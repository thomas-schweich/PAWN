"""Trial runner: process lifecycle, metrics monitoring, Optuna autopilot.

The runner manages GPU-isolated training processes, polls their metrics files,
detects failures, and optionally drives Optuna-based hyperparameter sweeps.
State is persisted to JSON so the runner can recover after MCP server restarts
while training processes continue running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("pawn.lab")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _is_alive(pid: int) -> bool:
    """Check if a process is alive. Reaps zombies as a side effect."""
    # First try to reap — if the process is our child zombie, waitpid clears it
    try:
        rpid, status = os.waitpid(pid, os.WNOHANG)
        if rpid != 0:
            return False  # reaped a zombie — process is done
    except ChildProcessError:
        pass  # not our child, fall through to kill check
    # Check via signal
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but can't signal


def _try_reap(pid: int) -> int | None:
    """Reap a child process (non-blocking). Returns exit code or None."""
    try:
        rpid, status = os.waitpid(pid, os.WNOHANG)
        if rpid == 0:
            return None
        return os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
    except ChildProcessError:
        return None  # not our child (recovered process)


def _params_to_cli(params: dict[str, Any]) -> list[str]:
    """Convert a param dict to CLI args: lr=3e-4 -> ['--lr', '3e-4']."""
    args: list[str] = []
    for k, v in params.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.extend([flag, str(v)])
    return args


def _parse_distribution(spec: dict[str, Any]) -> Any:
    """Convert a JSON distribution spec to an Optuna distribution object."""
    import optuna.distributions as d

    t = spec["type"]
    if t == "float":
        return d.FloatDistribution(spec["low"], spec["high"], log=spec.get("log", False))
    elif t == "int":
        return d.IntDistribution(spec["low"], spec["high"], step=spec.get("step", 1))
    elif t == "categorical":
        return d.CategoricalDistribution(spec["choices"])
    raise ValueError(f"Unknown distribution type: {t}")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Built-in search spaces (PAWN adapter strategies)
# ---------------------------------------------------------------------------

def _builtin_distributions(strategy: str) -> dict[str, Any]:
    """Return Optuna distributions for a PAWN adapter strategy."""
    import optuna.distributions as d
    Cat = d.CategoricalDistribution
    Float = d.FloatDistribution
    Int = d.IntDistribution

    common = {
        "lr": Float(1e-5, 1e-2, log=True),
        "batch_size": Cat([32, 64, 128, 256]),
        "weight_decay": Float(0.0, 0.1),
        "warmup_frac": Float(0.0, 0.15),
    }
    spaces: dict[str, dict] = {
        "lora": {**common,
            "lora_rank": Cat([2, 4, 8, 16, 32]),
            "lora_targets": Cat(["qkvo", "qv", "qkv"]),
            "lora_ffn": Cat([True, False]),
        },
        "bottleneck": {**common,
            "bottleneck_dim": Cat([4, 8, 16, 32, 64, 128, 256]),
            "no_adapt_attn": Cat([True, False]),
            "no_adapt_ffn": Cat([True, False]),
        },
        "film": {**common,
            "use_output_film": Cat([True, False]),
        },
        "sparse": {**common,
            "density": Float(0.001, 0.1, log=True),
            "sparse_targets": Cat(["qkvo", "qv", "qkv"]),
            "sparse_ffn": Cat([True, False]),
        },
        "hybrid": {
            "batch_size": Cat([32, 64, 128, 256]),
            "weight_decay": Float(0.0, 0.1),
            "warmup_frac": Float(0.0, 0.15),
            "lr": Float(1e-5, 1e-2, log=True),
            "film_lr": Float(1e-5, 1e-2, log=True),
            "lora_rank": Cat([2, 4, 8, 16]),
            "lora_targets": Cat(["qkvo", "qv", "qkv"]),
        },
        "specialized_clm": {**common,
            "d_model": Cat([32, 48, 84, 128, 192]),
            "n_layers": Int(1, 4),
            "n_heads": Cat([1, 2, 4, 8]),
        },
        "unfreeze": {**common,
            "unfreeze_layers": Cat(["6,7", "5,6,7", "4,5,6,7"]),
        },
    }
    rosa_common = {**common,
        "density": Float(0.001, 0.1, log=True),
        "lora_rank": Cat([2, 4, 8, 16]),
        "lora_targets": Cat(["qkvo", "qv", "qkv"]),
        "rosa_warmup_steps": Int(32, 256, step=32),
        "mask_samples": Cat([16, 32, 64]),
        "grad_alpha": Cat([1, 2]),
    }
    spaces["rosa"] = rosa_common
    spaces["retro-sparse"] = rosa_common
    spaces["retro-bottleneck"] = {**rosa_common, "bottleneck_dim": Cat([4, 8, 16])}

    if strategy not in spaces:
        raise ValueError(f"No built-in search space for strategy '{strategy}'. "
                         f"Available: {sorted(spaces)}")
    return spaces[strategy]


# ---------------------------------------------------------------------------
# Trial state
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    trial_id: int
    strategy: str
    params: dict[str, Any]
    cli_command: list[str]
    base_args: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    pid: int | None = None
    gpu_id: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    # Live metrics (updated by monitor)
    current_step: int = 0
    total_steps: int = 0
    steps_per_sec: float = 0.0
    last_train_loss: float | None = None
    best_val_loss: float | None = None
    best_accuracy: float | None = None
    actual_param_count: int | None = None
    # Files
    log_path: str = ""
    run_dir: str | None = None
    # Sweep
    optuna_number: int | None = None
    # Agent notes
    notes: str = ""

    def eta_seconds(self) -> float | None:
        if self.steps_per_sec > 0 and self.total_steps > self.current_step:
            return (self.total_steps - self.current_step) / self.steps_per_sec
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        eta = self.eta_seconds()
        d["eta_seconds"] = eta
        d["eta_human"] = _format_duration(eta)
        elapsed = (time.time() - self.start_time) if self.start_time else None
        d["elapsed_human"] = _format_duration(elapsed)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trial:
        # Strip computed fields that aren't constructor params
        d = {k: v for k, v in d.items()
             if k not in ("eta_seconds", "eta_human", "elapsed_human")}
        return cls(**d)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TrialRunner:
    """Manages GPU-isolated training processes with optional Optuna autopilot."""

    def __init__(
        self,
        workspace: str | None = None,
        code_dir: str | None = None,
        python: str = "python3",
    ):
        ws = workspace or os.environ.get("PAWN_WORKSPACE")
        if ws is None:
            ws = "/workspace" if Path("/workspace").exists() else str(Path.cwd())
        self.workspace = Path(ws)
        self.code_dir = Path(
            code_dir
            or os.environ.get("PAWN_CODE_DIR")
            or str(Path(__file__).resolve().parents[2])
        )
        self.python = python

        self.log_dir = self.workspace / "logs"
        self.results_dir = self.workspace / "sweep_results"
        self.state_path = self.workspace / "lab_state.json"
        self.events_path = self.workspace / "lab_events.jsonl"
        self.progress_log_path = self.workspace / "pod_manager.md"

        # State
        self.trials: dict[int, Trial] = {}
        self.next_trial_id: int = 0
        self.gpu_count: int = 0
        self.gpu_names: list[str] = []
        self.gpu_vram_mb: list[int] = []
        self.gpu_assignments: dict[int, int | None] = {}  # gpu -> trial_id|None

        # Sweep
        self.autopilot: bool = False
        self.sweep_strategies: list[str] = []
        self.sweep_base_args: dict[str, Any] = {}
        self.sweep_distributions: dict[str, Any] = {}
        self.pinned_params: dict[str, Any] = {}
        self.sweep_n_trials: int = 0
        self.sweep_launched: int = 0
        self.sweep_study_name: str = "sweep"
        self.sweep_directions: list[str] = ["minimize"]
        self._study: Any = None  # optuna.Study

        # Events
        self.events: list[dict[str, Any]] = []
        self.event_seq: int = 0
        self.start_time: float = time.time()
        self.cost_per_hour: float | None = None

        # Async
        self._monitor_tasks: dict[int, asyncio.Task[None]] = {}
        self._metrics_offsets: dict[int, int] = {}

        self._ensure_dirs()
        self._discover_gpus()

    # =======================================================================
    # Setup
    # =======================================================================

    def _ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace / "optuna-storage").mkdir(parents=True, exist_ok=True)

    def _discover_gpus(self) -> None:
        """Detect GPUs via PyTorch (works for both CUDA and ROCm)."""
        try:
            import torch
            if not torch.cuda.is_available():
                log.warning("No GPUs available (torch.cuda.is_available() = False)")
                return
            self.gpu_count = torch.cuda.device_count()
            for i in range(self.gpu_count):
                self.gpu_names.append(torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i)
                self.gpu_vram_mb.append(props.total_memory // (1024 * 1024))
                self.gpu_assignments.setdefault(i, None)
            log.info("Found %d GPUs: %s", self.gpu_count, self.gpu_names)
        except Exception as e:
            log.warning("GPU discovery failed: %s", e)
            self.gpu_count = 0

    # =======================================================================
    # State persistence
    # =======================================================================

    def _save_state(self) -> None:
        state = {
            "next_trial_id": self.next_trial_id,
            "trials": {str(k): v.to_dict() for k, v in self.trials.items()},
            "autopilot": self.autopilot,
            "sweep_strategies": self.sweep_strategies,
            "sweep_base_args": self.sweep_base_args,
            "pinned_params": self.pinned_params,
            "sweep_n_trials": self.sweep_n_trials,
            "sweep_launched": self.sweep_launched,
            "sweep_study_name": self.sweep_study_name,
            "sweep_directions": self.sweep_directions,
            "event_seq": self.event_seq,
            "start_time": self.start_time,
            "cost_per_hour": self.cost_per_hour,
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.rename(self.state_path)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
            self.next_trial_id = state.get("next_trial_id", 0)
            for k, v in state.get("trials", {}).items():
                self.trials[int(k)] = Trial.from_dict(v)
            self.autopilot = state.get("autopilot", False)
            self.sweep_strategies = state.get("sweep_strategies", [])
            self.sweep_base_args = state.get("sweep_base_args", {})
            self.pinned_params = state.get("pinned_params", {})
            self.sweep_n_trials = state.get("sweep_n_trials", 0)
            self.sweep_launched = state.get("sweep_launched", 0)
            self.sweep_study_name = state.get("sweep_study_name", "sweep")
            self.sweep_directions = state.get("sweep_directions", ["minimize"])
            self.event_seq = state.get("event_seq", 0)
            self.start_time = state.get("start_time", self.start_time)
            self.cost_per_hour = state.get("cost_per_hour")
            log.info("Loaded state: %d trials", len(self.trials))
        except Exception as e:
            log.error("Failed to load state: %s", e)

    # =======================================================================
    # GPU management
    # =======================================================================

    def _find_free_gpu(self) -> int | None:
        for gpu_id, trial_id in self.gpu_assignments.items():
            if trial_id is None:
                return gpu_id
        return None

    def _assign_gpu(self, trial_id: int, gpu_id: int) -> None:
        self.gpu_assignments[gpu_id] = trial_id

    def _release_gpu(self, gpu_id: int) -> None:
        self.gpu_assignments[gpu_id] = None

    def gpu_utilization(self) -> list[dict[str, Any]]:
        """Query current GPU memory usage via PyTorch."""
        try:
            import torch
            result = []
            for i in range(self.gpu_count):
                allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) // (1024 * 1024)
                total = self.gpu_vram_mb[i] if i < len(self.gpu_vram_mb) else 0
                result.append({
                    "gpu": i,
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "total_mb": total,
                    "assigned_trial": self.gpu_assignments.get(i),
                })
            return result
        except Exception:
            return []

    # =======================================================================
    # Trial lifecycle
    # =======================================================================

    async def launch(
        self,
        strategy: str,
        params: dict[str, Any] | None = None,
        base_args: dict[str, Any] | None = None,
    ) -> int:
        """Launch a single trial. Returns trial_id."""
        gpu_id = self._find_free_gpu()
        if gpu_id is None:
            raise RuntimeError(f"No free GPU (all {self.gpu_count} assigned)")

        trial_id = self.next_trial_id
        self.next_trial_id += 1

        merged_params = {**self.pinned_params, **(params or {})}
        merged_base = {**(base_args or self.sweep_base_args)}
        cmd = self._build_command(strategy, merged_params, merged_base, trial_id)

        trial = Trial(
            trial_id=trial_id,
            strategy=strategy,
            params=merged_params,
            cli_command=cmd,
            base_args=merged_base,
            gpu_id=gpu_id,
            log_path=str(self.results_dir / f"trial_{trial_id:04d}.log"),
            total_steps=merged_base.get("total_steps", 0)
                or merged_params.get("total_steps", 0),
        )
        self.trials[trial_id] = trial
        self._assign_gpu(trial_id, gpu_id)

        await self._spawn(trial)
        self._emit("trial_started", trial_id, {
            "strategy": strategy, "gpu": gpu_id, "params": merged_params,
        })
        self._save_state()
        self.render_progress_log()
        return trial_id

    def _build_command(
        self, strategy: str, params: dict, base_args: dict, trial_id: int,
    ) -> list[str]:
        script = str(self.code_dir / "scripts" / "train_adapter.py")
        trial_log_dir = str(self.log_dir / f"trial_{trial_id:04d}")

        cmd = [*self.python.split(), script, "--strategy", strategy]
        # Base args
        ba = dict(base_args)
        ba.setdefault("log_dir", trial_log_dir)
        ba.setdefault("local_checkpoints", True)
        cmd.extend(_params_to_cli(ba))
        # Suggested hyperparams
        cmd.extend(_params_to_cli(params))
        return cmd

    async def _spawn(self, trial: Trial) -> None:
        """Start the training process with GPU isolation."""
        env = os.environ.copy()
        if trial.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(trial.gpu_id)

        Path(trial.log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(trial.log_path, "w")

        proc = subprocess.Popen(
            trial.cli_command,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(self.code_dir),
        )
        log_fd.close()

        trial.pid = proc.pid
        trial.status = "running"
        trial.start_time = time.time()
        log.info("Spawned trial %d (PID %d) on GPU %d: %s",
                 trial.trial_id, proc.pid, trial.gpu_id, trial.strategy)

        self._monitor_tasks[trial.trial_id] = asyncio.create_task(
            self._monitor(trial.trial_id)
        )

    async def _monitor(self, trial_id: int) -> None:
        """Poll a running trial: check process + read metrics."""
        trial = self.trials[trial_id]
        try:
            while trial.status == "running":
                await asyncio.sleep(5.0)

                if trial.pid and not _is_alive(trial.pid):
                    break

                self._read_metrics(trial_id)
                issue = self._check_health(trial_id)
                if issue:
                    log.warning("Trial %d health issue: %s", trial_id, issue)
                    self._emit("health_warning", trial_id, {"issue": issue})

            # Process exited — final metrics read
            self._read_metrics(trial_id)

            # Determine outcome
            exit_code = _try_reap(trial.pid) if trial.pid else None
            if exit_code == 0 or trial.best_val_loss is not None:
                await self._on_complete(trial_id)
            else:
                reason = f"exit code {exit_code}" if exit_code is not None else "process exited"
                await self._on_failed(trial_id, reason)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Monitor error for trial %d: %s", trial_id, e, exc_info=True)
            await self._on_failed(trial_id, str(e))

    def _read_metrics(self, trial_id: int) -> None:
        """Read new lines from the trial's metrics.jsonl."""
        trial = self.trials[trial_id]

        # Find run dir if not yet discovered
        if trial.run_dir is None:
            trial_log_dir = self.log_dir / f"trial_{trial_id:04d}"
            for mf in trial_log_dir.glob("*/metrics.jsonl"):
                trial.run_dir = str(mf.parent)
                break
        if trial.run_dir is None:
            return

        metrics_path = Path(trial.run_dir) / "metrics.jsonl"
        if not metrics_path.exists():
            return

        offset = self._metrics_offsets.get(trial_id, 0)
        try:
            with open(metrics_path) as f:
                f.seek(offset)
                new_lines = f.readlines()
                self._metrics_offsets[trial_id] = f.tell()
        except OSError:
            return

        for line in new_lines:
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            rtype = rec.get("type")
            if rtype == "config":
                trial.total_steps = (
                    rec.get("total_steps")
                    or (rec.get("training") or {}).get("total_steps")
                    or trial.total_steps
                )
                trial.actual_param_count = rec.get("param_count", trial.actual_param_count)

            elif rtype == "train":
                trial.current_step = rec.get("step", trial.current_step)
                loss = rec.get("train/loss") or rec.get("train_loss")
                if loss is not None:
                    trial.last_train_loss = loss
                # Step rate: prefer step_time, fall back to elapsed/step
                st = rec.get("step_time")
                if st and st > 0:
                    trial.steps_per_sec = 1.0 / st
                elif rec.get("elapsed") and trial.current_step > 0:
                    trial.steps_per_sec = trial.current_step / rec["elapsed"]
                # Adapter scripts log val in train records
                vl = rec.get("val_loss") or rec.get("val/loss")
                if vl is not None and (trial.best_val_loss is None or vl < trial.best_val_loss):
                    trial.best_val_loss = vl
                acc = rec.get("val_top1") or rec.get("train/accuracy")
                if acc is not None:
                    trial.best_accuracy = acc

            elif rtype == "val":
                vl = rec.get("val/loss") or rec.get("val_loss") or rec.get("loss")
                if vl is not None and (trial.best_val_loss is None or vl < trial.best_val_loss):
                    trial.best_val_loss = vl
                acc = (rec.get("val/accuracy") or rec.get("val_top1")
                       or rec.get("accuracy"))
                if acc is not None:
                    trial.best_accuracy = acc

    def _check_health(self, trial_id: int) -> str | None:
        """Return a health issue string, or None if healthy."""
        trial = self.trials[trial_id]
        loss = trial.last_train_loss
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            if trial.current_step > 500:
                return "NaN/Inf loss"
        return None

    async def _on_complete(self, trial_id: int) -> None:
        trial = self.trials[trial_id]
        trial.status = "completed"
        trial.end_time = time.time()
        if trial.gpu_id is not None:
            self._release_gpu(trial.gpu_id)
        log.info("Trial %d completed: val_loss=%s acc=%s",
                 trial_id, trial.best_val_loss, trial.best_accuracy)
        self._emit("trial_completed", trial_id, {
            "best_val_loss": trial.best_val_loss,
            "best_accuracy": trial.best_accuracy,
            "param_count": trial.actual_param_count,
            "steps": trial.current_step,
        })
        # Report to Optuna
        if self._study and trial.optuna_number is not None:
            self._tell_optuna(trial)
        # Autopilot: launch next
        if self.autopilot and self.sweep_launched < self.sweep_n_trials:
            await self._autopilot_next()
        elif self.autopilot and self._all_done():
            self.autopilot = False
            self._emit("sweep_complete", data={"total_trials": self.sweep_launched})
        # Check if all GPUs idle
        if all(v is None for v in self.gpu_assignments.values()):
            self._emit("gpu_idle", data={"message": "All GPUs are idle"})
        self._save_state()
        self.render_progress_log()

    async def _on_failed(self, trial_id: int, reason: str) -> None:
        trial = self.trials[trial_id]
        trial.status = "failed"
        trial.end_time = time.time()
        if trial.gpu_id is not None:
            self._release_gpu(trial.gpu_id)
        log.warning("Trial %d failed: %s", trial_id, reason)
        self._emit("trial_failed", trial_id, {"reason": reason})
        # Report failure to Optuna
        if self._study and trial.optuna_number is not None:
            import optuna
            self._study.tell(trial.optuna_number, state=optuna.trial.TrialState.FAIL)
        # Autopilot: launch replacement
        if self.autopilot and self.sweep_launched < self.sweep_n_trials:
            await self._autopilot_next()
        self._save_state()
        self.render_progress_log()

    async def kill(self, trial_id: int) -> dict[str, Any]:
        """Kill a running trial via SIGTERM."""
        trial = self.trials.get(trial_id)
        if not trial:
            return {"error": f"Trial {trial_id} not found"}
        if trial.status != "running":
            return {"error": f"Trial {trial_id} is {trial.status}, not running"}
        if trial.pid:
            try:
                os.kill(trial.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        trial.status = "killed"
        trial.end_time = time.time()
        if trial.gpu_id is not None:
            self._release_gpu(trial.gpu_id)
        # Cancel monitor
        task = self._monitor_tasks.pop(trial_id, None)
        if task:
            task.cancel()
        self._emit("trial_killed", trial_id)
        # Autopilot: launch replacement
        if self.autopilot and self.sweep_launched < self.sweep_n_trials:
            await self._autopilot_next()
        self._save_state()
        self.render_progress_log()
        return {"killed": trial_id}

    # =======================================================================
    # Events
    # =======================================================================

    def _emit(
        self,
        event_type: str,
        trial_id: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.event_seq += 1
        event = {
            "seq": self.event_seq,
            "type": event_type,
            "trial_id": trial_id,
            "timestamp": _now_iso(),
            "data": data or {},
        }
        self.events.append(event)
        # Append to persistent log
        try:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError as e:
            log.error("Failed to write event: %s", e)

    def events_since(self, seq: int = 0) -> list[dict[str, Any]]:
        return [e for e in self.events if e["seq"] > seq]

    # =======================================================================
    # Sweep / autopilot
    # =======================================================================

    def _get_study(self) -> Any:
        if self._study is None:
            import optuna
            storage = f"sqlite:///{self.workspace}/optuna-storage/lab.db"
            self._study = optuna.create_study(
                study_name=self.sweep_study_name,
                storage=storage,
                directions=self.sweep_directions,
                load_if_exists=True,
            )
        return self._study

    async def configure_sweep(
        self,
        strategies: list[str],
        n_trials: int,
        base_args: dict[str, Any],
        pinned_params: dict[str, Any] | None = None,
        search_space: dict[str, Any] | None = None,
        study_name: str = "sweep",
        directions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Configure and start an autopilot sweep."""
        self.sweep_strategies = strategies
        self.sweep_base_args = base_args
        self.pinned_params = pinned_params or {}
        self.sweep_n_trials = n_trials
        self.sweep_launched = 0
        self.sweep_study_name = study_name
        self.sweep_directions = directions or ["minimize"]

        # Build distributions
        if search_space:
            # Custom search space provided as JSON specs
            self.sweep_distributions = {
                k: _parse_distribution(v) for k, v in search_space.items()
            }
        else:
            # Merge built-in distributions for all strategies
            merged: dict[str, Any] = {}
            for s in strategies:
                merged.update(_builtin_distributions(s))
            self.sweep_distributions = merged

        self.autopilot = True
        self._study = None  # reset to pick up new study_name
        self._emit("sweep_started", data={
            "strategies": strategies, "n_trials": n_trials,
        })

        # Fill all free GPUs
        launched = 0
        while self._find_free_gpu() is not None and self.sweep_launched < n_trials:
            await self._autopilot_next()
            launched += 1

        self._save_state()
        return {"status": "sweep_started", "launched": launched, "total": n_trials}

    async def _autopilot_next(self) -> None:
        """Ask Optuna for next trial and launch it."""
        if self._find_free_gpu() is None:
            return

        study = self._get_study()

        # Pick strategy (round-robin biased toward under-represented)
        strategy = self._pick_strategy()

        # Get distributions for this strategy, minus pinned params
        try:
            dists = _builtin_distributions(strategy)
        except ValueError:
            dists = self.sweep_distributions
        dists = {k: v for k, v in dists.items() if k not in self.pinned_params}

        trial = study.ask(dists)

        params = {**trial.params}
        self.sweep_launched += 1

        trial_id = await self.launch(strategy, params, self.sweep_base_args)
        self.trials[trial_id].optuna_number = trial.number

    def _pick_strategy(self) -> str:
        """Pick next strategy: round-robin biased toward least-explored."""
        if len(self.sweep_strategies) == 1:
            return self.sweep_strategies[0]
        counts: dict[str, int] = {s: 0 for s in self.sweep_strategies}
        for t in self.trials.values():
            if t.strategy in counts:
                counts[t.strategy] += 1
        min_count = min(counts.values())
        candidates = [s for s, c in counts.items() if c == min_count]
        return candidates[self.sweep_launched % len(candidates)]

    def _tell_optuna(self, trial: Trial) -> None:
        """Report trial results to Optuna."""
        if not self._study or trial.optuna_number is None:
            return
        values: list[float] = []
        if trial.best_val_loss is not None:
            values.append(trial.best_val_loss)
        else:
            values.append(float("inf"))
        # Multi-objective: add param_count if directions has 2 entries
        if len(self.sweep_directions) > 1 and trial.actual_param_count is not None:
            values.append(float(trial.actual_param_count))
        elif len(self.sweep_directions) > 1:
            values.append(float("inf"))
        try:
            self._study.tell(trial.optuna_number, values)
        except Exception as e:
            log.warning("Optuna tell failed for trial %d: %s", trial.trial_id, e)

    def _all_done(self) -> bool:
        """Check if all launched sweep trials are finished."""
        running = [t for t in self.trials.values()
                   if t.status == "running" and t.optuna_number is not None]
        return len(running) == 0 and self.sweep_launched >= self.sweep_n_trials

    async def pause(self) -> dict[str, Any]:
        self.autopilot = False
        self._save_state()
        return {"status": "paused", "running": sum(
            1 for t in self.trials.values() if t.status == "running"
        )}

    async def resume(self) -> dict[str, Any]:
        self.autopilot = True
        launched = 0
        while self._find_free_gpu() is not None and self.sweep_launched < self.sweep_n_trials:
            await self._autopilot_next()
            launched += 1
        self._save_state()
        return {"status": "resumed", "launched": launched}

    def pin(self, params: dict[str, Any]) -> dict[str, Any]:
        self.pinned_params.update(params)
        self._save_state()
        return {"pinned": self.pinned_params}

    async def seed_trial(
        self, strategy: str, params: dict[str, Any], values: list[float],
    ) -> dict[str, Any]:
        """Seed an existing result into the Optuna study."""
        import optuna
        study = self._get_study()
        dists = {}
        try:
            all_dists = _builtin_distributions(strategy)
            dists = {k: v for k, v in all_dists.items() if k in params}
        except ValueError:
            pass
        frozen = optuna.trial.create_trial(
            params=params,
            distributions=dists,
            values=values,
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(frozen)
        return {"seeded": True, "study": study.study_name}

    # =======================================================================
    # Reporting
    # =======================================================================

    def status(self) -> dict[str, Any]:
        running = [t.to_dict() for t in self.trials.values() if t.status == "running"]
        elapsed = time.time() - self.start_time
        cost = (self.cost_per_hour * elapsed / 3600) if self.cost_per_hour else None
        return {
            "gpus": self.gpu_utilization(),
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "running_trials": running,
            "total_trials": len(self.trials),
            "completed": sum(1 for t in self.trials.values() if t.status == "completed"),
            "failed": sum(1 for t in self.trials.values() if t.status == "failed"),
            "autopilot": self.autopilot,
            "sweep_progress": f"{self.sweep_launched}/{self.sweep_n_trials}"
                if self.sweep_n_trials else None,
            "elapsed": _format_duration(elapsed),
            "cost_per_hour": self.cost_per_hour,
            "estimated_cost": round(cost, 2) if cost else None,
        }

    def results(self) -> dict[str, Any]:
        rows = []
        for t in sorted(self.trials.values(), key=lambda t: t.trial_id):
            rows.append({
                "trial": t.trial_id,
                "strategy": t.strategy,
                "params": t.actual_param_count,
                "steps": t.current_step,
                "val_loss": t.best_val_loss,
                "accuracy": t.best_accuracy,
                "status": t.status,
                "notes": t.notes,
                "key_params": {k: v for k, v in t.params.items()
                               if k in ("lr", "lora_rank", "bottleneck_dim", "density",
                                        "d_model", "n_layers", "batch_size")},
            })
        # Compute Pareto front (for multi-objective)
        pareto: list[dict] = []
        completed = [r for r in rows if r["status"] == "completed" and r["val_loss"] is not None]
        completed.sort(key=lambda r: (r["params"] or float("inf")))
        best_loss = float("inf")
        for r in completed:
            if r["val_loss"] is not None and r["val_loss"] < best_loss:
                pareto.append(r)
                best_loss = r["val_loss"]
        return {"trials": rows, "pareto_front": pareto}

    def add_notes(self, trial_id: int, notes: str) -> dict[str, Any]:
        trial = self.trials.get(trial_id)
        if not trial:
            return {"error": f"Trial {trial_id} not found"}
        trial.notes = notes
        self._save_state()
        return {"ok": True}

    # =======================================================================
    # Progress log
    # =======================================================================

    def render_progress_log(self) -> str:
        """Render pod_manager.md from current state."""
        lines: list[str] = ["# Pod Manager Log\n"]

        # Environment
        lines.append("## Environment")
        lines.append(f"- GPUs: {self.gpu_count}x {self.gpu_names[0] if self.gpu_names else '?'}, "
                      f"{self.gpu_vram_mb[0] if self.gpu_vram_mb else '?'} MB each")
        lines.append(f"- Persistent storage: {self.workspace}")
        if self.sweep_strategies:
            lines.append(f"- Objective: sweep {', '.join(self.sweep_strategies)}")
        lines.append("")

        # Status
        elapsed = time.time() - self.start_time
        lines.append("## Current Status")
        lines.append(f"- Uptime: {_format_duration(elapsed)}")
        lines.append(f"- Autopilot: {'ON' if self.autopilot else 'OFF'}")
        if self.sweep_n_trials:
            lines.append(f"- Sweep: {self.sweep_launched}/{self.sweep_n_trials} launched")
        if self.cost_per_hour:
            cost = self.cost_per_hour * elapsed / 3600
            lines.append(f"- Cost: ${self.cost_per_hour}/hr, ~${cost:.2f} so far")
        lines.append("")

        # Active processes
        running = [t for t in self.trials.values() if t.status == "running"]
        if running:
            lines.append("## Active Processes")
            lines.append("| PID | GPU | Trial | Strategy | Step | Total | Step/s | ETA |")
            lines.append("|-----|-----|-------|----------|------|-------|--------|-----|")
            for t in running:
                eta = _format_duration(t.eta_seconds())
                lines.append(
                    f"| {t.pid} | {t.gpu_id} | {t.trial_id} | {t.strategy} "
                    f"| {t.current_step} | {t.total_steps} "
                    f"| {t.steps_per_sec:.1f} | {eta} |"
                )
            lines.append("")

        # Results table
        completed = [t for t in self.trials.values()
                     if t.status in ("completed", "failed", "killed")]
        if completed:
            lines.append("## Results")
            lines.append("| Trial | Strategy | Params | val_loss | Acc | Status | Notes |")
            lines.append("|-------|----------|--------|----------|-----|--------|-------|")
            for t in sorted(completed, key=lambda t: t.trial_id):
                vl = f"{t.best_val_loss:.4f}" if t.best_val_loss else "—"
                acc = f"{t.best_accuracy:.1%}" if t.best_accuracy else "—"
                pc = f"{t.actual_param_count:,}" if t.actual_param_count else "?"
                lines.append(
                    f"| {t.trial_id} | {t.strategy} | {pc} | {vl} "
                    f"| {acc} | {t.status} | {t.notes} |"
                )
            lines.append("")

        # Recent events
        recent = self.events[-10:]
        if recent:
            lines.append("## Recent Events")
            for e in recent:
                tid = e.get("trial_id")
                tid_str = f" (trial {tid})" if tid is not None else ""
                data_str = json.dumps(e.get("data", {}), default=str)
                lines.append(f"- [{e['timestamp']}] {e['type']}{tid_str} {data_str}")
            lines.append("")

        content = "\n".join(lines)
        try:
            self.progress_log_path.write_text(content)
        except OSError as e:
            log.error("Failed to write progress log: %s", e)
        return content

    # =======================================================================
    # Recovery
    # =======================================================================

    async def recover(self) -> None:
        """Re-attach to running processes from persisted state."""
        self._load_state()

        # Reload events from persistent log
        if self.events_path.exists():
            self.events = []
            for line in self.events_path.read_text().splitlines():
                try:
                    self.events.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
            if self.events:
                self.event_seq = max(e.get("seq", 0) for e in self.events)

        # Check which "running" trials are still alive
        for trial_id, trial in self.trials.items():
            if trial.status != "running":
                continue
            if trial.pid and _is_alive(trial.pid):
                log.info("Recovering trial %d (PID %d)", trial_id, trial.pid)
                self._monitor_tasks[trial_id] = asyncio.create_task(
                    self._monitor(trial_id)
                )
                if trial.gpu_id is not None:
                    self._assign_gpu(trial_id, trial.gpu_id)
            else:
                log.warning("Trial %d (PID %d) no longer running", trial_id, trial.pid)
                self._read_metrics(trial_id)
                if trial.best_val_loss is not None:
                    trial.status = "completed"
                else:
                    trial.status = "failed"
                trial.end_time = time.time()

        self._save_state()
        self.render_progress_log()
        log.info("Recovery complete: %d trials, %d still running",
                 len(self.trials),
                 sum(1 for t in self.trials.values() if t.status == "running"))

    def shutdown(self) -> None:
        """Save state on shutdown. Training processes continue independently."""
        self._save_state()
        self.render_progress_log()
        log.info("Runner shutdown (training processes continue)")
