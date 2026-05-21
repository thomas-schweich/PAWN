"""Trial runner: process lifecycle and metrics monitoring.

The runner manages GPU-isolated training processes, polls their metrics files,
and detects failures. State is persisted to JSON so the runner can recover
after MCP server restarts while training processes continue running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from pawn.lab.monitor import (
    check_health,
    is_alive,
    read_cotrain_val_summary,
    read_metrics,
    read_pretrain_val_summary,
)
from pawn.lab.state import Trial, _format_duration, _now_iso

log = logging.getLogger("pawn.lab")


_V2_ADAPTER_STRATEGIES = (
    "lora", "film", "unfreeze", "bottleneck",
    "hybrid", "sparse", "rosa", "specialized_clm",
)


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a trial-config dict against the v2 JAX-trainer surface.

    The v2 trainers (``scripts/train_jax.py`` / ``scripts/train_jax_adapter.py``)
    use argparse, not the deleted ``pawn.run_config`` pydantic schema.
    This validator checks the load-bearing shape: ``run_type`` is one of
    ``pretrain`` / ``adapter`` (no more ``cotrain``), and the per-type
    required fields are present. The trainers themselves do the
    deep-argparse validation when the subprocess fires; surface the
    obvious-misuse cases here.

    Returns ``config`` (shallow-copied) so callers can extend without
    mutating their input.
    """
    run_type = config.get("run_type")
    if run_type == "cotrain":
        raise ValueError(
            "run_type='cotrain' was removed in v2.0.0 — the supernet's "
            "multi-variant joint loss replaces the cotrain harness. Use "
            "run_type='pretrain' against the supernet, which trains all "
            "nested variants jointly."
        )
    if run_type not in ("pretrain", "adapter"):
        raise ValueError(
            f"run_type must be 'pretrain' or 'adapter', got {run_type!r}"
        )

    cfg = dict(config)
    # Defaults that mirror the train_jax* argparse defaults so a
    # minimal trial config is enough.
    cfg.setdefault("supernet", "tiny")
    cfg.setdefault("k", 50)
    if run_type == "adapter":
        if cfg.get("strategy") not in _V2_ADAPTER_STRATEGIES:
            raise ValueError(
                f"adapter trial requires strategy in {list(_V2_ADAPTER_STRATEGIES)}, "
                f"got {cfg.get('strategy')!r}"
            )
        cfg.setdefault("variant", "base")
    return cfg


class TrialRunner:
    """Manages GPU-isolated training processes."""

    def __init__(
        self,
        workspace: str | None = None,
        code_dir: str | None = None,
        python: str = "python3",
    ):
        ws = workspace or os.environ.get("PAWN_WORKSPACE")
        if ws is None:
            # On pods: /workspace. Locally: runs/ under the repo root.
            ws = "/workspace" if Path("/workspace").exists() else str(
                Path(__file__).resolve().parents[2] / "runs"
            )
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
        self.gpu_assignments: dict[int, int | None] = {}

        # Events
        self.events: list[dict[str, Any]] = []
        self.event_seq: int = 0
        self._last_events_seq: int = 0
        self.start_time: float = time.time()
        self.cost_per_hour: float | None = None

        # Async
        self._monitor_tasks: dict[int, asyncio.Task[None]] = {}
        self._metrics_offsets: dict[int, int] = {}
        # Rolling (step, elapsed) window per trial (or per cotrain
        # variant) for stable throughput estimation. See
        # ``pawn.lab.monitor._update_sps_window``.
        self._sps_windows: dict[Any, list[tuple[int, float]]] = {}

        self._ensure_dirs()
        self._gpus_discovered = False
        self._mps_active: bool | None = None

    # =======================================================================
    # Setup
    # =======================================================================

    def _ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _discover_gpus(self) -> None:
        """Detect GPUs via ``nvidia-smi`` / ``rocm-smi`` shell-out so the
        MCP server stays framework-agnostic (the v2 trainer is JAX,
        and JAX's GPU detection still spawns persistent worker
        threads — same anti-pattern that the old torch-based discovery
        ran in a subprocess to avoid).
        """
        if self._gpus_discovered:
            return
        self._gpus_discovered = True

        gpus: list[dict[str, Any]] = []
        # Try nvidia-smi first (CUDA pods), then rocm-smi (ROCm pods).
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=10,
            )
            for line in out.strip().splitlines():
                name, vram = line.split(",", 1)
                gpus.append({
                    "name": name.strip(),
                    "vram_mb": int(vram.strip()),
                })
        except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showproductname",
                     "--showmeminfo", "vram", "--json"],
                    text=True, timeout=10,
                )
                data = json.loads(out)
                for card_id, info in data.items():
                    if not card_id.startswith("card"):
                        continue
                    name = info.get("Card Series") or info.get("Card model") or card_id
                    vram_bytes = int(info.get("VRAM Total Memory (B)", 0))
                    gpus.append({
                        "name": str(name),
                        "vram_mb": vram_bytes // (1024 * 1024),
                    })
            except (
                subprocess.SubprocessError, FileNotFoundError, OSError, ValueError,
                json.JSONDecodeError,
            ):
                log.warning("GPU discovery failed: nvidia-smi and rocm-smi both unavailable")

        self.gpu_count = len(gpus)
        for i, g in enumerate(gpus):
            self.gpu_names.append(g["name"])
            self.gpu_vram_mb.append(g["vram_mb"])
            self.gpu_assignments.setdefault(i, None)
        log.info("Found %d GPUs: %s", self.gpu_count, self.gpu_names)

    # =======================================================================
    # State persistence
    # =======================================================================

    def _save_state(self) -> None:
        state = {
            "next_trial_id": self.next_trial_id,
            "trials": {str(k): v.to_dict() for k, v in self.trials.items()},
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
            self.event_seq = state.get("event_seq", 0)
            self.start_time = state.get("start_time", self.start_time)
            self.cost_per_hour = state.get("cost_per_hour")
            log.info("Loaded state: %d trials", len(self.trials))
        except Exception as e:
            log.error("Failed to load state: %s", e)

    # =======================================================================
    # GPU management
    # =======================================================================

    def _is_mps_active(self) -> bool:
        """Detect if CUDA MPS daemon is running."""
        if self._mps_active is None:
            try:
                out = subprocess.check_output(
                    ["pgrep", "-f", "nvidia-cuda-mps"],
                    text=True, timeout=5,
                )
                self._mps_active = bool(out.strip())
            except Exception:
                self._mps_active = False
            if self._mps_active:
                log.info("CUDA MPS detected — GPU isolation disabled")
        return self._mps_active

    def _find_free_gpu(self) -> int | None:
        self._discover_gpus()
        if self._is_mps_active():
            return 0 if self.gpu_count > 0 else None
        for gpu_id, trial_id in self.gpu_assignments.items():
            if trial_id is None:
                return gpu_id
        return None

    def _assign_gpu(self, trial_id: int, gpu_id: int) -> None:
        if not self._is_mps_active():
            self.gpu_assignments[gpu_id] = trial_id

    def _release_gpu(self, gpu_id: int) -> None:
        if not self._is_mps_active():
            self.gpu_assignments[gpu_id] = None

    def gpu_utilization(self) -> list[dict[str, Any]]:
        """Return GPU info without importing torch into this process."""
        self._discover_gpus()
        return [
            {
                "gpu": i,
                "total_mb": self.gpu_vram_mb[i] if i < len(self.gpu_vram_mb) else 0,
                "assigned_trial": self.gpu_assignments.get(i),
                "mps": self._is_mps_active(),
            }
            for i in range(self.gpu_count)
        ]

    # =======================================================================
    # Trial lifecycle
    # =======================================================================

    async def launch(
        self,
        config: dict[str, Any],
        *,
        # Legacy compat: if strategy/params/base_args are passed, merge
        # them into a config dict automatically.
        strategy: str | None = None,
        params: dict[str, Any] | None = None,
        base_args: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Launch a single trial. Returns trial_id.

        ``config`` is a dict matching ``RunConfig`` (either
        ``PretrainConfig`` or ``AdapterConfig``).  It is validated and
        written to a JSON file, then passed to ``scripts/train.py
        --config``.
        """
        # Legacy shim: build a config dict from strategy/params/base_args
        if strategy is not None:
            merged: dict[str, Any] = {"run_type": "adapter", "strategy": strategy}
            merged.update(base_args or {})
            merged.update(params or {})
            merged.update(config or {})
            config = merged

        gpu_id = self._find_free_gpu()
        if gpu_id is None:
            raise RuntimeError(f"No free GPU (all {self.gpu_count} assigned)")

        trial_id = self.next_trial_id
        self.next_trial_id += 1

        # Apply defaults
        trial_log_dir = str(self.log_dir / f"trial_{trial_id:04d}")
        config.setdefault("log_dir", trial_log_dir)
        config.setdefault("local_checkpoints", True)

        validated = _validate_config(config)
        cmd = self._build_command(validated, trial_id)

        # ``cotrain`` was rejected by _validate_config; only pretrain
        # and adapter trial types reach here.
        strategy_display = validated.get("strategy") or validated.get("variant", "pretrain")
        trial = Trial(
            trial_id=trial_id,
            strategy=strategy_display,
            config=validated,
            params=validated,
            cli_command=cmd,
            gpu_id=gpu_id,
            log_path=str(self.results_dir / f"trial_{trial_id:04d}.log"),
            total_steps=validated.get("total_steps", 0) or 0,
            tags=tags or [],
        )
        self.trials[trial_id] = trial
        self._assign_gpu(trial_id, gpu_id)

        await self._spawn(trial)
        self._emit("trial_started", trial_id, {
            "strategy": strategy_display, "gpu": gpu_id,
            "config": validated,
        })
        self._save_state()
        self.render_progress_log()
        return trial_id

    async def resume_trial(
        self,
        trial_id: int,
        total_steps: int | None = None,
        pause_after_steps: int | None = None,
    ) -> int:
        """Resume a completed/failed trial from its best checkpoint.

        For cotrain trials, discovers per-variant checkpoints and sets
        the resume path on each variant in the new config.
        """
        old = self.trials.get(trial_id)
        if not old:
            raise RuntimeError(f"Trial {trial_id} not found")
        if not old.run_dir:
            raise RuntimeError(f"Trial {trial_id} has no run directory")

        new_config = dict(old.config)
        new_config.pop("pause_after_steps", None)

        # Cotrain support was removed in v2.0.0; only pretrain + adapter
        # trial types persist, both single-run-dir.
        ckpt_dir = self._find_latest_checkpoint(Path(old.run_dir))
        new_config["resume"] = str(ckpt_dir)

        if total_steps is not None:
            new_config["total_steps"] = total_steps
        if pause_after_steps is not None:
            new_config["pause_after_steps"] = pause_after_steps

        return await self.launch(new_config, tags=old.tags)

    @staticmethod
    def _find_latest_checkpoint(run_dir: Path) -> Path:
        """Find the latest checkpoint under a run directory.

        Checks for ``best/`` and ``final/`` symlinks first (adapter runs),
        then falls back to the highest-numbered ``step_*`` directory
        (pretrain/cotrain runs, which don't create best/final symlinks).
        """
        ckpt_base = run_dir / "checkpoints"
        ckpt_dir = ckpt_base / "best"
        if not ckpt_dir.exists():
            ckpt_dir = ckpt_base / "final"
        if not ckpt_dir.exists():
            step_dirs = sorted(ckpt_base.glob("step_*"))
            if step_dirs:
                ckpt_dir = step_dirs[-1]
        if not ckpt_dir.exists():
            raise RuntimeError(f"No checkpoint found under {run_dir}")
        return ckpt_dir

    def _resolve_cotrain_resume_DEAD(
        self, old: "Trial", new_config: dict[str, Any],
    ) -> None:
        """DEAD CODE: v2 dropped cotrain trials. Retained as a stub
        so callers still get a clear NotImplementedError if they
        somehow reach this branch (they shouldn't — _validate_config
        rejects run_type='cotrain' upfront)."""
        raise NotImplementedError(
            "cotrain resume is unsupported in v2 — the supernet "
            "replaces cotrain"
        )
        # Unreachable below — kept to satisfy module-level imports.
        if not old.variants:
            raise RuntimeError(
                f"Trial {old.trial_id} is cotrain but has no variant state. "
                "Cannot determine per-variant checkpoints."
            )

        # Deep-copy variants list so we can mutate
        import copy
        variants = copy.deepcopy(new_config.get("variants", []))

        for v_cfg in variants:
            name = v_cfg.get("name")
            if name not in old.variants:
                raise RuntimeError(
                    f"Variant '{name}' not found in trial {old.trial_id} state"
                )
            vs = old.variants[name]
            v_run_dir = vs.get("run_dir")
            if not v_run_dir:
                raise RuntimeError(
                    f"Variant '{name}' in trial {old.trial_id} has no run directory"
                )
            ckpt_dir = self._find_latest_checkpoint(Path(v_run_dir))
            v_cfg["resume"] = str(ckpt_dir)

        new_config["variants"] = variants

    def _build_command(
        self, config: dict[str, Any], trial_id: int,
    ) -> list[str]:
        """Build a v2 train-script argv from a validated trial config.

        The v1 ``scripts/train.py --config <json>`` interface is gone;
        the v2 trainers use argparse and don't read a unified JSON
        config. This translates the trial dict to ``--flag value``
        argv. Unknown keys (anything outside the curated v2 surface)
        raise — silent passthrough would mask trial-config typos.
        """
        run_type = config["run_type"]
        config_dir = self.log_dir / f"trial_{trial_id:04d}"
        config_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot the config alongside the trial for reproducibility,
        # even though the v2 trainer doesn't read it back.
        (config_dir / "trial_config.json").write_text(
            json.dumps(config, indent=2, default=str)
        )

        if run_type == "pretrain":
            script = str(self.code_dir / "scripts" / "train_jax.py")
            argv: list[str] = [*self.python.split(), script]
            allowed = {
                "supernet", "total_steps", "batch_size", "seq_len", "k",
                "lr", "warmup_steps", "seed", "corpus_seed", "model_seed",
                "max_corpus_gb", "logs_dir",
            }
        else:  # adapter
            script = str(self.code_dir / "scripts" / "train_jax_adapter.py")
            argv = [
                *self.python.split(), script,
                "--strategy", str(config["strategy"]),
            ]
            allowed = {
                "supernet", "variant", "total_steps", "batch_size", "seq_len",
                "k", "lr", "warmup_steps", "seed", "corpus_seed", "model_seed",
                "max_corpus_gb", "logs_dir", "val_frac", "val_every",
                # adapter-shape flags
                "rank", "lora_alpha",
                "rosa_warmup_frac", "rosa_top_k_frac",
                "n_unfreeze", "bottleneck_dim", "bottleneck_n_hidden",
                "sparse_density",
                "specialized_d_model", "specialized_n_layers",
                "specialized_n_heads", "specialized_d_ff",
            }
            # list-valued flags handled below
            list_flags = {"lora_targets", "rosa_targets", "sparse_targets"}
            allowed |= list_flags
            # bool-flag pairs (CLI inverts the dest name)
            bool_flag_pairs = {
                "include_lm_head": ("--include-lm-head", "--no-include-lm-head"),
                "include_embeddings": ("--include-embeddings", "--no-include-embeddings"),
                "film_output": ("--film-output", "--no-film-output"),
                "sparse_hard": ("--sparse-hard", "--no-sparse-hard"),
                "bottleneck_no_attn": ("--bottleneck-no-attn", None),
                "bottleneck_no_ffn": ("--bottleneck-no-ffn", None),
                "quiet": ("--quiet", None),
            }

        # Ignore lab-bookkeeping keys.
        skip = {"run_type", "strategy", "log_dir"}

        for key, value in config.items():
            if key in skip:
                continue
            if key not in allowed:
                raise ValueError(
                    f"unknown v2 trainer flag {key!r} in trial config; "
                    f"allowed: {sorted(allowed)}"
                )
            cli = "--" + key.replace("_", "-")
            if run_type == "adapter" and key in list_flags:
                # ``--lora-targets q v`` shape. Accept either an
                # explicit list/tuple of letters or the concatenated
                # shorthand (``"qkvo"``) the sweep distributions
                # emit, since Optuna's CategoricalDistribution rejects
                # non-scalar choices on persistable storage.
                if isinstance(value, str):
                    letters: list[str] = list(value)
                else:
                    letters = [str(t) for t in value]
                for letter in letters:
                    if letter not in ("q", "k", "v", "o"):
                        raise ValueError(
                            f"{key}={value!r}: unexpected target letter "
                            f"{letter!r}; expected subset of qkvo"
                        )
                argv.append(cli)
                argv.extend(letters)
            elif run_type == "adapter" and key in bool_flag_pairs:
                pos, neg = bool_flag_pairs[key]
                if value:
                    argv.append(pos)
                elif neg is not None:
                    argv.append(neg)
                # else: bool false on a no-negation flag → omit
            else:
                argv.extend([cli, str(value)])
        return argv

    async def _spawn(self, trial: Trial) -> None:
        """Start the training process, with GPU isolation unless MPS is active."""
        env = os.environ.copy()
        if trial.gpu_id is not None and not self._is_mps_active():
            env["CUDA_VISIBLE_DEVICES"] = str(trial.gpu_id)

        Path(trial.log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(trial.log_path, "w")
        try:
            proc = subprocess.Popen(
                trial.cli_command,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(self.code_dir),
            )
        finally:
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
        exit_code: int | None = None
        try:
            while trial.status == "running":
                await asyncio.sleep(5.0)

                if trial.pid:
                    alive, code = is_alive(trial.pid)
                    if not alive:
                        exit_code = code
                        break

                read_metrics(trial, self.log_dir, self._metrics_offsets, self._sps_windows)
                issue = check_health(trial)
                if issue:
                    log.warning("Trial %d health issue: %s", trial_id, issue)
                    self._emit("health_warning", trial_id, {"issue": issue})

            # Process exited — final metrics read
            read_metrics(trial, self.log_dir, self._metrics_offsets, self._sps_windows)

            if trial.status == "killed":
                # Wait for the process to actually exit before releasing GPU.
                # kill() sends SIGTERM but graceful shutdown (checkpoint save)
                # can take 30-60s. The while loop above exits immediately when
                # status changes to "killed", so we poll here.
                if trial.pid:
                    while True:
                        alive, _ = is_alive(trial.pid)
                        if not alive:
                            break
                        await asyncio.sleep(1.0)
                if trial.gpu_id is not None:
                    self._release_gpu(trial.gpu_id)
                read_metrics(trial, self.log_dir, self._metrics_offsets, self._sps_windows)
                self._save_state()
            elif exit_code == 0:
                self._complete(trial_id)
            else:
                # Any non-zero exit = failure. Previously we fell back to
                # ``trial.best_val_loss is not None`` as a safety net, but
                # that misclassified crashed trials as "completed" once the
                # baseline eval populated ``best_val_loss`` before the real
                # failure point — e.g. OOMs at step 0 or Python tracebacks
                # after model load.
                reason = (
                    f"exit code {exit_code}"
                    if exit_code is not None
                    else "process exited (exit code unavailable)"
                )
                self._fail(trial_id, reason)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Monitor error for trial %d: %s", trial_id, e, exc_info=True)
            self._fail(trial_id, str(e))

    def _complete(self, trial_id: int) -> None:
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
        if all(v is None for v in self.gpu_assignments.values()):
            self._emit("gpu_idle", data={"message": "All GPUs are idle"})
        self._save_state()
        self.render_progress_log()

    def _fail(self, trial_id: int, reason: str) -> None:
        trial = self.trials[trial_id]
        trial.status = "failed"
        trial.end_time = time.time()
        if trial.gpu_id is not None:
            self._release_gpu(trial.gpu_id)
        log.warning("Trial %d failed: %s", trial_id, reason)
        self._emit("trial_failed", trial_id, {"reason": reason})
        self._save_state()
        self.render_progress_log()

    async def kill(self, trial_id: int) -> dict[str, Any]:
        """Kill a running trial via SIGTERM.

        Sets status to 'killed' and sends SIGTERM, but lets the monitor
        task detect the actual process exit and release the GPU. This
        avoids a window where the GPU appears free while the process is
        still doing graceful shutdown (checkpoint saving).
        """
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
        # Final metrics read before marking killed
        read_metrics(trial, self.log_dir, self._metrics_offsets, self._sps_windows)
        trial.status = "killed"
        trial.end_time = time.time()
        self._emit("trial_killed", trial_id)
        self._save_state()
        self.render_progress_log()
        return {
            "killed": trial_id,
            "step": trial.current_step,
            "train_loss": trial.last_train_loss,
            "train_acc": trial.last_train_acc,
            "val_loss": trial.best_val_loss,
            "val_acc": trial.best_accuracy,
        }

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
        try:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError as e:
            log.error("Failed to write event: %s", e)

    def events_since(self, seq: int | None = None) -> tuple[list[dict[str, Any]], int]:
        """Return events since seq and update the cursor.

        If seq is None, returns events since the last call (auto-tracking).
        Returns (events, latest_seq).
        """
        if seq is None:
            seq = self._last_events_seq
        events = [e for e in self.events if e["seq"] > seq]
        self._last_events_seq = self.event_seq
        return events, self.event_seq

    # =======================================================================
    # Reporting
    # =======================================================================

    def status(self) -> dict[str, Any]:
        running = []
        for t in self.trials.values():
            if t.status == "running":
                cfg = t.config or t.params
                row: dict[str, Any] = {
                    "trial": t.trial_id, "strategy": t.strategy,
                    "step": t.current_step, "total": t.total_steps,
                    "sps": round(t.steps_per_sec, 2),
                    "eta": _format_duration(t.eta_seconds()),
                    "train_loss": t.last_train_loss, "train_acc": t.last_train_acc,
                    "val_loss": t.best_val_loss, "val_acc": t.best_accuracy,
                    "params": t.actual_param_count, "pid": t.pid, "gpu": t.gpu_id,
                    "key_hp": {k: v for k, v in cfg.items()
                               if k in ("lr", "lora_rank", "bottleneck_dim",
                                        "bottleneck_n_hidden",
                                        "density", "d_model", "n_layers", "batch_size")},
                }
                # For pretraining runs, surface game-completion metrics and
                # the power-law forfeit-rate fit (matches the dashboard chart).
                if cfg.get("run_type") == "pretrain":
                    pretrain = read_pretrain_val_summary(t)
                    if pretrain:
                        row["pretrain"] = pretrain
                # Same surface for cotrain, per variant.
                elif cfg.get("run_type") == "cotrain":
                    cotrain = read_cotrain_val_summary(t)
                    if cotrain:
                        row["cotrain"] = cotrain
                running.append(row)
        elapsed = time.time() - self.start_time
        cost = (self.cost_per_hour * elapsed / 3600) if self.cost_per_hour else None
        return {
            "gpus": self.gpu_utilization(),
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "running": running,
            "total_trials": len(self.trials),
            "completed": sum(1 for t in self.trials.values() if t.status == "completed"),
            "failed": sum(1 for t in self.trials.values() if t.status == "failed"),
            "elapsed": _format_duration(elapsed),
            "cost_per_hour": self.cost_per_hour,
            "estimated_cost": round(cost, 2) if cost else None,
        }

    def results(self, strategy: str | None = None, tag: str | None = None) -> dict[str, Any]:
        rows = []
        trials = sorted(self.trials.values(), key=lambda t: t.trial_id)
        if tag:
            trials = [t for t in trials if tag in t.tags]
        for t in trials:
            elapsed = (t.end_time - t.start_time) if t.end_time and t.start_time else None
            cfg = t.config or t.params
            rows.append({
                "trial": t.trial_id, "strategy": t.strategy,
                "params": t.actual_param_count, "steps": t.current_step,
                "val_loss": t.best_val_loss, "accuracy": t.best_accuracy,
                "status": t.status, "notes": t.notes, "tags": t.tags,
                "wall_time": _format_duration(elapsed),
                "key_hp": {k: v for k, v in cfg.items()
                           if k in ("lr", "lora_rank", "bottleneck_dim",
                                    "bottleneck_n_hidden", "density",
                                    "d_model", "n_layers", "batch_size")},
            })
        # Pareto front: trials not dominated on (param_count, val_loss).
        # A trial is dominated if another trial has both fewer (or equal)
        # params AND lower (or equal) val_loss, with at least one strict.
        completed = [r for r in rows if r["status"] == "completed"
                     and r["val_loss"] is not None and r["params"] is not None]
        pareto: list[dict[str, Any]] = []
        for r in completed:
            dominated = False
            for other in completed:
                if other is r:
                    continue
                if (other["params"] <= r["params"]
                        and other["val_loss"] <= r["val_loss"]
                        and (other["params"] < r["params"]
                             or other["val_loss"] < r["val_loss"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        pareto.sort(key=lambda r: r["params"])

        # Infer strategy for suggestions from completed trials if not provided
        suggest_strategy = strategy
        if suggest_strategy is None and completed:
            strategies = {r["strategy"] for r in completed}
            if len(strategies) == 1:
                suggest_strategy = strategies.pop()
        result: dict[str, Any] = {"trials": rows, "pareto_front": pareto}
        if suggest_strategy:
            result["suggestions"] = self._suggest(suggest_strategy, completed)
        else:
            result["suggestions"] = []
        return result

    def _suggest(self, strategy: str, completed: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
        """Create an ephemeral Optuna study, seed it, and return N suggestions."""
        try:
            import optuna
            from pawn.lab.sweep import builtin_distributions
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            dists = builtin_distributions(strategy)
            study = optuna.create_study(study_name="suggest", direction="minimize")

            seeded = 0
            for r in completed:
                hp = r.get("key_hp", {})
                trial_dists = {k: v for k, v in dists.items() if k in hp}
                trial_params = {k: v for k, v in hp.items() if k in dists}
                if not trial_dists:
                    continue
                try:
                    frozen = optuna.trial.create_trial(
                        params=trial_params, distributions=trial_dists,
                        values=[r["val_loss"]], state=optuna.trial.TrialState.COMPLETE,
                    )
                    study.add_trial(frozen)
                    seeded += 1
                except Exception:
                    pass

            suggestions = []
            for _ in range(n):
                trial = study.ask(dists)
                suggestions.append(trial.params)
            return suggestions
        except Exception as e:
            log.debug("Suggestion failed: %s", e)
            return []

    def trial_log(self, trial_id: int, lines: int = 50) -> dict[str, Any]:
        """Return the last N lines of a trial's stdout log."""
        trial = self.trials.get(trial_id)
        if not trial:
            return {"error": f"Trial {trial_id} not found"}
        log_path = Path(trial.log_path)
        if not log_path.exists():
            return {"error": f"Log file not found: {trial.log_path}"}
        all_lines = log_path.read_text().splitlines()
        return {"trial": trial_id, "lines": all_lines[-lines:]}

    def add_notes(self, trial_id: int, notes: str) -> dict[str, Any]:
        trial = self.trials.get(trial_id)
        if not trial:
            return {"error": f"Trial {trial_id} not found"}
        trial.notes = notes
        self._save_state()
        return {"ok": True}

    # =======================================================================
    # Audit
    # =======================================================================

    def audit(
        self,
        trial_id: int | None = None,
        *,
        check_hf: bool = False,
    ) -> dict[str, Any]:
        """Per-trial pass/fail on completion invariants.

        Without arguments: audits every trial. With ``trial_id``: audits
        just that one. Skips ``running`` and ``queued`` trials.

        Invariants per trial:
          - ``schedule_complete``: ``schedule_health.json`` says
            ``actual_total_steps == planned_total_steps``. Equality, not
            tolerance — partial decay reports as a failure.
          - ``checkpoint_complete``: latest ``step_*`` directory has its
            ``.complete`` SHA-256 sentinel.
          - ``checkpoint_on_hf`` (when ``check_hf=True`` and the trial
            has an ``hf_repo`` configured): the latest local checkpoint
            directory name appears in ``list_repo_files`` for the
            run's branch. Off by default — the HF API call is the
            slow path.
        """
        trials = (
            [self.trials[trial_id]]
            if trial_id is not None and trial_id in self.trials
            else list(self.trials.values())
        )
        rows: list[dict[str, Any]] = []
        for t in trials:
            if t.status in ("running", "queued"):
                continue
            row = self._audit_trial(t, check_hf=check_hf)
            rows.append(row)

        any_fail = any(
            any(c.get("pass") is False for c in r["checks"].values())
            for r in rows
        )
        return {"trials": rows, "any_failure": any_fail}

    @staticmethod
    def _check(
        ok: bool | None, *, reason: str | None = None, **details: Any,
    ) -> dict[str, Any]:
        """Construct a single audit-check entry.

        Centralizes the ``{pass: bool|None, ...}`` shape so callers
        don't repeat the dict scaffolding. ``reason`` is omitted from
        the result when ``None`` so passing checks don't carry an
        empty-reason field.
        """
        out: dict[str, Any] = {"pass": ok}
        if reason is not None:
            out["reason"] = reason
        out.update(details)
        return out

    def _audit_trial(
        self, trial: Trial, *, check_hf: bool,
    ) -> dict[str, Any]:
        checks: dict[str, dict[str, Any]] = {}
        check = self._check

        # 1. schedule_complete via schedule_health.json
        if trial.run_dir:
            health_path = Path(trial.run_dir) / "schedule_health.json"
            if health_path.exists():
                try:
                    h = json.loads(health_path.read_text())
                    planned = int(h.get("planned_total_steps", 0))
                    actual = int(h.get("actual_total_steps", 0))
                    checks["schedule_complete"] = check(
                        planned == actual and planned > 0,
                        planned_total_steps=planned,
                        actual_total_steps=actual,
                        reason_for_stop=h.get("reason_for_stop"),
                        actual_final_lr=h.get("actual_final_lr"),
                    )
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    checks["schedule_complete"] = check(
                        None, reason=f"schedule_health.json unreadable: {e}",
                    )
            else:
                checks["schedule_complete"] = check(
                    None,
                    reason="schedule_health.json absent (older trainer or pre-init crash)",
                )
        else:
            checks["schedule_complete"] = check(None, reason="no run_dir")

        # 2. checkpoint_complete via .complete sentinel on latest step_*
        latest_ckpt: Path | None = None
        if trial.run_dir:
            ckpt_base = Path(trial.run_dir) / "checkpoints"
            if ckpt_base.exists():
                step_dirs = sorted(ckpt_base.glob("step_*"))
                if step_dirs:
                    latest_ckpt = step_dirs[-1]
        if latest_ckpt is None:
            checks["checkpoint_complete"] = check(
                None, reason="no step_* checkpoint found",
            )
        else:
            checks["checkpoint_complete"] = check(
                (latest_ckpt / ".complete").exists(),
                checkpoint=latest_ckpt.name,
                path=str(latest_ckpt),
            )

        # 3. checkpoint_on_hf (opt-in: HF API call is the slow path)
        hf_repo = (trial.config or {}).get("hf_repo")
        if not hf_repo:
            checks["checkpoint_on_hf"] = check(
                None, reason="hf_repo not configured",
            )
        elif not check_hf:
            checks["checkpoint_on_hf"] = check(
                None,
                reason="skipped (call audit with check_hf=True to verify)",
            )
        elif latest_ckpt is None:
            checks["checkpoint_on_hf"] = check(
                None, reason="no local checkpoint to compare",
            )
        else:
            checks["checkpoint_on_hf"] = self._audit_hf(
                hf_repo, trial.run_dir, latest_ckpt.name
            )

        return {
            "trial_id": trial.trial_id,
            "strategy": trial.strategy,
            "status": trial.status,
            "checks": checks,
        }

    @staticmethod
    def _audit_hf(
        hf_repo: str, run_dir: str | None, ckpt_name: str,
    ) -> dict[str, Any]:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            return {"pass": None, "reason": f"huggingface_hub unavailable: {e}"}
        try:
            api = HfApi()
            branch = (
                f"run/{Path(run_dir).name}" if run_dir else "main"
            )
            files = api.list_repo_files(
                hf_repo, repo_type="model", revision=branch
            )
            present = any(f"{ckpt_name}/" in f"/{f}" for f in files)
            return {
                "pass": present,
                "branch": branch,
                "checkpoint": ckpt_name,
                "n_files_on_branch": len(files),
            }
        except Exception as e:
            return {"pass": None, "reason": f"HF list failed: {e}"}

    # =======================================================================
    # Progress log
    # =======================================================================

    def render_progress_log(self) -> str:
        """Render pod_manager.md from current state."""
        lines: list[str] = ["# Pod Manager Log\n"]

        lines.append("## Environment")
        lines.append(f"- GPUs: {self.gpu_count}x {self.gpu_names[0] if self.gpu_names else '?'}, "
                      f"{self.gpu_vram_mb[0] if self.gpu_vram_mb else '?'} MB each")
        lines.append(f"- Persistent storage: {self.workspace}")
        lines.append("")

        elapsed = time.time() - self.start_time
        lines.append("## Current Status")
        lines.append(f"- Uptime: {_format_duration(elapsed)}")
        if self.cost_per_hour:
            cost = self.cost_per_hour * elapsed / 3600
            lines.append(f"- Cost: ${self.cost_per_hour}/hr, ~${cost:.2f} so far")
        lines.append("")

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

        completed = [t for t in self.trials.values()
                     if t.status in ("completed", "failed", "killed")]
        if completed:
            lines.append("## Results")
            lines.append("| Trial | Strategy | Params | val_loss | Acc | Status | Notes |")
            lines.append("|-------|----------|--------|----------|-----|--------|-------|")
            for t in sorted(completed, key=lambda t: t.trial_id):
                vl = f"{t.best_val_loss:.4f}" if t.best_val_loss else "---"
                acc = f"{t.best_accuracy:.1%}" if t.best_accuracy else "---"
                pc = f"{t.actual_param_count:,}" if t.actual_param_count else "?"
                lines.append(
                    f"| {t.trial_id} | {t.strategy} | {pc} | {vl} "
                    f"| {acc} | {t.status} | {t.notes} |"
                )
            lines.append("")

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

        if self.events_path.exists():
            self.events = []
            for line in self.events_path.read_text().splitlines():
                try:
                    self.events.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
            if self.events:
                self.event_seq = max(e.get("seq", 0) for e in self.events)

        for trial_id, trial in self.trials.items():
            if trial.status != "running":
                continue
            if trial.pid:
                alive, _ = is_alive(trial.pid)
            else:
                alive = False
            if alive:
                log.info("Recovering trial %d (PID %d)", trial_id, trial.pid)
                self._monitor_tasks[trial_id] = asyncio.create_task(
                    self._monitor(trial_id)
                )
                if trial.gpu_id is not None:
                    self._assign_gpu(trial_id, trial.gpu_id)
            else:
                log.warning("Trial %d (PID %d) no longer running", trial_id, trial.pid)
                read_metrics(trial, self.log_dir, self._metrics_offsets, self._sps_windows)
                trial.end_time = time.time()
                if trial.best_val_loss is not None:
                    trial.status = "completed"
                    self._emit("trial_completed", trial_id, {
                        "best_val_loss": trial.best_val_loss,
                        "best_accuracy": trial.best_accuracy,
                        "param_count": trial.actual_param_count,
                        "steps": trial.current_step,
                        "recovered": True,
                    })
                else:
                    trial.status = "failed"
                    self._emit("trial_failed", trial_id, {
                        "reason": "process exited during server downtime",
                        "recovered": True,
                    })

        self._save_state()
        self.render_progress_log()
        log.info("Recovery complete: %d trials, %d still running",
                 len(self.trials),
                 sum(1 for t in self.trials.values() if t.status == "running"))

    def shutdown(self) -> None:
        """Save state on shutdown. Training processes continue independently."""
        for task in self._monitor_tasks.values():
            task.cancel()
        self._monitor_tasks.clear()
        self._save_state()
        self.render_progress_log()
        log.info("Runner shutdown (training processes continue)")
