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

from pawn.lab.monitor import check_health, is_alive, read_metrics
from pawn.lab.state import Trial, _format_duration, _now_iso

log = logging.getLogger("pawn.lab")


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
        self.start_time: float = time.time()
        self.cost_per_hour: float | None = None

        # Async
        self._monitor_tasks: dict[int, asyncio.Task[None]] = {}
        self._metrics_offsets: dict[int, int] = {}

        self._ensure_dirs()
        self._gpus_discovered = False

    # =======================================================================
    # Setup
    # =======================================================================

    def _ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _discover_gpus(self) -> None:
        """Detect GPUs via a subprocess to avoid loading torch into this process.

        Torch's ROCm/HIP runtime spawns background threads that busy-spin,
        burning ~30% CPU permanently. Running discovery in a subprocess
        keeps the MCP server process clean.
        """
        if self._gpus_discovered:
            return
        self._gpus_discovered = True
        try:
            out = subprocess.check_output(
                [
                    self.python.split()[0], "-c",
                    "import json, torch; "
                    "gpus = [{"
                    "'name': torch.cuda.get_device_name(i), "
                    "'vram_mb': torch.cuda.get_device_properties(i).total_memory // (1024*1024)"
                    "} for i in range(torch.cuda.device_count())]; "
                    "print(json.dumps(gpus))",
                ],
                text=True, timeout=30,
            )
            gpus = json.loads(out.strip())
            self.gpu_count = len(gpus)
            for i, g in enumerate(gpus):
                self.gpu_names.append(g["name"])
                self.gpu_vram_mb.append(g["vram_mb"])
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

    def _find_free_gpu(self) -> int | None:
        self._discover_gpus()
        for gpu_id, trial_id in self.gpu_assignments.items():
            if trial_id is None:
                return gpu_id
        return None

    def _assign_gpu(self, trial_id: int, gpu_id: int) -> None:
        self.gpu_assignments[gpu_id] = trial_id

    def _release_gpu(self, gpu_id: int) -> None:
        self.gpu_assignments[gpu_id] = None

    def gpu_utilization(self) -> list[dict[str, Any]]:
        """Return GPU info without importing torch into this process."""
        self._discover_gpus()
        return [
            {
                "gpu": i,
                "total_mb": self.gpu_vram_mb[i] if i < len(self.gpu_vram_mb) else 0,
                "assigned_trial": self.gpu_assignments.get(i),
            }
            for i in range(self.gpu_count)
        ]

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

        merged_params = dict(params or {})
        merged_base = dict(base_args or {})
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

    async def resume_trial(
        self,
        trial_id: int,
        total_steps: int | None = None,
        base_args_overrides: dict[str, Any] | None = None,
    ) -> int:
        """Resume a completed/failed trial from its best checkpoint.

        Creates a new trial with the same strategy, params, and base_args,
        plus --resume pointing to the best checkpoint. Optionally override
        total_steps or other base_args for the continuation.
        """
        old = self.trials.get(trial_id)
        if not old:
            raise RuntimeError(f"Trial {trial_id} not found")
        if not old.run_dir:
            raise RuntimeError(f"Trial {trial_id} has no run directory")

        # Find best checkpoint
        ckpt_dir = Path(old.run_dir) / "checkpoints" / "best"
        if not ckpt_dir.exists():
            ckpt_dir = Path(old.run_dir) / "checkpoints" / "final"
        if not ckpt_dir.exists():
            raise RuntimeError(f"No checkpoint found for trial {trial_id}")

        # Build new base_args from old trial, with overrides.
        # Clear pause_after_steps by default so resumed trials don't
        # re-pause at the same step. The caller can set a new pause.
        new_base = dict(old.base_args)
        new_base.pop("pause_after_steps", None)
        new_base["resume"] = str(ckpt_dir)
        if total_steps is not None:
            new_base["total_steps"] = total_steps
        if base_args_overrides:
            new_base.update(base_args_overrides)

        return await self.launch(old.strategy, dict(old.params), new_base)

    def _build_command(
        self, strategy: str, params: dict[str, Any], base_args: dict[str, Any],
        trial_id: int,
    ) -> list[str]:
        script = str(self.code_dir / "scripts" / "train_adapter.py")
        trial_log_dir = str(self.log_dir / f"trial_{trial_id:04d}")

        cmd = [*self.python.split(), script, "--strategy", strategy]
        ba = dict(base_args)
        ba.setdefault("log_dir", trial_log_dir)
        ba.setdefault("local_checkpoints", True)
        cmd.extend(_params_to_cli(ba))
        cmd.extend(_params_to_cli(params))
        return cmd

    async def _spawn(self, trial: Trial) -> None:
        """Start the training process with GPU isolation."""
        env = os.environ.copy()
        if trial.gpu_id is not None:
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

                read_metrics(trial, self.log_dir, self._metrics_offsets)
                issue = check_health(trial)
                if issue:
                    log.warning("Trial %d health issue: %s", trial_id, issue)
                    self._emit("health_warning", trial_id, {"issue": issue})

            # Process exited — final metrics read
            read_metrics(trial, self.log_dir, self._metrics_offsets)

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
                read_metrics(trial, self.log_dir, self._metrics_offsets)
                self._save_state()
            elif exit_code == 0 or trial.best_val_loss is not None:
                self._complete(trial_id)
            else:
                reason = f"exit code {exit_code}" if exit_code is not None else "process exited"
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
        trial.status = "killed"
        trial.end_time = time.time()
        self._emit("trial_killed", trial_id)
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
        try:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError as e:
            log.error("Failed to write event: %s", e)

    def events_since(self, seq: int = 0) -> list[dict[str, Any]]:
        return [e for e in self.events if e["seq"] > seq]

    # =======================================================================
    # Reporting
    # =======================================================================

    def status(self) -> dict[str, Any]:
        running = []
        for t in self.trials.values():
            if t.status == "running":
                running.append({
                    "trial": t.trial_id, "strategy": t.strategy,
                    "step": t.current_step, "total": t.total_steps,
                    "sps": round(t.steps_per_sec, 2),
                    "eta": _format_duration(t.eta_seconds()),
                    "val_loss": t.best_val_loss, "acc": t.best_accuracy,
                    "params": t.actual_param_count, "pid": t.pid, "gpu": t.gpu_id,
                    "key_hp": {k: v for k, v in t.params.items()
                               if k in ("lr", "lora_rank", "bottleneck_dim",
                                        "density", "d_model", "n_layers", "batch_size")},
                })
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

    def results(self, strategy: str) -> dict[str, Any]:
        rows = []
        for t in sorted(self.trials.values(), key=lambda t: t.trial_id):
            elapsed = (t.end_time - t.start_time) if t.end_time and t.start_time else None
            rows.append({
                "trial": t.trial_id, "strategy": t.strategy,
                "params": t.actual_param_count, "steps": t.current_step,
                "val_loss": t.best_val_loss, "accuracy": t.best_accuracy,
                "status": t.status, "notes": t.notes,
                "wall_time": _format_duration(elapsed),
                "key_hp": {k: v for k, v in t.params.items()
                           if k in ("lr", "lora_rank", "bottleneck_dim", "density",
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

        result: dict[str, Any] = {"trials": rows, "pareto_front": pareto}
        result["suggestions"] = self._suggest(strategy, completed)
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
                read_metrics(trial, self.log_dir, self._metrics_offsets)
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
