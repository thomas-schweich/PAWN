"""Metrics reading, health checks, and process liveness."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from pawn.lab.state import Trial

log = logging.getLogger("pawn.lab")


def is_alive(pid: int) -> tuple[bool, int | None]:
    """Check if a process is alive. Returns (alive, exit_code).

    Reaps zombies as a side effect. exit_code is set when the process
    was reaped (our child) or None if still alive / not our child.
    """
    # First try to reap — if the process is our child zombie, waitpid clears it
    try:
        rpid, status = os.waitpid(pid, os.WNOHANG)
        if rpid != 0:
            code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
            return False, code
    except ChildProcessError:
        pass  # not our child, fall through to kill check
    # Check via signal
    try:
        os.kill(pid, 0)
        return True, None
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, None  # exists but can't signal


def read_metrics(
    trial: Trial,
    log_dir: Path,
    offsets: dict[int, int],
) -> None:
    """Read new lines from the trial's metrics.jsonl, updating trial in-place."""
    # Find run dir if not yet discovered — pick the most recent
    if trial.run_dir is None:
        trial_log_dir = log_dir / f"trial_{trial.trial_id:04d}"
        metrics_files = sorted(
            trial_log_dir.glob("*/metrics.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if metrics_files:
            trial.run_dir = str(metrics_files[-1].parent)
    if trial.run_dir is None:
        return

    metrics_path = Path(trial.run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return

    offset = offsets.get(trial.trial_id, 0)
    try:
        with open(metrics_path) as f:
            f.seek(offset)
            new_lines = f.readlines()
            offsets[trial.trial_id] = f.tell()
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
            train_acc = rec.get("train/accuracy") or rec.get("train_top1")
            if train_acc is not None:
                trial.last_train_acc = train_acc
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


def check_health(trial: Trial) -> str | None:
    """Return a health issue string, or None if healthy."""
    loss = trial.last_train_loss
    if loss is not None and (math.isnan(loss) or math.isinf(loss)):
        threshold = min(500, trial.total_steps // 5) if trial.total_steps > 0 else 500
        if trial.current_step > threshold:
            return "NaN/Inf loss"
    return None
