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


def read_pretrain_val_summary(trial: Trial) -> dict[str, Any] | None:
    """Scan the trial's metrics.jsonl for the latest pretraining val record
    and compute a log-linear fit on forfeit rate over the most recent half
    of the history.

    Returns a dict with:
        latest: the latest val record's key fields (game_completion_rate,
            avg_plies_completed, forfeit min/max/median, legal, late_legal,
            val_loss, step)
        forfeit_fit: {slope, half_life_steps, n_points} computed from the
            most recent half of the (step, forfeit_rate) series — matches
            the dashboard's log-linear fit. Omitted if fewer than 4 val
            records have game_completion_rate (not a pretraining run or
            too early).

    Returns None if no val records are available.
    """
    if trial.run_dir is None:
        return None
    metrics_path = Path(trial.run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    steps: list[int] = []
    forfeit_rates: list[float] = []
    latest: dict[str, Any] | None = None

    try:
        with open(metrics_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if rec.get("type") != "val":
                    continue
                gc = rec.get("val/game_completion_rate")
                if gc is None:
                    continue
                step = rec.get("step")
                if step is None:
                    continue
                steps.append(int(step))
                forfeit_rates.append(1.0 - float(gc))
                latest = rec
    except OSError:
        return None

    if not latest:
        return None

    summary: dict[str, Any] = {
        "latest": {
            "step": latest.get("step"),
            "val_loss": latest.get("val/loss"),
            "game_completion_rate": latest.get("val/game_completion_rate"),
            "avg_plies_completed": latest.get("val/avg_plies_completed"),
            "forfeit_ply_min": latest.get("val/min_forfeit_ply"),
            "forfeit_ply_max": latest.get("val/max_forfeit_ply"),
            "forfeit_ply_median": latest.get("val/median_forfeit_ply"),
            "legal_move_rate": latest.get("val/legal_move_rate"),
            "late_legal_move_rate": latest.get("val/late_legal_move_rate"),
        },
    }

    n = len(steps)
    if n >= 4:
        half = n // 2
        xs = steps[half:]
        ys = forfeit_rates[half:]
        # Only fit on strictly positive forfeit rates
        pos = [(x, y) for x, y in zip(xs, ys) if y > 0]
        if len(pos) >= 3:
            import math as _math
            xs_f = [float(x) for x, _ in pos]
            ys_log = [_math.log(y) for _, y in pos]
            n_pts = len(xs_f)
            mean_x = sum(xs_f) / n_pts
            mean_y = sum(ys_log) / n_pts
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_f, ys_log))
            den = sum((x - mean_x) ** 2 for x in xs_f)
            if den > 0:
                slope = num / den
                half_life = _math.log(2) / abs(slope) if slope != 0 else None
                summary["forfeit_fit"] = {
                    "slope_per_step": slope,
                    "half_life_steps": half_life,
                    "n_points": n_pts,
                    "current_forfeit": forfeit_rates[-1],
                }

    return summary


def check_health(trial: Trial) -> str | None:
    """Return a health issue string, or None if healthy."""
    loss = trial.last_train_loss
    if loss is not None and (math.isnan(loss) or math.isinf(loss)):
        threshold = min(500, trial.total_steps // 5) if trial.total_steps > 0 else 500
        if trial.current_step > threshold:
            return "NaN/Inf loss"
    return None
