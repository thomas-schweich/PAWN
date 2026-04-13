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
    offsets: dict,
) -> None:
    """Read new lines from the trial's metrics.jsonl, updating trial in-place.

    For cotrain trials, discovers multiple per-variant metrics files and
    aggregates them to the trial level while tracking per-variant state in
    ``trial.variants``.

    ``offsets`` keys are ``int`` (trial_id) for single-variant trials, or
    ``(trial_id, variant_name)`` for cotrain per-variant files.
    """
    is_cotrain = (trial.config or {}).get("run_type") == "cotrain"

    if is_cotrain:
        _read_cotrain_metrics(trial, log_dir, offsets)
    else:
        _read_single_metrics(trial, log_dir, offsets)


def _read_single_metrics(
    trial: Trial,
    log_dir: Path,
    offsets: dict,
) -> None:
    """Read metrics for a single-variant (pretrain/adapter) trial."""
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


def _read_cotrain_metrics(
    trial: Trial,
    log_dir: Path,
    offsets: dict,
) -> None:
    """Read metrics for a cotrain trial (multiple per-variant JSONL files)."""
    trial_log_dir = log_dir / f"trial_{trial.trial_id:04d}"

    # Discover all per-variant metrics files under the trial dir.
    # Each variant's MetricsLogger creates a run dir with suffix=variant_name,
    # e.g. run_20260410_151230_zesty-osprey_small/metrics.jsonl
    metrics_files = list(trial_log_dir.glob("*/metrics.jsonl"))
    if not metrics_files:
        return

    # Set trial.run_dir to the parent trial dir (not a specific variant)
    if trial.run_dir is None:
        trial.run_dir = str(trial_log_dir)

    # Initialize variants dict if needed
    if trial.variants is None:
        trial.variants = {}

    # Extract variant name from the run dir suffix: run_..._<name>/metrics.jsonl
    # The MetricsLogger uses suffix=name, producing dirs like
    # run_YYYYMMDD_HHMMSS_slug_variantname/
    for mf in metrics_files:
        variant_name = _extract_variant_name(mf.parent.name)
        if variant_name is None:
            continue

        # Initialize this variant's state dict
        if variant_name not in trial.variants:
            trial.variants[variant_name] = {
                "name": variant_name,
                "run_dir": str(mf.parent),
                "current_step": 0,
                "last_train_loss": None,
                "last_train_acc": None,
                "best_val_loss": None,
                "best_val_step": 0,
                "best_accuracy": None,
                "actual_param_count": None,
                "stopped": False,
                "steps_per_sec": 0.0,
            }

        vs = trial.variants[variant_name]
        offset_key = (trial.trial_id, variant_name)
        offset = offsets.get(offset_key, 0)

        try:
            with open(mf) as f:
                f.seek(offset)
                new_lines = f.readlines()
                offsets[offset_key] = f.tell()
        except OSError:
            continue

        for line in new_lines:
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            rtype = rec.get("type")
            if rtype == "config":
                ts = rec.get("total_steps") or (rec.get("training") or {}).get("total_steps")
                if ts:
                    trial.total_steps = ts
                pc = rec.get("param_count")
                if pc is not None:
                    vs["actual_param_count"] = pc

            elif rtype == "train":
                vs["current_step"] = rec.get("step", vs["current_step"])
                loss = rec.get("train/loss") or rec.get("train_loss")
                if loss is not None:
                    vs["last_train_loss"] = loss
                train_acc = rec.get("train/accuracy") or rec.get("train_top1")
                if train_acc is not None:
                    vs["last_train_acc"] = train_acc
                st = rec.get("step_time")
                if st and st > 0:
                    vs["steps_per_sec"] = 1.0 / st
                elif rec.get("elapsed") and vs["current_step"] > 0:
                    vs["steps_per_sec"] = vs["current_step"] / rec["elapsed"]

            elif rtype == "val":
                vl = rec.get("val/loss") or rec.get("val_loss") or rec.get("loss")
                if vl is not None and (vs["best_val_loss"] is None or vl < vs["best_val_loss"]):
                    vs["best_val_loss"] = vl
                    vs["best_val_step"] = rec.get("step", vs.get("best_val_step", 0))
                acc = (rec.get("val/accuracy") or rec.get("val_top1")
                       or rec.get("accuracy"))
                if acc is not None:
                    vs["best_accuracy"] = acc

    # Aggregate to trial level
    _aggregate_cotrain_metrics(trial)


def _extract_variant_name(run_dir_name: str) -> str | None:
    """Extract variant name from a run directory name.

    The MetricsLogger creates dirs like ``run_YYYYMMDD_HHMMSS_variantname_slug``.
    The layout is: ``run`` _ ``date`` _ ``time`` _ ``variant`` _ ``slug``.
    The variant name may itself contain underscores, but the slug (final segment)
    never does (it's two hyphenated words like ``calm-crane``).  So we rejoin
    everything between parts[3] and parts[-1].
    """
    # Expected: run_YYYYMMDD_HHMMSS_variant_slug (at least 5 parts)
    parts = run_dir_name.split("_")
    if len(parts) < 5 or parts[0] != "run":
        return None
    # parts[1]=date, parts[2]=time, parts[-1]=slug, parts[3:-1]=variant
    return "_".join(parts[3:-1])


def _aggregate_cotrain_metrics(trial: Trial) -> None:
    """Aggregate per-variant metrics to the trial level."""
    if not trial.variants:
        return

    variants = list(trial.variants.values())

    # current_step = min across variants (honest ETA — slowest determines progress)
    steps = [v["current_step"] for v in variants if v["current_step"] > 0]
    if steps:
        trial.current_step = min(steps)

    # best_val_loss = min across variants
    val_losses = [v["best_val_loss"] for v in variants if v["best_val_loss"] is not None]
    if val_losses:
        trial.best_val_loss = min(val_losses)

    # best_accuracy = max across variants
    accs = [v["best_accuracy"] for v in variants if v["best_accuracy"] is not None]
    if accs:
        trial.best_accuracy = max(accs)

    # last_train_loss = mean across active variants
    losses = [v["last_train_loss"] for v in variants
              if v["last_train_loss"] is not None and not v.get("stopped")]
    if losses:
        trial.last_train_loss = sum(losses) / len(losses)

    # last_train_acc = mean across active variants
    accs_train = [v["last_train_acc"] for v in variants
                  if v.get("last_train_acc") is not None and not v.get("stopped")]
    if accs_train:
        trial.last_train_acc = sum(accs_train) / len(accs_train)

    # steps_per_sec from any variant (they share the same step timing)
    for v in variants:
        if v.get("steps_per_sec", 0) > 0:
            trial.steps_per_sec = v["steps_per_sec"]
            break


def _read_val_forfeit_summary(metrics_path: Path) -> dict[str, Any] | None:
    """Scan a single metrics.jsonl for forfeit / game-completion stats.

    Returns ``{"latest": {...}, "forfeit_fit": {...}}`` or ``None`` if the
    file has no val records with ``val/game_completion_rate``. The fit
    block is omitted when there are fewer than 4 val records total or
    fewer than 3 strictly positive forfeit points in the half-window.

    Shared by :func:`read_pretrain_val_summary` (single metrics.jsonl) and
    :func:`read_cotrain_val_summary` (one call per variant).
    """
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
            xs_f = [float(x) for x, _ in pos]
            ys_log = [math.log(y) for _, y in pos]
            n_pts = len(xs_f)
            mean_x = sum(xs_f) / n_pts
            mean_y = sum(ys_log) / n_pts
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_f, ys_log))
            den = sum((x - mean_x) ** 2 for x in xs_f)
            if den > 0:
                slope = num / den
                half_life = math.log(2) / abs(slope) if slope != 0 else None
                summary["forfeit_fit"] = {
                    "slope_per_step": slope,
                    "half_life_steps": half_life,
                    "n_points": n_pts,
                    "current_forfeit": forfeit_rates[-1],
                }

    return summary


def read_pretrain_val_summary(trial: Trial) -> dict[str, Any] | None:
    """Scan the trial's metrics.jsonl for the latest pretraining val record
    and compute a log-linear fit on forfeit rate over the most recent half
    of the history.

    Returns a dict with:
        latest: the latest val record's key fields (game_completion_rate,
            avg_plies_completed, forfeit min/max/median, legal, late_legal,
            val_loss, step)
        forfeit_fit: {slope_per_step, half_life_steps, n_points,
            current_forfeit} computed from the most recent half of the
            (step, forfeit_rate) series — matches the dashboard's
            log-linear fit. The OLS itself is restricted to strictly
            positive forfeit rates (log(0) would blow up), but
            `current_forfeit` is always the last observed forfeit rate
            from the full series — including 0.0 if the most recent eval
            had no forfeits. Omitted if fewer than 4 val records have
            game_completion_rate (not a pretraining run or too early),
            or if fewer than 3 positive-forfeit points land in the
            half-window.

    Returns None if no val records are available.
    """
    if trial.run_dir is None:
        return None
    return _read_val_forfeit_summary(Path(trial.run_dir) / "metrics.jsonl")


def read_cotrain_val_summary(trial: Trial) -> dict[str, Any] | None:
    """Per-variant forfeit / log-linear summary for a cotrain trial.

    Returns ``{"variants": {name: summary, ...}}`` where each variant
    summary matches the shape of :func:`read_pretrain_val_summary` —
    ``{"latest": {...}, "forfeit_fit": {...}}`` keyed by variant name
    (e.g. ``small`` / ``base`` / ``large``). Variants with no
    game-completion records yet are omitted. Returns ``None`` if the
    trial has no variants registered or none of them have any
    forfeit-rate data yet.
    """
    if not trial.variants:
        return None

    variants_out: dict[str, dict[str, Any]] = {}
    for name, vs in trial.variants.items():
        run_dir = vs.get("run_dir")
        if not run_dir:
            continue
        summary = _read_val_forfeit_summary(Path(run_dir) / "metrics.jsonl")
        if summary is not None:
            variants_out[name] = summary

    if not variants_out:
        return None
    return {"variants": variants_out}


def check_health(trial: Trial) -> str | None:
    """Return a health issue string, or None if healthy."""
    loss = trial.last_train_loss
    if loss is not None and (math.isnan(loss) or math.isinf(loss)):
        threshold = min(500, trial.total_steps // 5) if trial.total_steps > 0 else 500
        if trial.current_step > threshold:
            return "NaN/Inf loss"
    return None
