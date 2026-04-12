"""Tests for pawn.lab.monitor — metrics reading, health checks, process liveness."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from pawn.lab.monitor import (
    check_health,
    is_alive,
    read_metrics,
    read_pretrain_val_summary,
)
from pawn.lab.state import Trial


# =====================================================================
# is_alive
# =====================================================================


class TestIsAlive:
    def test_current_process_is_alive(self):
        alive, code = is_alive(os.getpid())
        assert alive is True
        assert code is None

    def test_pid_one_is_alive_or_permission_denied(self):
        """init (pid 1) is always alive on Linux."""
        alive, _ = is_alive(1)
        assert alive is True

    def test_nonexistent_pid_is_dead(self):
        # High PID that's extremely unlikely to exist
        alive, code = is_alive(2**30)
        assert alive is False
        assert code is None

    def test_returns_tuple(self):
        result = is_alive(os.getpid())
        assert isinstance(result, tuple)
        assert len(result) == 2


# =====================================================================
# read_metrics
# =====================================================================


def _make_trial(trial_id: int = 0) -> Trial:
    return Trial(
        trial_id=trial_id,
        strategy="test",
        params={},
        cli_command=[],
    )


def _write_metrics_file(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestReadMetrics:
    def test_no_run_dir_finds_metrics(self, tmp_path):
        """When trial.run_dir is None, read_metrics discovers it."""
        log_dir = tmp_path / "logs"
        # Expected layout: logs/trial_0000/run_xyz/metrics.jsonl
        metrics_path = log_dir / "trial_0000" / "run_foo" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "config", "total_steps": 100, "param_count": 1234},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.run_dir == str(metrics_path.parent)
        assert trial.total_steps == 100
        assert trial.actual_param_count == 1234

    def test_no_metrics_file_noops(self, tmp_path):
        trial = _make_trial(0)
        trial.run_dir = str(tmp_path / "nonexistent")
        offsets: dict[int, int] = {}
        read_metrics(trial, tmp_path / "logs", offsets)  # should not crash
        assert trial.total_steps == 0

    def test_no_trial_dir_noops(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        trial = _make_trial(42)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)
        assert trial.run_dir is None

    def test_parses_config_record(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "config", "total_steps": 500, "param_count": 9999},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.total_steps == 500
        assert trial.actual_param_count == 9999

    def test_parses_config_nested_training_field(self, tmp_path):
        """Some configs nest total_steps under `training.total_steps`."""
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "config", "training": {"total_steps": 250}},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)
        assert trial.total_steps == 250

    def test_parses_train_record(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 50, "train/loss": 2.5, "step_time": 0.1,
             "train/accuracy": 0.3},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.current_step == 50
        assert trial.last_train_loss == pytest.approx(2.5)
        assert trial.steps_per_sec == pytest.approx(10.0)  # 1/0.1
        assert trial.last_train_acc == pytest.approx(0.3)

    def test_parses_train_record_train_loss_alt_key(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 10, "train_loss": 1.5},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.last_train_loss == pytest.approx(1.5)

    def test_parses_val_record(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "val", "val/loss": 1.8, "val/accuracy": 0.5},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.best_val_loss == pytest.approx(1.8)
        assert trial.best_accuracy == pytest.approx(0.5)

    def test_val_best_loss_takes_minimum(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "val", "val/loss": 3.0},
            {"type": "val", "val/loss": 2.0},
            {"type": "val", "val/loss": 2.5},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.best_val_loss == pytest.approx(2.0)

    def test_malformed_json_is_skipped(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({"type": "train", "step": 10, "train/loss": 1.0}) + "\n")
            f.write("{broken}\n")

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.current_step == 10
        assert trial.last_train_loss == pytest.approx(1.0)

    def test_offset_skips_already_read(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 10, "train/loss": 1.0},
            {"type": "train", "step": 20, "train/loss": 0.8},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)
        assert trial.current_step == 20

        # Now append more
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"type": "train", "step": 30, "train/loss": 0.6}) + "\n")

        read_metrics(trial, log_dir, offsets)
        assert trial.current_step == 30
        assert trial.last_train_loss == pytest.approx(0.6)

    def test_empty_metrics_file_is_ok(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.touch()

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)
        assert trial.current_step == 0

    def test_step_time_zero_uses_elapsed(self, tmp_path):
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 20, "train/loss": 1.0,
             "step_time": 0, "elapsed": 10.0},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        read_metrics(trial, log_dir, offsets)

        # 20 / 10 = 2.0 steps/sec
        assert trial.steps_per_sec == pytest.approx(2.0)


# =====================================================================
# read_metrics — cotrain
# =====================================================================


class TestReadMetricsCotrain:
    def _make_cotrain_trial(self, trial_id: int = 0) -> Trial:
        return Trial(
            trial_id=trial_id,
            strategy="cotrain:small+base",
            params={},
            cli_command=[],
            config={"run_type": "cotrain"},
        )

    def test_discovers_multiple_variant_dirs(self, tmp_path):
        """Cotrain read_metrics discovers per-variant JSONL files."""
        log_dir = tmp_path / "logs"
        trial_dir = log_dir / "trial_0000"

        # Create two variant run dirs with metrics
        for name in ("small", "base"):
            metrics_path = trial_dir / f"run_20260410_120000_{name}_calm-crane" / "metrics.jsonl"
            _write_metrics_file(metrics_path, [
                {"type": "config", "total_steps": 100, "param_count": 1000},
                {"type": "train", "step": 10, "train/loss": 2.5, "step_time": 0.1},
                {"type": "val", "step": 10, "val/loss": 2.0, "val/accuracy": 0.4},
            ])

        trial = self._make_cotrain_trial(0)
        offsets: dict = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.run_dir == str(trial_dir)
        assert trial.variants is not None
        assert "small" in trial.variants
        assert "base" in trial.variants
        assert trial.variants["small"]["current_step"] == 10
        assert trial.variants["base"]["current_step"] == 10

    def test_aggregates_current_step_as_min(self, tmp_path):
        """Trial.current_step is min across variants."""
        log_dir = tmp_path / "logs"
        trial_dir = log_dir / "trial_0000"

        small_path = trial_dir / "run_20260410_120000_small_calm-crane" / "metrics.jsonl"
        _write_metrics_file(small_path, [
            {"type": "train", "step": 50, "train/loss": 2.0, "step_time": 0.1},
        ])
        base_path = trial_dir / "run_20260410_120000_base_calm-crane" / "metrics.jsonl"
        _write_metrics_file(base_path, [
            {"type": "train", "step": 30, "train/loss": 2.5, "step_time": 0.1},
        ])

        trial = self._make_cotrain_trial(0)
        offsets: dict = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.current_step == 30  # min of 50, 30

    def test_aggregates_best_val_loss_as_min(self, tmp_path):
        """Trial.best_val_loss is min across variants."""
        log_dir = tmp_path / "logs"
        trial_dir = log_dir / "trial_0000"

        small_path = trial_dir / "run_20260410_120000_small_calm-crane" / "metrics.jsonl"
        _write_metrics_file(small_path, [
            {"type": "val", "step": 10, "val/loss": 3.0},
        ])
        base_path = trial_dir / "run_20260410_120000_base_calm-crane" / "metrics.jsonl"
        _write_metrics_file(base_path, [
            {"type": "val", "step": 10, "val/loss": 2.0},
        ])

        trial = self._make_cotrain_trial(0)
        offsets: dict = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.best_val_loss == pytest.approx(2.0)

    def test_per_variant_offsets_incremental(self, tmp_path):
        """Cotrain uses (trial_id, variant_name) offset keys for incremental reads."""
        log_dir = tmp_path / "logs"
        trial_dir = log_dir / "trial_0000"

        small_path = trial_dir / "run_20260410_120000_small_calm-crane" / "metrics.jsonl"
        _write_metrics_file(small_path, [
            {"type": "train", "step": 10, "train/loss": 2.5, "step_time": 0.1},
        ])

        trial = self._make_cotrain_trial(0)
        offsets: dict = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.variants["small"]["current_step"] == 10

        # Append more data
        with open(small_path, "a") as f:
            f.write(json.dumps({"type": "train", "step": 20, "train/loss": 2.0, "step_time": 0.1}) + "\n")

        read_metrics(trial, log_dir, offsets)
        assert trial.variants["small"]["current_step"] == 20

    def test_underscore_in_variant_name(self, tmp_path):
        """Variant names containing underscores are parsed correctly."""
        log_dir = tmp_path / "logs"
        trial_dir = log_dir / "trial_0000"

        metrics_path = trial_dir / "run_20260410_120000_my_model_calm-crane" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 5, "train/loss": 3.0, "step_time": 0.2},
        ])

        trial = self._make_cotrain_trial(0)
        offsets: dict = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.variants is not None
        assert "my_model" in trial.variants
        assert trial.variants["my_model"]["current_step"] == 5

    def test_empty_trial_dir_noops(self, tmp_path):
        log_dir = tmp_path / "logs"
        (log_dir / "trial_0000").mkdir(parents=True)

        trial = self._make_cotrain_trial(0)
        offsets: dict = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.variants is None
        assert trial.run_dir is None

    def test_variant_run_dir_tracked(self, tmp_path):
        """Each variant's run_dir is stored independently."""
        log_dir = tmp_path / "logs"
        trial_dir = log_dir / "trial_0000"

        small_dir = trial_dir / "run_20260410_120000_small_calm-crane"
        metrics_path = small_dir / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 5, "train/loss": 3.0, "step_time": 0.2},
        ])

        trial = self._make_cotrain_trial(0)
        offsets: dict = {}
        read_metrics(trial, log_dir, offsets)

        assert trial.variants["small"]["run_dir"] == str(small_dir)


# =====================================================================
# check_health
# =====================================================================


class TestCheckHealth:
    def test_no_loss_is_healthy(self):
        t = _make_trial()
        t.last_train_loss = None
        assert check_health(t) is None

    def test_normal_loss_is_healthy(self):
        t = _make_trial()
        t.last_train_loss = 2.5
        t.current_step = 1000
        t.total_steps = 10000
        assert check_health(t) is None

    def test_nan_loss_after_warmup_is_unhealthy(self):
        t = _make_trial()
        t.last_train_loss = float("nan")
        t.total_steps = 10000
        t.current_step = 600  # > 500 threshold
        issue = check_health(t)
        assert issue is not None
        # The message must mention NaN or Inf (the specific pathological value)
        assert "NaN" in issue or "nan" in issue.lower(), (
            f"Health issue for NaN loss should mention 'NaN', got: {issue!r}"
        )

    def test_inf_loss_after_warmup_is_unhealthy(self):
        t = _make_trial()
        t.last_train_loss = float("inf")
        t.total_steps = 10000
        t.current_step = 600
        issue = check_health(t)
        assert issue is not None
        # The message must mention Inf (the specific pathological value)
        assert "Inf" in issue or "inf" in issue.lower(), (
            f"Health issue for Inf loss should mention 'Inf', got: {issue!r}"
        )

    def test_nan_loss_during_warmup_is_tolerated(self):
        t = _make_trial()
        t.last_train_loss = float("nan")
        t.total_steps = 10000
        t.current_step = 100  # below threshold
        assert check_health(t) is None

    def test_threshold_uses_min_of_500_and_total_div_5(self):
        """threshold = min(500, total_steps // 5) when total_steps > 0."""
        t = _make_trial()
        t.last_train_loss = float("nan")
        t.total_steps = 1000  # total // 5 = 200 -> threshold = 200
        t.current_step = 150  # below threshold
        assert check_health(t) is None
        t.current_step = 250  # above threshold
        issue = check_health(t)
        assert issue is not None
        assert isinstance(issue, str) and len(issue) > 0
        assert "NaN" in issue or "Inf" in issue

    def test_no_total_steps_uses_500_threshold(self):
        t = _make_trial()
        t.last_train_loss = float("nan")
        t.total_steps = 0
        t.current_step = 400
        assert check_health(t) is None
        t.current_step = 600
        issue = check_health(t)
        assert issue is not None
        assert isinstance(issue, str) and len(issue) > 0
        assert "NaN" in issue or "Inf" in issue


# =====================================================================
# read_pretrain_val_summary
# =====================================================================


def _val_record(step: int, gc: float, **kwargs) -> dict:
    """Build a minimal val record with game_completion_rate."""
    rec = {
        "type": "val",
        "step": step,
        "val/loss": 3.0,
        "val/game_completion_rate": gc,
        "val/avg_plies_completed": 300.0,
        "val/min_forfeit_ply": 10.0,
        "val/max_forfeit_ply": 400.0,
        "val/median_forfeit_ply": 100.0,
        "val/legal_move_rate": 0.997,
        "val/late_legal_move_rate": 0.993,
    }
    rec.update(kwargs)
    return rec


class TestReadPretrainValSummary:
    def test_no_run_dir_returns_none(self):
        trial = _make_trial()
        assert read_pretrain_val_summary(trial) is None

    def test_missing_metrics_file_returns_none(self, tmp_path):
        trial = _make_trial()
        trial.run_dir = str(tmp_path / "does_not_exist")
        assert read_pretrain_val_summary(trial) is None

    def test_no_val_records_returns_none(self, tmp_path):
        run_dir = tmp_path / "run_x"
        _write_metrics_file(run_dir / "metrics.jsonl", [
            {"type": "train", "step": 10, "train/loss": 2.0},
        ])
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        assert read_pretrain_val_summary(trial) is None

    def test_val_records_without_game_completion_return_none(self, tmp_path):
        """Adapter runs log val records but no game_completion_rate."""
        run_dir = tmp_path / "run_x"
        _write_metrics_file(run_dir / "metrics.jsonl", [
            {"type": "val", "step": 100, "val/loss": 2.0, "val/accuracy": 0.5},
            {"type": "val", "step": 200, "val/loss": 1.9, "val/accuracy": 0.55},
        ])
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        assert read_pretrain_val_summary(trial) is None

    def test_returns_latest_without_fit_when_too_few_records(self, tmp_path):
        """Need n >= 4 val records to even attempt the fit."""
        run_dir = tmp_path / "run_x"
        _write_metrics_file(run_dir / "metrics.jsonl", [
            _val_record(1000, 0.5),
            _val_record(2000, 0.4),
            _val_record(3000, 0.3),
        ])
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        summary = read_pretrain_val_summary(trial)

        assert summary is not None
        assert "latest" in summary
        assert summary["latest"]["step"] == 3000
        assert summary["latest"]["game_completion_rate"] == pytest.approx(0.3)
        assert "forfeit_fit" not in summary

    def test_returns_fit_on_known_series(self, tmp_path):
        """Construct a known exponential decay and verify OLS recovers it.

        We pick forfeit_rate(step) = exp(-k * step + b) so log(forfeit) is
        exactly linear in step with slope -k. The fit uses the second half
        of the history.
        """
        run_dir = tmp_path / "run_x"
        k = 1e-5  # half-life = ln(2)/k ~= 69314 steps
        b = math.log(0.5)  # forfeit(0) = 0.5

        records = []
        n = 20
        for i in range(n):
            step = (i + 1) * 1000
            forfeit = math.exp(-k * step + b)
            gc = 1.0 - forfeit
            records.append(_val_record(step, gc))

        _write_metrics_file(run_dir / "metrics.jsonl", records)
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        summary = read_pretrain_val_summary(trial)

        assert summary is not None
        assert "forfeit_fit" in summary
        fit = summary["forfeit_fit"]
        assert fit["slope_per_step"] == pytest.approx(-k, rel=1e-6)
        assert fit["half_life_steps"] == pytest.approx(math.log(2) / k, rel=1e-6)
        # Second half of n=20 → 10 points
        assert fit["n_points"] == 10
        # current_forfeit is the last overall series value, not the fit window's
        expected_current = math.exp(-k * (n * 1000) + b)
        assert fit["current_forfeit"] == pytest.approx(expected_current)

    def test_all_zero_forfeit_omits_fit(self, tmp_path):
        """If every forfeit rate is exactly 0 (perfect completion), the OLS
        window has no positive points to fit and forfeit_fit is omitted."""
        run_dir = tmp_path / "run_x"
        _write_metrics_file(run_dir / "metrics.jsonl", [
            _val_record(1000, 1.0),
            _val_record(2000, 1.0),
            _val_record(3000, 1.0),
            _val_record(4000, 1.0),
            _val_record(5000, 1.0),
        ])
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        summary = read_pretrain_val_summary(trial)

        assert summary is not None
        assert summary["latest"]["game_completion_rate"] == pytest.approx(1.0)
        assert "forfeit_fit" not in summary

    def test_latest_records_carries_all_fields(self, tmp_path):
        """All documented latest fields are present when available."""
        run_dir = tmp_path / "run_x"
        _write_metrics_file(run_dir / "metrics.jsonl", [
            _val_record(
                5000, 0.9,
                **{"val/avg_plies_completed": 321.5,
                   "val/min_forfeit_ply": 25.0,
                   "val/max_forfeit_ply": 300.0,
                   "val/median_forfeit_ply": 120.0,
                   "val/loss": 2.9,
                   "val/legal_move_rate": 0.996,
                   "val/late_legal_move_rate": 0.992}),
        ])
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        summary = read_pretrain_val_summary(trial)

        assert summary is not None
        latest = summary["latest"]
        assert latest["step"] == 5000
        assert latest["val_loss"] == pytest.approx(2.9)
        assert latest["game_completion_rate"] == pytest.approx(0.9)
        assert latest["avg_plies_completed"] == pytest.approx(321.5)
        assert latest["forfeit_ply_min"] == pytest.approx(25.0)
        assert latest["forfeit_ply_max"] == pytest.approx(300.0)
        assert latest["forfeit_ply_median"] == pytest.approx(120.0)
        assert latest["legal_move_rate"] == pytest.approx(0.996)
        assert latest["late_legal_move_rate"] == pytest.approx(0.992)

    def test_current_forfeit_is_last_of_full_series(self, tmp_path):
        """`current_forfeit` tracks the unfiltered last value, even when it's
        zero and got dropped from the OLS window."""
        run_dir = tmp_path / "run_x"
        # 10 decaying records, then a final record at 100% completion (forfeit=0)
        records = [_val_record(i * 1000, 1.0 - math.exp(-i * 0.2)) for i in range(1, 11)]
        records.append(_val_record(11_000, 1.0))  # forfeit = 0.0
        _write_metrics_file(run_dir / "metrics.jsonl", records)
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        summary = read_pretrain_val_summary(trial)

        assert summary is not None
        # If forfeit_fit was computed, current_forfeit should be the unfiltered last value
        if "forfeit_fit" in summary:
            assert summary["forfeit_fit"]["current_forfeit"] == pytest.approx(0.0)
