"""Tests for pawn.lab.monitor — metrics reading, health checks, process liveness."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from pawn.lab.monitor import (
    check_health,
    fit_power_law,
    is_alive,
    read_cotrain_val_summary,
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
        # ``steps_per_sec`` is computed from a rolling window over
        # ``elapsed`` rather than per-record ``step_time`` (the old
        # path inflated sps by ~30× via log_interval-scale quantization).
        # Single train record → cumulative-rate fallback.
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 50, "elapsed": 5.0,
             "train/loss": 2.5, "train/accuracy": 0.3},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        sps_windows: dict = {}
        read_metrics(trial, log_dir, offsets, sps_windows)

        assert trial.current_step == 50
        assert trial.last_train_loss == pytest.approx(2.5)
        # 50 steps in 5s → 10 sps via the cumulative fallback.
        assert trial.steps_per_sec == pytest.approx(10.0)
        assert trial.last_train_acc == pytest.approx(0.3)

    def test_steps_per_sec_uses_window_delta(self, tmp_path):
        """Two-plus train records → window difference, not per-record
        ``1 / step_time``. This is the fix for the lab_status sps bug
        where log_interval-scale quantization inflated readings ~30×."""
        log_dir = tmp_path / "logs"
        metrics_path = log_dir / "trial_0000" / "run_x" / "metrics.jsonl"
        _write_metrics_file(metrics_path, [
            {"type": "train", "step": 100, "elapsed": 20.0,
             "train/loss": 2.0, "step_time": 0.05},
            {"type": "train", "step": 200, "elapsed": 40.0,
             "train/loss": 1.9, "step_time": 0.05},
        ])

        trial = _make_trial(0)
        offsets: dict[int, int] = {}
        sps_windows: dict = {}
        read_metrics(trial, log_dir, offsets, sps_windows)

        # (200 - 100) / (40 - 20) = 5 sps. The misleading
        # ``1 / step_time = 20`` reading is gone.
        assert trial.steps_per_sec == pytest.approx(5.0)

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

        assert trial.variants is not None
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

        assert trial.variants is not None
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
        """Construct a known power-law decay and verify log-log OLS recovers it.

        We pick forfeit_rate(step) = A * step^b so log(forfeit) is exactly
        linear in log(step) with slope b. The fit uses the second half of
        the history.
        """
        run_dir = tmp_path / "run_x"
        exponent = -0.8  # forfeit ∝ step^-0.8
        prefactor = 100.0  # large enough that forfeit stays < 1 over the window

        records = []
        n = 20
        for i in range(n):
            step = (i + 1) * 1000
            forfeit = prefactor * (step ** exponent)
            gc = 1.0 - forfeit
            records.append(_val_record(step, gc))

        _write_metrics_file(run_dir / "metrics.jsonl", records)
        trial = _make_trial()
        trial.run_dir = str(run_dir)
        summary = read_pretrain_val_summary(trial)

        assert summary is not None
        assert "forfeit_fit" in summary
        fit = summary["forfeit_fit"]
        assert fit["exponent"] == pytest.approx(exponent, rel=1e-6)
        assert fit["prefactor"] == pytest.approx(prefactor, rel=1e-6)
        # x_ratio_to_halve = 2^(-1/b); with b = -0.8 that's 2^1.25 ≈ 2.378
        assert fit["x_ratio_to_halve"] == pytest.approx(2.0 ** (-1.0 / exponent), rel=1e-6)
        # Second half of n=20 → 10 points
        assert fit["n_points"] == 10
        # current_forfeit is the last overall series value, not the fit window's
        expected_current = prefactor * ((n * 1000) ** exponent)
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


# =====================================================================
# read_cotrain_val_summary
# =====================================================================


class TestReadCotrainValSummary:
    def test_no_variants_returns_none(self):
        trial = _make_trial()
        assert read_cotrain_val_summary(trial) is None

    def test_variants_without_game_completion_returns_none(self, tmp_path):
        """Cotrain variants that only have loss/accuracy records but no
        game_completion_rate yet — the helper skips them and returns None."""
        trial = _make_trial()
        run_dir = tmp_path / "run_base"
        _write_metrics_file(run_dir / "metrics.jsonl", [
            {"type": "val", "step": 100, "val/loss": 2.0, "val/accuracy": 0.5},
        ])
        trial.variants = {"base": {"run_dir": str(run_dir), "current_step": 100}}
        assert read_cotrain_val_summary(trial) is None

    def test_per_variant_summary_structure(self, tmp_path):
        """Each variant's metrics.jsonl yields an independent forfeit summary."""
        trial = _make_trial()
        trial.variants = {}
        exponents = {"small": -0.4, "base": -0.6, "large": -0.9}
        prefactor = 50.0
        for name, exponent in exponents.items():
            run_dir = tmp_path / f"run_{name}"
            records = []
            for i in range(1, 21):
                step = i * 1000
                forfeit = prefactor * (step ** exponent)
                records.append(_val_record(step, 1.0 - forfeit))
            _write_metrics_file(run_dir / "metrics.jsonl", records)
            trial.variants[name] = {
                "run_dir": str(run_dir),
                "current_step": 20_000,
            }

        summary = read_cotrain_val_summary(trial)
        assert summary is not None
        assert set(summary["variants"].keys()) == {"small", "base", "large"}
        for name, want_exponent in exponents.items():
            v = summary["variants"][name]
            assert "latest" in v
            assert "forfeit_fit" in v
            assert v["forfeit_fit"]["exponent"] == pytest.approx(want_exponent, rel=1e-6)
            assert v["forfeit_fit"]["prefactor"] == pytest.approx(prefactor, rel=1e-6)
            assert v["forfeit_fit"]["n_points"] == 10

    def test_partial_variants_surface_what_is_ready(self, tmp_path):
        """A variant with records is surfaced even when a sibling has none."""
        trial = _make_trial()
        run_small = tmp_path / "run_small"
        _write_metrics_file(run_small / "metrics.jsonl", [
            _val_record(1000, 0.5),
        ])
        run_base = tmp_path / "run_base"
        _write_metrics_file(run_base / "metrics.jsonl", [
            {"type": "train", "step": 1000, "train/loss": 2.0},
        ])
        trial.variants = {
            "small": {"run_dir": str(run_small), "current_step": 1000},
            "base": {"run_dir": str(run_base), "current_step": 1000},
        }
        summary = read_cotrain_val_summary(trial)
        assert summary is not None
        # Only `small` had game_completion records.
        assert list(summary["variants"].keys()) == ["small"]


# =====================================================================
# fit_power_law
# =====================================================================


class TestFitPowerLaw:
    def test_recovers_known_exponent_and_prefactor(self):
        # y = 3.5 * x^-1.2
        prefactor, exponent = 3.5, -1.2
        xs = [float(x) for x in range(1, 30)]
        ys = [prefactor * (x ** exponent) for x in xs]
        fit = fit_power_law(xs, ys)
        assert fit is not None
        assert fit["exponent"] == pytest.approx(exponent, rel=1e-10)
        assert fit["prefactor"] == pytest.approx(prefactor, rel=1e-10)
        # Decaying → ratio > 1
        assert fit["x_ratio_to_halve"] == pytest.approx(2.0 ** (-1.0 / exponent), rel=1e-10)
        assert fit["n_points"] == float(len(xs))

    def test_omits_ratio_for_non_decaying_exponent(self):
        # y = x^+0.5 → growing; halving is meaningless.
        xs = [float(x) for x in range(1, 10)]
        ys = [x ** 0.5 for x in xs]
        fit = fit_power_law(xs, ys)
        assert fit is not None
        assert fit["exponent"] == pytest.approx(0.5, rel=1e-10)
        assert "x_ratio_to_halve" not in fit

    def test_drops_nonpositive_inputs(self):
        xs = [0.0, 1.0, 2.0, 4.0, 8.0]
        ys = [-1.0, 1.0, 0.5, 0.25, 0.125]  # first point dropped (y < 0)
        fit = fit_power_law(xs, ys)
        assert fit is not None
        assert fit["n_points"] == 4.0

    def test_returns_none_when_too_few_points(self):
        assert fit_power_law([1.0, 2.0], [1.0, 0.5]) is None
        # After dropping non-positives, only 2 remain
        assert fit_power_law([0.0, 1.0, -1.0, 2.0], [1.0, 0.5, 0.25, 0.125]) is None

    def test_returns_none_when_log_x_has_zero_variance(self):
        # All xs equal → log-x variance is zero
        fit = fit_power_law([2.0, 2.0, 2.0], [1.0, 1.5, 2.0])
        assert fit is None
