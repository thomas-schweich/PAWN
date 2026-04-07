"""Tests for pawn.lab.monitor — metrics reading, health checks, process liveness."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from pawn.lab.monitor import check_health, is_alive, read_metrics
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
        assert "NaN" in issue or "Inf" in issue

    def test_inf_loss_after_warmup_is_unhealthy(self):
        t = _make_trial()
        t.last_train_loss = float("inf")
        t.total_steps = 10000
        t.current_step = 600
        issue = check_health(t)
        assert issue is not None

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
        assert check_health(t) is not None

    def test_no_total_steps_uses_500_threshold(self):
        t = _make_trial()
        t.last_train_loss = float("nan")
        t.total_steps = 0
        t.current_step = 400
        assert check_health(t) is None
        t.current_step = 600
        assert check_health(t) is not None
