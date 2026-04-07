"""Tests for pawn.lab.state — Trial dataclass and helpers."""

from __future__ import annotations

import dataclasses
import re
import time

import pytest

from pawn.lab.state import Trial, _format_duration, _now_iso


# =====================================================================
# _format_duration
# =====================================================================


class TestFormatDuration:
    def test_zero(self):
        assert _format_duration(0) == "0m00s"

    def test_seconds_only(self):
        assert _format_duration(45) == "0m45s"

    def test_minutes_seconds(self):
        assert _format_duration(23 * 60 + 45) == "23m45s"

    def test_hours_minutes(self):
        # 1h 23m 45s -> the helper uses "{h}h{m:02d}m" format (no seconds on hours path)
        assert _format_duration(3600 + 23 * 60 + 45) == "1h23m"

    def test_none_returns_question_mark(self):
        assert _format_duration(None) == "?"

    def test_nan_returns_question_mark(self):
        assert _format_duration(float("nan")) == "?"

    def test_inf_returns_question_mark(self):
        assert _format_duration(float("inf")) == "?"

    def test_fractional_seconds_floor(self):
        # divmod(int(seconds), ...) truncates
        assert _format_duration(59.9) == "0m59s"

    def test_exactly_one_minute(self):
        assert _format_duration(60) == "1m00s"

    def test_exactly_one_hour(self):
        assert _format_duration(3600) == "1h00m"


# =====================================================================
# _now_iso
# =====================================================================


class TestNowIso:
    def test_returns_iso_8601_format(self):
        s = _now_iso()
        # Expect YYYY-MM-DDTHH:MM:SS
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", s)

    def test_is_string(self):
        assert isinstance(_now_iso(), str)


# =====================================================================
# Trial dataclass basic construction
# =====================================================================


class TestTrialConstruction:
    def test_minimal_construction(self):
        t = Trial(
            trial_id=0,
            strategy="lora",
            params={},
            cli_command=["python", "scripts/train.py"],
        )
        assert t.trial_id == 0
        assert t.strategy == "lora"
        assert t.params == {}
        assert t.cli_command == ["python", "scripts/train.py"]

    def test_default_status_is_queued(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.status == "queued"

    def test_default_pid_is_none(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.pid is None

    def test_default_gpu_id_is_none(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.gpu_id is None

    def test_default_current_step_is_zero(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.current_step == 0

    def test_default_total_steps_is_zero(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.total_steps == 0

    def test_default_steps_per_sec_is_zero(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.steps_per_sec == 0.0

    def test_default_best_val_loss_is_none(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.best_val_loss is None

    def test_default_tags_is_empty_list(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.tags == []

    def test_default_config_is_empty_dict(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.config == {}

    def test_default_notes_is_empty_string(self):
        t = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        assert t.notes == ""

    def test_tags_default_is_independent(self):
        """Each instance gets its own tags list (field(default_factory=list))."""
        t1 = Trial(trial_id=0, strategy="lora", params={}, cli_command=[])
        t2 = Trial(trial_id=1, strategy="lora", params={}, cli_command=[])
        t1.tags.append("phase1")
        assert t2.tags == []

    def test_all_expected_fields_present(self):
        names = {f.name for f in dataclasses.fields(Trial)}
        expected = {
            "trial_id", "strategy", "params", "cli_command", "config",
            "status", "pid", "gpu_id", "start_time", "end_time",
            "current_step", "total_steps", "steps_per_sec",
            "last_train_loss", "last_train_acc", "best_val_loss",
            "best_accuracy", "actual_param_count",
            "log_path", "run_dir", "optuna_number", "notes", "tags",
        }
        missing = expected - names
        assert not missing, f"Trial missing fields: {missing}"


# =====================================================================
# Trial.eta_seconds
# =====================================================================


class TestTrialEtaSeconds:
    def test_no_steps_per_sec_returns_none(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=1000, current_step=100, steps_per_sec=0.0,
        )
        assert t.eta_seconds() is None

    def test_negative_steps_per_sec_returns_none(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=1000, current_step=100, steps_per_sec=-1.0,
        )
        assert t.eta_seconds() is None

    def test_already_at_total_returns_none(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=1000, current_step=1000, steps_per_sec=5.0,
        )
        assert t.eta_seconds() is None

    def test_exceeded_total_returns_none(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=1000, current_step=1500, steps_per_sec=5.0,
        )
        assert t.eta_seconds() is None

    def test_basic_computation(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=1000, current_step=100, steps_per_sec=10.0,
        )
        # (1000 - 100) / 10.0 = 90.0
        assert t.eta_seconds() == pytest.approx(90.0)

    def test_zero_current_step(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=100, current_step=0, steps_per_sec=1.0,
        )
        assert t.eta_seconds() == pytest.approx(100.0)


# =====================================================================
# Trial.to_dict / from_dict
# =====================================================================


class TestTrialToDict:
    def test_basic_serialization(self):
        t = Trial(
            trial_id=1, strategy="lora", params={"lr": 1e-3},
            cli_command=["python", "train.py"],
            status="running",
            current_step=50, total_steps=100, steps_per_sec=5.0,
        )
        d = t.to_dict()
        assert d["trial_id"] == 1
        assert d["strategy"] == "lora"
        assert d["params"] == {"lr": 1e-3}
        assert d["status"] == "running"
        assert d["current_step"] == 50
        assert d["total_steps"] == 100

    def test_to_dict_includes_eta_seconds(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=100, current_step=0, steps_per_sec=10.0,
        )
        d = t.to_dict()
        assert "eta_seconds" in d
        assert d["eta_seconds"] == pytest.approx(10.0)

    def test_to_dict_includes_eta_human(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            total_steps=100, current_step=0, steps_per_sec=10.0,
        )
        d = t.to_dict()
        assert "eta_human" in d
        assert d["eta_human"] == "0m10s"

    def test_to_dict_eta_none_when_no_rate(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
        )
        d = t.to_dict()
        assert d["eta_seconds"] is None
        assert d["eta_human"] == "?"

    def test_to_dict_includes_elapsed_human(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            start_time=time.time() - 10.0,
        )
        d = t.to_dict()
        assert "elapsed_human" in d
        # Should be roughly 0m10s (possibly 0m09s or 0m11s)
        assert re.match(r"^0m\d{2}s$", d["elapsed_human"])

    def test_to_dict_elapsed_question_when_no_start_time(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
        )
        d = t.to_dict()
        assert d["elapsed_human"] == "?"


class TestTrialFromDict:
    def test_round_trip_basic(self):
        t = Trial(
            trial_id=3, strategy="bottleneck", params={"bottleneck_dim": 8},
            cli_command=["python", "x.py"],
            status="completed", current_step=100, total_steps=100,
            best_val_loss=0.5, best_accuracy=0.8, tags=["phase1"],
        )
        d = t.to_dict()
        t2 = Trial.from_dict(d)
        assert t2.trial_id == t.trial_id
        assert t2.strategy == t.strategy
        assert t2.params == t.params
        assert t2.cli_command == t.cli_command
        assert t2.status == t.status
        assert t2.current_step == t.current_step
        assert t2.total_steps == t.total_steps
        assert t2.best_val_loss == t.best_val_loss
        assert t2.best_accuracy == t.best_accuracy
        assert t2.tags == t.tags

    def test_from_dict_ignores_unknown_fields(self):
        """from_dict should silently drop keys not in the dataclass."""
        d = {
            "trial_id": 0, "strategy": "lora", "params": {},
            "cli_command": [], "status": "queued",
            "eta_seconds": 10.0,   # <-- synthetic, not a field
            "eta_human": "10s",    # <-- synthetic
            "elapsed_human": "5s", # <-- synthetic
            "unknown_key": "foo",  # <-- wholly unknown
        }
        t = Trial.from_dict(d)
        assert t.trial_id == 0
        assert t.strategy == "lora"

    def test_from_dict_preserves_optuna_number(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            optuna_number=7,
        )
        d = t.to_dict()
        t2 = Trial.from_dict(d)
        assert t2.optuna_number == 7

    def test_from_dict_preserves_run_dir(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            run_dir="/tmp/foo/run_123",
        )
        d = t.to_dict()
        t2 = Trial.from_dict(d)
        assert t2.run_dir == "/tmp/foo/run_123"

    def test_from_dict_preserves_config(self):
        t = Trial(
            trial_id=0, strategy="lora", params={}, cli_command=[],
            config={"run_type": "adapter", "strategy": "lora"},
        )
        d = t.to_dict()
        t2 = Trial.from_dict(d)
        assert t2.config == {"run_type": "adapter", "strategy": "lora"}
