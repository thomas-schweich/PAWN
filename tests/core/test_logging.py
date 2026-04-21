"""Unit tests for pawn.logging: MetricsLogger JSONL writer.

FROZEN MODULE — do not edit pawn/logging.py to make a test pass.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from pawn.logging import (
    MetricsLogger,
    _sanitize,
    get_git_info,
    random_slug,
    _ADJECTIVES,
    _ANIMALS,
)


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitize:
    def test_nan_becomes_none(self):
        assert _sanitize(float("nan")) is None

    def test_pos_inf_becomes_none(self):
        assert _sanitize(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _sanitize(float("-inf")) is None

    def test_finite_float_preserved(self):
        assert _sanitize(3.14) == 3.14
        assert _sanitize(0.0) == 0.0
        assert _sanitize(-1e-9) == -1e-9

    def test_int_preserved(self):
        assert _sanitize(42) == 42
        assert _sanitize(0) == 0

    def test_string_preserved(self):
        assert _sanitize("hello") == "hello"

    def test_none_preserved(self):
        assert _sanitize(None) is None

    def test_bool_preserved(self):
        assert _sanitize(True) is True
        assert _sanitize(False) is False

    def test_nested_dict_nan_replaced(self):
        result = _sanitize({"a": 1.0, "b": float("nan"), "c": "x"})
        assert result == {"a": 1.0, "b": None, "c": "x"}

    def test_nested_list_nan_replaced(self):
        result = _sanitize([1.0, float("nan"), float("inf"), 2.0])
        assert result == [1.0, None, None, 2.0]

    def test_deeply_nested(self):
        obj = {"outer": {"inner": [1.0, float("nan"), {"k": float("inf")}]}}
        result = _sanitize(obj)
        assert result == {"outer": {"inner": [1.0, None, {"k": None}]}}

    def test_dict_without_nan_unchanged_keys(self):
        obj = {"loss": 3.0, "accuracy": 0.1}
        result = _sanitize(obj)
        assert result == obj


# ---------------------------------------------------------------------------
# random_slug
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRandomSlug:
    def test_format_adjective_dash_animal(self):
        slug = random_slug()
        assert re.match(r"^[a-z]+-[a-z]+$", slug), f"Bad slug format: {slug}"

    def test_slug_components_in_wordlists(self):
        """Generate multiple slugs to catch edge cases a single draw might miss."""
        for i in range(20):
            slug = random_slug()
            adj, animal = slug.split("-")
            assert adj in _ADJECTIVES, f"slug #{i} adj {adj!r} not in adjectives"
            assert animal in _ANIMALS, f"slug #{i} animal {animal!r} not in animals"

    def test_slug_is_randomish(self):
        """Generate many slugs; expect at least some variation."""
        slugs = {random_slug() for _ in range(50)}
        assert len(slugs) > 5, "random_slug should produce varied output"

    def test_wordlist_nonempty(self):
        assert len(_ADJECTIVES) >= 10, "adjective list too small for sufficient variety"
        assert len(_ANIMALS) >= 10, "animal list too small for sufficient variety"
        # Verify the function actually draws from these lists
        slug = random_slug()
        adj, animal = slug.split("-")
        assert adj in _ADJECTIVES, f"{adj!r} not drawn from _ADJECTIVES"
        assert animal in _ANIMALS, f"{animal!r} not drawn from _ANIMALS"


# ---------------------------------------------------------------------------
# MetricsLogger basics
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_logger(tmp_path):
    logger = MetricsLogger(
        log_dir=tmp_path, run_prefix="test", slug="zesty-osprey", device="cpu",
    )
    yield logger
    logger.close()


@pytest.mark.unit
class TestMetricsLoggerInit:
    def test_creates_run_dir(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, slug="zesty-osprey") as logger:
            assert logger.run_dir.exists()
            assert logger.run_dir.is_dir()
            assert logger.run_dir.name.endswith("_zesty-osprey")

    def test_metrics_path_is_jsonl(self, fresh_logger):
        assert fresh_logger.metrics_path.name == "metrics.jsonl"
        assert fresh_logger.metrics_path.exists()

    def test_path_property_matches(self, fresh_logger):
        assert fresh_logger.path == fresh_logger.metrics_path

    def test_uses_provided_slug(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, slug="brisk-falcon") as logger:
            assert logger.slug == "brisk-falcon"
            assert "brisk-falcon" in logger.run_dir.name

    def test_generates_slug_when_none(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path) as logger:
            assert "-" in logger.slug
            adj, animal = logger.slug.split("-")
            assert adj in _ADJECTIVES
            assert animal in _ANIMALS

    def test_run_prefix_applied(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, run_prefix="myexp", slug="s-s") as logger:
            assert logger.run_dir.name.startswith("myexp_")

    def test_suffix_applied(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, slug="s-s", suffix="base") as logger:
            assert "_base_" in logger.run_dir.name

    def test_timestamp_in_dirname(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, slug="s-s") as logger:
            parts = logger.run_dir.name.split("_")
            # run_prefix_YYYYMMDD_HHMMSS_slug
            assert len(parts) >= 4
            date_part = parts[1]
            time_part = parts[2]
            assert re.match(r"^\d{8}$", date_part)
            assert re.match(r"^\d{6}$", time_part)


# ---------------------------------------------------------------------------
# log_config / log_train / log_val — record contents
# ---------------------------------------------------------------------------


def _read_all_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.unit
class TestLogConfig:
    def test_writes_config_record(self, fresh_logger):
        fresh_logger.log_config(run_type="pretrain", variant="base")
        records = _read_all_records(fresh_logger.metrics_path)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "config"
        assert r["run_type"] == "pretrain"
        assert r["variant"] == "base"

    def test_config_has_slug(self, fresh_logger):
        fresh_logger.log_config(run_type="pretrain")
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["slug"] == "zesty-osprey"

    def test_config_has_hostname(self, fresh_logger):
        fresh_logger.log_config(run_type="pretrain")
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert "hostname" in r
        assert isinstance(r["hostname"], str)
        assert len(r["hostname"]) > 0

    def test_config_has_timestamp(self, fresh_logger):
        fresh_logger.log_config(run_type="pretrain")
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert "timestamp" in r

    def test_config_has_git_info_keys(self, fresh_logger):
        fresh_logger.log_config(run_type="pretrain")
        r = _read_all_records(fresh_logger.metrics_path)[0]
        # git_hash/git_tag keys always present (may be None)
        assert "git_hash" in r
        assert "git_tag" in r

    def test_config_accepts_nested_dicts(self, fresh_logger):
        fresh_logger.log_config(
            run_type="pretrain",
            model={"d_model": 512, "n_layers": 8},
            training={"lr": 3e-4},
        )
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["model"]["d_model"] == 512
        assert r["training"]["lr"] == 3e-4


@pytest.mark.unit
class TestLogTrain:
    def test_writes_train_record(self, fresh_logger):
        fresh_logger.log_train(step=100, lr=3e-4, loss=3.5, accuracy=0.06)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["type"] == "train"
        assert r["step"] == 100
        assert r["lr"] == 3e-4
        assert r["loss"] == 3.5

    def test_train_record_has_baseline_fields(self, fresh_logger):
        fresh_logger.log_train(step=1, loss=1.0)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        # Baseline: type, step, timestamp, elapsed
        assert "timestamp" in r
        assert "elapsed" in r
        assert r["type"] == "train"

    def test_train_record_has_memory_stats(self, fresh_logger):
        fresh_logger.log_train(step=1, loss=1.0)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert "mem/system_rss_gb" in r
        assert "mem/system_used_gb" in r
        assert "mem/system_total_gb" in r
        assert "mem/cpu_percent" in r

    def test_train_epoch_recorded_when_set(self, fresh_logger):
        fresh_logger.log_train(step=500, epoch=3, loss=2.0)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["epoch"] == 3

    def test_train_epoch_absent_when_unset(self, fresh_logger):
        fresh_logger.log_train(step=500, loss=2.0)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert "epoch" not in r

    def test_train_custom_adapter_fields_pass_through(self, fresh_logger):
        fresh_logger.log_train(
            step=1, loss=1.0,
            **{"film/gamma_norm_L0": 0.5, "lora/B_norm_q": 0.2},
        )
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["film/gamma_norm_L0"] == 0.5
        assert r["lora/B_norm_q"] == 0.2

    def test_elapsed_monotonic_across_records(self, fresh_logger):
        import time
        fresh_logger.log_train(step=1, loss=1.0)
        time.sleep(0.01)
        fresh_logger.log_train(step=2, loss=1.0)
        records = _read_all_records(fresh_logger.metrics_path)
        assert records[1]["elapsed"] >= records[0]["elapsed"]


@pytest.mark.unit
class TestLogVal:
    def test_writes_val_record(self, fresh_logger):
        fresh_logger.log_val(step=100, loss=3.6, accuracy=0.05, patience=2)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["type"] == "val"
        assert r["step"] == 100
        assert r["loss"] == 3.6
        assert r["patience"] == 2

    def test_val_has_baseline_fields(self, fresh_logger):
        fresh_logger.log_val(step=100, loss=1.0)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert "timestamp" in r
        assert "elapsed" in r

    def test_val_epoch_recorded(self, fresh_logger):
        fresh_logger.log_val(step=500, epoch=5, loss=2.0)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["epoch"] == 5


# ---------------------------------------------------------------------------
# NaN/Inf sanitization on write
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitizeOnWrite:
    def test_nan_loss_written_as_null(self, fresh_logger):
        fresh_logger.log_train(step=1, loss=float("nan"))
        # Read raw text — JSON should not contain "NaN"
        raw = fresh_logger.metrics_path.read_text()
        assert "NaN" not in raw
        assert "null" in raw or '"loss": null' in raw
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["loss"] is None

    def test_inf_accuracy_written_as_null(self, fresh_logger):
        fresh_logger.log_train(step=1, accuracy=float("inf"), loss=1.0)
        raw = fresh_logger.metrics_path.read_text()
        assert "Infinity" not in raw
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["accuracy"] is None

    def test_nested_nan_sanitized(self, fresh_logger):
        fresh_logger.log_config(
            run_type="x",
            nested={"a": float("nan"), "b": 1.0},
        )
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["nested"]["a"] is None
        assert r["nested"]["b"] == 1.0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContextManager:
    def test_enter_returns_self(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, slug="a-b") as logger:
            assert isinstance(logger, MetricsLogger)

    def test_exit_closes_file(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, slug="a-b") as logger:
            logger.log_train(step=1, loss=1.0)
        assert logger._file.closed

    def test_close_idempotent(self, tmp_path):
        logger = MetricsLogger(log_dir=tmp_path, slug="a-b")
        logger.close()
        logger.close()  # should not raise
        assert logger._file.closed

    def test_write_after_explicit_close_keeps_file_closed(self, tmp_path):
        logger = MetricsLogger(log_dir=tmp_path, slug="a-b")
        logger.log_train(step=1, loss=1.0)
        logger.close()
        assert logger._file.closed


# ---------------------------------------------------------------------------
# write_config_json
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWriteConfigJson:
    def test_creates_config_json_file(self, fresh_logger):
        path = fresh_logger.write_config_json(run_type="pretrain", variant="base")
        assert path.exists()
        assert path.name == "config.json"
        assert path == fresh_logger.run_dir / "config.json"

    def test_config_json_has_slug_and_git(self, fresh_logger):
        fresh_logger.write_config_json(run_type="pretrain")
        data = json.loads((fresh_logger.run_dir / "config.json").read_text())
        assert data["slug"] == "zesty-osprey"
        assert "git_hash" in data
        assert "git_tag" in data

    def test_config_json_includes_kwargs(self, fresh_logger):
        fresh_logger.write_config_json(
            run_type="adapter", strategy="lora", lora_rank=4,
        )
        data = json.loads((fresh_logger.run_dir / "config.json").read_text())
        assert data["run_type"] == "adapter"
        assert data["strategy"] == "lora"
        assert data["lora_rank"] == 4

    def test_returns_path(self, fresh_logger):
        path = fresh_logger.write_config_json(run_type="x")
        assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# generic log()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenericLog:
    def test_log_train_default(self, fresh_logger):
        fresh_logger.log({"loss": 1.0}, step=10)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["type"] == "train"
        assert r["loss"] == 1.0
        assert r["step"] == 10

    def test_log_without_step(self, fresh_logger):
        fresh_logger.log({"loss": 1.0})
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert "step" not in r

    def test_log_without_resources(self, fresh_logger):
        fresh_logger.log({"loss": 1.0}, step=1, include_resources=False)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert "mem/system_rss_gb" not in r
        # but baseline (timestamp, elapsed) still present
        assert "timestamp" in r
        assert "elapsed" in r

    def test_log_record_type_custom(self, fresh_logger):
        fresh_logger.log({"metric": 1.0}, step=1, record_type="eval")
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["type"] == "eval"

    def test_log_with_epoch(self, fresh_logger):
        fresh_logger.log({"loss": 1.0}, step=10, epoch=2)
        r = _read_all_records(fresh_logger.metrics_path)[0]
        assert r["epoch"] == 2


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGitInfo:
    def test_returns_dict_with_hash_and_tag_keys(self):
        # Reset cache to ensure fresh call
        import pawn.logging as pl
        pl._git_info = None
        info = get_git_info()
        assert "git_hash" in info
        assert "git_tag" in info

    def test_cached_after_first_call(self):
        import pawn.logging as pl
        pl._git_info = None
        first = get_git_info()
        # Change env vars — cached value should remain
        second = get_git_info()
        assert first is second

    def test_env_var_override(self, monkeypatch):
        import pawn.logging as pl
        pl._git_info = None
        monkeypatch.setenv("PAWN_GIT_HASH", "deadbeef")
        monkeypatch.setenv("PAWN_GIT_TAG", "v1.0")
        info = get_git_info()
        assert info["git_hash"] == "deadbeef"
        assert info["git_tag"] == "v1.0"
        pl._git_info = None  # reset for other tests


# ---------------------------------------------------------------------------
# JSONL format — one record per line
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJsonlFormat:
    def test_multiple_records_one_per_line(self, fresh_logger):
        fresh_logger.log_train(step=1, loss=1.0)
        fresh_logger.log_train(step=2, loss=0.9)
        fresh_logger.log_val(step=2, loss=1.5)
        lines = fresh_logger.metrics_path.read_text().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # each line is standalone JSON

    def test_file_flushed_each_write(self, fresh_logger):
        """After log_train, the file contents should be readable immediately."""
        fresh_logger.log_train(step=1, loss=1.0)
        content = fresh_logger.metrics_path.read_text()
        assert content.strip() != ""
