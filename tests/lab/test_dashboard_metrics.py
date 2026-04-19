"""Tests for pawn.dashboard.metrics — run discovery, loading, type detection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from pawn.dashboard.metrics import (
    col,
    detect_run_type,
    get_run_hostname,
    get_run_meta,
    list_trials,
    load_metrics,
    load_notes,
    load_runs,
    notes_path,
    save_notes,
)


def _write_run(log_dir: Path, run_name: str, records: list[dict], mtime: float | None = None) -> Path:
    """Write metrics.jsonl at log_dir/run_name/metrics.jsonl."""
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# =====================================================================
# load_runs
# =====================================================================


class TestLoadRuns:
    def test_nonexistent_log_dir_returns_empty(self, tmp_path):
        assert load_runs(tmp_path / "nope") == []

    def test_log_dir_is_file_returns_empty(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        assert load_runs(f) == []

    def test_empty_log_dir_returns_empty(self, tmp_path):
        assert load_runs(tmp_path) == []

    def test_lists_runs_with_metrics(self, tmp_path):
        _write_run(tmp_path, "run_1", [{"type": "config"}])
        _write_run(tmp_path, "run_2", [{"type": "config"}])
        result = load_runs(tmp_path, max_age_hours=0)  # no filter
        assert set(result) == {"run_1", "run_2"}

    def test_skips_dirs_without_metrics(self, tmp_path):
        (tmp_path / "no_metrics").mkdir()
        _write_run(tmp_path, "has_metrics", [{"type": "config"}])
        result = load_runs(tmp_path, max_age_hours=0)
        assert result == ["has_metrics"]

    def test_max_age_filter_excludes_old(self, tmp_path):
        old_mtime = time.time() - 3600 * 48  # 2 days ago
        _write_run(tmp_path, "old_run", [{"type": "config"}], mtime=old_mtime)
        _write_run(tmp_path, "new_run", [{"type": "config"}])
        result = load_runs(tmp_path, max_age_hours=1.0)
        assert "new_run" in result
        assert "old_run" not in result

    def test_max_age_zero_includes_all(self, tmp_path):
        old_mtime = time.time() - 3600 * 48
        _write_run(tmp_path, "old_run", [{"type": "config"}], mtime=old_mtime)
        _write_run(tmp_path, "new_run", [{"type": "config"}])
        result = load_runs(tmp_path, max_age_hours=0)
        assert "old_run" in result
        assert "new_run" in result

    def test_sorted_newest_first(self, tmp_path):
        now = time.time()
        _write_run(tmp_path, "oldest", [{"type": "config"}], mtime=now - 300)
        _write_run(tmp_path, "middle", [{"type": "config"}], mtime=now - 100)
        _write_run(tmp_path, "newest", [{"type": "config"}], mtime=now)
        result = load_runs(tmp_path, max_age_hours=0)
        assert result == ["newest", "middle", "oldest"]

    def test_discovers_nested_trials(self, tmp_path):
        _write_run(tmp_path, "trial_0001/run_a", [{"type": "config"}])
        _write_run(tmp_path, "trial_0001/run_b", [{"type": "config"}])
        _write_run(tmp_path, "trial_0002/run_c", [{"type": "config"}])
        result = load_runs(tmp_path, max_age_hours=0)
        assert set(result) == {"trial_0001/run_a", "trial_0001/run_b", "trial_0002/run_c"}

    def test_mixes_flat_and_nested(self, tmp_path):
        _write_run(tmp_path, "flat_run", [{"type": "config"}])
        _write_run(tmp_path, "trial_0001/nested_run", [{"type": "config"}])
        result = load_runs(tmp_path, max_age_hours=0)
        assert set(result) == {"flat_run", "trial_0001/nested_run"}

    def test_checkpoints_subdir_does_not_trigger_false_discovery(self, tmp_path):
        # A normal run has a checkpoints/ dir sitting alongside metrics.jsonl;
        # it should never be treated as its own run.
        _write_run(tmp_path, "run_a", [{"type": "config"}])
        (tmp_path / "run_a" / "checkpoints").mkdir()
        (tmp_path / "run_a" / "checkpoints" / "model.bin").write_bytes(b"x")
        result = load_runs(tmp_path, max_age_hours=0)
        assert result == ["run_a"]


class TestListTrials:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert list_trials(tmp_path) == []

    def test_flat_runs_only_returns_empty(self, tmp_path):
        _write_run(tmp_path, "run_a", [{"type": "config"}])
        _write_run(tmp_path, "run_b", [{"type": "config"}])
        assert list_trials(tmp_path) == []

    def test_returns_trial_dirs_sorted_newest_first(self, tmp_path):
        now = time.time()
        _write_run(tmp_path, "trial_0001/run_a", [{"type": "config"}], mtime=now - 500)
        _write_run(tmp_path, "trial_0002/run_b", [{"type": "config"}], mtime=now - 50)
        _write_run(tmp_path, "trial_0003/run_c", [{"type": "config"}], mtime=now)
        assert list_trials(tmp_path) == ["trial_0003", "trial_0002", "trial_0001"]


# =====================================================================
# get_run_meta / get_run_hostname
# =====================================================================


class TestGetRunMeta:
    def test_missing_run_returns_empty(self, tmp_path):
        assert get_run_meta(tmp_path, "does_not_exist") == {}

    def test_reads_first_line_config(self, tmp_path):
        _write_run(tmp_path, "run_1", [
            {"type": "config", "hostname": "pod-abc", "slug": "shiny-fox",
             "variant": "base"},
        ])
        meta = get_run_meta(tmp_path, "run_1")
        assert meta["hostname"] == "pod-abc"
        assert meta["slug"] == "shiny-fox"
        assert meta["variant"] == "base"

    def test_returns_empty_strings_for_missing_fields(self, tmp_path):
        _write_run(tmp_path, "run_1", [{"type": "config"}])
        meta = get_run_meta(tmp_path, "run_1")
        assert meta["hostname"] == ""
        assert meta["slug"] == ""
        assert meta["variant"] == ""

    def test_malformed_json_returns_empty(self, tmp_path):
        run_dir = tmp_path / "bad_run"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text("not json\n")
        assert get_run_meta(tmp_path, "bad_run") == {}

    def test_empty_file_returns_empty(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").touch()
        assert get_run_meta(tmp_path, "empty_run") == {}

    def test_get_run_hostname_wraps_meta(self, tmp_path):
        _write_run(tmp_path, "run_1", [{"type": "config", "hostname": "h1"}])
        assert get_run_hostname(tmp_path, "run_1") == "h1"

    def test_get_run_hostname_missing_returns_empty(self, tmp_path):
        assert get_run_hostname(tmp_path, "nope") == ""


# =====================================================================
# load_metrics
# =====================================================================


class TestLoadMetricsNested:
    def test_load_metrics_round_trips_nested_path(self, tmp_path):
        _write_run(tmp_path, "trial_0001/run_x", [
            {"type": "config", "variant": "base"},
            {"type": "train", "step": 0},
        ])
        m = load_metrics(tmp_path, "trial_0001/run_x")
        assert len(m["train"]) == 1
        assert m["config"][0]["variant"] == "base"

    def test_get_run_meta_round_trips_nested_path(self, tmp_path):
        _write_run(tmp_path, "trial_0001/run_x", [
            {"type": "config", "hostname": "h", "slug": "s", "variant": "v"},
        ])
        meta = get_run_meta(tmp_path, "trial_0001/run_x")
        assert meta == {"hostname": "h", "slug": "s", "variant": "v"}


class TestNotes:
    def test_notes_path_for_flat_run(self, tmp_path):
        path = notes_path(tmp_path, "run_a")
        assert path == tmp_path / "run_a" / "notes.md"

    def test_notes_path_for_nested_trial(self, tmp_path):
        # Trial-scoped notes: every run under trial_0001 shares this file.
        assert notes_path(tmp_path, "trial_0001/run_a") == tmp_path / "trial_0001" / "notes.md"
        assert notes_path(tmp_path, "trial_0001/run_b") == tmp_path / "trial_0001" / "notes.md"

    def test_load_notes_missing_returns_empty(self, tmp_path):
        assert load_notes(tmp_path, "run_a") == ""

    def test_save_then_load_round_trips(self, tmp_path):
        _write_run(tmp_path, "run_a", [{"type": "config"}])
        save_notes(tmp_path, "run_a", "Hello world.")
        assert load_notes(tmp_path, "run_a") == "Hello world."

    def test_save_creates_parent_dir(self, tmp_path):
        # No run dir yet — save must still work (e.g. trial exists, run not discovered yet).
        save_notes(tmp_path, "trial_0001/run_a", "notes")
        assert (tmp_path / "trial_0001" / "notes.md").read_text() == "notes"

    def test_trial_notes_shared_across_runs(self, tmp_path):
        _write_run(tmp_path, "trial_0001/run_a", [{"type": "config"}])
        _write_run(tmp_path, "trial_0001/run_b", [{"type": "config"}])
        save_notes(tmp_path, "trial_0001/run_a", "shared notes")
        assert load_notes(tmp_path, "trial_0001/run_b") == "shared notes"


class TestLoadMetrics:
    def test_missing_run_returns_empty_dict(self, tmp_path):
        result = load_metrics(tmp_path, "nope")
        assert result == {}

    def test_buckets_by_type(self, tmp_path):
        _write_run(tmp_path, "run_1", [
            {"type": "config", "variant": "base"},
            {"type": "train", "step": 0},
            {"type": "train", "step": 1},
            {"type": "val", "step": 1},
        ])
        m = load_metrics(tmp_path, "run_1")
        assert "config" in m
        assert "train" in m
        assert "val" in m
        assert len(m["train"]) == 2
        assert len(m["val"]) == 1
        assert len(m["config"]) == 1

    def test_default_type_is_train(self, tmp_path):
        _write_run(tmp_path, "run_1", [{"step": 0}])  # no 'type' key
        m = load_metrics(tmp_path, "run_1")
        assert "train" in m
        assert len(m["train"]) == 1

    def test_skips_blank_lines(self, tmp_path):
        run_dir = tmp_path / "run_1"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text(
            '\n\n{"type": "train", "step": 0}\n\n'
        )
        m = load_metrics(tmp_path, "run_1")
        assert len(m["train"]) == 1

    def test_skips_malformed_json(self, tmp_path):
        run_dir = tmp_path / "run_1"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text(
            'bad json\n{"type": "train", "step": 0}\n'
        )
        m = load_metrics(tmp_path, "run_1")
        assert len(m["train"]) == 1


# =====================================================================
# detect_run_type
# =====================================================================


class TestDetectRunType:
    def test_film(self):
        assert detect_run_type({"run_type": "film"}) == "film"

    def test_lora(self):
        assert detect_run_type({"run_type": "lora"}) == "lora"

    def test_hybrid(self):
        assert detect_run_type({"run_type": "hybrid"}) == "hybrid"

    def test_sparse(self):
        assert detect_run_type({"run_type": "sparse"}) == "sparse"

    def test_bottleneck(self):
        assert detect_run_type({"run_type": "bottleneck"}) == "bottleneck"

    def test_tiny(self):
        assert detect_run_type({"run_type": "tiny"}) == "tiny"

    def test_rosa(self):
        assert detect_run_type({"run_type": "rosa"}) == "rosa"

    def test_unknown_run_type_falls_through(self):
        # formulation=clm => pawn
        assert detect_run_type({"run_type": "weird", "formulation": "clm"}) == "pawn"

    def test_formulation_clm_returns_pawn(self):
        assert detect_run_type({"formulation": "clm"}) == "pawn"

    def test_pgn_file_returns_bc(self):
        assert detect_run_type({"pgn_file": "games.pgn"}) == "bc"

    def test_empty_config_defaults_to_pawn(self):
        assert detect_run_type({}) == "pawn"


# =====================================================================
# col
# =====================================================================


class TestCol:
    def test_basic(self):
        records = [{"x": 1}, {"x": 2}, {"x": 3}]
        assert col(records, "x") == [1, 2, 3]

    def test_skips_missing(self):
        records = [{"x": 1}, {"y": 2}, {"x": 3}]
        assert col(records, "x") == [1, 3]

    def test_skips_none(self):
        records = [{"x": 1}, {"x": None}, {"x": 3}]
        assert col(records, "x") == [1, 3]

    def test_empty_records(self):
        assert col([], "x") == []

    def test_key_never_present(self):
        assert col([{"a": 1}, {"b": 2}], "x") == []

    def test_preserves_order(self):
        records = [{"x": 10}, {"x": 5}, {"x": 7}]
        assert col(records, "x") == [10, 5, 7]
