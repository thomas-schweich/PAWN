"""Tests for the v2 dashboard — import health + navigation helpers.

Covers the metrics-dashboard parity workstream (criterion 15):

- ``pawn.dashboard.sol`` imports cleanly (the 8-symbol ImportError that
  crashed every UI launch is a behavioral regression this suite catches).
- The v1 navigation surface — ``load_run_buckets`` / ``load_runs`` /
  ``list_trials`` / ``get_run_meta`` / ``detect_run_type`` — is restored.
- Per-run notes (``notes_path`` / ``load_notes`` / ``save_notes``) round-trip.
- ``sync_hf_metrics`` is wired and degrades gracefully when the HF client
  is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import health — the blocker the suite previously missed
# ---------------------------------------------------------------------------


def test_dashboard_sol_imports_without_error() -> None:
    """`pawn.dashboard.sol` must import. It pulls 8 symbols from
    `pawn.dashboard.metrics`; when those were removed in the v2 metrics
    rewrite, every UI launch (`solara run pawn.dashboard.sol`,
    `python -m pawn.dashboard`, `from pawn.dashboard import Dashboard`)
    raised ImportError at import time. This is the regression tripwire."""
    pytest.importorskip("solara")
    pytest.importorskip("plotly")
    sys.modules.pop("pawn.dashboard.sol", None)
    import pawn.dashboard.sol as sol  # noqa: F401

    # The lazily-exported UI components resolve through __getattr__.
    import pawn.dashboard as dash

    assert dash.Dashboard is sol.Dashboard
    assert dash.Page is sol.Page
    assert dash.Runner is sol.Runner


def test_dashboard_init_exports_navigation_helpers() -> None:
    """The package re-exports the v1 navigation helpers the shell and any
    notebook driver consume."""
    import pawn.dashboard as dash

    for name in (
        "load_metrics", "load_run_buckets", "load_runs",
        "list_trials", "detect_run_type", "col", "discover_runs",
    ):
        assert hasattr(dash, name), f"pawn.dashboard missing {name}"


# ---------------------------------------------------------------------------
# Bucketed loader (the v1-shaped loader the Solara shell consumes)
# ---------------------------------------------------------------------------


def test_load_run_buckets_keys_on_raw_type(tmp_path: Path) -> None:
    """`load_run_buckets(log_dir, run_name)` returns `{type: [records]}`
    — the exact shape `sol.Dashboard` reads via `.get("config"/"train"
    /"val"/"batch")`."""
    from pawn.dashboard.metrics import load_run_buckets

    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "config", "run_type": "pretrain", "slug": "s"}),
        json.dumps({"type": "train", "step": 1, "train/loss": 5.0}),
        json.dumps({"type": "train", "step": 2, "train/loss": 4.0}),
        json.dumps({"type": "val", "step": 2, "val/loss": 4.5}),
        json.dumps({"type": "batch", "global_step": 10, "lr": 1e-4}),
    ]))
    buckets = load_run_buckets(tmp_path, "run_a")
    assert len(buckets["config"]) == 1
    assert len(buckets["train"]) == 2
    assert len(buckets["val"]) == 1
    assert len(buckets["batch"]) == 1
    assert buckets["config"][0]["run_type"] == "pretrain"


def test_load_run_buckets_defaults_untyped_to_train(tmp_path: Path) -> None:
    """A record with no `type` lands in the `train` bucket (v1 parity)."""
    from pawn.dashboard.metrics import load_run_buckets

    run = tmp_path / "r"
    run.mkdir()
    (run / "metrics.jsonl").write_text(json.dumps({"step": 1, "loss": 1.0}))
    buckets = load_run_buckets(tmp_path, "r")
    assert len(buckets["train"]) == 1


def test_load_run_buckets_missing_file_returns_empty(tmp_path: Path) -> None:
    from pawn.dashboard.metrics import load_run_buckets

    assert load_run_buckets(tmp_path, "nope") == {}


def test_load_run_buckets_skips_malformed_lines(tmp_path: Path) -> None:
    from pawn.dashboard.metrics import load_run_buckets

    run = tmp_path / "r"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "train", "loss": 1.0}),
        "garbage {",
        json.dumps({"type": "train", "loss": 0.9}),
    ]))
    assert len(load_run_buckets(tmp_path, "r")["train"]) == 2


# ---------------------------------------------------------------------------
# Run discovery + trial grouping
# ---------------------------------------------------------------------------


def _write_run(log_dir: Path, rel: str) -> Path:
    run = log_dir / rel
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        json.dumps({"type": "config", "slug": "s", "hostname": "h"})
    )
    return run


def test_load_runs_returns_relative_posix_paths(tmp_path: Path) -> None:
    """`load_runs` returns POSIX paths relative to `log_dir` so nested
    trial layouts round-trip through the other helpers."""
    from pawn.dashboard.metrics import load_runs

    _write_run(tmp_path, "trial_0001/run_foo")
    _write_run(tmp_path, "run_bar")
    runs = load_runs(tmp_path, max_age_hours=0)  # 0 → no age filter
    assert "trial_0001/run_foo" in runs
    assert "run_bar" in runs


def test_load_runs_age_filter_excludes_old(tmp_path: Path) -> None:
    """A run whose metrics.jsonl mtime predates the cutoff is excluded."""
    import os
    import time

    from pawn.dashboard.metrics import load_runs

    run = _write_run(tmp_path, "old_run")
    old = time.time() - 3 * 3600  # 3 hours ago
    os.utime(run / "metrics.jsonl", (old, old))
    assert load_runs(tmp_path, max_age_hours=1.0) == []
    assert "old_run" in load_runs(tmp_path, max_age_hours=0)


def test_load_runs_does_not_descend_into_run_dirs(tmp_path: Path) -> None:
    """A run dir is a leaf — a `checkpoints/` subdir carrying its own
    `metrics.jsonl` (hypothetically) must not be discovered as a nested
    run that shadows the real one."""
    from pawn.dashboard.metrics import load_runs

    run = _write_run(tmp_path, "run_foo")
    nested = run / "checkpoints"
    nested.mkdir()
    (nested / "metrics.jsonl").write_text("{}")
    runs = load_runs(tmp_path, max_age_hours=0)
    assert runs == ["run_foo"]


def test_discover_runs_does_not_descend_into_run_dirs(tmp_path: Path) -> None:
    """`discover_runs` must stop at the first `metrics.jsonl` in a directory
    (shares `_iter_run_dirs` with `load_runs`/`list_trials`). A run dir that
    carries a nested `checkpoints/metrics.jsonl` must appear exactly once —
    an `rglob` walk would double-count it (once as the run dir, once as the
    checkpoints sub-path)."""
    from pawn.dashboard.metrics import discover_runs

    run = _write_run(tmp_path, "run_foo")
    nested = run / "checkpoints"
    nested.mkdir()
    (nested / "metrics.jsonl").write_text("{}")
    discovered = discover_runs(tmp_path)
    assert discovered == [run]


def test_list_trials_groups_nested_runs(tmp_path: Path) -> None:
    from pawn.dashboard.metrics import list_trials

    _write_run(tmp_path, "trial_a/run_1")
    _write_run(tmp_path, "trial_b/run_1")
    trials = list_trials(tmp_path)
    assert set(trials) == {"trial_a", "trial_b"}


def test_list_trials_empty_for_flat_layout(tmp_path: Path) -> None:
    """A flat layout (every run a direct child) has no trial grouping."""
    from pawn.dashboard.metrics import list_trials

    _write_run(tmp_path, "run_1")
    _write_run(tmp_path, "run_2")
    assert list_trials(tmp_path) == []


def test_get_run_meta_reads_config_record(tmp_path: Path) -> None:
    from pawn.dashboard.metrics import get_run_hostname, get_run_meta

    run = tmp_path / "trial/run"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "config", "hostname": "pod7",
                    "slug": "bold-fox", "variant": "base"}),
        json.dumps({"type": "train", "step": 1, "loss": 1.0}),
    ]))
    meta = get_run_meta(tmp_path, "trial/run")
    assert meta == {"hostname": "pod7", "slug": "bold-fox", "variant": "base"}
    assert get_run_hostname(tmp_path, "trial/run") == "pod7"


def test_get_run_meta_missing_returns_empty(tmp_path: Path) -> None:
    from pawn.dashboard.metrics import get_run_meta

    assert get_run_meta(tmp_path, "nope") == {}


# ---------------------------------------------------------------------------
# detect_run_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, expected",
    [
        # The live adapter-trainer config shape: `run_type` is the literal
        # "adapter" and the per-strategy name lives in a nested `config`
        # sub-dict (AdapterConfig.run_type is Literal["adapter"];
        # scripts/train_jax_adapter.py logs
        # `log_config(run_type="adapter", config=cfg.model_dump())`).
        ({"run_type": "adapter", "config": {"strategy": "lora"}}, "lora"),
        ({"run_type": "adapter", "config": {"strategy": "rosa"}}, "rosa"),
        ({"run_type": "adapter", "config": {"strategy": "unfreeze"}}, "unfreeze"),
        (
            {"run_type": "adapter", "config": {"strategy": "specialized_clm"}},
            "tiny",
        ),
        # Flat strategy key (older / sweep records) still resolves.
        ({"strategy": "film"}, "film"),
        ({"run_type": "specialized_clm"}, "tiny"),
        ({"run_type": "pretrain"}, "pawn"),
        ({"run_type": "pawn"}, "pawn"),
        ({"formulation": "clm"}, "pawn"),
        ({"pgn": "thomas-schweich/pawn-lichess-full"}, "bc"),
        # A non-string run_type must not leak through the `-> str` boundary.
        ({"run_type": 1968}, "pawn"),
        ({}, "pawn"),
    ],
)
def test_detect_run_type(config: dict, expected: str) -> None:
    from pawn.dashboard.metrics import detect_run_type

    result = detect_run_type(config)
    assert result == expected
    assert isinstance(result, str)


def test_col_skips_missing_and_none(tmp_path: Path) -> None:
    from pawn.dashboard.metrics import col

    records = [{"x": 1}, {"x": None}, {}, {"x": 3}]
    assert col(records, "x") == [1, 3]


# ---------------------------------------------------------------------------
# Per-run notes — save/load annotations round-trip
# ---------------------------------------------------------------------------


def test_notes_roundtrip_flat_run(tmp_path: Path) -> None:
    from pawn.dashboard.metrics import load_notes, notes_path, save_notes

    _write_run(tmp_path, "run_x")
    assert load_notes(tmp_path, "run_x") == ""  # missing → empty
    written = save_notes(tmp_path, "run_x", "tried lr=3e-4, diverged")
    assert written == notes_path(tmp_path, "run_x")
    assert written == tmp_path / "run_x" / "notes.md"
    assert load_notes(tmp_path, "run_x") == "tried lr=3e-4, diverged"


def test_notes_are_trial_scoped(tmp_path: Path) -> None:
    """A nested run stores notes at the trial level so every run in the
    trial shares one annotation file."""
    from pawn.dashboard.metrics import notes_path, save_notes

    _write_run(tmp_path, "trial_a/run_1")
    _write_run(tmp_path, "trial_a/run_2")
    save_notes(tmp_path, "trial_a/run_1", "shared note")
    # The second run resolves to the same (trial-level) notes file.
    assert notes_path(tmp_path, "trial_a/run_2") == tmp_path / "trial_a" / "notes.md"
    from pawn.dashboard.metrics import load_notes
    assert load_notes(tmp_path, "trial_a/run_2") == "shared note"


# ---------------------------------------------------------------------------
# HF metrics sync — wiring + graceful degradation
# ---------------------------------------------------------------------------


def test_sync_hf_metrics_targets_v2_repos() -> None:
    """The sync pulls from the v2 republished repos, not the frozen v1
    PyTorch ones (model-card / repo autodetect — no v1 hardcode)."""
    from pawn.dashboard.metrics import HF_REPOS

    assert all(r.endswith("-v2") for r in HF_REPOS)
    assert "thomas-schweich/pawn-base-v2" in HF_REPOS


def test_sync_hf_metrics_returns_empty_without_hf_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `huggingface_hub` can't be imported, the sync degrades to a
    no-op rather than crashing the dashboard."""
    import builtins

    from pawn.dashboard import metrics as metrics_mod

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name == "huggingface_hub":
            raise ImportError("simulated missing huggingface_hub")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert metrics_mod.sync_hf_metrics(tmp_path) == []


def test_sync_hf_metrics_downloads_run_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync enumerates `run/*` branches and downloads each branch's
    `metrics.jsonl` into `log_dir/<run_id>/`."""
    import sys
    import types

    from pawn.dashboard import metrics as metrics_mod

    class _Branch:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Refs:
        branches = [_Branch("run/run_001"), _Branch("main"), _Branch("run/run_002")]

    class _Api:
        def list_repo_refs(self, repo_id: str, repo_type: str = "model") -> _Refs:  # noqa: ANN001
            # Only the first repo has live run branches; the others raise.
            if repo_id == metrics_mod.HF_REPOS[0]:
                return _Refs()
            raise RuntimeError("no such repo")

    downloaded: list[tuple[str, str]] = []

    def _hf_hub_download(*, repo_id: str, filename: str, revision: str,
                         repo_type: str, local_dir: str) -> str:
        downloaded.append((revision, local_dir))
        Path(local_dir, filename).write_text("{}")
        return str(Path(local_dir, filename))

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _Api  # type: ignore[attr-defined]
    fake_hub.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    synced = metrics_mod.sync_hf_metrics(tmp_path)
    assert set(synced) == {"run_001", "run_002"}
    assert (tmp_path / "run_001" / "metrics.jsonl").is_file()
    assert (tmp_path / "run_002" / "metrics.jsonl").is_file()
