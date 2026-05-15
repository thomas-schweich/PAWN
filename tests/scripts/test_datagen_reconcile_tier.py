"""Pinning tests for `scripts/datagen_reconcile_tier.py`.

These tests guard the on-disk layout decisions that downstream consumers
rely on, in particular:

  - Canonical manifest + tier-state live under `_meta/<tier_name>/...`
    (not the legacy `<tier_name>/...` location). The motivation is HF's
    hard 10000-files-per-directory cap: a tier configured for 10000
    shards has no room for in-directory sidecars after the last shard
    lands, so sidecars must be hoisted out.

  - The `--all-tiers` skip-already-reconciled check honors *both* the
    current `_meta/<tier>/_manifest.json` location and the legacy
    `<tier>/_manifest.json` location, so older reconciled datasets are
    not re-reconciled into a duplicate location.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_reconcile_module() -> Any:
    """Load `scripts/datagen_reconcile_tier.py` as a module so we can
    inspect its internal helpers without spawning a subprocess."""
    if "datagen_reconcile_tier" in sys.modules:
        return sys.modules["datagen_reconcile_tier"]
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "datagen_reconcile_tier.py"
    spec = importlib.util.spec_from_file_location("datagen_reconcile_tier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reconciler from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["datagen_reconcile_tier"] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_paths_under_meta_prefix() -> None:
    """Canonical manifest + state belong under `_meta/<tier>/...`, NOT
    in the tier directory. Regression guard: if someone moves them back
    into `<tier>/` we'll hit the 10K-files-per-directory wall on every
    tier that fills to 10000 shards (the dominant case for this
    dataset's 100M-game runs)."""
    mod = _load_reconcile_module()

    assert mod.META_PREFIX == "_meta"
    assert mod.canonical_manifest_path("nodes_0001") == "_meta/nodes_0001/_manifest.json"
    assert mod.canonical_tier_state_path("nodes_0001") == "_meta/nodes_0001/_tier_state.json"

    # No accidental coupling to the legacy in-tier location.
    assert "_meta/" in mod.canonical_manifest_path("any_tier")
    assert not mod.canonical_manifest_path("any_tier").startswith("any_tier/")


def test_meta_layout_does_not_consume_tier_directory_budget() -> None:
    """A tier directory configured for 10000 shards must be able to
    reach exactly 10000 parquet files without the canonical sidecars
    eating into the budget. This test enforces that the canonical
    paths the reconciler writes to do NOT live under `<tier>/`."""
    mod = _load_reconcile_module()
    for tier in ("tier0_evallegal", "nodes_0001", "nodes_0128", "nodes_0256", "nodes_1024"):
        for path in (mod.canonical_manifest_path(tier), mod.canonical_tier_state_path(tier)):
            # The path must NOT be in the tier directory.
            assert not path.startswith(f"{tier}/"), (
                f"canonical sidecar {path!r} lives inside `{tier}/` and "
                f"will compete for the 10K-files-per-directory budget "
                f"that the tier itself needs for its 10000 shards"
            )
            # It must be under `_meta/<tier>/` so a future consumer
            # can find tier-scoped metadata by name.
            assert path.startswith(f"_meta/{tier}/"), (
                f"canonical sidecar {path!r} should be under `_meta/{tier}/`"
            )
