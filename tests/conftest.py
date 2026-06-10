"""Shared fixtures and hooks for the PAWN test suite.

Partition-local fixtures belong in ``tests/<partition>/conftest.py``.
Only genuinely cross-partition fixtures live here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# W&B: default to disabled mode so tests never hit the network.
# Individual tests override via monkeypatch to exercise specific modes.
# ---------------------------------------------------------------------------

os.environ.setdefault("PAWN_WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_MODE", "disabled")


# ---------------------------------------------------------------------------
# BUG-N enforcement hook
# ---------------------------------------------------------------------------
#
# Every xfail marker MUST cite a BUG-N identifier so it appears in
# ``tests/BUGS.md``. Unconditional xfails without a BUG-N reason are
# rejected at collection time — this prevents ``xfail`` from being used as
# a convenience escape hatch.

_BUG_REASON_RE = re.compile(r"^BUG-\d+:\s+.+")


def pytest_collection_modifyitems(config, items):
    failures = []
    for item in items:
        for marker in item.iter_markers(name="xfail"):
            # xfail with a `condition` that may be False at runtime is still
            # valid; only require the reason format when a reason is given.
            reason = marker.kwargs.get("reason", "")
            if not reason or not _BUG_REASON_RE.match(reason):
                failures.append(
                    f"{item.nodeid}: xfail reason must match "
                    f"'BUG-N: <summary>' (got {reason!r})"
                )
    if failures:
        raise pytest.UsageError(
            "xfail discipline violated — every xfail must cite a BUG-N:\n  "
            + "\n  ".join(failures)
        )


# ---------------------------------------------------------------------------
# JAX persistent compilation cache (session-scoped, autouse)
# ---------------------------------------------------------------------------
#
# Phase C tests compile slow ROCm kernels; without a persistent cache every
# pytest process re-pays the XLA/ROCm compile cost on first jit. Enabling
# ``setup_jax_caching()`` keys executables by ``(jaxlib version, GPU
# platform, HLO hash)`` and writes them to disk, so they're reused across
# every pytest invocation — and the cache dir is shared with the
# training/bench scripts that already call it, giving cross-hits on matching
# TINY configs.
#
# This MUST run before any ``jax.jit`` / ``eqx.filter_jit`` compile, hence
# ``autouse=True`` with session scope: pytest instantiates session-scoped
# autouse fixtures before the first test body executes. It's correctness-
# neutral (a stale-cache miss after a jaxlib/driver upgrade just falls back
# to recompilation) and ``setup_jax_caching`` is idempotent, so the single
# call here is sufficient for the whole session.


@pytest.fixture(scope="session", autouse=True)
def _jax_compilation_cache() -> None:
    """Enable JAX's persistent compilation cache for the whole test session.

    Importing :mod:`pawn.jax_setup` is cheap; the function itself imports
    ``jax`` lazily, so this doesn't drag JAX into tests that never touch it
    beyond the (already-paid) configuration call.
    """
    from pawn.jax_setup import setup_jax_caching

    setup_jax_caching()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rust_seed() -> int:
    """Canonical deterministic seed for chess_engine calls across the suite."""
    return 42


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Fresh empty directory suitable for checkpoint save/load round-trips."""
    d = tmp_path / "ckpt"
    d.mkdir()
    return d


@pytest.fixture
def freeze_numpy_rng() -> Iterator[None]:
    """Snapshot and restore numpy + python RNG around a test.

    Useful when a test needs to seed globally without contaminating sibling tests.
    """
    import random

    import numpy as np

    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
