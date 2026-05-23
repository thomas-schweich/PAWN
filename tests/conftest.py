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
