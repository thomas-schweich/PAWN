"""Shared fixtures and hooks for the PAWN test suite (post-JAX-migration).

The legacy PyTorch fixtures (``toy_model``, ``sample_clm_batch`` via
torch, ``freeze_rng`` over torch RNG) are gone with the PyTorch
removal in Phase 4. JAX-side equivalents live in
``tests/test_jax_*.py`` as local helpers — see ``_random_batch`` /
``_real_batch`` in ``tests/test_jax_trainer.py`` and
``test_jax_adapter_trainer.py``.
"""

from __future__ import annotations

import os
import re

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


@pytest.fixture(scope="session")
def rust_seed() -> int:
    """Canonical deterministic seed for chess_engine calls across the suite."""
    return 42
