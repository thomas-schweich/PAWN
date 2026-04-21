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
                failures.append(f"{item.nodeid}: xfail reason must match 'BUG-N: <summary>' (got {reason!r})")
    if failures:
        raise pytest.UsageError(
            "xfail discipline violated — every xfail must cite a BUG-N:\n  "
            + "\n  ".join(failures)
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cpu_device() -> str:
    """Canonical CPU device string for tests that must pin to CPU."""
    return "cpu"


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


@pytest.fixture(scope="session")
def toy_clm_config():
    """CLMConfig.toy() — d_model=64, n_layers=2. Cheap enough for CPU tests."""
    from pawn.config import CLMConfig

    return CLMConfig.toy()


@pytest.fixture(scope="session")
def toy_training_config():
    """TrainingConfig.toy() — short schedule, small batches, no AMP."""
    from pawn.config import TrainingConfig

    return TrainingConfig.toy()


@pytest.fixture
def toy_model(toy_clm_config, cpu_device):
    """Fresh PAWNCLM(toy_config) on CPU, eval mode.

    Not session-scoped: tests may mutate weights/gradients.
    """
    import torch

    from pawn.model import PAWNCLM

    torch.manual_seed(0)
    model = PAWNCLM(toy_clm_config).to(cpu_device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def sample_clm_batch(rust_seed):
    """Small deterministic CLM batch from the Rust engine.

    Returns a dict with: input_ids, targets, loss_mask, move_ids,
    game_lengths, term_codes. Session-scoped because generation is
    expensive and inputs are read-only.
    """
    import chess_engine  # type: ignore[import-not-found]

    input_ids, targets, loss_mask, move_ids, game_lengths, term_codes = (
        chess_engine.generate_clm_batch(
            batch_size=4,
            seq_len=64,
            seed=rust_seed,
        )
    )
    return {
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
        "move_ids": move_ids,
        "game_lengths": game_lengths,
        "term_codes": term_codes,
    }


@pytest.fixture
def freeze_rng() -> Iterator[None]:
    """Snapshot and restore torch + numpy + python RNG around a test.

    Useful when a test needs to seed globally without contaminating sibling tests.
    """
    import random

    import numpy as np
    import torch

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
