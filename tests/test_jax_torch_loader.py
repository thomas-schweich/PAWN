"""Tests for ``pawn.jax.torch_loader`` — load + logit-parity round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("torch")
pytest.importorskip("chess_engine")

import numpy as np
import torch

from pawn.config import CLMConfig
from pawn.jax.legacy import convert_legacy_checkpoint
from pawn.jax.torch_loader import (
    CheckpointIntegrityError,
    UnsupportedCheckpointVersionError,
    load_pawn,
)
from pawn.model import PAWNCLM
from tests._jax_helpers import (
    corrupt_safetensors,
    stamp_format_version,
    write_legacy_checkpoint,
)

pytestmark = pytest.mark.integration


def _build_jax_checkpoint(tmp_path: Path, seed: int = 0) -> tuple[Path, PAWNCLM]:
    """Build a fresh PyTorch toy model, convert to JAX format."""
    cfg = CLMConfig.toy()
    src = tmp_path / "legacy"
    model = write_legacy_checkpoint(src, cfg, seed=seed)
    dst = tmp_path / "jax"
    convert_legacy_checkpoint(src, dst)
    return dst, model


def test_loader_logit_parity(tmp_path: Path) -> None:
    """End-to-end: PyTorch -> legacy -> JAX -> torch_loader forward parity."""
    dst, torch_model = _build_jax_checkpoint(tmp_path)
    loader_model = load_pawn(dst).eval()

    b, t = 2, 24
    rng = np.random.default_rng(1)
    tokens = torch.from_numpy(rng.integers(0, 1968, size=(b, t), dtype=np.int64))
    mask = torch.ones((b, t), dtype=torch.bool)
    with torch.no_grad():
        ref, _ = torch_model(tokens, mask)
        out = loader_model(tokens, mask)
    diff = (ref - out).abs().max().item()
    assert diff < 1e-4, f"loader parity fail: max |Δ| = {diff}"


def test_loader_corrupt_safetensors_raises(tmp_path: Path) -> None:
    dst, _ = _build_jax_checkpoint(tmp_path)
    corrupt_safetensors(dst)
    with pytest.raises(CheckpointIntegrityError):
        load_pawn(dst)


def test_loader_bad_version_raises(tmp_path: Path) -> None:
    """With a re-signed sentinel, ``load_pawn`` reaches the version gate."""
    dst, _ = _build_jax_checkpoint(tmp_path)
    stamp_format_version(dst, 999)
    with pytest.raises(UnsupportedCheckpointVersionError):
        load_pawn(dst)


def test_loader_missing_sentinel_is_accepted(tmp_path: Path) -> None:
    """Bare HF-format directories (no .complete) load without integrity check."""
    dst, _ = _build_jax_checkpoint(tmp_path)
    (dst / ".complete").unlink()
    load_pawn(dst)  # must not raise


def test_loader_max_seq_len_guard_raises(tmp_path: Path) -> None:
    """``PAWNTorch.forward`` rejects sequences longer than ``cfg.max_seq_len``."""
    dst, _ = _build_jax_checkpoint(tmp_path)
    model = load_pawn(dst).eval()
    too_long = model.cfg.max_seq_len + 1
    tokens = torch.zeros((1, too_long), dtype=torch.long)
    mask = torch.ones((1, too_long), dtype=torch.bool)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        with torch.no_grad():
            model(tokens, mask)
