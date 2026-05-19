"""Tests for ``pawn.jax.torch_loader`` — load + logit-parity round-trip."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("torch")

import numpy as np
import torch
from safetensors.torch import save_file as torch_save_file

from pawn.config import CLMConfig
from pawn.jax.legacy import convert_legacy_checkpoint
from pawn.jax.torch_loader import (
    CheckpointIntegrityError,
    UnsupportedCheckpointVersionError,
    load_pawn,
)
from pawn.model import PAWNCLM

pytestmark = pytest.mark.integration


def _build_jax_checkpoint(tmp_path: Path, seed: int = 0) -> tuple[Path, PAWNCLM]:
    """Build a fresh PyTorch toy model, convert to JAX format; return (path, ref_model)."""
    cfg = CLMConfig.toy()
    torch.manual_seed(seed)
    model = PAWNCLM(cfg).eval()
    src = tmp_path / "legacy"
    src.mkdir()
    state = {
        k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
    }
    torch_save_file(state, src / "model.safetensors")
    (src / "config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint_type": "pretrain",
                "model_config": asdict(cfg),
            }
        ),
        encoding="utf-8",
    )
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
    with open(dst / "model.safetensors", "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(CheckpointIntegrityError):
        load_pawn(dst)


def test_loader_bad_version_raises(tmp_path: Path) -> None:
    dst, _ = _build_jax_checkpoint(tmp_path)
    # Removing the sentinel makes the loader skip integrity verification so
    # we can reach the format_version check with a tampered config.json.
    (dst / ".complete").unlink()
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["format_version"] = 999
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(UnsupportedCheckpointVersionError):
        load_pawn(dst)


def test_loader_missing_sentinel_is_accepted(tmp_path: Path) -> None:
    """Bare HF-format directories (no .complete) load without integrity check."""
    dst, _ = _build_jax_checkpoint(tmp_path)
    (dst / ".complete").unlink()
    load_pawn(dst)  # must not raise
