"""Tests for ``pawn.jax.legacy`` — PyTorch->JAX converter and logit parity."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("torch")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import save_file as torch_save_file

from pawn.config import CLMConfig
from pawn.jax.checkpoint import load_model
from pawn.jax.legacy import (
    IncompatibleCheckpointError,
    convert_legacy_checkpoint,
    legacy_to_model_config,
)
from pawn.model import PAWNCLM

pytestmark = pytest.mark.integration


def _write_legacy_checkpoint(
    dest: Path, cfg: CLMConfig, *, seed: int = 0
) -> PAWNCLM:
    """Materialise a legacy PyTorch checkpoint directory; return the model."""
    torch.manual_seed(seed)
    model = PAWNCLM(cfg).eval()
    dest.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
    }
    torch_save_file(state, dest / "model.safetensors")
    (dest / "config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint_type": "pretrain",
                "model_config": asdict(cfg),
            }
        ),
        encoding="utf-8",
    )
    return model


def test_logit_parity_pytorch_to_jax(tmp_path: Path) -> None:
    """End-to-end: PyTorch PAWNCLM -> legacy -> JAX -> forward parity."""
    src = tmp_path / "legacy"
    cfg = CLMConfig.toy()
    torch_model = _write_legacy_checkpoint(src, cfg)
    dst = tmp_path / "jax"
    convert_legacy_checkpoint(src, dst)
    jax_model = load_model(dst)
    assert jax_model.cfg.d_model == cfg.d_model
    assert jax_model.cfg.n_layers == cfg.n_layers
    assert jax_model.cfg.n_heads == cfg.n_heads

    b, t = 2, 24
    rng = np.random.default_rng(1)
    tokens_np = rng.integers(0, 1968, size=(b, t), dtype=np.int64)
    mask_np = np.ones((b, t), dtype=bool)
    with torch.no_grad():
        torch_logits, _ = torch_model(
            torch.from_numpy(tokens_np), torch.from_numpy(mask_np)
        )
    jax_logits = jax.jit(lambda m, tk, am: m(tk, am))(
        jax_model, jnp.asarray(tokens_np), jnp.asarray(mask_np)
    )
    diff = np.abs(torch_logits.numpy() - np.asarray(jax_logits)).max()
    assert diff < 1e-4, f"logit parity fail: max |Δ| = {diff}"


def test_legacy_to_model_config_missing_required_field() -> None:
    with pytest.raises(KeyError, match="missing required fields"):
        legacy_to_model_config({"d_model": 512})


def test_legacy_to_model_config_rejects_old_vocab() -> None:
    with pytest.raises(IncompatibleCheckpointError, match="vocab_size"):
        legacy_to_model_config(
            {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "d_ff": 2048,
                "vocab_size": 4284,
            }
        )


def test_legacy_to_model_config_drops_dropout() -> None:
    """Legacy CLMConfig has ``dropout``; the JAX ModelConfig must not."""
    cfg = legacy_to_model_config(
        {
            "d_model": 512,
            "n_layers": 8,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1,
        }
    )
    assert not hasattr(cfg, "dropout")


def test_convert_missing_model_config_key(tmp_path: Path) -> None:
    src = tmp_path / "legacy"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"format_version": 1}), encoding="utf-8"
    )
    (src / "model.safetensors").write_text("", encoding="utf-8")
    with pytest.raises(KeyError, match="model_config"):
        convert_legacy_checkpoint(src, tmp_path / "dst")
