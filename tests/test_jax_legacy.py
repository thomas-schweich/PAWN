"""Tests for ``pawn.jax.legacy`` — PyTorch->JAX converter and logit parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("torch")
pytest.importorskip("chess_engine")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from pawn.config import CLMConfig
from pawn.jax.checkpoint import IncompleteCheckpointError, load_model
from pawn.jax.legacy import (
    IncompatibleCheckpointError,
    convert_legacy_checkpoint,
    convert_state_dict,
    legacy_to_model_config,
)
from tests._jax_helpers import write_legacy_checkpoint

pytestmark = pytest.mark.integration


def test_logit_parity_pytorch_to_jax(tmp_path: Path) -> None:
    """End-to-end: PyTorch PAWNCLM -> legacy -> JAX -> forward parity."""
    src = tmp_path / "legacy"
    cfg = CLMConfig.toy()
    torch_model = write_legacy_checkpoint(src, cfg)
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
    diff = np.abs(torch_logits.cpu().numpy() - np.asarray(jax_logits)).max()
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


def test_legacy_to_model_config_rejects_old_n_outcomes() -> None:
    """The vocab gate fires on either ``vocab_size`` or ``n_outcomes`` drift."""
    with pytest.raises(IncompatibleCheckpointError, match="n_outcomes"):
        legacy_to_model_config(
            {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "d_ff": 2048,
                "n_outcomes": 5,
            }
        )


def test_legacy_to_model_config_drops_legacy_fields() -> None:
    """``dropout`` (a legacy CLMConfig field) must not survive into ModelConfig."""
    cfg = legacy_to_model_config(
        {
            "d_model": 512,
            "n_layers": 8,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1,  # legacy-only; must be silently dropped
        }
    )
    assert cfg.d_model == 512
    assert cfg.n_layers == 8
    assert cfg.n_heads == 8
    assert cfg.d_ff == 2048
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


def test_logit_parity_with_pad_and_outcome_tokens(tmp_path: Path) -> None:
    """The parity guarantee must hold when the input includes PAD and outcome
    tokens, exercising the embedding-override branches end-to-end."""
    src = tmp_path / "legacy"
    cfg = CLMConfig.toy()
    torch_model = write_legacy_checkpoint(src, cfg)
    dst = tmp_path / "jax"
    convert_legacy_checkpoint(src, dst)
    jax_model = load_model(dst)

    b, t = 2, 24
    rng = np.random.default_rng(2)
    tokens_np = rng.integers(0, 1968, size=(b, t), dtype=np.int64)
    tokens_np[:, 0] = 1969 + 3       # outcome at position 0
    tokens_np[:, -3:] = 1968         # trailing PAD
    mask_np = np.ones((b, t), dtype=bool)
    mask_np[:, -3:] = False
    with torch.no_grad():
        torch_logits, _ = torch_model(
            torch.from_numpy(tokens_np), torch.from_numpy(mask_np)
        )
    jax_logits = jax.jit(lambda m, tk, am: m(tk, am))(
        jax_model, jnp.asarray(tokens_np), jnp.asarray(mask_np)
    )
    # PAD positions diverge by design (legacy SDPA vs JAX plain attention treat
    # fully-masked-key rows differently); compare only real-token positions.
    real = mask_np
    diff = np.abs(
        torch_logits.cpu().numpy()[real] - np.asarray(jax_logits)[real]
    ).max()
    assert diff < 1e-4, (
        f"PAD/outcome-token parity fail at real positions: max |Δ| = {diff}"
    )


@pytest.mark.parametrize(
    "payload_name", ["training_state.json", "optimizer.safetensors"]
)
def test_convert_rejects_sentinel_missing_full_checkpoint(
    tmp_path: Path, payload_name: str
) -> None:
    """A source directory containing a full-checkpoint payload file
    (optimizer.safetensors / training_state.json) but lacking ``.complete``
    is a corrupted/interrupted save, not a bare HF snapshot — the converter
    must refuse rather than silently re-signing the bytes. Parametrised so a
    regression that drops either payload file from the discriminator set
    surfaces at the test level rather than at runtime."""
    src = tmp_path / "legacy"
    write_legacy_checkpoint(src, CLMConfig.toy())
    # Drop a payload file alongside model.safetensors + config.json, no
    # .complete — mimic an interrupted full-checkpoint save.
    (src / payload_name).write_text("noise", encoding="utf-8")
    with pytest.raises(IncompleteCheckpointError, match="payload"):
        convert_legacy_checkpoint(src, tmp_path / "dst")


def test_convert_accepts_hf_snapshot_with_metadata(tmp_path: Path) -> None:
    """A directory that looks like an ``hf_hub.snapshot_download(...)``
    result — model.safetensors + config.json plus README / LICENSE — must
    convert cleanly, not be mistaken for a corrupted full checkpoint."""
    src = tmp_path / "hf_snapshot"
    cfg = CLMConfig.toy()
    write_legacy_checkpoint(src, cfg)
    for filename in ("README.md", "LICENSE", ".gitattributes"):
        (src / filename).write_text("noise", encoding="utf-8")
    dst = tmp_path / "jax"
    convert_legacy_checkpoint(src, dst)
    loaded = load_model(dst)
    assert loaded.cfg.d_model == cfg.d_model


def test_convert_legacy_checkpoint_verifies_source_sentinel(
    tmp_path: Path,
) -> None:
    """When the legacy source carries a ``.complete`` sentinel, the converter
    refuses to convert a corrupted source rather than re-signing bad bytes
    into a "valid"-looking JAX checkpoint."""
    from pawn.jax.checkpoint import (
        CheckpointIntegrityError,
        _write_sentinel,
    )
    src = tmp_path / "legacy"
    cfg = CLMConfig.toy()
    write_legacy_checkpoint(src, cfg)
    _write_sentinel(src)              # mark the source as integrity-checked
    # Corrupt the source bytes after the sentinel was written.
    sf = src / "model.safetensors"
    with open(sf, "r+b") as f:
        f.seek(-4, 2)
        original = f.read(4)
        f.seek(-4, 2)
        f.write(bytes(b ^ 0xFF for b in original))
    with pytest.raises(CheckpointIntegrityError):
        convert_legacy_checkpoint(src, tmp_path / "dst")


def test_convert_state_dict_missing_layer_weight_raises() -> None:
    """``_check_keys`` must surface a missing per-layer tensor with a clear error."""
    cfg = legacy_to_model_config(
        {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 256}
    )
    # Build a complete state dict minus one per-layer key.
    state: dict[str, np.ndarray] = {
        "embed.src_embed.weight": np.zeros((64, 64), np.float32),
        "embed.dst_embed.weight": np.zeros((64, 64), np.float32),
        "embed.promo_embed.weight": np.zeros((5, 64), np.float32),
        "embed.pad_embed": np.zeros((64,), np.float32),
        "embed.outcome_embed.weight": np.zeros((11, 64), np.float32),
        "final_norm.weight": np.zeros((64,), np.float32),
        "lm_head.weight": np.zeros((1980, 64), np.float32),
    }
    per_layer = (
        "attn_norm.weight",
        "attn.wq.weight", "attn.wk.weight", "attn.wv.weight", "attn.wo.weight",
        "ffn_norm.weight",
        "ffn.w_gate.weight", "ffn.w_up.weight", "ffn.w_down.weight",
    )
    for i in range(cfg.n_layers):
        for suf in per_layer:
            if i == 0 and suf == "attn.wq.weight":
                continue  # deliberately missing
            shape = (
                (64, 256) if "gate" in suf or "up" in suf
                else (256, 64) if "down" in suf
                else (64,) if "norm" in suf
                else (64, 64)
            )
            state[f"layers.{i}.{suf}"] = np.zeros(shape, np.float32)
    with pytest.raises(KeyError, match=r"missing required keys.*attn\.wq"):
        convert_state_dict(state, cfg)
