"""Tests for `pawn.legacy` — v1 PyTorch → v2 JAX checkpoint converter.

Round-trip: build a synthetic v1-shape checkpoint, convert it, load the
v2 result via `pawn.checkpoint.load_model`, run forward and confirm
shape + finite logits. The full forward-parity test against a real v1
torch checkpoint runs as part of S13's
`scripts/convert_published_checkpoints.py`; here we exercise the
synthetic-fixture path which validates the transpose + stack logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pawn._torch_legacy_fixture import LegacyConfig, save_legacy_checkpoint
from pawn.checkpoint import load_model
from pawn.config import NUM_ACTIONS, VOCAB_SIZE
from pawn.legacy import convert_legacy_checkpoint


# ---------------------------------------------------------------------------
# Round-trip: convert + load + forward
# ---------------------------------------------------------------------------


def test_convert_round_trip_synthetic_v1(tmp_path: Path) -> None:
    """A synthetic v1 checkpoint converts, loads via `load_model`,
    and forward-evaluates without error."""
    legacy_cfg = LegacyConfig(
        d_model=64, n_layers=2, n_heads=2, d_ff=128
    )
    v1_dir = tmp_path / "v1_input"
    save_legacy_checkpoint(legacy_cfg, v1_dir, seed=42)

    out_dir = tmp_path / "v2_output"
    converted = convert_legacy_checkpoint(str(v1_dir), output_dir=out_dir)
    assert converted == out_dir

    model = load_model(out_dir)
    assert model.cfg.d_model == 64
    assert model.cfg.n_layers == 2
    assert model.cfg.n_heads == 2
    assert model.cfg.head_dim == 32  # 64 / 2

    tokens = jnp.zeros((1, 8), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (1, 8, VOCAB_SIZE)
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# Pre-vocab-transition rejection
# ---------------------------------------------------------------------------


def test_convert_rejects_pre_vocab_transition_checkpoint(tmp_path: Path) -> None:
    """A v1 checkpoint with the older ~60k vocab is rejected loudly."""
    legacy_cfg = LegacyConfig(d_model=64, n_layers=1, n_heads=1, d_ff=128)
    v1_dir = tmp_path / "v1_input"
    save_legacy_checkpoint(legacy_cfg, v1_dir, seed=0)
    # Rewrite the config.json with a non-1980 vocab size.
    cfg_path = v1_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["vocab_size"] = 4278  # the pre-vocab-transition era size
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-vocab-transition"):
        convert_legacy_checkpoint(str(v1_dir), output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_convert_cache_returns_existing_dir_on_second_call(tmp_path: Path) -> None:
    """A second call with the same source short-circuits."""
    legacy_cfg = LegacyConfig(d_model=64, n_layers=1, n_heads=1, d_ff=128)
    v1_dir = tmp_path / "v1_input"
    save_legacy_checkpoint(legacy_cfg, v1_dir, seed=0)
    out_dir = tmp_path / "v2_output"

    first = convert_legacy_checkpoint(str(v1_dir), output_dir=out_dir)
    # Delete the source — the cache hit should not need it again.
    import shutil

    shutil.rmtree(v1_dir)
    second = convert_legacy_checkpoint(str(v1_dir), output_dir=out_dir)
    assert first == second


def test_convert_force_reconverts(tmp_path: Path) -> None:
    """`force=True` re-runs the conversion even if the cache exists."""
    legacy_cfg = LegacyConfig(d_model=64, n_layers=1, n_heads=1, d_ff=128)
    v1_dir = tmp_path / "v1_input"
    save_legacy_checkpoint(legacy_cfg, v1_dir, seed=0)
    out_dir = tmp_path / "v2_output"

    convert_legacy_checkpoint(str(v1_dir), output_dir=out_dir)
    # force=True deletes + re-runs.
    convert_legacy_checkpoint(str(v1_dir), output_dir=out_dir, force=True)


# ---------------------------------------------------------------------------
# Transpose contract
# ---------------------------------------------------------------------------


def test_convert_transposes_lm_head(tmp_path: Path) -> None:
    """v1 stored lm_head as (V, d); v2 stores as (d, V). The converter
    transposes."""
    legacy_cfg = LegacyConfig(d_model=64, n_layers=1, n_heads=2, d_ff=128)
    v1_dir = tmp_path / "v1_input"
    save_legacy_checkpoint(legacy_cfg, v1_dir, seed=0)
    out_dir = tmp_path / "v2_output"
    convert_legacy_checkpoint(str(v1_dir), output_dir=out_dir)
    model = load_model(out_dir)
    # v2 lm_head shape: (d_model, vocab_size).
    assert model.lm_head.shape == (64, VOCAB_SIZE)


def test_convert_stacks_per_layer_linears(tmp_path: Path) -> None:
    """v1 had `layers.0.attn.wq.weight`, `layers.1.attn.wq.weight`, ...
    v2 has `layers.wq` shape `(n_layers, in, out)`. The converter
    stacks across the layer index."""
    legacy_cfg = LegacyConfig(d_model=64, n_layers=3, n_heads=2, d_ff=128)
    v1_dir = tmp_path / "v1_input"
    save_legacy_checkpoint(legacy_cfg, v1_dir, seed=0)
    out_dir = tmp_path / "v2_output"
    convert_legacy_checkpoint(str(v1_dir), output_dir=out_dir)
    model = load_model(out_dir)
    # 3 layers, each (in=d, out=d) → stacked (3, 64, 64).
    assert model.layers.wq.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# Missing source handling
# ---------------------------------------------------------------------------


def test_convert_rejects_missing_safetensors(tmp_path: Path) -> None:
    v1_dir = tmp_path / "v1_input"
    v1_dir.mkdir()
    (v1_dir / "config.json").write_text(
        json.dumps({"vocab_size": VOCAB_SIZE, "d_model": 64, "n_layers": 1,
                    "n_heads": 1, "d_ff": 128})
    )
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        convert_legacy_checkpoint(str(v1_dir), output_dir=tmp_path / "out")


def test_convert_rejects_missing_config(tmp_path: Path) -> None:
    v1_dir = tmp_path / "v1_input"
    v1_dir.mkdir()
    from safetensors.numpy import save_file
    save_file({"dummy": np.zeros(4, dtype=np.float32)},
              str(v1_dir / "model.safetensors"))
    with pytest.raises(FileNotFoundError, match="config.json"):
        convert_legacy_checkpoint(str(v1_dir), output_dir=tmp_path / "out")
