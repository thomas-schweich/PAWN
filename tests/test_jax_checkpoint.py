"""Tests for ``pawn.jax.checkpoint`` — round-trip and integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import jax
import jax.numpy as jnp

from pawn.jax.checkpoint import (
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    UnsupportedCheckpointVersionError,
    load_model,
    read_model_config,
    save_model,
    verify_checkpoint,
)
from pawn.jax.config import VARIANTS
from pawn.jax.model import PAWNModel, init_model

pytestmark = pytest.mark.integration


_PARAM_FIELDS = (
    "src_embed", "dst_embed", "promo_embed", "pad_embed", "outcome_embed",
    "attn_norm", "wq", "wk", "wv", "wo",
    "ffn_norm", "w_gate", "w_up", "w_down",
    "final_norm", "lm_head",
)


@pytest.fixture
def small_model() -> PAWNModel:
    return init_model(VARIANTS["small"], jax.random.PRNGKey(7))


def test_round_trip_bit_identical(small_model: PAWNModel, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    loaded = load_model(ckpt)
    assert loaded.cfg == small_model.cfg
    for field in _PARAM_FIELDS:
        assert bool(jnp.array_equal(
            getattr(small_model, field), getattr(loaded, field)
        )), f"{field} not bit-identical after round-trip"


def test_round_trip_forward_identical(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    loaded = load_model(ckpt)
    tokens = jax.random.randint(jax.random.PRNGKey(0), (1, 16), 0, 1968)
    mask = jnp.ones((1, 16), dtype=bool)
    assert bool(jnp.array_equal(
        small_model(tokens, mask), loaded(tokens, mask)
    ))


def test_save_creates_expected_files(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    assert (ckpt / ".complete").exists()
    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "config.json").exists()
    assert not (tmp_path / "ckpt.tmp").exists()
    assert not (tmp_path / "ckpt.bak").exists()


def test_missing_sentinel_raises(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    (ckpt / ".complete").unlink()
    with pytest.raises(IncompleteCheckpointError):
        load_model(ckpt)


def test_corrupted_file_raises(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    with open(ckpt / "model.safetensors", "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(CheckpointIntegrityError):
        load_model(ckpt)


def test_extra_file_raises(small_model: PAWNModel, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    (ckpt / "stray.txt").write_text("oops", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        load_model(ckpt)


def test_unsupported_version_raises(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    cfg_path = ckpt / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["format_version"] = 999
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(UnsupportedCheckpointVersionError):
        read_model_config(ckpt)


def test_atomic_overwrite_cleans_up(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    save_model(ckpt, small_model)  # overwrite onto the same path
    assert (ckpt / ".complete").exists()
    assert not (tmp_path / "ckpt.tmp").exists()
    assert not (tmp_path / "ckpt.bak").exists()
    load_model(ckpt)  # still loadable


def test_verify_checkpoint_public(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    verify_checkpoint(ckpt)  # no exception on a valid checkpoint
    (ckpt / ".complete").unlink()
    with pytest.raises(IncompleteCheckpointError):
        verify_checkpoint(ckpt)
