"""Tests for ``pawn.jax.checkpoint`` — round-trip and integrity."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("chess_engine")

import jax
import jax.numpy as jnp

from pawn.jax.checkpoint import (
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    UnsupportedCheckpointVersionError,
    _PARAM_FIELDS,
    load_model,
    read_model_config,
    save_model,
    verify_checkpoint,
)
from pawn.jax.config import VARIANTS
from pawn.jax.model import PAWNModel, init_model
from tests._jax_helpers import corrupt_safetensors, stamp_format_version

pytestmark = pytest.mark.integration


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
    corrupt_safetensors(ckpt)
    with pytest.raises(CheckpointIntegrityError):
        load_model(ckpt)


def test_extra_file_raises(small_model: PAWNModel, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    (ckpt / "stray.txt").write_text("oops", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        load_model(ckpt)


def test_unsupported_version_raises_on_read_config(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    stamp_format_version(ckpt, 999)
    with pytest.raises(UnsupportedCheckpointVersionError):
        read_model_config(ckpt)


def test_unsupported_version_raises_on_load_model(
    small_model: PAWNModel, tmp_path: Path
) -> None:
    """``load_model`` re-validates the version even with a valid sentinel."""
    ckpt = tmp_path / "ckpt"
    save_model(ckpt, small_model)
    # stamp_format_version re-signs the sentinel so integrity passes and we
    # reach the version gate inside load_model.
    stamp_format_version(ckpt, 999)
    with pytest.raises(UnsupportedCheckpointVersionError):
        load_model(ckpt)


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
