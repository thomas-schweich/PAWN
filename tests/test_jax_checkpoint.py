"""Tests for :mod:`pawn.checkpoint` — atomic safetensors save/load.

Coverage:
- Round-trip: save a fresh model, load it, every saved field
  byte-identical, forward pass matches.
- Atomic write: a crashed save leaves either no final dir or a complete
  one; an orphan ``.tmp`` from a prior save is cleaned up on next save.
- Sentinel verification: a corrupted file is rejected at load.
- Schema enforcement: extra / missing tensors are rejected; a tensor
  with the wrong shape is rejected.
- ``config.json`` version + structure validation.
- :func:`load_model_config` cheap inspection skips the safetensors load.
- :data:`SAVED_FIELDS` has exactly 16 entries (re-pinned here so a
  drift caught in pawn.model also shows up at the checkpoint contract).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pawn._sentinel import (
    SENTINEL_NAME,
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    write_sentinel,
)
from pawn.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CONFIG_FILE,
    MODEL_FILE,
    load_model,
    load_model_config,
    save_model,
)
from pawn.config import TINY_SUPERNET, TINY_VARIANTS
from pawn.model import SAVED_FIELDS, init_model, sliced


def _tamper_tensors_and_rehash(
    out_dir: Path,
    mutate: Callable[[dict[str, "np.ndarray"]], dict[str, "np.ndarray"]],
) -> None:
    """Rewrite `model.safetensors` via `mutate(tensors)` and refresh the
    .complete manifest. Used by integrity tests that need to exercise a
    branch *past* the sentinel hash check (e.g. wrong-shape, extra/missing
    tensor)."""
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(out_dir / MODEL_FILE))
    save_file(mutate(tensors), str(out_dir / MODEL_FILE))
    (out_dir / SENTINEL_NAME).unlink()
    write_sentinel(out_dir, [MODEL_FILE, CONFIG_FILE])


def _tamper_config_and_rehash(
    out_dir: Path, mutate: Callable[[dict], dict]
) -> None:
    """Rewrite `config.json` via `mutate(cfg)` and refresh the .complete
    manifest. Mirror of `_tamper_tensors_and_rehash` for the config branch.

    The helper rewrites the sentinel to cover only `MODEL_FILE` +
    `CONFIG_FILE`. If a test saved with optional payloads (optimizer /
    training_state) and then called this helper, the rewritten manifest
    would silently drop them. None of the current callers do that.
    """
    raw = json.loads((out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    (out_dir / CONFIG_FILE).write_text(
        json.dumps(mutate(raw)) + "\n", encoding="utf-8"
    )
    (out_dir / SENTINEL_NAME).unlink()
    write_sentinel(out_dir, [MODEL_FILE, CONFIG_FILE])


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Saving and loading a fresh model yields a model with byte-identical
    saved fields and an identical forward pass."""
    orig = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    final_dir = save_model(orig, out_dir)
    assert final_dir == out_dir
    assert (out_dir / SENTINEL_NAME).is_file()
    assert (out_dir / MODEL_FILE).is_file()
    assert (out_dir / CONFIG_FILE).is_file()

    loaded = load_model(out_dir)

    # Every SAVED_FIELDS tensor is byte-identical.
    for path in SAVED_FIELDS:
        o = orig
        l = loaded
        for piece in path.split("."):
            o = getattr(o, piece)
            l = getattr(l, piece)
        assert jnp.array_equal(jnp.asarray(o), jnp.asarray(l)), (
            f"saved field {path} changed across save/load"
        )

    # Forward pass on a small batch matches.
    tokens = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    out_a = orig(tokens)
    out_b = loaded(tokens)
    assert jnp.allclose(out_a, out_b, atol=1e-6)


def test_save_load_round_trip_sliced_variant(tmp_path: Path) -> None:
    """A sliced (small variant) model also round-trips cleanly — full
    forward pass on the loaded model matches the original.

    The forward-pass match is what would catch a transposed slice
    (e.g. `_tensor_dict_to_model` accidentally swapping w_gate / w_up)
    that an embedding spot-check wouldn't.
    """
    supernet = init_model(TINY_SUPERNET, key=0)
    variant = sliced(supernet, TINY_VARIANTS["small"])
    out_dir = tmp_path / "small_variant"
    save_model(variant, out_dir)
    loaded = load_model(out_dir)
    assert loaded.cfg.d_model == TINY_VARIANTS["small"].d_model

    # Every saved field is byte-identical
    for path in SAVED_FIELDS:
        o, l = variant, loaded
        for piece in path.split("."):
            o, l = getattr(o, piece), getattr(l, piece)
        assert jnp.array_equal(jnp.asarray(o), jnp.asarray(l)), (
            f"saved field {path} changed across save/load"
        )

    # Forward pass on the loaded variant matches the original at atol=1e-6.
    tokens = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    assert jnp.allclose(variant(tokens), loaded(tokens), atol=1e-6)


def test_save_with_run_config_and_optimizer(tmp_path: Path) -> None:
    """Optional payloads land on disk, the manifest covers them, and the
    full `load_model` path still works with the optional payloads present
    (regression-guards against future code that keys on manifest length)."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000020"
    fake_opt = {
        "adam_m": np.zeros(8, dtype=np.float32),
        "adam_v": np.ones(8, dtype=np.float32),
    }
    save_model(
        model,
        out_dir,
        run_config={"strategy": "lora", "lora_rank": 4},
        optimizer_state=fake_opt,
        training_state={"step": 1234, "rng_b64": "deadbeef"},
    )
    assert (out_dir / "optimizer.safetensors").is_file()
    assert (out_dir / "training_state.json").is_file()

    # config.json carries the run block.
    raw = json.loads((out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert raw["run"] == {"strategy": "lora", "lora_rank": 4}

    # training_state.json round-trips intact.
    ts = json.loads((out_dir / "training_state.json").read_text(encoding="utf-8"))
    assert ts == {"step": 1234, "rng_b64": "deadbeef"}

    # `load_model` still works with the optional payloads present — the
    # extra files in the manifest don't trip integrity or schema checks.
    loaded = load_model(out_dir)
    assert loaded.cfg == model.cfg
    assert jnp.array_equal(loaded.embed_src, model.embed_src)


# ---------------------------------------------------------------------------
# Atomic-write contract
# ---------------------------------------------------------------------------


def test_save_refuses_to_overwrite_existing_dir(tmp_path: Path) -> None:
    """A save into a dir that already exists is rejected — checkpoints are
    immutable by design."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_model(model, out_dir)


def test_save_cleans_orphan_tmp_dir(tmp_path: Path) -> None:
    """An orphan ``<target>.tmp`` from a prior crashed save gets blown
    away on the next save's start."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    orphan = tmp_path / "step_00000010.tmp"
    orphan.mkdir()
    (orphan / "stale_payload").write_text("garbage")

    final_dir = save_model(model, out_dir)
    assert final_dir == out_dir
    assert not orphan.exists()  # Cleaned up
    assert (out_dir / SENTINEL_NAME).is_file()


def test_save_renames_atomically(tmp_path: Path) -> None:
    """After `save_model` returns, the target dir exists and the tmp dir
    is gone."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    assert out_dir.is_dir()
    assert not (tmp_path / "step_00000010.tmp").exists()


def test_save_cleans_tmp_on_mid_save_failure(tmp_path: Path) -> None:
    """If a write step raises after `tmp.mkdir()`, the partial `.tmp`
    directory is removed before the exception propagates — orphans
    don't linger past the failing call."""
    import unittest.mock

    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    tmp_dir = tmp_path / "step_00000010.tmp"

    # Mock the safetensors save call so it raises after `tmp.mkdir()`
    # but before any payload file lands.
    with unittest.mock.patch(
        "pawn.checkpoint.st_save",
        side_effect=RuntimeError("simulated mid-save failure"),
    ):
        with pytest.raises(RuntimeError, match="simulated mid-save failure"):
            save_model(model, out_dir)

    # No `.tmp` orphan, no final dir.
    assert not tmp_dir.exists()
    assert not out_dir.exists()


def test_save_cleans_orphan_regular_file_at_tmp_path(tmp_path: Path) -> None:
    """A non-directory at the `.tmp` path (e.g. a regular file left by
    debug tooling) is removed instead of tripping `shutil.rmtree`'s
    NotADirectoryError."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    bogus_tmp = tmp_path / "step_00000010.tmp"
    bogus_tmp.write_text("an actual file, not a directory")
    save_model(model, out_dir)
    assert out_dir.is_dir()
    assert not bogus_tmp.exists()


# ---------------------------------------------------------------------------
# Integrity enforcement
# ---------------------------------------------------------------------------


def test_load_detects_corrupted_model_file(tmp_path: Path) -> None:
    """If a byte of `model.safetensors` is corrupted after the save, load
    raises `CheckpointIntegrityError` (the sentinel SHA mismatches)."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    # Tamper with the saved tensors.
    target = out_dir / MODEL_FILE
    data = target.read_bytes()
    # Flip a byte well past the header.
    pos = len(data) - 100
    mutated = data[:pos] + bytes([(data[pos] + 1) % 256]) + data[pos + 1:]
    target.write_bytes(mutated)
    with pytest.raises(CheckpointIntegrityError, match="sha256 mismatch"):
        load_model(out_dir)


def test_load_detects_missing_sentinel(tmp_path: Path) -> None:
    """A checkpoint dir without `.complete` is Incomplete, not Integrity
    — distinct error class so callers can distinguish 'crashed mid-write'
    from 'corrupted on disk'."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    (out_dir / SENTINEL_NAME).unlink()
    with pytest.raises(IncompleteCheckpointError):
        load_model(out_dir)


def test_load_rejects_missing_tensor(tmp_path: Path) -> None:
    """A model.safetensors with one of the SAVED_FIELDS missing is
    rejected. The sentinel is re-hashed so the failure fires on the
    schema check, not the SHA mismatch."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def drop_embed_pad(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(tensors)
        del d["embed_pad"]
        return d

    _tamper_tensors_and_rehash(out_dir, drop_embed_pad)
    with pytest.raises(CheckpointIntegrityError, match="missing expected tensors"):
        load_model(out_dir)


def test_load_rejects_extra_tensor(tmp_path: Path) -> None:
    """A model.safetensors with a tensor not in SAVED_FIELDS is rejected."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def add_surprise(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(tensors)
        d["surprise"] = np.zeros(4, dtype=np.float32)
        return d

    _tamper_tensors_and_rehash(out_dir, add_surprise)
    with pytest.raises(CheckpointIntegrityError, match="unexpected tensors"):
        load_model(out_dir)


def test_load_rejects_wrong_shape(tmp_path: Path) -> None:
    """A tensor with the right name but wrong shape (e.g. saved from a
    different d_model) is rejected — without this check the model would
    crash later or silently broadcast."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def wrong_embed_src(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(tensors)
        d["embed_src"] = np.zeros((64, 1234), dtype=np.float32)
        return d

    _tamper_tensors_and_rehash(out_dir, wrong_embed_src)
    with pytest.raises(CheckpointIntegrityError, match="wrong shape|expected"):
        load_model(out_dir)


def test_load_rejects_wrong_format_version(tmp_path: Path) -> None:
    """A future-versioned config.json is refused at load."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def bump_version(raw: dict) -> dict:
        raw["version"] = 9999
        return raw

    _tamper_config_and_rehash(out_dir, bump_version)
    with pytest.raises(CheckpointIntegrityError, match="version"):
        load_model(out_dir)


def test_load_rejects_unknown_config_keys(tmp_path: Path) -> None:
    """Extra keys in the `model` block of config.json fail load — better
    to refuse a future-format manifest than to silently drop fields."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def add_mystery_field(raw: dict) -> dict:
        raw["model"]["mystery_field"] = 7
        return raw

    _tamper_config_and_rehash(out_dir, add_mystery_field)
    with pytest.raises(CheckpointIntegrityError, match="unexpected keys"):
        load_model(out_dir)


def test_load_rejects_config_without_model_block(tmp_path: Path) -> None:
    """A config.json that's valid JSON but missing the `model` block is
    refused — distinct branch from unknown-keys / wrong-version."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def drop_model_block(raw: dict) -> dict:
        del raw["model"]
        return raw

    _tamper_config_and_rehash(out_dir, drop_model_block)
    with pytest.raises(CheckpointIntegrityError, match="missing the `model` block"):
        load_model(out_dir)


def test_load_rejects_config_with_non_dict_model_block(tmp_path: Path) -> None:
    """Same branch as the missing-block case but for a `model` value that
    is present but not a dict (e.g. a scalar). The guard is
    `not isinstance(model_block, dict)`, so both inputs trigger it."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def replace_model_with_scalar(raw: dict) -> dict:
        raw["model"] = 42
        return raw

    _tamper_config_and_rehash(out_dir, replace_model_with_scalar)
    with pytest.raises(CheckpointIntegrityError, match="missing the `model` block"):
        load_model(out_dir)


# ---------------------------------------------------------------------------
# Cheap inspection
# ---------------------------------------------------------------------------


def test_load_model_config_without_loading_tensors(tmp_path: Path) -> None:
    """`load_model_config` returns the cfg after verifying the sentinel,
    but doesn't load the safetensors file."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    cfg = load_model_config(out_dir)
    assert cfg == model.cfg


def test_load_rejects_manifest_missing_model_safetensors(tmp_path: Path) -> None:
    """A `.complete` manifest that doesn't claim `model.safetensors` is
    rejected even if the file is on disk and otherwise intact.

    Without this guard, a malformed manifest (e.g. hand-edited or
    produced by a broken writer) could let `load_model` read an
    unverified model.safetensors — defeating the SHA-256 contract."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    # Drop `model.safetensors` from the manifest by rewriting the sentinel
    # with only `config.json`.
    (out_dir / SENTINEL_NAME).unlink()
    write_sentinel(out_dir, [CONFIG_FILE])
    with pytest.raises(CheckpointIntegrityError, match="doesn't cover required payloads"):
        load_model(out_dir)


def test_load_rejects_manifest_missing_config_json(tmp_path: Path) -> None:
    """Same defense — a manifest that omits `config.json` is rejected."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    (out_dir / SENTINEL_NAME).unlink()
    write_sentinel(out_dir, [MODEL_FILE])
    with pytest.raises(CheckpointIntegrityError, match="doesn't cover required payloads"):
        load_model(out_dir)


def test_save_schema_is_sixteen_fields() -> None:
    """Re-pin the 16-field contract at the checkpoint API layer too —
    a drift in `pawn.model.SAVED_FIELDS` would already fire its own
    assertion at import; this is belt-and-braces."""
    assert len(SAVED_FIELDS) == 16
    assert CHECKPOINT_FORMAT_VERSION == 1
