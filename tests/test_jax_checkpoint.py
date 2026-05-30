"""Tests for :mod:`pawn.checkpoint` — atomic safetensors save/load.

Coverage:
- Round-trip: save a fresh model, load it, every saved field
  byte-identical, forward pass matches. Both tied (default) and untied
  (``tie_embeddings=False``) configs.
- Atomic write: a crashed save leaves either no final dir or a complete
  one; an orphan ``.tmp`` from a prior save is cleaned up on next save.
- Sentinel verification: a corrupted file is rejected at load.
- Schema enforcement: extra / missing tensors are rejected; a tensor
  with the wrong shape is rejected.
- Tied↔untied cross-load is a hard error (the save schema differs by the
  ``lm_head`` tensor, so a checkpoint saved tied refuses to load into an
  untied config and vice-versa).
- ``config.json`` version + ``mask_version`` + structure validation.
- ``load_model`` returns ``(model, run_block)`` so eval can read the
  checkpoint's own conditioning.
- bf16 first-moment (``mu``) optimizer state survives the safetensors
  round-trip bit-exact.
- :func:`load_model_config` cheap inspection skips the safetensors load.
- The save schema is the per-``tie_embeddings`` field list (re-pinned
  here so a drift caught in pawn.model also shows up at the checkpoint
  contract).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
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
    MASK_VERSION,
    MODEL_FILE,
    OPTIMIZER_FILE,
    load_model,
    load_model_config,
    save_model,
)
from pawn.config import TINY_SUPERNET, TINY_VARIANTS, ModelConfig
from pawn.model import init_model, saved_fields, sliced

TINY_UNTIED: ModelConfig = dataclasses.replace(TINY_SUPERNET, tie_embeddings=False)


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


def _assert_fields_byte_identical(orig: object, loaded: object, fields: tuple[str, ...]) -> None:
    """Every named (dotted) field is byte-identical between two models."""
    for path in fields:
        o, l = orig, loaded
        for piece in path.split("."):
            o, l = getattr(o, piece), getattr(l, piece)
        assert jnp.array_equal(jnp.asarray(o), jnp.asarray(l)), (
            f"saved field {path} changed across save/load"
        )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Saving and loading a fresh (tied) model yields a model with
    byte-identical saved fields and an identical forward pass."""
    orig = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    final_dir = save_model(orig, out_dir)
    assert final_dir == out_dir
    assert (out_dir / SENTINEL_NAME).is_file()
    assert (out_dir / MODEL_FILE).is_file()
    assert (out_dir / CONFIG_FILE).is_file()

    loaded, run_block = load_model(out_dir)
    assert run_block is None  # no run config was supplied

    # Tied model has no lm_head tensor on disk.
    assert loaded.lm_head is None
    _assert_fields_byte_identical(orig, loaded, saved_fields(orig.cfg.tie_embeddings))

    # Forward pass on a small batch matches.
    tokens = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    out_a = orig(tokens)
    out_b = loaded(tokens)
    assert jnp.allclose(out_a, out_b, atol=1e-6)


def test_save_load_round_trip_untied(tmp_path: Path) -> None:
    """An untied model (`tie_embeddings=False`) round-trips with its
    standalone `lm_head` tensor persisted and restored bit-exact."""
    orig = init_model(TINY_UNTIED, key=0)
    assert orig.lm_head is not None
    out_dir = tmp_path / "untied"
    save_model(orig, out_dir)

    # The untied schema includes `lm_head` on disk.
    from safetensors.numpy import load_file

    on_disk = set(load_file(str(out_dir / MODEL_FILE)).keys())
    assert "lm_head" in on_disk
    assert on_disk == set(saved_fields(False))

    loaded, _ = load_model(out_dir)
    assert loaded.lm_head is not None
    assert not loaded.cfg.tie_embeddings
    _assert_fields_byte_identical(orig, loaded, saved_fields(False))

    tokens = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    assert jnp.allclose(orig(tokens), loaded(tokens), atol=1e-6)


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
    loaded, _ = load_model(out_dir)
    assert loaded.cfg.d_model == TINY_VARIANTS["small"].d_model

    _assert_fields_byte_identical(variant, loaded, saved_fields(variant.cfg.tie_embeddings))

    # Forward pass on the loaded variant matches the original at atol=1e-6.
    tokens = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    assert jnp.allclose(variant(tokens), loaded(tokens), atol=1e-6)


def test_save_with_run_config_and_optimizer(tmp_path: Path) -> None:
    """Optional payloads land on disk, the manifest covers them, and the
    full `load_model` path still works with the optional payloads present
    (regression-guards against future code that keys on manifest length).

    `load_model` returns the persisted run block so eval can rebuild the
    checkpoint's own conditioning."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000020"
    fake_opt = {
        "adam_m": np.zeros(8, dtype=np.float32),
        "adam_v": np.ones(8, dtype=np.float32),
    }
    run = {"strategy": "lora", "lora_rank": 4, "conditioning": ["outcome"], "C": 2}
    save_model(
        model,
        out_dir,
        run_config=run,
        optimizer_state=fake_opt,
        training_state={"step": 1234, "rng_b64": "deadbeef"},
    )
    assert (out_dir / OPTIMIZER_FILE).is_file()
    assert (out_dir / "training_state.json").is_file()

    # config.json carries the run block.
    raw = json.loads((out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert raw["run"] == run

    # training_state.json round-trips intact.
    ts = json.loads((out_dir / "training_state.json").read_text(encoding="utf-8"))
    assert ts == {"step": 1234, "rng_b64": "deadbeef"}

    # `load_model` still works with the optional payloads present, and
    # returns the persisted run block (conditioning + C).
    loaded, run_block = load_model(out_dir)
    assert loaded.cfg == model.cfg
    assert run_block == run
    assert jnp.array_equal(loaded.embed_tokens, model.embed_tokens)


def test_eval_reads_checkpoint_conditioning_round_trip(tmp_path: Path) -> None:
    """Chunk 5: eval rebuilds the corpus from the checkpoint's *own*
    conditioning, read back through ``load_model`` →
    ``conditioning_from_run_block``. Save a checkpoint whose run block
    records ``conditioning=["outcome"]``; the helper must recover exactly
    that list so the eval corpus lands at the trained offset C=2 (plan
    §8.1). A checkpoint with no run block falls back to the BOS-only C=1
    layout."""
    from pawn.corpus import conditioning_from_run_block, conditioning_to_C

    model = init_model(TINY_SUPERNET, key=0)

    out_outcome = tmp_path / "outcome_ckpt"
    save_model(
        model, out_outcome,
        run_config={"conditioning": ["outcome"], "C": 2, "lr": 3e-4},
    )
    _, run_block = load_model(out_outcome)
    recovered = conditioning_from_run_block(run_block)
    assert recovered == ["outcome"]
    assert conditioning_to_C(recovered) == 2

    # No run block ⇒ BOS-only layout.
    out_bare = tmp_path / "bare_ckpt"
    save_model(model, out_bare)
    _, bare_block = load_model(out_bare)
    assert bare_block is None
    assert conditioning_from_run_block(bare_block) == []
    assert conditioning_to_C(conditioning_from_run_block(bare_block)) == 1


# ---------------------------------------------------------------------------
# config.json: tie_embeddings / vocab_size / mask_version persistence
# ---------------------------------------------------------------------------


def test_config_persists_format_and_layout_metadata(tmp_path: Path) -> None:
    """config.json persists the checkpoint format version, the layout
    `mask_version` tag, and (inside the `model` block) `tie_embeddings`
    + `vocab_size`."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)
    raw = json.loads((out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert raw["version"] == CHECKPOINT_FORMAT_VERSION
    assert raw["mask_version"] == MASK_VERSION
    assert raw["model"]["tie_embeddings"] is True
    assert raw["model"]["vocab_size"] == model.cfg.vocab_size


def test_load_rejects_wrong_mask_version(tmp_path: Path) -> None:
    """A checkpoint whose `mask_version` doesn't match this build is
    refused — closes the silent layout/RoPE drift the tag exists to
    catch."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def bump_mask_version(raw: dict) -> dict:
        raw["mask_version"] = MASK_VERSION + 7
        return raw

    _tamper_config_and_rehash(out_dir, bump_mask_version)
    with pytest.raises(CheckpointIntegrityError, match="mask_version"):
        load_model(out_dir)


# ---------------------------------------------------------------------------
# Tied↔untied cross-load rejection
# ---------------------------------------------------------------------------


def test_cross_load_untied_checkpoint_into_tied_config_rejected(tmp_path: Path) -> None:
    """A checkpoint saved untied (carries `lm_head`) cannot be loaded
    when config.json claims `tie_embeddings=True`."""
    model = init_model(TINY_UNTIED, key=0)
    out_dir = tmp_path / "untied_ckpt"
    save_model(model, out_dir)

    # Flip the persisted config to tied without dropping the lm_head tensor.
    def make_tied(raw: dict) -> dict:
        raw["model"]["tie_embeddings"] = True
        return raw

    _tamper_config_and_rehash(out_dir, make_tied)
    with pytest.raises(CheckpointIntegrityError, match="tie_embeddings=True"):
        load_model(out_dir)


def test_cross_load_tied_checkpoint_into_untied_config_rejected(tmp_path: Path) -> None:
    """A checkpoint saved tied (no `lm_head`) cannot be loaded when
    config.json claims `tie_embeddings=False`."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "tied_ckpt"
    save_model(model, out_dir)

    def make_untied(raw: dict) -> dict:
        raw["model"]["tie_embeddings"] = False
        return raw

    _tamper_config_and_rehash(out_dir, make_untied)
    with pytest.raises(CheckpointIntegrityError, match="tie_embeddings=False"):
        load_model(out_dir)


# ---------------------------------------------------------------------------
# bf16 first-moment optimizer-state round-trip
# ---------------------------------------------------------------------------


def test_optimizer_bf16_mu_round_trip(tmp_path: Path) -> None:
    """The AdamW first moment (`mu`), stored in bf16 to halve the
    optimizer state's footprint, survives the flatten → safetensors →
    unflatten round-trip bit-exact.

    Mirrors `pawn.trainer.make_optimizer`'s adamw branch (`mu_dtype=
    bfloat16`) and the trainer's `flatten_opt_state`/`unflatten_opt_state`
    serialisation path, but builds the optimizer directly so the test
    doesn't pull in the full run-config machinery."""
    import equinox as eqx

    from pawn.trainer import flatten_opt_state, unflatten_opt_state

    model = init_model(TINY_SUPERNET, key=0)
    params = eqx.filter(model, eqx.is_inexact_array)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=0.0, mu_dtype=jnp.bfloat16),
    )
    opt_state = optimizer.init(params)
    # Take one update step so the moments are non-zero (a zero-fill would
    # round-trip trivially regardless of dtype handling).
    grads = eqx.filter(model, eqx.is_inexact_array)
    _, opt_state = optimizer.update(grads, opt_state, params)

    # The first moment must actually be bf16, else the test isn't pinning
    # the bf16 path at all.
    flat = flatten_opt_state(opt_state)
    mu_keys = [k for k, v in flat.items() if v.dtype == jnp.bfloat16]
    assert mu_keys, "expected at least one bf16 (mu) leaf in the opt_state"

    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir, optimizer_state=flat)

    from safetensors.numpy import load_file

    restored_flat = load_file(str(out_dir / OPTIMIZER_FILE))
    for k in mu_keys:
        assert restored_flat[k].dtype == jnp.bfloat16, (
            f"bf16 leaf {k} lost its dtype across the safetensors round-trip"
        )
        assert np.array_equal(restored_flat[k], flat[k]), (
            f"bf16 leaf {k} changed across the round-trip"
        )

    # And it rebuilds back into a valid opt_state PyTree.
    template = optimizer.init(params)
    rebuilt = unflatten_opt_state(template, restored_flat)
    rebuilt_flat = flatten_opt_state(rebuilt)
    for k in mu_keys:
        assert rebuilt_flat[k].dtype == jnp.bfloat16
        assert np.array_equal(rebuilt_flat[k], flat[k])


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
    """A model.safetensors with one of the saved fields missing is
    rejected. The sentinel is re-hashed so the failure fires on the
    schema check, not the SHA mismatch."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir)

    def drop_embed_tokens(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(tensors)
        del d["embed_tokens"]
        return d

    _tamper_tensors_and_rehash(out_dir, drop_embed_tokens)
    with pytest.raises(CheckpointIntegrityError, match="missing expected tensors"):
        load_model(out_dir)


def test_load_rejects_extra_tensor(tmp_path: Path) -> None:
    """A model.safetensors with a tensor not in the save schema is rejected."""
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

    def wrong_embed_tokens(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(tensors)
        d["embed_tokens"] = np.zeros((64, 1234), dtype=np.float32)
        return d

    _tamper_tensors_and_rehash(out_dir, wrong_embed_tokens)
    with pytest.raises(CheckpointIntegrityError, match="shape|expected"):
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


def test_load_rejects_non_dict_run_block(tmp_path: Path) -> None:
    """A `run` block present but not a dict (e.g. a list) is rejected —
    `load_model` returns it to callers as a dict and a malformed shape
    would surface as a confusing downstream error otherwise."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir, run_config={"conditioning": ["outcome"]})

    def break_run_block(raw: dict) -> dict:
        raw["run"] = ["not", "a", "dict"]
        return raw

    _tamper_config_and_rehash(out_dir, break_run_block)
    with pytest.raises(CheckpointIntegrityError, match="`run` block is not a dict"):
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


def test_save_schema_field_count() -> None:
    """Re-pin the save-schema contract at the checkpoint API layer too —
    a drift in `pawn.model.saved_fields` would already fire its own
    assertion at import; this is belt-and-braces. Tied omits `lm_head`
    (11 fields), untied includes it (12)."""
    assert "lm_head" not in saved_fields(True)
    assert "lm_head" in saved_fields(False)
    assert len(saved_fields(True)) == len(saved_fields(False)) - 1
    assert CHECKPOINT_FORMAT_VERSION == 1
