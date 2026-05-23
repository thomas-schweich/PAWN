"""Atomic safetensors save/load for :class:`pawn.model.PAWNModel`.

The on-disk layout for one checkpoint directory ``step_<N>/`` is:

- ``model.safetensors`` — exactly the 16 trainable arrays listed in
  :data:`pawn.model.SAVED_FIELDS`, in declaration order. The
  :class:`pawn.model.PAWNModel` :attr:`decomp_table` buffer is *not*
  saved — it's rebuilt at load time from the engine vocab. RoPE phase
  tables aren't stored at all (they're a function of cfg, recomputed
  inside the forward pass).
- ``config.json`` — ``{"version": 1, "model": {ModelConfig fields},
  "run": {optional user-supplied dict}}``. The ``model`` block is
  enough to reconstruct a fresh ``PAWNModel`` and align the loaded
  tensors; the ``run`` block is reserved for the pydantic run config
  (filled in by the trainer in S6).
- ``optimizer.safetensors`` (optional) — flattened Optax state, written
  by :func:`save_model` when the caller passes an ``optimizer_state``
  dict. Skipped if absent. (Trainer integration arrives in S6.)
- ``training_state.json`` (optional) — ``{"step", "scheduler", "rng"}``
  plus any user-supplied metadata; same opt-in shape as the optimizer.
- ``.complete`` — the SHA-256 manifest from :func:`pawn._sentinel.write_sentinel`.
  Every load verifies it.

Atomic write workflow:

1. Compute ``<target_dir>.tmp`` as the staging path next to the final
   directory. If a stale ``.tmp`` from a prior crashed save is sitting
   there, blow it away with ``shutil.rmtree`` first — the docstring on
   the plan says "the next run sees either a complete `step_<N>` or
   just the orphaned `.tmp` (which gets cleaned up at the next save's
   start)".
2. ``mkdir`` the temp dir, write every payload file into it.
3. Call :func:`pawn._sentinel.write_sentinel` to compute hashes and
   drop the ``.complete`` manifest. The sentinel call lives **inside**
   the temp dir so a process kill mid-rename still leaves a recoverable
   state (either no final dir or a complete one — never a partial
   final dir).
4. ``os.rename(<target_dir>.tmp, <target_dir>)``. This is atomic on
   POSIX filesystems (same-filesystem rename is one inode-table
   operation).

Read workflow:

1. :func:`pawn._sentinel.verify_sentinel` re-hashes every payload and
   confirms the manifest. Raises :class:`IncompleteCheckpointError` if
   ``.complete`` is missing, :class:`CheckpointIntegrityError` on any
   mismatch.
2. Parse ``config.json`` → :class:`ModelConfig`.
3. Load the 16 tensors from ``model.safetensors`` and rebuild
   :class:`PAWNModel` with them + the recomputed decomp table.

:mod:`pawn.model` asserts ``len(SAVED_FIELDS) == 16`` at its own
import. Importing this module triggers ``pawn.model`` first, so the
same drift guard fires before ``pawn.checkpoint`` is fully loaded; a
typo'd or extra field surfaces as an ``AssertionError`` at startup,
not at save/load time.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file as st_load
from safetensors.numpy import save_file as st_save

from pawn._sentinel import (
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    verify_sentinel,
    write_sentinel,
)
from pawn.config import ModelConfig
from pawn.model import (
    SAVED_FIELDS,
    PAWNModel,
    TransformerLayer,
    _build_decomp_table,
)

# pawn.model already asserts `len(SAVED_FIELDS) == 16` at import. Importing
# checkpoint always imports model first, so that one assertion is the
# canonical guard against schema drift; no need to duplicate it here.
# The test `test_save_schema_is_sixteen_fields` re-pins the contract at
# the checkpoint API layer for documentation.

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "MODEL_FILE",
    "CONFIG_FILE",
    "OPTIMIZER_FILE",
    "TRAINING_STATE_FILE",
    "CheckpointIntegrityError",
    "IncompleteCheckpointError",
    "save_model",
    "load_model",
    "load_model_config",
]


CHECKPOINT_FORMAT_VERSION: Final[int] = 1

MODEL_FILE: Final[str] = "model.safetensors"
CONFIG_FILE: Final[str] = "config.json"
OPTIMIZER_FILE: Final[str] = "optimizer.safetensors"
TRAINING_STATE_FILE: Final[str] = "training_state.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_to_tensor_dict(model: PAWNModel) -> dict[str, np.ndarray]:
    """Flatten a :class:`PAWNModel` into a name → numpy-array dict for
    safetensors.

    Keys are the dotted paths from :data:`SAVED_FIELDS`. Arrays are
    materialised via :func:`numpy.asarray` (block until device transfer
    completes).
    """
    tensors: dict[str, np.ndarray] = {}
    for path in SAVED_FIELDS:
        node: Any = model
        for piece in path.split("."):
            node = getattr(node, piece)
        tensors[path] = np.asarray(node)
    return tensors


def _tensor_dict_to_model(
    tensors: dict[str, np.ndarray],
    cfg: ModelConfig,
) -> PAWNModel:
    """Rebuild a :class:`PAWNModel` from its loaded tensors + config.

    Validates that every name in :data:`SAVED_FIELDS` is present in the
    dict, and that every tensor's shape matches the expected
    ``cfg``-derived shape (catches a checkpoint produced by a model of
    a different size before the array is silently broadcast somewhere).
    """
    saved_set = set(SAVED_FIELDS)
    tensor_set = set(tensors.keys())
    missing = sorted(saved_set - tensor_set)
    extras = sorted(tensor_set - saved_set)
    if missing:
        raise CheckpointIntegrityError(
            f"checkpoint missing expected tensors: {missing}"
        )
    if extras:
        raise CheckpointIntegrityError(
            f"checkpoint has unexpected tensors: {extras}"
        )

    expected = _expected_shapes(cfg)
    for name, want in expected.items():
        got = tuple(tensors[name].shape)
        if got != want:
            raise CheckpointIntegrityError(
                f"checkpoint tensor {name!r} has shape {got}, expected {want} "
                f"for the saved ModelConfig"
            )

    def jnp_at(name: str) -> "jax.Array":
        return jnp.asarray(tensors[name])

    layers = TransformerLayer(
        attn_norm_w=jnp_at("layers.attn_norm_w"),
        wq=jnp_at("layers.wq"),
        wk=jnp_at("layers.wk"),
        wv=jnp_at("layers.wv"),
        wo=jnp_at("layers.wo"),
        ffn_norm_w=jnp_at("layers.ffn_norm_w"),
        w_gate=jnp_at("layers.w_gate"),
        w_up=jnp_at("layers.w_up"),
        w_down=jnp_at("layers.w_down"),
    )
    return PAWNModel(
        embed_src=jnp_at("embed_src"),
        embed_dst=jnp_at("embed_dst"),
        embed_promo=jnp_at("embed_promo"),
        embed_pad=jnp_at("embed_pad"),
        embed_outcome=jnp_at("embed_outcome"),
        layers=layers,
        final_norm_w=jnp_at("final_norm_w"),
        lm_head=jnp_at("lm_head"),
        decomp_table=_build_decomp_table(),
        cfg=cfg,
    )


def _expected_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Map every :data:`SAVED_FIELDS` name to the shape implied by ``cfg``.

    Used at load time to refuse a tensor whose shape doesn't match the
    ``ModelConfig`` we just parsed from ``config.json``.
    """
    d = cfg.d_model
    d_ff = cfg.d_ff
    L = cfg.n_layers
    V = cfg.vocab_size
    return {
        "embed_src": (64, d),
        "embed_dst": (64, d),
        "embed_promo": (5, d),
        "embed_pad": (d,),
        "embed_outcome": (cfg.n_outcomes, d),
        "layers.attn_norm_w": (L, d),
        "layers.wq": (L, d, d),
        "layers.wk": (L, d, d),
        "layers.wv": (L, d, d),
        "layers.wo": (L, d, d),
        "layers.ffn_norm_w": (L, d),
        "layers.w_gate": (L, d, d_ff),
        "layers.w_up": (L, d, d_ff),
        "layers.w_down": (L, d_ff, d),
        "final_norm_w": (d,),
        "lm_head": (d, V),
    }


def _cfg_to_dict(cfg: ModelConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def _cfg_from_dict(raw: dict[str, Any]) -> ModelConfig:
    """Rebuild a :class:`ModelConfig` from its JSON-serialised dict.

    Rejects extra keys (caller likely loaded a future-version manifest
    we don't understand) and missing required keys with
    :class:`CheckpointIntegrityError`.
    """
    expected_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    extras = set(raw.keys()) - expected_fields
    if extras:
        raise CheckpointIntegrityError(
            f"config.json `model` block has unexpected keys: {sorted(extras)}"
        )
    try:
        return ModelConfig(**raw)
    except (TypeError, ValueError) as e:
        raise CheckpointIntegrityError(
            f"config.json `model` block is not a valid ModelConfig: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_model(
    model: PAWNModel,
    target_dir: Path | str,
    *,
    run_config: dict[str, Any] | None = None,
    optimizer_state: dict[str, np.ndarray] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a :class:`PAWNModel` checkpoint to ``target_dir``.

    Returns the final ``target_dir`` :class:`Path`.

    The write is staged in ``<target_dir>.tmp``; an orphan ``.tmp`` from
    a prior crashed save is cleaned up first. The final rename is the
    POSIX-atomic step — once it completes, the directory is loadable.

    ``run_config`` / ``optimizer_state`` / ``training_state`` are
    optional; if any are passed, the corresponding files land alongside
    ``model.safetensors`` / ``config.json``. The trainer (S6) and the
    HF-push path (S12) supply them; S2 callers don't have to.

    Raises :class:`FileExistsError` if ``target_dir`` already exists —
    checkpoints are immutable by design, so an accidental overwrite is
    surfaced as a bug instead of silent data loss.
    """
    final = Path(target_dir)
    tmp = final.with_name(final.name + ".tmp")

    if final.exists():
        raise FileExistsError(
            f"checkpoint already exists at {final}; refusing to overwrite. "
            f"Save to a new step path or delete the existing dir explicitly."
        )
    # Discard any orphan from a prior crashed save. Tolerate a non-directory
    # at the .tmp path (regular file, symlink) — surface the cleanup as part
    # of the documented contract.
    if tmp.is_dir():
        shutil.rmtree(tmp)
    elif tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.mkdir(parents=True, exist_ok=False)

    # If any write step below raises, the partially-populated tmp dir
    # would otherwise linger until the next save to the same target path
    # cleans it up. Clean it ourselves on any exception and re-raise, so
    # the failure mode is "no .tmp left behind" regardless of which save
    # target the caller retries with.
    try:
        payload_files: list[str] = [MODEL_FILE, CONFIG_FILE]

        # 1. model.safetensors
        tensors = _model_to_tensor_dict(model)
        st_save(tensors, str(tmp / MODEL_FILE))

        # 2. config.json
        payload = {
            "version": CHECKPOINT_FORMAT_VERSION,
            "model": _cfg_to_dict(model.cfg),
        }
        if run_config is not None:
            payload["run"] = run_config
        (tmp / CONFIG_FILE).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

        # 3. Optional optimizer.safetensors
        if optimizer_state is not None:
            st_save(optimizer_state, str(tmp / OPTIMIZER_FILE))
            payload_files.append(OPTIMIZER_FILE)

        # 4. Optional training_state.json
        if training_state is not None:
            (tmp / TRAINING_STATE_FILE).write_text(
                json.dumps(training_state, indent=2) + "\n", encoding="utf-8"
            )
            payload_files.append(TRAINING_STATE_FILE)

        # 5. .complete sentinel — must be written into tmp BEFORE the rename
        # so the final dir is never seen in a partial state.
        write_sentinel(tmp, payload_files)

        # 6. POSIX-atomic rename. After this point the checkpoint is loadable.
        os.rename(tmp, final)
    except BaseException:
        # Includes KeyboardInterrupt / SystemExit; we want to clean up under
        # all unexpected exits before re-raising to the caller.
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def load_model_config(target_dir: Path | str) -> ModelConfig:
    """Verify the sentinel and parse the :class:`ModelConfig` only.

    Skips the safetensors load + JAX device transfer that
    :func:`load_model` performs, but **does** still re-hash every file
    in the manifest — the dashboard "show me this run's hyperparameters"
    use case still pays for the SHA-256 streaming over the (potentially
    multi-GB) ``model.safetensors``. If a caller knows the directory is
    fresh-from-disk and wants to skip integrity verification, they
    should read ``config.json`` themselves.
    """
    directory = Path(target_dir)
    manifest = verify_sentinel(directory)
    # Defense in depth: verify_sentinel only re-hashes whatever the
    # manifest claims. A malformed `.complete` that omits MODEL_FILE
    # would let `load_model` proceed to read an unverified file. Refuse
    # to load any checkpoint whose manifest doesn't cover both required
    # payloads.
    _require_payloads_in_manifest(manifest, directory)

    cfg_path = directory / CONFIG_FILE
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    version = raw.get("version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointIntegrityError(
            f"config.json version is {version!r}, expected {CHECKPOINT_FORMAT_VERSION}: "
            f"{cfg_path}"
        )
    model_block = raw.get("model")
    if not isinstance(model_block, dict):
        raise CheckpointIntegrityError(
            f"config.json is missing the `model` block: {cfg_path}"
        )
    return _cfg_from_dict(model_block)


def _require_payloads_in_manifest(
    manifest: dict[str, str], directory: Path
) -> None:
    """Reject a sentinel manifest that doesn't cover the required payload files.

    The required set is ``{MODEL_FILE, CONFIG_FILE}``. A manifest that
    omits either would allow :func:`load_model` to read an
    unverified file — exactly what the SHA-256 contract is supposed to
    prevent.
    """
    required = {MODEL_FILE, CONFIG_FILE}
    missing_in_manifest = required - set(manifest.keys())
    if missing_in_manifest:
        raise CheckpointIntegrityError(
            f"checkpoint sentinel at {directory} doesn't cover required "
            f"payloads: {sorted(missing_in_manifest)}"
        )


def load_model(target_dir: Path | str) -> PAWNModel:
    """Verify the sentinel and load a :class:`PAWNModel` from disk.

    Workflow:

    1. :func:`pawn._sentinel.verify_sentinel` re-hashes every payload
       file and confirms the manifest. Raises
       :class:`IncompleteCheckpointError` if ``.complete`` is missing
       (interrupted save) or :class:`CheckpointIntegrityError` on any
       SHA-256 mismatch.
    2. Parse ``config.json`` → :class:`ModelConfig`. The model block's
       keys must match :class:`ModelConfig`'s fields exactly.
    3. Load ``model.safetensors``. Every name in :data:`SAVED_FIELDS`
       must be present, with the shape ``cfg`` implies.
    4. Rebuild the :class:`PAWNModel` (decomp table is rebuilt from
       the engine vocab).
    """
    directory = Path(target_dir)
    cfg = load_model_config(directory)  # verify + parse cfg
    tensors = st_load(str(directory / MODEL_FILE))
    return _tensor_dict_to_model(tensors, cfg)
