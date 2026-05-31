"""Atomic safetensors save/load for :class:`pawn.model.PAWNModel`.

The on-disk layout for one checkpoint directory ``step_<N>/`` is:

- ``model.safetensors`` — the trainable arrays listed in
  :func:`pawn.model.saved_fields` for the config's ``tie_embeddings``,
  in declaration order. A tied model omits ``lm_head`` (logits reuse
  ``embed_tokens.T``); an untied model includes it. The
  :class:`pawn.model.PAWNModel` :attr:`decomp_table` buffer is *not*
  saved — it's rebuilt at load time from the engine vocab. RoPE phase
  tables aren't stored at all (they're a function of cfg, recomputed
  inside the forward pass).
- ``config.json`` — ``{"version": 1, "mask_version": M,
  "model": {ModelConfig fields}, "run": {optional user-supplied dict}}``.
  The ``model`` block (which carries ``tie_embeddings`` + ``vocab_size``)
  is enough to reconstruct a fresh ``PAWNModel`` and align the loaded
  tensors; the ``run`` block carries the run config (the run's
  ``conditioning`` + derived ``C``), returned by :func:`load_model` so
  eval can rebuild the checkpoint's own sequence layout. ``mask_version``
  is the layout/loss-mask contract tag (Chunk 4 owns its value).
- ``optimizer.safetensors`` (optional) — flattened Optax state, written
  by :func:`save_model` when the caller passes an ``optimizer_state``
  dict. Skipped if absent. (Trainer integration arrives in S6.)
- ``training_state.json`` (optional) — ``{"step", "scheduler",
  "rng_key", "numpy_rngs"}`` plus any user-supplied metadata; same opt-in
  shape as the optimizer. ``scheduler`` records the LR-schedule identity +
  peak (the schedule is a pure function of cfg, so its name is enough to
  reconstruct it); ``rng_key`` is the serialised JAX PRNG key and
  ``numpy_rngs`` the data-stream ``numpy.random.Generator`` states, so a
  ``--resume`` is bit-reproducible vs an uninterrupted run (H7). The
  payload is assembled by :func:`pawn.lifecycle.build_training_state`.
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
2. Parse ``config.json`` → :class:`ModelConfig` + run block.
3. Load the per-config tensors from ``model.safetensors`` and rebuild
   :class:`PAWNModel` with them + the recomputed decomp table.

:mod:`pawn.model` asserts ``len(SAVED_FIELDS) == 12`` (the untied
superset) at its own import, and :func:`pawn.model.saved_fields`
derives the per-config schema from it. Importing this module triggers
``pawn.model`` first, so the same drift guard fires before
``pawn.checkpoint`` is fully loaded; a typo'd or extra field surfaces
as an ``AssertionError`` at startup, not at save/load time.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any, Final

import equinox as eqx
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
from pawn.config import MASK_VERSION, ModelConfig
from pawn.model import (
    PAWNModel,
    TransformerLayer,
    _build_decomp_table,
    saved_fields,
)

# pawn.model already asserts the untied superset has 12 fields at import,
# and `saved_fields(tie_embeddings)` derives the per-config schema from it.
# Importing checkpoint always imports model first, so that assertion is the
# canonical guard against schema drift; no need to duplicate it here.
# The test `test_save_schema_field_count` re-pins the contract at the
# checkpoint API layer for documentation.

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "MASK_VERSION",
    "MODEL_FILE",
    "CONFIG_FILE",
    "OPTIMIZER_FILE",
    "TRAINING_STATE_FILE",
    "ADAPTER_RESUME_FILE",
    "CheckpointIntegrityError",
    "IncompleteCheckpointError",
    "save_model",
    "load_model",
    "load_model_config",
    "save_adapter_resume_state",
    "load_adapter_resume_state",
    "find_best_adapter_step",
]


CHECKPOINT_FORMAT_VERSION: Final[int] = 1

# Layout/mask contract version baked into every ``config.json``. Owned by
# :data:`pawn.config.MASK_VERSION` (single source of truth shared with the
# lichess cache key) and re-exported here for the checkpoint API surface.
# It bumps whenever the prefix-assembly / loss-mask / move-position
# convention changes; the load-time assert in ``_verify_and_read_config``
# refuses a checkpoint whose ``mask_version`` doesn't match the builder's.
# (Already listed in ``__all__`` above; the import above binds the name.)

MODEL_FILE: Final[str] = "model.safetensors"
CONFIG_FILE: Final[str] = "config.json"
OPTIMIZER_FILE: Final[str] = "optimizer.safetensors"
TRAINING_STATE_FILE: Final[str] = "training_state.json"
# Resume sidecar for weight-folding adapters (lora / sparse / unfreeze /
# specialized_clm / hybrid). ``model.safetensors`` carries the *folded*
# effective model for downstream eval / publish; this sidecar carries the
# *raw* frozen backbone + the trained adapter PyTree so ``--resume`` can
# re-derive the exact (backbone, adapter) split the warm Adam moments were
# trained against. See :func:`save_adapter_resume_state`.
ADAPTER_RESUME_FILE: Final[str] = "adapter_resume_state.eqx"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_to_tensor_dict(model: PAWNModel) -> dict[str, np.ndarray]:
    """Flatten a :class:`PAWNModel` into a name → numpy-array dict for
    safetensors.

    Keys are the dotted paths from :func:`saved_fields` for the model's
    ``tie_embeddings`` setting (tied models omit ``lm_head``). Arrays are
    materialised via :func:`numpy.asarray` (block until device transfer
    completes).
    """
    tensors: dict[str, np.ndarray] = {}
    for path in saved_fields(model.cfg.tie_embeddings):
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

    Validates that every name in :func:`saved_fields` (for ``cfg``'s
    ``tie_embeddings``) is present in the dict, and that every tensor's
    shape matches the expected ``cfg``-derived shape (catches a checkpoint
    produced by a model of a different size before the array is silently
    broadcast somewhere).

    Tied↔untied cross-load is a hard error: a tied checkpoint omits
    ``lm_head`` while an untied one includes it, so the missing/extra-tensor
    guards below already fire on a mismatch — but we raise an explicit,
    actionable message first so the failure mode is obvious rather than
    surfacing as a bare "missing lm_head".
    """
    expected_fields = saved_fields(cfg.tie_embeddings)
    saved_set = set(expected_fields)
    tensor_set = set(tensors.keys())

    # Tied↔untied cross-load: detect via the lone differing field (lm_head)
    # and raise a targeted error before the generic missing/extra messages.
    has_lm_head = "lm_head" in tensor_set
    if cfg.tie_embeddings and has_lm_head:
        raise CheckpointIntegrityError(
            "checkpoint carries an `lm_head` tensor but the saved ModelConfig "
            "has tie_embeddings=True (tied models reuse embed_tokens.T and "
            "store no lm_head); refusing to load an untied checkpoint into a "
            "tied config"
        )
    if not cfg.tie_embeddings and not has_lm_head:
        raise CheckpointIntegrityError(
            "checkpoint has no `lm_head` tensor but the saved ModelConfig has "
            "tie_embeddings=False (untied models need a standalone lm_head); "
            "refusing to load a tied checkpoint into an untied config"
        )

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
    lm_head = None if cfg.tie_embeddings else jnp_at("lm_head")
    return PAWNModel(
        embed_tokens=jnp_at("embed_tokens"),
        layers=layers,
        final_norm_w=jnp_at("final_norm_w"),
        lm_head=lm_head,
        decomp_table=_build_decomp_table(),
        cfg=cfg,
    )


def _expected_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Map every :func:`saved_fields` name (for ``cfg``'s ``tie_embeddings``)
    to the shape implied by ``cfg``.

    Used at load time to refuse a tensor whose shape doesn't match the
    ``ModelConfig`` we just parsed from ``config.json``. The factored
    embedding tables (``embed_src/dst/promo``) are gone — the uniform
    ``embed_tokens[V, d]`` table replaces them, so the 64/64/5 literals
    no longer appear here. ``lm_head`` is only expected for untied configs.
    """
    d = cfg.d_model
    d_ff = cfg.d_ff
    L = cfg.n_layers
    V = cfg.vocab_size
    shapes: dict[str, tuple[int, ...]] = {
        "embed_tokens": (V, d),
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
    }
    if not cfg.tie_embeddings:
        shapes["lm_head"] = (d, V)
    return shapes


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

        # 2. config.json. ``version`` is the checkpoint *format* version;
        # ``mask_version`` is the layout/prefix/loss-mask contract version
        # (Chunk 4 owns its value). The run block carries the run's
        # ``conditioning`` + derived ``C`` (filled by the trainer / adapter
        # driver) so eval can rebuild the exact sequence layout the
        # checkpoint was trained under (plan §8.1: eval reads the
        # checkpoint's own conditioning). ``tie_embeddings`` + ``vocab_size``
        # are already inside the ``model`` block via ``_cfg_to_dict``.
        payload: dict[str, Any] = {
            "version": CHECKPOINT_FORMAT_VERSION,
            "mask_version": MASK_VERSION,
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


def _verify_and_read_config(target_dir: Path | str) -> dict[str, Any]:
    """Verify the sentinel + manifest coverage and return the parsed
    ``config.json`` dict.

    Re-hashes every file in the manifest and refuses a manifest that
    doesn't cover both required payloads, then validates the top-level
    ``version`` / ``mask_version`` tags. The caller pulls the ``model`` /
    ``run`` blocks out of the returned dict.
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
    # ``mask_version`` is optional for forward-compat with older checkpoints
    # written before the tag existed (they predate the conditioning prefix,
    # so the layout is the implicit v0 contract). If present it must match.
    mask_version = raw.get("mask_version", MASK_VERSION)
    if mask_version != MASK_VERSION:
        raise CheckpointIntegrityError(
            f"config.json mask_version is {mask_version!r}, expected "
            f"{MASK_VERSION}: {cfg_path}"
        )
    return raw


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
    raw = _verify_and_read_config(target_dir)
    model_block = raw.get("model")
    if not isinstance(model_block, dict):
        raise CheckpointIntegrityError(
            f"config.json is missing the `model` block: "
            f"{Path(target_dir) / CONFIG_FILE}"
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


def load_model(
    target_dir: Path | str,
) -> tuple[PAWNModel, dict[str, Any] | None]:
    """Verify the sentinel and load a :class:`PAWNModel` from disk.

    Returns ``(model, run_block)`` — ``run_block`` is the persisted run
    config dict (``config.json``'s ``run`` block) or ``None`` if the
    checkpoint was written without one. Eval/generation read the run
    block to rebuild the exact sequence layout (conditioning + derived
    ``C``) the checkpoint was trained under, rather than assuming a
    hardcoded default (plan §8.1: eval reads the checkpoint's own
    conditioning).

    Workflow:

    1. :func:`pawn._sentinel.verify_sentinel` re-hashes every payload
       file and confirms the manifest. Raises
       :class:`IncompleteCheckpointError` if ``.complete`` is missing
       (interrupted save) or :class:`CheckpointIntegrityError` on any
       SHA-256 mismatch.
    2. Parse ``config.json`` → :class:`ModelConfig` + run block. The
       model block's keys must match :class:`ModelConfig`'s fields
       exactly; the top-level ``version`` / ``mask_version`` tags must
       match this build.
    3. Load ``model.safetensors``. Every name in
       :func:`pawn.model.saved_fields` (for the config's
       ``tie_embeddings``) must be present, with the shape ``cfg``
       implies. A tied↔untied mismatch is a hard error.
    4. Rebuild the :class:`PAWNModel` (decomp table is rebuilt from
       the engine vocab).
    """
    directory = Path(target_dir)
    raw = _verify_and_read_config(directory)  # verify sentinel + tags
    model_block = raw.get("model")
    if not isinstance(model_block, dict):
        raise CheckpointIntegrityError(
            f"config.json is missing the `model` block: {directory / CONFIG_FILE}"
        )
    cfg = _cfg_from_dict(model_block)
    run_block = raw.get("run")
    if run_block is not None and not isinstance(run_block, dict):
        raise CheckpointIntegrityError(
            f"config.json `run` block is not a dict: {directory / CONFIG_FILE}"
        )
    tensors = st_load(str(directory / MODEL_FILE))
    model = _tensor_dict_to_model(tensors, cfg)
    return model, run_block


def save_adapter_resume_state(
    backbone: PAWNModel, adapter: Any, ckpt_dir: Path | str
) -> Path:
    """Serialise the raw ``(backbone, adapter)`` PyTree into a resume sidecar.

    The published ``model.safetensors`` for a weight-folding adapter holds
    the *folded* effective model (``apply_fn(backbone, adapter)``) so
    downstream eval treats it as an ordinary checkpoint. That fold is one-way
    (and, for sparse, the trained mask isn't recoverable from it at all), so a
    faithful ``--resume`` can't reconstruct the pre-fold split from
    ``model.safetensors`` alone. This sidecar persists the *raw* frozen
    backbone and the trained adapter PyTree verbatim via
    :func:`equinox.tree_serialise_leaves`, so the resume path can restore the
    identical (backbone, adapter) the warm Adam moments were trained against —
    no cold-started params under a warm optimiser state (H3).

    The frozen backbone is constant across the run; persisting it per
    checkpoint trades a little disk for a correctness guarantee that holds
    even when the original ``--checkpoint`` source is no longer reachable.

    Returns the written sidecar :class:`Path`. Like the typed bottleneck / FiLM
    sidecars, this lands next to the finalised ``model.safetensors`` *after*
    :func:`save_model`'s atomic rename, so it is not part of the ``.complete``
    integrity manifest — the resume path predicates on its presence and falls
    back to a cold opt-state when it's absent.
    """
    out = Path(ckpt_dir) / ADAPTER_RESUME_FILE
    eqx.tree_serialise_leaves(out, (backbone, adapter))
    return out


def load_adapter_resume_state(
    backbone_like: PAWNModel, adapter_like: Any, ckpt_dir: Path | str
) -> tuple[PAWNModel, Any]:
    """Restore the raw ``(backbone, adapter)`` PyTree from the resume sidecar.

    ``backbone_like`` / ``adapter_like`` are freshly-built templates of the
    exact same PyTree structure as the saved pair — typically the loaded
    (folded) backbone and the freshly cold-initialised adapter. They provide
    the treedef + leaf shapes/dtypes; :func:`equinox.tree_deserialise_leaves`
    overwrites every array leaf with the saved value, so the returned pair is
    the raw frozen backbone and the trained adapter (including non-trainable
    leaves like sparse masks).

    Raises :class:`FileNotFoundError` if the sidecar isn't present — callers
    that need an "is this a resumable folded-adapter checkpoint" probe should
    test ``(ckpt_dir / ADAPTER_RESUME_FILE).is_file()`` first.
    """
    path = Path(ckpt_dir) / ADAPTER_RESUME_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"no {ADAPTER_RESUME_FILE} in {ckpt_dir} — this checkpoint was "
            "written without the weight-folding adapter resume sidecar"
        )
    backbone, adapter = eqx.tree_deserialise_leaves(
        path, (backbone_like, adapter_like)
    )
    return backbone, adapter


def resolve_checkpoint_source(source: str) -> Path:
    """Resolve a checkpoint identifier to a local directory.

    Local paths that exist are returned as-is. Anything else is treated
    as a HuggingFace repo ID and fetched via
    :func:`huggingface_hub.snapshot_download`. The result is a directory
    suitable for :func:`load_model` / :func:`load_model_config`.

    v2 checkpoints only. v1 PyTorch artifacts (the original
    ``thomas-schweich/pawn-{small,base,large}`` repos) are no longer
    loadable in v2 — the legacy converter was removed in the H.2
    housekeeping commit. To use v1 checkpoints, check out the
    ``v1.0.0`` git tag.
    """
    p = Path(source)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            f"checkpoint source {source!r} is not a local directory and "
            f"huggingface_hub is not installed — pip install huggingface_hub"
        ) from e
    local = snapshot_download(repo_id=source, repo_type="model")
    return Path(local)


def find_best_adapter_step(
    metrics_path: Path | str, *, metric: str = "val_loss"
) -> int | None:
    """Return the step with the lowest ``metric`` in a run's ``metrics.jsonl``.

    The v2 owner of v1's best-checkpoint selection (``find_best_adapter_step``).
    Adapter validation writes one ``type=val`` record per eval step carrying
    ``val_loss`` (plus the richer ``val_top1`` / ``val_top5`` /
    ``val_illegal_pred_rate``); this scans those records and returns the step
    that minimises ``metric``.

    Returns ``None`` when the file is absent or carries no ``type=val``
    record with a finite ``metric`` value (e.g. a run with validation
    disabled). Ties resolve to the *earliest* step (the first to reach the
    best loss), matching v1's strict ``<`` best-update.
    """
    path = Path(metrics_path)
    if not path.is_file():
        return None
    best_step: int | None = None
    best_val = float("inf")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "val":
                continue
            raw = rec.get(metric)
            step = rec.get("step")
            if raw is None or step is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            if val < best_val:
                best_val = val
                best_step = int(step)
    return best_step
