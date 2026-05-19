"""Checkpoint serialization for the JAX/Equinox PAWN model.

A checkpoint is a directory:

    model.safetensors   one fp32 tensor per PAWNModel array field, keyed by
                        the field name (the canonical schema below)
    config.json         {format_version, model_config}
    .complete           integrity sentinel — a JSON object
                        {format_version, files: {filename: sha256-hex}}

Writes are crash-safe: files land in a ``.tmp`` sibling directory; once the
``.complete`` sentinel is written, any existing checkpoint is renamed aside to
a ``.bak`` sibling, the new directory is renamed into place, and only then is
the old one removed — so an interrupted overwrite never destroys the previous
checkpoint. Loads verify the sentinel and every file hash before returning.

``save_model`` assumes a single writer per ``path``; concurrent saves to the
same path are not supported (the trainer uses per-step directories).

Canonical safetensors schema — 16 keys, the PAWNModel array fields, in
declaration order: ``src_embed dst_embed promo_embed pad_embed outcome_embed``
(embeddings); ``attn_norm wq wk wv wo ffn_norm w_gate w_up w_down`` (stacked
transformer layers, leading axis = n_layers); ``final_norm lm_head`` (head).
Linear weights use the JAX ``(in, out)`` convention. Tensors are stored fp32
(``save_model`` casts), matching the fp32-master-weight precision design. The
token-decomposition table is a global constant, not a parameter, and is not
stored.

This module is framework-neutral (no torch, no flax) and uses
``safetensors.numpy``. The atomic-write and sentinel helpers mirror
``pawn/checkpoint.py``; the two converge into one module when PyTorch is
removed at the end of the migration.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file

from pawn.jax.config import ModelConfig
from pawn.jax.model import PAWNModel

CHECKPOINT_FORMAT_VERSION = 1

_MODEL_FILE = "model.safetensors"
_CONFIG_FILE = "config.json"
_SENTINEL_FILE = ".complete"

# Canonical schema: one tensor per PAWNModel array field, keyed by field name,
# in declaration order. Derived from the model class — every non-static field,
# i.e. excluding ``cfg`` and any other ``eqx.field(static=True)``. The length
# assertion fails loudly at import if the model gains or loses an array field,
# so the schema cannot silently drift from the model definition.
_PARAM_FIELDS: tuple[str, ...] = tuple(
    f.name
    for f in dataclasses.fields(PAWNModel)
    if not f.metadata.get("static", False)
)
assert len(_PARAM_FIELDS) == 16, (
    f"expected 16 PAWNModel array fields, got {len(_PARAM_FIELDS)}: {_PARAM_FIELDS}"
)


class IncompleteCheckpointError(Exception):
    """Raised when a checkpoint directory is missing its .complete sentinel."""


class CheckpointIntegrityError(Exception):
    """Raised when a checkpoint's files do not match its .complete sentinel."""


class UnsupportedCheckpointVersionError(Exception):
    """Raised when a checkpoint's format_version is not understood."""


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sentinel(directory: Path) -> None:
    """Write the .complete sentinel with hashes of every other file."""
    hashes = {
        entry.name: _sha256_file(entry)
        for entry in sorted(directory.iterdir())
        if entry.is_file() and entry.name != _SENTINEL_FILE
    }
    (directory / _SENTINEL_FILE).write_text(
        json.dumps(
            {"format_version": CHECKPOINT_FORMAT_VERSION, "files": hashes},
            indent=2,
        ),
        encoding="utf-8",
    )


def _verify_sentinel(directory: Path) -> None:
    """Verify the .complete sentinel exists and matches the directory exactly.

    Every file listed in the sentinel must be present with a matching hash,
    and the directory must contain no files beyond those listed (plus the
    sentinel itself) — an injected file would otherwise load unverified.
    """
    sentinel = directory / _SENTINEL_FILE
    if not sentinel.exists():
        raise IncompleteCheckpointError(
            f"{directory} is missing its {_SENTINEL_FILE} sentinel — "
            "likely a partial write from an interrupted save."
        )
    sentinel_data = json.loads(sentinel.read_text(encoding="utf-8"))
    if "files" not in sentinel_data:
        raise CheckpointIntegrityError(
            f"{sentinel} is malformed — missing the 'files' key"
        )
    listed: dict[str, str] = sentinel_data["files"]
    present = {
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and entry.name != _SENTINEL_FILE
    }
    if present != set(listed):
        raise CheckpointIntegrityError(
            f"{directory} contains files {sorted(present)} but its "
            f"{_SENTINEL_FILE} sentinel lists {sorted(listed)}"
        )
    for name, expected in listed.items():
        actual = _sha256_file(directory / name)
        if actual != expected:
            raise CheckpointIntegrityError(
                f"SHA-256 mismatch for {name} in {directory}: "
                f"expected {expected[:16]}…, got {actual[:16]}…"
            )


def save_model(path: str | Path, model: PAWNModel) -> None:
    """Atomically write ``model`` to the checkpoint directory ``path``.

    Tensors are cast to fp32. Files are written to a ``.tmp`` sibling; once
    complete, any existing checkpoint is renamed to a ``.bak`` sibling, the
    new directory is renamed into place, and the old one is then removed — so
    an interrupted overwrite always leaves a recoverable checkpoint on disk.
    Single-writer per ``path`` (see module docstring).
    """
    target = Path(path)
    tmp = target.parent / f"{target.name}.tmp"
    backup = target.parent / f"{target.name}.bak"

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        # One batched device->host transfer, then cast to the fp32 on-disk
        # contract (matching the fp32-master-weight precision design).
        host = jax.device_get(
            {name: getattr(model, name) for name in _PARAM_FIELDS}
        )
        tensors: dict[str, np.ndarray] = {
            name: np.asarray(arr, dtype=np.float32) for name, arr in host.items()
        }
        save_file(tensors, tmp / _MODEL_FILE)
        config = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": asdict(model.cfg),
        }
        (tmp / _CONFIG_FILE).write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        _write_sentinel(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Rename-aside overwrite: the previous checkpoint stays on disk (as
    # ``.bak``) until the new directory is in place.
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.rename(target, backup)
    try:
        os.rename(tmp, target)
    except BaseException:
        if backup.exists():  # roll back: restore the previous checkpoint
            os.rename(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def read_model_config(path: str | Path) -> ModelConfig:
    """Return the ``ModelConfig`` stored in checkpoint ``path``.

    Reads only ``config.json`` — no weights, no hash verification — for
    callers that need to size a model before loading it. Do not call this on
    a ``.tmp`` / partially-written directory; the result would be a config
    built from an unverified file.

    Raises ``UnsupportedCheckpointVersionError`` if the checkpoint's
    ``format_version`` is not understood by this build.
    """
    config = json.loads(
        (Path(path) / _CONFIG_FILE).read_text(encoding="utf-8")
    )
    version = config.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise UnsupportedCheckpointVersionError(
            f"checkpoint {path} has format_version {version!r}; this build "
            f"reads version {CHECKPOINT_FORMAT_VERSION}"
        )
    return ModelConfig(**config["model_config"])


def load_model(path: str | Path) -> PAWNModel:
    """Load a ``PAWNModel`` from checkpoint ``path``, verifying integrity.

    Raises ``IncompleteCheckpointError`` if the ``.complete`` sentinel is
    missing, ``CheckpointIntegrityError`` if any file's hash mismatches or the
    tensor key set does not match the schema, and
    ``UnsupportedCheckpointVersionError`` on an unknown format version.
    """
    directory = Path(path)
    _verify_sentinel(directory)
    cfg = read_model_config(directory)
    tensors = load_file(directory / _MODEL_FILE)
    if set(tensors) != set(_PARAM_FIELDS):
        missing = sorted(set(_PARAM_FIELDS) - set(tensors))
        extra = sorted(set(tensors) - set(_PARAM_FIELDS))
        raise CheckpointIntegrityError(
            f"{directory} model.safetensors key set does not match the schema "
            f"— missing {missing}, unexpected {extra}"
        )
    arrays = {name: jnp.asarray(tensors[name]) for name in _PARAM_FIELDS}
    return PAWNModel(cfg=cfg, **arrays)
