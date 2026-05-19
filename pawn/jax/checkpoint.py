"""Checkpoint serialization for the JAX/Equinox PAWN model.

A checkpoint is a directory:

    model.safetensors   one fp32 tensor per PAWNModel array field, keyed by
                        the field name (the canonical schema below)
    config.json         {format_version, model_config}
    .complete           SHA-256 hashes of every other file (integrity sentinel)

Writes are atomic: files land in a ``.tmp`` sibling directory, then the whole
directory is renamed into place once the ``.complete`` sentinel is written.
Loads verify the sentinel and every file hash before returning.

Canonical safetensors schema — 16 keys, the PAWNModel array fields, in
declaration order: ``src_embed dst_embed promo_embed pad_embed outcome_embed``
(embeddings); ``attn_norm wq wk wv wo ffn_norm w_gate w_up w_down`` (stacked
transformer layers, leading axis = n_layers); ``final_norm lm_head`` (head).
Linear weights use the JAX ``(in, out)`` convention. The token-decomposition
table is a global constant, not a parameter, and is not stored.

This module is framework-neutral (no torch, no flax) and uses
``safetensors.numpy``. The atomic-write and sentinel helpers mirror
``pawn/checkpoint.py``; the two converge into one module when PyTorch is
removed at the end of the migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file

from pawn.jax.config import ModelConfig
from pawn.jax.model import PAWNModel

CHECKPOINT_FORMAT_VERSION = 1

_MODEL_FILE = "model.safetensors"
_CONFIG_FILE = "config.json"
_SENTINEL_FILE = ".complete"

# Canonical schema: one tensor per PAWNModel array field, keyed by field name.
# Derived from the model class (every field except the static ``cfg``) so the
# schema cannot silently drift from the model definition.
_PARAM_FIELDS: tuple[str, ...] = tuple(
    name for name in PAWNModel.__dataclass_fields__ if name != "cfg"
)


class IncompleteCheckpointError(Exception):
    """Raised when a checkpoint directory is missing its .complete sentinel."""


class CheckpointIntegrityError(Exception):
    """Raised when a checkpoint file's SHA-256 hash does not match .complete."""


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
        )
    )


def _verify_sentinel(directory: Path) -> None:
    """Verify the .complete sentinel exists and every listed hash matches."""
    sentinel = directory / _SENTINEL_FILE
    if not sentinel.exists():
        raise IncompleteCheckpointError(
            f"{directory} is missing its {_SENTINEL_FILE} sentinel — "
            "likely a partial write from an interrupted save."
        )
    listed: dict[str, str] = json.loads(sentinel.read_text())["files"]
    for name, expected in listed.items():
        path = directory / name
        if not path.exists():
            raise CheckpointIntegrityError(
                f"{name} is listed in {_SENTINEL_FILE} but missing from {directory}"
            )
        actual = _sha256_file(path)
        if actual != expected:
            raise CheckpointIntegrityError(
                f"SHA-256 mismatch for {name} in {directory}: "
                f"expected {expected[:16]}…, got {actual[:16]}…"
            )


def save_model(path: str | Path, model: PAWNModel) -> None:
    """Atomically write ``model`` to the checkpoint directory ``path``.

    Files are written to a ``.tmp`` sibling directory and renamed into place
    only after the ``.complete`` sentinel is written, so a crash mid-write
    never leaves a checkpoint that passes verification.
    """
    target = Path(path)
    tmp = target.parent / f"{target.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        tensors = {
            name: np.asarray(getattr(model, name)) for name in _PARAM_FIELDS
        }
        save_file(tensors, str(tmp / _MODEL_FILE))
        config = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": asdict(model.cfg),
        }
        (tmp / _CONFIG_FILE).write_text(json.dumps(config, indent=2))
        _write_sentinel(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if target.exists():
        shutil.rmtree(target)
    os.rename(tmp, target)


def read_model_config(path: str | Path) -> ModelConfig:
    """Return the ``ModelConfig`` stored in checkpoint ``path``.

    Reads only ``config.json`` — no weights, no integrity check — for callers
    that need to size a model before loading it.
    """
    config = json.loads((Path(path) / _CONFIG_FILE).read_text())
    return ModelConfig(**config["model_config"])


def load_model(path: str | Path) -> PAWNModel:
    """Load a ``PAWNModel`` from checkpoint ``path``, verifying integrity.

    Raises ``IncompleteCheckpointError`` if the ``.complete`` sentinel is
    missing, and ``CheckpointIntegrityError`` if any file's hash mismatches.
    """
    directory = Path(path)
    _verify_sentinel(directory)
    cfg = read_model_config(directory)
    tensors = load_file(str(directory / _MODEL_FILE))
    arrays = {name: jnp.asarray(tensors[name]) for name in _PARAM_FIELDS}
    return PAWNModel(cfg=cfg, **arrays)
