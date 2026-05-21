"""Atomic-write ``.complete`` sentinel — shared between the JAX
checkpoint module and the thin PyTorch loader.

Both ``pawn.checkpoint`` (writes + reads JAX-format checkpoints) and
``pawn.torch_loader`` (reads JAX-format checkpoints into a PyTorch
model for external non-JAX users) need to (a) compute file SHA-256s
and (b) verify the ``.complete`` sentinel's file-set + hashes match
the directory contents. The earlier Phase-1 implementation
deliberately duplicated this logic — the loader was meant to be
JAX-independent, and at the time the checkpoint module dragged JAX
through every import.

After the Phase-4 flatten of ``pawn.jax.*`` into ``pawn.*``, the
duplication can be removed cleanly: this module pulls only stdlib +
the two custom exception types, so importing it from
``pawn.torch_loader`` does NOT pull JAX.

Public API:

* ``IncompleteCheckpointError`` — sentinel file missing.
* ``CheckpointIntegrityError`` — sentinel content / file-set / hashes
  do not match the directory state.
* ``sha256_file(path)`` — stream a file's SHA-256 in 1 MiB chunks.
* ``write_sentinel(directory)`` — compute hashes of every non-sentinel
  file under ``directory`` and serialise them to ``directory/.complete``.
* ``verify_sentinel(directory)`` — raise ``IncompleteCheckpointError`` or
  ``CheckpointIntegrityError`` if the sentinel is missing or
  inconsistent.

The sentinel filename + JSON schema are pinned at module level so
both writer and verifier always agree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SENTINEL_FILE = ".complete"


class IncompleteCheckpointError(Exception):
    """Raised when a checkpoint directory has no ``.complete`` sentinel
    (typically a partial write from an interrupted save)."""


class CheckpointIntegrityError(Exception):
    """Raised when the sentinel's file-set or SHA-256 hashes do not
    match the directory contents."""


def sha256_file(path: Path) -> str:
    """Stream a file's SHA-256 hash, 1 MiB chunks. Returns hex digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sentinel(directory: Path) -> None:
    """Compute SHA-256 of every non-sentinel file in ``directory`` and
    serialise the ``{name: hex_digest, ...}`` map to ``directory/.complete``.

    Callers are expected to write the payload files first, then call
    this last so the sentinel's presence reliably implies a complete
    write. The save path in ``pawn.checkpoint`` does this inside a
    ``.tmp`` staging directory that is renamed-into-place atomically;
    a sentinel-present directory is therefore always integrity-checkable.
    """
    listed: dict[str, str] = {
        entry.name: sha256_file(entry)
        for entry in directory.iterdir()
        if entry.is_file() and entry.name != SENTINEL_FILE
    }
    (directory / SENTINEL_FILE).write_text(
        json.dumps({"files": listed}, indent=2),
        encoding="utf-8",
    )


def verify_sentinel(directory: Path) -> None:
    """Raise on missing or inconsistent ``.complete``.

    Specifically:
      * ``IncompleteCheckpointError`` — no sentinel file at all.
      * ``CheckpointIntegrityError`` — sentinel JSON malformed,
        the directory's file-set diverges from the sentinel's
        ``"files"`` keys, or any SHA-256 mismatches.

    Bytes-equal contents under a different filename count as a
    file-set mismatch (the sentinel is keyed by name, not by hash).
    """
    sentinel = directory / SENTINEL_FILE
    if not sentinel.exists():
        raise IncompleteCheckpointError(
            f"{directory} is missing its {SENTINEL_FILE} sentinel — "
            "likely a partial write from an interrupted save."
        )
    data = json.loads(sentinel.read_text(encoding="utf-8"))
    if "files" not in data:
        raise CheckpointIntegrityError(
            f"{sentinel} is malformed — missing the 'files' key"
        )
    listed: dict[str, str] = data["files"]
    present = {
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and entry.name != SENTINEL_FILE
    }
    if present != set(listed):
        raise CheckpointIntegrityError(
            f"{directory} contains files {sorted(present)} but its "
            f"{SENTINEL_FILE} sentinel lists {sorted(listed)}"
        )
    for name, expected in listed.items():
        actual = sha256_file(directory / name)
        if actual != expected:
            raise CheckpointIntegrityError(
                f"SHA-256 mismatch for {name} in {directory}: "
                f"expected {expected[:16]}…, got {actual[:16]}…"
            )
