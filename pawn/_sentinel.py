"""Stdlib-only ``.complete`` sentinel helpers for atomic checkpoint writes.

Every checkpoint directory written by :mod:`pawn.checkpoint` carries a
``.complete`` manifest listing the SHA-256 digest of every payload file.
The save path atomically renames a fully-populated temp directory into the
final ``step_<N>`` slot only after the sentinel has been written; readers
verify the manifest on every load.

This module deliberately depends on **only** :mod:`hashlib`, :mod:`json`
and :mod:`pathlib` so it can be imported in lightweight contexts — sweep
drivers, dashboards, the legacy converter — without dragging in JAX. The
exceptions raised here are the public contract every checkpoint consumer
relies on:

- :class:`IncompleteCheckpointError` — the ``.complete`` sentinel is
  absent. The checkpoint write was interrupted, or the directory was
  never produced by :func:`write_sentinel`.
- :class:`CheckpointIntegrityError` — the sentinel is present but at
  least one payload file is missing, modified, or hashes differently
  than the manifest claims.

The two error classes are distinct because a missing sentinel is an
expected outcome of a crash mid-save (caller may want to delete and
retry), while a hash mismatch is a corruption signal (caller should
refuse to load).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "SENTINEL_NAME",
    "SENTINEL_VERSION",
    "IncompleteCheckpointError",
    "CheckpointIntegrityError",
    "sha256_file",
    "write_sentinel",
    "verify_sentinel",
    "read_sentinel",
]


SENTINEL_NAME = ".complete"
SENTINEL_VERSION = 1
_CHUNK = 1 << 20  # 1 MiB streamed hashing block


class IncompleteCheckpointError(Exception):
    """The checkpoint directory has no ``.complete`` sentinel.

    A save was interrupted before the sentinel was written, or the
    directory was never produced by :func:`write_sentinel`. The caller
    typically discards the directory.
    """


class CheckpointIntegrityError(Exception):
    """The sentinel is present but the on-disk payload doesn't match it.

    Either a payload file referenced by the manifest is missing or its
    SHA-256 doesn't match the recorded digest. The manifest itself may
    also be malformed (not JSON, missing the ``files`` block, etc.).
    """


def sha256_file(path: Path | str) -> str:
    """Return the lowercase-hex SHA-256 of ``path``.

    Streams the file in 1 MiB chunks so checkpoints with multi-GB
    safetensors blobs don't hold the whole payload in memory.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_sentinel(
    directory: Path | str,
    payload_files: Iterable[str | Path],
) -> Path:
    """Hash every ``payload_files`` entry and write the ``.complete`` manifest.

    The manifest is sorted by filename so the file is bit-identical
    given the same inputs (helps content-addressed caching downstream).
    Filenames are recorded **relative to** ``directory`` even if the
    caller passes absolute paths — relative storage makes the manifest
    portable across moved/renamed checkpoint directories. Paths that
    escape ``directory`` (``..`` segments or absolute paths outside
    the dir) are rejected so a manifest can't be used to probe or hash
    arbitrary files on a future reader's filesystem.

    Empty ``payload_files`` is an error: a manifest with no files would
    fail its own :func:`verify_sentinel` (which rejects an empty files
    block), so we surface the bug at write time instead.

    Returns the sentinel path. The caller is responsible for placing
    this inside the temp dir *before* the atomic rename: if the rename
    happens first the directory looks complete while the sentinel is
    being written.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"sentinel target directory does not exist: {directory}")

    digests: dict[str, str] = {}
    for name in payload_files:
        rel = Path(name)
        if rel.is_absolute():
            try:
                rel = rel.relative_to(directory.resolve())
            except ValueError as e:
                raise CheckpointIntegrityError(
                    f"payload path escapes checkpoint directory: {name!r} "
                    f"is not under {directory}"
                ) from e
        if ".." in rel.parts:
            raise CheckpointIntegrityError(
                f"payload path may not contain '..' segments: {name!r}"
            )
        # Stash with forward-slash separators for cross-platform manifest portability.
        key = rel.as_posix()
        if key in digests:
            raise ValueError(f"duplicate payload file in manifest: {key}")
        digests[key] = sha256_file(directory / rel)

    if not digests:
        raise ValueError(
            "write_sentinel requires at least one payload file "
            "(an empty manifest would fail its own verify_sentinel)"
        )

    sentinel = directory / SENTINEL_NAME
    payload = {
        "version": SENTINEL_VERSION,
        "files": {k: digests[k] for k in sorted(digests)},
    }
    sentinel.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sentinel


def read_sentinel(directory: Path | str) -> dict[str, str]:
    """Return the ``{filename: sha256}`` manifest without re-verifying.

    Cheaper than :func:`verify_sentinel` when the caller already trusts
    the directory's integrity (e.g. a content-hashed cache lookup that
    happens after a successful verify). Still raises if the sentinel is
    missing or malformed.
    """
    directory = Path(directory)
    sentinel = directory / SENTINEL_NAME
    if not sentinel.is_file():
        raise IncompleteCheckpointError(f"missing sentinel: {sentinel}")
    try:
        raw = sentinel.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise CheckpointIntegrityError(
            f"sentinel is not valid UTF-8: {sentinel}: {e}"
        ) from e
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CheckpointIntegrityError(
            f"sentinel is not valid JSON: {sentinel}: {e}"
        ) from e
    if not isinstance(manifest, dict):
        raise CheckpointIntegrityError(
            f"sentinel manifest is not a JSON object: {sentinel}"
        )
    version = manifest.get("version")
    if version != SENTINEL_VERSION:
        raise CheckpointIntegrityError(
            f"unsupported sentinel version {version!r} (expected {SENTINEL_VERSION}): {sentinel}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise CheckpointIntegrityError(
            f"sentinel has no files entry: {sentinel}"
        )
    for name, digest in files.items():
        # JSON keys are always strings, but the value side is not guaranteed —
        # explicitly reject non-string digests (e.g. an int slipped into the
        # manifest by a bad writer).
        if not isinstance(name, str) or not isinstance(digest, str):
            raise CheckpointIntegrityError(
                f"sentinel files entry has non-string key/value: {sentinel}"
            )
        # Defense in depth: reject path-traversal in manifest names so an
        # externally-supplied .complete (e.g. a downloaded checkpoint) can't
        # cause verify_sentinel to hash files outside the directory.
        rel_path = Path(name)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise CheckpointIntegrityError(
                f"sentinel files entry has unsafe path: {name!r}"
            )
    return dict(files)


def verify_sentinel(directory: Path | str) -> dict[str, str]:
    """Re-hash every file in the manifest and confirm the digests match.

    On success returns the verified ``{filename: sha256}`` mapping (the
    same shape :func:`read_sentinel` returns). On failure raises
    :class:`IncompleteCheckpointError` (no sentinel) or
    :class:`CheckpointIntegrityError` (any mismatch / missing file /
    malformed manifest).
    """
    directory = Path(directory)
    manifest = read_sentinel(directory)
    for name, expected in manifest.items():
        full = directory / name
        if not full.is_file():
            raise CheckpointIntegrityError(
                f"sentinel references missing file: {full}"
            )
        actual = sha256_file(full)
        if actual != expected:
            raise CheckpointIntegrityError(
                f"sha256 mismatch for {full}: expected {expected}, got {actual}"
            )
    return manifest
