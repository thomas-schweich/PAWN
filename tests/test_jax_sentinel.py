"""Tests for :mod:`pawn._sentinel`.

The sentinel module is stdlib-only, so these tests don't need a GPU,
JAX, or any heavy dependency. They cover the integrity contract every
checkpoint reader assumes: a directory with a healthy ``.complete``
manifest is byte-for-byte what was written; corruption is detected
loudly; absence is distinguishable from corruption.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pawn._sentinel import (
    SENTINEL_NAME,
    SENTINEL_VERSION,
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    read_sentinel,
    sha256_file,
    verify_sentinel,
    write_sentinel,
)


def _make_payload(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """Lay out a checkpoint directory with the given filename->bytes mapping."""
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    for name, data in files.items():
        target = ckpt / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return ckpt


def test_sha256_file_known_content(tmp_path: Path) -> None:
    """`sha256_file` matches `hashlib.sha256` over the same bytes."""
    payload = b"PAWN test payload" * 1024
    target = tmp_path / "f.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_streams_larger_than_chunk(tmp_path: Path) -> None:
    """A multi-chunk file (>1 MiB) hashes to the same value as the in-memory
    digest. Confirms the streaming read doesn't truncate or double-hash."""
    payload = (b"x" * (1 << 20)) + b"y" * 17
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_write_read_verify_roundtrip(tmp_path: Path) -> None:
    """A freshly-written sentinel reads and verifies without error and
    returns the same manifest from both APIs."""
    ckpt = _make_payload(
        tmp_path,
        {"model.safetensors": b"abc", "config.json": b'{"k": 1}'},
    )
    sentinel_path = write_sentinel(ckpt, ["model.safetensors", "config.json"])
    assert sentinel_path == ckpt / SENTINEL_NAME
    read_manifest = read_sentinel(ckpt)
    verified_manifest = verify_sentinel(ckpt)
    assert read_manifest == verified_manifest
    assert set(verified_manifest.keys()) == {"model.safetensors", "config.json"}
    assert (
        verified_manifest["model.safetensors"] == hashlib.sha256(b"abc").hexdigest()
    )


def test_write_sentinel_normalises_subdir_separators(tmp_path: Path) -> None:
    """Filenames written under subdirectories appear as forward-slash POSIX
    paths in the manifest (cross-platform portability)."""
    ckpt = _make_payload(tmp_path, {"subdir/inner.bin": b"hello"})
    write_sentinel(ckpt, ["subdir/inner.bin"])
    raw = json.loads((ckpt / SENTINEL_NAME).read_text())
    assert raw["version"] == SENTINEL_VERSION
    assert list(raw["files"].keys()) == ["subdir/inner.bin"]


def test_write_sentinel_accepts_absolute_paths(tmp_path: Path) -> None:
    """An absolute path is rewritten to be relative to the checkpoint dir."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    write_sentinel(ckpt, [ckpt / "model.safetensors"])
    manifest = read_sentinel(ckpt)
    assert "model.safetensors" in manifest


def test_write_sentinel_rejects_duplicate_paths(tmp_path: Path) -> None:
    """Listing the same payload twice is an error (would silently lose
    one of the digests otherwise)."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    with pytest.raises(ValueError, match="duplicate"):
        write_sentinel(ckpt, ["model.safetensors", "model.safetensors"])


def test_write_sentinel_rejects_missing_target_dir(tmp_path: Path) -> None:
    """Pointing at a directory that doesn't exist is an error (calling-side
    bug — callers must mkdir the target first)."""
    with pytest.raises(FileNotFoundError):
        write_sentinel(tmp_path / "does_not_exist", ["a.bin"])


def test_write_sentinel_rejects_empty_payload(tmp_path: Path) -> None:
    """An empty payload list would produce a manifest that fails its own
    verify; surface the bug at write time so writer and reader can't
    disagree."""
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    with pytest.raises(ValueError, match="at least one payload file"):
        write_sentinel(ckpt, [])


def test_write_sentinel_rejects_path_traversal(tmp_path: Path) -> None:
    """`..` in a payload name is rejected so a malicious manifest can't be
    used to probe arbitrary files."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    with pytest.raises(CheckpointIntegrityError, match="'..' segments"):
        write_sentinel(ckpt, ["../escaped.bin"])


def test_write_sentinel_rejects_absolute_path_outside_dir(tmp_path: Path) -> None:
    """Absolute path that doesn't live under the checkpoint dir is rejected
    with a sentinel-specific error, not a raw ValueError."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    elsewhere = tmp_path / "other.bin"
    elsewhere.write_bytes(b"data")
    with pytest.raises(CheckpointIntegrityError, match="escapes checkpoint directory"):
        write_sentinel(ckpt, [elsewhere])


def test_verify_missing_sentinel_raises_incomplete(tmp_path: Path) -> None:
    """An empty directory has no sentinel; verify raises Incomplete (not
    Integrity) so the caller can distinguish 'crashed mid-write' from
    'corrupted on disk'."""
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    with pytest.raises(IncompleteCheckpointError):
        verify_sentinel(ckpt)
    with pytest.raises(IncompleteCheckpointError):
        read_sentinel(ckpt)


def test_verify_corrupted_payload_raises(tmp_path: Path) -> None:
    """If a payload byte is changed after the sentinel is written, verify
    catches the SHA mismatch."""
    ckpt = _make_payload(
        tmp_path, {"model.safetensors": b"original content"}
    )
    write_sentinel(ckpt, ["model.safetensors"])
    # Mutate one byte
    (ckpt / "model.safetensors").write_bytes(b"alteredcontent_!")
    with pytest.raises(CheckpointIntegrityError, match="sha256 mismatch"):
        verify_sentinel(ckpt)


def test_verify_missing_payload_raises(tmp_path: Path) -> None:
    """A payload file referenced by the manifest but absent on disk is a
    distinct integrity error from a hash mismatch."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    write_sentinel(ckpt, ["model.safetensors"])
    (ckpt / "model.safetensors").unlink()
    with pytest.raises(CheckpointIntegrityError, match="missing file"):
        verify_sentinel(ckpt)


def test_verify_bad_json_raises_integrity(tmp_path: Path) -> None:
    """A sentinel file present but unparseable is Integrity, not Incomplete
    — the directory looks complete to a presence-check, but its manifest
    is unusable."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    (ckpt / SENTINEL_NAME).write_text("not valid json {")
    with pytest.raises(CheckpointIntegrityError, match="not valid JSON"):
        verify_sentinel(ckpt)


def test_verify_zero_byte_sentinel_raises_integrity(tmp_path: Path) -> None:
    """A zero-byte sentinel (file creation succeeded but write never
    completed) is a likely crash artifact. The error class is Integrity
    — the file is present, just unusable — so callers don't conflate it
    with 'sentinel missing entirely'."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    (ckpt / SENTINEL_NAME).write_bytes(b"")
    with pytest.raises(CheckpointIntegrityError, match="not valid JSON"):
        verify_sentinel(ckpt)


def test_verify_path_traversal_in_manifest_raises(tmp_path: Path) -> None:
    """A `.complete` manifest produced elsewhere with a `..` segment is
    rejected at read time — defense in depth for externally-supplied
    checkpoints (e.g. downloaded from HF)."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    (ckpt / SENTINEL_NAME).write_text(
        json.dumps(
            {
                "version": SENTINEL_VERSION,
                "files": {"../escape.bin": "deadbeef" * 8},
            }
        )
    )
    with pytest.raises(CheckpointIntegrityError, match="unsafe path"):
        verify_sentinel(ckpt)


def test_verify_absolute_path_in_manifest_raises(tmp_path: Path) -> None:
    """An absolute path in a manifest entry is rejected for the same
    defense-in-depth reason."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    (ckpt / SENTINEL_NAME).write_text(
        json.dumps(
            {
                "version": SENTINEL_VERSION,
                "files": {"/etc/passwd": "deadbeef" * 8},
            }
        )
    )
    with pytest.raises(CheckpointIntegrityError, match="unsafe path"):
        verify_sentinel(ckpt)


def test_verify_wrong_version_raises(tmp_path: Path) -> None:
    """A future-versioned manifest a v2 reader doesn't understand is an
    integrity error, not silently ignored."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    (ckpt / SENTINEL_NAME).write_text(
        json.dumps({"version": 9999, "files": {"model.safetensors": "x"}})
    )
    with pytest.raises(CheckpointIntegrityError, match="version"):
        verify_sentinel(ckpt)


def test_verify_empty_files_raises(tmp_path: Path) -> None:
    """A manifest with no files block is malformed — every real checkpoint
    has at least one payload file."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    (ckpt / SENTINEL_NAME).write_text(
        json.dumps({"version": SENTINEL_VERSION, "files": {}})
    )
    with pytest.raises(CheckpointIntegrityError, match="no files entry"):
        verify_sentinel(ckpt)


def test_verify_non_object_manifest_raises(tmp_path: Path) -> None:
    """A JSON array or scalar at the manifest root is malformed."""
    ckpt = _make_payload(tmp_path, {"model.safetensors": b"abc"})
    (ckpt / SENTINEL_NAME).write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(CheckpointIntegrityError, match="not a JSON object"):
        verify_sentinel(ckpt)


def test_verify_returns_manifest_on_success(tmp_path: Path) -> None:
    """The verify return value is the same dict shape callers consume from
    read_sentinel, so caller code can chain verify → use without a second
    read."""
    ckpt = _make_payload(
        tmp_path,
        {"a.bin": b"aaaa", "b.bin": b"bbbb"},
    )
    write_sentinel(ckpt, ["a.bin", "b.bin"])
    manifest = verify_sentinel(ckpt)
    assert manifest["a.bin"] == hashlib.sha256(b"aaaa").hexdigest()
    assert manifest["b.bin"] == hashlib.sha256(b"bbbb").hexdigest()


def test_sentinel_filename_constant() -> None:
    """SENTINEL_NAME is hardcoded to `.complete` per the v1 contract; the
    legacy converter (S10) and the dashboard both grep for that exact
    name."""
    assert SENTINEL_NAME == ".complete"


def test_sentinel_module_has_no_jax_import() -> None:
    """The whole point of `pawn._sentinel` is that it imports without
    dragging in JAX or torch.

    Verifying this in-process is unreliable — pytest typically has
    already imported JAX via another test, and the obvious shortcut
    (inspect the module's globals) misses transitive imports inside
    `pawn/__init__.py`. Run a fresh subprocess and check `sys.modules`
    after the import: that's the actual contract every lightweight
    consumer relies on.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import sys
        import pawn._sentinel  # the only import on the lightweight path
        heavy = [m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib', 'torch', 'equinox', 'optax')]
        if heavy:
            print('LEAKED:' + ','.join(sorted(heavy)))
            raise SystemExit(1)
        print('OK')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"importing pawn._sentinel dragged in heavy modules:\n"
        f"  stdout: {result.stdout.strip()}\n"
        f"  stderr: {result.stderr.strip()}"
    )
    assert result.stdout.strip() == "OK"
