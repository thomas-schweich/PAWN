"""Pin the safety invariants of the datagen_with_hf_sync orchestrator.

The single most important invariant: **no zero-byte file is ever
committed to HF**. The previous orchestrator's "upload then truncate"
pattern silently destroyed remote data when a follow-up sync pushed
the truncated zero-byte local file over the populated remote (see
ANALYSIS.md A4). The new orchestrator's `_upload_folder_batch` has
both a watcher-level filter (zero-byte local files are excluded from
the upload list) and an explicit at-call-time guard inside
`_upload_folder_batch`. These tests pin both.

Round-2 review change: `_upload_folder_batch` now drives
`HfApi.create_commit` with explicit `CommitOperationAdd` objects
instead of `upload_folder(..., allow_patterns=files)`. The new shape
closes a defense-in-depth gap (allow_patterns is a glob, not a literal
list) and avoids `upload_folder`'s O(total_shards) folder walk on every
commit. The tests below assert on the new contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module() -> Any:
    """Import scripts/datagen_with_hf_sync.py as a module. The file name
    has hyphens so a normal `import` won't work; use the file-based loader."""
    path = SCRIPTS_DIR / "datagen_with_hf_sync.py"
    spec = importlib.util.spec_from_file_location("datagen_hf_sync_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeApi:
    """Captures create_commit invocations without touching the network."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_commit(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def test_upload_folder_batch_rejects_zero_byte_file(tmp_path: Path) -> None:
    """The at-call-time guard refuses to commit any zero-byte file —
    the regression we want pinned. If this test ever fails it means
    `_upload_folder_batch` would happily push a 0-byte file to HF and
    silently overwrite remote data."""
    mod = _load_module()
    (tmp_path / "good.parquet").write_bytes(b"some data")
    (tmp_path / "empty.parquet").write_bytes(b"")
    api = _FakeApi()
    with pytest.raises(RuntimeError, match="zero bytes"):
        mod._upload_folder_batch(
            api,
            repo_id="org/name",
            local_dir=tmp_path,
            path_in_repo="tier0",
            files=["good.parquet", "empty.parquet"],
            commit_message="test",
        )
    # The guard fires BEFORE the commit, so the fake API records no calls.
    assert api.calls == []


def test_upload_folder_batch_rejects_missing_file(tmp_path: Path) -> None:
    """Same defense-in-depth check, but for a file that doesn't exist
    locally — would otherwise crash inside huggingface_hub with a less
    clear error."""
    mod = _load_module()
    api = _FakeApi()
    with pytest.raises(RuntimeError, match="stat failed"):
        mod._upload_folder_batch(
            api,
            repo_id="org/name",
            local_dir=tmp_path,
            path_in_repo="tier0",
            files=["nonexistent.parquet"],
            commit_message="test",
        )


def test_upload_folder_batch_uploads_non_empty_files(tmp_path: Path) -> None:
    """Happy path: non-empty files reach `create_commit` with one
    explicit `CommitOperationAdd` per file. Each op names the file
    literally — no glob — so a `*` in a future filename can't
    over-match into adjacent files."""
    mod = _load_module()
    (tmp_path / "a.parquet").write_bytes(b"a" * 16)
    (tmp_path / "b.parquet").write_bytes(b"b" * 32)
    api = _FakeApi()
    mod._upload_folder_batch(
        api,
        repo_id="org/name",
        local_dir=tmp_path,
        path_in_repo="tier_X",
        files=["a.parquet", "b.parquet"],
        commit_message="test",
    )
    assert len(api.calls) == 1
    call = api.calls[0]
    assert call["repo_id"] == "org/name"
    assert call["repo_type"] == "dataset"
    ops = call["operations"]
    assert len(ops) == 2
    # Each op is a CommitOperationAdd carrying the literal repo path
    # (`tier_X/<name>`) — no glob, no allow_patterns.
    op_paths = {op.path_in_repo for op in ops}
    assert op_paths == {"tier_X/a.parquet", "tier_X/b.parquet"}
    op_locals = {op.path_or_fileobj for op in ops}
    assert op_locals == {str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")}


def test_scan_and_upload_excludes_zero_byte_placeholders(tmp_path: Path) -> None:
    """End-to-end: a tier dir containing a primed zero-byte placeholder
    next to a real shard. The watcher's scan must NOT include the
    zero-byte file in the upload batch — that's the property that
    keeps `--prune-local` from cratering remote data after the next
    cycle."""
    mod = _load_module()
    # Construct a tier layout with a mix of files.
    tier_dir = tmp_path / "nodes_0001"
    tier_dir.mkdir()
    (tier_dir / "shard-s000000-r000010.parquet").write_bytes(b"")  # primed placeholder
    (tier_dir / "shard-s000001-r000010.parquet").write_bytes(b"x" * 100)  # real shard
    (tier_dir / "_tier_state.json").write_text('{"foo": "bar"}')
    # Ignore: stray .tmp orphan.
    (tier_dir / "shard-s000002.parquet.tmp").write_bytes(b"orphan")

    tiers = [mod.TierLayout(name="nodes_0001", local_dir=tier_dir)]
    api = _FakeApi()
    uploaded: set[str] = set()
    upload_count = [0]
    mod._scan_and_upload(api, "org/name", tiers, uploaded, upload_count, prune_local=False)

    # Exactly one commit (shards+state batch); no separate manifest batch
    # since no `_manifest.json` exists yet.
    assert len(api.calls) == 1
    ops = api.calls[0]["operations"]
    repo_paths = {op.path_in_repo for op in ops}
    # Real shard + tier_state, NOT the zero-byte placeholder, NOT the .tmp.
    assert repo_paths == {
        "nodes_0001/shard-s000001-r000010.parquet",
        "nodes_0001/_tier_state.json",
    }
    # Both committed files made it into the in-memory uploaded set.
    assert uploaded == {
        "nodes_0001/shard-s000001-r000010.parquet",
        "nodes_0001/_tier_state.json",
    }
