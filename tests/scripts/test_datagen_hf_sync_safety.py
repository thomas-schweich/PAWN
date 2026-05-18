"""Pin the safety invariants of the datagen_with_hf_sync orchestrator.

The single most important invariant: **no zero-byte file is ever
committed to HF**. The previous orchestrator's "upload then truncate"
pattern silently destroyed remote data when a follow-up sync pushed
the truncated zero-byte local file over the populated remote. The new
orchestrator's `_upload_folder_batch` has
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

import httpx
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


class _FlakyApi(_FakeApi):
    """Raises queued errors before recording a successful commit."""

    def __init__(self, errors: list[BaseException]) -> None:
        super().__init__()
        self.errors = errors

    def create_commit(self, **kwargs: Any) -> None:
        if self.errors:
            raise self.errors.pop(0)
        super().create_commit(**kwargs)


def _hf_error(mod: Any, status_code: int, headers: dict[str, str] | None = None) -> Exception:
    response = httpx.Response(
        status_code,
        headers=headers or {},
        request=httpx.Request("POST", "https://huggingface.co/api/datasets/org/name/commit/main"),
    )
    return mod.HfHubHTTPError(f"HTTP {status_code}", response=response)


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


def test_upload_folder_batch_retries_529_with_overload_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 529 is HF overload, not a permanent commit failure. The
    orchestrator should back off with a pod-spreading delay and retry
    instead of counting the whole watcher cycle as failed immediately."""
    mod = _load_module()
    (tmp_path / "a.parquet").write_bytes(b"a" * 16)
    api = _FlakyApi([_hf_error(mod, 529)])
    sleeps: list[float] = []
    monkeypatch.setattr(mod.random, "uniform", lambda _lo, hi: hi)

    mod._upload_folder_batch(
        api,
        repo_id="org/name",
        local_dir=tmp_path,
        path_in_repo="tier_X",
        files=["a.parquet"],
        commit_message="test",
        retry_config=mod.HfRetryConfig(max_attempts=2),
        sleep=sleeps.append,
    )

    assert len(api.calls) == 1
    assert sleeps == [mod.HfRetryConfig.overload_min_delay_seconds]


def test_hf_retry_delay_honors_retry_after_before_backoff() -> None:
    """Server-provided Retry-After wins over exponential delay. This is
    especially important when HF asks clients to wait longer than a local
    backoff formula would."""
    mod = _load_module()
    response = httpx.Response(
        529,
        headers={"Retry-After": "91"},
        request=httpx.Request("POST", "https://huggingface.co/api/datasets/org/name/commit/main"),
    )

    delay, source = mod._hf_retry_delay_seconds(529, response, 0, mod.HfRetryConfig())

    assert delay == 91
    assert source == "Retry-After"


def test_hf_retry_delay_uses_ratelimit_reset_for_429() -> None:
    """HF documents RateLimit `t=<seconds>` on 429s. Use that window reset
    instead of guessing with exponential backoff."""
    mod = _load_module()
    response = httpx.Response(
        429,
        headers={"RateLimit": '"api";r=0;t=123'},
        request=httpx.Request("POST", "https://huggingface.co/api/datasets/org/name/commit/main"),
    )

    delay, source = mod._hf_retry_delay_seconds(429, response, 0, mod.HfRetryConfig())

    assert delay == 123
    assert source == "RateLimit"


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
    sentinel_state: dict[str, tuple[int, int]] = {}
    upload_count = [0]
    mod._scan_and_upload(
        api, "org/name", tiers, uploaded, sentinel_state, upload_count, prune_local=False
    )

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
    # Shard goes into the immutable `uploaded` set; sentinel goes into
    # `sentinel_state` keyed by (size, mtime_ns) so a future rewrite is
    # re-uploaded rather than skipped.
    assert uploaded == {"nodes_0001/shard-s000001-r000010.parquet"}
    assert set(sentinel_state.keys()) == {"nodes_0001/_tier_state.json"}


def test_boundary_rewrite_deletes_superseded_shard(tmp_path: Path) -> None:
    """Codex round-4 P2 regression: when a tier grows `n_games` such
    that the previous last shard was partial, the rust runner writes a
    NEW shard filename with a larger row count and `remove_file`'s the
    old smaller-row-count file. The orchestrator must include a
    `CommitOperationDelete` for the old filename in the same commit
    as the new file's `CommitOperationAdd`; otherwise both shards live
    on HF and downstream listers double-count the games in the
    boundary shard."""
    mod = _load_module()
    tier_dir = tmp_path / "nodes_0001"
    tier_dir.mkdir()
    # Cycle 1: only the small (boundary, partial) version of shard 0 exists.
    (tier_dir / "shard-s000000-r000006.parquet").write_bytes(b"x" * 100)

    tiers = [mod.TierLayout(name="nodes_0001", local_dir=tier_dir)]
    api = _FakeApi()
    uploaded: set[str] = set()
    sentinel_state: dict[str, tuple[int, int]] = {}
    upload_count = [0]
    mod._scan_and_upload(
        api, "org/name", tiers, uploaded, sentinel_state, upload_count, prune_local=False
    )
    assert len(api.calls) == 1
    cycle1 = api.calls[0]["operations"]
    assert len(cycle1) == 1
    assert cycle1[0].path_in_repo == "nodes_0001/shard-s000000-r000006.parquet"
    assert uploaded == {"nodes_0001/shard-s000000-r000006.parquet"}

    # Boundary rewrite: rust binary renames `r000006` -> `r000008`
    # (larger row count) and removes the old file.
    (tier_dir / "shard-s000000-r000006.parquet").unlink()
    (tier_dir / "shard-s000000-r000008.parquet").write_bytes(b"x" * 130)

    mod._scan_and_upload(
        api, "org/name", tiers, uploaded, sentinel_state, upload_count, prune_local=False
    )
    assert len(api.calls) == 2
    cycle2 = api.calls[1]["operations"]
    add_paths = [
        op.path_in_repo for op in cycle2 if type(op).__name__ == "CommitOperationAdd"
    ]
    del_paths = [
        op.path_in_repo for op in cycle2 if type(op).__name__ == "CommitOperationDelete"
    ]
    assert add_paths == ["nodes_0001/shard-s000000-r000008.parquet"]
    assert del_paths == ["nodes_0001/shard-s000000-r000006.parquet"], (
        f"superseded shard must be deleted in the same atomic commit; got dels={del_paths}"
    )
    # `uploaded` now reflects the new filename only.
    assert uploaded == {"nodes_0001/shard-s000000-r000008.parquet"}


def test_sentinel_rewrite_triggers_reupload(tmp_path: Path) -> None:
    """Codex round-3 P1 regression: when the rust binary rewrites a
    `_tier_state.json` or `_manifest.json` with updated content (e.g.
    refreshed `n_games` after a tier extension, or end-of-run manifest
    completion), the watcher MUST re-upload it. Tracking sentinels by
    `repo_path in uploaded` (as the previous implementation did) caused
    the rewritten content to be silently dropped — new shards landed
    remote while the manifest stayed stale, and future pods couldn't
    observe the completed extension."""
    mod = _load_module()
    tier_dir = tmp_path / "nodes_0001"
    tier_dir.mkdir()
    state_path = tier_dir / "_tier_state.json"
    state_path.write_text('{"n_games": 100}')
    (tier_dir / "shard-s000000-r000050.parquet").write_bytes(b"x" * 1000)

    tiers = [mod.TierLayout(name="nodes_0001", local_dir=tier_dir)]
    api = _FakeApi()
    uploaded: set[str] = set()
    sentinel_state: dict[str, tuple[int, int]] = {}
    upload_count = [0]

    # Cycle 1: shard + sentinel both upload.
    mod._scan_and_upload(
        api, "org/name", tiers, uploaded, sentinel_state, upload_count, prune_local=False
    )
    assert len(api.calls) == 1
    cycle1_paths = {op.path_in_repo for op in api.calls[0]["operations"]}
    assert cycle1_paths == {
        "nodes_0001/shard-s000000-r000050.parquet",
        "nodes_0001/_tier_state.json",
    }

    # Cycle 2: nothing changed — shard skipped (in `uploaded`), sentinel
    # skipped (signature matches `sentinel_state`).
    mod._scan_and_upload(
        api, "org/name", tiers, uploaded, sentinel_state, upload_count, prune_local=False
    )
    assert len(api.calls) == 1, "no new commit when nothing changed"

    # Rewrite the sentinel as the rust binary would (atomic rename via
    # write to .tmp then rename; here we just rewrite to ensure a fresh
    # mtime_ns). Also slightly different content so size+mtime differs
    # at least in mtime_ns.
    import time
    time.sleep(0.01)  # ensure mtime_ns advances even on coarse clocks
    state_path.write_text('{"n_games": 200, "extended": true}')

    # Cycle 3: sentinel signature differs → it should re-upload. Shard
    # is unchanged so it should NOT re-upload.
    mod._scan_and_upload(
        api, "org/name", tiers, uploaded, sentinel_state, upload_count, prune_local=False
    )
    assert len(api.calls) == 2, "rewritten sentinel must re-upload"
    cycle3_paths = {op.path_in_repo for op in api.calls[1]["operations"]}
    assert cycle3_paths == {"nodes_0001/_tier_state.json"}, (
        f"only the rewritten sentinel should be in the second commit; got {cycle3_paths}"
    )
