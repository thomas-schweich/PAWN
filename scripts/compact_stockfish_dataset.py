#!/usr/bin/env python3
"""Compact + rename the `pawn-stockfish-100m` HF dataset, in place.

One-off maintenance op. The dataset was published as 50,000 ~20 MB parquet
shards (2,000 games each) — a generation-time memory artifact, not a
download-friendly layout. This rewrites it as ~500 MB parquet files and
renames the two eval columns to a tier-uniform scheme:

    nnue_evals  -- raw NNUE per-legal-move evals (every legal move)
                   search tiers : was `static_legal_move_evals`
                   searchless   : was `legal_move_evals` (raw there)
    cp_evals    -- MultiPV / search-ranked cp evals (top-5)
                   search tiers : was `legal_move_evals`
                   searchless   : absent

Staged on a PR branch. Every compacted file is uploaded to an HF pull
request (`refs/pr/N`), never to `main` directly — so for the whole
multi-hour run `main` (what `load_dataset` reads) stays the consistent
pre-migration dataset. `--finalize` deletes the old shards on the PR,
merges it (the atomic swap to the compacted layout), then squashes
history. `main` is mutated exactly once, by the merge.

Two units of work:

  * A *slice* (~`--slice-mb`) is downloaded with a SINGLE `hf download`
    call. This matters: every `hf download` invocation re-walks the whole
    50,000-file repo tree, and HF caps API requests at 1000 / 5 min — so
    downloading per-bin (~2000 tree-walks) blows the quota instantly.
    Per-slice downloading keeps it to a few dozen tree-walks for the run.
    The next slice is prefetched while the current one compacts, so the
    cores are not idle during downloads.
  * A *bin* (~`--target-mb`) is the set of shards that compact into one
    output file. Bins of a downloaded slice are compacted in parallel by a
    process pool (`--workers`) — the workers read LOCAL files only and make
    no API calls, so the CPU-bound zstd-19 recompression uses all cores
    without touching the rate limit.

Per (split, tier) group: list shards, plan bins, group bins into slices,
then for each slice download -> pool-compact -> batched upload -> delete.

Resume is repo-state-based: a bin is "done" iff its `data-NNNNN-of-MMMMM`
file is already on the PR (plus a local `.done` marker that short-circuits
an already-finished group). The PR is rediscovered by title on resume; its
marker file records `--target-mb`, and a resume with a different value is
rejected (bin boundaries would shift). `--finalize` is safely re-runnable —
it detects an already-merged PR and skips to the post-merge cleanup
(marker + legacy-branch delete, history squash). `hf` download/upload
calls and the Hub list / merge / squash calls retry transient failures
with backoff; the marker file-ops are single-shot, but every `--finalize`
step is idempotent so a re-run completes whatever a failure left undone.

Prereqs on the pod (which has no `uv`):
    curl -LsSf https://hf.co/cli/install.sh | bash          # `hf` CLI
    pip install --break-system-packages pyarrow             # if missing
    export HF_TOKEN=...                                     # write token

Workflow:
    # 1. process every group (idempotent / resumable — re-run if interrupted)
    python3 compact_stockfish_dataset.py --workdir /dev/shm/compact
    # 2. inspect the PR, then finalize (IRREVERSIBLE)
    python3 compact_stockfish_dataset.py --workdir /dev/shm/compact --finalize
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
from pathlib import Path
from typing import NamedTuple, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile, RepoFolder

REPO = "thomas-schweich/pawn-stockfish-100m"
# Pre-compaction archive branch — deleted at finalize so its blobs are GC'd.
LEGACY_BRANCH = "archive/2026-05-10-pre-seed-rework"
# The migration PR — created once, rediscovered by this exact title on resume.
PR_TITLE = "Compact to ~500MB parquet + rename eval columns"
PR_MARKER = "_compact/.migration-pr"

TIERS: list[str] = [
    "tier0_evallegal",
    "nodes_0001",
    "nodes_0128",
    "nodes_0256",
    "nodes_1024",
]
SEARCHLESS_TIER = "tier0_evallegal"

# split name (HF) -> directory in the repo
SPLIT_DIRS: dict[str, str] = {"train": "train", "validation": "val", "test": "test"}
N_GROUPS = len(SPLIT_DIRS) * len(TIERS)

# Old eval-column names -> new names, per tier family.
RENAME_SEARCH: dict[str, str] = {
    "static_legal_move_evals": "nnue_evals",
    "legal_move_evals": "cp_evals",
}
RENAME_SEARCHLESS: dict[str, str] = {"legal_move_evals": "nnue_evals"}
# On the searchless tier `static_legal_move_evals` is all-null and redundant.
DROP_SEARCHLESS: list[str] = ["static_legal_move_evals"]

_DATA_RE = re.compile(r"data-(\d+)-of-(\d+)\.parquet$")

# spawn (not fork): worker processes re-import this module cleanly, avoiding
# any post-fork state hazard from pyarrow's internal thread pools.
_MP = mp.get_context("spawn")

api = HfApi()

_T = TypeVar("_T")


class BinResult(NamedTuple):
    """One pool task's outcome. Exactly one of (rows/out_path) or error is set."""

    idx: int
    rows: int | None
    out_path: str | None
    error: str | None


def hf_retry(cmd: list[str], attempts: int = 6) -> None:
    """Run an `hf` CLI command, retrying transient failures with backoff."""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        if attempt == attempts:
            # Lead of stderr — the actionable line (401, 404, quota) is first.
            raise RuntimeError(
                f"`hf {cmd[1]}` failed after {attempts} attempts: "
                f"{result.stderr.strip()[:600]}"
            )
        time.sleep(min(120, 5 * 2 ** attempt) + random.uniform(0, 8))


def api_retry(produce: Callable[[], _T], retries: int = 5) -> _T:
    """Call a Hub API thunk, retrying transient failures with backoff.

    `produce` must fully materialize its result (e.g. wrap generator-returning
    APIs in `list(...)`) so iteration-time errors are caught here too.
    """
    for attempt in range(1, retries + 1):
        try:
            return produce()
        except Exception:  # noqa: BLE001 — transient HF API / network error
            if attempt == retries:
                raise
            time.sleep(min(60, 2 ** attempt) + random.uniform(0, 4))
    raise AssertionError("unreachable")


def run(cmd: list[str]) -> None:
    """Run a subprocess, echoing the command, raising on non-zero exit."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def list_repo_tree(
    path: str, revision: str | None = None, recursive: bool = False
) -> list[RepoFile | RepoFolder]:
    """`api.list_repo_tree` with retry — raises on persistent failure.

    Never returns a silent empty list on error: an empty result must mean the
    directory genuinely has no matching entries, so callers (including the
    coverage checks) can trust it.
    """
    return api_retry(
        lambda: list(
            api.list_repo_tree(
                REPO,
                path_in_repo=path,
                repo_type="dataset",
                revision=revision,
                recursive=recursive,
            )
        )
    )


def find_migration_pr() -> tuple[int | None, str | None]:
    """Locate the migration PR by exact title.

    Returns (num, status). A *merged* PR is terminal and wins over any stray
    open PR; `draft` counts as open (resume must adopt it, not duplicate it).
    Among open PRs the lowest number wins — deterministic dedup if a resume
    race opened two.
    """
    open_nums: list[int] = []
    merged_nums: list[int] = []
    for d in api_retry(
        lambda: list(api.get_repo_discussions(REPO, repo_type="dataset"))
    ):
        if d.is_pull_request and d.title == PR_TITLE:
            if d.status == "merged":
                merged_nums.append(d.num)
            elif d.status in ("open", "draft"):
                open_nums.append(d.num)
    if merged_nums:
        return min(merged_nums), "merged"
    if open_nums:
        return min(open_nums), "open"
    return None, None


def _marker_target_mb(pr_revision: str) -> int | None:
    """The `--target-mb` recorded in the PR's marker file, or None if absent.

    `hf_hub_download` is called directly, not via `api_retry`: a missing
    marker raises `EntryNotFoundError` — a permanent error that must not be
    retried. Raises if the marker exists but carries no `target_mb=` line, so
    a malformed marker can't silently disable the resume guard.
    """
    try:
        path = hf_hub_download(
            REPO, PR_MARKER, repo_type="dataset", revision=pr_revision
        )
    except EntryNotFoundError:
        return None
    for line in Path(path).read_text().splitlines():
        if line.startswith("target_mb="):
            return int(line.split("=", 1)[1])
    raise RuntimeError(
        f"migration PR marker {PR_MARKER} has no target_mb= line — refusing "
        f"to resume against an unrecognized marker"
    )


def ensure_pr(target_mb: int) -> str:
    """Return the `refs/pr/N` revision of the open migration PR, creating once.

    Rediscovered by title so a resumed run reuses the same PR. Exits if the PR
    has already been merged. Rejects a resume whose `--target-mb` differs from
    the value recorded when the PR was created — a different value would shift
    every bin boundary while leaving the `data-*` filenames looking valid.
    """
    num, status = find_migration_pr()
    if status == "merged":
        raise SystemExit(
            "migration PR is already merged — compaction is complete; "
            "run --finalize if history still needs squashing."
        )
    if num is not None:
        pr_revision = f"refs/pr/{num}"
        marker_mb = _marker_target_mb(pr_revision)
        if marker_mb is not None and marker_mb != target_mb:
            raise SystemExit(
                f"PR #{num} was created with --target-mb={marker_mb}; resuming "
                f"requires that same value (got {target_mb})."
            )
        print(f"  using existing migration PR #{num}", flush=True)
        return pr_revision
    info = api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(PR_MARKER, f"target_mb={target_mb}\n".encode())
        ],
        commit_message=PR_TITLE,
        commit_description=(
            "Automated migration: 50k ~20MB shards -> ~2k ~500MB files, "
            "eval columns renamed to nnue_evals / cp_evals."
        ),
        create_pr=True,
    )
    if info.pr_num is None:
        raise RuntimeError("create_commit(create_pr=True) returned no pr_num")
    # Re-scan and adopt the lowest open PR: if a concurrent run also created
    # one, both runs converge on the same PR rather than splitting uploads.
    num, _ = find_migration_pr()
    chosen = num if num is not None else info.pr_num
    print(f"  opened migration PR #{chosen}", flush=True)
    return f"refs/pr/{chosen}"


def remap_columns(table: pa.Table, tier: str) -> pa.Table:
    """Apply the tier-appropriate column drop + rename to one table."""
    if tier == SEARCHLESS_TIER:
        drop = [c for c in DROP_SEARCHLESS if c in table.column_names]
        if drop:
            table = table.drop_columns(drop)
        mapping = RENAME_SEARCHLESS
    else:
        mapping = RENAME_SEARCH
    new_names = [mapping.get(name, name) for name in table.column_names]
    return table.rename_columns(new_names)


def list_group_shards(split_dir: str, tier: str) -> list[tuple[str, int]]:
    """(repo_path, size) for every original `shard-s*` shard in a group.

    Listed on `main` — the source shards live there untouched until finalize.
    """
    out: list[tuple[str, int]] = []
    for item in list_repo_tree(f"{split_dir}/{tier}"):
        if not isinstance(item, RepoFile):
            continue
        name = item.path.rsplit("/", 1)[-1]
        if name.startswith("shard-s") and name.endswith(".parquet"):
            out.append((item.path, item.size))
    out.sort()
    return out


def existing_compacted(
    split_dir: str, tier: str, revision: str
) -> tuple[set[int], set[int]]:
    """(bin indices, `-of-MMMMM` totals) of `data-*` files on the PR revision.

    A healthy group yields totals == {} (none yet) or {N} (one consistent
    total); anything else means a stale / mismatched prior compaction.
    """
    idxs: set[int] = set()
    totals: set[int] = set()
    for item in list_repo_tree(f"{split_dir}/{tier}", revision=revision):
        if not isinstance(item, RepoFile):
            continue
        m = _DATA_RE.search(item.path)
        if m:
            idxs.add(int(m.group(1)))
            totals.add(int(m.group(2)))
    return idxs, totals


def shards_remain(revision: str) -> bool:
    """True if any pre-compaction `shard-s*` file is still present on `revision`."""
    for item in list_repo_tree("", revision=revision, recursive=True):
        if not isinstance(item, RepoFile):
            continue
        name = item.path.rsplit("/", 1)[-1]
        if name.startswith("shard-s") and name.endswith(".parquet"):
            return True
    return False


def plan_bins(
    shards: list[tuple[str, int]], target_bytes: int
) -> list[list[tuple[str, int]]]:
    """Greedily group sorted shards into bins of ~`target_bytes` each.

    Deterministic given the (stable) repo listing, so bin `i` is reproducible
    across resumed runs. Each bin keeps (path, size) pairs.
    """
    bins: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_bytes = 0
    for name, size in shards:
        cur.append((name, size))
        cur_bytes += size
        if cur_bytes >= target_bytes:
            bins.append(cur)
            cur, cur_bytes = [], 0
    if cur:
        bins.append(cur)
    return bins


def compact_bin(
    shards: list[Path], out_path: Path, tier: str, rowgroup_bytes: int
) -> tuple[int, int]:
    """Concatenate one bin's shards into a single parquet file at `out_path`.

    Returns (rows_in, rows_out). Shards are buffered only up to
    `rowgroup_bytes` of decompressed Arrow data before being flushed as one
    row group.
    """
    writer: pq.ParquetWriter | None = None
    buf: list[pa.Table] = []
    buf_bytes = 0
    rows_in = 0

    def flush() -> None:
        nonlocal buf, buf_bytes
        if not buf:
            return
        assert writer is not None
        writer.write_table(pa.concat_tables(buf) if len(buf) > 1 else buf[0])
        buf, buf_bytes = [], 0

    try:
        for shard in shards:
            table = remap_columns(pq.read_table(shard), tier)
            rows_in += table.num_rows
            if writer is None:
                writer = pq.ParquetWriter(
                    out_path, table.schema, compression="zstd", compression_level=19
                )
            buf.append(table)
            buf_bytes += table.nbytes  # decompressed Arrow size — what's in RAM
            if buf_bytes >= rowgroup_bytes:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    return rows_in, pq.read_metadata(out_path).num_rows


def _compact_one_bin(
    task: tuple[int, int, list[str], str, str, str, int],
) -> BinResult:
    """Pool worker: compact one already-downloaded bin. Never raises.

    Reads local shards only — no network — so the pool can use every core
    without touching the HF API rate limit. Deletes the bin's raw shards
    once compacted to bound disk.
    """
    idx, total, shard_names, tier, raw_root, stage_root, rowgroup_bytes = task
    try:
        stage_dir = Path(stage_root)
        stage_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [Path(raw_root) / name for name in shard_names]
        out_path = stage_dir / f"data-{idx:05d}-of-{total:05d}.parquet"
        rows_in, rows_out = compact_bin(shard_paths, out_path, tier, rowgroup_bytes)
        if rows_in != rows_out:
            raise RuntimeError(f"row-count mismatch: in={rows_in} out={rows_out}")
        for shard in shard_paths:
            shard.unlink(missing_ok=True)
        return BinResult(idx, rows_out, str(out_path), None)
    except Exception as exc:  # noqa: BLE001 — isolate per-bin failure
        return BinResult(idx, None, None, repr(exc))


def download_slice(shard_names: list[str], raw_dir: Path, dl_workers: int) -> Path:
    """Download every shard of a slice in ONE `hf download` call (one tree-walk)."""
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    includes = [arg for name in shard_names for arg in ("--include", name)]
    hf_retry(
        [
            "hf", "download", REPO, "--type", "dataset", *includes,
            "--local-dir", str(raw_dir),
            "--max-workers", str(dl_workers), "--quiet",
        ]
    )
    return raw_dir


def upload_staged(
    paths: list[Path], split_dir: str, tier: str, workdir: str, revision: str,
    label: str,
) -> None:
    """Move a finished batch into a clean dir and upload it to the PR in one commit."""
    updir = Path(workdir) / "upload" / split_dir / tier
    if updir.exists():
        shutil.rmtree(updir)
    updir.mkdir(parents=True)
    for path in paths:
        shutil.move(str(path), str(updir / path.name))
    print(f"  uploading {label} -> {split_dir}/{tier}", flush=True)
    hf_retry(
        [
            "hf", "upload", REPO, str(updir), f"{split_dir}/{tier}",
            "--type", "dataset", "--revision", revision,
            "--commit-message", f"compact {split_dir}/{tier}: {label}",
        ]
    )
    shutil.rmtree(updir)


def _compact_slice(
    pool: Pool,
    slice_idxs: list[int],
    bins: list[list[tuple[str, int]]],
    total: int,
    tier: str,
    split_dir: str,
    raw_dir: Path,
    workdir: Path,
    pr_revision: str,
    rowgroup_bytes: int,
    upload_batch: int,
) -> None:
    """Compact every bin of one downloaded slice; upload to the PR in batches."""
    stage_root = workdir / "stage" / split_dir / tier
    tasks = [
        (i, total, [name for name, _ in bins[i]], tier, str(raw_dir),
         str(stage_root), rowgroup_bytes)
        for i in slice_idxs
    ]
    staged: list[Path] = []
    failures: list[tuple[int, str]] = []
    for res in pool.imap_unordered(_compact_one_bin, tasks):
        if res.error is not None or res.out_path is None:
            failures.append((res.idx, res.error or "unknown"))
            print(f"  bin {res.idx} FAILED: {res.error}", flush=True)
            continue
        staged.append(Path(res.out_path))
        print(f"  bin {res.idx} compacted ({res.rows} rows)", flush=True)
        if len(staged) >= upload_batch:
            upload_staged(
                staged, split_dir, tier, str(workdir), pr_revision,
                f"{len(staged)} files",
            )
            staged = []
    if staged:
        upload_staged(
            staged, split_dir, tier, str(workdir), pr_revision, f"{len(staged)} files"
        )
    if failures:
        raise RuntimeError(f"{split_dir}/{tier}: bin compaction failed: {failures[:5]}")


def process_group(
    args: argparse.Namespace, split: str, tier: str, pr_revision: str
) -> None:
    """Compact one (split, tier) group: prefetched per-slice downloads, pooled."""
    workdir = Path(args.workdir)
    done_marker = workdir / ".done" / f"{split}__{tier}"
    split_dir = SPLIT_DIRS[split]
    if done_marker.exists():
        print(f"[{split}/{tier}] already done — skipping")
        return

    bins = plan_bins(
        list_group_shards(split_dir, tier), args.target_mb * 1024 * 1024
    )
    total = len(bins)
    done_idxs, done_totals = existing_compacted(split_dir, tier, pr_revision)
    if done_totals - {total}:
        raise RuntimeError(
            f"{split}/{tier}: PR has data-* files with total(s) {done_totals}, "
            f"current plan has {total} — a stale compaction is present; clear the "
            f"data-* files for this group before resuming"
        )
    todo_bins = [i for i in range(total) if i not in done_idxs]
    print(
        f"\n=== {split}/{tier}: {total} files | {len(done_idxs)} present | "
        f"{len(todo_bins)} to build ===",
        flush=True,
    )

    if todo_bins:
        # Group the to-do bins into download slices of ~`--slice-mb`.
        slice_bytes = args.slice_mb * 1024 * 1024
        slices: list[list[int]] = []
        cur: list[int] = []
        cur_bytes = 0
        for i in todo_bins:
            cur.append(i)
            cur_bytes += sum(size for _, size in bins[i])
            if cur_bytes >= slice_bytes:
                slices.append(cur)
                cur, cur_bytes = [], 0
        if cur:
            slices.append(cur)

        rowgroup_bytes = args.rowgroup_mb * 1024 * 1024
        raw_dirs = [workdir / "raw" / "a", workdir / "raw" / "b"]

        def fetch(sn: int) -> Path:
            names = [name for i in slices[sn] for name, _ in bins[i]]
            print(
                f"  slice {sn + 1}/{len(slices)}: {len(slices[sn])} bins / "
                f"{len(names)} shards — downloading",
                flush=True,
            )
            return download_slice(names, raw_dirs[sn % 2], args.download_workers)

        # Prefetch the next slice's download while the pool compacts the
        # current one. The single-thread executor + 2-dir ping-pong keeps the
        # in-flight download and the in-flight compaction on separate dirs;
        # raising `max_workers` would break that invariant.
        with ThreadPoolExecutor(max_workers=1) as dlx, _MP.Pool(args.workers) as pool:
            pending = dlx.submit(fetch, 0)
            for sn in range(len(slices)):
                raw_dir = pending.result()
                if sn + 1 < len(slices):
                    pending = dlx.submit(fetch, sn + 1)
                _compact_slice(
                    pool, slices[sn], bins, total, tier, split_dir, raw_dir,
                    workdir, pr_revision, rowgroup_bytes, args.upload_batch,
                )
                shutil.rmtree(raw_dir, ignore_errors=True)

    present_idxs, present_totals = existing_compacted(split_dir, tier, pr_revision)
    if present_idxs != set(range(total)) or present_totals - {total}:
        raise RuntimeError(
            f"{split}/{tier}: coverage check failed — {len(present_idxs)}/{total} "
            f"files, totals={present_totals or set()}, "
            f"missing={sorted(set(range(total)) - present_idxs)[:10]}"
        )
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.touch()
    print(f"[{split}/{tier}] done — {total} files verified present", flush=True)


def _pr_status(pr_num: int) -> str | None:
    """Status of the PR with this exact number, or None if not found."""
    for d in api_retry(
        lambda: list(api.get_repo_discussions(REPO, repo_type="dataset"))
    ):
        if d.is_pull_request and d.num == pr_num:
            return d.status
    return None


def _merge_migration_pr(pr_num: int) -> None:
    """Merge the PR into main, retried; a concurrent merge is treated as success."""
    for attempt in range(1, 4):
        try:
            api.merge_pull_request(REPO, pr_num, repo_type="dataset")
            return
        except Exception:  # noqa: BLE001 — transient, or already merged
            try:
                merged = _pr_status(pr_num) == "merged"
            except Exception:  # noqa: BLE001 — status probe itself flaked
                merged = False
            if merged:
                print(f"  PR #{pr_num} is already merged.", flush=True)
                return
            if attempt == 3:
                raise
            time.sleep(15 * attempt)


def _post_merge_cleanup() -> None:
    """Delete the marker + legacy branch and squash history. Idempotent.

    The marker is removed only here, after the merge — keeping it present
    for the whole open-PR lifetime means the `--target-mb` resume guard
    (`_marker_target_mb`) stays enforceable right up to the merge.
    """
    try:
        api.delete_file(
            PR_MARKER, REPO, repo_type="dataset",
            commit_message="remove migration marker",
        )
    except EntryNotFoundError:
        pass  # already removed by an earlier finalize run
    branches = {
        b.name
        for b in api_retry(
            lambda: api.list_repo_refs(REPO, repo_type="dataset")
        ).branches
    }
    if LEGACY_BRANCH in branches:
        print(f"Deleting legacy branch {LEGACY_BRANCH} ...")
        hf_retry(
            ["hf", "repos", "branch", "delete", REPO, LEGACY_BRANCH, "--type", "dataset"]
        )
    else:
        print(f"Legacy branch {LEGACY_BRANCH} already absent — skipping.")
    # Skip the squash if history is already a single commit (a prior finalize
    # run squashed it but failed/interrupted afterwards).
    n_commits = len(
        api_retry(lambda: list(api.list_repo_commits(REPO, repo_type="dataset")))
    )
    if n_commits <= 1:
        print("Repo history already squashed — skipping.")
    else:
        print("Squashing repo history (IRREVERSIBLE) ...")
        api_retry(
            lambda: api.super_squash_history(
                repo_id=REPO,
                repo_type="dataset",
                commit_message="compact to ~500MB parquet; rename eval columns",
            ),
            retries=3,
        )
    print("Done. Old blobs are now unreferenced and will be GC'd by the Hub.")


def finalize(args: argparse.Namespace) -> None:
    """Verify the PR, delete old shards on it, merge it, then squash history.

    Safely re-runnable: if the PR is already merged (a prior run got past the
    merge), it skips to the legacy-branch + squash cleanup. While the PR is
    still open it refuses unless every (split, tier) group is fully compacted —
    the expected bin count is recomputed from the still-present source shards
    on `main`, and the published `data-*` set must cover exactly
    `range(expected)` with one consistent `-of-MMMMM` total.
    """
    pr_num, status = find_migration_pr()
    if pr_num is None:
        raise SystemExit("no migration PR found — run the compaction first")
    if status == "merged":
        print(f"Migration PR #{pr_num} already merged — finishing cleanup.")
        _post_merge_cleanup()
        return

    pr_revision = f"refs/pr/{pr_num}"
    print(f"Verifying all {N_GROUPS} groups are fully compacted on PR #{pr_num} ...")
    for split in SPLIT_DIRS:
        for tier in TIERS:
            split_dir = SPLIT_DIRS[split]
            expected = len(
                plan_bins(
                    list_group_shards(split_dir, tier), args.target_mb * 1024 * 1024
                )
            )
            idxs, totals = existing_compacted(split_dir, tier, pr_revision)
            if idxs != set(range(expected)) or totals - {expected}:
                raise RuntimeError(
                    f"refusing to finalize — {split}/{tier}: {len(idxs)}/{expected} "
                    f"compacted files present, totals={totals or set()}"
                )

    # Idempotent: skip the delete if the shards are already gone from the PR.
    if shards_remain(pr_revision):
        print(f"All {N_GROUPS} groups verified. Deleting old shards on the PR ...")
        hf_retry(
            [
                "hf", "repos", "delete-files", REPO, "**/shard-s*.parquet",
                "--type", "dataset", "--revision", pr_revision,
                "--commit-message", "drop pre-compaction 2k-game shards",
            ]
        )
    else:
        print(f"All {N_GROUPS} groups verified. Old shards already removed on the PR.")

    print(f"Merging migration PR #{pr_num} into main ...")
    _merge_migration_pr(pr_num)
    _post_merge_cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", required=True, help="scratch dir — point at /dev/shm/<name>"
    )
    parser.add_argument(
        "--workers", type=int, default=48, help="parallel bin-compaction processes"
    )
    parser.add_argument(
        "--slice-mb", type=int, default=30000, help="approx download slice size"
    )
    parser.add_argument(
        "--target-mb", type=int, default=500, help="approx output file size"
    )
    parser.add_argument(
        "--rowgroup-mb", type=int, default=128, help="approx row-group size"
    )
    parser.add_argument(
        "--upload-batch", type=int, default=25, help="output files per commit"
    )
    parser.add_argument(
        "--download-workers", type=int, default=16, help="hf download parallelism"
    )
    parser.add_argument(
        "--tiers", nargs="+", choices=TIERS, default=TIERS, help="subset of tiers"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="delete old shards + merge the PR + squash; checks ALL tiers "
             "regardless of --tiers (run only after every group is verified)",
    )
    args = parser.parse_args()

    if args.finalize:
        finalize(args)
        return 0

    # Worst case on disk: current raw slice + prefetched next slice + the
    # staged batch + upload_staged's transient copy of it.
    need_gb = args.slice_mb * 3.0 / 1024
    free_gb = shutil.disk_usage(Path(args.workdir).parent).free / 1024**3
    if free_gb < need_gb:
        raise SystemExit(
            f"workdir filesystem has {free_gb:.0f} GB free; this run needs "
            f"~{need_gb:.0f} GB (2 slices + staging). Lower --slice-mb or use a "
            f"bigger /dev/shm."
        )

    pr_revision = ensure_pr(args.target_mb)
    for split in SPLIT_DIRS:
        for tier in args.tiers:
            process_group(args, split, tier, pr_revision)
    print(f"\nAll groups compacted onto {pr_revision}. "
          f"Inspect the PR, then re-run with --finalize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
