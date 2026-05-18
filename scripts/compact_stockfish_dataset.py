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

The unit of work is a *bin* — the set of small shards that compact into one
~500 MB output file. Bins are processed by a **process pool** (`--workers`):
each worker downloads its bin, re-encodes it (the zstd-19 recompression is
CPU-heavy and single-bin-serial would idle most cores), and the main process
batches the uploads.

Resilience: every `hf` call retries transient failures with backoff, and a
worker never raises — a bin that still fails after retries is returned as a
failure and re-run in a second pass, so one bad download cannot kill the run.
Workers also jitter their first download to thin the startup herd.

Transfers go through the `hf` CLI. Hub metadata calls — the repo-tree
listing (also the resume signal) and the final history squash — use the
huggingface_hub Python API.

Resume is repo-state-based: a bin is "done" iff its `data-NNNNN-of-MMMMM`
file is already in the repo, so an interrupted run just re-lists and skips.

Prereqs on the pod:
    curl -LsSf https://hf.co/cli/install.sh | bash          # `hf` CLI
    pip install --break-system-packages pyarrow             # if missing
    export HF_TOKEN=...                                     # write token

Workflow:
    # 1. process every group (idempotent / resumable — re-run if interrupted)
    python3 compact_stockfish_dataset.py --workdir /dev/shm/compact
    # 2. inspect output, then finalize (IRREVERSIBLE)
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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

REPO = "thomas-schweich/pawn-stockfish-100m"
# Pre-compaction archive branch — deleted at finalize so its blobs are GC'd.
LEGACY_BRANCH = "archive/2026-05-10-pre-seed-rework"

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

# Old eval-column names -> new names, per tier family.
RENAME_SEARCH: dict[str, str] = {
    "static_legal_move_evals": "nnue_evals",
    "legal_move_evals": "cp_evals",
}
RENAME_SEARCHLESS: dict[str, str] = {"legal_move_evals": "nnue_evals"}
# On the searchless tier `static_legal_move_evals` is all-null and redundant.
DROP_SEARCHLESS: list[str] = ["static_legal_move_evals"]

_DATA_RE = re.compile(r"data-(\d+)-of-\d+\.parquet$")

api = HfApi()


def hf_retry(cmd: list[str], attempts: int = 6) -> None:
    """Run an `hf` CLI command, retrying transient failures with backoff.

    HF returns 429s / transient 5xx under load; without a retry a single
    blip would abort the whole migration.
    """
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise RuntimeError(
                f"`hf {cmd[1]}` failed after {attempts} attempts: "
                f"{result.stderr.strip()[-400:]}"
            )
        time.sleep(min(90, 2 ** attempt) + random.uniform(0, 6))


def run(cmd: list[str]) -> None:
    """Run a subprocess, echoing the command, raising on non-zero exit."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


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
    """(repo_path, size) for every original `shard-s*` shard in a group."""
    out: list[tuple[str, int]] = []
    for item in api.list_repo_tree(
        REPO, path_in_repo=f"{split_dir}/{tier}", repo_type="dataset", recursive=False
    ):
        size = getattr(item, "size", None)
        name = item.path.rsplit("/", 1)[-1]
        if size is not None and name.startswith("shard-s") and name.endswith(".parquet"):
            out.append((item.path, size))
    out.sort()
    return out


def existing_compacted(split_dir: str, tier: str) -> set[int]:
    """Bin indices already published as `data-NNNNN-of-MMMMM.parquet`."""
    out: set[int] = set()
    try:
        for item in api.list_repo_tree(
            REPO,
            path_in_repo=f"{split_dir}/{tier}",
            repo_type="dataset",
            recursive=False,
        ):
            m = _DATA_RE.search(item.path)
            if m:
                out.add(int(m.group(1)))
    except Exception:  # noqa: BLE001 — missing dir / transient list error
        pass
    return out


def plan_bins(shards: list[tuple[str, int]], target_bytes: int) -> list[list[str]]:
    """Greedily group sorted shards into bins of ~`target_bytes` each.

    Deterministic given the (stable) repo listing, so bin `i` is reproducible
    across resumed runs.
    """
    bins: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name, size in shards:
        cur.append(name)
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
    `rowgroup_bytes` of input before being flushed as one row group.
    """
    rows_in = sum(pq.read_metadata(s).num_rows for s in shards)
    writer: pq.ParquetWriter | None = None
    buf: list[pa.Table] = []
    buf_bytes = 0

    def flush() -> None:
        nonlocal buf, buf_bytes
        if not buf:
            return
        assert writer is not None
        writer.write_table(pa.concat_tables(buf) if len(buf) > 1 else buf[0])
        buf, buf_bytes = [], 0

    for shard in shards:
        table = remap_columns(pq.read_table(shard), tier)
        if writer is None:
            writer = pq.ParquetWriter(
                out_path, table.schema, compression="zstd", compression_level=19
            )
        buf.append(table)
        buf_bytes += shard.stat().st_size
        if buf_bytes >= rowgroup_bytes:
            flush()
    flush()
    assert writer is not None
    writer.close()
    return rows_in, pq.read_metadata(out_path).num_rows


def _compact_one_bin(
    task: tuple[int, int, list[str], str, str, str, int, int],
) -> tuple[int, int | None, str | None, str | None]:
    """Pool worker: download + compact one bin. Never raises.

    Returns (idx, rows_out, out_path, None) on success or
    (idx, None, None, error) on failure, so one bad bin cannot abort the run.
    """
    idx, total, bin_names, tier, split_dir, workdir, rowgroup_bytes, dl_workers = task
    raw_dir = Path(workdir) / "raw" / f"{split_dir}__{tier}__bin{idx:05d}"
    try:
        time.sleep(random.uniform(0, 10))  # thin the startup download herd
        stage_dir = Path(workdir) / "stage" / split_dir / tier
        raw_dir.mkdir(parents=True, exist_ok=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        includes: list[str] = []
        for name in bin_names:
            includes += ["--include", name]
        hf_retry(
            [
                "hf", "download", REPO, "--type", "dataset", *includes,
                "--local-dir", str(raw_dir),
                "--max-workers", str(dl_workers), "--quiet",
            ]
        )

        shard_paths = [raw_dir / name for name in bin_names]
        out_path = stage_dir / f"data-{idx:05d}-of-{total:05d}.parquet"
        rows_in, rows_out = compact_bin(shard_paths, out_path, tier, rowgroup_bytes)
        if rows_in != rows_out:
            raise RuntimeError(f"row-count mismatch: in={rows_in} out={rows_out}")
        shutil.rmtree(raw_dir, ignore_errors=True)
        return idx, rows_out, str(out_path), None
    except Exception as exc:  # noqa: BLE001 — isolate per-bin failure
        shutil.rmtree(raw_dir, ignore_errors=True)
        return idx, None, None, repr(exc)


def upload_staged(
    paths: list[Path], split_dir: str, tier: str, workdir: str, label: str
) -> None:
    """Move a finished batch into a clean dir and upload it in one commit."""
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
            "--type", "dataset",
            "--commit-message", f"compact {split_dir}/{tier}: {label}",
        ]
    )
    shutil.rmtree(updir)


def _run_bins(
    todo: list[int],
    bins: list[list[str]],
    total: int,
    tier: str,
    split_dir: str,
    workdir: str,
    rowgroup_bytes: int,
    args: argparse.Namespace,
) -> list[tuple[int, str]]:
    """Compact the given bin indices via the pool; return (idx, error) failures."""
    tasks = [
        (i, total, bins[i], tier, split_dir, workdir, rowgroup_bytes,
         args.download_workers)
        for i in todo
    ]
    staged: list[Path] = []
    failed: list[tuple[int, str]] = []
    with mp.Pool(args.workers) as pool:
        for n, (idx, rows_out, out_path, err) in enumerate(
            pool.imap_unordered(_compact_one_bin, tasks), 1
        ):
            if err is not None or out_path is None:
                failed.append((idx, err or "unknown"))
                print(f"  [{n}/{len(todo)}] bin {idx} FAILED: {err}", flush=True)
                continue
            staged.append(Path(out_path))
            print(
                f"  [{n}/{len(todo)}] bin {idx} compacted ({rows_out} rows)",
                flush=True,
            )
            if len(staged) >= args.upload_batch:
                upload_staged(staged, split_dir, tier, workdir, f"{len(staged)} files")
                staged = []
    if staged:
        upload_staged(staged, split_dir, tier, workdir, f"{len(staged)} files")
    return failed


def process_group(args: argparse.Namespace, split: str, tier: str) -> None:
    """Compact one (split, tier) group with a worker pool, batched uploads."""
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
    done = existing_compacted(split_dir, tier)
    todo = [i for i in range(total) if i not in done]
    print(
        f"\n=== {split}/{tier}: {total} files | {len(done)} present | "
        f"{len(todo)} to build ===",
        flush=True,
    )

    rowgroup_bytes = args.rowgroup_mb * 1024 * 1024
    if todo:
        failed = _run_bins(
            todo, bins, total, tier, split_dir, str(workdir), rowgroup_bytes, args
        )
        if failed:
            print(f"  retrying {len(failed)} failed bins ...", flush=True)
            failed = _run_bins(
                [i for i, _ in failed], bins, total, tier, split_dir,
                str(workdir), rowgroup_bytes, args,
            )
        if failed:
            raise RuntimeError(
                f"{split}/{tier}: bins still failing after retry: {failed[:5]}"
            )

    present = existing_compacted(split_dir, tier)
    if len(present) != total:
        raise RuntimeError(
            f"{split}/{tier}: {len(present)}/{total} compacted files present "
            f"after run — missing {sorted(set(range(total)) - present)[:10]}"
        )
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.touch()
    print(f"[{split}/{tier}] done — {total} files verified present", flush=True)


def finalize() -> None:
    """Delete old shards + legacy branch, then squash history to reclaim storage."""
    print("Deleting pre-compaction shards (shard-s*.parquet) ...")
    run(
        [
            "hf", "repos", "delete-files", REPO, "**/shard-s*.parquet",
            "--type", "dataset",
            "--commit-message", "drop pre-compaction 2k-game shards",
        ]
    )
    print(f"Deleting legacy branch {LEGACY_BRANCH} ...")
    run(["hf", "repos", "branch", "delete", REPO, LEGACY_BRANCH, "--type", "dataset"])
    print("Squashing repo history (IRREVERSIBLE) ...")
    api.super_squash_history(
        repo_id=REPO,
        repo_type="dataset",
        commit_message="compact to ~500MB parquet; rename eval columns",
    )
    print("Done. Old blobs are now unreferenced and will be GC'd by the Hub.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", required=True, help="scratch dir — point at /dev/shm/<name>"
    )
    parser.add_argument(
        "--workers", type=int, default=24, help="parallel bin-compaction processes"
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
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument(
        "--tiers", nargs="+", choices=TIERS, default=TIERS, help="subset of tiers"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="delete old shards + legacy branch + squash (run only after verifying)",
    )
    args = parser.parse_args()

    if args.finalize:
        finalize()
        return 0

    for split in SPLIT_DIRS:
        for tier in args.tiers:
            process_group(args, split, tier)
    print("\nAll groups compacted + uploaded. Verify, then re-run with --finalize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
