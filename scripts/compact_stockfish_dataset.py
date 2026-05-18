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

Two units of work:

  * A *slice* (~`--slice-mb`) is downloaded with a SINGLE `hf download`
    call. This matters: every `hf download` invocation re-walks the whole
    50,000-file repo tree, and HF caps API requests at 1000 / 5 min — so
    downloading per-bin (~1850 tree-walks) blows the quota instantly.
    Per-slice downloading keeps it to a few dozen tree-walks for the run.
  * A *bin* (~`--target-mb`) is the set of shards that compact into one
    output file. Bins of a downloaded slice are compacted in parallel by a
    process pool (`--workers`) — the workers read LOCAL files only and make
    no API calls, so the CPU-bound zstd-19 recompression uses all cores
    without touching the rate limit.

Per (split, tier) group: list shards, plan bins, group bins into slices,
then for each slice download -> pool-compact -> batched upload -> delete.

Resume is repo-state-based: a bin is "done" iff its `data-NNNNN-of-MMMMM`
file is already in the repo (plus a local `.done` marker per finished
group). Every download/upload `hf` call and Hub listing retries transient
failures with backoff; the irreversible `--finalize` ops do not.

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

_DATA_RE = re.compile(r"data-(\d+)-of-(\d+)\.parquet$")

api = HfApi()


def hf_retry(cmd: list[str], attempts: int = 6) -> None:
    """Run an `hf` CLI command, retrying transient failures with backoff."""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise RuntimeError(
                f"`hf {cmd[1]}` failed after {attempts} attempts: "
                f"{result.stderr.strip()[-400:]}"
            )
        time.sleep(min(120, 5 * 2 ** attempt) + random.uniform(0, 8))


def list_repo_tree(path: str, retries: int = 5) -> list:
    """`api.list_repo_tree` with retry — raises on persistent failure.

    Never returns a silent empty list on error: an empty result must mean the
    directory genuinely has no matching entries, so callers (including the
    final coverage check) can trust it.
    """
    for attempt in range(1, retries + 1):
        try:
            return list(
                api.list_repo_tree(
                    REPO, path_in_repo=path, repo_type="dataset", recursive=False
                )
            )
        except Exception:  # noqa: BLE001 — transient HF API / network error
            if attempt == retries:
                raise
            time.sleep(min(60, 2 ** attempt) + random.uniform(0, 4))
    return []  # unreachable


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
    for item in list_repo_tree(f"{split_dir}/{tier}"):
        size = getattr(item, "size", None)
        name = item.path.rsplit("/", 1)[-1]
        if size is not None and name.startswith("shard-s") and name.endswith(".parquet"):
            out.append((item.path, size))
    out.sort()
    return out


def existing_compacted(split_dir: str, tier: str) -> set[int]:
    """Bin indices already published as `data-NNNNN-of-MMMMM.parquet`."""
    out: set[int] = set()
    for item in list_repo_tree(f"{split_dir}/{tier}"):
        m = _DATA_RE.search(item.path)
        if m:
            out.add(int(m.group(1)))
    return out


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

    try:
        for shard in shards:
            table = remap_columns(pq.read_table(shard), tier)
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
) -> tuple[int, int | None, str | None, str | None]:
    """Pool worker: compact one already-downloaded bin. Never raises.

    Reads local shards only — no network — so the pool can use every core
    without touching the HF API rate limit. Returns
    (idx, rows_out, out_path, None) on success or (idx, None, None, error).
    Deletes the bin's raw shards once compacted to bound disk.
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
        return idx, rows_out, str(out_path), None
    except Exception as exc:  # noqa: BLE001 — isolate per-bin failure
        return idx, None, None, repr(exc)


def download_slice(shard_names: list[str], raw_root: Path, dl_workers: int) -> None:
    """Download every shard of a slice in ONE `hf download` call (one tree-walk)."""
    includes: list[str] = []
    for name in shard_names:
        includes += ["--include", name]
    hf_retry(
        [
            "hf", "download", REPO, "--type", "dataset", *includes,
            "--local-dir", str(raw_root),
            "--max-workers", str(dl_workers), "--quiet",
        ]
    )


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


def process_group(args: argparse.Namespace, split: str, tier: str) -> None:
    """Compact one (split, tier) group: per-slice download, pooled compaction."""
    workdir = Path(args.workdir)
    done_marker = workdir / ".done" / f"{split}__{tier}"
    split_dir = SPLIT_DIRS[split]
    if done_marker.exists():
        print(f"[{split}/{tier}] already done — skipping")
        return

    shards = list_group_shards(split_dir, tier)
    size_of = dict(shards)
    bins = plan_bins(shards, args.target_mb * 1024 * 1024)
    total = len(bins)
    done = existing_compacted(split_dir, tier)
    todo_bins = [i for i in range(total) if i not in done]
    print(
        f"\n=== {split}/{tier}: {total} files | {len(done)} present | "
        f"{len(todo_bins)} to build ===",
        flush=True,
    )

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

    raw_root = workdir / "raw"
    stage_root = workdir / "stage" / split_dir / tier
    rowgroup_bytes = args.rowgroup_mb * 1024 * 1024
    for sn, slice_idxs in enumerate(slices, 1):
        if raw_root.exists():
            shutil.rmtree(raw_root)
        shard_names = [name for i in slice_idxs for name, _ in bins[i]]
        print(
            f"  slice {sn}/{len(slices)}: {len(slice_idxs)} bins / "
            f"{len(shard_names)} shards — downloading",
            flush=True,
        )
        download_slice(shard_names, raw_root, args.download_workers)

        tasks = [
            (i, total, [name for name, _ in bins[i]], tier, str(raw_root),
             str(stage_root), rowgroup_bytes)
            for i in slice_idxs
        ]
        staged: list[Path] = []
        failures: list[tuple[int, str]] = []
        with mp.Pool(args.workers) as pool:
            for idx, rows_out, out_path, err in pool.imap_unordered(
                _compact_one_bin, tasks
            ):
                if err is not None or out_path is None:
                    failures.append((idx, err or "unknown"))
                    print(f"  bin {idx} FAILED: {err}", flush=True)
                    continue
                staged.append(Path(out_path))
                print(f"  bin {idx} compacted ({rows_out} rows)", flush=True)
                if len(staged) >= args.upload_batch:
                    upload_staged(
                        staged, split_dir, tier, str(workdir), f"{len(staged)} files"
                    )
                    staged = []
        if staged:
            upload_staged(
                staged, split_dir, tier, str(workdir), f"{len(staged)} files"
            )
        if failures:
            raise RuntimeError(
                f"{split}/{tier} slice {sn}: bin compaction failed: {failures[:5]}"
            )
    if raw_root.exists():
        shutil.rmtree(raw_root)

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
    """Delete old shards + legacy branch, then squash history to reclaim storage.

    Refuses to run unless every (split, tier) group is fully compacted — the
    `data-*` filenames encode their own `-of-MMMMM` total, so completeness is
    checked against that. The deletes below are otherwise unrecoverable.
    """
    print("Verifying all 15 groups are fully compacted ...")
    for split in SPLIT_DIRS:
        for tier in TIERS:
            split_dir = SPLIT_DIRS[split]
            idxs: set[int] = set()
            totals: set[int] = set()
            for item in list_repo_tree(f"{split_dir}/{tier}"):
                m = _DATA_RE.search(item.path)
                if m:
                    idxs.add(int(m.group(1)))
                    totals.add(int(m.group(2)))
            if len(totals) != 1:
                raise RuntimeError(
                    f"refusing to finalize — {split}/{tier}: {len(idxs)} compacted "
                    f"files, inconsistent/absent -of- totals {totals or set()}"
                )
            total = totals.pop()
            if idxs != set(range(total)):
                raise RuntimeError(
                    f"refusing to finalize — {split}/{tier}: only {len(idxs)}/{total} "
                    f"compacted files present"
                )
    print("All 15 groups verified complete. Deleting pre-compaction shards ...")
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
        "--workers", type=int, default=48, help="parallel bin-compaction processes"
    )
    parser.add_argument(
        "--slice-mb", type=int, default=35000, help="approx download slice size"
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
