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

Disk footprint is tiny. The unit of work is a *bin* — the set of small
shards that compact into one ~500 MB output file. Each bin is downloaded,
compacted, and its raw shards deleted immediately; compacted files are held
only until an upload batch fills. Peak working set is roughly
`--upload-batch` output files (~13 GB at the defaults) — point `--workdir`
at `/dev/shm` and run on any pod with a few hundred GB of RAM.

Transfers go through the `hf` CLI (download / upload). The Hub metadata
calls — listing the repo tree, and the final history squash — use the
huggingface_hub Python API: single requests, nothing to be inefficient at.

Upload batching keeps the commit count low: ~2,000 output files / 25 per
batch ~= 80 commits, well under HF's 128-commits/hour dataset limit.

Prereqs on the pod:
    curl -LsSf https://hf.co/cli/install.sh | bash          # `hf` CLI
    export HF_TOKEN=...                                     # write token
    uv run --with pyarrow --with huggingface_hub python scripts/compact_stockfish_dataset.py ...

Workflow:
    # 1. process every group (idempotent / resumable — re-run if interrupted)
    python scripts/compact_stockfish_dataset.py --workdir /dev/shm/compact
    # 2. inspect the verification output, then finalize (IRREVERSIBLE)
    python scripts/compact_stockfish_dataset.py --workdir /dev/shm/compact --finalize
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

REPO = "thomas-schweich/pawn-stockfish-100m"

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
# expected rows (= games) per tier, per split — coverage cross-check
EXPECTED_ROWS: dict[str, int] = {
    "train": 19_900_000,
    "validation": 50_000,
    "test": 50_000,
}

# Old eval-column names -> new names, per tier family.
RENAME_SEARCH: dict[str, str] = {
    "static_legal_move_evals": "nnue_evals",
    "legal_move_evals": "cp_evals",
}
RENAME_SEARCHLESS: dict[str, str] = {"legal_move_evals": "nnue_evals"}
# On the searchless tier `static_legal_move_evals` is all-null and redundant.
DROP_SEARCHLESS: list[str] = ["static_legal_move_evals"]

api = HfApi()


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
    """(repo_path, size) for every parquet shard in a (split, tier) group.

    A Hub metadata call — no file content is transferred.
    """
    out: list[tuple[str, int]] = []
    for item in api.list_repo_tree(
        REPO, path_in_repo=f"{split_dir}/{tier}", repo_type="dataset", recursive=False
    ):
        size = getattr(item, "size", None)
        if size is not None and item.path.endswith(".parquet"):
            out.append((item.path, size))
    out.sort()
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
    `rowgroup_bytes` of input before being flushed as one row group, so peak
    Arrow memory is a few hundred MB regardless of output file size.
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


def upload_batch(stage_dir: Path, split_dir: str, tier: str, label: str) -> None:
    """Upload everything staged for one group in a single commit, then prune."""
    run(
        [
            "hf", "upload", REPO, str(stage_dir), f"{split_dir}/{tier}",
            "--type", "dataset",
            "--commit-message", f"compact {split_dir}/{tier}: {label}",
        ]
    )
    for parquet in stage_dir.glob("*.parquet"):
        parquet.unlink()


def process_group(args: argparse.Namespace, split: str, tier: str) -> None:
    """Bin -> (download, compact, prune) per bin -> batched upload, one group."""
    workdir = Path(args.workdir)
    done_marker = workdir / ".done" / f"{split}__{tier}"
    if done_marker.exists():
        print(f"[{split}/{tier}] already done — skipping")
        return

    split_dir = SPLIT_DIRS[split]
    bins = plan_bins(
        list_group_shards(split_dir, tier), args.target_mb * 1024 * 1024
    )
    total = len(bins)
    print(f"\n=== {split}/{tier}: {total} output files ===", flush=True)

    progress_file = workdir / ".progress" / f"{split}__{tier}"
    done_bins, cum_rows = 0, 0
    if progress_file.exists():
        done_bins, cum_rows = (int(x) for x in progress_file.read_text().split())
        print(f"  resuming after bin {done_bins}")

    raw_root = workdir / "raw"
    stage_dir = workdir / "stage" / split_dir / tier
    stage_dir.mkdir(parents=True, exist_ok=True)
    rowgroup_bytes = args.rowgroup_mb * 1024 * 1024
    staged_since_upload = 0

    for idx in range(done_bins, total):
        bin_names = bins[idx]
        includes: list[str] = []
        for name in bin_names:
            includes += ["--include", name]
        run(
            [
                "hf", "download", REPO, "--type", "dataset", *includes,
                "--local-dir", str(raw_root),
                "--max-workers", str(args.download_workers),
            ]
        )

        shard_paths = [raw_root / name for name in bin_names]
        out_path = stage_dir / f"data-{idx:05d}-of-{total:05d}.parquet"
        rows_in, rows_out = compact_bin(shard_paths, out_path, tier, rowgroup_bytes)
        if rows_in != rows_out:
            raise RuntimeError(
                f"row-count mismatch {split}/{tier} bin {idx}: "
                f"in={rows_in} out={rows_out}"
            )
        cum_rows += rows_out
        for shard in shard_paths:  # prune raw immediately — bounds disk
            shard.unlink()
        staged_since_upload += 1

        is_last = idx == total - 1
        if staged_since_upload >= args.upload_batch or is_last:
            upload_batch(stage_dir, split_dir, tier, f"bins ..{idx}/{total}")
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            progress_file.write_text(f"{idx + 1} {cum_rows}")
            staged_since_upload = 0
            print(f"  uploaded through bin {idx + 1}/{total}")

    expected = EXPECTED_ROWS[split]
    if cum_rows != expected:
        raise RuntimeError(
            f"coverage check failed {split}/{tier}: "
            f"compacted {cum_rows:,} rows, expected {expected:,}"
        )
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.touch()
    print(f"[{split}/{tier}] done — {cum_rows:,} rows verified")


def finalize() -> None:
    """Delete the old shard-s* files, then squash history to reclaim storage."""
    print("Deleting pre-compaction shards (shard-s*.parquet) ...")
    run(
        [
            "hf", "repos", "delete-files", REPO, "**/shard-s*.parquet",
            "--type", "dataset",
            "--commit-message", "drop pre-compaction 2k-game shards",
        ]
    )
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
        "--target-mb", type=int, default=500, help="approx output file size"
    )
    parser.add_argument(
        "--rowgroup-mb", type=int, default=128, help="approx row-group size"
    )
    parser.add_argument(
        "--upload-batch",
        type=int,
        default=25,
        help="output files per commit (peak disk ~= this x --target-mb)",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--tiers", nargs="+", choices=TIERS, default=TIERS, help="subset of tiers"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="delete old shards + squash history (run only after verifying)",
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
