#!/usr/bin/env python3
"""Split a Parquet dataset into train/val/test and upload to HuggingFace.

Shuffles deterministically, splits 90/5/5, and uploads with HF-compatible
naming (data/train-*.parquet, data/val-*.parquet, data/test-*.parquet).

Usage:
    # Split local parquet files and upload
    python scripts/split_dataset.py \
        --input /dev/shm/lichess/*.parquet \
        --hf-repo thomas-schweich/lichess-1800-1900 \
        --seed 42

    # Split a single file
    python scripts/split_dataset.py \
        --input data/stockfish-nodes1/data/nodes_0001.parquet \
        --hf-repo thomas-schweich/stockfish-nodes1 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Split Parquet dataset into train/val/test")
    p.add_argument("--input", nargs="+", required=True, help="Input parquet file(s)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Local output directory (default: /dev/shm/split)")
    p.add_argument("--hf-repo", type=str, default=None,
                   help="Upload to this HuggingFace dataset repo")
    p.add_argument("--train-frac", type=float, default=0.90)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test_frac = 1.0 - args.train_frac - args.val_frac
    assert test_frac > 0, f"train+val fracs must be < 1.0, got {args.train_frac + args.val_frac}"

    output_dir = Path(args.output_dir) if args.output_dir else Path("/dev/shm/split")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all input files
    print(f"Loading {len(args.input)} file(s)...")
    tables = [pq.read_table(f) for f in args.input]
    table = pa.concat_tables(tables)
    n = len(table)
    print(f"  Total rows: {n:,}")

    # Deterministic shuffle
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n)

    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": table.take(train_idx),
        "val": table.take(val_idx),
        "test": table.take(test_idx),
    }

    print(f"  Split: train={n_train:,} val={n_val:,} test={n_test:,}")

    # Write split files
    paths = {}
    for name, split_table in splits.items():
        out_path = output_dir / f"{name}.parquet"
        pq.write_table(split_table, out_path, compression="zstd")
        size_mb = out_path.stat().st_size / 1e6
        paths[name] = out_path
        print(f"  Wrote {out_path} ({size_mb:.1f} MB, {len(split_table):,} rows)")

    # Upload to HuggingFace
    if args.hf_repo:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()
        try:
            create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"  Repo note: {e}")

        # Delete old data/ files first to avoid mixing old and new
        try:
            existing = api.list_repo_files(args.hf_repo, repo_type="dataset")
            old_data = [f for f in existing if f.startswith("data/")]
            for f in old_data:
                api.delete_file(f, args.hf_repo, repo_type="dataset")
                print(f"  Deleted old: {f}")
        except Exception:
            pass

        for name, out_path in paths.items():
            repo_path = f"data/{name}-00000-of-00001.parquet"
            print(f"  Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo=repo_path,
                repo_id=args.hf_repo,
                repo_type="dataset",
            )

        print(f"\n  Dataset: https://huggingface.co/datasets/{args.hf_repo}")

    print("\nDone.")


if __name__ == "__main__":
    main()
