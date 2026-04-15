#!/usr/bin/env python3
"""Rename extracted shards to HF canonical `<split>-NNNNN-of-NNNNN.parquet`.

Uses `git mv` on a pointer-only clone (`GIT_LFS_SKIP_SMUDGE=1`) so only
~130-byte LFS pointers are pulled and the real parquet blobs never move.
The resulting push is a metadata-only tree rewrite.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

from huggingface_hub import HfApi, get_token


def run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="thomas-schweich/pawn-lichess-full")
    p.add_argument("--revision", default="searchless_vocab_512")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api = HfApi()
    files = api.list_repo_files(args.repo, repo_type="dataset", revision=args.revision)
    shards = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))

    by_split: dict[str, list[str]] = defaultdict(list)
    for f in shards:
        name = f[len("data/"):]
        split = name.split("-", 1)[0]
        by_split[split].append(f)
    for split in by_split:
        by_split[split].sort()

    total = sum(len(v) for v in by_split.values())
    print(f"found {total} shards across splits: {sorted(by_split)}")
    for s, v in sorted(by_split.items()):
        print(f"  {s}: {len(v)}")

    rename_plan: list[tuple[str, str]] = []
    for split, paths in sorted(by_split.items()):
        n = len(paths)
        for i, old in enumerate(paths):
            new = f"data/{split}-{i:05d}-of-{n:05d}.parquet"
            if old != new:
                rename_plan.append((old, new))

    print(f"{len(rename_plan)} paths to rename")
    if args.dry_run:
        for old, new in rename_plan[:20]:
            print(f"  {old} -> {new}")
        if len(rename_plan) > 20:
            print(f"  ... +{len(rename_plan)-20} more")
        return 0

    if not rename_plan:
        print("nothing to do")
        return 0

    token = get_token()
    if not token:
        print("ERROR: no HF token available", file=sys.stderr)
        return 1

    tmpdir = tempfile.mkdtemp(prefix="pawn-rename-")
    try:
        repo_url = f"https://user:{token}@huggingface.co/datasets/{args.repo}"
        clone_dir = os.path.join(tmpdir, "repo")
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"

        print("cloning (pointer-only)...")
        run([
            "git", "clone",
            "--branch", args.revision,
            "--depth", "1",
            "--single-branch",
            repo_url, clone_dir,
        ], env=env)

        run(["git", "config", "user.email", "pawn-extract@local"], cwd=clone_dir)
        run(["git", "config", "user.name", "pawn-extract"], cwd=clone_dir)

        print(f"applying {len(rename_plan)} renames...")
        for old, new in rename_plan:
            subprocess.run(
                ["git", "mv", old, new], cwd=clone_dir, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )

        run([
            "git", "commit", "-m",
            "Rename shards to HF canonical <split>-NNNNN-of-NNNNN.parquet",
        ], cwd=clone_dir)

        print("pushing...")
        run(["git", "push", "origin", args.revision], cwd=clone_dir, env=env)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
