"""Reconcile per-pod tier manifests into a unified `_manifest.json`.

After a multi-pod cooperative run (`stockfish-datagen run --shard-id-range
A:B`), each pod writes its own `_manifest-s<A>-s<B>.json` covering the
shard ids it owned. This script walks the HF dataset repo, collects all
per-pod manifests for one tier, verifies they:

  1. Cover the full `[0, total_shards)` range with no gaps and no overlaps.
  2. Share the same `config_fingerprint` (i.e. were all generated from
     the same per-tier config).

…then commits a unified `_manifest.json` for that tier whose `shards`
list is the sorted concat of every pod's shards and whose
`n_games_written` is the sum.

Usage:
    python scripts/datagen_reconcile_tier.py \\
        --config stockfish-datagen/examples/stockfish_100m.json \\
        --repo-id thomas-schweich/pawn-stockfish \\
        --tier nodes_0001

Pass `--all-tiers` to reconcile every tier in the config (skipping any
that already have a canonical `_manifest.json` remotely).

Exits non-zero on any gap / overlap / fingerprint mismatch — those are
operator-visible problems that need investigation, not silent acceptance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token

LOG = logging.getLogger("datagen_reconcile")

# Per-pod manifest filename: `_manifest-s<A>-s<B>.json` where A,B are
# zero-padded to at least 6 digits. Mirrors the rust filename layout in
# `stockfish-datagen/src/resume.rs::TierManifest::path`.
PER_POD_MANIFEST_RE = re.compile(
    r"^_manifest-s(?P<start>\d{6,})-s(?P<end>\d{6,})\.json$"
)
CANONICAL_MANIFEST = "_manifest.json"


@dataclass(frozen=True)
class PerPodManifest:
    repo_path: str
    start: int
    end: int
    fingerprint: str
    n_games_written: int
    shards: list[str]
    completed_at: str


def load_run_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def total_shards(tier_cfg: dict[str, Any], shard_size_games: int) -> int:
    return math.ceil(tier_cfg["n_games"] / shard_size_games)


def fetch_remote_manifests(
    api: HfApi, repo_id: str, tier_name: str
) -> list[PerPodManifest]:
    """List the tier's directory in the remote repo, download each
    per-pod manifest, return parsed contents."""
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        raise SystemExit(f"failed to list repo {repo_id}: {e}") from e

    pod_paths: list[tuple[str, int, int]] = []  # (repo_path, start, end)
    prefix = f"{tier_name}/"
    for f in files:
        if not f.startswith(prefix):
            continue
        name = f[len(prefix):]
        m = PER_POD_MANIFEST_RE.match(name)
        if not m:
            continue
        pod_paths.append((f, int(m.group("start")), int(m.group("end"))))

    out: list[PerPodManifest] = []
    for repo_path, start, end in pod_paths:
        local = api.hf_hub_download(repo_id=repo_id, filename=repo_path, repo_type="dataset")
        # Explicit if-raise (not assert) so the type narrowing survives
        # `python -O`, which strips asserts and would otherwise let a
        # `DryRunFileInfo` slip through to `open()`.
        if not isinstance(local, str):
            raise RuntimeError(
                f"hf_hub_download returned non-str {type(local).__name__} for {repo_path}"
            )
        with open(local, "r") as fh:
            payload = json.load(fh)
        out.append(PerPodManifest(
            repo_path=repo_path,
            start=start,
            end=end,
            fingerprint=payload["config_fingerprint"],
            n_games_written=payload["n_games_written"],
            shards=payload["shards"],
            completed_at=payload["completed_at"],
        ))
    out.sort(key=lambda m: m.start)
    return out


def validate_ranges(manifests: list[PerPodManifest], total: int) -> None:
    """Fail loudly if the manifests don't tile `[0, total)` exactly."""
    if not manifests:
        raise SystemExit("no per-pod manifests found for this tier")
    expected = 0
    for m in manifests:
        # Zero-length ranges (`_manifest-s005000-s005000.json`) can arise
        # from a misconfigured pod that passed `--shard-id-range` with
        # `start >= tier total` and got clamped to an empty range. The
        # tiling check below would silently accept them (start == end ==
        # expected, no advance). Reject explicitly so the operator sees
        # the stray manifest rather than reconciling around it.
        if m.end <= m.start:
            raise SystemExit(
                f"empty range manifest {m.repo_path} (start={m.start}, "
                f"end={m.end}); a pod was launched with a shard-id-range "
                f"that clamped to zero work — investigate before reconciling"
            )
        if m.start != expected:
            if m.start < expected:
                raise SystemExit(
                    f"overlap or out-of-order: manifest {m.repo_path} starts at "
                    f"{m.start} but previous coverage ended at {expected}"
                )
            raise SystemExit(
                f"gap in shard coverage: previous manifest ended at "
                f"{expected}, next ({m.repo_path}) starts at {m.start}"
            )
        expected = m.end
    if expected != total:
        raise SystemExit(
            f"coverage incomplete: union of per-pod manifests covers "
            f"[0, {expected}) but tier has {total} shards"
        )


def validate_fingerprints(manifests: list[PerPodManifest]) -> str:
    """Verify every per-pod manifest shares the same fingerprint; return it."""
    fingerprints = {m.fingerprint for m in manifests}
    if len(fingerprints) != 1:
        raise SystemExit(
            f"per-pod manifests disagree on fingerprint: "
            f"{sorted(fingerprints)}; reconciliation refuses to merge "
            f"shards generated under different configs"
        )
    return fingerprints.pop()


def reconcile_one(
    api: HfApi,
    repo_id: str,
    tier_name: str,
    tier_cfg: dict[str, Any],
    shard_size_games: int,
    dry_run: bool,
) -> None:
    total = total_shards(tier_cfg, shard_size_games)
    LOG.info("reconciling tier %s (expected %d total shards)", tier_name, total)

    manifests = fetch_remote_manifests(api, repo_id, tier_name)
    LOG.info("found %d per-pod manifest(s) for %s", len(manifests), tier_name)
    validate_ranges(manifests, total)
    fingerprint = validate_fingerprints(manifests)

    merged_shards: list[str] = []
    n_games = 0
    completed_at = ""
    for m in manifests:
        merged_shards.extend(m.shards)
        n_games += m.n_games_written
        if m.completed_at > completed_at:
            completed_at = m.completed_at
    merged_shards.sort()
    # De-dup is defensive; the range-tiling check above already guarantees
    # uniqueness, but a corrupted per-pod manifest that double-lists a
    # shard would otherwise produce a degenerate unified manifest.
    if len(set(merged_shards)) != len(merged_shards):
        raise SystemExit(
            f"duplicate shard filenames in merged manifest for {tier_name}; "
            f"per-pod manifests reference overlapping shards"
        )

    unified = {
        "tier_name": tier_name,
        "config_fingerprint": fingerprint,
        "n_games_written": n_games,
        "shards": merged_shards,
        "completed_at": completed_at,
    }

    if dry_run:
        LOG.info(
            "DRY RUN: would commit %s/%s with %d shards / %d games, "
            "then delete %d per-pod manifest(s)",
            repo_id, f"{tier_name}/{CANONICAL_MANIFEST}",
            len(merged_shards), n_games, len(manifests),
        )
        return

    repo_path = f"{tier_name}/{CANONICAL_MANIFEST}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(unified, tf, indent=2)
        tmp_path = tf.name
    try:
        LOG.info(
            "committing unified %s (%d shards / %d games)",
            repo_path, len(merged_shards), n_games,
        )
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(
                f"reconcile {tier_name}: merge {len(manifests)} per-pod manifest(s)"
            ),
        )
    finally:
        # Always clean up the temp file, success or failure. Without this,
        # repeated failed reconcile invocations leak ~1 file per attempt
        # into /tmp until the pod is destroyed.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Clean up the per-pod manifests now that the unified one is committed.
    # Without this, the orchestrator's primer would re-download every
    # per-pod manifest on the next pod startup (SENTINEL_RE matches both
    # the canonical and per-pod variants), and a re-run of reconcile
    # would either merge the same set again (idempotent) or trip on a
    # mix of stale and fresh per-pod state if the run was extended.
    for m in manifests:
        try:
            api.delete_file(
                path_in_repo=m.repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=(
                    f"reconcile {tier_name}: remove per-pod manifest "
                    f"{m.repo_path.rsplit('/', 1)[-1]}"
                ),
            )
        except Exception as e:  # noqa: BLE001 — log every failure mode
            LOG.warning(
                "failed to delete per-pod manifest %s after unified commit: %s; "
                "the unified manifest is authoritative, but the dangling per-pod "
                "file may confuse a future reconcile",
                m.repo_path, e,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to the stockfish-datagen JSON run config.")
    ap.add_argument("--repo-id", required=True,
                    help="HuggingFace dataset repo holding the per-pod manifests.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tier", help="Tier name to reconcile (e.g. `nodes_0001`).")
    g.add_argument("--all-tiers", action="store_true",
                   help="Reconcile every tier in the config; skip tiers that "
                        "already have a remote `_manifest.json`.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate and print, don't commit.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    token = get_token()
    if token is None:
        LOG.error("no HF token found; set HF_TOKEN or `hf auth login` first")
        return 2

    api = HfApi(token=token)
    cfg = load_run_config(args.config)
    shard_size_games = int(cfg.get("shard_size_games", 10_000))

    if args.all_tiers:
        try:
            remote_files = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
        except Exception as e:
            LOG.error("failed to list repo: %s", e)
            return 3
        for tier_cfg in cfg["tiers"]:
            name = tier_cfg["name"]
            if f"{name}/{CANONICAL_MANIFEST}" in remote_files:
                LOG.info("tier %s already has canonical manifest; skipping", name)
                continue
            try:
                reconcile_one(api, args.repo_id, name, tier_cfg, shard_size_games, args.dry_run)
            except SystemExit as e:
                LOG.error("tier %s failed: %s", name, e)
                return 4
    else:
        # Single tier; fail fast if the operator named a nonexistent tier.
        tiers_by_name = {t["name"]: t for t in cfg["tiers"]}
        if args.tier not in tiers_by_name:
            LOG.error(
                "tier %s not in config (available: %s)",
                args.tier, sorted(tiers_by_name),
            )
            return 2
        # Catch SystemExit consistently with the --all-tiers branch so a
        # validation failure (gap/overlap/fingerprint mismatch) returns
        # the documented exit code 4 rather than Python's default 1 for
        # an unhandled `SystemExit("message")` — CI/monitoring that's
        # keyed on the documented codes would otherwise misclassify.
        try:
            reconcile_one(
                api, args.repo_id, args.tier, tiers_by_name[args.tier],
                shard_size_games, args.dry_run,
            )
        except SystemExit as e:
            LOG.error("tier %s failed: %s", args.tier, e)
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
