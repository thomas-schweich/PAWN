"""Reconcile per-pod tier manifests into a unified `_manifest.json`.

After a multi-pod cooperative run (`stockfish-datagen run --shard-id-range
A:B`), each pod writes its own `_manifest-s<A>-s<B>.json` covering the
shard ids it owned. This script walks the HF dataset repo, collects all
per-pod manifests for one tier, verifies they:

  1. Cover the full `[0, total_shards)` range with no gaps and no overlaps.
  2. Share the same `config_fingerprint` (i.e. were all generated from
     the same per-tier config).

…then commits a unified canonical manifest covering the entire tier under
`_meta/<tier_name>/_manifest.json` whose `shards` list is the sorted concat
of every pod's shards and whose `n_games_written` is the sum.

The canonical manifest lives under `_meta/` rather than in the tier
directory itself because HF (the underlying git layer) caps any directory
at 10000 files. A tier configured for 10000 shards has zero room for
in-directory sidecars after the last shard lands, so keeping canonical
state in a sibling tree keeps the tier directories pure-parquet and
fully utilizes the 10K budget. Per-pod manifests still live in the tier
directory during a run (they're deleted by this script post-merge), so a
tier that pushes near 10000 shards while multiple pods are writing
sidecars can still hit the wall — see the operational notes in the
project's reconciliation README. Future schema bump (v4) will move
per-pod files into `_meta/` too.

Usage:
    python scripts/datagen_reconcile_tier.py \\
        --config stockfish-datagen/examples/stockfish_100m.json \\
        --repo-id thomas-schweich/pawn-stockfish \\
        --tier nodes_0001

Pass `--all-tiers` to reconcile every tier in the config (skipping any
that already have a canonical manifest at either the new `_meta/<tier>/`
location or the legacy `<tier>/` location).

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

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    get_token,
)

LOG = logging.getLogger("datagen_reconcile")

# Per-pod manifest filename: `_manifest-s<A>-s<B>.json` where A,B are
# zero-padded to at least 6 digits. Mirrors the rust filename layout in
# `stockfish-datagen/src/resume.rs::TierManifest::path`.
PER_POD_MANIFEST_RE = re.compile(
    r"^_manifest-s(?P<start>\d{6,})-s(?P<end>\d{6,})\.json$"
)
PER_POD_TIER_STATE_RE = re.compile(
    r"^_tier_state-s(?P<start>\d{6,})-s(?P<end>\d{6,})\.json$"
)
CANONICAL_MANIFEST = "_manifest.json"
CANONICAL_TIER_STATE = "_tier_state.json"

# Canonical manifests + states are written under a sibling `_meta/` tree
# rather than into the tier directory itself. HF (and the underlying git
# layer) enforces a hard 10000-files-per-directory limit; a tier configured
# for 10000 shards has no room for extra sidecars after the last shard
# lands. Hoisting them to `_meta/<tier>/...` keeps the tier directory
# pure-parquet and lets every tier hit 10000/10000 cleanly. See
# `feedback_hf_commit_rate_limit.md` (memory) for the empirical 100M-run
# diagnosis that produced this layout.
META_PREFIX = "_meta"


def canonical_manifest_path(tier_name: str) -> str:
    return f"{META_PREFIX}/{tier_name}/{CANONICAL_MANIFEST}"


def canonical_tier_state_path(tier_name: str) -> str:
    return f"{META_PREFIX}/{tier_name}/{CANONICAL_TIER_STATE}"


@dataclass(frozen=True)
class PerPodManifest:
    repo_path: str
    start: int
    end: int
    fingerprint: str
    n_games_written: int
    shards: list[str]
    completed_at: str


@dataclass(frozen=True)
class PerPodTierState:
    repo_path: str
    start: int
    end: int
    fingerprint: str
    n_games: int
    started_at: str


def load_run_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def total_shards(tier_cfg: dict[str, Any], shard_size_games: int) -> int:
    return math.ceil(tier_cfg["n_games"] / shard_size_games)


def _list_repo_files(api: HfApi, repo_id: str) -> list[str]:
    try:
        return api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        raise SystemExit(f"failed to list repo {repo_id}: {e}") from e


def _download_json(api: HfApi, repo_id: str, repo_path: str) -> dict[str, Any]:
    local = api.hf_hub_download(repo_id=repo_id, filename=repo_path, repo_type="dataset")
    # Explicit if-raise (not assert) so the type narrowing survives
    # `python -O`, which strips asserts and would otherwise let a
    # `DryRunFileInfo` slip through to `open()`.
    if not isinstance(local, str):
        raise RuntimeError(
            f"hf_hub_download returned non-str {type(local).__name__} for {repo_path}"
        )
    with open(local, "r") as fh:
        return json.load(fh)


def fetch_remote_manifests(
    api: HfApi, repo_id: str, tier_name: str
) -> list[PerPodManifest]:
    """List the tier's directory in the remote repo, download each
    per-pod manifest, return parsed contents."""
    files = _list_repo_files(api, repo_id)

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
        payload = _download_json(api, repo_id, repo_path)
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


def fetch_remote_tier_states(
    api: HfApi, repo_id: str, tier_name: str
) -> list[PerPodTierState]:
    """Same as `fetch_remote_manifests` but for `_tier_state-s<A>-s<B>.json`
    sentinels. We need these to construct a canonical `_tier_state.json`
    after reconciliation — without one, a later unscoped run that grows
    `tier.n_games` would hit `TierState::load(..., None) -> None` and
    abort with "shards exist but no tier state file is present" because
    the per-pod state files were also deleted by reconcile."""
    files = _list_repo_files(api, repo_id)
    prefix = f"{tier_name}/"
    pod_paths: list[tuple[str, int, int]] = []
    for f in files:
        if not f.startswith(prefix):
            continue
        name = f[len(prefix):]
        m = PER_POD_TIER_STATE_RE.match(name)
        if not m:
            continue
        pod_paths.append((f, int(m.group("start")), int(m.group("end"))))

    out: list[PerPodTierState] = []
    for repo_path, start, end in pod_paths:
        payload = _download_json(api, repo_id, repo_path)
        out.append(PerPodTierState(
            repo_path=repo_path,
            start=start,
            end=end,
            fingerprint=payload["config_fingerprint"],
            n_games=payload.get("n_games", 0),
            started_at=payload["started_at"],
        ))
    out.sort(key=lambda s: s.start)
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

    states = fetch_remote_tier_states(api, repo_id, tier_name)
    LOG.info("found %d per-pod tier-state(s) for %s", len(states), tier_name)
    # Sanity-check states. They're optional fallbacks if missing (the run
    # might predate the n_games-in-state field), but if present they must
    # all agree on fingerprint and n_games.
    if states:
        state_fps = {s.fingerprint for s in states}
        if state_fps != {fingerprint}:
            raise SystemExit(
                f"per-pod tier states disagree with manifests on fingerprint: "
                f"manifests={fingerprint!r}, states={sorted(state_fps)!r}"
            )
        state_ngs = {s.n_games for s in states if s.n_games > 0}
        if len(state_ngs) > 1:
            raise SystemExit(
                f"per-pod tier states disagree on tier-level n_games: "
                f"{sorted(state_ngs)}; reconciliation refuses to merge"
            )

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

    unified_manifest = {
        "tier_name": tier_name,
        "config_fingerprint": fingerprint,
        "n_games_written": n_games,
        "shards": merged_shards,
        "completed_at": completed_at,
    }
    # Build a canonical `_tier_state.json` from the per-pod states (or
    # synthesize one if no per-pod states exist, e.g. legacy reconcile).
    # Without this, a later UNSCOPED run that grows `tier.n_games` would
    # fall through the manifest-skip check, find shards on disk, then
    # abort because `TierState::load(..., None)` returns None (the
    # per-pod state files get deleted below, leaving no sentinel for a
    # canonical-scope reader to find). The canonical state lets the
    # documented "grow n_games" flow work after multi-pod reconciliation.
    canonical_n_games = max(
        (s.n_games for s in states if s.n_games > 0),
        default=int(tier_cfg["n_games"]),
    )
    earliest_started_at = min((s.started_at for s in states), default=completed_at)
    canonical_state = {
        "config_fingerprint": fingerprint,
        "started_at": earliest_started_at,
        "n_games": canonical_n_games,
        # shard_range omitted — canonical sentinels never carry one.
    }

    if dry_run:
        LOG.info(
            "DRY RUN: would commit canonical %s + %s (%d shards / %d games), "
            "then delete %d per-pod manifest(s) and %d per-pod tier-state(s)",
            canonical_manifest_path(tier_name),
            canonical_tier_state_path(tier_name),
            len(merged_shards), n_games, len(manifests), len(states),
        )
        return

    # Phase 1: write canonical manifest + canonical state to local tempfiles,
    # then commit both in ONE atomic create_commit. Atomicity guarantees a
    # downstream reader never sees the canonical manifest without the
    # canonical state (or vice versa) — both appear together or not at all.
    manifest_tmp: str | None = None
    state_tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(unified_manifest, tf, indent=2)
            manifest_tmp = tf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(canonical_state, tf, indent=2)
            state_tmp = tf.name
        LOG.info(
            "committing canonical manifest + tier_state for %s (%d shards / %d games)",
            tier_name, len(merged_shards), n_games,
        )
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo=canonical_manifest_path(tier_name),
                    path_or_fileobj=manifest_tmp,
                ),
                CommitOperationAdd(
                    path_in_repo=canonical_tier_state_path(tier_name),
                    path_or_fileobj=state_tmp,
                ),
            ],
            commit_message=(
                f"reconcile {tier_name}: merge {len(manifests)} per-pod sentinel(s)"
            ),
        )
    finally:
        # Always clean up the temp files, success or failure. Without this,
        # repeated failed reconcile invocations leak ~2 files per attempt
        # into /tmp until the pod is destroyed.
        for p in (manifest_tmp, state_tmp):
            if p is None:
                continue
            try:
                os.unlink(p)
            except OSError:
                pass

    # Phase 2: delete per-pod manifests + per-pod states in one batched
    # commit. Two commits total per reconcile, regardless of pod count.
    delete_ops: list[CommitOperationDelete] = [
        CommitOperationDelete(path_in_repo=m.repo_path) for m in manifests
    ] + [
        CommitOperationDelete(path_in_repo=s.repo_path) for s in states
    ]
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=delete_ops,
            commit_message=(
                f"reconcile {tier_name}: remove {len(manifests)} per-pod manifest(s) "
                f"+ {len(states)} per-pod state(s)"
            ),
        )
    except Exception as e:  # noqa: BLE001 — log every failure mode
        LOG.warning(
            "failed to delete per-pod sentinels after canonical commit: %s; "
            "the canonical files are authoritative, but the dangling per-pod "
            "files may confuse a future reconcile. Manual cleanup: %s",
            e,
            ", ".join([m.repo_path for m in manifests] + [s.repo_path for s in states]),
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
                        "already have a canonical manifest at either "
                        "`_meta/<tier>/_manifest.json` (current) or "
                        "`<tier>/_manifest.json` (legacy).")
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
            # Treat either the new sibling-`_meta/` location OR the
            # legacy in-tier location as "already reconciled" so older
            # datasets with `<tier>/_manifest.json` aren't re-reconciled
            # and pushed into a duplicate location.
            if canonical_manifest_path(name) in remote_files:
                LOG.info("tier %s already has canonical manifest at %s; skipping",
                         name, canonical_manifest_path(name))
                continue
            if f"{name}/{CANONICAL_MANIFEST}" in remote_files:
                LOG.info("tier %s has legacy canonical manifest at %s/%s; skipping",
                         name, name, CANONICAL_MANIFEST)
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
