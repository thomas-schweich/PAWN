"""Run `stockfish-datagen run` with incremental sync to a HuggingFace dataset.

Pod-friendly orchestrator. Three phases:

1. **Primer.** List the target HF dataset repo. Download the per-tier
   sentinels (`_manifest.json`, `_tier_state.json`) into the local output
   dir. For every remote shard file (`shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet`)
   create a *zero-byte placeholder* at the same local path. The rust
   binary's resume logic reads `(worker_id, chunk_idx, n_rows)` from
   filenames alone — no parquet metadata reads — so a directory full of
   placeholders looks identical to a directory full of real shards. This
   is what lets a fresh pod resume without re-downloading any data.

2. **Subprocess.** Spawn `stockfish-datagen run --config <cfg>`. Forward
   SIGINT/SIGTERM so graceful-shutdown semantics still hold.

3. **Watcher.** A daemon thread polls the output dir every
   `--poll-interval` seconds. Anything matching the canonical shard naming
   (or one of the sentinel filenames) that isn't already in the in-memory
   "uploaded" set gets uploaded via `huggingface_hub.HfApi.upload_file`.
   Zero-byte placeholders (size == 0) are skipped — those came from the
   primer and are already remote. After a successful upload, if
   `--prune-local` is set, the local file is replaced with a zero-byte
   placeholder so disk usage stays flat as the run grows.

Auth: needs HF_TOKEN in the env (standard pod setup). Repo is created
with `repo_type="dataset"` if it doesn't exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

LOG = logging.getLogger("datagen_sync")

# Names that are part of every tier directory. Any other file is ignored
# by the watcher (it'll only ever look at these and the shard pattern).
SENTINEL_FILES: tuple[str, ...] = ("_manifest.json", "_tier_state.json")

# Matches `shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet`. Mirrors the rust
# parser in `stockfish-datagen/src/resume.rs::parse_shard_filename`.
SHARD_RE = re.compile(r"^shard-w\d{3,}-c\d{4,}-r\d{6,}\.parquet$")


@dataclass(frozen=True)
class TierLayout:
    """One tier's local + remote paths."""
    name: str
    local_dir: Path
    repo_subdir: str  # always equal to `name`; kept for clarity at call sites


def load_run_config(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def expand_output_dir(cfg: dict) -> Path:
    """Apply the same `~`-expansion rule the rust binary uses."""
    p = Path(cfg["output_dir"])
    if str(p).startswith("~"):
        p = Path(os.path.expanduser(str(p)))
    return p


def tier_layouts(cfg: dict) -> list[TierLayout]:
    out_dir = expand_output_dir(cfg)
    return [
        TierLayout(name=t["name"], local_dir=out_dir / t["name"], repo_subdir=t["name"])
        for t in cfg["tiers"]
    ]


def ensure_repo(api: HfApi, repo_id: str) -> None:
    """Create the dataset repo if it doesn't exist; no-op otherwise."""
    try:
        api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        LOG.info("creating dataset repo %s", repo_id)
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)


def primer(api: HfApi, repo_id: str, tiers: list[TierLayout]) -> set[str]:
    """Download sentinels and create zero-byte placeholders for remote shards.

    Returns the set of repo paths (e.g. `nodes_0001/shard-w000-...parquet`)
    we already know are remote — used to seed the watcher's "uploaded" set
    so it doesn't try to re-upload them.
    """
    try:
        remote_files = api.list_repo_files(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        # Brand-new run: nothing to prime.
        return set()

    by_subdir: dict[str, list[str]] = {}
    for f in remote_files:
        # Only files under one of our tier subdirs are interesting.
        # Anything at the repo root (.gitattributes, README, etc.) we skip.
        if "/" not in f:
            continue
        subdir, name = f.split("/", 1)
        by_subdir.setdefault(subdir, []).append(name)

    primed: set[str] = set()
    for tier in tiers:
        files = by_subdir.get(tier.repo_subdir, [])
        if not files:
            continue
        tier.local_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            repo_path = f"{tier.repo_subdir}/{name}"
            local_path = tier.local_dir / name
            if name in SENTINEL_FILES:
                # Real file — download. Tiny, so this is cheap.
                LOG.info("primer: downloading sentinel %s", repo_path)
                _download_replace(api, repo_id, repo_path, local_path)
                primed.add(repo_path)
            elif SHARD_RE.match(name):
                # Placeholder — touch a zero-byte file with the same name.
                # If a real (non-placeholder) file is already there, leave
                # it alone — could be mid-upload or pre-existing.
                if not local_path.exists():
                    local_path.touch()
                primed.add(repo_path)
            else:
                # Anything else under the tier subdir we leave alone.
                pass
    return primed


def _download_replace(api: HfApi, repo_id: str, repo_path: str, local_path: Path) -> None:
    """Download `repo_path` from the dataset repo to `local_path` atomically."""
    try:
        downloaded = api.hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            # No local_dir; we want the cache, then we copy to the canonical spot.
        )
    except EntryNotFoundError:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".dl.tmp")
    # `hf_hub_download` returns a path inside the HF cache; copy contents.
    with open(downloaded, "rb") as src, open(tmp, "wb") as dst:
        dst.write(src.read())
    os.replace(tmp, local_path)


def _upload_one(api: HfApi, repo_id: str, repo_path: str, local_path: Path) -> None:
    """Upload one file. Retries on transient errors; raises on permanent ones."""
    delay = 2.0
    for attempt in range(5):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"sync: {repo_path}",
            )
            return
        except HfHubHTTPError as e:
            # 5xx / rate-limit: retry. 4xx (except 429): give up.
            status = e.response.status_code if e.response is not None else 0
            if status not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            LOG.warning(
                "upload of %s failed with HTTP %d; retrying in %.1fs (attempt %d/5)",
                repo_path, status, delay, attempt + 1,
            )
            time.sleep(delay)
            delay *= 2


def watcher_loop(
    api: HfApi,
    repo_id: str,
    tiers: list[TierLayout],
    uploaded: set[str],
    stop: threading.Event,
    poll_interval: float,
    prune_local: bool,
) -> None:
    """Poll forever until `stop` is set. Each cycle, upload anything new."""
    while not stop.is_set():
        try:
            _scan_and_upload(api, repo_id, tiers, uploaded, prune_local)
        except Exception:
            # Don't let watcher death silently stall the run. Log + continue;
            # if HF is down, individual upload retries already handle it.
            LOG.exception("watcher cycle failed; continuing")
        # Wait either the full poll interval, or until stop fires.
        stop.wait(poll_interval)


def _scan_and_upload(
    api: HfApi,
    repo_id: str,
    tiers: list[TierLayout],
    uploaded: set[str],
    prune_local: bool,
) -> None:
    for tier in tiers:
        if not tier.local_dir.exists():
            continue
        for entry in sorted(tier.local_dir.iterdir()):
            name = entry.name
            repo_path = f"{tier.repo_subdir}/{name}"
            if repo_path in uploaded:
                continue
            is_sentinel = name in SENTINEL_FILES
            is_shard = bool(SHARD_RE.match(name))
            if not (is_sentinel or is_shard):
                continue
            if not entry.is_file():
                continue
            size = entry.stat().st_size
            if size == 0:
                # Zero-byte placeholder from primer, or a tier_state in
                # mid-write. Skip until it has content.
                continue
            LOG.info("uploading %s (%s bytes)", repo_path, size)
            _upload_one(api, repo_id, repo_path, entry)
            uploaded.add(repo_path)
            # Sentinels stay local — the rust binary reads them on every
            # tier start. Only shards get pruned.
            if prune_local and is_shard:
                # Truncate to zero-byte placeholder so resume still finds it.
                with open(entry, "wb"):
                    pass


def run_subprocess(binary: str, config_path: Path) -> int:
    """Spawn the rust binary and forward signals."""
    proc = subprocess.Popen([binary, "run", "--config", str(config_path)])
    LOG.info("spawned %s (pid=%d)", binary, proc.pid)

    def _forward(signum: int, _frame: object) -> None:
        LOG.warning("got signal %d, forwarding to child", signum)
        proc.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _forward)

    return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to the stockfish-datagen JSON run config.")
    ap.add_argument("--repo-id", required=True,
                    help="HuggingFace dataset repo, e.g. `thomas-schweich/pawn-stockfish`.")
    ap.add_argument("--binary", default="stockfish-datagen",
                    help="Name (on PATH) or path to the stockfish-datagen binary.")
    ap.add_argument("--poll-interval", type=float, default=30.0,
                    help="Watcher poll interval in seconds (default: 30).")
    ap.add_argument("--prune-local", action="store_true",
                    help="After uploading a shard, replace local file with a "
                         "zero-byte placeholder. Recommended on disk-constrained pods.")
    ap.add_argument("--no-primer", action="store_true",
                    help="Skip the primer phase. Useful for local testing where "
                         "you don't want network calls before the first shard.")
    ap.add_argument("--log-level", default="INFO",
                    help="Logging level (default: INFO).")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load_run_config(args.config)
    tiers = tier_layouts(cfg)
    api = HfApi()

    ensure_repo(api, args.repo_id)

    if args.no_primer:
        primed: set[str] = set()
    else:
        LOG.info("priming from %s", args.repo_id)
        primed = primer(api, args.repo_id, tiers)
        LOG.info("primer: %d remote files known (placeholders + sentinels)", len(primed))

    stop = threading.Event()
    watcher = threading.Thread(
        target=watcher_loop,
        args=(api, args.repo_id, tiers, primed, stop, args.poll_interval, args.prune_local),
        name="hf-sync-watcher",
        daemon=True,
    )
    watcher.start()

    rc = run_subprocess(args.binary, args.config)
    LOG.info("rust binary exited with code %d", rc)

    # Give the watcher one final pass to drain anything written between
    # its last cycle and the binary's exit.
    LOG.info("final drain")
    try:
        _scan_and_upload(api, args.repo_id, tiers, primed, args.prune_local)
    except Exception:
        LOG.exception("final drain failed")
        # Still exit with the rust binary's code; partial sync is recoverable.

    stop.set()
    watcher.join(timeout=5.0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
