"""Run `stockfish-datagen run` with incremental sync to a HuggingFace dataset.

Pod-friendly orchestrator. Three phases:

1. **Primer.** List the target HF dataset repo. Download the per-tier
   sentinels (`_manifest.json`, `_tier_state.json`, and their per-pod
   variants `_manifest-s<A>-s<B>.json` etc) into the local output dir.
   For every remote shard file (`shard-s<NNNNNN>-r<NNNNNN>.parquet`)
   create a *zero-byte placeholder* at the same local path. The rust
   binary's resume logic reads `(shard_id, n_rows)` from filenames
   alone — no parquet metadata reads — so a directory full of
   placeholders looks identical to a directory full of real shards.
   This is what lets a fresh pod resume without re-downloading any data.

2. **Subprocess.** Spawn `stockfish-datagen run --config <cfg> [--tiers
   <subset>] [--shard-id-range <A:B>]`. Forward SIGINT/SIGTERM so
   graceful-shutdown semantics still hold.

3. **Watcher.** A daemon thread polls the output dir every
   `--poll-interval` seconds, with optional jitter so multiple pods do
   not hit HF in lockstep. New shard / sentinel files are committed in
   batched `huggingface_hub.create_commit` calls — one commit per tier
   per cycle for shards, plus one trailing commit for each new manifest.
   This keeps the per-repo commit count under HF's 128/hour limit even
   on cheap tiers that produce hundreds of shards per minute. HF 429s,
   529s, and transient 5xx responses are retried with server-directed
   delays when available and jittered backoff otherwise. Zero-byte
   placeholders (size == 0) are skipped — those came from the primer and
   are already remote. After a successful upload, if `--prune-local` is
   set, the local shard file is truncated to a zero-byte placeholder so
   disk usage stays flat as the run grows.

Auth: needs HF_TOKEN in the env (standard pod setup). Repo is created
with `repo_type="dataset"` if it doesn't exist.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    get_token,
)
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

LOG = logging.getLogger("datagen_sync")

# Per-tier sentinel filenames. The single-pod canonical names plus their
# per-pod-cooperation variants `_manifest-s<A>-s<B>.json` /
# `_tier_state-s<A>-s<B>.json` (where A,B are shard-id range bounds).
# Mirrors the rust filename layout in `stockfish-datagen/src/resume.rs`.
SENTINEL_RE = re.compile(
    r"^_(manifest|tier_state)(-s\d{6,}-s\d{6,})?\.json$"
)
# Manifest filenames specifically (canonical + per-pod variants). The
# orchestrator commits these AFTER shards so a tier-complete marker
# never appears remote without its shards.
MANIFEST_RE = re.compile(r"^_manifest(-s\d{6,}-s\d{6,})?\.json$")
# Tier-state filenames specifically (canonical + per-pod variants).
# Uploaded eagerly before / alongside shards so fresh-pod resume sees
# the sentinel.
TIER_STATE_RE = re.compile(r"^_tier_state(-s\d{6,}-s\d{6,})?\.json$")

# Matches `shard-s<NNNNNN>-r<NNNNNN>.parquet`. Mirrors the rust parser in
# `stockfish-datagen/src/resume.rs::parse_shard_filename`. The
# `{6,}` quantifier accepts wider fields for >999_999 shards or rows
# (rare on a 100M-game run with 2K shard size = 50K shards, still 5
# digits; defensive).
SHARD_RE = re.compile(r"^shard-s(?P<shard_id>\d{6,})-r\d{6,}\.parquet$")
RATELIMIT_RESET_RE = re.compile(r"(?:^|[;,])\s*t=(?P<seconds>\d+(?:\.\d+)?)")

# HF's documented rate limits use 429 for quota exhaustion, but in practice
# heavily-loaded upload endpoints can also return 529 ("service overloaded").
# Treat both as backoff-worthy so a pod fleet does not amplify a transient.
RETRYABLE_HF_STATUS_CODES = {429, 500, 502, 503, 504, 529}


@dataclass(frozen=True)
class HfRetryConfig:
    """Retry policy for HuggingFace commit calls."""

    max_attempts: int = 8
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 300.0
    overload_min_delay_seconds: float = 30.0


@dataclass(frozen=True)
class TierLayout:
    """One tier's local + remote paths."""
    name: str
    local_dir: Path


def load_run_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def expand_output_dir(cfg: dict[str, Any]) -> Path:
    """Tilde-expand `output_dir` so the orchestrator and the rust binary
    agree on the absolute path.

    The rust side's `expand_tilde` only handles `~/...` (no `~user/...`),
    while `Path.expanduser()` handles both. In practice configs only use
    `~/...`, so the two agree on the path the rust binary actually
    writes to.
    """
    return Path(cfg["output_dir"]).expanduser()


def tier_layouts(cfg: dict[str, Any]) -> list[TierLayout]:
    out_dir = expand_output_dir(cfg)
    return [
        TierLayout(name=t["name"], local_dir=out_dir / t["name"])
        for t in cfg["tiers"]
    ]


def ensure_repo(api: HfApi, repo_id: str) -> None:
    """Create the dataset repo if it doesn't exist; no-op otherwise."""
    try:
        api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        LOG.info("creating dataset repo %s", repo_id)
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)


def _header(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except AttributeError:
        return None
    return str(value) if value is not None else None


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP Retry-After value into seconds."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())


def _parse_ratelimit_reset(value: str | None) -> float | None:
    """Parse HF's RateLimit `t=<seconds until reset>` parameter."""
    if value is None:
        return None
    match = RATELIMIT_RESET_RE.search(value)
    if match is None:
        return None
    try:
        return max(0.0, float(match.group("seconds")))
    except ValueError:
        return None


def _hf_retry_delay_seconds(
    status: int,
    response: object | None,
    attempt: int,
    retry_config: HfRetryConfig,
) -> tuple[float, str]:
    """Return the non-jittered retry delay and the source of that delay."""
    retry_after = _parse_retry_after(_header(response, "retry-after"))
    if retry_after is not None:
        return min(retry_after, retry_config.max_delay_seconds), "Retry-After"

    if status == 429:
        ratelimit_reset = _parse_ratelimit_reset(_header(response, "ratelimit"))
        if ratelimit_reset is not None:
            return min(ratelimit_reset, retry_config.max_delay_seconds), "RateLimit"

    delay = retry_config.base_delay_seconds * (2 ** attempt)
    if status == 529:
        delay = max(delay, retry_config.overload_min_delay_seconds)
    return min(delay, retry_config.max_delay_seconds), "exponential backoff"


def _jitter_retry_delay(delay: float, source: str) -> float:
    """Jitter retry sleeps so many pods do not retry in lockstep."""
    if delay <= 0:
        return 0.0
    if source in {"Retry-After", "RateLimit"}:
        # Honor server-directed sleeps as a floor, then add a small spread.
        return delay + random.uniform(0.0, min(30.0, max(1.0, delay * 0.10)))
    return random.uniform(delay * 0.5, delay)


def _jitter_poll_interval(poll_interval: float, poll_jitter_ratio: float) -> float:
    if poll_interval <= 0 or poll_jitter_ratio <= 0:
        return max(0.0, poll_interval)
    spread = poll_interval * poll_jitter_ratio
    return max(0.0, poll_interval + random.uniform(-spread, spread))


def primer(
    api: HfApi, repo_id: str, tiers: list[TierLayout]
) -> tuple[set[str], dict[str, tuple[int, int]]]:
    """Download sentinels and create zero-byte placeholders for remote shards.

    Returns:
      * `uploaded_shards`: repo paths of remote shards (immutable; the
        watcher must never re-upload these).
      * `sentinel_state`: for each downloaded sentinel, the local
        `(size, mtime_ns)` snapshot taken right after the download. The
        watcher compares each cycle's local sentinel signature against
        this baseline and re-uploads when it differs — so when the rust
        binary rewrites `_manifest*.json` or `_tier_state*.json` (with
        updated row counts / completed_at, or with a refreshed n_games
        after extension), the new content actually reaches HF instead of
        being skipped as "already primed".
    """
    try:
        remote_files = api.list_repo_files(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        # Brand-new run: nothing to prime.
        return set(), {}

    by_subdir: dict[str, list[str]] = {}
    for f in remote_files:
        # Only files under one of our tier subdirs are interesting.
        # Anything at the repo root (.gitattributes, README, etc.) we skip.
        if "/" not in f:
            continue
        subdir, name = f.split("/", 1)
        by_subdir.setdefault(subdir, []).append(name)

    uploaded_shards: set[str] = set()
    sentinel_state: dict[str, tuple[int, int]] = {}
    for tier in tiers:
        files = by_subdir.get(tier.name, [])
        if not files:
            continue
        tier.local_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            repo_path = f"{tier.name}/{name}"
            local_path = tier.local_dir / name
            if SENTINEL_RE.match(name):
                # Real file — download. Tiny, so this is cheap. Per-pod
                # variants (`_manifest-s<A>-s<B>.json` etc) match here too,
                # so a multi-pod run's resume picks up every pod's state.
                LOG.info("primer: downloading sentinel %s", repo_path)
                _download_replace(api, repo_id, repo_path, local_path)
                st = local_path.stat()
                sentinel_state[repo_path] = (st.st_size, st.st_mtime_ns)
            elif SHARD_RE.match(name):
                # Placeholder — touch a zero-byte file with the same name.
                # If a real (non-placeholder) file is already there, leave
                # it alone — could be mid-upload or pre-existing.
                if not local_path.exists():
                    local_path.touch()
                uploaded_shards.add(repo_path)
    return uploaded_shards, sentinel_state


def _download_replace(api: HfApi, repo_id: str, repo_path: str, local_path: Path) -> None:
    """Download `repo_path` from the dataset repo to `local_path` atomically.

    Raises `EntryNotFoundError` if the file disappeared between `list_repo_files`
    and the download. We do NOT silently swallow that — if a sentinel
    (`_manifest.json` / `_tier_state.json`) was advertised by `list_repo_files`
    but vanished mid-primer, treating it as absent would cause the rust binary
    to either re-generate a completed tier or hit a fingerprint-drift abort.
    Loud failure surfaces the (rare) race so the operator can investigate.
    """
    downloaded = api.hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        repo_type="dataset",
        # No local_dir; we want the cache, then we copy to the canonical spot.
    )
    # `hf_hub_download` is declared `Union[str, DryRunFileInfo]`; with
    # dry_run=False (the default) it always returns the cache path string.
    # Use an explicit if-raise (not assert) so the check survives `python -O`,
    # which strips asserts and would let a non-str silently propagate to the
    # downstream `open()` and crash with a less actionable TypeError.
    if not isinstance(downloaded, str):
        raise RuntimeError(
            f"hf_hub_download returned non-str {type(downloaded).__name__} for {repo_path}"
        )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".dl.tmp")
    # `hf_hub_download` returns a path inside the HF cache; copy contents.
    with open(downloaded, "rb") as src, open(tmp, "wb") as dst:
        dst.write(src.read())
    os.replace(tmp, local_path)


def _upload_folder_batch(
    api: HfApi,
    repo_id: str,
    local_dir: Path,
    path_in_repo: str,
    files: list[str],
    commit_message: str,
    deletes: list[str] | None = None,
    retry_config: HfRetryConfig | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Commit a batch of files from `local_dir` as a single HF commit.

    Uses `huggingface_hub.HfApi.create_commit` with one
    `CommitOperationAdd` per named file — NOT `upload_folder`. Two
    reasons:

      1. **Safety.** `upload_folder` accepts `allow_patterns` which is
         interpreted as a `fnmatch` glob list, not a literal filename
         list. Our current naming scheme (`shard-s000001-r000010.parquet`,
         `_tier_state.json`) doesn't include glob metacharacters, but a
         future naming scheme or a manual file with `*` / `?` / `[` in
         its name would silently over-match. With `create_commit` each
         file is named explicitly — no glob, no over-match.

      2. **Perf.** `upload_folder` walks the entire folder via
         `folder_path.glob("**/*")` before applying `allow_patterns` —
         O(total_shards) `stat` calls per commit, even when only a
         few new shards exist. On a 50K-shard tier dir at 30s polls,
         this cost dominates the watcher's hot path. `create_commit`
         skips the walk entirely.

    **Safety invariant — DO NOT BREAK**: every file in `files` MUST
    exist locally with non-zero size at call time. The previous
    orchestrator's "upload then truncate" pattern silently destroyed
    data when a future sync rebroadcast the truncated zero-byte file
    over the populated remote (see ANALYSIS.md A4). The watcher's
    filtering already excludes zero-byte placeholders before they
    reach this function; the explicit guard below catches any race or
    refactor regression rather than letting it land as a silent
    data-loss commit. Failing loud here is the right call — the
    operator can re-run after fixing the root cause; a quiet
    overwrite is unrecoverable.

    Why batched commits at all (vs per-shard `upload_file`): HF caps
    commits at 128 / hour per repo. A 100M-game run produces ~45
    shards/min; per-shard commits exceed the limit on the very first
    cycle. Batched commits commit many shards atomically — one commit
    per tier per cycle.
    """
    if not files and not deletes:
        return
    # Defense in depth: refuse to ever upload a zero-byte file.
    operations: list[CommitOperationAdd | CommitOperationDelete] = []
    for name in files:
        path = local_dir / name
        try:
            size = path.stat().st_size
        except OSError as e:
            raise RuntimeError(
                f"upload_folder safety check: {path} stat failed ({e}); "
                f"refusing to commit a batch with a missing file"
            ) from e
        if size == 0:
            raise RuntimeError(
                f"upload_folder safety check: {path} is zero bytes; "
                f"refusing to overwrite remote {path_in_repo}/{name} with "
                f"empty content (would silently destroy already-uploaded data)"
            )
        operations.append(CommitOperationAdd(
            path_in_repo=f"{path_in_repo}/{name}",
            path_or_fileobj=str(path),
        ))
    # Deletes for superseded files (e.g. boundary-rewrite shards whose
    # filename changed because the row count grew). Bundled into the
    # same atomic commit as the adds so a downstream lister never sees
    # both the old and the new filename simultaneously.
    for repo_relative in deletes or []:
        operations.append(CommitOperationDelete(
            path_in_repo=f"{path_in_repo}/{repo_relative}",
        ))
    retry_config = retry_config or HfRetryConfig()
    for attempt in range(retry_config.max_attempts):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
            return
        except HfHubHTTPError as e:
            response = getattr(e, "response", None)
            status = response.status_code if response is not None else 0
            if (
                status not in RETRYABLE_HF_STATUS_CODES
                or attempt == retry_config.max_attempts - 1
            ):
                raise
            delay, delay_source = _hf_retry_delay_seconds(
                status, response, attempt, retry_config
            )
            jittered_delay = _jitter_retry_delay(delay, delay_source)
            LOG.warning(
                "folder commit (%s, %d files) failed with HTTP %d; "
                "retrying in %.1fs via %s (attempt %d/%d)",
                path_in_repo, len(files), status, jittered_delay, delay_source,
                attempt + 1, retry_config.max_attempts,
            )
            sleep(jittered_delay)


# Default consecutive-failure threshold; CLI-tunable via
# `--max-consecutive-failures`. At default poll=30s, 10 failures is
# ~5 min — long enough to ride out brief HF blips, short enough to
# fail loudly on permanent auth/quota breakage. Tune up for runs
# expected to span longer documented HF outages.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10


class WatcherFailed(Exception):
    """Raised by the watcher thread when consecutive cycle failures exceed
    the threshold. Stored on the watcher's `error` attribute and re-raised
    by the main thread after `join`."""


def watcher_loop(
    api: HfApi,
    repo_id: str,
    tiers: list[TierLayout],
    uploaded: set[str],
    sentinel_state: dict[str, tuple[int, int]],
    upload_count: list[int],
    stop: threading.Event,
    poll_interval: float,
    prune_local: bool,
    max_consecutive_failures: int,
    error_holder: list[BaseException],
    on_terminal_failure: Callable[[], None] | None = None,
    retry_config: HfRetryConfig | None = None,
    poll_jitter_ratio: float = 0.0,
) -> None:
    """Poll until `stop` is set. Stores any terminal exception in `error_holder`.

    On terminal failure (consecutive_failures >= max_consecutive_failures),
    invokes `on_terminal_failure` so the orchestrator can SIGTERM the rust
    binary — otherwise the main thread is stuck in `proc.wait()` and the
    rust binary keeps generating data that nobody syncs.
    """
    consecutive_failures = 0
    while not stop.is_set():
        try:
            _scan_and_upload(
                api, repo_id, tiers, uploaded, sentinel_state, upload_count,
                prune_local, retry_config=retry_config,
            )
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            LOG.exception(
                "watcher cycle failed (%d/%d consecutive)",
                consecutive_failures, max_consecutive_failures,
            )
            if consecutive_failures >= max_consecutive_failures:
                error_holder.append(WatcherFailed(
                    f"{consecutive_failures} consecutive watcher cycles failed; "
                    f"last error: {e!r}"
                ))
                stop.set()
                if on_terminal_failure is not None:
                    try:
                        on_terminal_failure()
                    except Exception:
                        LOG.exception("on_terminal_failure callback raised; swallowing")
                return
        # Wait either the jittered poll interval, or until stop fires.
        stop.wait(_jitter_poll_interval(poll_interval, poll_jitter_ratio))


def _scan_and_upload(
    api: HfApi,
    repo_id: str,
    tiers: list[TierLayout],
    uploaded: set[str],
    sentinel_state: dict[str, tuple[int, int]],
    upload_count: list[int],
    prune_local: bool,
    retry_config: HfRetryConfig | None = None,
) -> None:
    """Per cycle, per tier, commit any new local files in two folder
    uploads max:

      1. **State+shards commit.** New `_tier_state*.json` and new shard
         files in one atomic HF commit. Bundling tier_state with shards
         (instead of as its own commit) saves one commit per tier and is
         safe because the rust binary writes `_tier_state.json` before
         any shards exist — there's never a state file without matching
         shard data in the same cycle.

      2. **Manifest commit.** New `_manifest*.json` in a separate atomic
         commit AFTER the shards commit. Decoupling guarantees the
         tier-complete marker never appears remote without all referenced
         shards already remote, even if cycle 1's batch is interrupted.

    Zero-byte placeholders (size == 0) are skipped — those came from the
    primer and are already remote. Manifest with no preceding shards
    commit (because no new shards in this cycle) commits on its own as
    cycle 2.

    Shards (`uploaded`) are immutable: a `repo_path in uploaded` short
    circuit is correct. Sentinels (tier_state / manifest) ARE mutable —
    the rust binary rewrites them with updated counts at end-of-run and
    on tier-extension — so they're tracked separately by
    `(size, mtime_ns)` in `sentinel_state`. A sentinel whose local
    signature differs from the recorded one re-enters the upload batch
    so the new content reaches HF.
    """
    for tier in tiers:
        if not tier.local_dir.exists():
            continue

        # Build a `shard_id -> uploaded-filename` map for THIS tier so we
        # can detect boundary-rewrite supersession: if the rust runner
        # produces a `shard-sNNN-rNEW.parquet` (different row count, same
        # shard id as a previously-uploaded `shard-sNNN-rOLD.parquet`),
        # we issue a `CommitOperationDelete` for the old filename in the
        # same atomic commit as the add. Otherwise both copies live on
        # HF and downstream listers see both, double-counting games.
        prefix = f"{tier.name}/"
        uploaded_by_shard_id: dict[int, str] = {}
        for repo_path in uploaded:
            if not repo_path.startswith(prefix):
                continue
            name = repo_path[len(prefix):]
            m = SHARD_RE.match(name)
            if m:
                uploaded_by_shard_id[int(m.group("shard_id"))] = name

        # Bucket new local files by category.
        new_shards_and_state: list[str] = []
        new_manifests: list[str] = []
        superseded_shards: list[str] = []  # OLD filenames to delete remote-side
        for entry in tier.local_dir.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            repo_path = f"{tier.name}/{name}"
            st = entry.stat()
            if st.st_size == 0:
                # Placeholder or mid-write atomic rename — skip until it
                # has content.
                continue
            shard_match = SHARD_RE.match(name)
            if shard_match:
                if repo_path in uploaded:
                    continue
                shard_id = int(shard_match.group("shard_id"))
                old_name = uploaded_by_shard_id.get(shard_id)
                if old_name is not None and old_name != name:
                    # Boundary-rewrite: same shard_id, different rows.
                    # The rust runner already `remove_file`'d the local
                    # OLD; the remote copy is what we need to retire.
                    superseded_shards.append(old_name)
                new_shards_and_state.append(name)
            elif TIER_STATE_RE.match(name):
                sig = (st.st_size, st.st_mtime_ns)
                if sentinel_state.get(repo_path) == sig:
                    continue
                new_shards_and_state.append(name)
            elif MANIFEST_RE.match(name):
                sig = (st.st_size, st.st_mtime_ns)
                if sentinel_state.get(repo_path) == sig:
                    continue
                new_manifests.append(name)
            # else: not ours — ignore.

        if new_shards_and_state or superseded_shards:
            LOG.info(
                "tier %s: committing %d new file(s) (shards + tier_state)%s",
                tier.name, len(new_shards_and_state),
                f" + deleting {len(superseded_shards)} superseded shard(s)"
                if superseded_shards else "",
            )
            _upload_folder_batch(
                api, repo_id, tier.local_dir,
                path_in_repo=tier.name,
                files=new_shards_and_state,
                deletes=superseded_shards,
                commit_message=f"sync {tier.name}: {len(new_shards_and_state)} file(s)"
                + (f", {len(superseded_shards)} superseded" if superseded_shards else ""),
                retry_config=retry_config,
            )
            for name in new_shards_and_state:
                repo_path = f"{tier.name}/{name}"
                upload_count[0] += 1
                if SHARD_RE.match(name):
                    uploaded.add(repo_path)
                    # Prune local shards (truncate to 0 bytes) after the
                    # commit lands. Sentinels stay local — the rust binary
                    # reads them on every tier start.
                    if prune_local:
                        with open(tier.local_dir / name, "wb"):
                            pass
                else:
                    # Sentinel: record the (size, mtime_ns) we just
                    # uploaded so the next cycle skips it iff unchanged.
                    st = (tier.local_dir / name).stat()
                    sentinel_state[repo_path] = (st.st_size, st.st_mtime_ns)
            for old_name in superseded_shards:
                uploaded.discard(f"{tier.name}/{old_name}")

        if new_manifests:
            LOG.info(
                "tier %s: committing %d new manifest file(s)",
                tier.name, len(new_manifests),
            )
            _upload_folder_batch(
                api, repo_id, tier.local_dir,
                path_in_repo=tier.name,
                files=new_manifests,
                commit_message=f"manifest {tier.name}",
                retry_config=retry_config,
            )
            for name in new_manifests:
                repo_path = f"{tier.name}/{name}"
                upload_count[0] += 1
                st = (tier.local_dir / name).stat()
                sentinel_state[repo_path] = (st.st_size, st.st_mtime_ns)


def install_signal_handlers(proc_holder: list["subprocess.Popen[bytes]"]) -> None:
    """Install SIGINT/SIGTERM forwarders that send the signal to `proc_holder[0]`.

    Installed BEFORE Popen so a signal arriving in the race window
    doesn't kill the parent without forwarding. If a signal fires
    before `proc_holder` is populated, we exit with the conventional
    128+signum so the caller's framework still sees a "terminated by
    signal" outcome — there's no child to clean up.
    """
    def _forward(signum: int, _frame: object) -> None:
        if not proc_holder:
            LOG.warning("got signal %d before child spawn; exiting", signum)
            sys.exit(128 + signum)
        LOG.warning("got signal %d, forwarding to child", signum)
        proc_holder[0].send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _forward)


def run_subprocess(
    binary: str,
    config_path: Path,
    proc_holder: list["subprocess.Popen[bytes]"],
    tiers: str | None = None,
    shard_id_range: str | None = None,
) -> int:
    """Spawn the rust binary into `proc_holder` (already-installed handlers).

    Optionally forwards `--tiers` and `--shard-id-range` to the rust
    binary for multi-pod cooperation.

    Signal handlers must already be installed when this is called.
    `proc_holder.append(subprocess.Popen(...))` shrinks but does not
    fully close the race between the child existing and the holder
    being populated. The defense is the `not proc_holder` check inside
    `_forward` exiting with `128 + signum`.
    """
    cmd = [binary, "run", "--config", str(config_path)]
    if tiers is not None:
        cmd.extend(["--tiers", tiers])
    if shard_id_range is not None:
        cmd.extend(["--shard-id-range", shard_id_range])
    proc_holder.append(subprocess.Popen(cmd))
    proc = proc_holder[0]
    LOG.info("spawned %s (pid=%d) %s", binary, proc.pid, " ".join(cmd[1:]))
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
    ap.add_argument("--poll-jitter-ratio", type=float, default=0.20,
                    help="Randomly vary each watcher sleep by this fraction "
                         "of --poll-interval (default: 0.20). This keeps "
                         "multiple pods from polling/uploading in lockstep.")
    ap.add_argument("--prune-local", action="store_true",
                    help="After uploading a shard, replace local file with a "
                         "zero-byte placeholder. Recommended on disk-constrained pods.")
    ap.add_argument("--no-primer", action="store_true",
                    help="Skip the primer phase (no `list_repo_files` + sentinel "
                         "downloads). The cheap repo existence check still runs. "
                         "Useful for local testing or restarting against a known-empty repo.")
    ap.add_argument("--max-consecutive-failures", type=int,
                    default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
                    help="Watcher gives up + kills the rust child after this "
                         "many consecutive cycle failures (default: %(default)s; "
                         "tune up for runs that may span longer HF outages).")
    ap.add_argument("--hf-commit-retries", type=int,
                    default=HfRetryConfig.max_attempts,
                    help="Max attempts for each HuggingFace commit batch "
                         "(default: %(default)s). Retries cover 429, 529, "
                         "and transient 5xx responses.")
    ap.add_argument("--hf-retry-max-delay", type=float,
                    default=HfRetryConfig.max_delay_seconds,
                    help="Cap for a single HuggingFace retry sleep in seconds "
                         "(default: %(default)s).")
    ap.add_argument("--hf-overload-min-delay", type=float,
                    default=HfRetryConfig.overload_min_delay_seconds,
                    help="Minimum non-header retry delay for HTTP 529 overload "
                         "responses, before jitter (default: %(default)s).")
    ap.add_argument("--watcher-drain-timeout-hours", type=float, default=4.0,
                    help="After the rust binary exits, how long to wait for the "
                         "watcher to drain the backlog of locally-staged shards "
                         "before giving up and exiting with code 4 "
                         "(default: %(default)s). Pod will not exit until "
                         "either the backlog drains or this timeout fires.")
    ap.add_argument("--tiers", default=None,
                    help="Comma-separated tier indices to forward to the rust "
                         "binary's --tiers flag (e.g. `0,1` or `3`). Default: "
                         "all tiers in the config. Use for multi-pod runs "
                         "where each pod handles a tier subset.")
    ap.add_argument("--shard-id-range", default=None,
                    help="Half-open shard-id range A:B to forward to the rust "
                         "binary (e.g. `0:5000`). Default: full tier range. "
                         "Use to fan a single tier across multiple pods.")
    ap.add_argument("--log-level", default="INFO",
                    help="Logging level (default: INFO).")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Validate `--shard-id-range` format before any HF calls. Otherwise an
    # invalid input (e.g. `500-1000` with a dash, or `:5000` missing the
    # start) only fails inside the rust binary's clap parser — seconds
    # after this script's HF auth, primer, etc., with an error message
    # that points back to the rust binary, not the orchestrator flag.
    if args.shard_id_range is not None:
        if not re.fullmatch(r"\d+:\d+", args.shard_id_range):
            LOG.error(
                "FATAL: --shard-id-range must be A:B with non-negative integers "
                "(e.g. `0:5000`); got %r. Refusing to start the rust binary.",
                args.shard_id_range,
            )
            return 2
    if args.poll_jitter_ratio < 0:
        LOG.error("FATAL: --poll-jitter-ratio must be >= 0; got %s", args.poll_jitter_ratio)
        return 2
    if args.hf_commit_retries < 1:
        LOG.error("FATAL: --hf-commit-retries must be >= 1; got %s", args.hf_commit_retries)
        return 2
    if args.hf_retry_max_delay <= 0:
        LOG.error("FATAL: --hf-retry-max-delay must be > 0; got %s", args.hf_retry_max_delay)
        return 2
    if args.hf_overload_min_delay < 0:
        LOG.error(
            "FATAL: --hf-overload-min-delay must be >= 0; got %s",
            args.hf_overload_min_delay,
        )
        return 2

    retry_config = HfRetryConfig(
        max_attempts=args.hf_commit_retries,
        max_delay_seconds=args.hf_retry_max_delay,
        overload_min_delay_seconds=args.hf_overload_min_delay,
    )

    # Two-part fail-fast: (a) no token at all, (b) token present but rejected.
    # Either case must fire before the rust binary spawns — running for hours
    # only to discover at the end that nothing was uploaded would be the
    # worst possible failure mode for this tool.
    #
    # `get_token()` is the canonical helper: it checks env vars (HF_TOKEN
    # / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_HUB_TOKEN) AND the on-disk
    # credential store written by `hf auth login`, in the same order the
    # rest of huggingface_hub uses at request time. Don't use
    # `HfApi().token` — that only inspects env, not the disk-cached login.
    token = get_token()
    if token is None:
        LOG.error(
            "FATAL: no HF token found in env (HF_TOKEN / HUGGING_FACE_HUB_TOKEN "
            "/ HUGGINGFACE_HUB_TOKEN) or local credential store "
            "(`hf auth login`). Set HF_TOKEN before running this orchestrator. "
            "Refusing to start the rust binary — running data generation "
            "without a working sync target would silently waste pod hours."
        )
        return 2

    cfg = load_run_config(args.config)
    tiers = tier_layouts(cfg)
    # Pass the resolved token explicitly to make the connection between
    # the fail-fast check and the API client visible. Without this, HfApi
    # would re-resolve via `get_token()` itself — same end result, but
    # the wiring is implicit.
    api = HfApi(token=token)

    # Token-validity preflight. `whoami()` is the canonical "this token works"
    # call — returns user info on success, raises 401 on a bad token. Doing
    # this BEFORE `ensure_repo` and BEFORE spawning the rust binary means a
    # revoked / expired / typo'd token surfaces in seconds rather than as an
    # opaque 401 stack trace from inside an upload retry chain hours later.
    try:
        whoami = api.whoami()
        LOG.info("authenticated to HuggingFace as %s", whoami.get("name", "<unknown>"))
    except HfHubHTTPError as e:
        # Same defensive guard as `_upload_folder_batch`: `e.response` is documented
        # always-present on HfHubHTTPError but typed Optional.
        response = getattr(e, "response", None)
        status = response.status_code if response is not None else 0
        if status == 401:
            LOG.error(
                "FATAL: HF token is set but rejected by HuggingFace (HTTP 401 "
                "from /api/whoami). Token is likely expired, revoked, or "
                "mistyped. Refusing to start the rust binary."
            )
            return 2
        # Any other status (network blip, 5xx) we propagate — this is
        # before any expensive work, so a clean exit with a stack trace
        # is fine and tells the operator something they need to know.
        raise

    # We always touch the repo (cheap: one info-or-create round-trip)
    # before launching the rust binary. `HfApi.upload_file` does NOT
    # auto-create missing repos, so without this the first watcher
    # upload would 404, count toward the failure threshold, and
    # eventually SIGTERM the rust child — leaving the user with a
    # confusing "watcher failed" exit on what was actually a missing
    # repo. `--no-primer` only suppresses the heavier `list_repo_files`
    # + sentinel-download phase, not the repo existence guarantee.
    ensure_repo(api, args.repo_id)

    if args.no_primer:
        LOG.info("--no-primer: skipping primer phase (no remote listing or sentinel download)")
        primed: set[str] = set()
        sentinel_state: dict[str, tuple[int, int]] = {}
    else:
        LOG.info("priming from %s", args.repo_id)
        primed, sentinel_state = primer(api, args.repo_id, tiers)
        LOG.info(
            "primer: %d remote shards known, %d sentinel(s) downloaded",
            len(primed), len(sentinel_state),
        )

    # Signal handlers must be installed BEFORE Popen — the proc_holder
    # is shared between the handler closure (forwards signals to the
    # child), the watcher (kills the child on terminal failure), and
    # the subprocess runner (populates it).
    proc_holder: list[subprocess.Popen[bytes]] = []
    install_signal_handlers(proc_holder)

    def _kill_child_on_watcher_failure() -> None:
        """Forwarded to the watcher so a permanent HF failure terminates
        the rust binary instead of letting it run for hours producing
        unsynced data.

        Race-tolerant: between `poll()` returning None and `terminate()`
        firing, the rust child can finish on its own and the main thread
        can reap it via `proc.wait()`. After the reap the PID may be
        recycled, in which case `terminate()` would SIGTERM an unrelated
        process. We swallow `ProcessLookupError` (already-reaped) and
        the broader case where terminate fails because the child is
        gone — those are benign races, not failures worth surfacing."""
        if not proc_holder:
            return
        proc = proc_holder[0]
        if proc.poll() is not None:
            return  # child already exited; main thread will reap.
        LOG.error("watcher gave up; sending SIGTERM to rust binary")
        try:
            proc.terminate()
        except ProcessLookupError:
            # Child finished between our poll() and terminate() — fine.
            LOG.debug("rust binary exited before SIGTERM landed; race benign")
        except Exception:
            LOG.exception("failed to terminate rust binary; will continue")

    stop = threading.Event()
    watcher_error: list[BaseException] = []
    upload_count = [0]
    watcher = threading.Thread(
        target=watcher_loop,
        args=(api, args.repo_id, tiers, primed, sentinel_state, upload_count, stop,
              args.poll_interval, args.prune_local,
              args.max_consecutive_failures, watcher_error,
              _kill_child_on_watcher_failure, retry_config,
              args.poll_jitter_ratio),
        name="hf-sync-watcher",
        daemon=True,
    )
    watcher.start()

    rc = run_subprocess(
        args.binary,
        args.config,
        proc_holder,
        tiers=args.tiers,
        shard_id_range=args.shard_id_range,
    )
    LOG.info("rust binary exited with code %d", rc)

    # Stop and join the watcher BEFORE the final drain. Otherwise the
    # watcher's last cycle and the main thread's drain race on the same
    # `uploaded` set and on prune-local's truncate-after-upload step —
    # both threads could be mid-upload of the same shard, and one could
    # truncate the local file while the other is still reading it.
    #
    # Critical pod-exit guarantee: when the rust binary finishes, there's
    # almost always a backlog of locally-staged shards that the watcher
    # hasn't gotten to yet (cheap tiers can produce shards faster than HF
    # can commit them). We MUST wait for the watcher to drain that backlog
    # before exiting, or those shards die with the pod.
    #
    # The watcher will exit at the top of its next iteration after
    # `stop.set()`, but it has to finish its current `_scan_and_upload`
    # cycle first — which uploads everything currently visible. With a
    # big backlog this cycle can take many minutes. We give it up to
    # `--watcher-drain-timeout-hours` hours (default 4); long enough for
    # tens of thousands of shards even at HF's worst-case ~2s/commit. If
    # the watcher hasn't finished by then, HF is broken or our retry
    # logic is malformed; we skip the drain (rather than racing it) and
    # exit with code 4 so the operator sees the partial-sync state.
    stop.set()
    drain_timeout_s = args.watcher_drain_timeout_hours * 3600.0
    LOG.info(
        "stopping watcher; waiting up to %.1fh for backlog to drain "
        "(%d files uploaded so far)",
        args.watcher_drain_timeout_hours, upload_count[0],
    )
    watcher.join(timeout=drain_timeout_s)
    drained_ok = True
    if watcher.is_alive():
        LOG.error(
            "watcher still alive after %.1fh of drain attempt — uploaded %d "
            "files so far. Either HF is extraordinarily slow or our retry "
            "logic is broken. SKIPPING final drain to avoid racing in-flight "
            "upload; any remaining locally-staged shards will be lost when "
            "the pod terminates. Re-run with a larger "
            "--watcher-drain-timeout-hours if HF is just slow.",
            args.watcher_drain_timeout_hours, upload_count[0],
        )
        drained_ok = False
    else:
        LOG.info("watcher exited cleanly; running final drain")
        try:
            _scan_and_upload(
                api, args.repo_id, tiers, primed, sentinel_state, upload_count,
                args.prune_local, retry_config=retry_config,
            )
        except Exception:
            LOG.exception("final drain failed")
            drained_ok = False

    LOG.info(
        "sync summary: %d files uploaded this run; %d remote files known via "
        "primer; final drain ok=%s; watcher_error=%s",
        upload_count[0], len(primed), drained_ok,
        watcher_error[0] if watcher_error else None,
    )

    if watcher_error:
        # Watcher gave up after consecutive failures (auth / quota /
        # permanent 4xx). Surface that as a non-zero exit so the pod
        # doesn't appear to have completed successfully.
        #
        # Always 3, regardless of `rc`: when the watcher SIGTERMs the
        # rust child, `proc.wait()` returns a negative value (-15),
        # which Python translates to exit code 241 — pod monitoring
        # would see "weird signal" instead of the documented "watcher
        # failed" semantic. The watcher failure is the *root cause*
        # we want to surface; the SIGTERM-induced rust exit is
        # consequence, not signal.
        LOG.error("watcher terminated abnormally: %r", watcher_error[0])
        return 3
    if not drained_ok:
        return rc if rc != 0 else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
