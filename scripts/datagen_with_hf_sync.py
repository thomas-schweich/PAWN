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
from typing import Any, Callable

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

LOG = logging.getLogger("datagen_sync")

# Names that are part of every tier directory. Any other file is ignored
# by the watcher (it'll only ever look at these and the shard pattern).
SENTINEL_FILES: tuple[str, ...] = ("_manifest.json", "_tier_state.json")

# Upload-order priority: lower number first.
#   `_tier_state.json` is the FIRST upload — fresh-pod resume requires
#       it to exist remotely so the primer downloads it before the rust
#       binary scans the tier dir; without it `run_tier` aborts with
#       "shards exist but no tier state file is present".
#   shard files are the BULK — uploaded in any order between sentinels.
#   `_manifest.json` is the LAST upload — it's the tier-complete marker;
#       seeing it remotely without all referenced shards already remote
#       would make the next primer skip the tier (silent data loss).
#
# `_tier_state.json` happens to be alphabetically after `_manifest.json`,
# which is why we can't rely on lex sort.
def _upload_priority(name: str) -> int:
    if name == "_tier_state.json":
        return 0
    if name == "_manifest.json":
        return 2
    return 1  # shards (and anything else, though only shards reach here)

# Matches `shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet`. Mirrors the rust
# parser in `stockfish-datagen/src/resume.rs::parse_shard_filename`.
SHARD_RE = re.compile(r"^shard-w\d{3,}-c\d{4,}-r\d{6,}\.parquet$")


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
        files = by_subdir.get(tier.name, [])
        if not files:
            continue
        tier.local_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            repo_path = f"{tier.name}/{name}"
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
    return primed


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
    assert isinstance(downloaded, str), (
        f"hf_hub_download returned non-str {type(downloaded).__name__} for {repo_path}"
    )
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
            #
            # `e.response` is documented as always-set on HfHubHTTPError, but
            # the field is `Optional[Response]` in the type stubs. A library
            # update or test-monkeypatched error path could violate the
            # invariant; guarding here lets us log + give up cleanly instead
            # of `AttributeError`-tripping the consecutive-failure threshold.
            response = getattr(e, "response", None)
            status = response.status_code if response is not None else 0
            if status not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            LOG.warning(
                "upload of %s failed with HTTP %d; retrying in %.1fs (attempt %d/5)",
                repo_path, status, delay, attempt + 1,
            )
            time.sleep(delay)
            delay *= 2


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
    upload_count: list[int],
    stop: threading.Event,
    poll_interval: float,
    prune_local: bool,
    max_consecutive_failures: int,
    error_holder: list[BaseException],
    on_terminal_failure: Callable[[], None] | None = None,
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
            _scan_and_upload(api, repo_id, tiers, uploaded, upload_count, prune_local)
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
        # Wait either the full poll interval, or until stop fires.
        stop.wait(poll_interval)


def _scan_and_upload(
    api: HfApi,
    repo_id: str,
    tiers: list[TierLayout],
    uploaded: set[str],
    upload_count: list[int],
    prune_local: bool,
) -> None:
    for tier in tiers:
        if not tier.local_dir.exists():
            continue
        # Strict per-cycle upload order: tier_state → shards → manifest.
        # See `_upload_priority` for the rationale on each anchor.
        entries = list(tier.local_dir.iterdir())
        entries.sort(key=lambda e: (_upload_priority(e.name), e.name))
        for entry in entries:
            name = entry.name
            repo_path = f"{tier.name}/{name}"
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
                # Zero-byte placeholder from primer (or a sentinel in
                # mid-write — atomic-rename means we'll see it next cycle
                # at full size). Skip until it has content.
                continue
            LOG.info("uploading %s (%s bytes)", repo_path, size)
            _upload_one(api, repo_id, repo_path, entry)
            uploaded.add(repo_path)
            upload_count[0] += 1
            # Sentinels stay local — the rust binary reads them on every
            # tier start. Only shards get pruned.
            if prune_local and is_shard:
                # Truncate to zero-byte placeholder so resume still finds it.
                with open(entry, "wb"):
                    pass


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
    binary: str, config_path: Path, proc_holder: list["subprocess.Popen[bytes]"],
) -> int:
    """Spawn the rust binary into `proc_holder` (already-installed handlers).

    Signal handlers must already be installed when this is called.
    `proc_holder.append(subprocess.Popen(...))` shrinks but does not
    fully close the race between the child existing and the holder
    being populated — Python evaluates the inner `Popen(...)` first
    (child spawns), then calls `.append(result)`. A signal delivered
    between those two bytecode steps would find `proc_holder` empty
    and orphan the freshly-spawned child. The window is nanoseconds in
    practice; the defense is the `not proc_holder` check inside
    `_forward` exiting with `128 + signum`, which still produces the
    conventional shell exit even though the child leaks. Truly
    closing the race needs `posix_spawn` (atomic) or holding off
    signals around Popen via `signal.pthread_sigmask`, neither of
    which we currently bother with.
    """
    proc_holder.append(subprocess.Popen([binary, "run", "--config", str(config_path)]))
    proc = proc_holder[0]
    LOG.info("spawned %s (pid=%d)", binary, proc.pid)
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
                    help="Skip the primer phase (no `list_repo_files` + sentinel "
                         "downloads). The cheap repo existence check still runs. "
                         "Useful for local testing or restarting against a known-empty repo.")
    ap.add_argument("--max-consecutive-failures", type=int,
                    default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
                    help="Watcher gives up + kills the rust child after this "
                         "many consecutive cycle failures (default: %(default)s; "
                         "tune up for runs that may span longer HF outages).")
    ap.add_argument("--watcher-drain-timeout-hours", type=float, default=4.0,
                    help="After the rust binary exits, how long to wait for the "
                         "watcher to drain the backlog of locally-staged shards "
                         "before giving up and exiting with code 4 "
                         "(default: %(default)s). Pod will not exit until "
                         "either the backlog drains or this timeout fires.")
    ap.add_argument("--log-level", default="INFO",
                    help="Logging level (default: INFO).")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
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
        # Same defensive guard as `_upload_one`: `e.response` is documented
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
    else:
        LOG.info("priming from %s", args.repo_id)
        primed = primer(api, args.repo_id, tiers)
        LOG.info("primer: %d remote files known (placeholders + sentinels)", len(primed))

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
        args=(api, args.repo_id, tiers, primed, upload_count, stop,
              args.poll_interval, args.prune_local,
              args.max_consecutive_failures, watcher_error,
              _kill_child_on_watcher_failure),
        name="hf-sync-watcher",
        daemon=True,
    )
    watcher.start()

    rc = run_subprocess(args.binary, args.config, proc_holder)
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
            _scan_and_upload(api, args.repo_id, tiers, primed, upload_count, args.prune_local)
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
