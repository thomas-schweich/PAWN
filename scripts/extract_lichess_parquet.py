#!/usr/bin/env python3
"""Extract the Lichess parquet dataset from raw monthly PGN dumps.

Streams raw monthly PGN dumps from `thomas-schweich/raw-lichess` directly
(no full local copy of the .zst), parses each game via the Rust chess
engine, and incrementally uploads each parquet shard to the destination
dataset repo as soon as it finishes — deleting the local copy immediately
after a successful upload. Disk usage is bounded to roughly one in-flight
shard per worker.

Parallelism
-----------
Months run concurrently via a spawn-context process pool (`--jobs N`). One
training month or one holdout month = one work unit. Each worker opens its
own HF stream, its own parser, and its own uploader. Commits to the dest repo
are interleaved across workers; HfApi handles optimistic concurrency.

Resume
------
Each completed work unit writes a sentinel file `data/_complete/<kind>-<YYYY-MM>.done`
to the dest repo. On startup the script lists the repo, skips any unit whose
sentinel already exists, and atomically deletes any orphaned partial shards
for units that didn't finish (so the rerun starts clean). Pass `--force` to
ignore sentinels and re-extract every requested unit.

Branching
---------
By default the script writes every shard, sentinel, and cleanup commit to a
`run/extract` branch (created automatically). `main` stays untouched until
you squash-merge the branch once you're satisfied. Override with
`--revision main` to write directly to the primary branch, or any other
branch name to run multiple extractions in parallel.

Shard names include the source month so a mid-run crash in month M never
collides with shards from month M+1: `data/<split>-<YYYY-MM>-<NNNN>.parquet`.

Standalone: imports only `chess_engine`, `polars`, `numpy`, `zstandard`, and
`huggingface_hub`. No dependency on the `pawn` Python package.

Games longer than `--max-ply` are truncated; their outcome is classified as
PLY_LIMIT by the Rust parser. Default is 512, which keeps natural outcomes
for the vast majority of Lichess games.

Usage
-----

    uv run python scripts/extract_lichess_parquet.py \\
        --months 2025-01 2025-02 2025-03 \\
        --holdout-month 2026-01 \\
        --dest-repo thomas-schweich/pawn-lichess \\
        --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import chess_engine
import numpy as np
import polars as pl
from huggingface_hub import CommitOperationDelete, HfApi, HfFileSystem
from huggingface_hub.errors import RepositoryNotFoundError


HF_RAW_BUCKET = "thomas-schweich/raw-lichess"
HF_RAW_FILENAME_TEMPLATE = "lichess_{year_month}.pgn.zst"
DEFAULT_MAX_PLY = 512
DEFAULT_BATCH_SIZE = 500_000
DEFAULT_SHARD_SIZE = 1_000_000
DEFAULT_JOBS = 4
DEFAULT_THREADS_PER_WORKER = 2
DEFAULT_REVISION = "run/extract"
DEFAULT_SCRATCH_DIR = Path("/dev/shm/pawn-lichess-extract")


def log(msg: str, prefix: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    tag = f"[{prefix}] " if prefix else ""
    print(f"[{ts}] {tag}{msg}", flush=True)


# ---------------------------------------------------------------------------
# Run config / work units (picklable across spawn boundary)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkUnit:
    kind: str  # "train" or "holdout"
    year_month: str

    @property
    def tag(self) -> str:
        return f"{self.kind}/{self.year_month}"

    @property
    def sentinel_path(self) -> str:
        return f"data/_complete/{self.kind}-{self.year_month}.done"


@dataclass(frozen=True)
class RunConfig:
    dest_repo: str
    revision: str
    max_ply: int
    batch_size: int
    shard_size: int
    scratch_dir: Path
    max_games_per_month: int | None
    val_days: tuple[int, int]
    test_days: tuple[int, int]


# ---------------------------------------------------------------------------
# Streaming PGN ingest
# ---------------------------------------------------------------------------

def open_hf_stream(year_month: str) -> Any:
    """Open a monthly Lichess zstd PGN dump from HF as a byte stream.

    The raw dumps live in a HuggingFace *bucket*, not a dataset repo, so
    the path is `buckets/<namespace>/<name>/...`. Returns an `fsspec` file
    handle backed by ranged HTTP requests — bytes are pulled on demand, so
    the full compressed dump never lands on disk.
    """
    filename = HF_RAW_FILENAME_TEMPLATE.format(year_month=year_month)
    hf_path = f"buckets/{HF_RAW_BUCKET}/{filename}"
    fs = HfFileSystem()
    return fs.open(hf_path, "rb")


def stream_pgn_games(fileobj: Any, batch_size: int) -> Iterator[tuple[str, int]]:
    """Yield (pgn_text, game_count) batches from a zstd-compressed stream."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(fileobj)
    text_reader = io.TextIOWrapper(reader, encoding="latin-1", errors="replace")

    buf: list[str] = []
    game_count = 0
    in_movetext = False

    for line in text_reader:
        stripped = line.strip()

        if not stripped:
            if in_movetext:
                game_count += 1
                in_movetext = False
                buf.append(line)
                if game_count >= batch_size:
                    yield "".join(buf), game_count
                    buf.clear()
                    game_count = 0
                continue
            buf.append(line)
            continue

        if stripped.startswith("["):
            in_movetext = False
        else:
            in_movetext = True
        buf.append(line)

    if buf:
        yield "".join(buf), game_count


# ---------------------------------------------------------------------------
# Parsed batch -> Polars DataFrame
# ---------------------------------------------------------------------------

def batch_to_dataframe(parsed: dict[str, Any]) -> pl.DataFrame:
    tokens: np.ndarray = parsed["tokens"]
    n = int(tokens.shape[0])
    if n == 0:
        return pl.DataFrame()

    game_lengths: np.ndarray = parsed["game_lengths"]
    # `tokens` is pure moves (prepend_outcome=False), so slice to game_length.
    token_rows = [tokens[i, : int(game_lengths[i])].tolist() for i in range(n)]

    clocks: np.ndarray = parsed["clocks"]
    clock_rows = [clocks[i, : int(game_lengths[i])].tolist() for i in range(n)]

    # SAN and UCI arrive as Python lists from the Rust side, already trimmed
    # to the number of successfully tokenized moves.
    san_rows: list[list[str]] = parsed["san"]
    uci_rows: list[list[str]] = parsed["uci"]

    datetimes: list[datetime | None] = []
    for dt_str in parsed["date_time"]:
        parsed_dt: datetime | None = None
        if dt_str and len(dt_str) >= 10:
            try:
                parsed_dt = datetime.strptime(dt_str, "%Y.%m.%d %H:%M:%S")
            except ValueError:
                try:
                    parsed_dt = datetime.strptime(dt_str[:10], "%Y.%m.%d")
                except ValueError:
                    parsed_dt = None
        datetimes.append(parsed_dt)

    df = pl.DataFrame({
        "tokens": pl.Series("tokens", token_rows, dtype=pl.List(pl.Int16)),
        "san": pl.Series("san", san_rows, dtype=pl.List(pl.Utf8)),
        "uci": pl.Series("uci", uci_rows, dtype=pl.List(pl.Utf8)),
        "clock": pl.Series("clock", clock_rows, dtype=pl.List(pl.UInt16)),
        "game_length": pl.Series("game_length", parsed["game_lengths"], dtype=pl.UInt16),
        "outcome_token": pl.Series("outcome_token", parsed["outcome_tokens"], dtype=pl.UInt16),
        "result": pl.Series("result", parsed["result"], dtype=pl.Utf8),
        "white_player": pl.Series("white_player", parsed["white"], dtype=pl.Utf8),
        "black_player": pl.Series("black_player", parsed["black"], dtype=pl.Utf8),
        "white_elo": pl.Series("white_elo", parsed["white_elo"], dtype=pl.UInt16),
        "black_elo": pl.Series("black_elo", parsed["black_elo"], dtype=pl.UInt16),
        "white_rating_diff": pl.Series(
            "white_rating_diff", parsed["white_rating_diff"], dtype=pl.Int16
        ),
        "black_rating_diff": pl.Series(
            "black_rating_diff", parsed["black_rating_diff"], dtype=pl.Int16
        ),
        "eco": pl.Series("eco", parsed["eco"], dtype=pl.Utf8),
        "opening": pl.Series("opening", parsed["opening"], dtype=pl.Utf8),
        "time_control": pl.Series("time_control", parsed["time_control"], dtype=pl.Utf8),
        "termination": pl.Series("termination", parsed["termination"], dtype=pl.Utf8),
        "date": pl.Series("date", datetimes, dtype=pl.Datetime("ms")),
        "site": pl.Series("site", parsed["site"], dtype=pl.Utf8),
    })

    df = df.with_columns(
        pl.col("white_player").hash().alias("white_player"),
        pl.col("black_player").hash().alias("black_player"),
    )
    return df


# ---------------------------------------------------------------------------
# Streaming shard uploader
# ---------------------------------------------------------------------------

class StreamingShardUploader:
    """Buffer DataFrames into fixed-size shards and upload each as it completes.

    Shard names include the source month so parallel/resumed runs never
    collide: `<split>-<YYYY-MM>-<NNNN>.parquet`.
    """

    def __init__(
        self,
        split: str,
        month_tag: str,
        dest_repo: str,
        revision: str,
        shard_size: int,
        scratch_dir: Path,
        api: HfApi,
        log_prefix: str,
    ) -> None:
        self.split = split
        self.month_tag = month_tag
        self.dest_repo = dest_repo
        self.revision = revision
        self.shard_size = shard_size
        self.scratch_dir = scratch_dir
        self.api = api
        self.log_prefix = log_prefix
        self.frames: list[pl.DataFrame] = []
        self.buffered = 0
        self.total_games = 0
        self.shard_idx = 0
        self.uploaded: list[str] = []
        scratch_dir.mkdir(parents=True, exist_ok=True)

    def add(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            return
        self.frames.append(df)
        self.buffered += len(df)
        self.total_games += len(df)

        while self.buffered >= self.shard_size:
            combined = pl.concat(self.frames)
            shard_df = combined.head(self.shard_size)
            leftover = combined.slice(self.shard_size)
            self._flush(shard_df)
            if len(leftover) > 0:
                self.frames = [leftover]
                self.buffered = len(leftover)
            else:
                self.frames = []
                self.buffered = 0

    def finalize(self) -> None:
        if self.frames:
            combined = pl.concat(self.frames)
            if len(combined) > 0:
                self._flush(combined)
        self.frames = []
        self.buffered = 0

    def _flush(self, df: pl.DataFrame) -> None:
        name = f"{self.split}-{self.month_tag}-{self.shard_idx:04d}.parquet"
        local = self.scratch_dir / name
        df.write_parquet(local, compression="zstd", compression_level=3)
        size_mb = local.stat().st_size / 1024 / 1024
        log(
            f"[{self.split}] shard {self.shard_idx}: {len(df):,} games, "
            f"{size_mb:.1f} MB — uploading",
            self.log_prefix,
        )

        t0 = time.monotonic()
        self.api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"data/{name}",
            repo_id=self.dest_repo,
            repo_type="dataset",
            revision=self.revision,
            commit_message=f"Add {name}",
        )
        dt = time.monotonic() - t0
        rate = size_mb / dt if dt > 0 else 0.0
        log(
            f"[{self.split}] shard {self.shard_idx} uploaded in {dt:.1f}s "
            f"({rate:.1f} MB/s)",
            self.log_prefix,
        )

        local.unlink(missing_ok=True)
        self.uploaded.append(name)
        self.shard_idx += 1


# ---------------------------------------------------------------------------
# Month processors (run inside a worker)
# ---------------------------------------------------------------------------

def process_train_month(
    year_month: str,
    uploader: StreamingShardUploader,
    cfg: RunConfig,
    log_prefix: str,
) -> int:
    log("Streaming from HuggingFace", log_prefix)

    total = 0
    with open_hf_stream(year_month) as stream:
        for pgn_text, _ in stream_pgn_games(stream, cfg.batch_size):
            if not pgn_text.strip():
                continue

            t0 = time.monotonic()
            parsed = chess_engine.parse_pgn_lichess(
                pgn_text, max_ply=cfg.max_ply, max_games=cfg.batch_size * 2, min_ply=1
            )
            dt = time.monotonic() - t0

            df = batch_to_dataframe(parsed)
            if df.is_empty():
                continue

            n = len(df)
            if cfg.max_games_per_month is not None and total + n > cfg.max_games_per_month:
                df = df.head(cfg.max_games_per_month - total)
                n = len(df)

            total += n
            uploader.add(df)

            rate = n / dt if dt > 0 else 0.0
            log(f"Parsed {n:,} games ({rate:,.0f}/s) | total: {total:,}", log_prefix)

            if cfg.max_games_per_month is not None and total >= cfg.max_games_per_month:
                break

    log(f"Done — {total:,} games", log_prefix)
    return total


def process_holdout_month(
    year_month: str,
    val_uploader: StreamingShardUploader,
    test_uploader: StreamingShardUploader,
    cfg: RunConfig,
    log_prefix: str,
) -> tuple[int, int]:
    log("Streaming from HuggingFace", log_prefix)

    year, mon = year_month.split("-")
    val_start = datetime(int(year), int(mon), cfg.val_days[0])
    val_end = datetime(int(year), int(mon), cfg.val_days[1] + 1)
    test_start = datetime(int(year), int(mon), cfg.test_days[0])
    test_end = datetime(int(year), int(mon), cfg.test_days[1] + 1)

    with open_hf_stream(year_month) as stream:
        for pgn_text, _ in stream_pgn_games(stream, cfg.batch_size):
            if not pgn_text.strip():
                continue

            t0 = time.monotonic()
            parsed = chess_engine.parse_pgn_lichess(
                pgn_text, max_ply=cfg.max_ply, max_games=cfg.batch_size * 2, min_ply=1
            )
            dt = time.monotonic() - t0

            df = batch_to_dataframe(parsed)
            if df.is_empty():
                continue

            val_df = df.filter((pl.col("date") >= val_start) & (pl.col("date") < val_end))
            test_df = df.filter((pl.col("date") >= test_start) & (pl.col("date") < test_end))
            if not val_df.is_empty():
                val_uploader.add(val_df)
            if not test_df.is_empty():
                test_uploader.add(test_df)

            rate = len(df) / dt if dt > 0 else 0.0
            log(
                f"Parsed {len(df):,} ({rate:,.0f}/s) | "
                f"val {val_uploader.total_games:,}, test {test_uploader.total_games:,}",
                log_prefix,
            )

            max_date_any = df["date"].max()
            if isinstance(max_date_any, datetime) and max_date_any >= test_end:
                log("Past test date range — stopping early", log_prefix)
                break

    return val_uploader.total_games, test_uploader.total_games


# ---------------------------------------------------------------------------
# Worker entrypoint (top-level so spawn can pickle it)
# ---------------------------------------------------------------------------

def process_unit(unit: WorkUnit, cfg: RunConfig) -> dict[str, int]:
    api = HfApi()
    prefix = unit.tag
    worker_scratch = cfg.scratch_dir / f"{unit.kind}-{unit.year_month}"
    worker_scratch.mkdir(parents=True, exist_ok=True)

    try:
        if unit.kind == "train":
            uploader = StreamingShardUploader(
                "train", unit.year_month, cfg.dest_repo, cfg.revision,
                cfg.shard_size, worker_scratch, api, log_prefix=prefix,
            )
            total = process_train_month(unit.year_month, uploader, cfg, prefix)
            uploader.finalize()
            shard_count = len(uploader.uploaded)
            stats = {"train_games": total, "train_shards": shard_count}
        elif unit.kind == "holdout":
            val_uploader = StreamingShardUploader(
                "validation", unit.year_month, cfg.dest_repo, cfg.revision,
                cfg.shard_size, worker_scratch, api, log_prefix=prefix,
            )
            test_uploader = StreamingShardUploader(
                "test", unit.year_month, cfg.dest_repo, cfg.revision,
                cfg.shard_size, worker_scratch, api, log_prefix=prefix,
            )
            val_total, test_total = process_holdout_month(
                unit.year_month, val_uploader, test_uploader, cfg, prefix,
            )
            val_uploader.finalize()
            test_uploader.finalize()
            stats = {
                "val_games": val_total,
                "test_games": test_total,
                "val_shards": len(val_uploader.uploaded),
                "test_shards": len(test_uploader.uploaded),
            }
        else:
            raise ValueError(f"unknown unit kind: {unit.kind}")

        # Sentinel: marks this unit as fully uploaded
        api.upload_file(
            path_or_fileobj=io.BytesIO(b"ok\n"),
            path_in_repo=unit.sentinel_path,
            repo_id=cfg.dest_repo,
            repo_type="dataset",
            revision=cfg.revision,
            commit_message=f"Complete {unit.tag}",
        )
        log(f"Sentinel written: {unit.sentinel_path}", prefix)
        return stats
    finally:
        for leftover in worker_scratch.glob("*.parquet"):
            leftover.unlink(missing_ok=True)
        try:
            worker_scratch.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Resume: inspect existing repo state, clean up partial shards
# ---------------------------------------------------------------------------

def list_repo_files(api: HfApi, dest_repo: str, revision: str) -> list[str]:
    try:
        return list(
            api.list_repo_files(dest_repo, repo_type="dataset", revision=revision)
        )
    except RepositoryNotFoundError:
        return []


def partition_units(
    units: list[WorkUnit],
    existing_files: list[str],
    force: bool,
) -> tuple[list[WorkUnit], list[WorkUnit], list[str]]:
    """Split units into (to_run, already_done) and collect stale shards to delete.

    A unit is "done" iff its sentinel file exists. A partial unit (shards
    present, no sentinel) is queued to run and its orphan shards are returned
    for deletion before the rerun.
    """
    existing = set(existing_files)

    def unit_shard_prefix(u: WorkUnit) -> tuple[str, ...]:
        if u.kind == "train":
            return (f"data/train-{u.year_month}-",)
        return (
            f"data/validation-{u.year_month}-",
            f"data/test-{u.year_month}-",
        )

    to_run: list[WorkUnit] = []
    done: list[WorkUnit] = []
    stale: list[str] = []

    for u in units:
        has_sentinel = u.sentinel_path in existing
        if has_sentinel and not force:
            done.append(u)
            continue

        # Either force, or incomplete: queue to run and mark any orphan shards
        to_run.append(u)
        prefixes = unit_shard_prefix(u)
        for f in existing_files:
            if any(f.startswith(p) for p in prefixes):
                stale.append(f)
        if force and has_sentinel:
            stale.append(u.sentinel_path)

    return to_run, done, stale


def delete_stale_shards(
    api: HfApi, dest_repo: str, revision: str, paths: list[str]
) -> None:
    if not paths:
        return
    log(f"Deleting {len(paths)} orphan file(s) from previous runs")
    for p in paths:
        log(f"  - {p}")
    api.create_commit(
        repo_id=dest_repo,
        repo_type="dataset",
        revision=revision,
        operations=[CommitOperationDelete(path_in_repo=p) for p in paths],
        commit_message=f"Clean up {len(paths)} orphan file(s) before re-extract",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_day_range(s: str) -> tuple[int, int]:
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"day range must be 'start-end', got {s!r}")
    return int(parts[0]), int(parts[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the Lichess parquet dataset from raw monthly PGN dumps"
    )
    parser.add_argument(
        "--months", nargs="+", required=True,
        help="Training months to extract, e.g. 2025-01 2025-02 2025-03",
    )
    parser.add_argument(
        "--holdout-month", default=None,
        help="Month for validation/test splits (optional)",
    )
    parser.add_argument(
        "--val-days", type=parse_day_range, default=(1, 3),
        help="Day range for validation split (default 1-3)",
    )
    parser.add_argument(
        "--test-days", type=parse_day_range, default=(15, 17),
        help="Day range for test split (default 15-17)",
    )
    parser.add_argument(
        "--dest-repo", required=True,
        help="Destination HuggingFace dataset repo (created if missing)",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help=(
            f"Branch to write to (default {DEFAULT_REVISION!r}); will be "
            "created if missing. Use 'main' to write directly to the primary "
            "branch. Squash-merge this branch to main once the extraction is verified."
        ),
    )
    parser.add_argument(
        "--max-ply", type=int, default=DEFAULT_MAX_PLY,
        help=f"Ply cap (default {DEFAULT_MAX_PLY}; games beyond get PLY_LIMIT outcome)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Games per parse batch",
    )
    parser.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help="Games per output parquet shard",
    )
    parser.add_argument(
        "--scratch-dir", type=Path, default=DEFAULT_SCRATCH_DIR,
        help=(
            f"Local staging directory for in-flight shards (default "
            f"{DEFAULT_SCRATCH_DIR}). /dev/shm avoids touching the network "
            "volume; fall back to a disk path if tmpfs is too small."
        ),
    )
    parser.add_argument(
        "--max-games-per-month", type=int, default=None,
        help="Cap games per training month (for smoke tests)",
    )
    parser.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS,
        help=f"Parallel worker processes (default {DEFAULT_JOBS})",
    )
    parser.add_argument(
        "--threads-per-worker", type=int, default=DEFAULT_THREADS_PER_WORKER,
        help=(
            f"Cap rayon / polars threads per worker (default "
            f"{DEFAULT_THREADS_PER_WORKER}). Total CPU budget is roughly "
            "jobs * threads_per_worker; keep it below the pod's core count "
            "minus whatever the training run is using so the trainer isn't "
            "starved. Set to 0 to leave the defaults alone."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore existing sentinels and re-extract every requested unit",
    )
    args = parser.parse_args()

    cfg = RunConfig(
        dest_repo=args.dest_repo,
        revision=args.revision,
        max_ply=args.max_ply,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        scratch_dir=args.scratch_dir,
        max_games_per_month=args.max_games_per_month,
        val_days=args.val_days,
        test_days=args.test_days,
    )

    units: list[WorkUnit] = [WorkUnit("train", m) for m in args.months]
    if args.holdout_month:
        units.append(WorkUnit("holdout", args.holdout_month))

    if args.threads_per_worker > 0:
        # Must be set before the ProcessPoolExecutor spawns children so the
        # env vars are inherited. Rayon initialises its global pool lazily on
        # first use, so capping it here prevents chess_engine from spinning
        # up one thread per physical core inside each worker.
        os.environ["RAYON_NUM_THREADS"] = str(args.threads_per_worker)
        os.environ["POLARS_MAX_THREADS"] = str(args.threads_per_worker)

    log("=== Lichess parquet extraction ===")
    log(f"Training months: {args.months}")
    if args.holdout_month:
        log(f"Holdout month:  {args.holdout_month} (val {cfg.val_days}, test {cfg.test_days})")
    log(f"Destination:    {cfg.dest_repo}")
    log(f"Revision:       {cfg.revision}")
    log(f"max_ply:        {cfg.max_ply}")
    log(f"Batch / shard:  {cfg.batch_size:,} / {cfg.shard_size:,}")
    log(f"Scratch dir:    {cfg.scratch_dir}")
    log(f"Jobs:           {args.jobs} (threads/worker: {args.threads_per_worker or 'default'})")
    log(f"Force re-extract: {args.force}")

    api = HfApi()
    api.create_repo(cfg.dest_repo, repo_type="dataset", exist_ok=True)
    if cfg.revision != "main":
        api.create_branch(
            repo_id=cfg.dest_repo,
            branch=cfg.revision,
            repo_type="dataset",
            exist_ok=True,
        )

    existing = list_repo_files(api, cfg.dest_repo, cfg.revision)
    to_run, already_done, stale = partition_units(units, existing, args.force)

    if already_done:
        log("")
        log(f"Resuming — {len(already_done)} unit(s) already complete:")
        for u in already_done:
            log(f"  ✓ {u.tag}")
    if to_run:
        log("")
        log(f"To process: {len(to_run)} unit(s):")
        for u in to_run:
            log(f"  • {u.tag}")
    else:
        log("")
        log("Nothing to do — every requested unit has a sentinel.")
        return 0

    delete_stale_shards(api, cfg.dest_repo, cfg.revision, stale)

    n_workers = max(1, min(args.jobs, len(to_run)))
    log("")
    log(f"Spawning {n_workers} worker(s)...")

    ctx = mp.get_context("spawn")
    failures: list[tuple[WorkUnit, BaseException]] = []
    aggregate: dict[str, int] = {}

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        future_to_unit = {
            executor.submit(process_unit, u, cfg): u for u in to_run
        }
        for fut in cf.as_completed(future_to_unit):
            u = future_to_unit[fut]
            try:
                stats = fut.result()
                for k, v in stats.items():
                    aggregate[k] = aggregate.get(k, 0) + v
                log(f"[{u.tag}] complete: {stats}")
            except BaseException as e:
                log(f"[{u.tag}] FAILED: {type(e).__name__}: {e}")
                failures.append((u, e))

    log("")
    log("=== Summary ===")
    for k in sorted(aggregate):
        log(f"  {k}: {aggregate[k]:,}")
    log(f"Dataset: https://huggingface.co/datasets/{cfg.dest_repo}")

    if failures:
        log("")
        log(f"{len(failures)} unit(s) failed — rerun the same command to resume:")
        for u, e in failures:
            log(f"  ✗ {u.tag}: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
