#!/usr/bin/env python3
"""Extract Lichess monthly PGN database dumps into PAWN-compatible Parquet.

Downloads zstd-compressed PGN files from database.lichess.org, parses games
via the Rust chess engine, and writes sharded Parquet to disk with
train/val/test splits. All months are downloaded and parsed in parallel.

The output schema stores complete PAWN training sequences: the tokens column
begins with the outcome token followed by move tokens. Outcome tokens are
classified by replaying the full game in the Rust engine to detect checkmate,
stalemate, etc. Games terminated by rules infraction, abandonment, or
incomplete records are filtered out.

Player usernames are hashed to uint64 via Polars xxHash64.

Usage:
    python scripts/extract_lichess_parquet.py \\
        --months 2025-01 2025-02 2025-03 \\
        --holdout-month 2026-01 \\
        --hf-repo thomas-schweich/pawn-lichess-full
"""

import argparse
import io
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import chess_engine
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_RAW_BUCKET = "thomas-schweich/raw-lichess"
HF_RAW_FILENAME_TEMPLATE = "lichess_{year_month}.pgn.zst"
MAX_PLY = 255
SHARD_TARGET_GAMES = 1_000_000


def log(msg: str, prefix: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    tag = f"[{prefix}] " if prefix else ""
    print(f"[{ts}] {tag}{msg}", flush=True)


# ---------------------------------------------------------------------------
# PGN streaming + download
# ---------------------------------------------------------------------------

def stream_pgn_games(fileobj, batch_size: int):
    """Yield (pgn_text, game_count) batches from a zstd-compressed stream."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(fileobj)
    text_reader = io.TextIOWrapper(reader, encoding="latin-1", errors="replace")

    buf = []
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


def download_zst(year_month: str, output_dir: Path, prefix: str = "") -> Path:
    """Download a Lichess zstd PGN dump from HuggingFace bucket."""
    from huggingface_hub import download_bucket_files

    hf_filename = HF_RAW_FILENAME_TEMPLATE.format(year_month=year_month)
    zst_path = output_dir / hf_filename
    if zst_path.exists():
        log(f"Using cached {zst_path} ({zst_path.stat().st_size / 1e9:.1f} GB)", prefix)
        return zst_path

    log(f"Downloading {HF_RAW_BUCKET}/{hf_filename} -> {zst_path}", prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    download_bucket_files(
        HF_RAW_BUCKET,
        files=[(hf_filename, str(zst_path))],
        raise_on_missing_files=True,
    )
    dt = time.monotonic() - t0

    size_gb = zst_path.stat().st_size / 1e9
    rate = (size_gb * 1000) / dt if dt > 0 else 0
    log(f"Downloaded {size_gb:.2f} GB in {dt:.0f}s ({rate:.0f} MB/s)", prefix)
    return zst_path


# ---------------------------------------------------------------------------
# DataFrame construction (v2: outcome token prepended, no eval)
# ---------------------------------------------------------------------------

def batch_to_dataframe(parsed: dict) -> pl.DataFrame:
    """Convert a parsed batch dict from parse_pgn_lichess into a Polars DataFrame.

    The tokens array is (N, seq_len) where seq_len = max_ply + 1.
    tokens[i, 0] is the outcome token, tokens[i, 1:game_length+1] are moves.
    """
    tokens: np.ndarray = parsed["tokens"]  # (N, seq_len) i16
    n = tokens.shape[0]
    if n == 0:
        return pl.DataFrame()

    game_lengths: np.ndarray = parsed["game_lengths"]  # (N,) u16 — move count
    seq_len = tokens.shape[1]

    # Token lists: outcome + moves, trimmed to actual length
    # Length of token list = game_length + 1 (outcome token)
    token_rows = [
        tokens[i, : game_lengths[i] + 1].tolist()
        for i in range(n)
    ]

    # Clock lists: aligned with tokens (0 for outcome slot, then per-ply clocks)
    clocks: np.ndarray = parsed["clocks"]  # (N, max_ply) u16
    clock_rows = [
        [0] + clocks[i, : game_lengths[i]].tolist()
        for i in range(n)
    ]

    # Parse datetime strings
    datetimes = []
    for dt_str in parsed["date_time"]:
        if dt_str and len(dt_str) >= 10:
            try:
                datetimes.append(datetime.strptime(dt_str, "%Y.%m.%d %H:%M:%S"))
            except ValueError:
                try:
                    datetimes.append(datetime.strptime(dt_str[:10], "%Y.%m.%d"))
                except ValueError:
                    datetimes.append(None)
        else:
            datetimes.append(None)

    df = pl.DataFrame({
        "tokens": pl.Series("tokens", token_rows, dtype=pl.List(pl.Int16)),
        "clock": pl.Series("clock", clock_rows, dtype=pl.List(pl.UInt16)),
        "game_length": pl.Series("game_length", parsed["game_lengths"], dtype=pl.UInt16),
        "result": pl.Series("result", parsed["result"], dtype=pl.Utf8),
        "white_player": pl.Series("white_player", parsed["white"], dtype=pl.Utf8),
        "black_player": pl.Series("black_player", parsed["black"], dtype=pl.Utf8),
        "white_elo": pl.Series("white_elo", parsed["white_elo"], dtype=pl.UInt16),
        "black_elo": pl.Series("black_elo", parsed["black_elo"], dtype=pl.UInt16),
        "white_rating_diff": pl.Series("white_rating_diff", parsed["white_rating_diff"], dtype=pl.Int16),
        "black_rating_diff": pl.Series("black_rating_diff", parsed["black_rating_diff"], dtype=pl.Int16),
        "eco": pl.Series("eco", parsed["eco"], dtype=pl.Utf8),
        "opening": pl.Series("opening", parsed["opening"], dtype=pl.Utf8),
        "time_control": pl.Series("time_control", parsed["time_control"], dtype=pl.Utf8),
        "termination": pl.Series("termination", parsed["termination"], dtype=pl.Utf8),
        "date": pl.Series("date", datetimes, dtype=pl.Datetime("ms")),
        "site": pl.Series("site", parsed["site"], dtype=pl.Utf8),
    })

    # Hash usernames: vectorized xxHash64 via Polars (pinned to Polars 1.39.3).
    df = df.with_columns(
        pl.col("white_player").hash().alias("white_player"),
        pl.col("black_player").hash().alias("black_player"),
    )

    return df


# ---------------------------------------------------------------------------
# Worker: process a single training month
# ---------------------------------------------------------------------------

def process_train_month(
    year_month: str,
    output_dir: Path,
    batch_size: int,
    shard_size: int,
    max_games: int | None,
) -> list[Path]:
    """Download, parse, and write shards for a training month."""
    prefix = f"train/{year_month}"
    log("Starting", prefix)

    zst_path = download_zst(year_month, output_dir, prefix)

    shard_paths = []
    shard_idx = 0
    buffer_frames: list[pl.DataFrame] = []
    buffer_games = 0
    total_games = 0

    def flush_shard(df: pl.DataFrame) -> None:
        nonlocal shard_idx
        path = output_dir / "data" / f"train-{year_month}-{shard_idx:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd", compression_level=3)
        size_mb = path.stat().st_size / 1024 / 1024
        log(f"Shard {shard_idx}: {len(df):,} games, {size_mb:.1f} MB", prefix)
        shard_paths.append(path)
        shard_idx += 1

    with open(zst_path, "rb") as f:
        for pgn_text, _ in stream_pgn_games(f, batch_size):
            if not pgn_text.strip():
                continue

            t0 = time.monotonic()
            parsed = chess_engine.parse_pgn_lichess(
                pgn_text, max_ply=MAX_PLY, max_games=batch_size * 2, min_ply=1
            )
            dt = time.monotonic() - t0

            df = batch_to_dataframe(parsed)
            if df.is_empty():
                continue

            n = len(df)
            if max_games and total_games + n > max_games:
                df = df.head(max_games - total_games)
                n = len(df)

            total_games += n
            buffer_frames.append(df)
            buffer_games += n

            rate = n / dt if dt > 0 else 0
            log(f"Parsed {n:,} games ({rate:,.0f}/s) | total: {total_games:,}", prefix)

            while buffer_games >= shard_size:
                combined = pl.concat(buffer_frames)
                flush_shard(combined.head(shard_size))
                leftover = combined.slice(shard_size)
                buffer_frames = [leftover] if len(leftover) > 0 else []
                buffer_games = len(leftover) if len(leftover) > 0 else 0

            if max_games and total_games >= max_games:
                break

    if buffer_frames:
        combined = pl.concat(buffer_frames)
        if len(combined) > 0:
            flush_shard(combined)

    log(f"Done — {total_games:,} games, {len(shard_paths)} shards", prefix)
    return shard_paths


# ---------------------------------------------------------------------------
# Worker: process holdout month (full date-range extraction, no sampling)
# ---------------------------------------------------------------------------

def process_holdout_month(
    year_month: str,
    output_dir: Path,
    batch_size: int,
    shard_size: int,
    val_days: tuple[int, int],
    test_days: tuple[int, int],
) -> tuple[list[Path], list[Path]]:
    """Download, parse, and write val/test shards from date ranges."""
    prefix = f"holdout/{year_month}"
    log("Starting", prefix)

    zst_path = download_zst(year_month, output_dir, prefix)

    year, mon = year_month.split("-")
    val_start = datetime(int(year), int(mon), val_days[0])
    val_end = datetime(int(year), int(mon), val_days[1] + 1)  # exclusive
    test_start = datetime(int(year), int(mon), test_days[0])
    test_end = datetime(int(year), int(mon), test_days[1] + 1)  # exclusive

    log(f"Val: day {val_days[0]}-{val_days[1]}, Test: day {test_days[0]}-{test_days[1]}", prefix)

    val_buf = SplitBuffer("validation", shard_size, output_dir)
    test_buf = SplitBuffer("test", shard_size, output_dir)

    total = 0
    past_test_end = False

    with open(zst_path, "rb") as f:
        for pgn_text, _ in stream_pgn_games(f, batch_size):
            if not pgn_text.strip():
                continue

            t0 = time.monotonic()
            parsed = chess_engine.parse_pgn_lichess(
                pgn_text, max_ply=MAX_PLY, max_games=batch_size * 2, min_ply=1
            )
            dt = time.monotonic() - t0

            df = batch_to_dataframe(parsed)
            if df.is_empty():
                continue

            total += len(df)

            val_df = df.filter(
                (pl.col("date") >= val_start) & (pl.col("date") < val_end)
            )
            test_df = df.filter(
                (pl.col("date") >= test_start) & (pl.col("date") < test_end)
            )

            if not val_df.is_empty():
                val_buf.add(val_df)
            if not test_df.is_empty():
                test_buf.add(test_df)

            rate = len(df) / dt if dt > 0 else 0
            log(f"Parsed {len(df):,} ({rate:,.0f}/s) | val: {val_buf.total_games:,}, test: {test_buf.total_games:,}", prefix)

            # Stop early once past test range
            max_date = df["date"].max()
            if isinstance(max_date, datetime) and max_date >= test_end:
                log("Past test date range — stopping early", prefix)
                break

    val_buf.flush_remaining()
    test_buf.flush_remaining()

    log(f"Done — val: {val_buf.total_games:,}, test: {test_buf.total_games:,}", prefix)

    val_paths = val_buf.rename_shards() if val_buf.shard_paths else []
    test_paths = test_buf.rename_shards() if test_buf.shard_paths else []

    return [Path(p) for p in val_paths], [Path(p) for p in test_paths]


# ---------------------------------------------------------------------------
# Shard buffer
# ---------------------------------------------------------------------------

class SplitBuffer:
    """Accumulates DataFrames for a single split and flushes to Parquet shards."""

    def __init__(self, split: str, shard_size: int, output_dir: Path):
        self.split = split
        self.shard_size = shard_size
        self.output_dir = output_dir / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[pl.DataFrame] = []
        self.buffered = 0
        self.total_games = 0
        self.shard_paths: list[Path] = []
        self.shard_idx = 0

    def add(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            return
        self.frames.append(df)
        self.buffered += len(df)
        self.total_games += len(df)
        self._flush_full()

    def _flush_full(self) -> None:
        while self.buffered >= self.shard_size:
            combined = pl.concat(self.frames)
            shard_df = combined.head(self.shard_size)
            leftover = combined.slice(self.shard_size)
            self._write_shard(shard_df)
            self.frames = [leftover] if len(leftover) > 0 else []
            self.buffered = len(leftover) if len(leftover) > 0 else 0

    def flush_remaining(self) -> None:
        if self.frames:
            combined = pl.concat(self.frames)
            if len(combined) > 0:
                self._write_shard(combined)
            self.frames.clear()
            self.buffered = 0

    def _write_shard(self, df: pl.DataFrame) -> None:
        path = self.output_dir / f"{self.split}-temp-{self.shard_idx:05d}.parquet"
        df.write_parquet(path, compression="zstd", compression_level=3)
        size_mb = path.stat().st_size / 1024 / 1024
        log(f"  [{self.split}] shard {self.shard_idx}: {len(df):,} games, {size_mb:.1f} MB")
        self.shard_paths.append(path)
        self.shard_idx += 1

    def rename_shards(self) -> list[Path]:
        n = len(self.shard_paths)
        final = []
        for i, path in enumerate(self.shard_paths):
            new_name = f"{self.split}-{i:05d}-of-{n:05d}.parquet"
            new_path = path.parent / new_name
            path.rename(new_path)
            final.append(new_path)
            log(f"  {path.name} -> {new_name}")
        self.shard_paths = final
        return final


# ---------------------------------------------------------------------------
# Multiprocessing wrappers
# ---------------------------------------------------------------------------

def _worker_train(args):
    year_month, output_dir, batch_size, shard_size, max_games = args
    paths = process_train_month(year_month, Path(output_dir), batch_size, shard_size, max_games)
    return [str(p) for p in paths]


def _worker_holdout(args):
    year_month, output_dir, batch_size, shard_size, val_days, test_days = args
    return process_holdout_month(
        year_month, Path(output_dir), batch_size, shard_size, val_days, test_days
    )


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------

def upload_to_hf(output_dir: Path, hf_repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    log(f"Uploading to HuggingFace: {hf_repo}")
    api.create_repo(hf_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo,
        folder_path=str(output_dir),
        repo_type="dataset",
    )
    log(f"Upload complete: https://huggingface.co/datasets/{hf_repo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Lichess PGN dumps to PAWN-compatible Parquet"
    )
    parser.add_argument(
        "--months", nargs="+", required=True,
        help="Training month(s) to download, e.g. 2025-01 2025-02 2025-03"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/lichess-parquet"),
        help="Output directory for Parquet shards"
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo to push to"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500_000,
        help="Games per batch during parsing"
    )
    parser.add_argument(
        "--shard-size", type=int, default=SHARD_TARGET_GAMES,
        help="Target games per output shard"
    )
    parser.add_argument(
        "--holdout-month", type=str, default=None,
        help="Month for val/test (e.g. 2026-01)"
    )
    parser.add_argument(
        "--val-days", type=str, default="1-3",
        help="Day range for validation (e.g. 1-3 for Jan 1-3)"
    )
    parser.add_argument(
        "--test-days", type=str, default="15-17",
        help="Day range for test (e.g. 15-17 for Jan 15-17)"
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Stop after this many training games per month"
    )
    args = parser.parse_args()

    # Parse day ranges
    val_days = tuple(int(x) for x in args.val_days.split("-"))
    test_days = tuple(int(x) for x in args.test_days.split("-"))

    log("=== Lichess Parquet Extraction (v2: outcome tokens) ===")
    log(f"Training months: {args.months}")
    if args.holdout_month:
        log(f"Holdout month: {args.holdout_month} (val days {val_days[0]}-{val_days[1]}, test days {test_days[0]}-{test_days[1]})")
    log(f"Output: {args.output}")
    log(f"Batch size: {args.batch_size:,}, Shard size: {args.shard_size:,}")
    n_workers = len(args.months) + (1 if args.holdout_month else 0)
    log(f"Parallelism: {n_workers} workers")
    log("")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(parents=True, exist_ok=True)

    # ── Launch all workers in parallel ─────────────────────────────────
    train_args = [
        (m, str(args.output), args.batch_size, args.shard_size, args.max_games)
        for m in args.months
    ]
    holdout_args = None
    if args.holdout_month:
        holdout_args = (
            args.holdout_month, str(args.output), args.batch_size,
            args.shard_size, val_days, test_days,
        )

    log(f"Spawning {n_workers} workers...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        train_results = pool.map_async(_worker_train, train_args)
        holdout_result = None
        if holdout_args:
            holdout_result = pool.apply_async(_worker_holdout, (holdout_args,))

        train_shard_lists = train_results.get()
        val_paths = []
        test_paths = []
        if holdout_result:
            val_str, test_str = holdout_result.get()
            val_paths = [Path(p) for p in val_str]
            test_paths = [Path(p) for p in test_str]

    # ── Rename train shards ────────────────────────────────────────────
    all_train_paths = []
    for shard_list in train_shard_lists:
        all_train_paths.extend(Path(p) for p in shard_list)
    all_train_paths.sort(key=lambda p: p.name)
    n_train = len(all_train_paths)

    log(f"\n=== Renaming {n_train} train shards ===")
    final_paths = []
    for i, path in enumerate(all_train_paths):
        new_name = f"train-{i:05d}-of-{n_train:05d}.parquet"
        new_path = path.parent / new_name
        path.rename(new_path)
        final_paths.append(new_path)

    final_paths.extend(val_paths)
    final_paths.extend(test_paths)

    # ── Summary ────────────────────────────────────────────────────────
    log(f"\n=== Summary ===")
    log(f"Train: {n_train} shards")
    log(f"Validation: {len(val_paths)} shards")
    log(f"Test: {len(test_paths)} shards")

    if final_paths:
        total_size = sum(p.stat().st_size for p in final_paths)
        log(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    if args.hf_repo:
        upload_to_hf(args.output, args.hf_repo)

    log("\nDone!")


if __name__ == "__main__":
    main()
