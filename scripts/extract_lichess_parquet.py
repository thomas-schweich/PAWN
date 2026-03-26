#!/usr/bin/env python3
"""Extract Lichess monthly PGN database dumps into PAWN-compatible Parquet.

Downloads a zstd-compressed PGN from database.lichess.org, parses games via
the Rust chess engine (tokens, clocks, evals, headers), builds a Polars
DataFrame, and writes sharded Parquet to disk with train/val/test splits.

The output schema stores pre-tokenized move sequences as list[uint16],
clock annotations as list[uint16] (seconds remaining), eval annotations
as list[int16] (centipawns, mate=+-32000, missing=-32768), and metadata
columns. Player usernames are SHA-256 hashed to uint64.

Shards are written in chronological order (no shuffle) so that the last
shards can be held out as val/test by time. The script auto-assigns splits
based on --val-weeks and --test-weeks.

Designed to run on a CPU pod with the pawn Docker image.

Usage:
    python scripts/extract_lichess_parquet.py \\
        --months 2025-01 2025-02 2025-03 \\
        --output /workspace/lichess-parquet \\
        --hf-repo thomas-schweich/lichess-pawn \\
        --batch-size 500000
"""

import argparse
import io
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import chess_engine
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LICHESS_URL_TEMPLATE = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_{year_month}.pgn.zst"
)
MAX_PLY = 255  # Max plies per game (token sequence = outcome + 255 plies)
EVAL_MISSING = -32768  # i16::MIN sentinel for missing eval
SHARD_TARGET_GAMES = 1_000_000  # Target games per shard


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# PGN streaming
# ---------------------------------------------------------------------------

def stream_pgn_games(fileobj, batch_size: int):
    """Yield batches of complete PGN game strings from a text stream.

    Each batch is a single string containing `batch_size` complete games
    (delimited by blank lines between the last movetext and next header).
    """
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(fileobj)
    text_reader = io.TextIOWrapper(reader, encoding="latin-1", errors="replace")

    buf = []
    game_count = 0
    # Track blank-line state for game boundary detection
    last_was_blank = False
    in_movetext = False

    for line in text_reader:
        stripped = line.strip()

        if not stripped:
            if in_movetext:
                # End of movetext — game boundary
                game_count += 1
                in_movetext = False
                last_was_blank = True
                buf.append(line)
                if game_count >= batch_size:
                    yield "".join(buf), game_count
                    buf.clear()
                    game_count = 0
                continue
            last_was_blank = True
            buf.append(line)
            continue

        if stripped.startswith("["):
            in_movetext = False
        else:
            in_movetext = True

        last_was_blank = False
        buf.append(line)

    # Final batch
    if buf:
        yield "".join(buf), game_count


def download_month(year_month: str, output_dir: Path, batch_size: int):
    """Download and parse a single month's PGN dump, yielding parsed batches."""
    url = LICHESS_URL_TEMPLATE.format(year_month=year_month)
    log(f"Downloading {url}")

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "pawn-lichess-extract/1.0")

    response = urllib.request.urlopen(req)
    total_games = 0
    batch_num = 0

    for pgn_text, n_games_in_chunk in stream_pgn_games(response, batch_size):
        if not pgn_text.strip():
            continue

        t0 = time.monotonic()
        parsed = chess_engine.parse_pgn_enriched(
            pgn_text, max_ply=MAX_PLY, max_games=batch_size * 2, min_ply=1
        )
        dt = time.monotonic() - t0

        n = parsed["tokens"].shape[0]
        total_games += n
        batch_num += 1
        rate = n / dt if dt > 0 else 0
        log(f"  [{year_month}] batch {batch_num}: {n:,} games parsed in {dt:.1f}s ({rate:,.0f} games/s) | total: {total_games:,}")

        yield parsed

    response.close()
    log(f"  [{year_month}] Done — {total_games:,} games total")


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def numpy_rows_to_list_series(
    arr: np.ndarray, lengths: np.ndarray, name: str, dtype: pl.DataType
) -> pl.Series:
    """Convert a 0-padded (N, max_ply) numpy array to a Polars List series,
    trimming each row to its actual game length."""
    inner = dtype.inner  # type: ignore[attr-defined]
    rows = [arr[i, :lengths[i]].tolist() for i in range(len(arr))]
    return pl.Series(name, rows, dtype=pl.List(inner))


def batch_to_dataframe(parsed: dict) -> pl.DataFrame:
    """Convert a parsed batch dict from Rust into a Polars DataFrame.

    Rust returns numpy arrays: tokens/clocks/evals as (N, max_ply),
    scalar fields as (N,) arrays, and strings as Python lists.
    """
    tokens: np.ndarray = parsed["tokens"]  # (N, max_ply) i16
    n = tokens.shape[0]
    if n == 0:
        return pl.DataFrame()

    lengths: np.ndarray = parsed["game_lengths"]  # (N,) u16

    # Parse datetime strings -> proper datetime
    # Format: "YYYY.MM.DD HH:MM:SS"
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
        "tokens": numpy_rows_to_list_series(tokens, lengths, "tokens", pl.List(pl.Int16)),
        "clock": numpy_rows_to_list_series(parsed["clocks"], lengths, "clock", pl.List(pl.UInt16)),
        "eval": numpy_rows_to_list_series(parsed["evals"], lengths, "eval", pl.List(pl.Int16)),
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

    # Hash usernames: vectorized xxHash64 via Polars.
    # NOTE: hash() output is deterministic within a Polars version but the
    # algorithm is not guaranteed stable across major versions. Originally
    # recorded with Polars 1.39.3. Pin Polars version (via uv.lock) and
    # tag the repo to ensure reproducibility. See test_enriched_pgn.py
    # TestPlayerHashRegression for the snapshot test.
    df = df.with_columns(
        pl.col("white_player").hash().alias("white_player"),
        pl.col("black_player").hash().alias("black_player"),
    )

    return df


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------

def write_shard(
    df: pl.DataFrame,
    output_dir: Path,
    split: str,
    shard_idx: int,
    total_shards: int,
) -> Path:
    """Write a single Parquet shard with HF-compatible naming."""
    name = f"{split}-{shard_idx:05d}-of-{total_shards:05d}.parquet"
    path = output_dir / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", compression_level=3)
    size_mb = path.stat().st_size / 1024 / 1024
    log(f"  Wrote {name}: {len(df):,} games, {size_mb:.1f} MB")
    return path


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------

def upload_to_hf(output_dir: Path, hf_repo: str) -> None:
    """Upload the output directory to HuggingFace as a dataset."""
    from huggingface_hub import HfApi

    api = HfApi()
    log(f"Uploading to HuggingFace: {hf_repo}")

    # Create repo if it doesn't exist
    api.create_repo(hf_repo, repo_type="dataset", exist_ok=True)

    # Upload the data directory
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
        help="Month(s) to download, e.g. 2024-12 2025-01 2025-02 2025-03"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/lichess-parquet"),
        help="Output directory for Parquet shards"
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo to push to (e.g. thomas-schweich/pawn-lichess-full)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500_000,
        help="Games per batch during parsing (controls memory usage)"
    )
    parser.add_argument(
        "--shard-size", type=int, default=SHARD_TARGET_GAMES,
        help="Target games per output shard"
    )
    parser.add_argument(
        "--holdout-months", nargs="*", default=None,
        help="Month(s) to use for val/test splits (e.g. 2024-12). "
             "First half of holdout shards become validation, second half test. "
             "All other months become train."
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Stop after this many games total (for testing)"
    )
    args = parser.parse_args()

    holdout = set(args.holdout_months) if args.holdout_months else set()

    log(f"=== Lichess Parquet Extraction ===")
    log(f"Months: {args.months}")
    log(f"Output: {args.output}")
    log(f"Batch size: {args.batch_size:,}")
    log(f"Shard size: {args.shard_size:,}")
    if holdout:
        log(f"Holdout months (val/test): {sorted(holdout)}")
    else:
        log(f"No holdout months — all shards will be train")
    if args.max_games:
        log(f"Max games: {args.max_games:,}")
    log("")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(parents=True, exist_ok=True)

    # Phase 1: Download, parse, and write temporary shards.
    # We accumulate games into a buffer and flush when it reaches shard_size.
    # Track (path, month) for each shard so we can assign splits later.
    temp_shards: list[tuple[Path, str]] = []  # (path, source_month)
    buffer_frames: list[pl.DataFrame] = []
    buffer_games = 0
    total_games = 0
    shard_idx = 0
    current_month = ""
    stop = False

    for month in args.months:
        if stop:
            break
        current_month = month
        log(f"\n=== Processing {month} ===")

        for parsed in download_month(month, args.output, args.batch_size):
            df = batch_to_dataframe(parsed)
            if df.is_empty():
                continue

            # Apply max_games limit
            remaining = None
            if args.max_games:
                remaining = args.max_games - total_games
                if remaining <= 0:
                    stop = True
                    break
                if len(df) > remaining:
                    df = df.head(remaining)

            buffer_frames.append(df)
            buffer_games += len(df)
            total_games += len(df)

            # Flush shard when buffer is full
            while buffer_games >= args.shard_size:
                combined = pl.concat(buffer_frames)
                shard_df = combined.head(args.shard_size)
                leftover = combined.slice(args.shard_size)

                path = args.output / "data" / f"shard-{shard_idx:05d}.parquet"
                shard_df.write_parquet(path, compression="zstd", compression_level=3)
                size_mb = path.stat().st_size / 1024 / 1024
                log(f"  Shard {shard_idx}: {len(shard_df):,} games, {size_mb:.1f} MB | Total: {total_games:,}")
                temp_shards.append((path, current_month))
                shard_idx += 1

                buffer_frames = [leftover] if len(leftover) > 0 else []
                buffer_games = len(leftover) if len(leftover) > 0 else 0

            if remaining is not None and remaining <= len(df):
                stop = True
                break

    # Flush remaining buffer
    if buffer_frames:
        combined = pl.concat(buffer_frames)
        if len(combined) > 0:
            path = args.output / "data" / f"shard-{shard_idx:05d}.parquet"
            combined.write_parquet(path, compression="zstd", compression_level=3)
            size_mb = path.stat().st_size / 1024 / 1024
            log(f"  Shard {shard_idx}: {len(combined):,} games, {size_mb:.1f} MB (final) | Total: {total_games:,}")
            temp_shards.append((path, current_month))
            shard_idx += 1

    n_shards = len(temp_shards)
    log(f"\n=== Assigning splits ({n_shards} shards) ===")

    # Phase 2: Assign splits based on holdout months.
    # Holdout month shards are split evenly into validation and test.
    # All other shards become train.
    train_shards = [(p, m) for p, m in temp_shards if m not in holdout]
    holdout_shards = [(p, m) for p, m in temp_shards if m in holdout]

    n_holdout = len(holdout_shards)
    n_val = n_holdout // 2
    n_test = n_holdout - n_val
    n_train = len(train_shards)

    log(f"  Train: {n_train} shards, Val: {n_val} shards, Test: {n_test} shards")
    if holdout:
        log(f"  Holdout months {sorted(holdout)}: {n_holdout} shards -> {n_val} val + {n_test} test")

    final_paths = []

    # Rename train shards
    for i, (path, _month) in enumerate(train_shards):
        new_name = f"train-{i:05d}-of-{n_train:05d}.parquet"
        new_path = path.parent / new_name
        path.rename(new_path)
        final_paths.append(new_path)
        log(f"  {path.name} -> {new_name}")

    # Rename holdout shards: first half -> validation, second half -> test
    for i, (path, _month) in enumerate(holdout_shards):
        if i < n_val:
            split = "validation"
            split_idx = i
            split_total = n_val
        else:
            split = "test"
            split_idx = i - n_val
            split_total = n_test

        new_name = f"{split}-{split_idx:05d}-of-{split_total:05d}.parquet"
        new_path = path.parent / new_name
        path.rename(new_path)
        final_paths.append(new_path)
        log(f"  {path.name} -> {new_name}")

    # Phase 3: Summary
    log(f"\n=== Summary ===")
    log(f"Total games: {total_games:,}")
    log(f"Shards: {n_shards} ({n_train} train, {n_val} val, {n_test} test)")

    total_size = sum(p.stat().st_size for p in final_paths)
    log(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    # Phase 4: Upload to HuggingFace
    if args.hf_repo:
        upload_to_hf(args.output, args.hf_repo)

    log("\nDone!")


if __name__ == "__main__":
    main()
