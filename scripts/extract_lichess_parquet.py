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


def download_zst(year_month: str, output_dir: Path) -> Path:
    """Download a Lichess zstd PGN dump to disk. Returns path to .zst file."""
    url = LICHESS_URL_TEMPLATE.format(year_month=year_month)
    zst_path = output_dir / f"lichess_{year_month}.pgn.zst"
    if zst_path.exists():
        log(f"  Using cached {zst_path} ({zst_path.stat().st_size / 1e9:.1f} GB)")
        return zst_path

    log(f"  Downloading {url}")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "pawn-lichess-extract/1.0")
    response = urllib.request.urlopen(req)

    t0 = time.monotonic()
    downloaded = 0
    with open(zst_path, "wb") as f:
        while True:
            chunk = response.read(8 * 1024 * 1024)  # 8 MB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.monotonic() - t0
            rate_mb = (downloaded / 1e6) / elapsed if elapsed > 0 else 0
            print(f"\r  Downloaded {downloaded / 1e9:.2f} GB ({rate_mb:.0f} MB/s)", end="", flush=True)
    print(flush=True)

    response.close()
    log(f"  Saved {zst_path} ({downloaded / 1e9:.2f} GB in {time.monotonic() - t0:.0f}s)")
    return zst_path


def stream_pgn_from_zst(zst_path: Path, batch_size: int):
    """Yield PGN text batches from a local .zst file."""
    with open(zst_path, "rb") as f:
        yield from stream_pgn_games(f, batch_size)


def download_month(year_month: str, output_dir: Path, batch_size: int):
    """Download and parse a single month's PGN dump, yielding parsed batches."""
    zst_path = download_zst(year_month, output_dir)
    total_games = 0
    batch_num = 0

    for pgn_text, n_games_in_chunk in stream_pgn_from_zst(zst_path, batch_size):
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

    log(f"  [{year_month}] Done — {total_games:,} games total")


def sample_holdout(
    zst_path: Path,
    date_start: str,
    date_end: str,
    n_games: int,
    batch_size: int,
    seed: int,
) -> pl.DataFrame:
    """Uniformly sample n_games from a date range within a .zst PGN dump.

    Two-pass approach:
    1. Count games in the date range (header-only scan, no tokenization)
    2. Generate random indices, parse only those games
    """
    # Pass 1: count
    log(f"  Pass 1: counting games in [{date_start}, {date_end}]")
    total_in_range = 0
    t0 = time.monotonic()
    for pgn_text, _ in stream_pgn_from_zst(zst_path, batch_size):
        if not pgn_text.strip():
            continue
        total_in_range += chess_engine.count_pgn_games_in_date_range(
            pgn_text, date_start, date_end
        )
    dt = time.monotonic() - t0
    log(f"  Found {total_in_range:,} games in range ({dt:.1f}s)")

    if total_in_range == 0:
        return pl.DataFrame()

    # Generate random sample indices
    actual_n = min(n_games, total_in_range)
    rng = np.random.default_rng(seed)
    indices = set(rng.choice(total_in_range, size=actual_n, replace=False).tolist())
    log(f"  Sampling {actual_n:,} of {total_in_range:,} games (seed={seed})")

    # Pass 2: parse only sampled games
    log(f"  Pass 2: parsing sampled games")
    frames = []
    game_offset = 0
    t0 = time.monotonic()
    for pgn_text, _ in stream_pgn_from_zst(zst_path, batch_size):
        if not pgn_text.strip():
            continue

        # Count how many date-matching games are in this chunk
        chunk_count = chess_engine.count_pgn_games_in_date_range(
            pgn_text, date_start, date_end
        )

        # Check if any of our target indices fall in this chunk's range
        chunk_indices = [
            i for i in range(game_offset, game_offset + chunk_count)
            if i in indices
        ]

        if chunk_indices:
            parsed = chess_engine.parse_pgn_sampled(
                pgn_text,
                chunk_indices,
                date_start,
                date_end,
                game_offset=game_offset,
                max_ply=MAX_PLY,
                min_ply=1,
            )
            n = parsed["tokens"].shape[0]
            if n > 0:
                frames.append(batch_to_dataframe(parsed))

        game_offset += chunk_count

    dt = time.monotonic() - t0
    if not frames:
        log(f"  No games parsed")
        return pl.DataFrame()

    result = pl.concat(frames)
    log(f"  Parsed {len(result):,} games ({dt:.1f}s)")
    return result


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
        # Write with placeholder name; rename after all shards are counted
        path = self.output_dir / f"{self.split}-temp-{self.shard_idx:05d}.parquet"
        df.write_parquet(path, compression="zstd", compression_level=3)
        size_mb = path.stat().st_size / 1024 / 1024
        log(f"  [{self.split}] shard {self.shard_idx}: {len(df):,} games, {size_mb:.1f} MB")
        self.shard_paths.append(path)
        self.shard_idx += 1

    def rename_shards(self) -> list[Path]:
        """Rename temp shards to HF-compatible names with correct total count."""
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
        "--holdout-month", type=str, default=None,
        help="Month to use for val/test (e.g. 2023-12). First half of month "
             "-> val, second half -> test. Randomly samples --holdout-games "
             "from each half."
    )
    parser.add_argument(
        "--holdout-games", type=int, default=50_000,
        help="Number of games to sample for each of val and test (default: 50000)"
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Stop after this many training games (for testing)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for holdout sampling (default: 42)"
    )
    args = parser.parse_args()

    log("=== Lichess Parquet Extraction ===")
    log(f"Training months: {args.months}")
    log(f"Output: {args.output}")
    log(f"Batch size: {args.batch_size:,}")
    log(f"Shard size: {args.shard_size:,}")
    if args.holdout_month:
        log(f"Holdout month: {args.holdout_month}")
        log(f"Holdout games per split: {args.holdout_games:,}")
        log(f"Holdout seed: {args.seed}")
    if args.max_games:
        log(f"Max training games: {args.max_games:,}")
    log("")

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Process holdout month (val/test) ──────────────────────
    buffers = {
        "train": SplitBuffer("train", args.shard_size, args.output),
        "validation": SplitBuffer("validation", args.shard_size, args.output),
        "test": SplitBuffer("test", args.shard_size, args.output),
    }

    if args.holdout_month:
        log(f"\n=== Processing holdout month {args.holdout_month} ===")

        # Download the zstd file (reused for both count and parse passes)
        zst_path = download_zst(args.holdout_month, args.output)

        # Date ranges: first half of month -> val, second half -> test
        # UTCDate format is "YYYY.MM.DD"
        year, mon = args.holdout_month.split("-")
        val_start = f"{year}.{mon}.01"
        val_end = f"{year}.{mon}.14"
        test_start = f"{year}.{mon}.15"
        test_end = f"{year}.{mon}.31"

        log(f"  Val date range: [{val_start}, {val_end}]")
        val_df = sample_holdout(
            zst_path, val_start, val_end,
            args.holdout_games, args.batch_size, args.seed,
        )
        if not val_df.is_empty():
            buffers["validation"].add(val_df)

        log(f"  Test date range: [{test_start}, {test_end}]")
        test_df = sample_holdout(
            zst_path, test_start, test_end,
            args.holdout_games, args.batch_size, args.seed + 1,
        )
        if not test_df.is_empty():
            buffers["test"].add(test_df)

    # ── Phase 2: Process training months ───────────────────────────────
    total_train = 0
    stop = False

    for month in args.months:
        if stop:
            break
        log(f"\n=== Processing {month} (train) ===")

        for parsed in download_month(month, args.output, args.batch_size):
            df = batch_to_dataframe(parsed)
            if df.is_empty():
                continue

            if args.max_games:
                remaining = args.max_games - total_train
                if remaining <= 0:
                    stop = True
                    break
                if len(df) > remaining:
                    df = df.head(remaining)

            total_train += len(df)
            buffers["train"].add(df)

            if args.max_games and total_train >= args.max_games:
                stop = True
                break

    # Flush remaining data in each buffer
    for buf in buffers.values():
        buf.flush_remaining()

    log(f"\n=== Renaming shards ===")
    final_paths = []
    for buf in buffers.values():
        if buf.shard_paths:
            final_paths.extend(buf.rename_shards())

    # Summary
    log(f"\n=== Summary ===")
    total_games = sum(buf.total_games for buf in buffers.values())
    log(f"Total games: {total_games:,}")
    for name, buf in buffers.items():
        if buf.total_games > 0:
            log(f"  {name}: {buf.total_games:,} games, {len(buf.shard_paths)} shards")

    if final_paths:
        total_size = sum(p.stat().st_size for p in final_paths)
        log(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    # Upload to HuggingFace
    if args.hf_repo:
        upload_to_hf(args.output, args.hf_repo)

    log("\nDone!")


if __name__ == "__main__":
    main()
