#!/usr/bin/env python3
"""Extract Lichess monthly PGN database dumps into PAWN-compatible Parquet.

Downloads zstd-compressed PGN files from database.lichess.org, parses games
via the Rust chess engine (tokens, clocks, evals, headers), builds Polars
DataFrames, and writes sharded Parquet to disk with train/val/test splits.

All months are downloaded and parsed in parallel (one process per month).
Holdout val/test data is uniformly sampled in a single pass via oversampling.

The output schema stores pre-tokenized move sequences as list[int16],
clock annotations as list[uint16] (seconds remaining, 0x8000=missing),
eval annotations as list[int16] (centipawns, mate=±(32767-N),
0x8000=missing), and metadata columns. Player usernames are hashed to
uint64 via Polars xxHash64 (deterministic within a Polars version).

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
MAX_PLY = 255
SHARD_TARGET_GAMES = 1_000_000
# Oversample rate for holdout: accumulate ~10x target, then downsample.
# Each game independently has probability OVERSAMPLE_FACTOR * target / total
# of being kept, giving a uniform final sample.
OVERSAMPLE_FACTOR = 10


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
    """Download a Lichess zstd PGN dump to disk. Atomic write."""
    url = LICHESS_URL_TEMPLATE.format(year_month=year_month)
    zst_path = output_dir / f"lichess_{year_month}.pgn.zst"
    if zst_path.exists():
        log(f"Using cached {zst_path} ({zst_path.stat().st_size / 1e9:.1f} GB)", prefix)
        return zst_path

    log(f"Downloading {url}", prefix)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "pawn-lichess-extract/1.0")
    response = urllib.request.urlopen(req)

    tmp_path = zst_path.with_suffix(".zst.tmp")
    t0 = time.monotonic()
    downloaded = 0
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
        tmp_path.rename(zst_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    response.close()
    dt = time.monotonic() - t0
    rate = (downloaded / 1e6) / dt if dt > 0 else 0
    log(f"Downloaded {downloaded / 1e9:.2f} GB in {dt:.0f}s ({rate:.0f} MB/s)", prefix)
    return zst_path


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def numpy_rows_to_list_series(
    arr: np.ndarray, lengths: np.ndarray, name: str, inner_dtype: pl.DataType
) -> pl.Series:
    """Convert a 0-padded (N, max_ply) numpy array to a Polars List series,
    trimming each row to its actual game length."""
    rows = [arr[i, :lengths[i]].tolist() for i in range(len(arr))]
    return pl.Series(name, rows, dtype=pl.List(inner_dtype))


def batch_to_dataframe(parsed: dict) -> pl.DataFrame:
    """Convert a parsed batch dict from Rust into a Polars DataFrame."""
    tokens: np.ndarray = parsed["tokens"]
    n = tokens.shape[0]
    if n == 0:
        return pl.DataFrame()

    lengths: np.ndarray = parsed["game_lengths"]

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
        "tokens": numpy_rows_to_list_series(tokens, lengths, "tokens", pl.Int16),
        "clock": numpy_rows_to_list_series(parsed["clocks"], lengths, "clock", pl.UInt16),
        "eval": numpy_rows_to_list_series(parsed["evals"], lengths, "eval", pl.Int16),
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
# Worker: process a single training month
# ---------------------------------------------------------------------------

def process_train_month(
    year_month: str,
    output_dir: Path,
    batch_size: int,
    shard_size: int,
    max_games: int | None,
) -> list[Path]:
    """Download, parse, and write shards for a training month. Runs in a subprocess."""
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
            parsed = chess_engine.parse_pgn_enriched(
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

    # Flush remainder
    if buffer_frames:
        combined = pl.concat(buffer_frames)
        if len(combined) > 0:
            flush_shard(combined)

    log(f"Done — {total_games:,} games, {len(shard_paths)} shards", prefix)
    return shard_paths


# ---------------------------------------------------------------------------
# Worker: process holdout month (single-pass oversample)
# ---------------------------------------------------------------------------

def process_holdout_month(
    year_month: str,
    output_dir: Path,
    batch_size: int,
    holdout_games: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Download, parse, oversample, and write val/test shards. Runs in a subprocess.

    Single-pass: each batch is parsed, split by date half, and independently
    oversampled with fraction = OVERSAMPLE_FACTOR * target / estimated_total.
    After all batches, the oversampled pools are downsampled to exactly
    holdout_games each. This is mathematically equivalent to uniform sampling.
    """
    prefix = f"holdout/{year_month}"
    log("Starting", prefix)

    zst_path = download_zst(year_month, output_dir, prefix)

    year, mon = year_month.split("-")
    midpoint = datetime(int(year), int(mon), 15)

    # Estimate: ~90M rated games per month, ~45M per half
    est_per_half = 45_000_000
    oversample_frac = min(1.0, OVERSAMPLE_FACTOR * holdout_games / est_per_half)
    log(f"Oversample fraction: {oversample_frac:.4f} (target {holdout_games:,} × {OVERSAMPLE_FACTOR}x from ~{est_per_half:,})", prefix)

    val_frames: list[pl.DataFrame] = []
    test_frames: list[pl.DataFrame] = []
    val_total = 0
    test_total = 0
    total_games = 0

    with open(zst_path, "rb") as f:
        for pgn_text, _ in stream_pgn_games(f, batch_size):
            if not pgn_text.strip():
                continue

            t0 = time.monotonic()
            parsed = chess_engine.parse_pgn_enriched(
                pgn_text, max_ply=MAX_PLY, max_games=batch_size * 2, min_ply=1
            )
            dt = time.monotonic() - t0

            df = batch_to_dataframe(parsed)
            if df.is_empty():
                continue

            total_games += len(df)

            # Split by date half
            val_batch = df.filter(pl.col("date") < midpoint)
            test_batch = df.filter(pl.col("date") >= midpoint)

            # Oversample each half independently
            if len(val_batch) > 0:
                sampled = val_batch.sample(fraction=oversample_frac)
                if len(sampled) > 0:
                    val_frames.append(sampled)
                    val_total += len(sampled)

            if len(test_batch) > 0:
                sampled = test_batch.sample(fraction=oversample_frac)
                if len(sampled) > 0:
                    test_frames.append(sampled)
                    test_total += len(sampled)

            rate = len(df) / dt if dt > 0 else 0
            log(f"Parsed {len(df):,} ({rate:,.0f}/s) | val pool: {val_total:,}, test pool: {test_total:,}", prefix)

    # Final downsample to exactly holdout_games
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    val_paths = []
    test_paths = []

    if val_frames:
        val_pool = pl.concat(val_frames)
        n_val = min(holdout_games, len(val_pool))
        val_df = val_pool.sample(n=n_val, seed=seed)
        path = data_dir / "validation-00000.parquet"
        val_df.write_parquet(path, compression="zstd", compression_level=3)
        val_paths.append(path)
        log(f"Val: sampled {n_val:,} from {len(val_pool):,} pool ({len(val_pool) / max(val_total, 1) * 100:.0f}% kept)", prefix)

    if test_frames:
        test_pool = pl.concat(test_frames)
        n_test = min(holdout_games, len(test_pool))
        test_df = test_pool.sample(n=n_test, seed=seed + 1)
        path = data_dir / "test-00000.parquet"
        test_df.write_parquet(path, compression="zstd", compression_level=3)
        test_paths.append(path)
        log(f"Test: sampled {n_test:,} from {len(test_pool):,} pool", prefix)

    log(f"Done — {total_games:,} games processed", prefix)
    return val_paths, test_paths


# ---------------------------------------------------------------------------
# Multiprocessing wrappers (picklable top-level functions)
# ---------------------------------------------------------------------------

def _worker_train(args):
    """Wrapper for process_train_month that returns serializable results."""
    year_month, output_dir, batch_size, shard_size, max_games = args
    paths = process_train_month(year_month, Path(output_dir), batch_size, shard_size, max_games)
    return [str(p) for p in paths]


def _worker_holdout(args):
    """Wrapper for process_holdout_month that returns serializable results."""
    year_month, output_dir, batch_size, holdout_games, seed = args
    val_paths, test_paths = process_holdout_month(
        year_month, Path(output_dir), batch_size, holdout_games, seed
    )
    return [str(p) for p in val_paths], [str(p) for p in test_paths]


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
        help="Games per batch during parsing (controls memory usage)"
    )
    parser.add_argument(
        "--shard-size", type=int, default=SHARD_TARGET_GAMES,
        help="Target games per output shard"
    )
    parser.add_argument(
        "--holdout-month", type=str, default=None,
        help="Month for val/test (e.g. 2026-01). First half -> val, second half -> test."
    )
    parser.add_argument(
        "--holdout-games", type=int, default=50_000,
        help="Games per holdout split (default: 50000)"
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Stop after this many training games per month (for testing)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for holdout sampling (default: 42)"
    )
    args = parser.parse_args()

    log("=== Lichess Parquet Extraction ===")
    log(f"Training months: {args.months}")
    if args.holdout_month:
        log(f"Holdout month: {args.holdout_month} ({args.holdout_games:,} games/split)")
    log(f"Output: {args.output}")
    log(f"Batch size: {args.batch_size:,}, Shard size: {args.shard_size:,}")
    log(f"Parallelism: {len(args.months) + (1 if args.holdout_month else 0)} workers")
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
            args.holdout_games, args.seed,
        )

    n_workers = len(train_args) + (1 if holdout_args else 0)
    log(f"Spawning {n_workers} workers...")

    # Use spawn context to avoid Rust rayon + fork deadlocks
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        # Submit all jobs
        train_results = pool.map_async(_worker_train, train_args)
        holdout_result = None
        if holdout_args:
            holdout_result = pool.apply_async(_worker_holdout, (holdout_args,))

        # Collect results
        train_shard_lists = train_results.get()
        val_paths = []
        test_paths = []
        if holdout_result:
            val_str, test_str = holdout_result.get()
            val_paths = [Path(p) for p in val_str]
            test_paths = [Path(p) for p in test_str]

    # ── Rename train shards to HF-compatible names ─────────────────────
    all_train_paths = []
    for shard_list in train_shard_lists:
        all_train_paths.extend(Path(p) for p in shard_list)

    # Sort by name to maintain chronological order
    all_train_paths.sort(key=lambda p: p.name)
    n_train = len(all_train_paths)

    log(f"\n=== Renaming {n_train} train shards ===")
    final_paths = []
    for i, path in enumerate(all_train_paths):
        new_name = f"train-{i:05d}-of-{n_train:05d}.parquet"
        new_path = path.parent / new_name
        path.rename(new_path)
        final_paths.append(new_path)

    # Rename val/test (already single shards, just fix the -of-N suffix)
    for i, path in enumerate(val_paths):
        new_name = f"validation-{i:05d}-of-{len(val_paths):05d}.parquet"
        new_path = path.parent / new_name
        if path != new_path:
            path.rename(new_path)
        final_paths.append(new_path)

    for i, path in enumerate(test_paths):
        new_name = f"test-{i:05d}-of-{len(test_paths):05d}.parquet"
        new_path = path.parent / new_name
        if path != new_path:
            path.rename(new_path)
        final_paths.append(new_path)

    # ── Summary ────────────────────────────────────────────────────────
    log(f"\n=== Summary ===")
    log(f"Train: {n_train} shards")
    log(f"Validation: {len(val_paths)} shards")
    log(f"Test: {len(test_paths)} shards")

    if final_paths:
        total_size = sum(p.stat().st_size for p in final_paths)
        log(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    # ── Upload to HuggingFace ──────────────────────────────────────────
    if args.hf_repo:
        upload_to_hf(args.output, args.hf_repo)

    log("\nDone!")


if __name__ == "__main__":
    main()
