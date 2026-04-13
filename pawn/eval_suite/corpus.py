"""Evaluation corpus generation, loading, and sanity checks.

Storage layout:
    corpus_dir/
        move_ids.npy          — int16[n_games, max_ply]   (mmap for replay)
        games.parquet         — game-level metadata        (1M rows)
        positions/*.parquet   — one row per valid position (~237M rows)
        metadata.json         — generation parameters
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import chess_engine as engine

# ---------------------------------------------------------------------------
# Popcount helper (for legal-move counting from bit-packed grids)
# ---------------------------------------------------------------------------

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(arr.shape, dtype=np.uint32)
    for shift in range(0, 64, 8):
        byte = ((arr >> shift) & 0xFF).astype(np.uint8)
        result += _POPCOUNT_LUT[byte].astype(np.uint32)
    return result


def _count_legal_moves(move_ids: np.ndarray, game_lengths: np.ndarray) -> np.ndarray:
    """Legal move count per ply via bit-packed grids + promo mask."""
    grid, promo_mask = engine.compute_legal_move_masks(move_ids, game_lengths)
    grid_counts = np.zeros(grid.shape[:2], dtype=np.uint32)
    for sq in range(64):
        grid_counts += _popcount_u64(grid[:, :, sq])

    promo_pairs = engine.export_move_vocabulary()["promo_pairs"]
    adj = np.zeros(grid.shape[:2], dtype=np.int32)
    for i, (src, dst) in enumerate(promo_pairs):
        bit = ((grid[:, :, src] >> dst) & 1).astype(np.int32)
        n_pt = promo_mask[:, :, i, :].sum(axis=-1).astype(np.int32)
        has = (n_pt > 0).astype(np.int32)
        adj += (n_pt - 1) * bit * has
    return (grid_counts.astype(np.int32) + adj).astype(np.uint16)


def _term_to_outcome(tc: int, gl: int) -> str:
    if tc == 0:
        return "WHITE_CHECKMATES" if gl % 2 == 1 else "BLACK_CHECKMATES"
    if tc == 1:
        return "STALEMATE"
    if tc in (2, 3, 4):
        return "DRAW_BY_RULE"
    return "PLY_LIMIT"


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

_LMC_SUB = 2000   # sub-batch size for compute_legal_move_masks (memory)
_POS_BATCH = 50_000  # games per positions parquet part


def generate_corpus(
    output_dir: str | Path,
    n_games: int = 1_000_000,
    max_ply: int = 255,
    seed: int = 99_999,
    batch_size: int = 10_000,
) -> Path:
    """Generate corpus and write parquet + npy to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pos_dir = output_dir / "positions"
    pos_dir.mkdir(exist_ok=True)

    all_move_ids, all_gl, all_tc = [], [], []
    # Accumulate per-game data for games.parquet
    all_game_rows: list[dict] = []

    n_batches = (n_games + batch_size - 1) // batch_size
    t0 = time.time()
    pos_part = 0  # parquet partition counter
    game_offset = 0

    for batch_idx in range(n_batches):
        batch_n = min(batch_size, n_games - batch_idx * batch_size)
        batch_seed = seed + batch_idx * 1_000_000

        print(
            f"  Batch {batch_idx + 1}/{n_batches}: {batch_n} games "
            f"(seed={batch_seed})...",
            end="", flush=True,
        )

        move_ids, gl, tc = engine.generate_random_games(batch_n, max_ply, batch_seed)

        # Legal move counts in sub-batches
        lmc_parts = []
        for s in range(0, batch_n, _LMC_SUB):
            e = min(s + _LMC_SUB, batch_n)
            lmc_parts.append(_count_legal_moves(move_ids[s:e], gl[s:e]))
        legal_counts = np.concatenate(lmc_parts, axis=0)
        del lmc_parts

        # is_check from board states
        _, _, _, _, is_check, _ = engine.extract_board_states(move_ids, gl)

        # Accumulate game-level metadata
        for g in range(batch_n):
            gidx = game_offset + g
            g_len = int(gl[g])
            all_game_rows.append({
                "game_idx": np.uint32(gidx),
                "game_length": np.uint16(g_len),
                "term_code": np.uint8(tc[g]),
                "outcome": _term_to_outcome(int(tc[g]), g_len),
            })

        # Write positions parquet in _POS_BATCH-game chunks
        for sub_start in range(0, batch_n, _POS_BATCH):
            sub_end = min(sub_start + _POS_BATCH, batch_n)
            sub_gl = gl[sub_start:sub_end].astype(np.int32)
            sub_n = sub_end - sub_start
            max_len = int(sub_gl.max())

            ply_grid = np.arange(max_len, dtype=np.uint16)[None, :]
            valid = ply_grid < sub_gl[:, None]

            gidx_arr = np.broadcast_to(
                np.arange(game_offset + sub_start,
                          game_offset + sub_end, dtype=np.uint32)[:, None],
                (sub_n, max_len),
            )[valid]

            df = pl.DataFrame({
                "game_idx": gidx_arr,
                "ply": np.broadcast_to(ply_grid, (sub_n, max_len))[valid],
                "k": np.asarray(legal_counts[sub_start:sub_end, :max_len])[valid].astype(np.uint16),
                "is_check": np.asarray(is_check[sub_start:sub_end, :max_len])[valid],
            })
            df.write_parquet(pos_dir / f"part_{pos_part:04d}.parquet")
            pos_part += 1
            del df

        all_move_ids.append(move_ids)
        all_gl.append(gl)
        all_tc.append(tc)
        game_offset += batch_n
        del legal_counts, is_check

        print(f" done ({time.time() - t0:.0f}s)")

    # Write move_ids.npy
    move_ids_all = np.concatenate(all_move_ids, axis=0)
    np.save(output_dir / "move_ids.npy", move_ids_all)
    del all_move_ids, move_ids_all

    # Write games.parquet
    games_df = pl.DataFrame(all_game_rows)
    games_df.write_parquet(output_dir / "games.parquet")

    # Write game_lengths and term_codes as small .npy (still needed for engine calls)
    gl_all = np.concatenate(all_gl)
    tc_all = np.concatenate(all_tc)
    np.save(output_dir / "game_lengths.npy", gl_all)
    np.save(output_dir / "termination_codes.npy", tc_all)

    metadata = {
        "seed": seed, "n_games": n_games, "max_ply": max_ply,
        "batch_size": batch_size,
        "generation_time_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "format": "parquet",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Corpus saved to {output_dir} ({time.time() - t0:.0f}s)")
    return output_dir


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_corpus(corpus_dir: str | Path) -> dict:
    """Load corpus from parquet format.

    Returns dict with:
        move_ids:           np.ndarray (mmap)
        game_lengths:       np.ndarray
        termination_codes:  np.ndarray
        games:              pl.LazyFrame  (scan of games.parquet)
        positions:          pl.LazyFrame  (scan of positions/*.parquet)
        metadata:           dict | None
    """
    d = Path(corpus_dir)

    corpus = {
        "move_ids": np.load(d / "move_ids.npy", mmap_mode="r"),
        "game_lengths": np.load(d / "game_lengths.npy"),
        "termination_codes": np.load(d / "termination_codes.npy"),
        "games": pl.scan_parquet(d / "games.parquet"),
        "positions": pl.scan_parquet(d / "positions" / "*.parquet"),
    }
    meta_path = d / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            corpus["metadata"] = json.load(f)
    else:
        corpus["metadata"] = None
    return corpus


# ---------------------------------------------------------------------------
# Sanity checks (byte-level comparison — stays numpy)
# ---------------------------------------------------------------------------


def sanity_checks(corpus: dict) -> dict:
    """Duplicate detection and common-prefix analysis on raw move sequences."""
    move_ids = corpus["move_ids"]
    game_lengths = corpus["game_lengths"]
    n = len(move_ids)

    print("  Converting games to byte strings...")
    game_bytes = [bytes(move_ids[i, : int(game_lengths[i])]) for i in range(n)]

    print("  Sorting...")
    order = sorted(range(n), key=lambda i: game_bytes[i])
    sorted_bytes = [game_bytes[order[i]] for i in range(n)]
    del game_bytes

    print("  Comparing adjacent pairs...")
    max_prefix = 0
    dupes = 0
    dupe_pairs: list[tuple[int, int]] = []
    prefix_hist: dict[int, int] = {}

    for i in range(n - 1):
        a, b = sorted_bytes[i], sorted_bytes[i + 1]
        common = 0
        for c in range(min(len(a), len(b))):
            if a[c] != b[c]:
                break
            common += 1
        else:
            common = min(len(a), len(b))
        pm = common // 2
        max_prefix = max(max_prefix, pm)
        prefix_hist[pm] = prefix_hist.get(pm, 0) + 1
        if a == b:
            dupes += 1
            dupe_pairs.append((order[i], order[i + 1]))

    del sorted_bytes
    return {
        "duplicates": dupes,
        "max_prefix_moves": max_prefix,
        "prefix_length_histogram": dict(sorted(prefix_hist.items())),
        "duplicate_pairs": dupe_pairs[:20],
    }


# ---------------------------------------------------------------------------
# Summary statistics (Polars lazy scan)
# ---------------------------------------------------------------------------


def summary_stats(corpus: dict) -> dict:
    """Compute summary statistics by iterating over position parquet parts.

    Each part (~12M rows) is loaded, aggregated, and freed — peak memory
    is bounded by the largest single part file (~100 MB).
    """
    games_lf = corpus["games"]
    gl = corpus["game_lengths"]
    n = len(gl)
    gl_hist, gl_bins = np.histogram(gl, bins=50)

    outcome_rates = {
        r["outcome"]: r["count"] / n
        for r in (
            games_lf.group_by("outcome")
            .agg(pl.len().alias("count"))
            .collect()
            .iter_rows(named=True)
        )
    }

    acc = _new_accumulator()
    for part_df in _iter_position_parts(corpus):
        _accumulate(acc, part_df.filter(pl.col("k") > 0))

    return {
        "n_games": n,
        "total_positions": acc["n"],
        "game_length": {
            "mean": float(gl.mean()), "median": float(np.median(gl)),
            "std": float(gl.std()), "min": int(gl.min()), "max": int(gl.max()),
            "histogram_counts": gl_hist.tolist(), "histogram_edges": gl_bins.tolist(),
        },
        "legal_move_counts": _finalize_k_stats(acc),
        "k_histogram": _finalize_k_hist(acc),
        "outcome_rates": outcome_rates,
        "phase_stats": _finalize_phases(acc),
        "check_stats": _finalize_checks(acc),
    }


# ---------------------------------------------------------------------------
# File-level accumulator (shared by summary_stats and bounds)
# ---------------------------------------------------------------------------

_PHASES = [("ply_1_20", 0, 20), ("ply_21_80", 20, 80),
           ("ply_81_150", 80, 150), ("ply_150_plus", 150, 9999)]


def _iter_position_parts(corpus: dict) -> Iterator[pl.DataFrame]:
    """Yield each position parquet part as an eager DataFrame."""
    from pathlib import Path
    # Find corpus dir from the LazyFrame's file path
    # Positions are at corpus_dir/positions/*.parquet
    # We stored move_ids.npy at corpus root — derive the dir from it
    pos_dir = Path(str(corpus["move_ids"].filename)).parent / "positions"
    for f in sorted(pos_dir.glob("*.parquet")):
        yield pl.read_parquet(f)


def _new_accumulator() -> dict[str, Any]:
    return {
        "n": 0, "sum_k": 0.0, "sum_k_sq": 0.0, "k_min": 999, "k_max": 0,
        "sum_inv_k": 0.0, "sum_inv_k_sq": 0.0,
        "sum_ln_k": 0.0, "sum_ln_k_sq": 0.0,
        "sum_top5": 0.0, "sum_top5_sq": 0.0,
        "k_hist": np.zeros(300, dtype=np.int64),
        **{f"chk_{s}": 0.0 for s in ("n", "sum_k", "sum_inv_k")},
        **{f"nochk_{s}": 0.0 for s in ("n", "sum_k", "sum_inv_k")},
        **{f"{p}_{s}": 0.0 for p, _, _ in _PHASES for s in ("n", "sum_k", "sum_inv_k", "sum_ln_k")},
    }


def _accumulate(acc: dict, df: pl.DataFrame) -> None:
    """Accumulate stats from one chunk (already filtered to k > 0)."""
    k = df["k"].to_numpy().astype(np.float64)
    ply = df["ply"].to_numpy()
    chk = df["is_check"].to_numpy()
    nk = len(k)
    if nk == 0:
        return

    inv_k = 1.0 / k
    ln_k = np.log(k)
    top5 = np.minimum(5.0, k) / k

    acc["n"] += nk
    acc["sum_k"] += k.sum()
    acc["sum_k_sq"] += (k ** 2).sum()
    acc["k_min"] = min(acc["k_min"], int(k.min()))
    acc["k_max"] = max(acc["k_max"], int(k.max()))
    acc["sum_inv_k"] += inv_k.sum()
    acc["sum_inv_k_sq"] += (inv_k ** 2).sum()
    acc["sum_ln_k"] += ln_k.sum()
    acc["sum_ln_k_sq"] += (ln_k ** 2).sum()
    acc["sum_top5"] += top5.sum()
    acc["sum_top5_sq"] += (top5 ** 2).sum()
    np.add.at(acc["k_hist"], np.clip(k.astype(np.int64), 0, 299), 1)

    # Check
    for label, mask in [("chk", chk), ("nochk", ~chk)]:
        if mask.any():
            km = k[mask]
            acc[f"{label}_n"] += int(mask.sum())
            acc[f"{label}_sum_k"] += km.sum()
            acc[f"{label}_sum_inv_k"] += (1.0 / km).sum()

    # Phase
    for name, lo, hi in _PHASES:
        mask = (ply >= lo) & (ply < hi)
        if mask.any():
            km = k[mask]
            acc[f"{name}_n"] += int(mask.sum())
            acc[f"{name}_sum_k"] += km.sum()
            acc[f"{name}_sum_inv_k"] += (1.0 / km).sum()
            acc[f"{name}_sum_ln_k"] += np.log(km).sum()

    del k, ply, chk, inv_k, ln_k, top5


def _finalize_k_stats(acc: dict) -> dict[str, float | int]:
    N = acc["n"]
    mean = acc["sum_k"] / N
    var = acc["sum_k_sq"] / N - mean ** 2
    # Median from histogram
    cs = np.cumsum(acc["k_hist"][1:])  # skip k=0
    median = int(np.searchsorted(cs, N / 2) + 1) if len(cs) > 0 else 0
    return {"mean": float(mean), "median": median, "std": float(np.sqrt(max(var, 0))),
            "min": acc["k_min"], "max": acc["k_max"]}


def _finalize_k_hist(acc: dict) -> dict:
    h = acc["k_hist"]
    nz = h > 0
    return {"values": np.arange(300)[nz].tolist(), "counts": h[nz].tolist(), "total": acc["n"]}


def _finalize_phases(acc: dict) -> dict:
    result = {}
    for name, _, _ in _PHASES:
        c = acc[f"{name}_n"]
        if c > 0:
            result[name] = {
                "mean_k": acc[f"{name}_sum_k"] / c,
                "e_1_over_k": acc[f"{name}_sum_inv_k"] / c,
                "e_ln_k": acc[f"{name}_sum_ln_k"] / c,
                "n_positions": int(c),
            }
    return result


def _finalize_checks(acc: dict) -> dict:
    N = acc["n"]
    result = {}
    for label in ("chk", "nochk"):
        c = acc[f"{label}_n"]
        if c > 0:
            name = "in_check" if label == "chk" else "not_in_check"
            result[name] = {
                "mean_k": acc[f"{label}_sum_k"] / c,
                "e_1_over_k": acc[f"{label}_sum_inv_k"] / c,
                "frequency": c / N,
            }
    return result
