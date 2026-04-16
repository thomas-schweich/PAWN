#!/usr/bin/env python3
"""Compute the unconditional top-1 accuracy ceiling for random chess play.

PAWN's training distribution is uniformly random legal play. At each
position with N legal moves, the next move is drawn uniformly from the
N legal moves, so the Bayes-optimal predictor that does not know the
game outcome can do no better than 1/N at that position. Averaged over
the position distribution induced by random play, the top-1 accuracy
ceiling is therefore::

    E[1/N_legal]

where the expectation is over positions sampled from random games.

This is **not** equal to ``1 / E[N_legal]``: by Jensen's inequality
(``1/x`` is convex), ``E[1/N] >= 1/E[N]``, with equality only if N is
constant. Computing the ceiling honestly requires evaluating ``1/N`` at
each position and then averaging.

The previous version of this script also computed an outcome-
conditioned ceiling via Monte Carlo rollouts, which is meaningful when
the model gets to see the game's actual outcome as a prefix token (the
v0.x outcome-conditioned PAWN backbones did). The v1.0.0 backbones do
not use outcome conditioning, so the MC ceiling collapses to the
unconditional ceiling and the rollouts are wasted work. This script
no longer does them.

Usage::

    uv run python scripts/compute_theoretical_ceiling.py
    uv run python scripts/compute_theoretical_ceiling.py --n-games 50000
    uv run python scripts/compute_theoretical_ceiling.py --max-ply 512 --n-games 20000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import chess_engine as engine
from pawn.config import CLMConfig, PAD_TOKEN


def bootstrap_ci_clustered(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean + CI resampling by cluster (game), not by position.

    Position-level resampling underestimates variance because positions
    within a single random game are correlated (legal-move counts shift
    smoothly along a game). Resampling whole games at a time gives an
    honest CI.

    Vectorized: precompute per-cluster sum and count once, then each
    bootstrap iteration is a pair of integer-indexed sums of length
    n_clusters — O(n_boot × n_clusters) instead of
    O(n_boot × n_positions). For 50K games × ~360 plies/game and
    n_boot=2000 that's ~100M ops instead of ~36 billion, which is the
    difference between "instant" and "wall clock fall-off-a-cliff."
    """
    rng = np.random.default_rng(seed)
    unique_ids, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_ids)

    cluster_sum = np.zeros(n_clusters, dtype=np.float64)
    cluster_count = np.zeros(n_clusters, dtype=np.int64)
    np.add.at(cluster_sum, inv, values)
    np.add.at(cluster_count, inv, 1)

    sampled = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    sums = cluster_sum[sampled].sum(axis=1)
    counts = cluster_count[sampled].sum(axis=1)
    boot_means = sums / counts

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def compute_ceiling(
    n_games: int,
    max_ply: int,
    seed: int,
    vocab_size: int,
) -> dict:
    """Generate random games and compute E[1/N_legal] over their positions.

    All heavy lifting is done in Rust:
    - ``generate_random_games`` emits ``n_games`` games with up to
      ``max_ply`` plies each.
    - ``compute_legal_token_masks_sparse`` returns flat int64 indices
      into a (batch, seq_len, vocab_size) tensor; one index per
      (game, ply, legal_token). Counting indices per (game, ply) gives
      the per-position legal-move count.

    Then 1/N is computed in numpy and averaged.
    """
    seq_len = max_ply  # no outcome prefix in v1.0.0

    move_ids, game_lengths, _term_codes = engine.generate_random_games(
        n_games, max_ply, seed, False, 0.0,
    )
    move_ids_np = np.asarray(move_ids, dtype=np.int16)
    game_lengths_np = np.asarray(game_lengths, dtype=np.int16)

    sparse = engine.compute_legal_token_masks_sparse(
        move_ids_np, game_lengths_np, seq_len, vocab_size,
    )
    sparse_np = np.asarray(sparse, dtype=np.int64)

    # Each index = b * seq_len * vocab_size + t * vocab_size + token_id.
    # Strip the token_id by integer-dividing by vocab_size, leaving a
    # flat (b * seq_len + t) index per legal token. Counting these gives
    # the legal-move count at each (b, t).
    flat_pos = sparse_np // vocab_size
    counts = np.bincount(flat_pos, minlength=n_games * seq_len)
    counts = counts.reshape(n_games, seq_len)

    # The sparse mask intentionally adds a PAD entry at position
    # t == length (so loss at the end-of-game target is finite). For the
    # ceiling we only care about positions with a real legal move, so
    # mask to t < length.
    game_lengths_int = game_lengths_np.astype(np.int64)
    ply_idx = np.arange(seq_len, dtype=np.int64)[None, :]
    valid = ply_idx < game_lengths_int[:, None]

    n_legal_per_pos = counts[valid]
    if (n_legal_per_pos == 0).any():
        n_zero = int((n_legal_per_pos == 0).sum())
        raise RuntimeError(
            f"{n_zero} sampled positions had zero legal moves — should be "
            "impossible for non-terminal positions in random play."
        )

    inv_n = 1.0 / n_legal_per_pos.astype(np.float64)

    # Build per-position cluster IDs (game index) for clustered bootstrap.
    game_idx_2d = np.broadcast_to(
        np.arange(n_games, dtype=np.int64)[:, None], (n_games, seq_len),
    )
    game_idx_per_pos = game_idx_2d[valid]

    # Per-game length and per-position ply (for the ply-bucket breakdown).
    ply_2d = np.broadcast_to(ply_idx, (n_games, seq_len))
    ply_per_pos = ply_2d[valid]

    return {
        "inv_n": inv_n,
        "n_legal": n_legal_per_pos,
        "game_idx_per_pos": game_idx_per_pos,
        "ply_per_pos": ply_per_pos,
        "game_lengths": game_lengths_int,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute the unconditional top-1 accuracy ceiling for "
                    "PAWN (E[1/N_legal] over random-game positions)."
    )
    parser.add_argument("--n-games", type=int, default=20_000,
                        help="Number of random games to generate.")
    parser.add_argument("--max-ply", type=int, default=None,
                        help="Maximum game length (default: CLMConfig().max_seq_len).")
    parser.add_argument("--seed", type=int, default=77777)
    parser.add_argument("--output", type=str, default="cards/theoretical_ceiling.json",
                        help="Where to save the JSON artifact. Defaults to "
                             "cards/theoretical_ceiling.json (read by "
                             "scripts/generate_model_cards.py to fill in the "
                             "accuracy ceiling section of each model card).")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of clustered-bootstrap resamples for the CI "
                             "(0 to skip).")
    parser.add_argument("--model-accuracy", type=float, default=None,
                        help="If given, also report the ratio of this top-1 "
                             "accuracy to the computed ceiling.")
    args = parser.parse_args()

    cfg = CLMConfig()
    max_ply = args.max_ply if args.max_ply is not None else cfg.max_seq_len
    vocab_size = cfg.vocab_size

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Computing unconditional accuracy ceiling")
    print(f"  Games:      {args.n_games:,}")
    print(f"  Max ply:    {max_ply}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Seed:       {args.seed}")
    print()

    t0 = time.time()
    result = compute_ceiling(args.n_games, max_ply, args.seed, vocab_size)
    elapsed = time.time() - t0

    inv_n = result["inv_n"]
    n_legal = result["n_legal"]
    game_idx_per_pos = result["game_idx_per_pos"]
    ply_per_pos = result["ply_per_pos"]
    game_lengths = result["game_lengths"]

    n_positions = int(inv_n.size)
    ceiling = float(inv_n.mean())
    mean_n_legal = float(n_legal.mean())
    one_over_mean_n = 1.0 / mean_n_legal
    mean_game_len = float(game_lengths.mean())

    print(f"Generated {args.n_games:,} games "
          f"({n_positions:,} non-terminal positions, "
          f"avg {mean_game_len:.1f} ply/game) in {elapsed:.1f}s")
    print()
    print(f"Unconditional ceiling (E[1/N_legal]): {ceiling:.6f}  "
          f"({ceiling * 100:.4f}%)")
    print(f"  Lower bound (1/E[N_legal]):         {one_over_mean_n:.6f}  "
          f"({one_over_mean_n * 100:.4f}%)  [Jensen lower bound, not the ceiling]")
    print(f"  Avg legal moves per position:       {mean_n_legal:.2f}")
    print()

    ci_low: float | None = None
    ci_high: float | None = None
    if args.bootstrap > 0:
        print(f"Bootstrap CI ({args.bootstrap} resamples, clustered by game):")
        _, ci_low, ci_high = bootstrap_ci_clustered(
            inv_n, game_idx_per_pos, n_boot=args.bootstrap,
        )
        print(f"  ceiling = {ceiling:.6f}  95% CI [{ci_low:.6f}, {ci_high:.6f}]")
        print()

    # Per-ply-bucket breakdown — useful for sanity-checking that the
    # ceiling is dominated by the long tail of mid-game positions and
    # not by early-opening or near-terminal positions.
    print("Ceiling by ply bucket:")
    bucket_data = {}
    bucket_edges = [0, 10, 20, 40, 80, 160, 256, 384, max_ply + 1]
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (ply_per_pos >= lo) & (ply_per_pos < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bucket_ceiling = float(inv_n[mask].mean())
        bucket_n_legal = float(n_legal[mask].mean())
        print(f"  plies [{lo:>3}, {hi - 1:>3}]: ceiling={bucket_ceiling * 100:.3f}%  "
              f"avg N={bucket_n_legal:.1f}  n={n:,}")
        bucket_data[f"{lo}-{hi - 1}"] = {
            "ceiling": bucket_ceiling,
            "mean_n_legal": bucket_n_legal,
            "n_positions": n,
        }
    print()

    if args.model_accuracy is not None:
        ma = args.model_accuracy
        ratio = ma / ceiling if ceiling > 0 else 0
        print(f"Model accuracy {ma * 100:.2f}%  →  {ratio * 100:.1f}% of the unconditional ceiling")
        print()

    data = {
        "unconditional_ceiling": ceiling,
        "ceiling_ci_low_95": ci_low,
        "ceiling_ci_high_95": ci_high,
        "mean_n_legal": mean_n_legal,
        "one_over_mean_n_legal": one_over_mean_n,
        "n_games": args.n_games,
        "n_positions": n_positions,
        "max_ply": max_ply,
        "vocab_size": vocab_size,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "by_ply_bucket": bucket_data,
        "method": "E[1/N_legal] over positions sampled from uniformly random games",
    }
    if args.model_accuracy is not None:
        data["model_accuracy"] = args.model_accuracy
        data["model_ratio_vs_ceiling"] = ma / ceiling

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
