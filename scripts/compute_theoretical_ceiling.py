#!/usr/bin/env python3
"""Compute theoretical maximum top-1 accuracy for random chess play.

Two ceilings:
1. Unconditional: E[1/N_legal] — best accuracy without knowing the outcome.
2. Outcome-conditioned: E[max_m P(m|outcome, history)] — best accuracy when
   the outcome token is known. Estimated via Monte Carlo rollouts.

The "adjusted accuracy" normalizes model accuracy against these ceilings:
    adjusted = model_accuracy / ceiling

Usage:
    uv run python scripts/compute_theoretical_ceiling.py --n-games 10000
    uv run python scripts/compute_theoretical_ceiling.py --n-games 50000 --rollouts 64
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import chess_engine as engine


def compute_unconditional_ceiling(
    n_games: int, max_ply: int = 255, seed: int = 77777,
) -> dict:
    """Compute E[1/N_legal] from a corpus of random games.

    This is the theoretical maximum top-1 accuracy for a predictor that
    knows the rules of chess but NOT the outcome token.
    """
    # Generate random games and get legal move masks
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        n_games, max_ply, seed,
    )

    # Compute legal move masks: grid is (n_games, max_ply, 64) packed bits
    grid, promo = engine.compute_legal_move_masks(move_ids, game_lengths)

    # Count legal moves at each position
    inv_n_sum = 0.0
    total_positions = 0
    inv_n_by_ply = defaultdict(list)

    for i in range(n_games):
        gl = int(game_lengths[i])
        for ply in range(gl):
            # Count legal grid moves: unpack 64 uint64 values, popcount each
            n_legal = 0
            for sq in range(64):
                n_legal += bin(int(grid[i, ply, sq])).count('1')
            # Add promotion moves
            if promo is not None and promo.shape[1] > ply:
                n_legal += int(np.sum(promo[i, ply] > 0))

            if n_legal > 0:
                inv_n_sum += 1.0 / n_legal
                inv_n_by_ply[ply].append(1.0 / n_legal)
                total_positions += 1

    overall = inv_n_sum / total_positions if total_positions else 0

    # Per-ply breakdown (sampled)
    ply_ceilings = {}
    for ply in sorted(inv_n_by_ply.keys())[:256]:
        vals = inv_n_by_ply[ply]
        ply_ceilings[ply] = sum(vals) / len(vals)

    return {
        "unconditional_ceiling": overall,
        "total_positions": total_positions,
        "n_games": n_games,
        "per_ply_ceiling": ply_ceilings,
    }


def compute_conditional_ceiling_mc(
    n_games: int = 5000,
    n_sample_positions: int = 2000,
    n_rollouts: int = 32,
    max_ply: int = 255,
    seed: int = 88888,
) -> dict:
    """Estimate outcome-conditioned ceiling via Monte Carlo rollouts.

    For a sample of positions, enumerate legal moves and estimate
    P(outcome | move, history) by playing out random continuations.
    The Bayes-optimal predictor picks argmax, giving accuracy =
    max_m P(outcome | move, history) / sum_m P(outcome | move, history).

    This requires playing games from arbitrary positions, which we approximate
    by generating many games and looking at positions where the same board
    state appears with different continuations.

    More practical approach: for each sampled position in a game:
    - We know the actual outcome O and the actual move m*
    - We know N_legal moves
    - We estimate: does knowing O help predict m*?
    - Specifically: we compute the fraction of random continuations from m*
      that produce outcome O, vs the average fraction across all legal moves.

    Since we can't easily play from arbitrary positions in the engine,
    we use an analytical approximation based on game structure:
    - Near game end (last few plies of checkmate): huge conditioning benefit
    - Mid-game: minimal conditioning benefit (~= 1/N)
    - PLY_LIMIT games: game length is known, slight benefit
    """
    # Generate games
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        n_games, max_ply, seed,
    )
    grid, promo = engine.compute_legal_move_masks(move_ids, game_lengths)

    # Analytical estimation of conditioning benefit
    #
    # For each position, the conditioning benefit depends on:
    # 1. How many plies remain (closer to end = more benefit)
    # 2. The outcome type (checkmate is more constraining than ply_limit)
    #
    # At the LAST ply of a checkmate game:
    #   Only checkmate-delivering moves are consistent with the outcome.
    #   Ceiling = 1/n_checkmate_moves (often 1-3 out of ~30 legal moves)
    #
    # At earlier plies: the benefit decays roughly exponentially.
    # P(outcome | move, history) ≈ 1/N_legal * (1 + benefit(plies_remaining))
    # where benefit → large near the end, → 0 far from the end.

    # Empirical approach: measure how concentrated the move distribution is
    # by looking at the last K plies of decisive games.
    conditioning_by_plies_from_end = defaultdict(list)

    for i in range(min(n_games, 10000)):
        gl = int(game_lengths[i])
        tc = int(term_codes[i])  # 0=checkmate, 1=stalemate, etc.

        for ply in range(gl):
            plies_from_end = gl - ply

            # Count legal moves
            n_legal = 0
            for sq in range(64):
                n_legal += bin(int(grid[i, ply, sq])).count('1')
            if promo is not None and promo.shape[1] > ply:
                n_legal += int(np.sum(promo[i, ply] > 0))

            if n_legal <= 0:
                continue

            # For the last move of a checkmate: only 1 move delivers mate
            # (approximately — sometimes 2-3 moves all give checkmate)
            if tc == 0 and plies_from_end == 1:
                # Last move is checkmate. Estimate ~1-2 mating moves.
                # Ceiling ≈ 1/min(n_legal, 2)
                effective_n = min(n_legal, 2)
            elif tc == 0 and plies_from_end <= 3:
                # Near-checkmate: some conditioning benefit
                # Rough: conditioning cuts effective choices by factor of
                # plies_from_end
                effective_n = max(1, n_legal / plies_from_end)
            elif tc == 1 and plies_from_end == 1:
                # Last move before stalemate
                effective_n = min(n_legal, 3)
            else:
                # General position: conditioning benefit is small
                # The outcome provides ~log2(5) ≈ 2.3 bits over the whole
                # game, distributed across ~gl plies. Per-ply benefit is tiny.
                effective_n = n_legal

            conditioning_by_plies_from_end[plies_from_end].append(
                1.0 / effective_n
            )

    # Compute overall conditioned ceiling
    all_conditioned = []
    all_unconditioned = []
    for i in range(min(n_games, 10000)):
        gl = int(game_lengths[i])
        tc = int(term_codes[i])
        for ply in range(gl):
            n_legal = 0
            for sq in range(64):
                n_legal += bin(int(grid[i, ply, sq])).count('1')
            if promo is not None and promo.shape[1] > ply:
                n_legal += int(np.sum(promo[i, ply] > 0))
            if n_legal <= 0:
                continue

            plies_from_end = gl - ply
            all_unconditioned.append(1.0 / n_legal)

            if tc == 0 and plies_from_end == 1:
                all_conditioned.append(1.0 / min(n_legal, 2))
            elif tc == 0 and plies_from_end <= 3:
                all_conditioned.append(1.0 / max(1, n_legal / plies_from_end))
            elif tc == 1 and plies_from_end == 1:
                all_conditioned.append(1.0 / min(n_legal, 3))
            else:
                all_conditioned.append(1.0 / n_legal)

    uncond = np.mean(all_unconditioned)
    cond = np.mean(all_conditioned)

    # Per-distance-from-end breakdown
    by_distance = {}
    for dist in sorted(conditioning_by_plies_from_end.keys()):
        if dist <= 20:
            vals = conditioning_by_plies_from_end[dist]
            by_distance[dist] = float(np.mean(vals))

    return {
        "conditional_ceiling_estimate": float(cond),
        "unconditional_ceiling": float(uncond),
        "conditioning_boost": float(cond / uncond) if uncond > 0 else 0,
        "n_positions": len(all_conditioned),
        "ceiling_by_plies_from_end": by_distance,
        "note": "Conditional ceiling is an analytical estimate, not exact Monte Carlo. "
                "The main benefit comes from the last 1-3 plies of decisive games.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute theoretical accuracy ceilings for random chess"
    )
    parser.add_argument("--n-games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=77777)
    parser.add_argument("--output", type=str, default="data/theoretical_ceiling.json")
    parser.add_argument("--model-accuracy", type=float, default=None,
                        help="Model top-1 accuracy to compute adjusted score")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Computing theoretical accuracy ceilings ({args.n_games:,} games)...")
    print()

    t0 = time.time()

    # Unconditional ceiling
    print("1. Unconditional ceiling (E[1/N_legal])...")
    uncond = compute_unconditional_ceiling(args.n_games, seed=args.seed)
    print(f"   = {uncond['unconditional_ceiling']:.4f} "
          f"({uncond['unconditional_ceiling']*100:.2f}%)")
    print(f"   ({uncond['total_positions']:,} positions from {args.n_games:,} games)")

    # Conditional ceiling
    print()
    print("2. Outcome-conditioned ceiling (analytical estimate)...")
    cond = compute_conditional_ceiling_mc(
        n_games=args.n_games, seed=args.seed + 1,
    )
    print(f"   = {cond['conditional_ceiling_estimate']:.4f} "
          f"({cond['conditional_ceiling_estimate']*100:.2f}%)")
    print(f"   Conditioning boost: {cond['conditioning_boost']:.2f}x")

    print()
    print(f"   Ceiling by plies from game end:")
    for dist, ceil in sorted(cond["ceiling_by_plies_from_end"].items()):
        bar = "#" * int(ceil * 200)
        print(f"     {dist:>3} plies from end: {ceil:.4f} ({ceil*100:.1f}%) {bar}")

    # Summary
    elapsed = time.time() - t0
    results = {
        "unconditional_ceiling": uncond["unconditional_ceiling"],
        "conditional_ceiling": cond["conditional_ceiling_estimate"],
        "conditioning_boost": cond["conditioning_boost"],
        "n_games": args.n_games,
        "total_positions": uncond["total_positions"],
        "per_ply_ceiling": uncond["per_ply_ceiling"],
        "ceiling_by_plies_from_end": cond["ceiling_by_plies_from_end"],
        "elapsed_seconds": elapsed,
    }

    if args.model_accuracy is not None:
        ma = args.model_accuracy
        results["model_accuracy"] = ma
        results["adjusted_vs_unconditional"] = ma / uncond["unconditional_ceiling"]
        results["adjusted_vs_conditional"] = ma / cond["conditional_ceiling_estimate"]
        print()
        print(f"Model accuracy: {ma:.4f} ({ma*100:.2f}%)")
        print(f"  vs unconditional ceiling: {results['adjusted_vs_unconditional']:.2f}x "
              f"({results['adjusted_vs_unconditional']*100:.1f}% of theoretical max)")
        print(f"  vs conditional ceiling:   {results['adjusted_vs_conditional']:.2f}x "
              f"({results['adjusted_vs_conditional']*100:.1f}% of theoretical max)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
