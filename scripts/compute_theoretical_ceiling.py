#!/usr/bin/env python3
"""Compute theoretical maximum top-1 accuracy for random chess play.

Two ceilings computed via Monte Carlo rollouts in the Rust engine:

1. Unconditional: E[1/N_legal] — best accuracy without knowing the outcome.
2. Outcome-conditioned: E[max_m P(m|outcome, history)] — best accuracy when
   the outcome token is known. Estimated by playing out random continuations
   from each legal move and measuring which outcomes result.

The "adjusted accuracy" normalizes model accuracy against these ceilings:
    adjusted = model_accuracy / ceiling

Usage:
    uv run python scripts/compute_theoretical_ceiling.py
    uv run python scripts/compute_theoretical_ceiling.py --n-games 5000 --rollouts 64
    uv run python scripts/compute_theoretical_ceiling.py --model-accuracy 0.070
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import chess_engine as engine


def main():
    parser = argparse.ArgumentParser(
        description="Compute theoretical accuracy ceilings for random chess"
    )
    parser.add_argument("--n-games", type=int, default=2000,
                        help="Number of random games to generate")
    parser.add_argument("--rollouts", type=int, default=32,
                        help="Monte Carlo rollouts per legal move")
    parser.add_argument("--sample-rate", type=float, default=0.02,
                        help="Fraction of positions to sample (1.0=all, 0.02=2%%)")
    parser.add_argument("--seed", type=int, default=77777)
    parser.add_argument("--output", type=str, default="data/theoretical_ceiling.json")
    parser.add_argument("--model-accuracy", type=float, default=None,
                        help="Model top-1 accuracy to compute adjusted score")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Computing theoretical accuracy ceilings")
    print(f"  Games: {args.n_games:,}")
    print(f"  Rollouts/move: {args.rollouts}")
    print(f"  Sample rate: {args.sample_rate:.0%}")
    print(f"  Seed: {args.seed}")
    print()

    t0 = time.time()
    result = engine.compute_accuracy_ceiling(
        n_games=args.n_games,
        max_ply=255,
        n_rollouts=args.rollouts,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    uncond = result["unconditional_ceiling"]
    naive_cond = result["naive_conditional_ceiling"]
    cond = result["conditional_ceiling"]
    boost_naive = naive_cond / uncond if uncond > 0 else 0
    boost = cond / uncond if uncond > 0 else 0

    print(f"Positions sampled: {result['n_positions']:,}")
    print(f"Unconditional ceiling:       {uncond:.4f} ({uncond*100:.2f}%)")
    print(f"Naive conditional ceiling:   {naive_cond:.4f} ({naive_cond*100:.2f}%)  {boost_naive:.2f}x")
    print(f"MCTS conditional ceiling:    {cond:.4f} ({cond*100:.2f}%)  {boost:.2f}x")
    print(f"Time: {elapsed:.0f}s")
    print()

    # Per-outcome breakdown
    outcomes = np.asarray(result["outcome"])
    conditionals = np.asarray(result["conditional"])
    naive_conditionals = np.asarray(result["naive_conditional"])
    unconditionals = np.asarray(result["unconditional"])
    outcome_names = [
        "W checkmated", "B checkmated", "Stalemate", "75-move",
        "5-fold rep", "Insuff mat", "Ply limit",
    ]

    print("Per-outcome breakdown:")
    outcome_data = {}
    for oi in range(7):
        mask = outcomes == oi
        n = int(mask.sum())
        if n > 0:
            uc = float(unconditionals[mask].mean())
            nc = float(naive_conditionals[mask].mean())
            cc = float(conditionals[mask].mean())
            print(f"  {outcome_names[oi]:>12}: uncond={uc:.4f}  naive={nc:.4f}  "
                  f"mcts={cc:.4f}  (n={n})")
            outcome_data[outcome_names[oi]] = {
                "unconditional": uc, "naive_conditional": nc,
                "conditional": cc, "n_positions": n,
            }
    print()

    # Per-ply-from-end breakdown
    plies = np.asarray(result["ply"])
    game_lengths = np.asarray(result["game_length"])
    plies_from_end = game_lengths - plies

    print("Ceiling by distance from game end:")
    distance_data = {}
    for dist in range(1, 21):
        mask = plies_from_end == dist
        n = int(mask.sum())
        if n > 10:
            uc = float(unconditionals[mask].mean())
            nc = float(naive_conditionals[mask].mean())
            cc = float(conditionals[mask].mean())
            bar = "#" * int(cc * 200)
            print(f"  {dist:>3} plies from end: uncond={uc:.4f}  naive={nc:.4f}  mcts={cc:.4f}  {bar}")
            distance_data[dist] = {"unconditional": uc, "naive_conditional": nc, "conditional": cc, "n": n}
    print()

    # Model adjusted accuracy
    if args.model_accuracy is not None:
        ma = args.model_accuracy
        adj_uncond = ma / uncond if uncond > 0 else 0
        adj_naive = ma / naive_cond if naive_cond > 0 else 0
        adj_cond = ma / cond if cond > 0 else 0
        print(f"Model accuracy: {ma:.4f} ({ma*100:.2f}%)")
        print(f"  vs unconditional ceiling:     {adj_uncond:.1%} of theoretical max")
        print(f"  vs naive conditional ceiling: {adj_naive:.1%} of theoretical max")
        print(f"  vs MCTS conditional ceiling:  {adj_cond:.1%} of theoretical max")
        print()

    # Save results
    data = {
        "unconditional_ceiling": float(uncond),
        "naive_conditional_ceiling": float(naive_cond),
        "conditional_ceiling": float(cond),
        "naive_conditioning_boost": float(boost_naive),
        "mcts_conditioning_boost": float(boost),
        "n_positions": int(result["n_positions"]),
        "n_games": args.n_games,
        "n_rollouts": args.rollouts,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "per_outcome": outcome_data,
        "per_distance_from_end": {str(k): v for k, v in distance_data.items()},
    }
    if args.model_accuracy is not None:
        data["model_accuracy"] = args.model_accuracy
        data["adjusted_vs_unconditional"] = adj_uncond
        data["adjusted_vs_naive_conditional"] = adj_naive
        data["adjusted_vs_conditional"] = adj_cond

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
