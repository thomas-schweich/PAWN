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
    cond = result["conditional_ceiling"]
    boost = cond / uncond if uncond > 0 else 0

    print(f"Positions sampled: {result['n_positions']:,}")
    print(f"Unconditional ceiling: {uncond:.4f} ({uncond*100:.2f}%)")
    print(f"Conditional ceiling:   {cond:.4f} ({cond*100:.2f}%)")
    print(f"Conditioning boost:    {boost:.2f}x")
    print(f"Time: {elapsed:.0f}s")
    print()

    # Per-outcome breakdown
    outcomes = result["outcome"]
    conditionals = result["conditional"]
    unconditionals = result["unconditional"]
    outcome_names = [
        "Checkmate", "Stalemate", "75-move", "5-fold rep",
        "Insuff mat", "Ply limit",
    ]

    print("Per-outcome breakdown:")
    outcome_data = {}
    for oi in range(6):
        mask = outcomes == oi
        n = int(mask.sum())
        if n > 0:
            uc = float(unconditionals[mask].mean())
            cc = float(conditionals[mask].mean())
            ob = cc / uc if uc > 0 else 0
            print(f"  {outcome_names[oi]:>12}: uncond={uc:.4f}  cond={cc:.4f}  "
                  f"boost={ob:.2f}x  (n={n})")
            outcome_data[outcome_names[oi]] = {
                "unconditional": uc, "conditional": cc,
                "boost": ob, "n_positions": n,
            }
    print()

    # Per-ply-from-end breakdown
    plies = result["ply"]
    game_lengths = result["game_length"]
    plies_from_end = game_lengths - plies

    print("Ceiling by distance from game end:")
    distance_data = {}
    for dist in range(1, 21):
        mask = plies_from_end == dist
        n = int(mask.sum())
        if n > 10:
            uc = float(unconditionals[mask].mean())
            cc = float(conditionals[mask].mean())
            bar = "#" * int(cc * 200)
            print(f"  {dist:>3} plies from end: uncond={uc:.4f}  cond={cc:.4f}  {bar}")
            distance_data[dist] = {"unconditional": uc, "conditional": cc, "n": n}
    print()

    # Model adjusted accuracy
    if args.model_accuracy is not None:
        ma = args.model_accuracy
        adj_uncond = ma / uncond if uncond > 0 else 0
        adj_cond = ma / cond if cond > 0 else 0
        print(f"Model accuracy: {ma:.4f} ({ma*100:.2f}%)")
        print(f"  vs unconditional ceiling: {adj_uncond:.1%} of theoretical max")
        print(f"  vs conditional ceiling:   {adj_cond:.1%} of theoretical max")
        print()

    # Save results
    data = {
        "unconditional_ceiling": float(uncond),
        "conditional_ceiling": float(cond),
        "conditioning_boost": float(boost),
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
        data["adjusted_vs_conditional"] = adj_cond

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
