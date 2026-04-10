#!/usr/bin/env python3
"""Compute theoretical maximum top-1 accuracy for random chess play.

Three ceilings computed via Monte Carlo rollouts in the Rust engine:

1. Unconditional: E[1/N_legal] — best accuracy without knowing the outcome.
2. Naive conditional (0-depth): prune moves that immediately terminate with
   the wrong outcome, then 1/N_remaining.
3. MC conditional: E[max_m P(m|outcome, history)] — best accuracy when the
   outcome token is known. Estimated via random rollouts from each legal move.

The MC conditional is reported as a bracket [corrected, naive] because:
- The naive estimator (max of noisy estimates) is biased upward.
- The split-half corrected estimator (A selects, B evaluates) is biased downward.
- The true Bayes-optimal ceiling lies between the two.

Usage:
    uv run python scripts/compute_theoretical_ceiling.py
    uv run python scripts/compute_theoretical_ceiling.py --n-games 5000 --rollouts 128 --sample-rate 0.05
    uv run python scripts/compute_theoretical_ceiling.py --model-accuracy 0.070
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import chess_engine as engine


def bootstrap_ci_clustered(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI resampling by cluster (game), not by position.

    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(cluster_ids)
    n_clusters = len(unique_ids)

    # Build cluster->position index for fast resampling
    cluster_positions: dict[int, np.ndarray] = {}
    for cid in unique_ids:
        cluster_positions[int(cid)] = np.where(cluster_ids == cid)[0]

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_ids, size=n_clusters, replace=True)
        indices = np.concatenate([cluster_positions[int(c)] for c in sampled])
        boot_means[b] = values[indices].mean()

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


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
    parser.add_argument("--max-ply", type=int, default=255,
                        help="Maximum game length in plies (default: 255)")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of bootstrap resamples for CIs (0 to skip)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Computing theoretical accuracy ceilings")
    print(f"  Games: {args.n_games:,}")
    print(f"  Rollouts/move: {args.rollouts}")
    print(f"  Sample rate: {args.sample_rate:.0%}")
    print(f"  Seed: {args.seed}")
    print()

    # Process in batches for progress reporting
    batch_size = max(100, args.n_games // 10)
    n_batches = (args.n_games + batch_size - 1) // batch_size

    all_outcomes = []
    all_conditionals = []
    all_corrected = []
    all_naive = []
    all_unconditionals = []
    all_game_ids = []
    all_plies = []
    all_game_lengths = []
    total_positions = 0
    game_id_offset = 0

    t0 = time.time()
    games_done = 0

    for batch_idx in range(n_batches):
        batch_n = min(batch_size, args.n_games - games_done)
        batch_seed = args.seed + batch_idx * 100000

        bt = time.time()
        result = engine.compute_accuracy_ceiling(
            n_games=batch_n,
            max_ply=args.max_ply,
            n_rollouts=args.rollouts,
            sample_rate=args.sample_rate,
            seed=batch_seed,
        )
        batch_elapsed = time.time() - bt

        n_pos = result["n_positions"]
        total_positions += n_pos
        games_done += batch_n

        # Accumulate per-position arrays
        all_outcomes.append(np.asarray(result["outcome"]))
        all_conditionals.append(np.asarray(result["conditional"]))
        all_corrected.append(np.asarray(result["conditional_corrected"]))
        all_naive.append(np.asarray(result["naive_conditional"]))
        all_unconditionals.append(np.asarray(result["unconditional"]))
        all_game_ids.append(np.asarray(result["game_idx"]) + game_id_offset)
        all_plies.append(np.asarray(result["ply"]))
        all_game_lengths.append(np.asarray(result["game_length"]))
        game_id_offset += batch_n

        # Running estimates
        uc_so_far = np.concatenate(all_unconditionals).mean()
        mc_so_far = np.concatenate(all_conditionals).mean()
        mc_corr_so_far = np.concatenate(all_corrected).mean()
        bracket = (mc_so_far - mc_corr_so_far) * 100

        elapsed_so_far = time.time() - t0
        rate = games_done / elapsed_so_far
        eta = (args.n_games - games_done) / rate if rate > 0 else 0

        print(
            f"[batch {batch_idx+1}/{n_batches}] "
            f"{games_done}/{args.n_games} games, {total_positions:,} positions | "
            f"uncond={uc_so_far:.4f}  mc=[{mc_corr_so_far:.4f}, {mc_so_far:.4f}] "
            f"bracket={bracket:.3f}pp | "
            f"{batch_elapsed:.0f}s/batch, ETA {eta/60:.0f}m",
            flush=True,
        )

    elapsed = time.time() - t0

    # Concatenate all batches
    outcomes = np.concatenate(all_outcomes)
    conditionals = np.concatenate(all_conditionals)
    corrected_conditionals = np.concatenate(all_corrected)
    naive_conditionals = np.concatenate(all_naive)
    unconditionals = np.concatenate(all_unconditionals)
    game_ids = np.concatenate(all_game_ids)

    uncond = float(unconditionals.mean())
    naive_cond = float(naive_conditionals.mean())
    cond = float(conditionals.mean())
    cond_corr = float(corrected_conditionals.mean())
    boost_naive = naive_cond / uncond if uncond > 0 else 0
    boost = cond / uncond if uncond > 0 else 0
    boost_corr = cond_corr / uncond if uncond > 0 else 0

    print()
    print(f"Positions sampled: {total_positions:,}")
    print(f"Unconditional ceiling:         {uncond:.4f} ({uncond*100:.2f}%)")
    print(f"Naive conditional ceiling:     {naive_cond:.4f} ({naive_cond*100:.2f}%)  {boost_naive:.2f}x")
    print(f"MC conditional (naive est.):   {cond:.4f} ({cond*100:.2f}%)  {boost:.2f}x  [biased up]")
    print(f"MC conditional (corrected):    {cond_corr:.4f} ({cond_corr*100:.2f}%)  {boost_corr:.2f}x  [biased down]")
    print(f"  Bias bracket width:          {(cond - cond_corr)*100:.3f}pp")
    print(f"Time: {elapsed:.0f}s")
    print()

    outcome_names = [
        "W checkmated", "B checkmated", "Stalemate", "75-move",
        "5-fold rep", "Insuff mat", "Ply limit",
    ]

    # Bootstrap CIs (clustered by game)
    ci_data = {}
    if args.bootstrap > 0:
        print(f"Bootstrap CIs ({args.bootstrap} resamples, clustered by game):")
        for label, vals in [
            ("unconditional", unconditionals),
            ("naive_conditional", naive_conditionals),
            ("mc_conditional", conditionals),
            ("mc_corrected", corrected_conditionals),
        ]:
            mean, lo, hi = bootstrap_ci_clustered(vals, game_ids, n_boot=args.bootstrap)
            ci_data[label] = {"mean": mean, "ci_low": lo, "ci_high": hi}
            print(f"  {label:>20}: {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print()

    # Per-outcome breakdown
    print("Per-outcome breakdown:")
    outcome_data = {}
    for oi in range(7):
        mask = outcomes == oi
        n = int(mask.sum())
        if n > 0:
            uc = float(unconditionals[mask].mean())
            nc = float(naive_conditionals[mask].mean())
            cc = float(conditionals[mask].mean())
            cc_corr = float(corrected_conditionals[mask].mean())
            print(f"  {outcome_names[oi]:>12}: uncond={uc:.4f}  naive={nc:.4f}  "
                  f"mc={cc:.4f}  corrected={cc_corr:.4f}  (n={n})")
            outcome_data[outcome_names[oi]] = {
                "unconditional": uc, "naive_conditional": nc,
                "conditional": cc, "conditional_corrected": cc_corr,
                "n_positions": n,
            }
    print()

    # Per-ply-from-end breakdown
    plies = np.concatenate(all_plies)
    game_lengths = np.concatenate(all_game_lengths)
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
            cc_corr = float(corrected_conditionals[mask].mean())
            bar = "#" * int(cc * 200)
            print(f"  {dist:>3} plies from end: uncond={uc:.4f}  naive={nc:.4f}  "
                  f"mc={cc:.4f}  corrected={cc_corr:.4f}  {bar}")
            distance_data[dist] = {
                "unconditional": uc, "naive_conditional": nc,
                "conditional": cc, "conditional_corrected": cc_corr, "n": n,
            }
    print()

    # Model adjusted accuracy
    if args.model_accuracy is not None:
        ma = args.model_accuracy
        adj_uncond = ma / uncond if uncond > 0 else 0
        adj_naive = ma / naive_cond if naive_cond > 0 else 0
        adj_cond = ma / cond if cond > 0 else 0
        adj_corr = ma / cond_corr if cond_corr > 0 else 0
        print(f"Model accuracy: {ma:.4f} ({ma*100:.2f}%)")
        print(f"  vs unconditional ceiling:      {adj_uncond:.1%} of theoretical max")
        print(f"  vs naive conditional ceiling:  {adj_naive:.1%} of theoretical max")
        print(f"  vs MC conditional (naive):     {adj_cond:.1%} of theoretical max")
        print(f"  vs MC conditional (corrected): {adj_corr:.1%} of theoretical max")
        print()

    # Save results
    data = {
        "unconditional_ceiling": float(uncond),
        "naive_conditional_ceiling": float(naive_cond),
        "conditional_ceiling": float(cond),
        "conditional_corrected_ceiling": float(cond_corr),
        "naive_conditioning_boost": float(boost_naive),
        "mc_conditioning_boost": float(boost),
        "mc_corrected_conditioning_boost": float(boost_corr),
        "bias_bracket_pp": float((cond - cond_corr) * 100),
        "n_positions": total_positions,
        "n_games": args.n_games,
        "n_rollouts": args.rollouts,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "per_outcome": outcome_data,
        "per_distance_from_end": {str(k): v for k, v in distance_data.items()},
    }
    if ci_data:
        data["bootstrap_ci"] = ci_data
    if args.model_accuracy is not None:
        data["model_accuracy"] = args.model_accuracy
        data["adjusted_vs_unconditional"] = adj_uncond
        data["adjusted_vs_naive_conditional"] = adj_naive
        data["adjusted_vs_conditional"] = adj_cond
        data["adjusted_vs_conditional_corrected"] = adj_corr

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
