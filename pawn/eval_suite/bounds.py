"""Theoretical performance bounds via file-level iteration over positions.parquet."""

import numpy as np

from .corpus import _iter_position_parts, _new_accumulator, _accumulate, _finalize_k_stats, _finalize_k_hist, _finalize_phases, _finalize_checks


def compute_theoretical_bounds(corpus: dict) -> dict:
    """Compute theoretical bounds by iterating over position parquet parts.

    Reuses the same accumulator as summary_stats — one scan, bounded memory.
    """
    import polars as pl

    acc = _new_accumulator()
    for part_df in _iter_position_parts(corpus):
        _accumulate(acc, part_df.filter(pl.col("k") > 0))

    N = acc["n"]
    e_inv_k = acc["sum_inv_k"] / N
    e_top5 = acc["sum_top5"] / N
    e_ln_k = acc["sum_ln_k"] / N
    perplexity = float(np.exp(e_ln_k))

    var_inv_k = acc["sum_inv_k_sq"] / N - e_inv_k ** 2
    var_top5 = acc["sum_top5_sq"] / N - e_top5 ** 2
    var_ln_k = acc["sum_ln_k_sq"] / N - e_ln_k ** 2

    se_inv_k = float(np.sqrt(max(var_inv_k, 0) / N))
    se_top5 = float(np.sqrt(max(var_top5, 0) / N))
    se_ln_k = float(np.sqrt(max(var_ln_k, 0) / N))
    se_perp = perplexity * se_ln_k

    k_stats = _finalize_k_stats(acc)
    k_hist = _finalize_k_hist(acc)

    k_distribution = dict(zip(k_hist["values"], k_hist["counts"]))

    return {
        "n_positions": N,
        "top1_accuracy": {"value": float(e_inv_k), "se": se_inv_k},
        "top5_accuracy": {"value": float(e_top5), "se": se_top5},
        "loss_nats": {"value": float(e_ln_k), "se": se_ln_k},
        "perplexity": {"value": perplexity, "se": se_perp},
        "k_stats": k_stats,
        "k_distribution": k_distribution,
        "k_histogram": k_hist,
        "phase_bounds": _finalize_phases(acc),
        "check_bounds": _finalize_checks(acc),
    }


def format_bounds_report(bounds: dict, seed: int, n_games: int) -> str:
    """Format the theoretical bounds as a text report matching §3.3."""
    lines = [
        "=== Theoretical Bounds for Random Chess Next-Token Prediction ===",
        "",
        f"Sample: {n_games:,} games, {bounds['n_positions']:,} non-terminal non-padding positions",
        f"Seed: {seed}",
        "",
        "Bounds (95% CI):",
        f"  Max top-1 accuracy:  {bounds['top1_accuracy']['value']:.4%} ± {1.96*bounds['top1_accuracy']['se']:.4%}",
        f"  Max top-5 accuracy:  {bounds['top5_accuracy']['value']:.4%} ± {1.96*bounds['top5_accuracy']['se']:.4%}",
        f"  Min loss (nats):     {bounds['loss_nats']['value']:.4f} ± {1.96*bounds['loss_nats']['se']:.4f}",
        f"  Min perplexity:      {bounds['perplexity']['value']:.3f} ± {1.96*bounds['perplexity']['se']:.3f}",
        "",
        "Legal move count statistics:",
        f"  Mean K:              {bounds['k_stats']['mean']:.2f}",
        f"  Median K:            {bounds['k_stats']['median']}",
        f"  Min K (non-terminal): {bounds['k_stats']['min']}",
        f"  Max K:               {bounds['k_stats']['max']}",
        "",
    ]

    lines.append("Distribution of K (most common):")
    k_dist = bounds["k_distribution"]
    total = bounds["n_positions"]
    for kv in sorted(k_dist.keys())[:30]:
        pct = k_dist[kv] / total * 100
        lines.append(f"  K={kv:<4d}: {pct:.2f}%")
    if len(k_dist) > 30:
        lines.append(f"  ... ({len(k_dist)} distinct values total)")

    lines.append("")
    lines.append("Distribution of K by game phase:")
    for name in ("ply_1_20", "ply_21_80", "ply_81_150", "ply_150_plus"):
        if name not in bounds["phase_bounds"]:
            continue
        stats = bounds["phase_bounds"][name]
        label = name.replace("_", " ").replace("ply ", "Ply ")
        lines.append(
            f"  {label:>16s}: mean K = {stats['mean_k']:.2f}, "
            f"E[1/K] = {stats['e_1_over_k']:.4f}"
        )

    lines.append("")
    lines.append("Distribution of K by check status:")
    for label in ("in_check", "not_in_check"):
        if label not in bounds["check_bounds"]:
            continue
        stats = bounds["check_bounds"][label]
        lines.append(
            f"  {label:>14s}: mean K = {stats['mean_k']:.2f}, "
            f"E[1/K] = {stats['e_1_over_k']:.4f}, "
            f"frequency = {stats['frequency']:.2%}"
        )

    return "\n".join(lines)
