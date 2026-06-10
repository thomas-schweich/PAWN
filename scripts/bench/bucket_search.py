"""Sample random games and compute optimal bucket edges for pretraining.

Approach:
1. Generate N games via the Rust engine. Capture game_lengths.
2. Build a length histogram.
3. Cost model: per-row cost(T) = a*T + b*T² (FFN linear + attention quadratic).
   Coefficients estimated from FLOP counts at the production SUPERNET shape.
4. For each candidate set of bucket edges (constrained to 128-multiples), compute
   total compute = sum_games cost(B(L)) where B(L) is the smallest bucket >= L.
5. Search over edge sets with K=1,2,3,4 buckets. Report optimal edges and the
   compute savings vs the current single-bucket T=512 baseline.
"""
from __future__ import annotations
import argparse
import itertools
import sys
from typing import Sequence

import numpy as np
import chess_engine as engine

from pawn.config import SUPERNET


def cost_flops_per_row(T: int, cfg) -> float:
    """FLOPs per row of seq_len T at the production SUPERNET shape.

    Counts only the dominant ops: attention (Q,K,V,O matmuls + QK^T + AV)
    and FFN (gate, up, down). Per layer. Multiplied by n_layers.

    Forward only — backward is roughly 2x (autograd) so we'd multiply by 3
    for fwd+bwd, but the constant doesn't affect optimal-edge search.
    """
    d, ff, L, H, hd = cfg.d_model, cfg.d_ff, cfg.n_layers, cfg.n_heads, cfg.head_dim
    # Attention per layer per row:
    #   QKV projection: 3 * T * d * d
    #   Output projection: T * d * d
    #   QK^T: H * T * T * hd  (per head)
    #   softmax: O(T²) — negligible
    #   AV: H * T * T * hd
    attn = 4 * T * d * d + 2 * H * T * T * hd
    # FFN per layer per row:
    #   gate: T * d * ff
    #   up:   T * d * ff
    #   down: T * ff * d
    ffn = 3 * T * d * ff
    # lm_head: T * d * V (small relative to per-layer, ignored — we compare per-game)
    return L * (attn + ffn)


def bucket_for(L: int, edges: Sequence[int]) -> int:
    """Return the smallest edge >= L. Edges must be sorted ascending."""
    for e in edges:
        if L <= e:
            return e
    return edges[-1]  # truncate to top bucket


def evaluate(lengths: np.ndarray, edges: list[int], cfg) -> tuple[float, dict]:
    """Total cost across all games. Also returns per-bucket count + waste."""
    edges = sorted(set(edges))
    assert edges[-1] >= int(lengths.max()), f"top edge {edges[-1]} < max length {lengths.max()}"
    per_game_T = np.array([bucket_for(int(L), edges) for L in lengths])
    per_game_cost = np.array([cost_flops_per_row(T, cfg) for T in per_game_T])
    total = float(per_game_cost.sum())
    counts: dict[int, int] = {e: 0 for e in edges}
    waste: dict[int, int] = {e: 0 for e in edges}  # PAD tokens
    for L, T in zip(lengths, per_game_T, strict=False):
        counts[T] += 1
        waste[T] += T - int(L)
    return total, {"counts": counts, "waste": waste, "edges": edges}


def search_optimal(lengths: np.ndarray, K: int, alignment: int, T_max: int, cfg):
    """Exhaustive search over K-bucket edge sets (alignment-multiple, ≤ T_max)."""
    candidates = list(range(alignment, T_max + 1, alignment))
    if T_max not in candidates:
        candidates.append(T_max)
    candidates = sorted(set(candidates))
    # K buckets means K-1 internal edges + the top edge fixed at >= T_max
    # We require the top edge to be the smallest candidate >= T_max so every game fits
    top = next(c for c in candidates if c >= T_max)
    inner_candidates = [c for c in candidates if c < top]
    best_cost = float("inf")
    best_edges = None
    n_combos = 0
    for inner in itertools.combinations(inner_candidates, K - 1):
        edges = list(inner) + [top]
        cost, _ = evaluate(lengths, edges, cfg)
        n_combos += 1
        if cost < best_cost:
            best_cost = cost
            best_edges = edges
    return best_edges, best_cost, n_combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=100_000)
    ap.add_argument("--max-ply", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-buckets", type=int, default=4)
    ap.add_argument("--alignment", type=int, default=128,
                    help="bucket edges must be multiples of this for tensor-core alignment")
    args = ap.parse_args()

    print(f"Generating {args.n_games:,} random games, max_ply={args.max_ply}…")
    _move_ids, game_lengths, _term_codes = engine.generate_random_games(
        n_games=args.n_games, max_ply=args.max_ply,
        seed=args.seed, discard_ply_limit=False, mate_boost=0.0,
    )
    L = np.asarray(game_lengths)
    print(f"length stats: min={L.min()} median={int(np.median(L))} "
          f"mean={L.mean():.1f} p90={int(np.percentile(L,90))} "
          f"p95={int(np.percentile(L,95))} p99={int(np.percentile(L,99))} max={L.max()}")
    print()

    cfg = SUPERNET
    print(f"cost model: SUPERNET d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} "
          f"d_ff={cfg.d_ff} head_dim={cfg.head_dim}")

    # Baseline: single bucket at max_ply.
    baseline_edges = [args.max_ply]
    baseline_cost, baseline_info = evaluate(L, baseline_edges, cfg)
    print(f"\nbaseline (K=1, T={args.max_ply}): cost = {baseline_cost:.3e}")
    print(f"  total games: {sum(baseline_info['counts'].values())}")
    print(f"  total PAD tokens: {sum(baseline_info['waste'].values()):,}")

    # Search over K=2..max_buckets
    print()
    for K in range(2, args.max_buckets + 1):
        edges, cost, n = search_optimal(L, K, args.alignment, args.max_ply, cfg)
        if edges is None:
            continue
        savings = (baseline_cost - cost) / baseline_cost * 100
        _, info = evaluate(L, edges, cfg)
        print(f"K={K}: optimal edges = {edges}  cost = {cost:.3e}  "
              f"savings vs baseline = {savings:.2f}%  (searched {n:,} combos)")
        for e in info["edges"]:
            n_in_bucket = info["counts"][e]
            w = info["waste"][e]
            pct = n_in_bucket / len(L) * 100
            avg_waste_per_row = w / max(n_in_bucket, 1)
            print(f"    T={e:>4}  {n_in_bucket:>7,} games ({pct:5.1f}%)  "
                  f"avg PAD/row = {avg_waste_per_row:.1f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
