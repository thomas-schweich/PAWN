# Theoretical Accuracy Ceiling

PAWN is trained on uniformly random chess games. Since each move is drawn
uniformly from the legal move set, top-1 accuracy has a hard theoretical
ceiling — no model, however large, can exceed it.

## Two ceilings

### Unconditional ceiling: E[1/N_legal] = 6.43%

At each position, the move is drawn uniformly from N legal moves. The best
a predictor can do without any context is pick one at random: accuracy = 1/N.
Averaged over all positions in random games, this gives **6.43%**.

A model that exceeds this ceiling has learned something beyond just "which
moves are legal" — it has learned to estimate the number of legal moves at
each position and bias predictions toward positions with fewer options.

### Outcome-conditioned ceiling: 7.92%

PAWN's input sequence begins with an outcome token (`WHITE_CHECKMATES`,
`STALEMATE`, `PLY_LIMIT`, etc.). This leaks information about the game's
trajectory, making some moves more predictable:

- **Checkmate games**: The final move must deliver checkmate. Knowing this
  raises the ceiling at the last ply from ~5% to ~14%.
- **Ply limit games**: Knowing the game lasts 255 plies constrains the move
  distribution slightly.
- **Stalemate games**: The final position has no legal moves but isn't check
  — very constraining on late moves.

The conditioned ceiling is estimated via Monte Carlo rollouts: at each
sampled position, every legal move is tried and 32 random continuations are
played out to estimate P(outcome | move, history). The Bayes-optimal
predictor picks the move most consistent with the known outcome.

## Adjusted accuracy

| Metric | Value |
|--------|-------|
| Unconditional ceiling (E[1/N_legal]) | 6.43% |
| Outcome-conditioned ceiling (MC, 32 rollouts) | 7.92% |
| Conditioning boost | 1.23x |

For a model with top-1 accuracy A:

- **Adjusted (unconditional)** = A / 6.43% — measures how much the model
  has learned about chess legality. Values > 100% mean it has learned
  structure beyond just legal moves.
- **Adjusted (conditioned)** = A / 7.92% — measures how close the model is
  to the Bayes-optimal predictor with perfect outcome knowledge. This is
  the tighter bound.

### Current model results (step ~69K)

| Variant | Top-1 | vs Uncond | vs Conditioned |
|---------|-------|-----------|----------------|
| large (68M) | 6.9% | 107% | 87% |
| base (36M) | 6.9% | 107% | 87% |
| small (10M) | 6.5% | 101% | 82% |

All models exceed the unconditional ceiling, confirming they learn chess
structure beyond move legality. The large and base models reach 87% of the
outcome-conditioned ceiling.

## Per-outcome breakdown

| Outcome | Uncond | Conditioned | Boost | Positions |
|---------|--------|-------------|-------|-----------|
| White checkmated | 5.26% | 13.79% | 2.62x | 328 |
| Black checkmated | 5.02% | 13.64% | 2.72x | 388 |
| Stalemate | 7.22% | 18.67% | 2.59x | 125 |
| Insufficient material | 7.17% | 18.61% | 2.60x | 256 |
| Ply limit | 6.51% | 6.97% | 1.07x | 8,618 |

Decisive outcomes (checkmate, stalemate, insufficient material) show 2.6x
conditioning boost. Ply limit games — the vast majority — show only 1.07x
because knowing the game goes the distance provides minimal per-move
information.

## Reproducing

```bash
# Default: 2000 games, 32 rollouts/move, 2% sample rate
uv run python scripts/compute_theoretical_ceiling.py --model-accuracy 0.069

# Higher precision (slower)
uv run python scripts/compute_theoretical_ceiling.py --n-games 10000 --rollouts 64 --sample-rate 0.05
```

Results are saved to `data/theoretical_ceiling.json`.

## Caveats

- The MCTS ceiling is an estimate, not exact. With more rollouts and higher
  sample rates, the estimate improves but computation time increases
  quadratically.
- The ceiling assumes the model has perfect knowledge of P(outcome | move,
  history). In practice, the model must learn this from data, so the
  achievable accuracy for a finite model is somewhat below the ceiling.
- Game length information is implicit in the outcome token (e.g., PLY_LIMIT
  implies 255 plies). A model could theoretically use position in the
  sequence to estimate remaining game length, further improving predictions.
