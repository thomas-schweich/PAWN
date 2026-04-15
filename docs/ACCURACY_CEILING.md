# Theoretical Accuracy Ceiling

PAWN is trained on uniformly random chess games. Since each move is drawn
uniformly from the legal move set, top-1 accuracy has a hard theoretical
ceiling — no model, however large, can exceed it.

## Three ceilings

### Unconditional ceiling: E[1/N_legal] = 6.52%

At each position, the move is drawn uniformly from N legal moves. The best
a predictor can do is pick one at random: accuracy = 1/N. Averaged over all
positions in random games, this gives **6.52%** (95% CI [6.44, 6.61]).

A model that exceeds this ceiling may have learned to use the outcome token 
to make non-uniform predictions — assigning higher probability to moves that
are more consistent with the known game outcome.

### Naive conditional ceiling: 6.53%

A zero-cost analytical estimate of outcome conditioning. At each position,
legal moves that lead to an immediate terminal state with a *different*
outcome than the actual game are excluded, and accuracy = 1/(N_legal - N_wrong).

This barely exceeds the unconditional ceiling (1.00x boost) because
immediate terminal states are rare — most moves at most positions lead to
non-terminal continuations, so the filter has almost nothing to exclude.

### MC conditional ceiling: [6.67%, 7.34%]

The full Monte Carlo estimate. At each sampled position, every legal move is
tried and random continuations are played out to estimate
P(outcome | move, history). The Bayes-optimal predictor picks the move most
consistent with the known outcome:

    P(m_i | outcome, history) = P(outcome | m_i, history) / Σ_j P(outcome | m_j, history)

The ceiling at each position is max_i P(m_i | outcome, history), and the
overall ceiling is the mean over all positions.

PAWN's input sequence begins with an outcome token (`WHITE_CHECKMATES`,
`STALEMATE`, `PLY_LIMIT`, etc.). This leaks information about the game's
trajectory, making some moves more predictable:

- **Checkmate games**: The final move must deliver checkmate — constraining
  on the last few plies.
- **Ply limit games**: Knowing the game lasts 255 plies constrains the move
  distribution slightly.
- **Stalemate games**: The final position has no legal moves but isn't check
  — constraining on late moves.

#### Bias bracket

The MC conditional ceiling is reported as a bracket, not a point estimate.
The naive Monte Carlo estimator (max of noisy P̂(outcome | m_i) estimates)
is **biased upward** because the `max` operator preferentially selects
whichever estimate had favorable noise (Jensen's inequality:
E[max X̂_i] ≥ max E[X̂_i]).

To bound this bias, we use a **split-half** correction: rollouts are split
into two independent halves (A and B). Half A selects the argmax move;
half B evaluates it. This breaks the selection-evaluation feedback loop,
producing an estimate that is **biased downward** (sometimes A picks the
wrong argmax). The true ceiling lies between the two:

    corrected (biased down)  ≤  true ceiling  ≤  naive (biased up)

With 128 rollouts per move, the bracket is **0.66pp** wide. The corrected
and naive 95% CIs do not overlap, confirming the bias is real and
non-negligible at this rollout count.

## Summary

| Metric | Value | 95% CI |
|--------|-------|--------|
| Unconditional ceiling (E[1/N_legal]) | 6.52% | [6.44, 6.61] |
| Naive conditional ceiling (1-ply filter) | 6.53% | [6.44, 6.61] |
| MC conditional ceiling (naive est.) | 7.34% | [7.24, 7.43] |
| MC conditional ceiling (corrected) | 6.67% | [6.58, 6.76] |
| Bias bracket width | 0.66pp | |
| Conditioning boost (naive) | 1.12x | |
| Conditioning boost (corrected) | 1.02x | |

For a model with top-1 accuracy A:

- **Adjusted (unconditional)** = A / 6.52% — measures how much the model
  has learned beyond predicting uniformly over legal moves. Values > 100%
  mean the model exploits the outcome token to make non-uniform predictions.
- **Adjusted (MC conditional)** = A / ceiling — measures how close the
  model is to the Bayes-optimal predictor with perfect outcome knowledge.
  Report against both the naive and corrected estimates for transparency.

### Legacy model results (100K steps, 4,278-token vocab, prepend_outcome=True)

| Variant | Top-1 | vs Uncond | vs MC Naive | vs MC Corrected |
|---------|-------|-----------|-------------|-----------------|
| large (68M) | 6.94% | 106% | 95% | 104% |
| base (36M) | 6.86% | 105% | 94% | 103% |
| small (10M) | 6.73% | 103% | 92% | 101% |

These numbers are from the legacy backbones (pre-`pre-vocab-transition`),
which were trained with the outcome token prepended at position 0 — so the
"vs MC corrected" column was the load-bearing comparison: it measures how
close the model is to the Bayes-optimal predictor *that also knows the
outcome*. All three exceeded the unconditional ceiling, confirming they
exploited the outcome token; against the MC corrected ceiling they sat
around it, which (given the corrected estimate is biased downward) means
the legacy models were close to the achievable ceiling for an outcome-
conditioned predictor.

### v1.0.0 model results (200K steps, 1,980-token vocab, prepend_outcome=False)

The current published checkpoints were trained **without** an outcome
prefix, so the comparison framework above doesn't apply unchanged: there
is no leaked outcome information for the model to exploit, and the MC
ceiling collapses to the unconditional ceiling.

The vocabulary also changed from 4,278 coordinate pairs to the 1,968-entry
searchless_chess action set (no impossible moves like `a1a1` or `b1a7`).
This both lowers the random baseline (fewer plausible candidates per
position) and removes the "model has to learn the action set" tax. For a
plain top-1 number to be comparable across vocabularies, you have to
recompute `E[1/N_legal]` against the *legal* moves, which is what the
new vocabulary already enforces — the unconditional ceiling is still the
right comparison, but the constant has shifted slightly.

| Variant | Params | Top-1 | Top-5 | Per-move legal | Game completion |
|---------|--------|-------|-------|----------------|-----------------|
| Small | 8.94M | 8.53% | 35.44% | 99.7457% | 51.66% |
| Base | 34.65M | 8.54% | 35.49% | 99.9962% | 99.02% |
| Large | 66.91M | 8.65% | 35.56% | 99.9996% | 99.80% |

Top-1 numbers across all three sizes are tightly clustered (~8.5%, vs
the legacy ~6.8%) and the loss curves were essentially indistinguishable
between sizes. The signal that actually separates capacity is **game
completion rate**: a model with a 99.7% per-move legal rate forfeits about
half its games on the way to terminal, while 99.999% clears nearly all of
them. See [docs/ARCHITECTURE.md](ARCHITECTURE.md#game-completion-rate)
for that analysis.

The unconditional / MC ceilings reported in the table at the top of this
document were computed against the **legacy** vocabulary and prepended-
outcome layout. Recomputing them under the v1.0.0 setup is a known TODO;
the methodology in the rest of this document still applies, only the
constants change.

## Per-outcome breakdown

| Outcome | Uncond | MC Naive | MC Corrected | Bracket | n |
|---------|--------|----------|--------------|---------|---|
| White checkmated | 5.44% | 9.63% | 6.34% | 3.29pp | 2,167 |
| Black checkmated | 5.05% | 9.25% | 6.07% | 3.18pp | 2,382 |
| Stalemate | 7.98% | 14.05% | 8.50% | 5.55pp | 1,029 |
| Insufficient material | 7.18% | 12.35% | 8.03% | 4.32pp | 1,651 |
| Ply limit | 6.59% | 6.87% | 6.63% | 0.24pp | 52,740 |

The bias bracket is narrow (0.24pp) for ply-limit games, which make up 88%
of positions — most legal moves lead to ply-limit regardless, so outcome
probabilities are all near 0.9 and the max/sum ratio is close to 1/N. The
bracket is wide (3-5pp) for decisive outcomes, where a few moves have high
P(outcome | m_i) and the rest are near zero, making the max sensitive to
noise.

The conditioning benefit for decisive outcomes is real but modest. Taking
the corrected estimates at face value:

| Outcome | Corrected / Unconditional |
|---------|--------------------------|
| White checkmated | 1.17x |
| Black checkmated | 1.20x |
| Stalemate | 1.07x |
| Insufficient material | 1.12x |
| Ply limit | 1.01x |

However, the per-outcome corrected estimates are noisy for rare outcomes
(especially stalemate, n=1,029) and should be interpreted cautiously.

## Limitations

- **The bias bracket is too wide for strong quantitative claims.** At 128
  rollouts, we can say the model is between 92-104% of the Bayes-optimal
  ceiling (depending on which estimate is used), but not pin it down more
  precisely. A 256- or 512-rollout run would narrow this.
- **The MC ceiling is an estimate, not exact.** Both the naive and corrected
  estimators have known biases. The true ceiling lies between them, but the
  exact value is unknown without infinite rollouts.
- **Per-outcome estimates are noisy for rare outcomes.** Checkmate and
  stalemate positions have large per-position variance with 128 rollouts.
  Stratified sampling (oversampling rare outcomes) would improve precision.
- **The ceiling assumes perfect outcome knowledge.** The model must *learn*
  P(outcome | move, history) from data, so achievable accuracy for a finite
  model is somewhat below the theoretical ceiling.
- **Other sources of signal are not accounted for.** The model may exploit
  sequential structure in random games (e.g., position-dependent move
  popularity, game-length correlations) beyond what the outcome token
  provides. The ceiling analysis does not isolate this.

## Reproducing

```bash
# Moderate precision: 5000 games, 128 rollouts/move (64 per half), 5% sample rate
uv run python scripts/compute_theoretical_ceiling.py \
    --n-games 5000 --rollouts 128 --sample-rate 0.05 \
    --model-accuracy 0.069

# Quick check (low precision, ~2 min)
uv run python scripts/compute_theoretical_ceiling.py --model-accuracy 0.069
```

Results are saved to `data/theoretical_ceiling.json` and include bootstrap
95% CIs clustered by game. Runtime: ~38 min on 16-core CPU for the moderate
configuration.

## Methodology

The computation (implemented in Rust in `engine/src/random.rs`) works as
follows:

1. Generate N random games (uniform legal move selection).
2. Sample a fraction of positions across all games.
3. At each sampled position with K legal moves and known outcome O:
   - **Unconditional**: ceiling = 1/K.
   - **Naive conditional**: try each move; if it immediately terminates with
     outcome ≠ O, prune it. Ceiling = 1/(K - pruned).
   - **MC conditional**: for each legal move, play R/2 random continuations
     from two independent seeds. Estimate P(O | m_i) from each half.
     Naive estimate = max(P̂_combined) / Σ P̂_combined.
     Corrected = P̂_B[argmax(P̂_A)] / Σ P̂_combined.
4. Report means with bootstrap 95% CIs (resampled by game to account for
   within-game correlation).
