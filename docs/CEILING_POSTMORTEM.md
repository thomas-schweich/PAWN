# Accuracy Ceiling Postmortem

*2026-03-28. Recomputation with bias correction and higher precision.*

## Background

PAWN is trained on uniformly random chess games. The model's input sequence
begins with an outcome token (e.g., `WHITE_CHECKMATES`, `PLY_LIMIT`), which
leaks information about the game trajectory. We compute theoretical accuracy
ceilings to understand how much of the model's predictive power comes from
this outcome conditioning versus learned chess structure.

The original ceiling computation used 2,000 games, 32 Monte Carlo rollouts
per move, and a 2% position sample rate (~9,700 positions). It reported
an "MCTS conditional ceiling" of **7.92%**, implying that outcome
conditioning provides a 1.23x boost over the unconditional baseline of
6.43%. All three PAWN models were reported as reaching 85-88% of this
ceiling.

This document reports the results of a recomputation that increases
precision, corrects a systematic bias in the estimator, and adds confidence
intervals.

## Metric definitions

### Unconditional ceiling: E[1/N_legal]

The Bayes-optimal top-1 accuracy for a predictor that knows the board
position (and thus the legal moves) but not the game outcome. Since moves
are drawn uniformly, the best strategy is to predict uniformly over legal
moves, achieving 1/N_legal at each position. The ceiling is the mean over
the position distribution in random games.

### Naive conditional ceiling

A zero-depth analytical bound. At each position, any legal move that
immediately terminates the game with an outcome different from the actual
game's outcome is pruned (it could not have been the move played). The
ceiling is 1/(N_legal - N_pruned). This costs nothing beyond move
generation and gives an exact (non-estimated) lower bound on the
conditioning benefit.

### MC conditional ceiling

The Bayes-optimal top-1 accuracy for a predictor that knows both the
position and the game outcome. Computed via Bayes' rule:

    P(m_i | outcome, history) = P(outcome | m_i, history) / Σ_j P(outcome | m_j, history)

where P(outcome | m_i, history) is estimated by playing R random
continuations from each legal move m_i and observing how often each outcome
occurs. The ceiling at each position is max_i P(m_i | outcome, history),
and the overall ceiling is the mean.

This is the quantity the old computation called the "MCTS ceiling." The name
was a misnomer — the method uses flat Monte Carlo rollouts, not Monte Carlo
tree search.

### Split-half bias correction

The naive MC estimator computes max over noisy probability estimates. By
Jensen's inequality, E[max(X̂_1,...,X̂_n)] ≥ max(E[X̂_1],...,E[X̂_n]), so
the estimator is **biased upward**: it systematically overestimates the true
ceiling because the max operator preferentially selects whichever move got
lucky noise.

The split-half correction breaks the R rollouts into two independent halves
(A and B). Half A selects the argmax move; half B evaluates it. Because B's
noise is independent of A's selection, the feedback loop is broken. This
estimator is **biased downward** (A sometimes picks the wrong argmax, and B
honestly evaluates the wrong move).

The true Bayes-optimal ceiling lies between the two estimates:

    corrected (biased down)  ≤  true ceiling  ≤  naive (biased up)

As the number of rollouts increases, both converge to the truth and the
bracket narrows.

### Bootstrap confidence intervals

All CIs are computed by resampling **games** (not positions) with
replacement (2,000 bootstrap replicates). This accounts for within-game
correlation — positions from the same game share trajectory structure and
are not independent.

## Configuration

| Parameter | Old | New |
|-----------|-----|-----|
| Games | 2,000 | 5,000 |
| Rollouts/move | 32 | 128 |
| Sample rate | 2% | 5% |
| Positions sampled | 9,715 | 59,969 |
| Split-half correction | No | Yes |
| Bootstrap CIs | No | Yes (2,000 resamples, clustered) |
| Wall-clock time | ~2.5 min | ~38 min |

## Headline results

| Metric | Old value | New value | 95% CI |
|--------|-----------|-----------|--------|
| Unconditional | 6.43% | 6.52% | [6.44, 6.61] |
| Naive conditional | 6.44% | 6.53% | [6.44, 6.61] |
| MC conditional (naive est.) | 7.92% | **7.34%** | [7.24, 7.43] |
| MC conditional (corrected) | — | **6.67%** | [6.58, 6.76] |
| Bias bracket width | — | **0.66pp** | |
| Conditioning boost (naive) | 1.23x | 1.12x | |
| Conditioning boost (corrected) | — | 1.02x | |

### Model performance vs ceilings (base model, 6.90% top-1)

| Comparison | Old | New |
|------------|-----|-----|
| vs unconditional | 107% | 106% |
| vs MC naive | 87% | 94% |
| vs MC corrected | — | **103%** |

## Key findings

### 1. The old 7.92% ceiling was substantially inflated

The naive MC estimator dropped from 7.92% (32 rollouts) to 7.34%
(128 rollouts), a 0.58pp reduction. This is entirely attributable to bias
reduction: with more rollouts, each per-move probability estimate is less
noisy, so the max operator's upward bias shrinks. The remaining naive
estimate (7.34%) is still biased upward.

The split-half corrected estimate is 6.67%, yielding a bias bracket of
0.66pp. This bracket is still wide — the naive and corrected 95% CIs don't
overlap — indicating that even 128 rollouts leaves meaningful estimation
noise per move.

### 2. Outcome conditioning provides far less signal than originally reported

The old narrative: "outcome conditioning provides a 1.23x boost, and models
reach 85-88% of the Bayes-optimal ceiling."

The revised picture: the true conditioning boost is somewhere between 1.02x
(corrected) and 1.12x (naive). The bulk of the model's accuracy comes from
learning legal move distributions and position-dependent move popularity,
not from exploiting the outcome token.

### 3. The base model exceeds the corrected ceiling

The base model's 6.90% top-1 accuracy is **103.4% of the corrected MC
ceiling** (6.67%). This is expected and not paradoxical:

- The corrected estimator is biased downward. It underestimates the true
  ceiling because half-A sometimes selects the wrong argmax, and half-B
  faithfully evaluates the wrong move.
- The true ceiling lies somewhere in [6.67%, 7.34%], and the model at
  6.90% is well within this bracket.

This does mean, however, that we cannot currently make a strong quantitative
claim about what fraction of the Bayes-optimal ceiling the model achieves.
The bracket is too wide for that.

### 4. The unconditional ceiling shifted from 6.43% to 6.52%

This is a sampling effect, not a methodological change — the unconditional
ceiling (E[1/N_legal]) has no estimation bias. The shift reflects the
larger, differently-sampled corpus (5,000 games at 5% vs 2,000 at 2%). The
95% CI [6.44, 6.61] comfortably contains the old 6.43% value. With enough
games, both estimates converge to the population mean.

### 5. Per-outcome breakdown confirms bias is concentrated in rare outcomes

| Outcome | Naive MC | Corrected | Bracket | n |
|---------|----------|-----------|---------|---|
| White checkmated | 9.63% | 6.34% | 3.29pp | 2,167 |
| Black checkmated | 9.25% | 6.07% | 3.18pp | 2,382 |
| Stalemate | 14.05% | 8.50% | 5.55pp | 1,029 |
| Insufficient material | 12.35% | 8.03% | 4.32pp | 1,651 |
| Ply limit | 6.87% | 6.63% | 0.24pp | 52,740 |

The bias bracket is 0.24pp for ply-limit games (88% of positions) but
3-5pp for decisive outcomes. This makes sense: in a ply-limit game, most
legal moves lead to ply-limit continuations regardless, so outcome
probabilities are all ~0.9 and the max/sum ratio is close to 1/N. In
checkmate/stalemate games, most moves have P(outcome | m_i) ≈ 0 and a few
have P ≫ 0, making the max highly sensitive to noise.

The old computation's headline "2.6x boost for decisive outcomes" was almost
entirely bias artifact. The corrected estimates show:

| Outcome | Corrected / Unconditional |
|---------|--------------------------|
| White checkmated | 1.17x |
| Black checkmated | 1.20x |
| Stalemate | 1.07x |
| Insufficient material | 1.12x |
| Ply limit | 1.01x |

There is still a real conditioning benefit for decisive outcomes, but it's
~1.1-1.2x, not 2.6x. And for the dominant ply-limit class, conditioning
is essentially irrelevant.

### 6. The naive conditional ceiling remains useless

The naive conditional (0-depth pruning) is 6.53% vs unconditional 6.52% —
a 0.01pp difference. Immediate terminal states are too rare to provide
meaningful signal. This result is robust and unchanged from the original
computation.

## Incongruities and open questions

### The bracket is too wide for strong claims

The primary goal of the ceiling analysis is to answer: "how close is the
model to optimal?" With a bracket of [6.67%, 7.34%], we can only say the
base model (6.90%) is somewhere between 94% and 103% of optimal. This is
not precise enough to draw meaningful conclusions about remaining headroom.

Narrowing the bracket requires more rollouts. Based on the trend (32
rollouts → 7.92% naive, 128 rollouts → 7.34% naive), diminishing returns
are likely, but the corrected estimate should converge faster than the naive
one.

### Models exceed the unconditional ceiling — but barely

All models exceed the unconditional ceiling (small: 103%, base: 106%,
large: 106%), confirming they exploit outcome conditioning. But the margin
is modest. This raises the question: is the model primarily learning
P(outcome | move) (outcome conditioning), or is it learning other structure
in random games?

For instance, move order within random games is not fully exchangeable —
early moves affect which positions arise later, and some move sequences are
more common than others even under uniform play. A model could exploit these
sequential dependencies without using the outcome token at all.

A controlled experiment would help: train a model variant with a fixed
(uninformative) outcome token and compare its accuracy to the standard
model. The gap, if any, quantifies the outcome conditioning contribution.

### Per-outcome ceilings are noisy for rare outcomes

Stalemate (n=1,029) and checkmate (n=2,167/2,382) have large per-position
variance. The per-outcome corrected estimates should be interpreted
cautiously. A run with higher sample rates for rare outcomes (stratified
sampling) would improve precision without proportionally increasing compute.

## Followup work

### Required (before merging)

- **Backfill ACCURACY_CEILING.md** with the final numbers from this run.
- **Regenerate model cards** with updated ceiling values.
- **Decide which ceiling to use as the canonical reference.** The midpoint
  of the bracket ((6.67 + 7.34) / 2 = 7.01%) is one option; reporting the
  bracket explicitly is more honest.

### Recommended

- **Higher-rollout run (256 or 512)** to narrow the bracket. If the bracket
  at 256 rollouts is < 0.2pp, the ceiling is well-determined and no further
  computation is needed. Estimated wall-clock: ~75 min (256) or ~2.5 hr
  (512) on the same machine.
- **Ablation without outcome token** to disentangle outcome conditioning
  from other sequential structure.
- **Stratified sampling** for rare outcomes: oversample
  checkmate/stalemate/insufficient-material games to get tighter
  per-outcome estimates without increasing total compute proportionally.

### Nice to have

- **Analytical bias correction.** For binomial estimates with known K
  (rollouts), the bias of max over n Binomial(K, p_i)/K estimators can be
  bounded analytically. This could provide tighter brackets than the
  split-half approach at the same rollout count.
- **Progress callback from Rust to Python.** Currently the Rust
  computation is a single blocking FFI call. A callback mechanism would
  enable real-time progress bars in the Python script without relying on
  stderr logging.

## Reproduction

```bash
cd engine && uv run --with maturin maturin develop --release && cd ..
uv run python scripts/compute_theoretical_ceiling.py \
    --n-games 5000 --rollouts 128 --sample-rate 0.05 \
    --model-accuracy 0.069 --bootstrap 2000
```

Results: `data/theoretical_ceiling.json`. Runtime: ~38 min on 16-core CPU.
