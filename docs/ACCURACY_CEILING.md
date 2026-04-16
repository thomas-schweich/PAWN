# Theoretical Accuracy Ceiling

PAWN is trained on uniformly random chess games. At each position with
N legal moves, the next move is drawn uniformly from those N moves, so
the Bayes-optimal predictor (without outcome information) can do no
better than `1/N` at that position. Top-1 accuracy therefore has a
hard theoretical ceiling — no model, however large, can exceed it on
the average random-game position.

## The ceiling: E[1/N_legal]

Averaged over the position distribution induced by random play, the
top-1 ceiling is

    E[1/N_legal]

where the expectation is over positions sampled from random games.
Importantly this is **not** equal to `1 / E[N_legal]`. By Jensen's
inequality (`1/x` is convex):

    E[1/N] ≥ 1 / E[N]

with equality only if N is constant. For a concrete example, suppose
half of all positions have N=10 and half have N=40:

- `E[1/N] = 0.5 · (1/10) + 0.5 · (1/40) = 0.0625` (6.25%)
- `1 / E[N] = 1 / 25 = 0.04` (4.00%)

The reciprocal-of-the-mean version under-counts the ceiling. The
correct computation evaluates `1/N` at every position and then
averages.

## v1.0.0 result

For the v1.0.0 backbones, which use a 512-token context and no outcome
conditioning, the ceiling computed over **50,000 fresh random games
(17.6M positions)** is:

| Quantity | Value |
|---|---|
| Unconditional ceiling `E[1/N_legal]` | **8.43%** |
| 95% CI (clustered bootstrap, by game) | [8.41%, 8.45%] |
| Average legal moves per position | 22.1 |
| Lower bound `1 / E[N]` (Jensen, *not* the ceiling) | 4.53% |
| Position distribution | uniformly random play, max 512 ply |

The ceiling is much higher than the legacy ~6.5% number because
context length matters a lot: random games of up to 512 ply spend a
large fraction of their plies in late-game positions where the board
has few pieces and few legal moves (avg N ≈ 13 in plies 384–512 vs
N ≈ 30 around plies 20–80), and `1/N` weights those late positions
heavily.

### Ceiling by ply bucket

| Ply range | Ceiling | Avg N_legal | Positions |
|-----------|--------:|------------:|----------:|
|   0 –   9 | 4.47% | 23.8 |   500K |
|  10 –  19 | 4.00% | 29.6 |   499K |
|  20 –  39 | 4.06% | 32.3 |   996K |
|  40 –  79 | 4.54% | 32.6 |  1.97M |
|  80 – 159 | 6.04% | 27.4 |  3.78M |
| 160 – 255 | 9.11% | 19.3 |  4.19M |
| 256 – 383 | 11.96% | 14.8 |  4.12M |
| 384 – 512 | 13.57% | 12.7 |  1.54M |

Late-game positions completely dominate the average. This is the same
pattern that shows up in the legacy 256-context analysis, just with
many more positions in the long tail.

## v1.0.0 model results (200K steps, prepend_outcome=False)

| Variant | Params | Top-1 | vs Ceiling |
|---------|-------:|------:|-----------:|
| Small | 8.94M | 8.54% | 101.4% |
| Base | 34.65M | 8.57% | 101.7% |
| Large | 66.91M | 8.63% | 102.4% |

(Headline metrics are from the best 5K-cadence checkpoint by val loss
for each variant — step 195,000 of 200,000 for all three.)

All three variants sit essentially at the unconditional ceiling, with
a very slight excess of ~1–2 percentage points relative to it. A few
plausible sources of that excess:

- **Sequential structure in random play.** The next-move distribution
  isn't quite uniform-given-history once you condition on the actual
  sequence of past moves: a position reached via random play tends to
  have certain pieces on certain squares with non-uniform probability,
  and the model can pick up on that. The ceiling assumes the
  predictor has no such information.
- **Different position distributions.** The ceiling above is computed
  over 50K fresh games, while the model val accuracy is averaged over
  a 2,048-game val set. Positional drift between the two samples is
  small but non-zero.
- **Statistical noise.** Both the val accuracy and the ceiling have
  CIs of order 0.1pp, so a 1–2pp gap is real but not enormous.

The headline takeaway: under the v1.0.0 setup, all three sizes are
within 1–2pp of the achievable Bayes-optimal accuracy on random games.
**Top-1 accuracy is essentially saturated and cannot meaningfully
distinguish capacity between sizes.** The metric that *does* separate
them is **game completion rate**; see [docs/ARCHITECTURE.md](ARCHITECTURE.md#game-completion-rate).

## Why this is the only ceiling that matters for v1.0.0

A previous version of this document also discussed an outcome-
conditioned ceiling computed via Monte Carlo rollouts. That ceiling
matters when the model is told the game's actual outcome ahead of time
(via a prepended outcome token), because outcome knowledge lets a
Bayes-optimal predictor non-trivially favor moves consistent with that
outcome — especially near the end of the game.

The v1.0.0 backbones use **no outcome prefix** (`prepend_outcome=False`),
so there is no outcome information leaking into the model's input and
the MC ceiling collapses to the unconditional ceiling. Computing it
would just produce the same number with extra steps. The legacy
`-legacy` backbones did use outcome conditioning; their results sat
~3–5pp above the unconditional ceiling, which was the load-bearing
evidence that they were exploiting the prefix.

## Reproducing

```bash
# Default: 20K games, ~512-ply context, ~5s on a modern multicore.
uv run python scripts/compute_theoretical_ceiling.py

# Tighter CI (50K games, ~14s, what the published v1.0.0 number uses)
uv run python scripts/compute_theoretical_ceiling.py --n-games 50000

# Replicate the legacy 255-ply distribution for comparison
uv run python scripts/compute_theoretical_ceiling.py --max-ply 255

# Compare a specific top-1 accuracy against the ceiling
uv run python scripts/compute_theoretical_ceiling.py --model-accuracy 0.0857
```

Output is saved to `cards/theoretical_ceiling.json` with the ceiling,
its 95% bootstrap CI, the per-ply-bucket breakdown, and the run
parameters. The model card generator
(`scripts/generate_model_cards.py`) reads the JSON from that path to
fill in the "Accuracy ceiling" section of each model card. The
artifact is checked in so that re-rendering cards on a fresh clone
does not require a fresh ceiling computation.

## Methodology

The computation is straightforward and uses only existing Rust
primitives — no rollouts, no MC, no bias correction.

1. `chess_engine.generate_random_games(n, max_ply, seed)` produces a
   batch of uniformly random legal games.
2. `chess_engine.compute_legal_token_masks_sparse(...)` returns flat
   int64 indices into a `(batch, seq_len, vocab_size)` legal-move
   tensor, one entry per legal token at every ply.
3. Counting indices per `(game, ply)` gives `N_legal` at every
   non-terminal position.
4. The ceiling is `mean(1 / N_legal)` over those positions.
5. The 95% CI is a clustered bootstrap that resamples whole games at
   a time (positions within a game are correlated, so per-position
   resampling under-counts variance).

The bootstrap is vectorized: per-cluster sums and counts are computed
once, then each bootstrap iteration is a sum-of-sums divided by
sum-of-counts. This is `O(n_boot × n_clusters)` instead of
`O(n_boot × n_positions)`, which is the difference between "instant"
and "many minutes" at 50K games.

## Limitations

- **The ceiling assumes the predictor has no information beyond the
  current position's legal-move set.** A model trained on millions of
  random games can pick up subtle distributional biases (e.g. piece
  locations are not uniformly distributed conditional on a particular
  random move history) that the ceiling doesn't account for. This is
  why the v1.0.0 models clear the ceiling by ~1–2pp.
- **Tied to the position distribution of random play.** The number is
  not a property of chess; it's a property of the random-play
  distribution at a given context length. A model evaluated on a
  different distribution (e.g. human games) needs a different
  ceiling.
- **Per-ply-bucket numbers are descriptive only.** They are not
  separate ceilings — the ceiling is a single average, and bucketing
  is just for understanding which positions dominate it.
