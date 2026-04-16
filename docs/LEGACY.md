# Legacy PAWN Architecture (v0.x)

Before the v1.0.0 release, PAWN backbones used a different vocabulary, a
narrower context window, and outcome conditioning. Those models are still
available and useful for reproducing prior experiments, but they are no
longer the primary checkpoints, and the current `main` branch can no
longer load them.

This document explains what the legacy architecture was, why it was
replaced, and how to access it if you need it.

## What the legacy backbones looked like

| Item | Legacy (v0.x) | v1.0.0 |
|------|---------------|--------|
| Vocabulary | 4,278-token coordinate vocab (every UCI-style coordinate pair, including impossible ones like `a1a1` and `b1a7`) | 1,980-token searchless_chess action vocab (1,968 reachable actions + 1 PAD + 11 outcome tokens) |
| Context window | 256 tokens | 512 tokens |
| Outcome conditioning | **On.** Each example began with one of 5 outcome tokens (`white_checkmate`, `black_checkmate`, `stalemate`, `draw_by_rule`, `ply_limit`), retroactively prepended | **Off** by default. The vocab still contains 11 outcome tokens (the original 5 plus 6 Lichess-specific ones) so they can be used for opt-in outcome-conditioned variants, but the published checkpoints don't use them |
| Training steps | 100K | 200K |
| Warmup | 1,000 steps | 10,000 steps |
| Eval interval | every 500 steps on 512 random games | every 1,000 steps on 2,048 random games |
| HuggingFace repos | `thomas-schweich/pawn-{small,base,large}-legacy` | `thomas-schweich/pawn-{small,base,large}` |
| Code snapshot | git tag `pre-vocab-transition` | git tag `v1.0.0` (and forward) |

The transformer architecture itself (RMSNorm, SwiGLU, RoPE, factored
embeddings, layer/head/d_model counts per variant) is unchanged. The
v1.0.0 backbones are slightly smaller in parameter count only because
the smaller vocab shrinks the `lm_head` projection.

## Why the legacy architecture was replaced

Three problems motivated the redesign.

### 1. The 256-token context made random games pathological

Random play almost never produces a chess game that ends naturally inside
255 ply. The legacy training data was therefore dominated by games that
hit the ply limit without resolution — about 80% of training examples had
the `ply_limit` outcome token. The model was essentially learning to
predict random moves under a "this game probably won't terminate" prior,
and the actual terminal-state signal was very sparse.

The 512-token window changes this: the vast majority of random games now
terminate naturally inside the context, so checkmate/stalemate/draw
become a meaningful share of the training distribution and `ply_limit`
shrinks to a tail outcome.

### 2. The compound-legality story was much worse than headline metrics suggested

The legacy backbones all reported >98% top-1 legal move rate, which
sounds essentially solved. But "98% per move" compounds into a forfeit
about every 50 plies on average — and a single illegal move at any point
in the game forfeits the whole game. When measured directly as a **game
completion rate** (no illegal predictions across an entire game's plies
on one side), the legacy models looked like:

In the table below, "avg plies completed" is the average number of legal plies the model produced before either reaching the end of the game or making its first illegal prediction. It is shown out of the legacy 255-ply context limit, since legacy validation games could not extend further.

| Variant | Steps | Params | Forfeit rate | Per-move illegal rate | Avg plies completed (out of 255) | Median forfeit ply | Min/max forfeit ply |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pawn-small`-legacy  | 100K | 9.5M  | 0.6797 | 6.8e-3 | 146.7 | 105 | 13 / 252 |
| `pawn-base`-legacy   | 100K | 35.8M | 0.0542 | 2.71e-4 | 229.7 | 126 | 39 / 251 |
| `pawn-large`-legacy  | 100K | 68.4M | 0.0327 | ~2.0e-4 | 232.4 | 121 | 37 / 254 |

Once it became clear the legacy models really weren't "done" training,
doubling the step count to 200K (with the wider context and the cleaner
vocabulary) brought them dramatically closer to "knows the rules":

| Variant | Steps | Params | Game completion | Per-move legal rate |
|---|---:|---:|---:|---:|
| pawn-small  | 200K | 8.94M  | 51.66% | 99.7457% |
| pawn-base   | 200K | 34.65M | 99.02% | 99.9962% |
| pawn-large  | 200K | 66.91M | 99.80% | 99.9996% |

Note also that the v1.0.0 numbers above were measured against games of
**up to 512 ply** — a significantly harder evaluation than the legacy
table's 255-ply cap. See [docs/ARCHITECTURE.md](ARCHITECTURE.md#game-completion-rate) for the full discussion.

### 3. Outcome conditioning made comparisons to standard chess models awkward

The legacy backbones were trained with the game's actual outcome prepended
to every sequence. The original idea was that the outcome token could be
adapted into a "system prompt" of sorts — e.g., "play as white" — and
during play against Stockfish, the model was always given "I win" as the
outcome.

This is unusual. Standard chess engines are zero-sum and side-agnostic;
their architecture predicts the side-to-move's best moves regardless of
who that side is, and they never get to know the game outcome ahead of
time. Legacy PAWN's accuracy was therefore inflated relative to e.g.
the MAIA models, because PAWN had access to information they didn't.
Direct numerical comparisons across architectures became fraught.

The v1.0.0 models do not use outcome conditioning by default. The
vocabulary still includes outcome tokens (and `prepend_outcome=True` is
still a supported training option) so that future experiments can revisit
outcome conditioning intentionally — but it is no longer the default
mode, and the published checkpoints are now directly comparable to
standard chess models.

## Adapter experiments on the legacy backbones

The legacy backbones were the testbed for the adapter experiments described in
[docs/ADAPTERS.md](ADAPTERS.md): bottleneck adapters, LoRA, FiLM, sparse,
hybrid, and RoSA. Behavioral cloning of Lichess players in a fixed Elo band
reached nearly 50% top-1 accuracy with enough adapter capacity. Linear
probes on the frozen hidden states recovered piece tracking, board
geometry, and rules (even the 75-move rule was easy to derive).

Triangulating against Stockfish via `UCI_LimitStrength`, an 1800–1900-Elo
adapter trained to ~50% accuracy played at roughly **1250 Elo** — well
below the rating it was cloning. By contrast, MAIA's behavioral-clone
models (~55% accuracy) tend to play *above* the skill they were trained
on, an effect the MAIA authors call the "committee effect" (the model
plays more like a committee of players at that rating than a single
player at that rating, and rarely blunders catastrophically).

The legacy adapter numbers are the most recent end-to-end adapter sweep
the project has, and they are still in [docs/ADAPTERS.md](ADAPTERS.md).
They have not yet been re-run on the v1.0.0 backbones; expect updated
numbers in a future release.

## How to access the legacy code and weights

### Legacy weights

The legacy checkpoints are mirrored as their own HuggingFace repos:

- [`thomas-schweich/pawn-small-legacy`](https://huggingface.co/thomas-schweich/pawn-small-legacy)
- [`thomas-schweich/pawn-base-legacy`](https://huggingface.co/thomas-schweich/pawn-base-legacy)
- [`thomas-schweich/pawn-large-legacy`](https://huggingface.co/thomas-schweich/pawn-large-legacy)

These are byte-for-byte the same as the original `main`-branch revisions
of the non-`-legacy` repos before the v1.0.0 cutover.

### Legacy code

The current `main` branch of this repo cannot load the legacy weights —
the vocabulary, sequence layout, and outcome handling were ripped out in
PRs #67 and #68. To work with the legacy backbones, check out the
`pre-vocab-transition` git tag:

```bash
git checkout pre-vocab-transition
cd engine && uv run --with maturin maturin develop --release && cd ..
uv sync --extra cu128   # or --extra rocm
```

That tag preserves the exact code the legacy checkpoints were trained
with, including the dual-vocabulary support (one vocab for pretraining,
one for finetuning) and the `prepend_outcome=True` defaults.

### Legacy Lichess data

The Lichess parquet dataset at `thomas-schweich/pawn-lichess-full`
historically used the legacy 4,278-token vocab in its `tokens` column.
Older revisions of that dataset are still accessible by checking out the
revision in question (the `main` revision was rewritten to use the new
vocabulary as part of the v1.0.0 cutover, but prior revisions remain in
git history). There is intentionally no separate `-legacy` dataset repo
— the raw Lichess `.pgn.zst` files in the repo are vocabulary-agnostic
and useful regardless of which token layout you want to consume.
