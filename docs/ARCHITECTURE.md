# PAWN Architecture

> **v1 metrics disclaimer.** All metrics and benchmarks attached to
> the published `pawn-{small, base, large}` HF repos are v1 PyTorch
> numbers. v2 republishes to new HF repos
> (`pawn-{small, base, large}-v2` or similar). The v1 repos are **not
> loadable in v2** (the Phase-A `V=2000` un-factored-embedding redesign is
> architecturally incompatible and the legacy converter was removed in the
> H.2 commit); use the `v1.0.0` git tag for the v1 artifacts.


PAWN (Playstyle-Agnostic World-model Network for Chess) is a causal transformer trained on random chess games via next-token prediction. It learns chess rules, legal moves, and board state representations purely from move sequences, with no hand-crafted features or external game databases.

This document describes the model architecture in detail.

## Input Format

Each training example is a fixed-length sequence of 512 tokens:

```
[move_1, move_2, ..., move_N, PAD, PAD, ..., PAD]
 pos 0   pos 1        pos N-1 pos N           pos 511
```

Sequences are pure moves by default. Position 0 holds the first ply; the rest are the game's moves in order, right-padded with PAD tokens. The model is trained with standard next-token prediction: given positions 0 through t, predict the token at position t+1.

The 512-token context fits the vast majority of random games end-to-end. Random play hits truly degenerate game lengths only rarely at this width, so the `PLY_LIMIT` outcome — which dominated the legacy 256-token models — is now a small minority of the training distribution. (See [docs/LEGACY.md](LEGACY.md) for why this changed.)

An optional outcome-conditioning mode — enabled via the `prepend_outcome` training config — prepends a single outcome token at position 0 and shifts all moves right by one. This is useful for downstream tasks that condition on game outcome, at the cost of one ply of context. **The published v1.0.0 checkpoints were trained with `prepend_outcome=False`** (no outcome prefix), which makes them directly comparable to standard chess models like MAIA and the searchless_chess agents.

## Token Vocabulary

The model's uniform input/output vocabulary is `VOCAB_SIZE = 2000`. The first 1,968 entries are the move actions borrowed from Google DeepMind's [searchless_chess](https://github.com/google-deepmind/searchless_chess) project; the remainder are PAD, outcomes, and the Phase-A control tokens:

| Range | Count | Description |
|-------|-------|-------------|
| 0--1967 | 1,968 | Move actions (one entry per legally-reachable (src, dst[, promotion]) tuple) |
| 1968 | 1 | PAD token |
| 1969--1979 | 11 | Outcome tokens |
| 1980--1999 | 20 | Phase-A control tokens (BOS, NULL, + reserved conditioning slots) |

The outcome tokens are:

| Token ID | Meaning | Source |
|----------|---------|--------|
| 1969 | White delivers checkmate | Pretraining + Lichess |
| 1970 | Black delivers checkmate | Pretraining + Lichess |
| 1971 | Stalemate | Pretraining + Lichess |
| 1972 | Draw by rule (75-move, fivefold rep, insufficient material) | Pretraining + Lichess |
| 1973 | Ply limit reached | Pretraining + Lichess (truncated) |
| 1974 | White wins by resignation | Lichess only |
| 1975 | Black wins by resignation | Lichess only |
| 1976 | Draw by agreement | Lichess only |
| 1977 | White wins on time | Lichess only |
| 1978 | Black wins on time | Lichess only |
| 1979 | Draw on time (insufficient material) | Lichess only |

Tokens 1969-1973 are used during pretraining on random games. Tokens 1974-1979 appear only in Lichess finetuning data. The pretrained model has no trained embeddings for tokens 1974-1979; these are initialized during adapter training.

Move tokenization is handled entirely by the Rust chess engine, which maps UCI move strings (e.g., `e2e4`, `a7a8q`) to token indices.

## Token Embeddings

> **Historical note.** v1 (and early v2) used *factored* move embeddings that
> decomposed each move token into `src_embed[source] + dst_embed[destination]
> + promo_embed[promotion]`. The Phase-A redesign replaced that with the
> uniform table described below; the factored decomposition no longer exists
> in the code.

PAWN uses a single uniform embedding table `embed_tokens` of shape
`[VOCAB_SIZE, d_model]` over the whole `VOCAB_SIZE = 2000` vocabulary — the
1,968 move actions, PAD, the 11 outcome tokens, and the reserved
`BOS` / `NULL` / control slots all index the same table:

```
embed(token) = embed_tokens[token]
```

**Tied output head.** By default (`tie_embeddings = True` on `ModelConfig`)
the model has no separate `lm_head` array — output logits reuse the input
table transposed:

```
logits = x @ embed_tokens.T          # shape [..., VOCAB_SIZE]
```

Argmax callers restrict the result to `[0, NUM_ACTIONS)` so PAD, outcome,
`BOS`/`NULL`, and reserved tokens can never be sampled as a move. An untied
config keeps a standalone `lm_head[d_model, VOCAB_SIZE]` instead; the
checkpoint schema drops the `lm_head` field in the tied case (see
`pawn.model.saved_fields`).

## Transformer Architecture

PAWN uses a decoder-only transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) with the following design choices:

**Normalization.** Pre-norm with [RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019). Each transformer block applies RMSNorm before the attention sublayer and before the feed-forward sublayer:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

A final RMSNorm is applied after the last transformer block, before the output projection.

**Attention.** Standard multi-head self-attention with no bias terms in any of the projection matrices (Q, K, V, output). The v2 stack (JAX/Equinox/Optax) materialises the attention `QK^T` scores plainly with a causal × padding mask combined via `jax.numpy.where`; the v1 PyTorch stack used `torch.nn.functional.scaled_dot_product_attention`. At seq 512 attention is ~12% of step FLOPs and plain attention sidesteps fused-kernel maturity under JAX-on-ROCm. The padding mask ensures that PAD tokens are not attended to.

**Positional encoding.** [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2104.09864) (Su et al., 2021) with base frequency 10000. RoPE is applied to the query and key vectors after projection, before the attention computation. Frequency tensors are precomputed for the full sequence length and stored as non-persistent buffers.

**Feed-forward network.** [SwiGLU](https://arxiv.org/abs/2002.05202) (Shazeer, 2020), a gated linear unit with SiLU activation, implemented as:

```
FFN(x) = W_down(SiLU(W_gate(x)) * W_up(x))
```

This uses three weight matrices per block instead of the standard two, with no bias terms. The intermediate dimension d_ff is 4x the model dimension.

**Output head.** Logits over the full `VOCAB_SIZE = 2000` vocabulary. By default (`tie_embeddings`) the output head is the input `embed_tokens` table transposed — `logits = x @ embed_tokens.T`, no separate `lm_head` parameter (see *Token Embeddings* above). An untied config keeps a standalone `lm_head[d_model, VOCAB_SIZE]` linear projection instead. *(v1 used an untied `d_model → 1,980` projection.)*

**Weight initialization.** All parameters with more than one dimension are initialized from N(0, 0.02).

## Model Variants

**v1 (published HF checkpoints — PyTorch):**

| Variant | d_model | Layers | Heads | Head dim | d_ff | Parameters |
|---------|---------|--------|-------|----------|------|------------|
| Small   | 256     | 8      | 4     | 64       | 1024 | 8.94M      |
| Base    | 512     | 8      | 8     | 64       | 2048 | 34.65M     |
| Large   | 640     | 10     | 8     | 80       | 2560 | 66.91M     |

**v2 (JAX/Equinox/Optax — supernet + nested slices, all share depth + head_dim):**

| Variant | d_model | Layers | Heads | Head dim | d_ff |
|---------|---------|--------|-------|----------|------|
| Small   | 256     | 10     | 4     | 64       | 1024 |
| Base    | 512     | 10     | 8     | 64       | 2048 |
| Large   | 640     | 10     | 10    | 64       | 2560 |

The v2 stack pins `head_dim = 64` so width slices align to whole heads and RoPE is variant-invariant; all variants share the supernet's depth (10 layers) so the inner `[:d_V, :d_V]` of every weight matrix gives a valid sub-model. A `TINY_SUPERNET` (d=192, 4 layers, 3 heads) exists for verification runs that don't need production scale.

The v1.0.0 parameter counts are slightly lower than the legacy `-legacy` repos with the same `d_model`/`n_layers`/`n_heads`, because the move vocabulary has roughly half the entries of the old 4,278-token vocab. In v2 the uniform `embed_tokens[VOCAB_SIZE, d_model]` table is where vocab size enters the parameter count; with the default `tie_embeddings` the output head reuses that same table (transposed), so the vocab contributes a single `VOCAB_SIZE × d_model` block rather than separate input and output tables.

## Forward Pass Variants

The model provides three forward pass modes:

**`forward()`** -- Standard inference. Processes a full sequence and returns logits at every position along with intermediate hidden states from each layer. The hidden states are useful for linear probing experiments.

**`forward_train()`** -- Memory-optimized training. Runs the transformer backbone identically, but projects only non-padding positions through the output head. This avoids materializing the full (B, T, 1980) logit tensor, saving memory during training.

**`forward_generate()`** -- Autoregressive generation with KV-cache. On the first call (prefill), processes the full input sequence and builds the key-value cache. On subsequent calls (decode), processes a single new token and extends the cache. Returns logits only for the last position.

## Training

**Objective.** Standard cross-entropy next-token prediction. The loss is computed only at non-padding positions via a loss mask.

**Data.** Training games are generated on-the-fly by a Rust chess engine that plays random legal moves. Each batch is a fresh set of games produced from a deterministic seed, so no game is seen twice and no external data is required. The engine runs in parallel via rayon and produces batches fast enough to keep the GPU saturated.

**Seeding.** Each batch gets a deterministic seed computed as `base_seed + step * num_workers + worker_id`, ensuring exact reproducibility across restarts and different worker counts.

## What the Model Learns

Despite training exclusively on random games, PAWN learns rich chess representations:

- **Legal move prediction.** The base and large variants achieve >99.99% per-move legal rate. Since the training distribution is uniform over legal moves, the only way to predict a legal move with high probability is to know which moves are currently legal — i.e., to track the rules and the underlying board state from the move history alone.

- **Board state tracking.** Linear probes on intermediate hidden states can decode board-level features with high accuracy, including piece type on each square, whether either side is in check, castling rights, and the en passant target square. The model reconstructs the full board state internally even though it never sees explicit board representations.

These properties make PAWN useful as a frozen backbone for downstream tasks. Adapter methods (LoRA, FiLM, bottleneck adapters) can be trained on top of the frozen PAWN representations to produce strategic play from human game data, without modifying the base model's weights.

## Game Completion Rate

Top-1 legal-move rate is the obvious metric for a backbone trained on random games, but it badly understates how much capacity matters. A model with 99.7% legal rate sounds essentially perfect — but if a 200-ply game has to clear ~200 independent dice rolls at 0.997 each, the per-game survival probability is only `0.997**200 ≈ 55%`. A single illegal move at any point in autoregressive play would forfeit, so per-move legality compounds badly.

To measure this directly, the v1.0.0 evaluation includes a **game completion rate**: for each game in the validation set, the model is shown every position along one side's turns (non-autoregressively — the model sees the actual ground-truth history at each ply) and asked to predict that side's next move. If *every* prediction across the game is legal, the game counts as completed; if any prediction at any ply is illegal, the game counts as a forfeit.

This metric compounds illegality the way real generation would, and turns out to depend much more strongly on model capacity than per-move accuracy does:

| Variant | Top-1 legal rate | Game completion rate |
|---------|------------------|----------------------|
| Small (8.94M) | 99.7451% | 52.34% |
| Base (34.65M) | 99.9962% | 98.97% |
| Large (66.91M) | 99.9990% | 99.76% |

(Numbers are from the published `model.safetensors` for each variant — the best 5K-cadence checkpoint by val loss, at step 195,000 of 200,000 for all three.)

The small model has a respectable per-move legal rate but forfeits roughly half its games on the way to terminal. The base and large models clear nearly every game. This gap is essentially invisible at the loss level — train/val curves for all three sizes look similar — and only shows up once you measure compound legality. It was the main signal that the legacy 100K-step models were nowhere near "done" training; doubling the step count to 200K made the difference.

The numbers above are non-autoregressive: the model sees the ground-truth history at each ply, so each prediction is independent given that history. **Autoregressive game completion has not been measured for these checkpoints.** It could come out higher or lower than the non-AR numbers — there are arguments both ways (errors in AR generation corrupt the visible history for every subsequent ply, but a competent model may also be biased toward the move distribution it was trained on, which would help it stay on-distribution as it generates) and the only honest answer is to actually run it. Measuring the AR forfeit rate is a known TODO.
