# PAWN Architecture

PAWN (Playstyle-Agnostic World-model Network for Chess) is a causal transformer trained on random chess games via next-token prediction. It learns chess rules, legal moves, and board state representations purely from move sequences, with no hand-crafted features or external game databases.

This document describes the model architecture in detail.

## Input Format

Each training example is a fixed-length sequence of 256 tokens:

```
[outcome, move_1, move_2, ..., move_N, PAD, PAD, ..., PAD]
 pos 0    pos 1   pos 2        pos N   pos N+1         pos 255
```

Position 0 holds an **outcome token** that tells the model how the game ends. The remaining 255 positions hold the game's moves in order, right-padded with PAD tokens. This outcome-first format gives the model access to the game result at every position via the causal attention mask, which is useful for downstream tasks that condition on game outcome.

The model is trained with standard next-token prediction: given positions 0 through t, predict the token at position t+1.

## Token Vocabulary

The vocabulary contains 4278 tokens:

| Range | Count | Description |
|-------|-------|-------------|
| 0 | 1 | PAD token |
| 1--4096 | 4096 | Grid moves (64 source squares x 64 destination squares) |
| 4097--4272 | 176 | Promotion moves (44 promotion src/dst pairs x 4 piece types) |
| 4273--4283 | 11 | Outcome tokens |

The outcome tokens are:

| Token ID | Meaning | Source |
|----------|---------|--------|
| 4273 | White delivers checkmate | Pretraining + Lichess |
| 4274 | Black delivers checkmate | Pretraining + Lichess |
| 4275 | Stalemate | Pretraining + Lichess |
| 4276 | Draw by rule (75-move, fivefold rep, insufficient material) | Pretraining + Lichess |
| 4277 | Ply limit reached (255 plies) | Pretraining + Lichess (truncated) |
| 4278 | White wins by resignation | Lichess only |
| 4279 | Black wins by resignation | Lichess only |
| 4280 | Draw by agreement | Lichess only |
| 4281 | White wins on time | Lichess only |
| 4282 | Black wins on time | Lichess only |
| 4283 | Draw on time (insufficient material) | Lichess only |

Tokens 4273-4277 are used during pretraining on random games. Tokens 4278-4283 appear only in Lichess finetuning data. The pretrained model has no trained embeddings for tokens 4278-4283; these are initialized during adapter training.

Move tokenization is handled entirely by the Rust chess engine, which maps UCI move strings (e.g., `e2e4`, `a7a8q`) to token indices.

## Factored Input Embeddings

Instead of a single embedding table of size 4278, PAWN uses **factored embeddings** that decompose each move token into its structural components. This exploits the fact that chess moves have compositional structure: a source square, a destination square, and an optional promotion piece.

For each move token, a static decomposition table maps it to a (source, destination, promotion) triple. The embedding is computed as:

```
embed(move) = src_embed[source] + dst_embed[destination] + promo_embed[promotion]
```

The embedding tables are:

- `src_embed`: 64 entries (one per square), each of dimension d_model
- `dst_embed`: 64 entries (one per square), each of dimension d_model
- `promo_embed`: 5 entries (none, queen, rook, bishop, knight), each of dimension d_model

This reduces the embedding parameter count from 4284 x d_model to 133 x d_model -- a roughly 32x reduction. It also provides structural inductive bias: moves that share a source or destination square share embedding components.

PAD and outcome tokens are not decomposed. PAD uses a standalone learned parameter vector. The 11 outcome tokens use a separate small embedding table.

## Transformer Architecture

PAWN uses a decoder-only transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) with the following design choices:

**Normalization.** Pre-norm with [RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019). Each transformer block applies RMSNorm before the attention sublayer and before the feed-forward sublayer:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

A final RMSNorm is applied after the last transformer block, before the output projection.

**Attention.** Standard multi-head self-attention with no bias terms in any of the projection matrices (Q, K, V, output). Attention uses PyTorch's `scaled_dot_product_attention` with a causal mask combined with a padding mask. The padding mask ensures that PAD tokens are not attended to.

**Positional encoding.** [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2104.09864) (Su et al., 2021) with base frequency 10000. RoPE is applied to the query and key vectors after projection, before the attention computation. Frequency tensors are precomputed for the full sequence length and stored as non-persistent buffers.

**Feed-forward network.** [SwiGLU](https://arxiv.org/abs/2002.05202) (Shazeer, 2020), a gated linear unit with SiLU activation, implemented as:

```
FFN(x) = W_down(SiLU(W_gate(x)) * W_up(x))
```

This uses three weight matrices per block instead of the standard two, with no bias terms. The intermediate dimension d_ff is 4x the model dimension.

**Output head.** A single linear projection from d_model to vocab_size (4278), producing logits over the full token vocabulary. No weight tying with the input embeddings.

**Weight initialization.** All parameters with more than one dimension are initialized from N(0, 0.02).

## Model Variants

| Variant | d_model | Layers | Heads | Head dim | d_ff | Parameters |
|---------|---------|--------|-------|----------|------|------------|
| Small   | 256     | 8      | 4     | 64       | 1024 | ~9.5M      |
| Base    | 512     | 8      | 8     | 64       | 2048 | ~35.8M     |
| Large   | 640     | 10     | 8     | 80       | 2560 | ~68.4M     |

All variants use the same vocabulary, sequence length (256), and architectural choices. They differ only in width, depth, and head count. A `toy` variant (d=64, 2 layers, 4 heads) exists for testing.

## Forward Pass Variants

The model provides three forward pass modes:

**`forward()`** -- Standard inference. Processes a full sequence and returns logits at every position along with intermediate hidden states from each layer. The hidden states are useful for linear probing experiments.

**`forward_train()`** -- Memory-optimized training. Runs the transformer backbone identically, but projects only non-padding positions through the output head. This avoids materializing the full (B, T, 4278) logit tensor, saving roughly 25% memory during training.

**`forward_generate()`** -- Autoregressive generation with KV-cache. On the first call (prefill), processes the full input sequence and builds the key-value cache. On subsequent calls (decode), processes a single new token and extends the cache. Returns logits only for the last position.

## Training

**Objective.** Standard cross-entropy next-token prediction. The loss is computed only at non-padding positions via a loss mask.

**Data.** Training games are generated on-the-fly by a Rust chess engine that plays random legal moves. Each batch is a fresh set of games produced from a deterministic seed, so no game is seen twice and no external data is required. The engine runs in parallel via rayon and produces batches fast enough to keep the GPU saturated.

**Seeding.** Each batch gets a deterministic seed computed as `base_seed + step * num_workers + worker_id`, ensuring exact reproducibility across restarts and different worker counts.

## What the Model Learns

Despite training exclusively on random games, PAWN learns rich chess representations:

- **Legal move prediction.** The model achieves over 98% top-1 accuracy at predicting the next randomly chosen legal move. Since the training distribution is uniform over legal moves, high accuracy implies the model has learned to enumerate legal moves from the move history alone.

- **Board state tracking.** Linear probes on intermediate hidden states can decode board-level features with high accuracy, including piece positions on each square, whether either side is in check, and castling rights. The model reconstructs the full board state internally even though it never sees explicit board representations.

These properties make PAWN useful as a frozen backbone for downstream tasks. Adapter methods (LoRA, FiLM, bottleneck adapters) can be trained on top of the frozen PAWN representations to produce strategic play from human game data, without modifying the base model's weights.
