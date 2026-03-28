---
library_name: pawn
license: apache-2.0
base_model:
  - thomas-schweich/pawn-small
  - thomas-schweich/pawn-base
  - thomas-schweich/pawn-large
tags:
  - chess
  - transformer
  - world-model
  - causal-lm
  - next-token-prediction
  - representation-learning
  - pytorch
  - rust
model_name: PAWN-{VARIANT_NAME}
pipeline_tag: other
citation: |
  @software{schweich2025pawn,
    author = {Schweich, Thomas},
    title = {{PAWN}: Playstyle-Agnostic World-model Network for Chess},
    year = {2025},
    url = {https://github.com/thomas-schweich/PAWN},
    license = {Apache-2.0}
  }
model_params: {PARAMS_NUM}
d_model: {D_MODEL_NUM}
n_layers: {N_LAYERS_NUM}
n_heads: {N_HEADS_NUM}
d_ff: {D_FF_NUM}
context_length: 256
vocab_size: 4278
datasets:
  - random-chess-games
language:
  - en
metrics:
  - accuracy
model-index:
  - name: PAWN-{VARIANT_NAME}
    results:
      - task:
          type: next-token-prediction
          name: Chess Move Prediction (Random Games)
        metrics:
          - name: Legal Move Rate
            type: accuracy
            value: {LEGAL_MOVE_RATE}
          - name: Accuracy / Unconditioned Ceiling
            type: accuracy
            value: {UNCOND_RATIO_NUM}
          - name: Accuracy / Bayes-Optimal Ceiling
            type: accuracy
            value: {MCTS_RATIO_NUM}
          - name: Top-1 Accuracy
            type: accuracy
            value: {TOP1_ACC_NUM}
          - name: Top-5 Accuracy
            type: accuracy
            value: {TOP5_NUM}
          - name: Val Loss
            type: loss
            value: {VAL_LOSS_NUM}
          - name: Games Seen
            type: other
            value: 25600000
---

# PAWN-{VARIANT_NAME}

**PAWN** (Playstyle-Agnostic World-model Network for Chess) is a causal transformer trained on random chess games. It learns legal moves, board state representations, and game dynamics purely from uniformly random legal move sequences -- no strategic play, no hand-crafted features, no external game databases.

This is the **{VARIANT_LABEL}** variant ({PARAMS} parameters). PAWN is designed as a frozen backbone for parameter-efficient finetuning into player models with arbitrary playstyles.

**[GitHub Repository](https://github.com/thomas-schweich/PAWN)** -- full source code, training scripts, adapter implementations, and documentation.

## All Variants

| Variant | Parameters | Link |
|---------|------------|------|
| PAWN-Small | ~9.5M | [thomas-schweich/pawn-small](https://huggingface.co/thomas-schweich/pawn-small) |
| PAWN (Base) | ~35.8M | [thomas-schweich/pawn-base](https://huggingface.co/thomas-schweich/pawn-base) |
| PAWN-Large | ~68.4M | [thomas-schweich/pawn-large](https://huggingface.co/thomas-schweich/pawn-large) |

## Headline Metrics

PAWN is trained on uniformly random chess games, so top-1 accuracy has a hard theoretical ceiling -- no model can exceed it. We report accuracy relative to three ceilings to contextualize model performance. For full details, see [Accuracy Ceiling Analysis](https://github.com/thomas-schweich/PAWN/blob/main/docs/ACCURACY_CEILING.md).

| Metric | Value |
|--------|-------|
| Legal move rate | {LEGAL_MOVE_RATE} |
| Top-1 accuracy | {TOP1_ACC} |

### Accuracy Ratios

The ceilings below represent the best possible top-1 accuracy under different assumptions about what the model can know. Ratios above 100% on the unconditioned ceiling indicate the model has learned structure beyond simply identifying legal moves.

| Ceiling | Accuracy / Ceiling | Ratio |
|---------|-------------------|-------|
| Unconditioned (E\[1/N_legal\] = 6.43%) | {ACC} / 6.43% | {UNCOND_RATIO} |
| Naive-conditioned (1-ply filter = 6.44%) | {ACC} / 6.44% | {NAIVE_RATIO} |
| Bayes-optimal conditioned (MCTS, 32 rollouts = 7.92%) | {ACC} / 7.92% | {MCTS_RATIO} |

**Unconditioned ceiling** (6.43%): The expected accuracy of a predictor that knows only which moves are legal at each position and picks uniformly. A model exceeding this has learned to estimate the number of legal moves and bias predictions toward constrained positions.

**Naive-conditioned ceiling** (6.44%): An analytical estimate that excludes moves leading to an immediate terminal state inconsistent with the game's actual outcome. This barely exceeds the unconditioned ceiling because immediate terminal states are rare.

**Bayes-optimal conditioned ceiling** (7.92%): The Monte Carlo estimate of the best achievable accuracy given perfect knowledge of P(outcome | move, history). This is the tightest bound. PAWN's input sequence begins with an outcome token, which leaks information about the game's trajectory. The MCTS ceiling quantifies the maximum benefit of this conditioning.

## Probe Results

Linear probes trained on frozen hidden states measure how well the model's internal representations encode board-level features. All probes are single linear layers trained on {PROBE_N_TRAIN} positions and evaluated on {PROBE_N_VAL} held-out positions.

| Probe | Result | Description |
|-------|--------|-------------|
| Piece type | {PROBE_PIECE_TYPE} | Per-square piece type (13 classes x 64 squares) |
| Side to move | {PROBE_SIDE_TO_MOVE} | Whose turn it is |
| Is check | {PROBE_IS_CHECK} | Whether the side to move is in check |
| Castling rights | {PROBE_CASTLING} | KQkq castling availability |
| En passant square | {PROBE_EP_SQUARE} | En passant target square (64 + none) |
| Material count | {PROBE_MATERIAL} | Piece counts per type per color |
| Legal move count | {PROBE_LEGAL_MOVES} | Number of legal moves available |
| Halfmove clock | {PROBE_HALFMOVE} | Plies since last capture or pawn move |
| Game phase | {PROBE_GAME_PHASE} | Opening / middlegame / endgame |

## Diagnostic Results

Edge-case diagnostics measure the model's accuracy and legal move rate in specific tactical situations. Positions are extracted from a corpus of random games and evaluated in isolation.

| Category | Positions | Legal Rate |
|----------|-----------|------------|
| In check | {DIAG_CHECK_N} | {DIAG_CHECK_LEGAL} |
| Double check | {DIAG_DCHECK_N} | {DIAG_DCHECK_LEGAL} |
| Pin restricts movement | {DIAG_PIN_N} | {DIAG_PIN_LEGAL} |
| En passant available | {DIAG_EP_N} | {DIAG_EP_LEGAL} |
| Castling legal (kingside) | {DIAG_CASTLE_K_N} | {DIAG_CASTLE_K_LEGAL} |
| Castling legal (queenside) | {DIAG_CASTLE_Q_N} | {DIAG_CASTLE_Q_LEGAL} |
| Castling blocked by check | {DIAG_CASTLE_BLOCK_N} | {DIAG_CASTLE_BLOCK_LEGAL} |
| Promotion available | {DIAG_PROMO_N} | {DIAG_PROMO_LEGAL} |
| Checkmate (terminal) | {DIAG_MATE_N} | {DIAG_MATE_PAD_PROB} |
| Stalemate (terminal) | {DIAG_STALE_N} | {DIAG_STALE_PAD_PROB} |

For terminal positions (checkmate, stalemate), there are no legal moves. The "Legal Rate" column instead reports the probability the model assigns to the PAD token — i.e., how often it correctly recognizes the game is over.

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only transformer |
| d_model | {D_MODEL} |
| Layers | {N_LAYERS} |
| Attention heads | {N_HEADS} |
| Head dimension | {HEAD_DIM} |
| d_ff | {D_FF} |
| Parameters | {PARAMS} |
| Vocabulary | 4,278 tokens |
| Context length | 256 tokens |
| Normalization | Pre-norm RMSNorm |
| FFN | SwiGLU (4x expansion) |
| Positional encoding | Rotary (RoPE, base 10000) |
| Embeddings | Factored (src + dst + promo) |
| Dropout | 0.0 |

The token vocabulary consists of 1 PAD token, 4,096 grid moves (64 source squares x 64 destination squares), 176 promotion moves (44 src/dst pairs x 4 piece types), and 5 outcome tokens (WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE, DRAW_BY_RULE, PLY_LIMIT).

Each input sequence has the format `[outcome] [ply_1] ... [ply_N] [PAD] ... [PAD]`, where the outcome token is prepended during training so the model can condition on how the game ends. Move embeddings are factored into source square + destination square + promotion piece components, reducing embedding parameters by roughly 32x while providing structural inductive bias.

The model receives no board representation, piece type information, or geometric features. All state tracking is learned internally from move sequences alone.

## Training Details

| Parameter | Value |
|-----------|-------|
| Training data | On-the-fly uniformly random legal games (no external dataset) |
| Objective | Next-token cross-entropy (non-padding positions only) |
| Total steps | 100,000 |
| Batch size | 256 |
| Games seen | 25,600,000 |
| Learning rate | 3e-4 (cosine decay with 1,000-step warmup) |
| Optimizer | AdamW (weight decay 0.01) |
| Precision | Mixed (AMP) |
| Max gradient norm | 1.0 |
| Hardware | NVIDIA H200 |
| Chess engine | Rust (shakmaty + rayon), ~43K games/sec |

Training data is generated on-the-fly by a Rust chess engine that plays uniformly random legal moves. Each batch is a fresh set of games produced from a deterministic seed, so no game is seen twice. The engine runs with rayon parallelism and produces batches fast enough to keep the GPU fully saturated.

Games are retroactively prepended with their actual outcome token. The model is not masked to legal moves during training; it must learn which moves are legal based on the sequence of prior moves.

## Usage

### Loading the model

```python
import torch
from safetensors.torch import load_file
from pawn.config import CLMConfig
from pawn.model import PAWNCLM

# Initialize and load weights
cfg = CLMConfig.{VARIANT_FACTORY}()
model = PAWNCLM(cfg).cuda().eval()
weights = load_file("model.safetensors", device="cuda")
model.load_state_dict(weights)
```

### Autoregressive generation

```python
from pawn.config import WHITE_CHECKMATES, PAD_TOKEN

# Start a game conditioned on white delivering checkmate
input_ids = torch.tensor([[WHITE_CHECKMATES]], device="cuda")
pad_mask = torch.ones(1, 1, dtype=torch.bool, device="cuda")

generated = [WHITE_CHECKMATES]
for _ in range(255):
    logits, _ = model.forward_generate(input_ids, pad_mask)
    next_token = logits[0, -1].argmax().item()
    if next_token == PAD_TOKEN:
        break
    generated.append(next_token)
    input_ids = torch.tensor([[next_token]], device="cuda")
    pad_mask = torch.ones(1, 1, dtype=torch.bool, device="cuda")
```

### Extracting hidden states for probing

```python
import torch
from pawn.config import CLMConfig
from pawn.model import PAWNCLM

cfg = CLMConfig.{VARIANT_FACTORY}()
model = PAWNCLM(cfg).cuda().eval()
# ... load weights ...

# input_ids: (B, T) tensor of token IDs
# pad_mask: (B, T) boolean tensor (True = real token)
logits, layer_hiddens = model(input_ids, pad_mask)
# layer_hiddens: list of (B, T, d_model) tensors, one per layer
```

### Finetuning with an adapter

```bash
# Install dependencies
cd engine && uv run --with maturin maturin develop --release && cd ..
uv sync --extra cu128 --extra dev

# Train a bottleneck adapter on Lichess games
uv run python scripts/train_bottleneck.py \
    --checkpoint path/to/pawn-{VARIANT_LOWER} \
    --pgn data/lichess_1800_1900.pgn \
    --bottleneck-dim 32 --lr 1e-4 --local-checkpoints
```

## Accuracy Ceiling Analysis

The accuracy ratios reported above are derived from a theoretical analysis of the maximum achievable top-1 accuracy on uniformly random chess games. Since each move is drawn uniformly from the legal move set, there is a hard ceiling that no model -- however large -- can exceed.

The **unconditioned ceiling** (6.43%) is the average of 1/N_legal across all positions in random games: the best a predictor can do without any context beyond the current position's legal moves. The **Bayes-optimal conditioned ceiling** (7.92%) accounts for the information leaked by the outcome token at position 0, estimated via Monte Carlo rollouts.

Models that exceed the unconditioned ceiling have learned structure beyond simple move legality. The gap between the unconditioned and MCTS ceilings quantifies the value of outcome conditioning. For most games (ply-limit outcomes, which dominate the distribution), the conditioning boost is small (1.07x). For decisive outcomes (checkmate, stalemate), the boost is substantial (2.6x).

Full methodology and per-outcome breakdowns are available in the [accuracy ceiling documentation](https://github.com/thomas-schweich/PAWN/blob/main/docs/ACCURACY_CEILING.md).

## Acknowledgments

PAWN builds on ideas and tools from the following projects and publications:

| Component | Reference |
|-----------|-----------|
| Transformer | [Vaswani et al., "Attention Is All You Need", NeurIPS 2017](https://arxiv.org/abs/1706.03762) |
| RMSNorm | [Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019](https://arxiv.org/abs/1910.07467) |
| RoPE | [Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021](https://arxiv.org/abs/2104.09864) |
| SwiGLU | [Shazeer, "GLU Variants Improve Transformer", 2020](https://arxiv.org/abs/2002.05202) |
| AdamW | [Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019](https://arxiv.org/abs/1711.05101) |
| Cosine schedule | [Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017](https://arxiv.org/abs/1608.03983) |
| Mixed precision | [Micikevicius et al., "Mixed Precision Training", ICLR 2018](https://arxiv.org/abs/1710.03740) |
| Bottleneck adapters | [Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019](https://arxiv.org/abs/1902.00751) |
| LoRA | [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022](https://arxiv.org/abs/2106.09685) |
| FiLM | [Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018](https://arxiv.org/abs/1709.07871) |
| RoSA | [Nikdan et al., "RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation", 2024](https://arxiv.org/abs/2401.04679) |
| Linear probes | [Alain & Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes", ICLR Workshop 2017](https://arxiv.org/abs/1610.01644) |
| Intrinsic dimensionality | [Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", ACL 2021](https://arxiv.org/abs/2012.13255) |
| MAIA | [McIlroy-Young et al., "Aligning Superhuman AI with Human Behavior: Chess as a Model System", KDD 2020](https://arxiv.org/abs/2006.01855) |
| AlphaZero | [Silver et al., "A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play", Science 2018](https://arxiv.org/abs/1712.01815) |
| Leela Chess Zero | [github.com/LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) |
| shakmaty | [github.com/niklasf/shakmaty](https://github.com/niklasf/shakmaty) |
| PyO3 | [github.com/PyO3/pyo3](https://github.com/PyO3/pyo3) |
| Lichess | [lichess.org](https://lichess.org/) / [database.lichess.org](https://database.lichess.org/) |

## Citation

```bibtex
@software{schweich2026pawn,
  author = {Schweich, Thomas},
  title = {{PAWN}: Playstyle-Agnostic World-model Network for Chess},
  year = {2026},
  url = {https://github.com/thomas-schweich/PAWN},
  license = {Apache-2.0}
}
```

## License

Apache 2.0. See [LICENSE](https://github.com/thomas-schweich/PAWN/blob/main/LICENSE).
