---
library_name: pawn
license: apache-2.0
tags:
  - chess
  - transformer
  - world-model
  - causal-lm
  - next-token-prediction
  - representation-learning
  - pytorch
  - rust
model_name: PAWN-Base
pipeline_tag: other
citation: |
  @software{schweich2026pawn,
    author = {Schweich, Thomas},
    title = {{PAWN}: Playstyle-Agnostic World-model Network for Chess},
    year = {2026},
    url = {https://github.com/thomas-schweich/PAWN},
    license = {Apache-2.0}
  }
model_params: 34651136
d_model: 512
n_layers: 8
n_heads: 8
d_ff: 2048
context_length: 512
vocab_size: 1980
datasets:
  - random-chess-games
language:
  - en
metrics:
  - accuracy
model-index:
  - name: PAWN-Base
    results:
      - task:
          type: next-token-prediction
          name: Chess Move Prediction (Random Games)
        metrics:
          - name: Game Completion Rate
            type: accuracy
            value: 0.989746
          - name: Legal Move Rate
            type: accuracy
            value: 0.999962
          - name: Top-1 Accuracy
            type: accuracy
            value: 0.0857
          - name: Top-5 Accuracy
            type: accuracy
            value: 0.3545
          - name: Val Loss
            type: loss
            value: 2.8679
          - name: Total Training Sequences
            type: other
            value: 51200000
---

# PAWN-Base

**PAWN** (Playstyle-Agnostic World-model Network for Chess) is a causal transformer trained on random chess games. It learns legal moves, board state representations, and game dynamics purely from uniformly random legal move sequences -- no strategic play, no hand-crafted features, no external game databases.

This is the **base (default)** variant (~34.7M parameters). PAWN is designed as a frozen backbone for parameter-efficient finetuning into player models with arbitrary playstyles.

**[GitHub Repository](https://github.com/thomas-schweich/PAWN)** -- full source code, training scripts, adapter implementations, and documentation.

## All Variants

| Variant | Parameters | Link |
|---------|------------|------|
| PAWN-Small | ~9M | [thomas-schweich/pawn-small](https://huggingface.co/thomas-schweich/pawn-small) |
| PAWN (Base) | ~35M | [thomas-schweich/pawn-base](https://huggingface.co/thomas-schweich/pawn-base) |
| PAWN-Large | ~67M | [thomas-schweich/pawn-large](https://huggingface.co/thomas-schweich/pawn-large) |

A previous generation of PAWN backbones (`pawn-{small,base,large}-legacy`) used a 4,278-token coordinate vocabulary, a 256-token context window, and outcome conditioning. They are still available on HuggingFace; see [docs/LEGACY.md](https://github.com/thomas-schweich/PAWN/blob/main/docs/LEGACY.md) for the full story.

## Headline Metrics

These come from the published `model.safetensors` (step 195,000 out of 200,000 — the best 5,000-step-cadence checkpoint by val loss), measured on a fresh validation set of random games.

| Metric | Value |
|--------|-------|
| Game completion rate | 98.97% |
| Per-move legal rate | 99.9962% |
| Late-game legal rate | 100.0000% |
| Top-1 accuracy | 8.57% |
| Top-5 accuracy | 35.45% |
| Val loss | 2.868 |
| Val perplexity | 17.60 |

**Game completion rate** is the share of validation games in which *every* prediction along one side's plies was a legal move. The measurement is **non-autoregressive**: at each ply the model is shown the true ground-truth history and asked for that side's next move, and an illegal prediction at any ply forfeits the game. Errors do not corrupt subsequent positions — each prediction is independent given the true history. Autoregressive game completion has not been measured for these checkpoints and could be higher or lower; see the [game completion section of the architecture doc](https://github.com/thomas-schweich/PAWN/blob/main/docs/ARCHITECTURE.md#game-completion-rate) for the full definition. Game completion rate is a much stricter metric than per-move legal rate, and is the main signal that separates capacity between sizes.

| Compound-legality detail | Value |
|--------------------------|-------|
| Average plies completed per game | 347 |
| Average % of game completed | 99.27% |
| Median forfeit ply (when forfeit) | 101 |

### Accuracy Ratios

PAWN is trained on uniformly random chess games, so top-1 accuracy has a hard theoretical ceiling. The unconditional ratio below is computed against the **legacy** (4,278-token vocab) ceiling estimate and is kept for continuity; the v1.0.0 vocabulary changes the constant slightly, and a re-derivation is a known TODO. See the [accuracy ceiling analysis](https://github.com/thomas-schweich/PAWN/blob/main/docs/ACCURACY_CEILING.md) for methodology.

| Ceiling | Ratio |
|---------|-------|
| Unconditioned (E\[1/N_legal\] ≈ 6.43%) | 133% |



## Probe Results

Linear probes trained on frozen hidden states measure how well the model's internal representations encode board-level features. The model is never explicitly told about pieces, sides, or rules — these representations emerge purely from next-token prediction on random games.

| Probe | Accuracy | Description |
|-------|----------|-------------|
| Piece type | 89.7% | Per-square piece type (13 classes x 64 squares) |
| Side to move | 100.0% | Whose turn it is |
| Is check | 94.2% | Whether the side to move is in check |
| Castling rights | 96.6% | KQkq castling availability |
| En passant square | 99.7% | En passant target square (64 + none) |
| Material count | 86.1% (MAE 6.1) | Piece counts per type per color |
| Legal move count | 37.9% (MAE 6.8) | Number of legal moves available |
| Halfmove clock | 11.8% (MAE 4.1) | Plies since last capture or pawn move |
| Game phase | 90.7% | Opening / middlegame / endgame |




## Diagnostic Results

Edge-case diagnostics measure the model's legal move rate in specific tactical situations.

| Category | Positions | Legal Rate |
|----------|-----------|------------|
| In check | 1000 | 97.7% |
| Double check | 71 | 91.2% |
| Pin restricts movement | 1000 | 97.2% |
| En passant available | 940 | 99.2% |
| Castling legal (kingside) | 1000 | 99.7% |
| Castling legal (queenside) | 1000 | 99.6% |
| Castling blocked by check | 892 | 99.4% |
| Promotion available | 1000 | 99.4% |
| Checkmate (terminal) | 276 | 91.2% |
| Stalemate (terminal) | 41 | 84.2% |



## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only transformer |
| d_model | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Head dimension | 64 |
| d_ff | 2048 |
| Parameters | ~34.7M |
| Vocabulary | 1,980 tokens (1,968 searchless_chess actions + 1 PAD + 11 outcome tokens) |
| Context length | 512 tokens |
| Normalization | Pre-norm RMSNorm |
| FFN | SwiGLU (4x expansion) |
| Positional encoding | Rotary (RoPE, base 10000) |
| Embeddings | Factored (src + dst + promo) |
| Dropout | 0.0 |

## Training Details

| Parameter | Value |
|-----------|-------|
| Training data | On-the-fly uniformly random legal games (no external dataset) |
| Objective | Next-token cross-entropy (non-padding positions only) |
| Outcome conditioning | Disabled (prepend_outcome=False) — pure moves, no outcome leakage |
| Total steps | 200,000 |
| Batch size | 256 |
| Total training sequences | 51,200,000 (= total steps × batch size; the published checkpoint is the best 5K-cadence step by val loss, at step 195,000 ≈ 49,920,000 sequences) |
| Max ply per example | 512 |
| Learning rate | 0.0003 (cosine decay with 10,000-step warmup) |
| Optimizer | AdamW (weight decay 0.01) |
| Precision | Mixed (AMP) |

## Usage

### Loading the model

```python
import torch
from safetensors.torch import load_file
from pawn.config import CLMConfig
from pawn.model import PAWNCLM

cfg = CLMConfig.base()
model = PAWNCLM(cfg).cuda().eval()
weights = load_file("model.safetensors", device="cuda")
model.load_state_dict(weights)
```

Or load directly from HuggingFace:

```python
from pawn.checkpoint import load_backbone_weights
from pawn.config import CLMConfig
from pawn.model import PAWNCLM

weights, config = load_backbone_weights("thomas-schweich/pawn-base")
cfg = CLMConfig.base()
model = PAWNCLM(cfg).eval()
model.load_state_dict(weights)
```

### Finetuning with an adapter

```bash
uv run python scripts/train.py --run-type adapter --strategy bottleneck \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full \
    --bottleneck-dim 32 --lr 1e-4 --local-checkpoints
```

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
| Searchless Chess (action vocab) | [Ruoss et al., "Amortized Planning with Large-Scale Transformers: A Case Study on Chess", 2024](https://arxiv.org/abs/2402.04494) |
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
