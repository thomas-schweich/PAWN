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
model_name: PAWN-Small
pipeline_tag: other
citation: |
  @software{schweich2026pawn,
    author = {Schweich, Thomas},
    title = {{PAWN}: Playstyle-Agnostic World-model Network for Chess},
    year = {2026},
    url = {https://github.com/thomas-schweich/PAWN},
    license = {Apache-2.0}
  }
model_params: 9523712
d_model: 256
n_layers: 8
n_heads: 4
d_ff: 1024
context_length: 256
vocab_size: 4284
datasets:
  - random-chess-games
language:
  - en
metrics:
  - accuracy
model-index:
  - name: PAWN-Small
    results:
      - task:
          type: next-token-prediction
          name: Chess Move Prediction (Random Games)
        metrics:

          - name: Top-1 Accuracy
            type: accuracy
            value: 0.0668

          - name: Val Loss
            type: loss
            value: 3.1606
          - name: Games Seen
            type: other
            value: 25600000
---

# PAWN-Small

**PAWN** (Playstyle-Agnostic World-model Network for Chess) is a causal transformer trained on random chess games. It learns legal moves, board state representations, and game dynamics purely from uniformly random legal move sequences -- no strategic play, no hand-crafted features, no external game databases.

This is the **small** variant (~9.5M parameters). PAWN is designed as a frozen backbone for parameter-efficient finetuning into player models with arbitrary playstyles.

**[GitHub Repository](https://github.com/thomas-schweich/PAWN)** -- full source code, training scripts, adapter implementations, and documentation.

## All Variants

| Variant | Parameters | Link |
|---------|------------|------|
| PAWN-Small | ~9.5M | [thomas-schweich/pawn-small](https://huggingface.co/thomas-schweich/pawn-small) |
| PAWN (Base) | ~35.8M | [thomas-schweich/pawn-base](https://huggingface.co/thomas-schweich/pawn-base) |
| PAWN-Large | ~68.4M | [thomas-schweich/pawn-large](https://huggingface.co/thomas-schweich/pawn-large) |

## Headline Metrics

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 6.68% |
| Val loss | 3.161 |

### Accuracy Ratios

PAWN is trained on uniformly random chess games, so top-1 accuracy has a hard theoretical ceiling. Ratios above 100% on the unconditioned ceiling indicate the model has learned structure beyond simply identifying legal moves. See [Accuracy Ceiling Analysis](https://github.com/thomas-schweich/PAWN/blob/main/docs/ACCURACY_CEILING.md).

| Ceiling | Ratio |
|---------|-------|
| Unconditioned (E\[1/N_legal\] = 6.43%) | 104% |
| Naive-conditioned (1-ply filter = 6.44%) | 104% |
| Bayes-optimal conditioned (MCTS, 32 rollouts = 7.92%) | 84% |


## Probe Results

Linear probes trained on frozen hidden states measure how well the model's internal representations encode board-level features.

| Probe | Accuracy | Description |
|-------|----------|-------------|
| Piece type | 89.1% | Per-square piece type (13 classes x 64 squares) |
| Side to move | 100.0% | Whose turn it is |
| Is check | 94.3% | Whether the side to move is in check |
| Castling rights | 96.5% | KQkq castling availability |
| En passant square | 99.8% | En passant target square (64 + none) |
| Material count | 86.5% (MAE 4.9) | Piece counts per type per color |
| Legal move count | 30.7% (MAE 7.4) | Number of legal moves available |
| Halfmove clock | 13.3% (MAE 3.9) | Plies since last capture or pawn move |
| Game phase | 91.1% | Opening / middlegame / endgame |




## Diagnostic Results

Edge-case diagnostics measure the model's legal move rate in specific tactical situations.

| Category | Positions | Legal Rate |
|----------|-----------|------------|
| In check | 1000 | 82.4% |
| Double check | 71 | 65.1% |
| Pin restricts movement | 1000 | 86.2% |
| En passant available | 940 | 97.1% |
| Castling legal (kingside) | 1000 | 98.8% |
| Castling legal (queenside) | 1000 | 98.2% |
| Castling blocked by check | 892 | 95.7% |
| Promotion available | 1000 | 96.2% |
| Checkmate (terminal) | 276 | 66.4% |
| Stalemate (terminal) | 41 | 53.8% |



## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only transformer |
| d_model | 256 |
| Layers | 8 |
| Attention heads | 4 |
| Head dimension | 64 |
| d_ff | 1024 |
| Parameters | ~9.5M |
| Vocabulary | 4,284 tokens |
| Context length | 256 tokens |
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
| Total steps | 100,000 |
| Batch size | 256 |
| Games seen | 25,600,000 |
| Learning rate | 3e-4 (cosine decay with 1,000-step warmup) |
| Optimizer | AdamW (weight decay 0.01) |
| Precision | Mixed (AMP) |
| Hardware | NVIDIA H200 |

## Usage

### Loading the model

```python
import torch
from safetensors.torch import load_file
from pawn.config import CLMConfig
from pawn.model import PAWNCLM

cfg = CLMConfig.small()
model = PAWNCLM(cfg).cuda().eval()
weights = load_file("model.safetensors", device="cuda")
model.load_state_dict(weights)
```

Or load directly from HuggingFace:

```python
from pawn.checkpoint import load_backbone_weights
from pawn.config import CLMConfig
from pawn.model import PAWNCLM

weights, config = load_backbone_weights("thomas-schweich/pawn-small")
cfg = CLMConfig.small()
model = PAWNCLM(cfg).eval()
model.load_state_dict(weights)
```

### Finetuning with an adapter

```bash
uv run python scripts/train_bottleneck.py \
    --checkpoint thomas-schweich/pawn-small \
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
