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
  - parameter-efficient-finetuning
  - pytorch
  - rust
language:
  - en
---

# PAWN: Playstyle-Agnostic World-model Network for Chess

PAWN is a small causal transformer trained on random chess games. It learns legal moves, board state representations, and game dynamics purely from random legal move sequences -- no strategic play, no hand-crafted features, no external game databases.

PAWN is designed as a testbed for finetuning and augmentation methods at small scale. Because the pretrained model is entirely unopinionated (trained only on uniformly random legal moves), it serves as a blank slate that can be adapted, augmented, and finetuned into arbitrary player models with unique playstyles.

Finetuning PAWN has proven significantly more parameter-efficient than training new models from scratch and requires minimal compute resources.

**[GitHub Repository](https://github.com/thomas-schweich/PAWN)**

## Model Variants

| Variant | d_model | Layers | Heads | Parameters | Link |
|---------|---------|--------|-------|------------|------|
| PAWN-Small | 256 | 8 | 4 | ~9.5M | [thomas-schweich/pawn-small](https://huggingface.co/thomas-schweich/pawn-small) |
| PAWN (Base) | 512 | 8 | 8 | ~35.8M | [thomas-schweich/pawn-base](https://huggingface.co/thomas-schweich/pawn-base) |
| PAWN-Large | 640 | 10 | 8 | ~68.4M | [thomas-schweich/pawn-large](https://huggingface.co/thomas-schweich/pawn-large) |

All variants share the same architecture (RMSNorm, SwiGLU, RoPE, factored move embeddings) and vocabulary (4,278 tokens). They differ only in width, depth, and head count.

## Quickstart

```bash
# Clone and build
git clone https://github.com/thomas-schweich/PAWN.git && cd PAWN

# Build the Rust chess engine (required -- handles all game logic)
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python dependencies
uv sync --extra cu128   # NVIDIA (or --extra rocm for AMD)

# Pull a pretrained checkpoint
git submodule update --init checkpoints/pawn-base
```

### Load and generate moves

```python
import torch
from safetensors.torch import load_file
from pawn.config import CLMConfig, WHITE_CHECKMATES
from pawn.model import PAWNCLM

# Load the model
cfg = CLMConfig.base()
model = PAWNCLM(cfg).cuda().eval()
weights = load_file("checkpoints/pawn-base/model.safetensors", device="cuda")
model.load_state_dict(weights)

# Condition on outcome and generate a game
input_ids = torch.tensor([[WHITE_CHECKMATES]], device="cuda")
pad_mask = torch.ones(1, 1, dtype=torch.bool, device="cuda")

logits, _ = model.forward_generate(input_ids, pad_mask)
next_token = logits[0, -1].argmax()
```

### Train an adapter

```bash
uv sync --extra dev
git submodule update --init checkpoints/pawn-base

uv run python scripts/train_bottleneck.py \
    --checkpoint checkpoints/pawn-base \
    --pgn data/lichess_1800_1900.pgn \
    --bottleneck-dim 32 --lr 1e-4 --local-checkpoints
```

## Architecture

PAWN is a decoder-only transformer trained with next-token prediction on chess move sequences. Each sequence has the format:

```
[outcome] [ply_1] [ply_2] ... [ply_N] [PAD] ... [PAD]
```

The token vocabulary covers all possible source-destination square pairs on the 8x8 board (4,096 grid moves), promotion moves (176 tokens for 4 piece types across 44 eligible square pairs), 5 outcome tokens, and 1 padding token.

Move embeddings are factored: each move token is decomposed into source square + destination square + promotion piece, with embeddings summed. This provides structural inductive bias (moves sharing a source or destination share embedding components) while reducing embedding parameters by roughly 32x.

The model uses pre-norm RMSNorm, SwiGLU feed-forward layers (4x expansion), Rotary Position Embeddings (RoPE), and a 256-token context window. All chess logic -- game simulation, move generation, tokenization, and legal move computation -- is handled by a bundled Rust engine built on [shakmaty](https://github.com/niklasf/shakmaty).

For full architectural details, see [docs/ARCHITECTURE.md](https://github.com/thomas-schweich/PAWN/blob/main/docs/ARCHITECTURE.md).

## What the Model Learns

Despite training exclusively on random games, PAWN develops rich internal representations:

- **Legal move prediction**: The model achieves over 98% legal move rate, accurately predicting which moves are legal from move history alone.
- **Board state tracking**: Linear probes on hidden states decode piece positions, check status, castling rights, material counts, and game phase with high accuracy -- even though the model never sees explicit board representations.

These properties make PAWN useful as a frozen backbone for downstream tasks. See the [adapter documentation](https://github.com/thomas-schweich/PAWN/blob/main/docs/ADAPTERS.md) for fine-tuning results.

## Adapter Methods

PAWN ships with five adapter implementations for fine-tuning the frozen backbone on human game data:

| Method | Parameters | Description |
|--------|-----------|-------------|
| Bottleneck | ~131K | Houlsby-style residual MLP adapters |
| Sparse | 503K--2.7M | Random binary mask on frozen weights |
| LoRA | ~65K | Low-rank attention projection adapters |
| Hybrid | ~65K | LoRA + FiLM combined |
| FiLM | ~17K | Per-channel affine modulation |

## Citation

```bibtex
@software{schweich2025pawn,
  author = {Schweich, Thomas},
  title = {{PAWN}: Playstyle-Agnostic World-model Network for Chess},
  year = {2025},
  url = {https://github.com/thomas-schweich/PAWN},
  license = {Apache-2.0}
}
```

## License

Apache 2.0. See [LICENSE](https://github.com/thomas-schweich/PAWN/blob/main/LICENSE).
