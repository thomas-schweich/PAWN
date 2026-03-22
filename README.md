# PAWN: Playstyle-Agnostic World-model Network for Chess

A small causal transformer trained on random chess games that learns legal moves, board state representations, and game dynamics purely from random legal move sequences absent any form of strategic play.

I've found PAWN to be a viable testbed for finetuning and augmentation methods at small scale. Since it is entirely unopinionated, it's a blank slate ready to be adapted, augmented, and finetuned into arbitrary player models with unique playstyles.

Finetuning PAWN has proven significantly more parameter-efficient than training new models from scratch and requires minimal compute resources.

Feel free to use PAWN in your own experiments. Note that PAWN was developed as a personal project by a single developer and has not been published or audited. If you spot a bug, please help out by creating an issue or PR.

## Model Variants

To aid in exploring how model size affects different finetuning methods, we trained three versions of PAWN:

| Variant | d_model | Layers | Heads | Params | Download |
|---------|---------|--------|-------|--------|----------|
| **PAWN** | 512 | 8 | 8 | ~35.8M | [pawn-base.pt]() |
| **PAWN-Small** | 256 | 8 | 4 | ~9.5M | [pawn-small.pt]() |
| **PAWN-Large** | 640 | 10 | 8 | ~68.4M | [pawn-large.pt]() |

All variants share the same architecture: RMSNorm, SwiGLU FFN, RoPE, factored move embeddings, and a 4278-token vocabulary covering all possible (src, dst) pairs for an 8x8 chess board plus promotions (one per promotion piece type per square on 1st or 8th rank), along with a token for each game outcome (white wins, black wins, stalemate, draw, ply limit) and a padding token. Notably, the vocabulary includes impossible moves like `(a1, a1)`. PAWN learns to avoid these as part of its understanding of legality.

## Quickstart

```bash
# Clone and build
git clone https://github.com/<user>/pawn.git && cd pawn
cd engine && uv run --with maturin maturin develop --release && cd ..
uv sync --extra cu128   # NVIDIA GPU (or --extra rocm for AMD)

# Train an adapter on a pre-trained checkpoint
uv run python scripts/train_bottleneck.py \
    --checkpoint checkpoints/pawn-base.pt \
    --pgn data/lichess_1800_1900.pgn \
    --bottleneck-dim 32 --lr 1e-4

# Or pretrain from scratch (generates random games on-the-fly)
uv run python scripts/train.py --variant base
```

## Architecture

PAWN is a standard decoder-only transformer trained with next-token prediction on chess move sequences. Each training example is:

```
[outcome] [ply_1] [ply_2] ... [ply_N] [PAD] ... [PAD]
```

The outcome token (white wins, black wins, stalemate, draw, ply limit) tells the model how the game ends.

Ply tokens use a factored embedding: each move is decomposed into source square + destination square + promotion piece, with embeddings summed. This gives the model explicit spatial structure while keeping the vocabulary compact.

The context window of all variants is 256 tokens wide. Training examples all include the outcome token followed by up to 255 ply or padding tokens.

During training, examples are retroactively prepended with their actual outcome. During inference, the outcome token has a measurable impact on subsequent completions.


## Adapter Methods

PAWN ships with five adapter implementations for fine-tuning the frozen backbone:

| Method | Params (typical) | Accuracy (1800 Elo) | Description |
|--------|-----------------|---------------------|-------------|
| **Bottleneck** | 131K | 41.7% | Houlsby-style residual MLP adapters |
| **Sparse** | 503K-2.7M | 40.2-44.7% | Random binary mask on frozen weights |
| **LoRA** | ~65K | 34.1% | Low-rank attention projection adapters |
| **Hybrid** | ~65K | 34.1% | LoRA + FiLM combined |
| **FiLM** | ~17K | 30.3% | Per-channel affine modulation |

A 524K bottleneck adapter on PAWN achieves 42.2% accuracy, vs. 30.9% for a standalone model with the same parameter count. The frozen backbone provides ~11 percentage points of "free" accuracy.

See [docs/ADAPTERS.md](docs/ADAPTERS.md) for detailed comparisons and training instructions.

## Repository Structure

```
pawn/
├── pawn/                 # Core Python package
│   ├── config.py         # Model configs (small/base/large)
│   ├── model.py          # PAWN transformer
│   ├── data.py           # Random game data pipeline
│   ├── lichess_data.py   # Lichess PGN data pipeline
│   ├── trainer.py        # Pretraining loop
│   ├── gpu.py            # GPU auto-detection
│   ├── adapters/         # Bottleneck, LoRA, FiLM, sparse, hybrid
│   └── eval_suite/       # Probes, generation tests, diagnostics
├── engine/               # Rust chess engine (PyO3 bindings)
├── scripts/              # Training and evaluation scripts
├── deploy/               # Runpod deployment scripts
├── tests/                # Unit tests
└── docs/                 # Architecture, training, adapter docs
```

## Chess Engine

PAWN includes a bundled Rust chess engine (`engine/`) that handles all game simulation, move generation, legal move computation, and PGN parsing via `shakmaty`. No Python chess libraries are used. The engine generates training data on-the-fly via `chess_engine.generate_random_games()`, which is capable of producing well over 100 million random games per hour on a modern CPU.


## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- model design, embeddings, training objective
- [Training](docs/TRAINING.md) -- pretraining, adapter training, deployment
- [Adapters](docs/ADAPTERS.md) -- adapter methods, results, quick start

## License

Apache 2.0. See [LICENSE](LICENSE).
