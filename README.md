# PAWN: Playstyle-Agnostic World-model Network for Chess

A small causal transformer trained on random chess games that learns legal moves, board state representations, and game dynamics purely from random legal move sequences absent any form of strategic play.

I've found PAWN to be a viable testbed for finetuning and augmentation methods at small scale. Since it is entirely unopinionated, it's a blank slate ready to be adapted, augmented, and finetuned into arbitrary player models with unique playstyles.

Finetuning PAWN has proven significantly more parameter-efficient than training new models from scratch and requires minimal compute resources.

Feel free to use PAWN in your own experiments. Note that PAWN was developed as a personal project by a single developer and has not been published or audited. If you spot a bug, please help out by creating an issue or PR.

## Model Variants

To aid in exploring how model size affects different finetuning methods, we trained three versions of PAWN:

| Variant | d_model | Layers | Heads | Params | Download |
|---------|---------|--------|-------|--------|----------|
| **PAWN** | 512 | 8 | 8 | ~35.8M | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-base) |
| **PAWN-Small** | 256 | 8 | 4 | ~9.5M | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-small) |
| **PAWN-Large** | 640 | 10 | 8 | ~68.4M | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-large) |


All variants share the same architecture: RMSNorm, SwiGLU FFN, RoPE, factored move embeddings, and a 4278-token vocabulary covering:

- all possible (src, dst) pairs for an 8x8 grid (the chess board),
- promotion moves (one per promotion piece type per square on 1st or 8th rank), 
- a token for each game outcome (`WHITE_CHECKMATE`, `BLACK_CHECKMATE`, `STALEMATE`, `DRAW_BY_RULE`, `PLY_LIMIT`),
- and a padding token.

Notably, the vocabulary includes impossible moves like `a1a1` and `b1a5`. PAWN naturally learns to avoid these since they don't appear in its training examples.

Conceptually, each token is best thought of as a move in UCI notation--they are effectively coordinates. They do not include any information on the type of peice, side to play, or any direct geometric or board state information other than the factored nature of the embeddings (see the architecture section below for details). 

For example, `e2e4` is the token that represents the king's pawn opening, but only when it's the first ply in the sequence (moving a rook between from e2 to e4 in the late game would use the same token). The model learns to track which type of peice is on each square any given moment entirely of its own accord. For that matter, it isn't even told what piece types exist and what movement patterns they follow, or indeed even the concept of a peice. All of that 'understanding' comes purely from observation.

## Quickstart

```bash
# Clone and build
git clone https://github.com/<user>/pawn.git && cd pawn

# Build the Rust chess engine
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install core dependencies
uv sync --extra cu128   # NVIDIA GPU (or --extra rocm for AMD)

# Install dependencies for running tests, performing analysis on the results, and running the training monitoring dashboard (optional but recommended) 
uv sync --extra dev --extra eval --extra dashboard

# Train an adapter on a pre-trained checkpoint
uv run python scripts/train_bottleneck.py \
    --checkpoint checkpoints/pawn-base.pt \
    --pgn data/lichess_1800_1900.pgn \
    --bottleneck-dim 32 --lr 1e-4

# Or pretrain a PAWN variant from scratch (generates random games on-the-fly; no dataset required)
uv run python scripts/train.py --variant base

# Launch the real-time monitoring dashboard (optional dashboard dependency must be installed)
uv run python -m pawn.dashboard --log-dir logs --port 8765
```

## Architecture

<sub>Main article: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)</sub>

PAWN is a standard decoder-only transformer trained with next-token prediction on chess move sequences. Each training example is:

```
[outcome] [ply_1] [ply_2] ... [ply_N] [PAD] ... [PAD]
```

The outcome token is one of `WHITE_CHECKMATE`, `BLACK_CHECKMATE`, `STALEMATE`, `DRAW_BY_RULE`, or `PLY_LIMIT`.

Ply tokens use a factored embedding: each move is decomposed into source square + destination square + promotion piece, with embeddings summed. This gives the model some degree of explicit spatial structure while keeping the vocabulary compact.

The summed embeddings effectively represent UCI strings like `e2e4` (peice moves from `e2` to `e4`) or `f7f8q` (promotion to queen on `f8`). In factored form, the vector `e2e4` is given by `(e2xx + xxe4)`. Likewise, `f7f8q` is given by `(f7xx + xxf8 + q)`.

The context window of all variants is 256 tokens wide. Training examples all include the outcome token followed by up to 255 ply or padding tokens.

During training, simulated games are retroactively prepended with their actual outcome. During inference, the outcome token has a measurable impact on subsequent completions.

The models predictions are not masked to legal moves during training; it has to determine what moves are currently legal based on the seqeunce of moves so far.

No attempt is made to provide the model with information about other peices. In other words, it only thinks in moves. There is no equivalent of 7-dimensional manifold board representation used by e.g. Alpha Zero and Lc0. Any and all state representation is learned by the model internally.

## Adapter Methods
<sub>Main article: [docs/ADAPTERS.md](docs/ADAPTERS.md)</sub>

PAWN ships with five adapter implementations for fine-tuning the frozen backbone:

| Method | Params (typical) | Accuracy (1800 Elo) | Description |
|--------|-----------------|---------------------|-------------|
| **Bottleneck** | 131K | 41.7% | Houlsby-style residual MLP adapters |
| **Sparse** | 503K-2.7M | 40.2-44.7% | Random binary mask on frozen weights |
| **LoRA** | ~65K | 34.1% | Low-rank attention projection adapters |
| **Hybrid** | ~65K | 34.1% | LoRA + FiLM combined |
| **FiLM** | ~17K | 30.3% | Per-channel affine modulation |

Preliminary results show that a 524K bottleneck adapter on PAWN achieves 42.2% accuracy when predicting moves by 1800-level players on Lichess vs. 30.9% for a standalone model with the same architecture and parameter count. Thus the frozen backbone provides ~11 percentage points of "free" accuracy.

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

PAWN includes a bundled Rust chess engine (`engine/`) that handles all game simulation, move generation, legal move computation, and PGN parsing. The engine extensively uses `shakmaty` under the hood. No Python chess libraries are used. The engine generates training data on-the-fly via `chess_engine.generate_random_games()`, which is capable of producing well over 100 million random games per hour on a modern CPU.


## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- model design, embeddings, training objective
- [Training](docs/TRAINING.md) -- pretraining, adapter training, deployment
- [Adapters](docs/ADAPTERS.md) -- adapter methods, results, quick start

## License

Apache 2.0. See [LICENSE](LICENSE).
