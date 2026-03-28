# PAWN: Playstyle-Agnostic World-model Network for Chess

A small causal transformer trained on random chess games that learns legal moves, board state representations, and game dynamics purely from random legal move sequences absent any form of strategic play.

I've found PAWN to be a viable testbed for finetuning and augmentation methods at small scale. Since it is entirely unopinionated, it's a blank slate ready to be adapted, augmented, and finetuned into arbitrary player models with unique playstyles.

Finetuning PAWN has proven significantly more parameter-efficient than training new models from scratch and requires minimal compute resources.

Feel free to use PAWN in your own experiments. Note that PAWN was developed as a personal project by a single developer and has not been published or audited. If you spot a bug, please help out by creating an issue or PR.

**PAWN is under active development and is not yet stable.**

## Model Variants

Three sizes, trained for 100K steps on random games (~25.6M games each):

| Variant | d_model | Layers | Heads | Params | Top-1 | Legal Rate | Download |
|---------|---------|--------|-------|--------|-------|------------|----------|
| **PAWN-Small** | 256 | 8 | 4 | ~9.5M | 6.73% | 99.29% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-small) |
| **PAWN (Base)** | 512 | 8 | 8 | ~35.8M | 6.86% | 99.97% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-base) |
| **PAWN-Large** | 640 | 10 | 8 | ~68.4M | 6.94% | 99.98% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-large) |

All variants share the same architecture: [RMSNorm](https://arxiv.org/abs/1910.07467), [SwiGLU](https://arxiv.org/abs/2002.05202) FFN, [RoPE](https://arxiv.org/abs/2104.09864), factored move embeddings, and a 4278-token vocabulary covering:

- all possible (src, dst) pairs for an 8x8 grid (the chess board),
- promotion moves: 4 piece types (queen, bishop, rook, knight) x 44 eligible (source square, destination square) pairs for pawns reaching the 1st & 8th ranks,
- a token for each game outcome (`WHITE_CHECKMATES`, `BLACK_CHECKMATES`, `STALEMATE`, `DRAW_BY_RULE`, `PLY_LIMIT`),
- and a padding token.

Notably, the vocabulary includes impossible moves like `a1a1` and `b1a5`. PAWN naturally learns to avoid these since they don't appear in its training examples.

Conceptually, each token is best thought of as a move in UCI notation -- they are effectively coordinates. They do not include any information on the type of piece, side to play, or any direct geometric or board state information other than the factored nature of the embeddings.

For example, `e2e4` is the token that represents the king's pawn opening, but only when it's the first ply in the sequence (moving a rook from e2 to e4 in the late game would use the same token). The model learns to track which type of piece is on each square at any given moment entirely of its own accord.

For that matter, it isn't told what piece types exist, what movement patterns they follow, or indeed the concept of a piece. All of that understanding comes purely from observation and can be isolated via [linear probes](https://arxiv.org/abs/1610.01644) (Alain & Bengio, 2016).

## Quickstart

```bash
# Clone and build
git clone https://github.com/thomas-schweich/PAWN.git && cd PAWN

# Build the Rust chess engine (required -- handles all game logic)
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python dependencies
uv sync --extra cu128   # NVIDIA GPU (or --extra rocm for AMD)
```

### Train an adapter

Weights and data load directly from HuggingFace -- no submodules or local files needed:

```bash
uv run python scripts/train_bottleneck.py \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full \
    --bottleneck-dim 32 --lr 1e-4 --local-checkpoints
```

### Pretrain from scratch

Random games are generated on-the-fly; no dataset required:

```bash
uv run python scripts/train.py --variant base --local-checkpoints

# Or train all three variants simultaneously on shared data
uv run python scripts/train_all.py --local-checkpoints
```

### Run probes and diagnostics

```bash
uv run python scripts/eval_probes.py --log-dir logs --device cuda
uv run python -m pawn.dashboard --log-dir logs  # real-time monitoring
```

## Datasets

These datasets are for **adapter training (behavioral cloning)**, not for pretraining PAWN itself. PAWN is pretrained exclusively on random legal games generated on-the-fly -- it never sees human or engine games during pretraining. The datasets below provide real gameplay data for finetuning the frozen PAWN backbone into player models that mimic specific playstyles or skill levels.

| Dataset | Games | Description | Link |
|---------|-------|-------------|------|
| Lichess Full | ~289M train + 50K val + 50K test | Rated games from Q1 2025 (all Elos), holdout from Jan 2026 | [pawn-lichess-full](https://huggingface.co/datasets/thomas-schweich/pawn-lichess-full) |
| Stockfish nodes=1 | 900K train + 50K val + 50K test | NNUE self-play, 1 node/move | [stockfish-nodes1](https://huggingface.co/datasets/thomas-schweich/stockfish-nodes1) |

All datasets use the PAWN token format: pre-tokenized `list[int16]` move sequences, ready for training without any parsing. The Lichess dataset also includes clock annotations, Stockfish eval annotations (~8% of games), player hashes, Elo ratings, and game metadata.

Datasets load directly from HuggingFace via Polars lazy scan -- predicate pushdown on columns like `white_elo` and `date` lets you efficiently filter to specific Elo bands or time periods without downloading the full dataset.

## Architecture

<sub>More info: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)</sub>

PAWN is a standard decoder-only [transformer](https://arxiv.org/abs/1706.03762) trained with next-token prediction on chess move sequences. Each training example is:

```
[outcome] [ply_1] [ply_2] ... [ply_N] [PAD] ... [PAD]
```

Ply tokens use a factored embedding: each move is decomposed into source square + destination square + promotion piece, with embeddings summed. This gives the model explicit spatial structure while keeping the vocabulary compact. The context window of all variants is 256 tokens.

The model's predictions are not masked to legal moves during training; it has to determine what moves are currently legal based on the sequence of moves so far.

No attempt is made to provide the model with information about other pieces. In other words, it only thinks in moves. There is no equivalent of the multi-plane 8x8xN board representation used by e.g. [AlphaZero](https://arxiv.org/abs/1712.01815) (Silver et al., 2018) and [Lc0](https://github.com/LeelaChessZero/lc0). Any and all state representation and geometry is learned by the model internally.

## What the Model Learns

Despite training exclusively on random games, PAWN develops rich internal representations. Linear probes on the base model's hidden states decode:

| Probe | Accuracy |
|-------|----------|
| Side to move | 100.0% |
| En passant square | 99.7% |
| Castling rights | 96.6% |
| Game phase | 90.7% |
| Piece type at square | 89.7% |
| Is check | 94.2% |
| Material count (MAE) | 6.1 |

The model also achieves >99.9% legal move rate on the base and large variants, correctly identifying legal moves from move history alone.

The [theoretical accuracy ceiling](docs/ACCURACY_CEILING.md) for random game prediction is 6.43% (unconditional) to 7.92% (MCTS-conditioned on outcome). All three models exceed the unconditional ceiling, confirming they learn structure beyond move legality.

## Adapter Methods

<sub>More info: [docs/ADAPTERS.md](docs/ADAPTERS.md)</sub>

PAWN ships with six adapter implementations for fine-tuning the frozen backbone on human game data:

| Method | Params (typical) | Accuracy (1800 Elo) | Description |
|--------|-----------------|---------------------|-------------|
| **[Bottleneck](https://arxiv.org/abs/1902.00751)** | 131K | 41.7% | Houlsby-style residual MLP adapters |
| **[RoSA](https://arxiv.org/abs/2401.04679)** | configurable | -- | Gradient-informed sparse + LoRA |
| **Sparse** | 503K-2.7M | 40.2-44.7% | Random binary mask on frozen weights |
| **[LoRA](https://arxiv.org/abs/2106.09685)** | ~65K | 34.1% | Low-rank attention projection adapters |
| **Hybrid** | ~65K | 34.1% | LoRA + FiLM combined |
| **[FiLM](https://arxiv.org/abs/1709.07871)** | ~17K | 30.3% | Per-channel affine modulation |

A 524K bottleneck adapter achieves 42.2% accuracy predicting moves by 1800-rated Lichess players, vs. 30.9% for a standalone model with the same architecture and parameter count -- an ~11 percentage point "free" accuracy lift from the frozen backbone.

## Repository Structure

```
pawn/
├── pawn/                 # Core Python package
│   ├── config.py         # Model configs (small/base/large)
│   ├── model.py          # PAWN transformer
│   ├── data.py           # Random game data pipeline
│   ├── lichess_data.py   # Lichess/Parquet data pipeline
│   ├── trainer.py        # Pretraining loop
│   ├── gpu.py            # GPU auto-detection
│   ├── adapters/         # Bottleneck, LoRA, FiLM, sparse, hybrid, RoSA
│   ├── eval_suite/       # Probes, generation tests, diagnostics
│   └── dashboard/        # Solara training dashboard
├── engine/               # Rust chess engine (PyO3 bindings via shakmaty)
├── scripts/              # Training, evaluation, and data extraction
├── deploy/               # Docker, RunPod deployment, serverless handler
├── tests/                # Unit tests
└── docs/                 # Architecture, training, adapter docs
```

## Chess Engine

PAWN includes a bundled Rust chess engine (`engine/`) that handles all game simulation, move generation, legal move computation, tokenization, and PGN parsing. The engine uses [`shakmaty`](https://github.com/niklasf/shakmaty) under the hood, with [PyO3](https://github.com/PyO3/pyo3) bindings to Python. No Python chess libraries are used.

The engine generates training data on-the-fly via `chess_engine.generate_random_games()`, producing well over 100 million random games per hour. It also includes enriched PGN parsing (extracting clock annotations, Stockfish evals, and headers in a single pass) and UCI engine self-play generation.

## More info

- [Architecture](docs/ARCHITECTURE.md) -- model design, embeddings, training objective
- [Training](docs/TRAINING.md) -- pretraining, adapter training, deployment
- [Adapters](docs/ADAPTERS.md) -- adapter methods, results, quick start
- [Accuracy Ceiling](docs/ACCURACY_CEILING.md) -- theoretical limits for random game prediction

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
  year = 2026,
  url = {https://github.com/thomas-schweich/PAWN},
  license = {Apache-2.0}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
