# PAWN: Playstyle-Agnostic World-model Network for Chess

A small causal transformer trained on random chess games that learns legal moves, board state representations, and game dynamics purely from random legal move sequences absent any form of strategic play.

I've found PAWN to be a viable testbed for finetuning and augmentation methods at small scale. Since it is entirely unopinionated, it's a blank slate ready to be adapted, augmented, and finetuned into arbitrary player models with unique playstyles.

Finetuning PAWN has proven significantly more parameter-efficient than training new models from scratch and requires minimal compute resources.

Feel free to use PAWN in your own experiments. PAWN is developed as a personal project by a single developer and his imaginary friend (Claude) and has not been published or audited — treat the numbers below as a single-developer snapshot rather than peer-reviewed results. If you spot a bug or inaccuracy, please help out by creating an issue or PR.


## Model Variants

Three sizes, all trained from scratch on random chess games generated on-the-fly by a Rust engine. The published v1.0.0 weights were trained together for 200K steps at batch size 256 on a single B200 — all three variants see the same random-game batches each step, with one forward/backward pass per variant in sequence on the same GPU (see [cotrain config](configs/cotrain_three_variants.json)). The numbers below come from the best 5K-cadence checkpoint by val loss (step 195,000 ≈ 49.9M sequences) for all three variants:

| Variant | d_model | Layers | Heads | Params | Top-1 | Legal rate | Game completion | Download |
|---------|---------|--------|-------|--------|-------|------------|-----------------|----------|
| **PAWN-Small** | 256 | 8 | 4 | 8.94M | 8.54% | 99.7451% | 52.34% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-small) |
| **PAWN (Base)** | 512 | 8 | 8 | 34.65M | 8.57% | 99.9962% | 98.97% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-base) |
| **PAWN-Large** | 640 | 10 | 8 | 66.91M | 8.63% | 99.9990% | 99.76% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-large) |

*Metrics measured on a 2,048-game validation set of fresh random games. **Game completion** is non-autoregressive: each ply sees the true history, so predictions are independent. It is the main signal that separates capacity between sizes — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#game-completion-rate).*

All variants share the same architecture: [RMSNorm](https://arxiv.org/abs/1910.07467), [SwiGLU](https://arxiv.org/abs/2002.05202) FFN, [RoPE](https://arxiv.org/abs/2104.09864), factored move embeddings, and a vocabulary covering:

- 1,968 move actions (the searchless_chess vocabulary, one entry per legally-reachable (src, dst[, promotion]) tuple),
- 11 game-outcome tokens (pretraining outcomes: `WHITE_CHECKMATES`, `BLACK_CHECKMATES`, `STALEMATE`, `DRAW_BY_RULE`, `PLY_LIMIT`; Lichess-specific outcomes: `WHITE_RESIGNS`, `BLACK_RESIGNS`, `DRAW_BY_AGREEMENT`, `WHITE_WINS_ON_TIME`, `BLACK_WINS_ON_TIME`, `DRAW_BY_TIME`),
- and a single PAD token — 1,980 tokens total.

PAWN learns to avoid impossible moves like `a1a1` and `b1a5` since they don't appear in its training examples (and now don't appear in the action vocabulary either).

> [!note]
> A previous generation of PAWN backbones (`pawn-{small,base,large}-legacy`) used a 4,278-token coordinate vocabulary, a 256-token context window, and outcome conditioning. They are still available on HuggingFace, and the `pre-vocab-transition` git tag marks the last commit before the vocabulary transition — it will load and work with the old checkpoints. See [docs/LEGACY.md](docs/LEGACY.md) for background on what changed and why.

Tokens are coordinate pairs (UCI notation) with no piece type or side-to-move information — `e2e4` means the same token whether it's a pawn double-push or a rook move. The model learns to track piece placement, movement rules, and game state entirely from observation, which can be isolated via [linear probes](https://arxiv.org/abs/1610.01644) (Alain & Bengio, 2016).

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

Weights and data can be loaded directly from HuggingFace:

```bash
uv run python scripts/train.py --run-type adapter --strategy bottleneck \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full \
    --bottleneck-dim 32 --lr 1e-4 --local-checkpoints
```

### Pretrain from scratch

Random games are generated on-the-fly; no dataset required:

```bash
uv run python scripts/train.py --variant base --local-checkpoints

# Or train all three variants simultaneously on shared data
uv run python scripts/train.py --config configs/cotrain_three_variants.json
```

### Run probes and diagnostics

```bash
uv run python scripts/eval_probes.py --log-dir logs --device cuda
uv run python -m pawn.dashboard --log-dir logs  # real-time monitoring
```

## Datasets

For adapter training (behavioral cloning), not pretraining. PAWN is pretrained on random games generated on-the-fly — it never sees human or engine games.

| Dataset | Games | Description | Link |
|---------|-------|-------------|------|
| Lichess Full | 286M train + 9.3M val + 9.0M test | Rated games from Q1 2025 (all Elos), holdout from Jan 2026 | [pawn-lichess-full](https://huggingface.co/datasets/thomas-schweich/pawn-lichess-full) |
| Stockfish nodes=1 | 900K train + 50K val + 50K test | NNUE self-play, 1 node/move | [stockfish-nodes1](https://huggingface.co/datasets/thomas-schweich/stockfish-nodes1) |

All datasets use pre-tokenized `list[int16]` move sequences (`tokens` column). The Lichess dataset also includes raw `san`/`uci` strings, clock annotations, Elo ratings, and full game metadata. Datasets load directly from HuggingFace via Polars lazy scan with predicate pushdown.

## Architecture

<sub>More info: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)</sub>

Standard decoder-only [transformer](https://arxiv.org/abs/1706.03762) with next-token prediction. Each training example is a move sequence padded to 512 tokens. Factored embeddings decompose each move into source square + destination square + promotion piece. Predictions are not masked to legal moves — the model must infer legality from the move history alone. There is no board representation like [AlphaZero](https://arxiv.org/abs/1712.01815)'s 8x8xN planes; all state tracking is learned internally.

## What the Model Learns

Despite training exclusively on random games, PAWN develops rich internal representations. Linear probes on frozen hidden states decode chess concepts the model is never explicitly told about:

| Probe | Small | Base | Large |
|-------|-------|------|-------|
| Side to move | 100.0% | 100.0% | 100.0% |
| En passant square | 99.8% | 99.7% | 99.7% |
| Castling rights | 96.5% | 96.6% | 96.8% |
| Is check | 94.3% | 94.2% | 93.9% |
| Game phase | 91.1% | 90.7% | 91.3% |
| Piece type | 89.1% | 89.7% | 90.3% |
| Material count | 86.5% | 86.1% | 86.9% |

Full probe results including diagnostics are on each variant's [HuggingFace model card](https://huggingface.co/thomas-schweich/pawn-base).

All three variants sit at ~101–102% of the [theoretical top-1 ceiling](docs/ACCURACY_CEILING.md) (`E[1/N_legal] = 8.43%`, 95% CI [8.41%, 8.45%]), essentially saturating the metric. **Game completion rate is the only headline metric that meaningfully separates sizes** — see the [model variant table](#model-variants) above.

## Adapter Methods

<sub>More info: [docs/ADAPTERS.md](docs/ADAPTERS.md)</sub>

PAWN ships with six adapter implementations for fine-tuning the frozen backbone on human game data. *(The numbers below are from the legacy backbone — see [docs/ADAPTERS.md](docs/ADAPTERS.md) for full context.)*

| Method | Params (typical) | Accuracy (1800 Elo) | Description |
|--------|-----------------|---------------------|-------------|
| **[Bottleneck](https://arxiv.org/abs/1902.00751)** | 131K | 41.7% | Houlsby-style residual MLP adapters |
| **[RoSA](https://arxiv.org/abs/2401.04679)** | configurable | -- | Gradient-informed sparse + LoRA |
| **Sparse** | 503K-2.7M | 40.2-44.7% | Random binary mask on frozen weights |
| **[LoRA](https://arxiv.org/abs/2106.09685)** | ~65K | 34.1% | Low-rank attention projection adapters |
| **Hybrid** | ~65K | 34.1% | LoRA + FiLM combined |
| **[FiLM](https://arxiv.org/abs/1709.07871)** | ~17K | 30.3% | Per-channel affine modulation* |

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

- [Architecture](docs/ARCHITECTURE.md) -- model design, embeddings, training objective, game completion analysis
- [Training](docs/TRAINING.md) -- pretraining, adapter training, deployment
- [Adapters](docs/ADAPTERS.md) -- adapter methods, results, quick start
- [Accuracy Ceiling](docs/ACCURACY_CEILING.md) -- theoretical limits for random game prediction
- [Legacy Architecture](docs/LEGACY.md) -- the v0.x backbones, why they were retired, and how to load them

\*None of the existing experiments use FiLM to condition on anything. The existing FiLM experiments ask the question, 'how does FiLM perform when all parameters are learned'.

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
| Searchless Chess | [Ruoss et al., Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494) |
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

Apache 2.0. See [LICENSE](LICENSE).
