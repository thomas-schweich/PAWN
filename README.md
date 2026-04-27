# PAWN: Playstyle-Agnostic World-model Network for Chess

A small causal transformer trained on random chess games that learns legal moves, board state representations, and game dynamics purely from random legal move sequences absent any form of strategic play.

I've found PAWN to be a viable testbed for finetuning and augmentation methods at small scales. Since it is entirely unopinionated, it's a blank slate ready to be adapted, augmented, and finetuned into arbitrary player models with unique playstyles.

Finetuning PAWN has proven significantly more parameter-efficient than training new models from scratch and requires minimal compute resources.

Feel free to use PAWN in your own experiments. PAWN is developed as a personal project by a single developer and his imaginary friend (Claude) and has not been published or audited. If you spot a bug or inaccuracy, please help out by creating an issue or PR.


## Model Variants

The model comes in three sizes, all trained from scratch on random chess games generated on-the-fly by a Rust-based chess backend. The v1.0.0 weights were trained together for 200K steps at batch size 256 on a single B200 — all three variants see the same random-game batches each step, with one forward/backward pass per variant in sequence on the same GPU (see [cotrain config](configs/cotrain_three_variants.json)). The numbers below come from the best 5K-cadence checkpoint by val loss (step 195,000 ≈ 49.9M sequences) for all three variants:

| Variant | d_model | Layers | Heads | Params | Top-1 | Legal rate | Game completion | Download |
|---------|---------|--------|-------|--------|-------|------------|-----------------|----------|
| **PAWN-Small** | 256 | 8 | 4 | 8.94M | 8.54% | 99.7451% | 52.34% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-small) |
| **PAWN (Base)** | 512 | 8 | 8 | 34.65M | 8.57% | 99.9962% | 98.97% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-base) |
| **PAWN-Large** | 640 | 10 | 8 | 66.91M | 8.63% | 99.9990% | 99.76% | [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/thomas-schweich/pawn-large) |

*Metrics measured on a 2,048-game validation set of random games. **Game completion** is the ability to choose a legal move in every position throughout a random game. It is the primary signal that separates capacity between sizes. The number given above is non-autoregressive. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#game-completion-rate).*

All variants share the same architecture: [RMSNorm](https://arxiv.org/abs/1910.07467), [SwiGLU](https://arxiv.org/abs/2002.05202) FFN, [RoPE](https://arxiv.org/abs/2104.09864), factored move embeddings, and a vocabulary covering:

- 1,968 move actions (the `searchless_chess` vocabulary, one entry per legally-reachable (src, dst[, promotion]) tuple),
- 11 game-outcome tokens (pretraining outcomes: `WHITE_CHECKMATES`, `BLACK_CHECKMATES`, `STALEMATE`, `DRAW_BY_RULE`, `PLY_LIMIT`; Lichess-specific outcomes: `WHITE_RESIGNS`, `BLACK_RESIGNS`, `DRAW_BY_AGREEMENT`, `WHITE_WINS_ON_TIME`, `BLACK_WINS_ON_TIME`, `DRAW_BY_TIME`),
- and a single PAD token — 1,980 tokens total.

Tokens are coordinate pairs (UCI notation) with no piece type or side-to-move information — `e2e4` means the same token whether it's a pawn double-push or a rook move. The model learns to track piece placement, movement rules, and game state entirely from observation, which can be isolated via [linear probes](https://arxiv.org/abs/1610.01644).

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

## Architecture

<sub>More info: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)</sub>

Standard decoder-only [transformer](https://arxiv.org/abs/1706.03762) with next-token prediction. Each training example is a move sequence padded to 512 tokens. Factored embeddings decompose each move into source square + destination square + promotion piece. Predictions are not masked to legal moves — the model must infer legality from the move history alone. There is no board representation like [AlphaZero](https://arxiv.org/abs/1712.01815)'s 8x8xN planes; all state tracking is learned internally.

## What the Model Learns

Despite training exclusively on random games, PAWN develops rich internal representations. Linear probes on frozen hidden states decode chess concepts the model is never explicitly told about:

| Probe | Small | Base | Large |
|-------|-------|------|-------|
| Side to move | 100.0% | 100.0% | 100.0% |
| En passant square | 99.9% | 99.9% | 99.9% |
| Castling rights | 98.3% | 99.3% | 99.3% |
| Is check | 95.2% | 95.0% | 94.9% |
| Game phase | 94.8% | 95.8% | 95.9% |
| Piece type | 88.8% | 91.8% | 92.2% |
| Material count (R²) | 0.80 | 0.84 | 0.83 |

Full probe results including diagnostics are on each variant's [HuggingFace model card](https://huggingface.co/thomas-schweich/pawn-base).

## Adapter Methods

<sub>More info: [docs/ADAPTERS.md](docs/ADAPTERS.md)</sub>

PAWN ships with several adapter implementations for finetuning the frozen backbone on human game data. The table below is the current Pareto frontier across param budgets, all on Lichess games filtered to Elo 1800-1900 with both players in band (≈16 M games per epoch). Top entries use the `pawn-large` backbone with bottleneck adapters concentrated on the top 4 of 10 layers, which generalizes the layer-placement finding (top-4 ≈ 40% of depth wins on every backbone size tested).

| Method | Backbone | Params | Val top-1 (1800-1900 Elo) | Description |
|--------|----------|-------:|--------------------------:|-------------|
| **[Bottleneck](https://arxiv.org/abs/1902.00751)** dim=2500 top-4 (2 ep) | pawn-large | 25.6M | **52.37%** | Houlsby-style; current ceiling on this dataset |
| **Combo** retro-bn dim=1280 top-4 + d=0.05 sparse (1.4 ep) | pawn-large | 13.93M | 51.16% | Gradient-informed sparse + bottleneck |
| **Bottleneck** dim=1024 top-4 (2 ep) | pawn-base | 8.4M | 51.03% | Top-4 placement closes the backbone-size gap with extension |
| **Bottleneck** dim=2580 top-4 (1 ep) | pawn-large | 26.4M | 50.76% | Decomposition test — pure bottleneck baseline for the combo |
| **Bottleneck** dim=2500 top-4 (1 ep) | pawn-large | 25.6M | 50.74% | One-epoch ceiling; +1.6 pp from 2 epochs of training |
| **Bottleneck** dim=640 dist (1 ep) | pawn-large | 16.4M | 49.88% | All-layer distributed placement |
| **Bottleneck** dim=1024 top-4 (1 ep) | pawn-base | 8.4M | 49.23% | Top-4 +0.7 pp over distributed at the same param count |
| **[LoRA](https://arxiv.org/abs/2106.09685)** rank=16 qkvo | pawn-base | 524K | 35.62% | Low-rank attention projection adapters |
| **[RoSA](https://arxiv.org/abs/2401.04679)** retro-sparse d=0.01 | pawn-base | 84K | 30.45% | Gradient-informed sparse mask from LoRA warmup |

Hybrid (LoRA+FiLM) and [FiLM](https://arxiv.org/abs/1709.07871) remain in the codebase but aren't on the current frontier. See [docs/ADAPTERS.md](docs/ADAPTERS.md) for the full methodology — backbone-size H2H, layer-placement ladder, adapter-vs-from-scratch matched-budget comparison, and decay-shape analysis.

### Datasets

The "Lichess Full" dataset below was filtered to matches between players rated 1800-1900 and truncated to 1-10 million games, depending on adapter type and size (some smaller adapters saturate too rapidly to benefit from more games). But I parsed all ~300 million games and converted them to UCI as well as PAWN's training format (a) to make future experiments easier and (b) since others might find the pre-converted raw UCI helpful for other projects. And because the Rust engine is super fast anyway. I also kept the SAN notation and metadata from the original PGNs.

| Dataset | Games | Description | Link |
|---------|-------|-------------|------|
| Lichess Full | 286M train + 9.3M val + 9.0M test | Rated games from Q1 2025 (all Elos), holdout from Jan 2026 | [pawn-lichess-full](https://huggingface.co/datasets/thomas-schweich/pawn-lichess-full) |
| Stockfish nodes=1 | 900K train + 50K val + 50K test | NNUE self-play, 1 node/move | [stockfish-nodes1](https://huggingface.co/datasets/thomas-schweich/stockfish-nodes1) |

All datasets use pre-tokenized `list[int16]` move sequences (`tokens` column). The Lichess dataset also includes raw `san`/`uci` strings, clock annotations, Elo ratings, and full game metadata. Datasets load directly from HuggingFace via Polars lazy scan. Predicate pushdown makes it so that only the subset of data you select is actually downloaded.

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
