# PAWN (Playstyle-Agnostic World-model Network for Chess)

A causal transformer trained on random chess games, designed as a testbed for finetuning and augmentation methods at small scales. Apache 2.0.

## Repository Structure

```
pawn/
├── engine/          # Rust chess engine with PyO3 bindings (via shakmaty)
├── pawn/            # Core Python package
│   ├── config.py    # CLMConfig (small/base/large), TrainingConfig
│   ├── model.py     # PAWNCLM transformer (RMSNorm, SwiGLU, RoPE, factored embeddings)
│   ├── data.py      # On-the-fly random game data pipeline
│   ├── lichess_data.py  # Lichess PGN data pipeline + legal mask computation
│   ├── trainer.py   # Pretraining loop
│   ├── gpu.py       # GPU auto-detection (compile/AMP/SDPA backend)
│   ├── logging.py   # MetricsLogger (JSONL output)
│   ├── checkpoint.py # Atomic save/load, .complete sentinel, HF push
│   ├── adapters/    # Bottleneck, LoRA, FiLM, sparse, hybrid
│   ├── eval_suite/  # Probes, generation tests, diagnostics, lichess eval
│   └── dashboard/   # Solara training dashboard (metrics, charts, runner)
├── scripts/         # Training and evaluation entry points
├── tests/           # Unit tests
├── deploy/          # Runpod deployment scripts
└── docs/            # Architecture, training, adapter docs
```

## Building

This is a uv workspace. The root project is the `pawn` Python package; `engine/` is the sole workspace member.

```bash
# Build the Rust chess engine (required before anything else)
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python deps (dev tools like pytest, seaborn, solara are in base dependencies):
uv sync --extra rocm      # AMD (ROCm 7.1)
uv sync --extra cu128     # NVIDIA (CUDA 12.8)

# Run tests
uv run pytest tests/

# Pretrain from scratch (local dev)
uv run python scripts/train.py --variant base --local-checkpoints
```

PyTorch is a **base dependency** — `uv sync` always installs it (CPU build from PyPI by default). The extras (`rocm`, `cu128`) only control which GPU-accelerated build is pulled from the PyTorch index. You cannot accidentally end up without torch.

**GPU requirement**: `configure_gpu()` (called by every training and eval script) raises `RuntimeError` if no CUDA/ROCm GPU is detected. This prevents accidentally running GPU workloads on CPU, which is almost always a mistake. The environment variable `PAWN_ALLOW_CPU=1` overrides this check as a last resort for the rare case where CPU execution is genuinely intended (e.g. a lightweight backfill script). Unit tests do not call `configure_gpu()` and run fine on CPU without the override.

## Engine (`engine/`)

**Single source of truth** for all chess logic. All game simulation, move generation, legality checks, tokenization, PGN parsing, and board state extraction happen in Rust. No Python chess libraries.

- Uses rayon for parallel game generation (~43K games/sec, 150M+/hr)
- PyO3 bindings expose `chess_engine` module to Python
- Key functions: `generate_random_games()`, `parse_pgn_file()`, `compute_legal_token_masks_sparse()`, `extract_board_states()`, `export_move_vocabulary()`, `compute_accuracy_ceiling()`

## Model

### Architecture
- Decoder-only transformer, next-token prediction over 4,278 tokens
- Token vocabulary: 1 PAD + 4,096 grid (64x64 src/dst) + 176 promotions + 5 outcomes
- Factored embeddings: `src_embed[s] + dst_embed[d] + promo_embed[p]`
- Sequence format: `[outcome] [ply_1] ... [ply_N] [PAD] ... [PAD]` (256 tokens)

### Variants
- `CLMConfig.small()`: d=256, 8 layers, 4 heads, ~9.5M params
- `CLMConfig.base()`: d=512, 8 layers, 8 heads, ~35.8M params (default)
- `CLMConfig.large()`: d=640, 10 layers, 8 heads, ~68.4M params
- `CLMConfig.toy()`: d=64, 2 layers, for tests only

## Training

All training scripts require one of `--hf-repo REPO_ID` or `--local-checkpoints` (mutually exclusive). Use `--local-checkpoints` for local dev; use `--hf-repo` for any run where you need durable checkpoints.

### Pretraining

```bash
# Single model
uv run python scripts/train.py --variant base --local-checkpoints

# All three variants simultaneously (shared data batches, sequential GPU)
uv run python scripts/train_all.py --local-checkpoints

# Resume from checkpoint
uv run python scripts/train.py --variant base --resume checkpoints/step_00050000 --local-checkpoints
```

**`scripts/train.py`** key args:
- `--variant {small|base|large|toy}` — model size (default: base)
- `--resume PATH` — resume from checkpoint directory
- `--total-steps N` — training steps (default: 100,000)
- `--batch-size N` — batch size (default: 256)
- `--discard-ply-limit` — only train on naturally-ended games (no ply-limit truncation)
- Architecture overrides: `--d-model`, `--n-layers`, `--n-heads`, `--d-ff`, `--lr`, `--weight-decay`, `--warmup-steps`

**`scripts/train_all.py`** additional args:
- `--shm-checkpoints` — write checkpoints to `/dev/shm` (requires `--hf-repo`, volatile)
- `--run-evals` — auto-run probes + diagnostics after training completes
- `--publish-results` — push eval results to HF
- `--patience N` — per-model early stopping patience (eval intervals without improvement)

### Adapter Training

All adapter scripts require `--checkpoint PATH` (pretrained weights) and `--pgn PATH` (Lichess PGN file). They freeze the backbone and train only adapter parameters.

```bash
# Example: train a LoRA adapter on Lichess 1800-1900 games
uv run python scripts/train_lora.py \
    --checkpoint checkpoints/pawn-base \
    --pgn data/lichess_1800_1900.pgn \
    --lora-rank 4 --lr 3e-4 --local-checkpoints
```

| Script | Adapter | Key args | Typical params |
|--------|---------|----------|----------------|
| `train_bottleneck.py` | Houlsby MLP | `--bottleneck-dim 8` | ~131K |
| `train_lora.py` | Low-rank attention | `--lora-rank 4 --lora-targets qkvo` | ~65K |
| `train_film.py` | Channel-wise affine | `--no-output-film` | ~17K |
| `train_sparse.py` | Binary mask | `--density 0.01 --sparse-targets qkvo` | ~503K-2.7M |
| `train_hybrid.py` | LoRA + FiLM | `--lora-rank 4 --film-lr 1e-3` | ~65K |
| `train_tiny.py` | None (from scratch) | `--d-model 84 --n-layers 2` | ~524K |

Common adapter args: `--epochs 50`, `--batch-size 64`, `--lr 3e-4`, `--patience 10`, `--val-every 1`, `--max-games 12000`, `--min-ply 10`

### Common CLI Patterns

- `--sdpa-math` — force MATH SDPA backend (required for ROCm + torch.compile)
- `--no-compile` — disable torch.compile
- `--no-amp` — disable mixed precision
- `--num-workers N` — DataLoader workers (default: 8 for adapters, 4 for pretraining)
- `--device {cuda|cpu}` — device selection
- `--wandb` — enable Weights & Biases logging

## Evaluation & Metrics

### Linear Probes

```bash
uv run python scripts/eval_probes.py --log-dir logs --device cuda
```

Trains linear probes on frozen hidden states to measure internal representations (piece type, check status, castling rights, material count, game phase, etc.). Args: `--n-games 4096`, `--n-val-games 1024`, `--n-epochs 20`, `--run RUN_NAME` (specific run).

### Move Prediction Accuracy

```bash
uv run python scripts/eval_accuracy.py \
    --checkpoint checkpoints/pawn-base \
    --pgn data/lichess_1800_1900.pgn \
    --adapter-checkpoint logs/run_*/checkpoints/best
```

MAIA-compatible evaluation with per-phase and per-ply accuracy. Args: `--min-eval-ply 10`, `--max-games 50000`, `--per-ply`.

### Theoretical Accuracy Ceilings

```bash
uv run python scripts/compute_theoretical_ceiling.py
```

Computes upper bounds on top-1 accuracy for random games: unconditional (E[1/N_legal] = 6.43%), naive-conditioned (1-ply filter = 6.44%), MCTS-conditioned (32 rollouts = 7.92%). CPU-intensive.

### Export to HuggingFace

```bash
uv run python scripts/export_hf_repo.py --run-dir logs/run_YYYYMMDD_HHMMSS
```

Converts a training run to HuggingFace repo format (safetensors + metrics). Finds best checkpoint by val loss.

## Checkpoints

Pre-trained weights are HuggingFace git submodules under `checkpoints/`:
- `checkpoints/pawn-small` — 9.5M params, `CLMConfig.small()`
- `checkpoints/pawn-base` — 35.8M params, `CLMConfig.base()`
- `checkpoints/pawn-large` — 68.4M params, `CLMConfig.large()`

Pull with: `git submodule update --init --remote checkpoints/pawn-base`

### Checkpoint Format (safetensors)

Checkpoints are directories, not single files:
```
step_00065000/
├── model.safetensors        # model weights
├── optimizer.safetensors    # flattened optimizer state
├── training_state.json      # step, scheduler, scaler, RNG (base64)
├── config.json              # model + training config
└── .complete                # SHA-256 hashes of all files (integrity sentinel)
```

Central module: `pawn/checkpoint.py`. All save/load goes through this module.
Legacy `.pt` files are still loadable (backward compatible).

### Checkpoint Storage Modes

All training scripts require one of:
- `--hf-repo REPO_ID` — push checkpoints to a HuggingFace branch as they're written (durable)
- `--local-checkpoints` — save locally only (for development without an HF account)

HF mode creates a `run/{run_id}` branch. HF pushes happen in background threads (one per model slot) so training is not blocked by uploads. Squash-merge into main when satisfied.

Optional: `--shm-checkpoints` writes checkpoints to `/dev/shm` (RAM-backed filesystem, instant writes). Requires `--hf-repo` since `/dev/shm` is volatile. Old checkpoints are cleaned up after successful HF push, keeping only the latest and the best (by val loss) for post-training evals.

### Data Integrity

**Every checkpoint write is atomic**: files are written to a `.tmp` directory, then renamed.
The `.complete` sentinel contains SHA-256 hashes of every file in the checkpoint.
**Hashes are always verified on load — no exceptions.**

- `IncompleteCheckpointError` — raised when `.complete` sentinel is missing
- `CheckpointIntegrityError` — raised when any hash mismatches

**Never use `kill -9` on training processes.** SIGTERM is handled gracefully: a flag is set,
the training loop checks it between steps, saves a checkpoint, pushes to HF, and exits cleanly.

**Never rsync checkpoint files from running pods.** Checkpoints are pushed to HuggingFace
from the trainer. Pull via `deploy/sync.sh` (submodule update).

## RunPod Operations

### Docker Build & Push

```bash
# Build runner target (auto-stop after training completes)
docker build --platform linux/amd64 \
    --build-arg GIT_HASH=$(git rev-parse HEAD) \
    --build-arg GIT_TAG=$(git tag --points-at HEAD) \
    --target runner \
    -t thomasschweich/pawn:latest-runner .

# Build interactive target (SSH + Jupyter, stays alive)
docker build --platform linux/amd64 \
    --build-arg GIT_HASH=$(git rev-parse HEAD) \
    --target interactive \
    -t thomasschweich/pawn:latest .

docker push thomasschweich/pawn:latest-runner
```

Code lives at `/opt/pawn` on pods (outside the `/workspace` volume mount).

### Pod Lifecycle

Use `deploy/pod.sh` for all pod management. Requires `runpodctl` (`wget -qO- cli.runpod.net | sudo bash`).

```bash
# Create a pod
bash deploy/pod.sh create myexp --gpu h100

# SSH into it
bash deploy/pod.sh ssh myexp

# Launch training
bash deploy/pod.sh launch myexp scripts/train_all.py --hf-repo thomas-schweich/pawn-{variant}

# Stop (preserves volume, stops billing)
bash deploy/pod.sh stop myexp

# Delete (destroys everything)
bash deploy/pod.sh delete myexp
```

GPU shortcuts: `a5000`, `a40`, `a6000`, `4090`, `5090`, `l40s`, `h100`. Pod configs are cached in `~/.config/pawn/pods/<name>.env`.

### GPU Selection

Benchmarks from pretraining 3 models concurrently (`train_all.py`, batch=256):

| GPU | VRAM | $/hr | Step time | 100K cost | Notes |
|-----|------|------|-----------|-----------|-------|
| B200 | 192GB | $4.99 | 0.28s | ~$39 | Fastest |
| H200 SXM | 80GB | $3.59 | 0.34s | ~$34 | Best wall-clock/cost balance |
| RTX PRO 6000 | 48GB | $1.89 | 0.62s | ~$33 | Cheapest viable |
| A100 PCIe | 80GB | $1.39 | 0.79s | ~$30 | Cheapest overall |
| L40S | 48GB | $0.86 | 1.37s | ~$33 | Slow but cheap |
| RTX 5090/4090/3090 | 24-32GB | — | OOM | — | Insufficient VRAM for 3 models |

Total cost is remarkably consistent ($30-39) across viable GPUs. The choice is wall-clock time vs cost, not cost vs cost. Single-model training fits on 24GB GPUs.

### Required Pod Configuration

- **Always attach a network volume.** Checkpoints write to disk during atomic rename and HF push. Ephemeral container disk is lost on pod termination.
- **Set `HF_TOKEN` as a pod environment variable** for automatic HuggingFace authentication. The entrypoint persists it to `~/.cache/huggingface/token`.
- `PAWN_MODEL=thomas-schweich/pawn-base` — auto-pull a checkpoint on startup (runner target).
- `PAWN_CMD` — training command to execute (alternative to Docker CMD args).

### Pod Safety

- Stop pods with `runpodctl pod stop` or `bash deploy/pod.sh stop` — sends SIGTERM, trainer saves and pushes before exiting.
- **Never `runpodctl pod delete` while training is running** — data loss risk.
- **Never `kill -9` training processes** — use SIGTERM (plain `kill`), which triggers graceful shutdown.
- **Never rsync checkpoint files from running pods** — pull via HF submodule instead.

## Monitoring Training Progress

### Key Principle: Write Scripts to Disk for Pre-Approval

When setting up recurring monitoring, **always write the monitoring script to a file first** so the user can review and pre-approve it. This avoids repeated permission prompts when `/loop` fires.

**Pattern:**
1. Write a bash script to disk (e.g., `scripts/check_my_run.sh`)
2. User reviews and approves the script
3. Schedule with `/loop 15m bash scripts/check_my_run.sh`

**Example monitoring script:**

```bash
#!/usr/bin/env bash
# scripts/check_my_run.sh — monitor a specific training run
set -euo pipefail
bash /home/tas/pawn/scripts/monitor_training.sh <POD_ID>
```

Or for local-only monitoring:

```bash
#!/usr/bin/env bash
set -euo pipefail
bash /home/tas/pawn/scripts/check_progress.sh --sync
```

### Available Monitoring Tools

| Tool | What it does |
|------|-------------|
| `scripts/monitor_training.sh [POD_ID]` | SSH to pod, sync metrics via rsync, show per-variant step/loss/acc/ETA, check HF checkpoint branches |
| `scripts/check_progress.sh [--sync]` | Show progress from local `logs/` and HF submodules. `--sync` pulls submodules first. |
| `deploy/sync.sh [name]` | Pull latest checkpoints/metrics from HuggingFace submodules |
| `python -m pawn.dashboard --log-dir logs` | Solara web dashboard with interactive charts |

### Dashboard

```bash
python -m pawn.dashboard --log-dir logs
```

Reads `metrics.jsonl` files, no dependency on training packages. Auto-detects run type from config fields. Shows loss curves, accuracy, LR schedules, GPU utilization, patience clocks, and adapter-specific diagnostics. Requires restart for code changes (no hot reload).

## Logs

Training metrics in `logs/` (gitignored). Each run gets a timestamped directory with `metrics.jsonl` and a random slug (e.g., `run_20260325_140000_zesty-osprey/`).

`MetricsLogger` (`pawn/logging.py`) writes one JSON object per line. Every record includes timestamp, step, elapsed time, and memory stats. Config records include hostname, git hash, git tag, and run slug.

## Hyperparameter Sweeps

Optuna integration via `pawn/sweep.py` and `scripts/sweep.py`:

```bash
uv run python scripts/sweep.py \
    --adapter lora --n-trials 30 --n-jobs 2 --n-gpus 2 \
    --total-steps 20000 --pruner hyperband \
    --checkpoint checkpoints/pawn-base --pgn data/lichess_1800_1900.pgn \
    --local-checkpoints
```

Supports all adapter types + architecture search. GPU affinity assigns `CUDA_VISIBLE_DEVICES = trial.number % n_gpus`. SQLite-backed study persistence. Pruner options: `hyperband`, `median`, `none`.

## Key Patterns & Gotchas

- **DataLoader workers must use `multiprocessing_context='spawn'`** — the Rust engine uses rayon, and fork after rayon init causes deadlocks.
- **`SDPA_BACKEND` must be set before `torch.compile()`** — compiled code captures the backend at trace time. `apply_gpu_config()` handles this.
- **ROCm flash attention bug**: with `torch.compile`, flash attention backward has stride issues. Use `--sdpa-math` to force the MATH SDPA backend.
- **Sparse logit projection**: `forward_hidden()` returns `(B,T,d_model)`, then only loss-masked positions project through `lm_head` — avoids full `(B,T,V)` materialization.
- **Legal mask via Rust**: `LegalMaskBuilder` replays games in Rust, returns sparse indices (~2 MB) scattered into a pre-allocated GPU buffer (vs ~70 MB dense).
- **GPU auto-detection**: `pawn.gpu.configure_gpu()` selects compile/AMP/SDPA settings. `apply_gpu_config()` applies them. NVIDIA uses flash attention + compile; AMD uses MATH SDPA + compile.
- **Factored embeddings**: each move token decomposes into `src_embed[s] + dst_embed[d] + promo_embed[p]`, reducing embedding parameters by ~32x.
