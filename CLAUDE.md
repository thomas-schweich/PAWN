# PAWN (Playstyle-Agnostic World-model Network for Chess)

A causal transformer trained on random chess games, designed as a testbed for finetuning and augmentation methods at small scales. Apache 2.0.

## Repository Structure

```
pawn/
├── engine/          # Rust chess engine with PyO3 bindings (via shakmaty)
├── pawn/            # Core Python package
│   ├── config.py    # CLMConfig (small/base/large), TrainingConfig
│   ├── model.py     # PAWNCLM transformer (RMSNorm, SwiGLU, RoPE, factored embeddings)
│   ├── data.py      # On-the-fly random game data pipeline (prepend_outcome flag)
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

The only extras are GPU backends (`rocm` or `cu128`). Everything else (pytest, solara, optuna, seaborn, etc.) is in base dependencies. PyTorch lives in the extras because uv can't resolve CPU/CUDA/ROCm from a single lockfile — always specify `--extra rocm` or `--extra cu128`.

**GPU requirement**: `configure_gpu()` (called by every training and eval script) raises `RuntimeError` if no CUDA/ROCm GPU is detected. This prevents accidentally running GPU workloads on CPU, which is almost always a mistake. The environment variable `PAWN_ALLOW_CPU=1` overrides this check as a last resort for the rare case where CPU execution is genuinely intended (e.g. a lightweight backfill script). Unit tests do not call `configure_gpu()` and run fine on CPU without the override.

## Engine (`engine/`)

**Single source of truth** for all chess logic. All game simulation, move generation, legality checks, tokenization, PGN parsing, and board state extraction happen in Rust. No Python chess libraries.

- Uses rayon for parallel game generation (~43K games/sec, 150M+/hr)
- PyO3 bindings expose `chess_engine` module to Python
- Key functions: `generate_random_games()`, `parse_pgn_file()`, `compute_legal_token_masks_sparse()`, `extract_board_states()`, `export_move_vocabulary()`, `compute_accuracy_ceiling()`
- `export_move_vocabulary()` returns the 1,968-entry searchless_chess action table used by the factored embeddings.

## Model

### Architecture
- Decoder-only transformer, next-token prediction over 1,968 move tokens (1,980 total vocab)
- Token vocabulary: 1,968 searchless_chess actions (0-1967) + 1 PAD (1968) + 11 outcomes (1969-1979) = 1,980 total
- Factored embeddings: `src_embed[s] + dst_embed[d] + promo_embed[p]`
- Sequence format: `[ply_1] ... [ply_N] [PAD] ... [PAD]` (512 tokens) — outcome prefix is optional via `prepend_outcome` flag

> **Legacy note.** Earlier versions of this codebase used a ~60k-entry move
> vocabulary and two separate parquet layouts ("v1" = pure-moves tokens with
> outcomes derived coarsely from the PGN `result` header, "v2" = tokens with
> the outcome prepended at position 0). **Both are gone.** The current code
> only knows about the 1,968-action vocabulary and the single canonical
> parquet schema written by `scripts/extract_lichess_parquet.py` (pure-moves
> `tokens` + granular `outcome_token` column + per-game metadata). If you
> find a reference to a "v1 vocab", "v2 format", `_result_to_outcome`,
> `strip_outcome_token`, or `no_outcome_token`, it's a bug — those were all
> removed during the 0.x → stable transition. Legacy checkpoints trained
> against the old vocabulary are accessible only via the
> `pre-vocab-transition` git tag; they cannot be loaded or trained against
> from the current tree.

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

# All three variants simultaneously (shared data batches, sequential GPU).
# Cotrain always takes a JSON config because the `variants` list is
# shaped like [{"name": ..., "variant": ..., ...}, ...] and isn't
# expressible on a flat CLI. A default 3-variant config ships at
# configs/cotrain_three_variants.json.
uv run python scripts/train.py --config configs/cotrain_three_variants.json

# Resume from checkpoint
uv run python scripts/train.py --variant base --resume checkpoints/step_00050000 --local-checkpoints
```

**`scripts/train.py`** key args (all run types):
- `--config PATH` — load a JSON run config (required for cotrain)
- `--run-type {pretrain|adapter|cotrain}` — dispatch target
- `--variant {small|base|large|toy|custom}` — pretrain model size (default: base)
- `--resume PATH` — resume from checkpoint directory
- `--total-steps N` — training steps (default: 100,000)
- `--batch-size N` — batch size (default: 256)
- `--discard-ply-limit` — only train on naturally-ended games (no ply-limit truncation)
- Architecture overrides: `--d-model`, `--n-layers`, `--n-heads`, `--d-ff`, `--lr`, `--weight-decay`, `--warmup-steps`

**Cotrain-specific config fields** (in the JSON):
- `shm_checkpoints: true` — write checkpoints to `/dev/shm` (requires `hf_repo`, volatile)
- `run_evals: true` — run per-slot probes + diagnostics after training completes
- `lichess_pgn: "..."` — Lichess PGN path for Maia-style accuracy eval (requires `run_evals`)
- `publish_results: true` — push `eval_results.json` to HF (requires `hf_repo`)
- `patience: N` — per-variant early stopping patience (eval intervals without improvement)

### Distillation

Distill a frozen teacher (any HF repo or local checkpoint) into a smaller student. The student sees the same on-the-fly random-game batches as pretraining; the teacher provides soft targets every step. Loss is `alpha * CE(student, ground_truth) + (1 - alpha) * T**2 * KL(softmax(teacher/T), softmax(student/T))` (Hinton et al., 2015).

```bash
# Distill base → small with default knobs
uv run python scripts/train.py --run-type distill --variant small \
    --teacher-checkpoint thomas-schweich/pawn-base \
    --temperature 4.0 --alpha 0.5 --local-checkpoints

# JSON config (configs/distill_small_from_base.json) for full reproducibility
uv run python scripts/train.py --config configs/distill_small_from_base.json
```

Key knobs (all on `DistillConfig` in `pawn/run_config.py`):
- `--teacher-checkpoint` — HF repo id or local checkpoint dir (required)
- `--variant {toy|small|base|large|custom}` — student arch (default `small`)
- `--temperature` — softmax temperature for both teacher and student logits (default `4.0`)
- `--alpha` — mixing coefficient: `alpha * CE + (1 - alpha) * KL`. `alpha=0` is pure soft-target distillation; `alpha=1` is rejected (just use pretrain)
- `--kd-direction {forward|reverse|jsd}` — KL direction. `forward` = `KL(teacher || student)` (Hinton form, default), `reverse` = mode-seeking, `jsd` = symmetric Jensen-Shannon
- `--top-k-teacher N` — restrict KL to the teacher's top-k logits per position (cuts memory; useful for very large vocabs — for PAWN's 1,980-vocab usually leave at `None`)
- `--hidden-loss-weight λ` — additional MSE between teacher and student final hidden states (a learned linear projector handles `d_model` mismatch). `0.0` disables it (default)

The teacher's `prepend_outcome` setting is auto-detected from its saved `training_config` and the student is forced to match — passing an explicit conflicting `prepend_outcome` prints a warning. The student's `vocab_size` and `max_seq_len` must be compatible with the teacher's (both checked at construction).

### Adapter Training

All adapter strategies dispatch through the unified `scripts/train.py` with `--run-type adapter --strategy STRATEGY`. They freeze the backbone and train only adapter parameters. Both `--checkpoint PATH` and `--pgn PATH` are required.

```bash
# Example: train a LoRA adapter on Lichess 1800-1900 games
uv run python scripts/train.py --run-type adapter --strategy lora \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 1900 \
    --lora-rank 4 --lr 3e-4 --local-checkpoints
```

| `--strategy` value  | Adapter | Key args | Typical params |
|---------------------|---------|----------|----------------|
| `bottleneck`        | Houlsby MLP | `--bottleneck-dim 8` | ~131K |
| `lora`              | Low-rank attention | `--lora-rank 4 --lora-targets qkvo` | ~65K |
| `film`              | Channel-wise affine | `--no-output-film` | ~17K |
| `sparse`            | Binary mask | `--density 0.01 --sparse-targets qkvo` | ~503K-2.7M |
| `hybrid`            | LoRA + FiLM | `--lora-rank 4` | ~65K |
| `rosa`              | Gradient-informed sparse + LoRA (3-phase) | `--rosa-mode rosa` | varies |
| `specialized_clm`   | From-scratch standalone transformer (no backbone) | `--d-model 84 --n-layers 2` | ~524K |
| `unfreeze`          | Fine-tune top N backbone layers | `--unfreeze-layers 6,7` | varies |

Common adapter args: `--epochs 50`, `--batch-size 64`, `--lr 3e-4`, `--patience 10`, `--val-every 1`, `--max-games 12000`, `--min-ply 10`, `--checkpoint-interval 5000`

Adapter checkpoints are written to `logs/run_*/checkpoints/step_{global_step:08d}/` (matching the pretraining layout — never overwritten). A save fires whenever val hits a new best, whenever the step is a `--checkpoint-interval` multiple, or at termination (step limit, patience, shutdown). To find the best step from a run's `metrics.jsonl`, use `pawn.checkpoint.find_best_adapter_step`.

LR schedule: `--lr-schedule {cosine,wsd,constant,one_cycle,infinite}`. Default `cosine`.
- `wsd` — Warmup-Stable-Decay. Holds peak LR for `1 - warmup_frac - decay_frac` of training, then decays over the last `--decay-frac` (default 0.1). `--wsd-decay-shape {linear,cosine}` controls the tail curve.
- `constant` — linear warmup → hold peak indefinitely. Pair with `--patience` to actually stop.
- `one_cycle` — Smith (2018) one-cycle: ramp from `peak/25` → `peak` over `--warmup-frac` of steps (try 0.3), then cosine-decay to `peak/10000`.
- `infinite` — warmup → cosine cooldown to `--stable-lr-ratio` (default 0.1) × peak over `--cooldown-frac` of steps (default 0.2) → flat stable plateau → final decay to 0 over the last `--decay-frac` of steps (default 0.1, shape set by `--wsd-decay-shape`). The stable-plateau LR depends only on `--stable-lr-ratio`, not on `total_steps`, so any checkpoint taken during the plateau is a valid resumption point — extend `total_steps` on resume and the plateau simply lasts longer before the final decay kicks in. Useful when you don't want to commit to a total-step count upfront. See Hägele et al. (2024) arXiv:2405.18392.

Legal-move handling (defaults match pre-existing behavior):
- `--disable-legal-mask` — drop the `-inf` hard mask on illegal logits and compute CE over the full 1,980-token vocabulary (same as pretraining). Useful for probing whether the adapter is leaning on the mask.
- `--illegal-penalty λ` — adds `λ · E[P_illegal]` to the loss (mean softmax mass on illegal tokens). Only valid together with `--disable-legal-mask` — under the hard mask this term is analytically zero. Eval reports `illegal_pred_rate` / `illegal_prob_mass` in this regime.

### Common CLI Patterns

- `--sdpa-math` — force MATH SDPA backend (debugging escape hatch; not required anymore on ROCm)
- `--no-compile` — disable torch.compile
- `--no-amp` — disable mixed precision
- `--num-workers N` — DataLoader workers (default: 8 for adapters, 4 for pretraining)
- `--device {cuda|cpu}` — device selection
- `--wandb` — enable Weights & Biases metrics logging (pretrain, cotrain, and adapter). Each process invocation creates a fresh run — no W&B state is persisted to checkpoints, so pause/resume is unaffected. Cotrain slots in a single invocation share `group=cotrain-<slug>` so variants cluster together. Resumed runs are independent runs; link them in the UI by filtering on the shared `git:<hash>` tag or the same HF branch URL. Set `PAWN_WANDB_MODE=disabled` to force offline behavior (CI / no network). Project name defaults to `pawn`; override via `WANDB_PROJECT` env var or `TrainingConfig.wandb_project`.

## Evaluation & Metrics

### Linear Probes

```bash
uv run python scripts/eval_probes.py --log-dir logs --device cuda
```

Trains linear probes on frozen hidden states to measure internal representations (piece type, check status, castling rights, material count, game phase, etc.). Args: `--n-games 4096`, `--n-val-games 1024`, `--n-epochs 20`, `--run RUN_NAME` (specific run).

### Move Prediction Accuracy

```bash
uv run python scripts/eval_accuracy.py \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 1900 \
    --adapter-checkpoint logs/run_*/checkpoints/step_00020000
```

MAIA-compatible evaluation with per-phase and per-ply accuracy. Args: `--min-eval-ply 10`, `--max-games 50000`, `--per-ply`.

### Theoretical Accuracy Ceilings

```bash
uv run python scripts/compute_theoretical_ceiling.py
```

Computes theoretical accuracy ceilings for random games via Monte Carlo rollouts: unconditional (E[1/N_legal]), naive-conditioned (1-ply filter), and MC-conditioned (Bayes-optimal with outcome knowledge). Reports a bias bracket (naive vs split-half corrected estimates) and bootstrap 95% CIs clustered by game. CPU-intensive.

### Export to HuggingFace

```bash
uv run python scripts/export_hf_repo.py --run-dir logs/run_YYYYMMDD_HHMMSS
```

Converts a training run to HuggingFace repo format (safetensors + metrics). Finds best checkpoint by val loss.

## Checkpoints

Pre-trained weights are hosted on HuggingFace and loaded directly by repo ID:
- `thomas-schweich/pawn-small` — 9.5M params, `CLMConfig.small()`
- `thomas-schweich/pawn-base` — 35.8M params, `CLMConfig.base()`
- `thomas-schweich/pawn-large` — 68.4M params, `CLMConfig.large()`

All scripts accept HF repo IDs for `--checkpoint` (e.g. `--checkpoint thomas-schweich/pawn-base`). Weights are downloaded and cached automatically via `huggingface_hub`.

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
from the trainer. Load via HF repo ID (e.g. `--checkpoint thomas-schweich/pawn-base`).

## RunPod Operations

### Docker Image

Docker images are **automatically built and pushed to Docker Hub by CI** on every merge to main. No manual builds needed.

| Tag | Target | Base | GPU |
|-----|--------|------|-----|
| `thomasschweich/pawn:latest` | `runtime` | `python:3.12-slim` | CUDA (cu128 wheels bundle runtime) |
| `thomasschweich/pawn:dev` | `dev` | `python:3.12-slim` | CUDA + Claude Code + tmux |
| `thomasschweich/pawn:rocm` | `runtime-rocm` | `python:3.12-slim` | ROCm 7.1 (wheel bundles runtime) |
| `thomasschweich/pawn:dev-rocm` | `dev-rocm` | `python:3.12-slim` | ROCm 7.1 + Claude Code + tmux |

All images use `python:3.12-slim` — PyTorch cu128 wheels bundle CUDA runtime as separate `nvidia-*` pip packages, and PyTorch ROCm wheels bundle HIP/rocBLAS/MIOpen/etc. inside the wheel itself (~2.8 GB). No nvidia/cuda or rocm base image needed. The only host requirement is the GPU kernel driver.

Code lives at `/opt/pawn` on all images. SSH in and run experiments directly.

To build locally (rarely needed):
```bash
# CUDA
docker build --platform linux/amd64 --target runtime \
    --build-arg GIT_HASH=$(git rev-parse HEAD) \
    -t thomasschweich/pawn:latest .

# ROCm
docker build --platform linux/amd64 --target runtime-rocm \
    --build-arg GIT_HASH=$(git rev-parse HEAD) \
    -t thomasschweich/pawn:rocm .
```

### Pod Lifecycle

Use `deploy/pod.sh` for all pod management. Requires `runpodctl` (`wget -qO- cli.runpod.net | sudo bash`).

```bash
# Create a pod
bash deploy/pod.sh create myexp --gpu h100

# SSH into it
bash deploy/pod.sh ssh myexp

# Launch training
bash deploy/pod.sh launch myexp scripts/train.py --config configs/cotrain_three_variants.json --hf-repo thomas-schweich/pawn-{variant}

# Stop (preserves volume, stops billing)
bash deploy/pod.sh stop myexp

# Delete (destroys everything)
bash deploy/pod.sh delete myexp
```

GPU shortcuts: `a5000`, `a40`, `a6000`, `4090`, `5090`, `l40s`, `h100`. Pod configs are cached in `~/.config/pawn/pods/<name>.env`.

### GPU Selection

Benchmarks from pretraining 3 models concurrently (cotrain, batch=256):

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
- `PAWN_DASHBOARD=0` — disable the auto-started dashboard + Caddy proxy (enabled by default).

### Pod Safety

- Stop pods with `runpodctl pod stop` or `bash deploy/pod.sh stop` — sends SIGTERM, trainer saves and pushes before exiting.
- **Never `runpodctl pod delete` while training is running** — data loss risk.
- **Never `kill -9` training processes** — use SIGTERM (plain `kill`), which triggers graceful shutdown.
- **Never rsync checkpoint files from running pods** — load via HF repo ID instead.

## Monitoring Training Progress

### Key Principle: Write Scripts to Disk for Pre-Approval

When setting up recurring monitoring, **always write the monitoring script to a file first** so the user can review and pre-approve it. This avoids repeated permission prompts when `/loop` fires.

**Pattern:**
1. Write a bash script to disk (e.g., `scripts/check_my_run.sh`)
2. User reviews and approves the script
3. Schedule with `/loop 15m bash scripts/check_my_run.sh`

The primary monitoring interface is the Solara dashboard (`python -m pawn.dashboard --log-dir logs`); any ad-hoc polling helpers are one-offs the model is expected to write on demand and leave out of the repo.

### Dashboard

```bash
python -m pawn.dashboard --log-dir logs
```

Reads `metrics.jsonl` files, no dependency on training packages. Auto-detects run type from config fields. Shows loss curves, accuracy, LR schedules, GPU utilization, patience clocks, and adapter-specific diagnostics. Requires restart for code changes (no hot reload).

**On RunPod pods**, the dashboard starts automatically and is proxied through Caddy on port 8888. Access it via the RunPod HTTP proxy URL (the "Connect" button → port 8888). Set `PAWN_DASHBOARD=0` as a pod environment variable to disable it.

## Logs

Training metrics in `logs/` (gitignored). Each run gets a timestamped directory with `metrics.jsonl` and a random slug (e.g., `run_20260325_140000_zesty-osprey/`).

`MetricsLogger` (`pawn/logging.py`) writes one JSON object per line. Every record includes timestamp, step, elapsed time, and memory stats. Config records include hostname, git hash, git tag, and run slug.

## Hyperparameter Sweeps

Optuna integration via `pawn/sweep.py` and `scripts/sweep.py`:

```bash
uv run python scripts/sweep.py \
    --adapter lora --n-trials 30 --n-jobs 2 --n-gpus 2 \
    --total-steps 20000 --pruner hyperband \
    --checkpoint thomas-schweich/pawn-base --pgn thomas-schweich/pawn-lichess-full \
    --local-checkpoints
```

Supports all adapter types + architecture search. GPU affinity assigns `CUDA_VISIBLE_DEVICES = trial.number % n_gpus`. SQLite-backed study persistence. Pruner options: `hyperband`, `median`, `none`.

## Key Patterns & Gotchas

- **DataLoader workers must use `multiprocessing_context='spawn'`** — the Rust engine uses rayon, and fork after rayon init causes deadlocks.
- **`SDPA_BACKEND` must be set before `torch.compile()`** — compiled code captures the backend at trace time. `apply_gpu_config()` handles this.
- **ROCm works**: Previously the flash-attention backward on ROCm hit a stride mismatch when combined with `torch.compile` + AMP. We worked around it by forcing RoPE outputs to be contiguous before SDPA in `pawn.model.Attention.forward` — flash is now the default on AMD too. `--sdpa-math` remains available as a debugging escape hatch but is no longer required. Everything else — training, eval, adapters, data loading — works identically on ROCm and CUDA. **Do not assume bugs are ROCm-specific.** Every other time something has failed on AMD it turned out to be a bug in our code (wrong torch version installed, stale lockfile, missing dependency, etc.), not a ROCm issue.
- **Sparse logit projection**: `forward_hidden()` returns `(B,T,d_model)`, then only loss-masked positions project through `lm_head` — avoids full `(B,T,1980)` materialization.
- **Legal mask via Rust**: `LegalMaskBuilder` replays games in Rust, returns sparse indices (~2 MB) scattered into a pre-allocated GPU buffer (vs ~70 MB dense).
- **GPU auto-detection**: `pawn.gpu.configure_gpu()` selects compile/AMP/SDPA settings. `apply_gpu_config()` applies them. Both NVIDIA and AMD use flash attention + compile by default (flash on AMD relies on the RoPE contiguous fix in `Attention.forward`). Both paths are tested and production-validated.
- **Factored embeddings**: each move token decomposes into `src_embed[s] + dst_embed[d] + promo_embed[p]`, reducing embedding parameters by ~32x.
