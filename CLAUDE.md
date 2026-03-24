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

# Install Python deps:
uv sync --extra rocm      # AMD (ROCm 7.1)
uv sync --extra cu128     # NVIDIA (CUDA 12.8)
uv sync --extra dev       # + pytest, ipykernel

# Run tests
uv run pytest tests/

# Pretrain from scratch
uv run python scripts/train.py --variant base --local-checkpoints
```

## Engine (`engine/`)

**Single source of truth** for all chess logic. All game simulation, move generation, legality checks, tokenization, PGN parsing, and board state extraction happen in Rust. No Python chess libraries.

- Uses rayon for parallel game generation (~43K games/sec, 150M+/hr)
- PyO3 bindings expose `chess_engine` module to Python
- Key functions: `generate_random_games()`, `parse_pgn_file()`, `compute_legal_token_masks_sparse()`, `extract_board_states()`, `export_move_vocabulary()`

## Model (`pawn/`)

### Architecture
- Decoder-only transformer, next-token prediction over 4278 tokens
- Token vocabulary: 1 PAD + 4096 grid (64x64 src/dst) + 176 promotions + 5 outcomes
- Factored embeddings: `src_embed[s] + dst_embed[d] + promo_embed[p]`
- Sequence format: `[outcome] [ply_1] ... [ply_N] [PAD] ... [PAD]` (256 tokens)

### Variants
- `CLMConfig.small()`: d=256, 8 layers, 4 heads, ~9.5M params
- `CLMConfig.base()`: d=512, 8 layers, 8 heads, ~35.8M params (default)
- `CLMConfig.large()`: d=640, 10 layers, 8 heads, ~68.4M params
- `CLMConfig.toy()`: d=64, 2 layers, for tests only

### Key Patterns

- **Sparse logit projection**: `forward_hidden()` returns `(B,T,d_model)`, then only loss-masked positions project through `lm_head` -- avoids full `(B,T,V)` materialization
- **Legal mask via Rust**: `LegalMaskBuilder` replays games in Rust, returns sparse indices scattered into a pre-allocated GPU buffer
- **DataLoader workers**: Must use `multiprocessing_context='spawn'` -- the engine uses rayon, and fork after rayon init causes deadlocks
- **GPU auto-detection**: `pawn.gpu.configure_gpu()` selects compile/AMP/SDPA settings. ROCm uses MATH SDPA backend (flash attention backward has stride issues with torch.compile)
- **SDPA backend global**: `pawn.model.SDPA_BACKEND` is set by `apply_gpu_config()` and used in `Attention.forward()` via `sdpa_kernel()` context

## Adapters (`pawn/adapters/`)

All adapters freeze the backbone and initialize to identity (zero-init or gamma=1, beta=0):
- `bottleneck.py` -- Houlsby-style down/up MLP, best parameter efficiency below ~1M params
- `lora.py` -- Low-rank attention projection adapters
- `film.py` -- Feature-wise Linear Modulation (lightest, ~17K params)
- `sparse.py` -- Random binary mask on frozen weights
- `hybrid.py` -- LoRA + FiLM combined

## Scripts (`scripts/`)

- `train.py` -- Pretrain from scratch (`--variant small|base|large|toy`)
- `train_all.py` -- Train small/base/large simultaneously on shared data batches. Supports `--run-evals` for automatic post-training probes, diagnostics, and Lichess eval, and `--publish-results` to push eval results to HF.
- `train_bottleneck.py`, `train_film.py`, `train_lora.py`, `train_sparse.py`, `train_hybrid.py` -- Adapter behavioral cloning on Lichess PGN
- `train_tiny.py` -- Standalone tiny transformer baseline (no frozen backbone)
- `eval_accuracy.py` -- MAIA-compatible evaluation (per-phase, per-ply accuracy)
- `eval_probes.py` -- Run linear probes on all checkpoints
- `export_hf_repo.py` -- Convert training run to HuggingFace repo format (safetensors + metrics)

All training scripts require `--hf-repo REPO` or `--local-checkpoints`.

## Deploy (`deploy/`)

Docker-based deployment to Runpod GPU VMs:
- `Dockerfile` -- Multi-target build: `interactive` (SSH+Jupyter, default) and `runner` (auto-stop)
- `entrypoint-run.sh` -- Runner entrypoint, pulls from HF via `PAWN_MODEL` env var
- `sync.sh` -- Pull latest checkpoints/metrics from HuggingFace submodules
- `pod.sh` -- Pod lifecycle (create/start/stop/delete/ssh)

Code lives at `/opt/pawn` on pods (outside the `/workspace` volume mount).

## Dashboard (`pawn/dashboard/`)

Solara-based training dashboard. Reads `metrics.jsonl` files, no dependency on training packages.

```bash
uv sync --extra dashboard
python -m pawn.dashboard --log-dir logs
```

- `metrics.py` -- Load runs, parse JSONL, detect run type from config record
- `charts.py` -- Plotly chart builders (loss, accuracy, LR, GPU, adapter-specific diagnostics)
- `sol.py` -- Solara components: RunSelector, ConfigSummary, MetricsCharts, Runner, Dashboard
- `__main__.py` -- CLI entry point, sets `PAWN_LOG_DIR` env var, launches `solara run`

Auto-detects run type from config fields (`run_type`, `formulation`, `pgn_file`). Dashboard requires restart for code changes (no hot reload).

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

## Logs

Training metrics in `logs/` (gitignored). Each run gets a timestamped directory with `metrics.jsonl`.

## Runpod Pod Management

### Setup

- Docker image: multi-target build in `Dockerfile`
  - `interactive` (default) — SSH + Jupyter, stays alive
  - `runner` — executes command then exits (pod auto-stops)
- Build: `docker build --target runner --build-arg GIT_HASH=$(git rev-parse HEAD) ...`

### Required Configuration

- **Always attach a network volume.** Checkpoints write to disk during atomic rename and HF push. Ephemeral container disk is lost on pod termination.
- **Set `HF_TOKEN` as a pod environment variable** for automatic HuggingFace authentication.
- Set `PAWN_MODEL=thomas-schweich/pawn-base` env var in the runner to auto-pull a checkpoint on startup.

### Lifecycle

- Create: `runpodctl pod create --name pawn-exp --gpu-id "NVIDIA RTX A5000" --image thomasschweich/pawn:<tag> --volume-in-gb 75 --ports "8888/http,22/tcp"`
- Stop: `runpodctl pod stop <ID>` — sends SIGTERM → trainer saves and pushes before exiting
- **Never `runpodctl pod delete` while training is running** — data loss risk
- Monitor: pull HF submodule (`deploy/sync.sh`) and read `metrics.jsonl`
