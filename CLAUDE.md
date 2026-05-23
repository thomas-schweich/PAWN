# PAWN (Playstyle-Agnostic World-model Network for Chess)

A causal transformer trained on random chess games, designed as a testbed for finetuning and augmentation methods at small scales. Apache 2.0.

The training stack is JAX/Equinox + Optax. The PyTorch implementation is removed from the training and eval surface; the only torch touchpoints are `pawn.torch_loader` (a thin loader for external non-JAX consumers) and `pawn._torch_legacy_fixture` (a frozen reference architecture used by the legacy-converter parity tests). `docs/jax-migration.md` is the canonical reference for the migration; this file is a quick-orientation map of what's shipped on the integration branch.

> **Published-checkpoint disclaimer.** The pre-trained weights on HuggingFace (`thomas-schweich/pawn-{small,base,large}`) and the metrics/cards that ship with them come from v1 PyTorch training runs. v2 JAX runs from the new supernet will be published to **new** HF repos (`pawn-{small,base,large}-v2`); the v1 repos stay frozen and remain reachable via `pawn.legacy.convert_legacy_checkpoint` for backward compatibility. The eval numbers in `docs/ACCURACY_CEILING.md` and the model cards are v1 numbers until the v2 publishing round produces fresh ones.

## Repository Structure

```
pawn/
├── engine/           # Rust chess engine with PyO3 bindings (via shakmaty)
├── pawn/             # Core Python package (JAX/Equinox)
│   ├── _sentinel.py             # Shared .complete sentinel helpers (stdlib only)
│   ├── _torch_legacy_fixture.py # Frozen torch arch for converter parity tests
│   ├── config.py                # ModelConfig, SUPERNET / TINY_SUPERNET, VARIANTS / TINY_VARIANTS, validate_nested
│   ├── model.py                 # PAWNModel transformer (RMSNorm, SwiGLU, RoPE, factored embeddings, stacked lax.scan layers)
│   ├── corpus.py                # Rust-engine corpus → trainer-shaped int32/bool arrays
│   ├── checkpoint.py            # Atomic save/load (.tmp → rename, .complete sentinel)
│   ├── legacy.py                # PyTorch → JAX checkpoint converter
│   ├── torch_loader.py          # Thin PyTorch loader for non-JAX consumers
│   ├── run_config.py            # Pydantic BaseRunConfig / PretrainConfig / AdapterConfig (single source of truth)
│   ├── logging.py               # MetricsLogger — JSONL metrics with type discriminator + psutil/GPU stats
│   ├── trainer.py               # Pretraining: cross-entropy, AdamW + warmup-cosine, K-step lax.scan, joint multi-variant loss
│   ├── adapter_trainer.py       # Two-tier frozen/trainable training; scan + eval
│   ├── adapters/                # 8 strategies: lora / film / unfreeze / bottleneck / hybrid / sparse / rosa / specialized_clm
│   ├── eval.py                  # Move-prediction accuracy + per-phase breakdown
│   ├── probes.py                # Linear probes on hidden states
│   ├── generation.py            # Generation diagnostics + KV-cache
│   ├── lichess_eval.py          # Elo-stratified Maia-style eval
│   ├── lichess_data.py          # Lichess parquet → JAX Corpus (Elo/ply filter + on-disk cache)
│   ├── sweep.py                 # Standalone Optuna driver (subprocess; AdapterObjective)
│   ├── wandb_utils.py           # OPTIONAL EXTRA: W&B metrics integration
│   ├── eval_suite/              # Legacy position-parquet pipeline (bounds, viz, corpus) — base deps
│   ├── dashboard/               # OPTIONAL EXTRA (`dashboard`): Solara training dashboard
│   └── lab/                     # OPTIONAL EXTRA (`lab`): FastMCP MCP server
├── scripts/          # CLI drivers (train_jax, train_jax_adapter, eval_jax, sweep, convert_published_checkpoints, data tools)
├── tests/            # JAX test suite (torch surface limited to torch_loader + legacy converter parity tests)
├── deploy/           # RunPod + vast.ai deployment scripts
└── docs/             # docs/jax-migration.md is authoritative
```

## Building

This is a uv workspace. The root project is the `pawn` Python package; `engine/` is the sole workspace member.

```bash
# Build the Rust chess engine (required before anything else)
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python deps. The base install ships CPU jaxlib. The
# rocm / cu128 extras add the torch dep used by the thin loader +
# legacy-converter parity tests, but they do NOT pull GPU jaxlib
# (see the comment in pyproject.toml above the extras — GPU jaxlib
# is installed manually after sync). The torch-loader extra adds
# only torch (useful if you want the thin loader without GPU jaxlib).
uv sync --extra rocm          # AMD (torch + triton-rocm)
uv sync --extra cu128         # NVIDIA (torch + cu128 index)
uv sync --extra torch-loader  # CPU jax + torch (thin loader only)

# Optional extras (compose as needed). Polars / matplotlib / seaborn
# / jinja2 / zstandard are in base — the Lichess parquet path,
# legacy eval_suite, and the data/publishing scripts all work out of
# a bare `uv sync --extra <gpu>`.
#   --extra dashboard   solara dashboard
#   --extra lab         pawn-lab MCP server (fastmcp + optuna-dashboard;
#                       optuna itself is base, shared with pawn.sweep)
#   --extra wandb       W&B integration

# Run the base + JAX test suite. The lab + dashboard tests need
# their matching extras.
uv run --extra rocm pytest tests/                # core + JAX + sweep + Lichess
uv run --extra rocm --extra lab --extra dashboard \
    --extra wandb pytest tests/                   # everything

# Pretrain the supernet on Rust-engine random games (verification scale)
uv run --extra rocm python scripts/train_jax.py \
    --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 --k 50

# Train a LoRA adapter on a frozen sliced backbone variant (v1 names restored)
uv run --extra rocm python scripts/train_jax_adapter.py \
    --supernet tiny --variant base --lora-rank 4 --total-steps 500

# Evaluate a converted JAX checkpoint
uv run --extra rocm python scripts/eval_jax.py \
    --checkpoint ~/.cache/huggingface/pawn-jax-converted/pawn-small
```

CPU jaxlib ships in the base `dependencies`; the `rocm` / `cu128` extras add torch (for the loader + legacy-converter parity fixture). GPU jaxlib is installed separately for now — see the `pyproject.toml` comment above the `rocm`/`cu128` extras for the manual step.

> **Migration-state note.** During the JAX migration the integration branch carries the v1 PyTorch surface alongside the JAX work. Sections S3-S12 replace the v1 modules one at a time; each section ends with its own tests green, but a full `pytest tests/` may have failing modules between sections. Run only the section's tests during an in-progress section; the green full suite is restored at S15.

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
- Sequence format: `[ply_1] ... [ply_N] [PAD] ... [PAD]` (configurable max length) — outcome prefix is optional via `prepend_outcome` flag

> **Legacy note.** Earlier versions of this codebase used a ~60k-entry move
> vocabulary and two separate parquet layouts ("v1" = pure-moves tokens with
> outcomes derived coarsely from the PGN `result` header, "v2" = tokens with
> the outcome prepended at position 0). **Both are gone.** The current code
> only knows about the 1,968-action vocabulary and the canonical parquet
> schema (pure-moves `tokens` + granular `outcome_token` column + per-game
> metadata). If you find a reference to a "v1 vocab", "v2 format",
> `_result_to_outcome`, `strip_outcome_token`, or `no_outcome_token`, it's
> a bug — those were all removed during the 0.x → stable transition. Legacy
> checkpoints trained against the old vocabulary are accessible only via
> the `pre-vocab-transition` git tag; they cannot be loaded or trained
> against from the current tree.

### Supernet + Variants

The JAX model is a single `PAWNModel` shaped by `pawn.config.ModelConfig`. The supernet hosts three nested slices (validated by `validate_nested`):

| Constant | d_model | n_layers | n_heads | d_ff | Notes |
|---|---|---|---|---|---|
| `SUPERNET` (`VARIANTS["large"]`) | 640 | 10 | 10 | 2560 | production supernet |
| `VARIANTS["base"]` | 512 | 8 | 8 | 2048 | nested slice |
| `VARIANTS["small"]` | 256 | 8 | 4 | 1024 | nested slice |
| `TINY_SUPERNET` (`TINY_VARIANTS["large"]`) | 192 | 4 | 3 | 768 | verification scale |
| `TINY_VARIANTS["base"]` | 128 | 3 | 2 | 512 | nested slice |
| `TINY_VARIANTS["small"]` | 64 | 2 | 1 | 256 | nested slice |

All current production variants share `head_dim = 64`. Legacy published checkpoints `pawn-{small,base,large}` predate the supernet — `pawn-large` uses `head_dim = 80` and is not a nested slice; the converter preserves its exact hyperparameters.

## Config + Logging — `pawn.run_config` and `pawn.logging`

These two modules are the load-bearing infrastructure for everything that follows.

### `pawn.run_config` (pydantic)

Single source of truth for every parameter that training scripts, sweep tooling, and the lab MCP server care about. `BaseRunConfig`, `PretrainConfig`, `AdapterConfig`, and `SpecializedCLMConfig` are pydantic BaseModels with `extra="forbid"` and cross-field `model_validator(mode="after")` invariants. JSON Schema is derived automatically — call `PretrainConfig.model_json_schema()` / `AdapterConfig.model_json_schema()` for introspection (used by the lab).

Both `scripts/train_jax.py` and `scripts/train_jax_adapter.py` accept `--config <json>`: the JSON is parsed through `TypeAdapter(PretrainConfig).validate_python(...).model_dump()` and the result drives the existing argparse code path. Bare-CLI invocation still works.

### `pawn.logging.MetricsLogger`

Every metric write goes through `MetricsLogger.log_config / log_train / log_val`. Each JSONL record carries `type: "config" | "train" | "val"` (discriminator the dashboard's train/val split depends on), `timestamp` (absolute), `elapsed`, `slug`, `hostname`, `git_hash`, plus host memory/CPU stats from `psutil` and GPU stats via shell-out to `nvidia-smi` / `rocm-smi`. NaN/Inf is sanitised to `None` so records stay RFC-7159 valid. Per-record flush (SIGKILL-durable). Slug-based run-dir naming (`run_20260520_140000_zesty-osprey`).

## Training

### Pretraining (`scripts/train_jax.py`)

Drives multi-variant joint training: every step computes loss on the supernet plus each nested slice and sums them (the §5.3 supernet signal).

```bash
# Verification run on the tiny supernet
uv run python scripts/train_jax.py \
    --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 --k 50

# JSON config (single source of truth — derived automatically from PretrainConfig)
uv run python scripts/train_jax.py --config configs/pretrain_supernet.json

# Larger run on the production SUPERNET. The 100K × B=256 × T=512 corpus
# is ~122 GiB on disk; override the 64 GiB safety guard explicitly so the
# trainer doesn't abort before generating data.
uv run python scripts/train_jax.py \
    --supernet supernet --total-steps 100000 --batch-size 256 --seq-len 512 --k 50 \
    --max-corpus-gb 128
```

Key args (`scripts/train_jax.py --help` for the full surface; pydantic-validated either way):
- `--config <json>` — load a JSON run config (single source of truth)
- `--supernet {tiny,supernet}` — which supernet config + variants to train
- `--total-steps N` — total training steps (must be a multiple of `--k`)
- `--batch-size B` — per-step batch size; chunk on device is `K × B`
- `--seq-len T` — sequence length; must be ≤ `supernet.max_seq_len`
- `--k K` — inner steps per `lax.scan` call (amortises JIT dispatch)
- `--lr`, `--warmup-steps` — AdamW + warmup-cosine schedule peak / warmup span
- `--seed` / `--corpus-seed` / `--model-seed` — RNG seeds
- `--max-corpus-gb` — abort upfront if the estimated Rust corpus footprint exceeds this
- `--logs-dir` — root dir for `metrics.jsonl` + `config.json` per run

The corpus is generated by the Rust engine each run. Run output lands in `logs/jax_run_<slug>/` and is auto-detectable by the dashboard.

### Adapter Training (`scripts/train_jax_adapter.py`)

Adapter training driver. Loads the supernet, slices to a named variant, wraps with the chosen adapter strategy, and runs the two-tier frozen-backbone / trainable-adapter optimisation under a K-step `lax.scan`. The driver dispatches all eight strategies (`lora` / `film` / `unfreeze` / `bottleneck` / `hybrid` / `sparse` / `rosa` / `specialized_clm`) via the `--strategy` flag; RoSA additionally carries the three-phase training schedule (LoRA warmup → gradient-magnitude mask gen → joint training) controlled by `--rosa-warmup-frac` / `--rosa-top-k-frac`.

**Data source.** Two paths. The realistic adapter task — human-move prediction at a target Elo — is `--pgn <hf-repo|path>`: the canonical pre-tokenized Lichess parquet (`thomas-schweich/pawn-lichess-full`), optionally narrowed to an Elo band via `--elo-min` / `--elo-max`. `pawn.lichess_data` filters + packs it into a `Corpus` and caches the result under `$HF_HOME/pawn-lichess-cache/<key>/`.

**Train / val split.** By default the trainer reads val from a separate held-out `validation` split (`--pgn-val-split validation`, the HF dataset's layout) — no leakage from the train pool. The train pool is then tiled across epochs (per-epoch permutation) to fill `total_steps × batch_size` game-slots. Pass `--pgn-val-split ""` to disable the held-out split and carve val out of train instead — the right choice for a single-file local source with no split structure.

Omitting `--pgn` falls back to Rust-engine random games — a verification proxy, not a real adapter benchmark.

```bash
# LoRA on a real Lichess Elo band (1800-2000) — the realistic task
uv run python scripts/train_jax_adapter.py \
    --supernet tiny --variant base --lora-rank 4 --total-steps 500 \
    --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 2000

# LoRA on the random-game proxy (no --pgn) — quick verification
uv run python scripts/train_jax_adapter.py \
    --supernet tiny --variant base --lora-rank 4 --total-steps 500

# RoSA three-phase
uv run python scripts/train_jax_adapter.py --strategy rosa \
    --supernet tiny --variant base --lora-rank 4 --total-steps 1000 \
    --rosa-warmup-frac 0.4 --rosa-top-k-frac 0.01

# specialized_clm (from-scratch; no backbone) — v1 names inside the strategy config
uv run python scripts/train_jax_adapter.py --strategy specialized_clm \
    --d-model 64 --n-layers 2 --n-heads 2 --d-ff 128 --total-steps 1000

# JSON config
uv run python scripts/train_jax_adapter.py --config configs/adapter_lora.json
```

Key args (v1 names restored):
- `--strategy {lora,film,unfreeze,bottleneck,hybrid,sparse,rosa,specialized_clm}` — adapter strategy
- `--supernet {tiny,supernet}`, `--variant` — backbone selection (ignored for `specialized_clm`)
- `--lora-rank`, `--lora-alpha` — shared by LoRA / Hybrid / RoSA (rank of the low-rank update)
- `--lora-targets {q,k,v,o ...}` — LoRA / Hybrid target sublayers
- `--rosa-targets {q,k,v,o ...}`, `--rosa-warmup-frac`, `--rosa-top-k-frac` — RoSA target sublayers + 3-phase schedule + sparsity
- `--use-output-film` / `--no-use-output-film` — FiLM / Hybrid: γ⊙logits+β post-`lm_head` modulation (default: enabled; per-layer γ⊙h+β stays on either way)
- `--n-unfreeze`, `--include-lm-head`, `--include-embeddings` — Unfreeze
- `--bottleneck-dim`, `--bottleneck-n-hidden`, `--no-adapt-attn`, `--no-adapt-ffn` — Bottleneck
- `--sparse-targets {q,k,v,o ...}`, `--density`, `--sparse-hard` — Sparse
- `--d-model`, `--n-layers`, `--n-heads`, `--d-ff` — SpecializedCLM (no `--specialized-` prefix; the values live inside `SpecializedCLMConfig`)
- `--pgn`, `--elo-min`, `--elo-max`, `--min-ply`, `--max-games`, `--pgn-split`, `--pgn-val-split`, `--cache-dir` — Lichess data source. Default `--pgn-val-split=validation` reads val from the held-out HF split; pass `""` to carve from train. Omit `--pgn` for the random-game proxy.
- `--total-steps`, `--batch-size`, `--seq-len`, `--k`, `--lr`, `--warmup-steps`
- `--val-frac`, `--val-every` — held-out validation slice + frequency

The Lichess Elo-stratified *eval* (move-prediction accuracy on held-out human games) is at `pawn.lichess_eval`; the Lichess *training-data* path (`pawn.lichess_data`) feeds the adapter trainer above.

The two-tier optimisation partitions the PyTree via `eqx.partition(model, adapter_filter(model))`. Gradients for the frozen backbone are dropped by XLA dead-code elimination (~33% FLOP cut on the backward pass). The structural invariant — every array field of `state.trainable.backbone` is `None` after partitioning (the `PAWNModel` object itself stays, but its leaves are sentinel `None`s) — is pinned by `test_backbone_weights_are_frozen`.

## Evaluation

### Move-accuracy + per-phase breakdown (`scripts/eval_jax.py`)

Loads a converted JAX checkpoint (or a freshly-initialised model for verification) and reports overall + per-phase accuracy on a Rust-engine corpus.

```bash
# Convert published PyTorch checkpoints once
uv run python scripts/convert_published_checkpoints.py

# Evaluate
uv run python scripts/eval_jax.py \
    --checkpoint ~/.cache/huggingface/pawn-jax-converted/pawn-small
```

Argmax is restricted to `[0, NUM_ACTIONS)` so PAD + outcome tokens cannot leak (pinned by `test_argmax_restricted_to_action_band`).

### Probes + generation + Lichess eval

- `scripts/eval_probes_jax.py` — linear probes on frozen hidden states via Optax fit.
- `scripts/eval_generation_jax.py` — generation diagnostics (outcome-signal, prefix-continuation, poisoned-prefix, impossible-task, improbable-task). All five share the `outcome_prefix_trained` gate so they skip cleanly when the model wasn't trained that way.
- `scripts/eval_vs_stockfish.py` — Elo-stratified Maia-style accuracy via `pawn.lichess_eval`.

### Legacy position-parquet pipeline (`pawn.eval_suite`)

Off-the-hot-path tooling: theoretical accuracy bounds (`bounds.py`), matplotlib/seaborn plotting (`viz.py`), polars parquet iterator (`corpus.py`). polars / matplotlib / seaborn moved to base deps in S18, so the JAX surface installs them automatically; nothing here is gated.

## Sweeps (`pawn.sweep` + `scripts/sweep.py`)

Standalone Optuna driver. `AdapterObjective` builds CLI argv from a trial's params, runs `scripts/train_jax_adapter.py` as a subprocess, parses the resulting `metrics.jsonl`, and returns `val_loss` to Optuna. `MedianPruner` / `HyperbandPruner` hookups via `trial.report(val_loss, step) + should_prune()`. Persistent study state via SQLite.

```bash
uv run python scripts/sweep.py --strategy lora --n-trials 50 \
    --storage sqlite:///sweeps/lora.db
```

The lab MCP server (`pawn.lab`, `--extra lab`) drives a different code path: it uses Optuna's `ask()` to produce candidate suggestions on demand via `lab_results` rather than calling `study.optimize()`. Both modes spawn `scripts/train_jax_adapter.py` as subprocesses; they don't conflict.

## Checkpoints

Pre-trained v1 weights are hosted on HuggingFace:
- `thomas-schweich/pawn-small` — 9.5M params (d=256, 8 layers, 4 heads)
- `thomas-schweich/pawn-base` — 35.8M params (d=512, 8 layers, 8 heads)
- `thomas-schweich/pawn-large` — 68.4M params (d=640, 10 layers, 8 heads, head_dim=80)

Convert them once via `scripts/convert_published_checkpoints.py`; results land in `$HF_HOME/pawn-jax-converted/<variant>/` and are reusable across runs. v2 supernet-derived checkpoints will be published to new repos (`pawn-{small,base,large}-v2`); the v1 repos stay frozen for backward compatibility.

### Checkpoint Format (safetensors)

JAX checkpoints are directories:
```
step_00065000/
├── model.safetensors   # one fp32 tensor per PAWNModel array field, declaration order
├── config.json         # {format_version, model_config}
└── .complete           # integrity sentinel — JSON {format_version: 1, files: {name: sha256-hex}}
```

Central module: `pawn/checkpoint.py`. All save/load goes through this module. Sentinel helpers (`sha256_file`, `write_sentinel`, `verify_sentinel`, `IncompleteCheckpointError`, `CheckpointIntegrityError`) live in `pawn/_sentinel.py` — stdlib-only so the thin PyTorch loader can import them without pulling JAX. In practice only `verify_sentinel` is shared between `pawn.checkpoint` and `pawn.torch_loader`; `pawn.checkpoint` keeps a private `_write_sentinel` that bakes a `format_version` field into the JSON alongside the file-hash map. The exceptions + `sha256_file` come from `pawn._sentinel` so the schema (`{"files": {...}}`) and integrity rules stay in one place.

### Data Integrity

**Every checkpoint write is atomic**: files are written to a `.tmp` sibling directory; once the `.complete` sentinel is written, any existing checkpoint is renamed aside to `.bak`, the new directory is renamed into place, then `.bak` is removed. An interrupted overwrite always leaves a recoverable checkpoint on disk.

The `.complete` sentinel contains SHA-256 hashes of every other file in the checkpoint. **Hashes are always verified on load — no exceptions.**

- `IncompleteCheckpointError` — raised when `.complete` sentinel is missing
- `CheckpointIntegrityError` — raised when any hash mismatches or the file-set diverges

**Never rsync checkpoint files from running pods.** Use the published HF repos or the JAX-converted local cache.

## Logs

Per-run output lands in `logs/` (gitignored). The slug prefix + per-run files depend on the driver:
- `jax_run_<YYYYMMDD>_<HHMMSS>_<microseconds>_<slug>/` — `scripts/train_jax.py` (pretraining). Writes `metrics.jsonl` (one JSON record per scan chunk, plus a `type: "config"` baseline at start) and `config.json` (every `ModelConfig` field for the supernet + every variant, the resolved `PretrainConfig`, and the run seeds).
- `jax_adapter_run_<...>/` — `scripts/train_jax_adapter.py`. Writes `metrics.jsonl` (per-chunk train rows + separate `type: "val"` rows on val chunks) and `config.json` (supernet + selected variant cfg + the resolved `AdapterConfig`).
- `jax_eval_run_<...>/` — `scripts/eval_jax.py`. Writes a single-shot `eval_result.json` with overall + per-phase accuracy; no `metrics.jsonl`.

## Cloud GPU Operations

PAWN can run on either RunPod or vast.ai. The same Docker image works on both — pick the provider that has the GPU you want at the price you want.

| | RunPod | vast.ai |
|---|---|---|
| Manager script | `deploy/pod.sh` | `deploy/vast.sh` |
| CLI | `runpodctl` | `vastai` (or `uvx vastai`) |
| Local config dir | `~/.config/pawn/pods/` | `~/.config/pawn/vast/` |
| Volume model | Network volume mounted at `/workspace` | Single instance disk (use `--disk N`) |
| Pricing | Fixed per-GPU rates | Marketplace; pass `--max-price` and/or `--interruptible` |

Both share the same Docker image (`thomasschweich/pawn:latest`) and entrypoint. `vast.sh` mirrors `pod.sh`'s command surface (`create / start / stop / delete / ssh / list / status / setup / deploy / launch`) plus a `search` subcommand.

### Docker Image

Docker images are **automatically built and pushed to Docker Hub by CI** on every merge to main. No manual builds needed.

| Tag | Target | Base | GPU |
|-----|--------|------|-----|
| `thomasschweich/pawn:latest` | `runtime` | `python:3.12-slim` | CUDA (cu128 wheels bundle runtime) |
| `thomasschweich/pawn:dev` | `dev` | `python:3.12-slim` | CUDA + Claude Code + tmux |
| `thomasschweich/pawn:rocm` | `runtime-rocm` | `python:3.12-slim` | ROCm 7.1 (wheel bundles runtime) |
| `thomasschweich/pawn:dev-rocm` | `dev-rocm` | `python:3.12-slim` | ROCm 7.1 + Claude Code + tmux |

The runtime images install the base deps + GPU torch only; polars / matplotlib / seaborn / jinja2 / zstandard are in base since S18, so the published image runs `extract_lichess_parquet`, `compute_theoretical_ceiling`, `generate_model_cards`, the legacy `pawn.eval_suite` pipeline, and the `--pgn` adapter path without a manual sync. Dashboard / lab / wandb extras are not in the runtime image; the `:dev` tags include them.

Code lives at `/opt/pawn` on all images. SSH in and run experiments directly.

### Pod Lifecycle (RunPod)

```bash
bash deploy/pod.sh create myexp --gpu h100
bash deploy/pod.sh ssh myexp
# The wrapper appends --logs-dir logs automatically; do not pass it yourself.
bash deploy/pod.sh launch myexp \
    scripts/train_jax.py --supernet supernet --total-steps 100000 \
    --batch-size 256 --seq-len 512 --k 50 --max-corpus-gb 128
bash deploy/pod.sh stop myexp   # preserves volume, stops billing
bash deploy/pod.sh delete myexp # destroys everything
```

GPU shortcuts: `a5000`, `a40`, `a6000`, `4090`, `5090`, `l40s`, `a100`, `a100-pcie`, `a100-sxm`, `h100`, `h200`. Pod configs are cached in `~/.config/pawn/pods/<name>.env`.

### Instance Lifecycle (vast.ai)

```bash
bash deploy/vast.sh search --gpu 4090 --max-price 0.5
bash deploy/vast.sh create myexp --gpu 4090 --max-price 0.5
bash deploy/vast.sh deploy myexp   # rsync local checkout to /workspace/pawn
bash deploy/vast.sh ssh myexp
# The wrapper appends --logs-dir logs automatically; do not pass it yourself.
bash deploy/vast.sh launch myexp \
    scripts/train_jax.py --supernet supernet --total-steps 100000 \
    --batch-size 256 --seq-len 512 --k 50 --max-corpus-gb 128
bash deploy/vast.sh stop myexp
```

Vast.ai has no separate network volume — instance disk is sized via `--disk` (default 100 GB) and persists across stop/start. `delete` destroys the disk. `HF_TOKEN` and `PUBLIC_KEY` from your local environment are passed through to the instance at create time.

### Required Instance Configuration

- **Persistent storage.** On RunPod, attach a network volume (mounted at `/workspace`). On vast.ai, pick a disk size with `--disk` that comfortably holds checkpoints.
- **Set `HF_TOKEN` as an environment variable** for automatic HuggingFace authentication.
- `PAWN_MODEL=thomas-schweich/pawn-base` — auto-pull a published checkpoint on startup.
- `PAWN_CMD` — training command to execute (alternative to Docker CMD args).

### Instance Safety

- `bash deploy/pod.sh stop <name>` / `bash deploy/vast.sh stop <name>` halts billing. Whether stop triggers a graceful trainer shutdown depends on whether the SIGTERM handler is wired in `pawn.trainer` / `pawn.adapter_trainer` (still TODO on the JAX side; the v1 PyTorch handler ported alongside HF-backed checkpoint pushing).
- **Never delete/destroy an instance while training is running** — data loss risk on either provider.
- **Never rsync checkpoint files from running instances** — load via the published HF repos.
- On vast.ai with `--interruptible`, the host can preempt you at any time.

## Key Patterns & Gotchas

- **Single-framework promise.** The training and eval surface is JAX-only. The two torch touchpoints — `pawn.torch_loader` and `pawn._torch_legacy_fixture` — exist precisely so the JAX surface stays torch-free under a CPU-jax install. Don't reintroduce torch dependencies into `pawn.{model,trainer,adapter_trainer,adapters,eval,probes,generation,lichess_eval,checkpoint,run_config,logging,sweep}`.
- **`pawn.run_config` is the single source of truth.** Don't scatter `raise SystemExit(...)` validation across script bodies — encode the invariant as a `@model_validator(mode="after")` in the relevant pydantic config. This is the lesson from the first JAX-migration attempt (#111) that we explicitly reverted; see `docs/jax-migration.md` §8.1.
- **`pawn.logging.MetricsLogger` is the only metric writer.** Both trainers go through `log_config / log_train / log_val`. Don't open `metrics.jsonl` directly — losing the `type` discriminator silently breaks the dashboard's train/val split.
- **v1 field names are canonical.** `lora_rank` (not `rank`), `density` (not `sparse_density`), `use_output_film` (not `--no-film-output` polarity-flipped), `no_adapt_attn` / `no_adapt_ffn` (not the `bottleneck_` prefix), and `d_model`/`n_layers`/`n_heads` inside `SpecializedCLMConfig` (no `specialized_` prefix). The migration restored these after the first attempt cosmetically renamed them; see `docs/jax-migration.md` §8.3.
- **Sentinel logic lives in `pawn._sentinel`.** Stdlib-only. Both `pawn.checkpoint` and `pawn.torch_loader` import `verify_sentinel` from it (aliased to a local underscore name on both sides; the public API on the module itself has no underscores). `pawn.checkpoint` *also* has its own private `_write_sentinel` that adds a `format_version` field — `pawn._sentinel.write_sentinel` writes a strict `{"files": ...}` payload and that asymmetry is intentional. Don't duplicate the SHA-256 / verify logic elsewhere.
- **Two-tier PyTree partition** is how adapter training freezes the backbone: `eqx.partition(model, adapter_filter(model))` produces a trainable subtree (adapters) and a frozen subtree (backbone). XLA dead-code-eliminates the backbone weight gradients, which is the source of the ~33% FLOP cut on the backward pass.
- **K-step `lax.scan` amortises JIT dispatch.** The pretraining + adapter trainers both run `K` inner steps per scan invocation. `K * B` games are consumed per scan call; size `K` so per-step throughput is roughly host-bound rather than launch-bound.
- **`state.step` must be a JAX scalar inside `jit`.** Storing it as a Python int caused a ~70× slowdown during Phase-2 development by retriggering recompilation every step. The fix is pinned by tests; keep it that way.
- **`optax.warmup_cosine_decay_schedule`'s `decay_steps` is the end-to-end length, not the post-warmup tail.** A `decay_steps = total_steps - warmup` would double-subtract warmup. The trainer passes `decay_steps = total_steps`.
- **Padded-batch AdamW weight-decay drift.** Without a `lax.cond` guard, weight decay still fires on a padded final chunk and erodes the parameters; the trainer guards against it.
- **Gradient clipping** is `optax.chain(optax.clip_by_global_norm(1.0), adamw)`. Matches the v1 PyTorch contract.
- **Factored embeddings**: each move token decomposes into `src_embed[s] + dst_embed[d] + promo_embed[p]`, shrinking the move-embedding table from `1968 × d_model` to `(64 + 64 + 5) × d_model` — ~14.8× fewer params on that table (input side only; `lm_head` is still a full `d_model → vocab` projection).
- **Legacy converter rejects pre-vocab-transition checkpoints.** The current JAX model uses the 1,968-action `decomp_table`; converting an older ~60k-token PyTorch checkpoint would silently embed every move incorrectly. See `pawn.legacy.convert_legacy_checkpoint` and the `pre-vocab-transition` git tag.
- **fp32 cross-framework parity bar is `1e-3`, not `1e-4`.** The toy-config converter test gets `~2.4e-7` because of its small accumulation budget; the three published checkpoints land at mean `|Δlogit| ≈ 5×10⁻⁶` (max around `1e-4`). Don't tighten the published-checkpoint tolerance.
- **stockfish-datagen 50-move rule is eval-strategic, not unconditional.** At halfmove 100 (the FIDE-claimable threshold), the side about to move claims iff Stockfish's top-candidate `score_cp` (side-to-move POV) is `< 0`. Winning/even sides keep playing; losing sides claim. The 75-move *automatic* rule (halfmove 150) is the hard upper bound, fires regardless of eval. Means the dataset has 50-move-rule draws scattered across halfmoves 100–150 (correlated with eval), giving the model a learnable signal for *when* to claim rather than baking in "halfmove 100 → game over." 3-fold repetition stays unconditional. See `stockfish-datagen/src/game.rs` (`detect_pre_eval_terminal` + `should_strategic_claim_50mv`).
- **stockfish-datagen worker pinning shifts the n_workers sweet spot down.** Each (worker, Stockfish) pair is pinned to `worker_id % n_logical` on Linux so they share L1/L2. Rule of thumb: **`n_workers = total_threads − threads_per_core`** — fully occupy every physical core except one, and leave that one core entirely free for the kernel + watcher thread + HF upload networking + parquet I/O. For typical vast.ai 128-thread / 64-physical / 2-SMT pods that's 126 workers; the bundled `stockfish-datagen/examples/stockfish_100m.json` config defaults to that.

  As of the shard-id partitioning refactor, `cfg.n_workers` is operational-only — workers pull from a shared atomic shard-id counter, so changing it between runs never changes any game's content (the per-game seed is `mix(tier_seed, global_game_index)` and tier seeds are keyed by sha256(`tier.name`), not by index). Multi-pod cooperation uses `--shard-id-range A:B` on the `stockfish-datagen run` subcommand; each pod writes per-pod sentinels with the range zero-padded to 6 digits (`_tier_state-s000000-s005000.json`, `_manifest-s000000-s005000.json`) and `scripts/datagen_reconcile_tier.py` merges them into a canonical `_manifest.json` post-run. The orchestrator (`scripts/datagen_with_hf_sync.py`) commits shards via batched `huggingface_hub.HfApi.create_commit` calls with one `CommitOperationAdd` per shard (~1 commit per tier per cycle), staying well under HF's 128 commits/hour limit.

  NUMA: `stockfish-datagen` calls `set_mempolicy(MPOL_INTERLEAVE)` at startup (Linux only). Children inherit across `execve`, so stockfish workers first-touch their NNUE page-cache fills under interleave policy. No-op on single-socket pods.

  Stockfish binary: the `:datagen` Docker image JIT-builds the patched stockfish for the host CPU on first launch (`scripts/build_stockfish_for_host.sh`) and caches it under `/workspace/.cache/stockfish/`. One image tag covers every supported microarch (vnni512 / avx512 / avx2 / modern).
