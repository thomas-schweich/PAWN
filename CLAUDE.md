# PAWN (Playstyle-Agnostic World-model Network for Chess)

A causal transformer trained on random chess games, designed as a testbed for finetuning and augmentation methods at small scales. Apache 2.0.

The training stack is JAX/Equinox + Optax. The legacy PyTorch implementation was removed in Phase 4 of the JAX migration. `docs/jax-migration.md` is the canonical reference for everything below; this file is a quick-orientation map of what's actually shipped on the integration branch.

## Repository Structure

```
pawn/
├── engine/           # Rust chess engine with PyO3 bindings (via shakmaty)
├── pawn/             # Core Python package (JAX/Equinox)
│   ├── config.py     # ModelConfig, SUPERNET / TINY_SUPERNET, VARIANTS / TINY_VARIANTS, validate_nested
│   ├── model.py      # PAWNModel transformer (RMSNorm, SwiGLU, RoPE, factored embeddings, stacked lax.scan layers)
│   ├── corpus.py     # Rust-engine corpus → trainer-shaped int32/bool arrays
│   ├── trainer.py    # Pretraining: cross-entropy, AdamW + warmup-cosine, K-step lax.scan, joint multi-variant loss
│   ├── adapter_trainer.py  # Two-tier frozen/trainable training; scan + eval
│   ├── adapters/     # LoRA (only adapter strategy ported so far)
│   ├── eval.py       # Move-prediction accuracy + per-phase breakdown
│   ├── checkpoint.py # Atomic save/load (.tmp → rename, .complete sentinel)
│   ├── legacy.py     # PyTorch → JAX checkpoint converter
│   ├── torch_loader.py        # Thin PyTorch loader for non-JAX consumers
│   ├── _sentinel.py           # Shared .complete sentinel helpers (stdlib only)
│   └── _torch_legacy_fixture.py  # Test fixture: legacy PyTorch architecture for converter-parity tests
├── scripts/          # CLI drivers (train_jax, train_jax_adapter, eval_jax, convert_published_checkpoints)
├── tests/            # JAX test suite (torch surface limited to torch_loader + legacy converter parity tests)
├── deploy/           # RunPod + vast.ai deployment scripts
└── docs/             # docs/jax-migration.md is authoritative
```

## Building

This is a uv workspace. The root project is the `pawn` Python package; `engine/` is the sole workspace member.

```bash
# Build the Rust chess engine (required before anything else)
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python deps. The base install ships CPU jaxlib; the rocm / cu128
# extras add the GPU jaxlib AND the torch dep used by the thin loader +
# legacy-converter parity tests. The torch-loader extra adds only torch
# (useful if you want the thin loader without GPU jaxlib).
uv sync --extra rocm        # AMD (ROCm 7.1)
uv sync --extra cu128       # NVIDIA (CUDA 12.8)
uv sync --extra torch-loader  # CPU jax + torch (thin loader only)

# Run tests
uv run --extra rocm pytest tests/

# Pretrain the supernet on Rust-engine random games (verification scale)
uv run --extra rocm python scripts/train_jax.py \
    --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 --k 50

# Train a LoRA adapter on a frozen sliced backbone variant
uv run --extra rocm python scripts/train_jax_adapter.py \
    --supernet tiny --variant base --rank 4 --total-steps 500

# Evaluate a converted JAX checkpoint
uv run --extra rocm python scripts/eval_jax.py \
    --checkpoint ~/.cache/huggingface/pawn-jax-converted/pawn-small
```

CPU jaxlib ships in the base `dependencies`; the `rocm` / `cu128` extras add GPU jax + torch. The torch dep is optional — the JAX training and eval surface installs without it.

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

## Training

### Pretraining (`scripts/train_jax.py`)

Drives multi-variant joint training: every step computes loss on the supernet plus each nested slice and sums them (the §5.3 supernet signal). Phase-2 verification entry point — production HF-backed checkpoints, resume, and W&B integration come in a later phase.

```bash
# Verification run on the tiny supernet
uv run python scripts/train_jax.py \
    --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 --k 50

# Larger run on the production SUPERNET. The 100K × B=256 × T=512 corpus
# is ~122 GiB on disk; override the 64 GiB safety guard explicitly so the
# trainer doesn't abort before generating data.
uv run python scripts/train_jax.py \
    --supernet supernet --total-steps 100000 --batch-size 256 --seq-len 512 --k 50 \
    --max-corpus-gb 128
```

Key args (`scripts/train_jax.py --help` for the full surface):
- `--supernet {tiny,supernet}` — which supernet config + variants to train
- `--total-steps N` — total training steps (must be a multiple of `--k`)
- `--batch-size B` — per-step batch size; chunk on device is `K × B`
- `--seq-len T` — sequence length; must be ≤ `supernet.max_seq_len`
- `--k K` — inner steps per `lax.scan` call (amortises JIT dispatch)
- `--lr`, `--warmup-steps` — AdamW + warmup-cosine schedule peak / warmup span
- `--seed` / `--corpus-seed` / `--model-seed` — RNG seeds
- `--max-corpus-gb` — abort upfront if the estimated Rust corpus footprint exceeds this
- `--logs-dir` — root dir for `metrics.jsonl` + `config.json` per run

The corpus is generated by the Rust engine each run (no cache reuse path yet). Run output lands in `logs/jax_run_<timestamp>_<pid>/`.

### Adapter Training (`scripts/train_jax_adapter.py`)

Phase-3 driver. Loads the supernet, slices to a named variant, wraps with the only currently-ported adapter strategy (LoRA), and runs the two-tier frozen-backbone / trainable-adapter optimisation under a K-step `lax.scan`.

```bash
# Tiny verification run
uv run python scripts/train_jax_adapter.py \
    --supernet tiny --variant base --rank 4 --total-steps 500
```

Key args:
- `--supernet {tiny,supernet}`, `--variant` — backbone selection
- `--rank`, `--lora-targets {q,k,v,o ...}`, `--lora-alpha` — LoRA shape
- `--total-steps`, `--batch-size`, `--seq-len`, `--k`, `--lr`, `--warmup-steps`
- `--val-frac`, `--val-every` — held-out validation slice + frequency

The verification proxy is a Rust-engine random-game corpus; Lichess Elo-slice cache + the broader adapter strategies (bottleneck / FiLM / hybrid / sparse / unfreeze) port in follow-up PRs onto the same two-tier trainer. The broader strategies live in Phase 3 of the migration plan (Phase 3 = "Adapters. Two-tier PyTree, finite-dataset data path, …"); the Lichess data cache lands with Phase 4's eval-suite port. See `docs/jax-migration.md` §9.

The two-tier optimisation partitions the PyTree via `eqx.partition(model, adapter_filter(model))`. Gradients for the frozen backbone are dropped by XLA dead-code elimination (~33% FLOP cut on the backward pass). The structural invariant — every array field of `state.trainable.backbone` is `None` after partitioning (the `PAWNModel` object itself stays, but its leaves are sentinel `None`s) — is pinned by `test_backbone_weights_are_frozen`.

## Evaluation (`scripts/eval_jax.py`)

Move-prediction accuracy + per-phase breakdown. Loads a converted JAX checkpoint (or a freshly-initialised model for verification) and reports overall + per-phase accuracy on a Rust-engine corpus.

```bash
# Convert published PyTorch checkpoints once
uv run python scripts/convert_published_checkpoints.py

# Evaluate
uv run python scripts/eval_jax.py \
    --checkpoint ~/.cache/huggingface/pawn-jax-converted/pawn-small
```

Argmax is restricted to `[0, NUM_ACTIONS)` so PAD + outcome tokens cannot leak (pinned by `test_argmax_restricted_to_action_band`).

The rest of the legacy `pawn.eval_suite` surface — linear probes on hidden states, generation diagnostics (outcome-signal / prefix-continuation / poisoned-prefix / impossible-task), Lichess Elo-stratified eval — ports incrementally onto this entry point in follow-up PRs. See `docs/jax-migration.md` Phase 4.

## Checkpoints

Pre-trained weights are hosted on HuggingFace and ship in the legacy PyTorch format:
- `thomas-schweich/pawn-small` — 9.5M params (d=256, 8 layers, 4 heads)
- `thomas-schweich/pawn-base` — 35.8M params (d=512, 8 layers, 8 heads)
- `thomas-schweich/pawn-large` — 68.4M params (d=640, 10 layers, 8 heads, head_dim=80)

Convert them once via `scripts/convert_published_checkpoints.py`; results land in `$HF_HOME/pawn-jax-converted/<variant>/` and are reusable across runs.

### Checkpoint Format (safetensors)

JAX checkpoints are directories:
```
step_00065000/
├── model.safetensors   # one fp32 tensor per PAWNModel array field, declaration order
├── config.json         # {format_version, model_config}
└── .complete           # integrity sentinel — JSON {files: {name: sha256-hex}}
```

Central module: `pawn/checkpoint.py`. All save/load goes through this module. Sentinel helpers (`sha256_file`, `write_sentinel`, `verify_sentinel`, `IncompleteCheckpointError`, `CheckpointIntegrityError`) live in `pawn/_sentinel.py` — stdlib-only so the thin PyTorch loader can import them without pulling JAX. In practice only `verify_sentinel` is shared between `pawn.checkpoint` and `pawn.torch_loader`; `pawn.checkpoint` keeps a private `_write_sentinel` that bakes a `format_version` field into the JSON alongside the file-hash map, so it does not call `pawn._sentinel.write_sentinel` directly. The exceptions + `sha256_file` come from `pawn._sentinel` so the schema (`{"files": {...}}`) and integrity rules stay in one place.

### Data Integrity

**Every checkpoint write is atomic**: files are written to a `.tmp` sibling directory; once the `.complete` sentinel is written, any existing checkpoint is renamed aside to `.bak`, the new directory is renamed into place, then `.bak` is removed. An interrupted overwrite always leaves a recoverable checkpoint on disk.

The `.complete` sentinel contains SHA-256 hashes of every other file in the checkpoint. **Hashes are always verified on load — no exceptions.**

- `IncompleteCheckpointError` — raised when `.complete` sentinel is missing
- `CheckpointIntegrityError` — raised when any hash mismatches or the file-set diverges

**Signal handling is not yet wired in the current JAX drivers.** Phase-2 / Phase-3 entry points (`scripts/train_jax.py`, `scripts/train_jax_adapter.py`) install no SIGTERM handler — the legacy trainer's "save + push + exit" graceful-shutdown loop ports in a later phase along with HF-backed checkpoint pushing. Until then, killing the driver mid-step will drop any in-flight chunk's metrics row and any unsaved model state.

**Never rsync checkpoint files from running pods.** Use the published HF repos or the JAX-converted local cache.

## Logs

Training metrics in `logs/` (gitignored). Each run gets a timestamped directory with `metrics.jsonl` and a per-pid suffix (e.g., `jax_run_20260520_140000_123456_<pid>/`).

The trainer writes one JSON record per scan chunk. The config record at the top of the file includes every `ModelConfig` field for both the supernet and every variant, the run seed, and the resolved CLI flags.

## Cloud GPU Operations

PAWN can run on either RunPod or vast.ai. The same Docker image works on both — pick the provider that has the GPU you want at the price you want.

| | RunPod | vast.ai |
|---|---|---|
| Manager script | `deploy/pod.sh` | `deploy/vast.sh` |
| CLI | `runpodctl` | `vastai` (or `uvx vastai`) |
| Local config dir | `~/.config/pawn/pods/` | `~/.config/pawn/vast/` |
| Volume model | Network volume mounted at `/workspace` | Single instance disk (use `--disk N`) |
| Pricing | Fixed per-GPU rates | Marketplace; pass `--max-price` and/or `--interruptible` |

Both share the same Docker image (`thomasschweich/pawn:latest`) and entrypoint (which honors the `PUBLIC_KEY` env var that both providers set). `vast.sh` mirrors `pod.sh`'s command surface (`create / start / stop / delete / ssh / list / status / setup / deploy / launch`) plus a `search` subcommand for browsing offers before committing.

### Docker Image

Docker images are **automatically built and pushed to Docker Hub by CI** on every merge to main. No manual builds needed.

| Tag | Target | Base | GPU |
|-----|--------|------|-----|
| `thomasschweich/pawn:latest` | `runtime` | `python:3.12-slim` | CUDA (cu128 wheels bundle runtime) |
| `thomasschweich/pawn:dev` | `dev` | `python:3.12-slim` | CUDA + Claude Code + tmux |
| `thomasschweich/pawn:rocm` | `runtime-rocm` | `python:3.12-slim` | ROCm 7.1 (wheel bundles runtime) |
| `thomasschweich/pawn:dev-rocm` | `dev-rocm` | `python:3.12-slim` | ROCm 7.1 + Claude Code + tmux |

All images use `python:3.12-slim`. The cu128 / ROCm extras bundle the GPU runtime inside the wheel; the only host requirement is the GPU kernel driver.

Code lives at `/opt/pawn` on all images. SSH in and run experiments directly.

To build locally (rarely needed):
```bash
docker build --platform linux/amd64 --target runtime \
    --build-arg GIT_HASH=$(git rev-parse HEAD) \
    -t thomasschweich/pawn:latest .
```

### Pod Lifecycle (RunPod)

```bash
bash deploy/pod.sh create myexp --gpu h100
bash deploy/pod.sh ssh myexp
# Run training directly over ssh (the `pod.sh launch` wrapper currently
# appends `--log-dir` for legacy compatibility; JAX scripts use `--logs-dir`,
# so launch via ssh until the wrapper is rewired):
ssh root@<pod-host> "cd /workspace/pawn && nohup uv run python \
    scripts/train_jax.py --supernet supernet --total-steps 100000 \
    --batch-size 256 --seq-len 512 --k 50 --max-corpus-gb 128 \
    > logs/train.log 2>&1 &"
bash deploy/pod.sh stop myexp   # preserves volume, stops billing
bash deploy/pod.sh delete myexp # destroys everything
```

GPU shortcuts: `a5000`, `a40`, `a6000`, `4090`, `5090`, `l40s`, `h100`. Pod configs are cached in `~/.config/pawn/pods/<name>.env`.

### Instance Lifecycle (vast.ai)

```bash
bash deploy/vast.sh search --gpu 4090 --max-price 0.5
bash deploy/vast.sh create myexp --gpu 4090 --max-price 0.5
bash deploy/vast.sh deploy myexp   # rsync local checkout to /workspace/pawn
bash deploy/vast.sh ssh myexp
# Same caveat as the RunPod wrapper — `vast.sh launch` appends `--log-dir`
# (legacy); JAX scripts use `--logs-dir`, so ssh + manual run for now:
ssh <vast-instance> "cd /workspace/pawn && nohup uv run python \
    scripts/train_jax.py --supernet supernet --total-steps 100000 \
    --batch-size 256 --seq-len 512 --k 50 --max-corpus-gb 128 \
    > logs/train.log 2>&1 &"
bash deploy/vast.sh stop myexp
```

Vast.ai has no separate network volume — instance disk is sized via `--disk` (default 100 GB) and persists across `stop`/`start`. `delete` destroys the disk. `HF_TOKEN` and `PUBLIC_KEY` from your local environment are passed through to the instance at create time.

### Required Instance Configuration

- **Persistent storage.** On RunPod, attach a network volume (mounted at `/workspace`). On vast.ai, the instance disk persists across stop/start; pick a size with `--disk` that comfortably holds checkpoints.
- **Set `HF_TOKEN` as an environment variable** for automatic HuggingFace authentication.
- `PAWN_MODEL=thomas-schweich/pawn-base` — auto-pull a published checkpoint on startup (runner target).
- `PAWN_CMD` — training command to execute (alternative to Docker CMD args).

### Instance Safety

- `bash deploy/pod.sh stop <name>` / `bash deploy/vast.sh stop <name>` halts billing but **does not yet trigger a graceful trainer shutdown** under the JAX entry points (no signal handler installed yet — see the gotcha below). Stop only when no training is in flight, or accept the in-flight chunk loss.
- **Never delete/destroy an instance while training is running** — data loss risk on either provider.
- **Never rsync checkpoint files from running instances** — load via the published HF repos.
- On vast.ai with `--interruptible`, the host can preempt you at any time.

## Key Patterns & Gotchas

- **Single-framework promise.** The training and eval surface is JAX-only. The two torch touchpoints — `pawn.torch_loader` (a thin loader for external non-JAX consumers) and `pawn._torch_legacy_fixture` (a frozen reference architecture used by the legacy-converter parity tests) — exist precisely so the JAX surface stays torch-free under a CPU-jax install. Don't reintroduce torch dependencies into `pawn.{model,trainer,adapter_trainer,adapters,eval,checkpoint}`.
- **Sentinel logic lives in `pawn._sentinel`.** Stdlib-only. Both `pawn.checkpoint` and `pawn.torch_loader` import `verify_sentinel` from it (aliased to a local underscore name on both sides; the public API on the module itself has no underscores). `pawn.checkpoint` *also* has its own private `_write_sentinel` that adds a `format_version` field — `pawn._sentinel.write_sentinel` writes a strict `{"files": ...}` payload and that asymmetry is intentional. Don't duplicate the SHA-256 / verify logic elsewhere; if a new caller needs to write a sentinel, decide explicitly whether to use `_sentinel.write_sentinel` or to keep its own variant.
- **Two-tier PyTree partition** is how adapter training freezes the backbone: `eqx.partition(model, adapter_filter(model))` produces a trainable subtree (adapters) and a frozen subtree (backbone). XLA dead-code-eliminates the backbone weight gradients, which is the source of the ~33% FLOP cut on the backward pass.
- **K-step `lax.scan` amortises JIT dispatch.** The pretraining + adapter trainers both run `K` inner steps per scan invocation. `K * B` games are consumed per scan call; size `K` so per-step throughput is roughly host-bound rather than launch-bound.
- **`state.step` must be a JAX scalar inside `jit`.** Storing it as a Python int caused a ~70× slowdown during Phase-2 development by retriggering recompilation every step. The fix is pinned by tests; keep it that way.
- **`optax.warmup_cosine_decay_schedule`'s `decay_steps` is the end-to-end length, not the post-warmup tail.** A `decay_steps = total_steps - warmup` would double-subtract warmup. The trainer passes `decay_steps = total_steps`.
- **Padded-batch AdamW weight-decay drift.** Without a `lax.cond` guard, weight decay still fires on a padded final chunk and erodes the parameters; the trainer guards against it.
- **Gradient clipping** is `optax.chain(optax.clip_by_global_norm(1.0), adamw)`. Matches the legacy PyTorch contract.
- **Factored embeddings**: each move token decomposes into `src_embed[s] + dst_embed[d] + promo_embed[p]`, shrinking the move-embedding table from `1968 × d_model` to `(64 + 64 + 5) × d_model` — ~14.8× fewer params on that table (input side only; `lm_head` is still a full `d_model → vocab` projection).
- **Legacy converter rejects pre-vocab-transition checkpoints.** The current JAX model uses the 1,968-action `decomp_table`; converting an older ~60k-token PyTorch checkpoint would silently embed every move incorrectly. See `pawn.legacy.convert_legacy_checkpoint` and the `pre-vocab-transition` git tag.
- **fp32 cross-framework parity bar is `1e-3`, not `1e-4`.** The toy-config converter test gets `~2.4e-7` because of its small accumulation budget; the three published checkpoints land at mean `|Δlogit| ≈ 5×10⁻⁶` (max around `1e-4`). Don't tighten the published-checkpoint tolerance.
- **stockfish-datagen 50-move rule is eval-strategic, not unconditional.** At halfmove 100 (the FIDE-claimable threshold), the side about to move claims iff Stockfish's top-candidate `score_cp` (side-to-move POV) is `< 0`. Winning/even sides keep playing; losing sides claim. The 75-move *automatic* rule (halfmove 150) is the hard upper bound, fires regardless of eval. Means the dataset has 50-move-rule draws scattered across halfmoves 100–150 (correlated with eval), giving the model a learnable signal for *when* to claim rather than baking in "halfmove 100 → game over." 3-fold repetition stays unconditional. See `stockfish-datagen/src/game.rs` (`detect_pre_eval_terminal` + `should_strategic_claim_50mv`).
- **stockfish-datagen worker pinning shifts the n_workers sweet spot down.** Each (worker, Stockfish) pair is pinned to `worker_id % n_logical` on Linux so they share L1/L2. Rule of thumb: **`n_workers = total_threads − threads_per_core`** — fully occupy every physical core except one, and leave that one core entirely free for the kernel + watcher thread + HF upload networking + parquet I/O. For typical vast.ai 128-thread / 64-physical / 2-SMT pods that's 126 workers; the bundled `stockfish-datagen/examples/stockfish_100m.json` config defaults to that.

  As of the shard-id partitioning refactor, `cfg.n_workers` is operational-only — workers pull from a shared atomic shard-id counter, so changing it between runs never changes any game's content (the per-game seed is `mix(tier_seed, global_game_index)` and tier seeds are keyed by sha256(`tier.name`), not by index). Multi-pod cooperation uses `--shard-id-range A:B` on the `stockfish-datagen run` subcommand; each pod writes per-pod sentinels with the range zero-padded to 6 digits (`_tier_state-s000000-s005000.json`, `_manifest-s000000-s005000.json`) and `scripts/datagen_reconcile_tier.py` merges them into a canonical `_manifest.json` post-run. The orchestrator (`scripts/datagen_with_hf_sync.py`) commits shards via batched `huggingface_hub.HfApi.create_commit` calls with one `CommitOperationAdd` per shard (~1 commit per tier per cycle), staying well under HF's 128 commits/hour limit.

  NUMA: `stockfish-datagen` calls `set_mempolicy(MPOL_INTERLEAVE)` at startup (Linux only). Children inherit across `execve`, so stockfish workers first-touch their NNUE page-cache fills under interleave policy. No-op on single-socket pods.

  Stockfish binary: the `:datagen` Docker image JIT-builds the patched stockfish for the host CPU on first launch (`scripts/build_stockfish_for_host.sh`) and caches it under `/workspace/.cache/stockfish/`. One image tag covers every supported microarch (vnni512 / avx512 / avx2 / modern).
