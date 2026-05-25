# PAWN (Playstyle-Agnostic World-model Network for Chess)

> **Post-swap orientation map (canonical).** This document describes the v2
> JAX/Equinox/Optax repo. The framework swap is tracked in
> `docs/jax_migration_plan.md`; until that migration lands, individual modules
> referenced below may live on `feat/jax-migration/*` section branches rather
> than on the integration branch.
>
> **Published v1 checkpoints.** `thomas-schweich/pawn-{small,base,large}` are
> v1 PyTorch artifacts. They stay frozen — v2 republishes to new HF repos
> (`pawn-{small,base,large}-v2`). The bridge that lets v2 code load the v1
> repos is `pawn.legacy.convert_legacy_checkpoint`; every benchmark or metric
> attached to a v1 repo card is a v1 PyTorch number, not a v2 number.

A causal transformer trained on random chess games, designed as a testbed for
finetuning and augmentation methods at small scales. Apache 2.0.

## Repository Structure

```
pawn/
├── engine/                  # Rust chess engine with PyO3 bindings (via shakmaty)
├── pawn/                    # Core Python package
│   ├── _sentinel.py         # stdlib-only SHA-256 .complete sentinel helpers
│   ├── config.py            # ModelConfig, SUPERNET, VARIANTS, TINY_*, validate_nested
│   ├── model.py             # Equinox PAWNModel (RMSNorm + RoPE + SwiGLU + factored embeddings)
│   ├── run_config.py        # pydantic configs: BaseRunConfig / PretrainConfig / AdapterConfig / SpecializedCLMConfig
│   ├── logging.py           # MetricsLogger (JSONL, type-discriminated, NaN-sanitised)
│   ├── checkpoint.py        # Atomic safetensors save/load + async HF push
│   ├── corpus.py            # Rust-engine random games → Corpus (JAX arrays)
│   ├── lichess_data.py      # Lichess parquet → Corpus, on-disk cache, multi-epoch tiling
│   ├── trainer.py           # Pretraining: lax.scan K-step training loop + supernet joint loss
│   ├── adapter_trainer.py   # Two-tier frozen/trainable PyTree adapter training
│   ├── adapters/            # lora / film / unfreeze / bottleneck / hybrid / sparse / rosa / specialized_clm
│   ├── eval.py              # Move-accuracy + per-phase
│   ├── probes.py            # Linear probes via Optax fit on frozen hidden states
│   ├── generation.py        # 5 generation diagnostics (all gated on outcome_prefix_trained) + KV-cached decoder
│   ├── lichess_eval.py      # Elo-stratified Maia-style accuracy
│   ├── eval_suite/          # Edge-case diagnostics, theoretical accuracy bounds, viz helpers
│   ├── sweep.py             # Standalone Optuna driver (subprocess + in-process objectives)
│   ├── legacy.py            # Single bridge: v1 torch HF checkpoint → JAX safetensors
│   ├── lab/                 # FastMCP lab manager (trial orchestration via pydantic configs)
│   ├── dashboard/           # Solara dashboard (reads metrics.jsonl)
│   └── wandb_utils.py       # Optional W&B integration
├── scripts/                 # Training and eval entry points (train_jax*, eval_jax*, sweep, etc.)
├── tests/                   # Unit + integration + scripts/smoke tests
├── deploy/                  # RunPod + vast.ai deployment scripts
└── docs/                    # Architecture, training, adapter docs + the migration plan
```

## Building

This is a uv workspace. The root project is the `pawn` Python package;
`engine/` is the sole workspace member.

```bash
# Build the Rust chess engine (required before anything else)
cd engine && uv run --with maturin maturin develop --release && cd ..

# Install Python deps — choose your GPU backend:
uv sync --extra rocm      # AMD ROCm 7.1 (torch + triton + triton-rocm + jaxlib + jax-rocm7-plugin)
uv sync --extra cu128     # NVIDIA CUDA 12.8 (torch + jaxlib + jax-cuda12-plugin)

# Optional extras can be combined with a GPU extra:
#   --extra dashboard   solara + plotly + pandas + anywidget + optuna-dashboard
#   --extra lab         fastmcp (for `python -m pawn.lab`)
#   --extra wandb       wandb metric mirror

# Run tests
uv run --extra rocm pytest tests/

# Pretrain the tiny supernet (local dev)
uv run --extra rocm python scripts/train_jax.py --supernet tiny --total-steps 1000 \
    --batch-size 16 --seq-len 64 --k 50 --local-checkpoints
```

The two GPU extras are mutually exclusive (declared in `[tool.uv].conflicts`).
The base dependency set includes JAX + Equinox + Optax + pydantic + polars +
matplotlib + seaborn + jinja2 + zstandard + optuna — these are load-bearing
for the realistic Lichess adapter path, the legacy eval suite, and the
data/publishing scripts, so they live in `[project.dependencies]` rather than
behind an extra. Only the GPU plugins, the dashboard / lab UIs, and the wandb
mirror are optional.

**GPU requirement.** JAX manages its own device detection. Training entry
points refuse to run on CPU unless `PAWN_ALLOW_CPU=1` is set (parity with the
v1 escape hatch). Unit tests run on CPU without the override.

## Engine (`engine/`)

**Single source of truth** for all chess logic. All game simulation, move
generation, legality checks, tokenization, PGN parsing, and board state
extraction happen in Rust. No Python chess libraries.

- Uses rayon for parallel game generation
- PyO3 bindings expose `chess_engine` module to Python
- Key functions: `generate_random_games()`, `parse_pgn_file()`,
  `compute_legal_token_masks_sparse()`, `extract_board_states()`,
  `export_move_vocabulary()`, `compute_accuracy_ceiling()`, `edge_case_bits()`
- `export_move_vocabulary()` returns the 1,968-entry searchless_chess action
  table used by the factored embeddings.

## Model

### Architecture

- Decoder-only transformer, next-token prediction over 1,968 move tokens
  (1,980 total vocab).
- Token vocabulary: 1,968 searchless_chess actions (0–1967) + 1 PAD (1968) +
  11 outcomes (1969–1979) = 1,980 total.
- Factored embeddings: `src_embed[s] + dst_embed[d] + promo_embed[p]`.
- Sequence format: `[ply_1] ... [ply_N] [PAD] ... [PAD]` (512 tokens) —
  outcome prefix is optional via the `prepend_outcome` field in
  `BaseRunConfig`.
- Equinox `PAWNModel` is a single module covering the supernet, every sliced
  variant, and any standalone (converted-legacy) model. Stacked layers are
  applied with `jax.lax.scan` over a leading `n_layers` axis. Attention
  defaults to plain materialised `QK^T`; pass `--use-sdpa` (or set
  `use_sdpa=True` on `BaseRunConfig`) to switch to
  `jax.nn.dot_product_attention` (XLA implementation). The default is plain
  because on RDNA 3 the fused kernel OOMs at training shapes (B≥2 T=512 wants
  128 KB shared memory; the GPU has 64 KB/CU). At seq 512 attention is ~12%
  of step FLOPs.
- KV-cached decode: `PAWNModel.forward_with_cache(input_ids, cache,
  pos_start)` + `pawn.model.KVCache` + `init_kv_cache(...)` give an
  autoregressive generation path that's O(T) per step instead of O(T²) per
  step. `pawn.generation.autoregressive_generate` auto-detects the cached
  path; `use_kv_cache=False` keeps the legacy full-forward for parity
  testing.

### Supernet + variants

The supernet is large's dimensions. small and base are nested slices of it —
the inner `[:d_V, :d_V]` of every weight matrix. Joint training sums
per-variant cross-entropies on the same batch; gradients accumulate into the
one shared weight tensor. The supernet is sliced into three standalone
safetensors checkpoints at publish time; downstream consumers see ordinary
independent checkpoints.

- `SUPERNET`: d=640, 10 layers, 10 heads, head_dim=64
- `VARIANTS["small"]`: d=256, 4 heads
- `VARIANTS["base"]`: d=512, 8 heads
- `VARIANTS["large"]`: d=640, 10 heads (= the supernet itself)
- `TINY_SUPERNET`: d=192, 4 layers, 3 heads — for verification runs that
  don't need production scale. `TINY_VARIANTS` slice the same way.

`head_dim = 64` is fixed across nested variants so width slices align to
whole heads and RoPE is variant-invariant.

> **Legacy note.** Earlier versions of this codebase used a ~60k-entry move
> vocabulary. The current code only knows about the 1,968-action vocabulary
> and the single canonical parquet schema. Pre-vocab-transition checkpoints
> are rejected loudly by `pawn.legacy.convert_legacy_checkpoint` and are
> accessible only via the `pre-vocab-transition` git tag.

## Training

All training scripts require one of `--hf-repo REPO_ID` or
`--local-checkpoints` (mutually exclusive). Use `--local-checkpoints` for
local dev; use `--hf-repo` for any run where you need durable checkpoints.

### Pretraining the supernet

```bash
uv run --extra rocm python scripts/train_jax.py \
    --supernet base --total-steps 100000 --batch-size 256 --local-checkpoints

# Resume from a checkpoint:
uv run --extra rocm python scripts/train_jax.py --supernet base \
    --resume checkpoints/step_00050000 --local-checkpoints
```

The training loop is a `jax.lax.scan` over `K` inner steps inside one
`@eqx.filter_jit` function with buffers donated. Per-step body never returns
to the host; per-chunk metrics flush to disk between chunks.

Cotrain (the v1 multi-variant joint trainer) is GONE BY DESIGN. The supernet
joint loss is what cotrain provided. Pretrain the supernet, then publish
its three nested slices.

### Adapter training

All adapter strategies dispatch through `scripts/train_jax_adapter.py`. They
freeze the backbone (via `eqx.partition(model, adapter_filter(model))`) and
train only adapter parameters. `jax.grad` only differentiates the trainable
PyTree, so XLA DCEs the backbone weight-gradients — a ~33% backward-pass FLOP
cut for free.

```bash
uv run --extra rocm python scripts/train_jax_adapter.py --strategy lora \
    --supernet tiny --variant base --lora-rank 4 \
    --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 1900 \
    --total-steps 200 --local-checkpoints
```

The 8 strategies and their key args (full surface in `pawn/run_config.py`):

| `--strategy`      | Adapter                                   | Key args                                                |
|-------------------|-------------------------------------------|---------------------------------------------------------|
| `lora`            | Low-rank attention                        | `--lora-rank 4 --lora-targets qkvo`                     |
| `film`            | Channel-wise affine                       | `--use-output-film` (default True)                      |
| `bottleneck`      | Houlsby MLP                               | `--bottleneck-dim 8 --no-adapt-attn`                    |
| `hybrid`          | LoRA + FiLM                               | `--lora-rank 4`                                         |
| `sparse`          | Binary mask                               | `--density 0.01 --sparse-targets qkvo`                  |
| `rosa`            | Gradient-informed sparse + LoRA (3-phase) | `--rosa-mode rosa` &#124; `retro-sparse` &#124; `retro-bottleneck` |
| `unfreeze`        | Fine-tune explicit layer picks            | `--unfreeze-layers 5,6,7`                               |
| `specialized_clm` | From-scratch standalone transformer       | `--d-model 64 --n-layers 2 --n-heads 4 --d-ff 256`      |

The `rosa` strategy has three sub-modes selected by `--rosa-mode`: `rosa`
(standard), `retro-sparse`, and `retro-bottleneck`. All three are
in-scope and tested.

Lichess data is cache-first: the first run with a given (Elo, `min_ply`)
combination filters and tokenizes the dataset to disk under
`$HF_HOME/pawn-lichess-cache/<sha-of-filter-params>/`. The dataset's
`validation` split is the default held-out source; pass `--pgn-val-split ""`
to carve from train (only for single-file local sources).

LR schedules: `--lr-schedule {cosine,wsd,constant,one_cycle,infinite}` —
warmup + cosine / WSD / constant / one_cycle / infinite-cooldown. Sum-fraction
validation in `BaseRunConfig` keeps `warmup_frac`, `cooldown_frac`,
`decay_frac` consistent.

### Common CLI patterns

- `--config <path.json>` — load a JSON run config (validated through pydantic
  `extra="forbid"`); CLI flags merge in and take precedence.
- `--wandb` — enable Weights & Biases metric mirror (requires `--extra wandb`).
- `PAWN_ALLOW_CPU=1` — last-resort CPU escape hatch.

## Evaluation

### Move accuracy

```bash
uv run --extra rocm python scripts/eval_jax.py --checkpoint thomas-schweich/pawn-base
```

Compatible with v1's eval_accuracy schema; reports overall + per-phase
breakdown. Argmax restricted to `[0, NUM_ACTIONS)` so PAD and outcome tokens
can't be sampled.

### Linear probes

```bash
uv run --extra rocm python scripts/eval_probes_jax.py --checkpoint <converted>
```

### Generation diagnostics + edge-case diagnostics

```bash
uv run --extra rocm python scripts/eval_generation_jax.py \
    --checkpoint <converted> --outcome-prefix-trained --edge-cases
```

The five generation diagnostics — `outcome_signal_test`,
`prefix_continuation_test`, `poisoned_prefix_test`, `impossible_task_test`,
`improbable_task_test` — **all** condition on the outcome token at position 0
and **all** return a `{"_skipped": ...}` sentinel when
`--no-outcome-prefix-trained` is set.

Edge-case diagnostics use `engine.edge_case_bits()` for guaranteed coverage
of `in_check` / `double_check` / `pin_restricts` / `ep_available` /
`castle_legal_*`.

### Elo-stratified Lichess accuracy

```bash
uv run --extra rocm python scripts/eval_vs_stockfish.py \
    --checkpoint <converted> --pgn thomas-schweich/pawn-lichess-full
```

Maia-style per-Elo-bin accuracy.

### Compatibility loader (v1 HF → JAX)

```bash
python -c "from pawn.legacy import convert_legacy_checkpoint; \
           convert_legacy_checkpoint('thomas-schweich/pawn-base')"
```

Reads a v1 torch `.safetensors` checkpoint, transposes linear weights from
`(out, in)` to `(in, out)` JAX convention, writes a JAX checkpoint under
`$HF_HOME/pawn-jax-converted/<variant>/`. Cached by content hash. Rejects
pre-vocab-transition checkpoints loudly. This is the **only** v1↔v2 bridge.

## Checkpoints

Pre-trained weights are hosted on HuggingFace and loaded by repo ID through
the legacy converter:

- `thomas-schweich/pawn-small` — v1 PyTorch, ~9.5M params
- `thomas-schweich/pawn-base` — v1 PyTorch, ~35.8M params
- `thomas-schweich/pawn-large` — v1 PyTorch, ~68.4M params

v2 supernet-derived checkpoints publish to new repos
(`pawn-{small,base,large}-v2` or similar).

### Checkpoint format (safetensors)

Each checkpoint is a directory:

```
step_00065000/
├── model.safetensors        # one tensor per PAWNModel array field (~16 fields, declaration order)
├── optimizer.safetensors    # flattened Optax state
├── training_state.json      # step, scheduler, RNG (base64)
├── config.json              # ModelConfig + run config
└── .complete                # SHA-256 hashes of all files (integrity sentinel)
```

Atomic save: payload files land in `step_<N>.tmp`, then a `.complete` sentinel
with SHA-256s, then `os.rename(step_<N>.tmp, step_<N>)`. Old checkpoints are
never overwritten or deleted by the trainer; pruning is the user's call.

**Every load verifies the sentinel.** `IncompleteCheckpointError` /
`CheckpointIntegrityError` are raised on missing or mismatched hashes. The
sentinel helpers live in `pawn/_sentinel.py` (stdlib-only) so they can be
imported without dragging in JAX.

### Storage modes

All training scripts require one of:

- `--hf-repo REPO_ID` — push checkpoints to a HuggingFace branch as they're
  written (async; failures don't block training)
- `--local-checkpoints` — save locally only

HF mode creates a `run/{run_id}` branch. Squash-merge into main when
satisfied.

### Operational guarantees

- **SIGTERM is handled gracefully** — the training loop finishes the current
  chunk, saves a checkpoint, pushes to HF, and exits 0. Never use `kill -9`.
- **`--resume <ckpt>`** loads `TrainState` from the checkpoint and splices
  `state.step` from the saved value so the metrics log stays monotonic across
  the resume.
- **Never rsync checkpoint files from running pods.** Load via HF repo ID.

## Metrics & dashboard

`MetricsLogger` (`pawn/logging.py`) is the **only** path metrics take to
disk. Every record in `metrics.jsonl` has a `type ∈ {"config","train","val"}`
discriminator, a timestamp, slug, hostname, git_hash, and (on train/val)
host + GPU memory stats. NaN / Inf sanitised to `null`. Per-record flush.

```bash
uv run --extra dashboard python -m pawn.dashboard --log-dir logs
```

Reads `metrics.jsonl` files, no dependency on training packages. Auto-detects
run type from config fields. Shows loss curves, accuracy, LR schedules, GPU
utilisation, patience clocks, and adapter-specific diagnostics.

## Hyperparameter sweeps

```bash
uv run --extra rocm python scripts/sweep.py --strategy lora --n-trials 30 \
    --supernet base --storage sqlite:///./lora.db --logs-dir ./sweep
```

Optuna driver with per-strategy `suggest_*` functions matching the v1 search
spaces. Two objective shapes: `AdapterObjective` (subprocess per trial, parses
`metrics.jsonl` for best `val_loss`) and `InProcessRoSAObjective` (skips
per-trial JAX startup for big RoSA sweeps). Persistent study state via SQLite.

## Lab (FastMCP)

```bash
uv run --extra lab python -m pawn.lab
```

`lab_launch(config={...})` validates the incoming trial dict through
pydantic (`extra="forbid"` rejects stale field names). `lab_schema` returns
JSON Schema generated from `PretrainConfig.model_json_schema()` /
`AdapterConfig.model_json_schema()`.

## Logs

Training metrics in `logs/` (gitignored). Each run gets a timestamped
directory with `metrics.jsonl` and a random slug.

## Cloud GPU operations

PAWN runs on either RunPod or vast.ai. The same Docker image works on both —
pick the provider that has the GPU you want at the price you want.

| | RunPod | vast.ai |
|---|---|---|
| Manager script | `deploy/pod.sh` | `deploy/vast.sh` |
| CLI | `runpodctl` | `vastai` (or `uvx vastai`) |
| Local config dir | `~/.config/pawn/pods/` | `~/.config/pawn/vast/` |
| Volume model | Network volume mounted at `/workspace` | Single instance disk |
| Pricing | Fixed per-GPU rates | Marketplace |

### Pod lifecycle (RunPod)

```bash
bash deploy/pod.sh create myexp --gpu h100
bash deploy/pod.sh ssh myexp
bash deploy/pod.sh launch myexp scripts/train_jax.py --supernet base \
    --hf-repo thomas-schweich/pawn-base-v2
bash deploy/pod.sh stop myexp        # SIGTERM, graceful shutdown
bash deploy/pod.sh delete myexp      # destroy everything
```

### Instance lifecycle (vast.ai)

```bash
bash deploy/vast.sh search --gpu 4090 --max-price 0.5
bash deploy/vast.sh create myexp --gpu 4090 --max-price 0.5
bash deploy/vast.sh deploy myexp     # rsync local checkout to /workspace/pawn
bash deploy/vast.sh launch myexp scripts/train_jax.py --supernet base \
    --hf-repo thomas-schweich/pawn-base-v2
```

`HF_TOKEN` and `PUBLIC_KEY` are forwarded automatically at create time. Both
providers honour the same Docker image (`thomasschweich/pawn:latest`) and
entrypoint.

### Instance safety

- Stop with `pod.sh stop` / `vast.sh stop` — sends SIGTERM, trainer saves and
  pushes before exiting.
- **Never delete/destroy an instance while training is running.**
- **Never `kill -9` training processes.**
- **Never rsync checkpoint files from running instances** — load via HF repo
  ID instead.

## Key Patterns & Gotchas

- **Adapter training is cache-first.** First run with a given (Elo,
  `min_ply`) combination filters and tokenizes to disk under
  `$HF_HOME/pawn-lichess-cache/<key>/`. Subsequent runs mmap the cache. Filter
  parameters bake into the cache key.
- **`steps_per_epoch` is canonical for adapters.** `"all"` resolves to
  `n_train_games // batch_size` once the cache materialises.
- **`schedule_health.json`** is written at trainer exit. Records
  `{planned_total_steps, actual_total_steps, reason_for_stop, lr_peak,
  actual_final_lr}`. `actual != planned` AND `reason_for_stop == "completed"`
  is a structural-bug signal.
- **All five generation diagnostics gate on outcome_prefix_trained.** Not
  just the obvious two (`impossible_task_test` / `improbable_task_test`).
- **The held-out validation split is the default.** Carving val out of train
  silently leaks; only opt back into carve-from-train for single-file local
  sources without split structure.
- **One framework.** JAX/Equinox/Optax everywhere. The only torch touchpoints
  are `pawn/legacy.py` (the v1 converter) and `pawn/_torch_legacy_fixture.py`
  (the converter's parity-test reference architecture).
- **PAD-token init in the engine.** Every PGN-token-init site initialises
  with `vocab::PAD_TOKEN`, not `0` — the vocab assigns `0` to a legal move,
  so a 0-initialised tail looks like real moves downstream.
- **Factored embeddings.** Each move token decomposes into
  `src_embed[s] + dst_embed[d] + promo_embed[p]`, shrinking the
  move-embedding table from `1968 × d_model` to `(64 + 64 + 5) × d_model`.
- **WSL2 + ROCm.** JAX-on-ROCm works on WSL2 but emits a benign
  "sysfs nodes path does not exist" warning at import time; ignore it. The
  RocmDevice still resolves and jit'd kernels run normally.

## When in doubt

The original PyTorch/Stockfish-datagen pipeline lives at the `v1.0.0` git
tag. The `pre-vocab-transition` tag preserves the older 60k-token vocabulary.
The full framework swap is tracked in `docs/jax_migration_plan.md` — every
commit on `jax_migration` and its section branches references that document
under a `Plan:` line in the commit body.
