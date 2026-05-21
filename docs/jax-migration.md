# PAWN → JAX migration (v2)

> **Status (2026-05-21):** `jax_migration` was hard-reset to
> `origin/main` and the migration is being re-implemented from scratch
> under `/review-driven-development`. The prior `jax_migration` HEAD
> survives on `jax_migration_backup_2026-05-21` for reference, but its
> framework-swap PR (#111) was withdrawn — that commit history dropped
> too much of the v1 surface (pydantic configs, structured
> `MetricsLogger`, standalone Optuna sweep driver, eval-suite
> `bounds`/`viz`, parts of `eval_suite/corpus`) and cosmetically
> renamed a handful of CLI/config fields without justification.
> The framework swap will be re-landed here with *full v1.x parity*
> on the surfaces that are not explicitly out of scope (§2 Scope, §8
> Backward-compatibility).
>
> The design that was reviewed and accepted in #111 — supernet +
> nested-slice variants, two-tier frozen/trainable PyTree, fused
> `lax.scan` chunks, the `pawn.legacy` converter, the thin
> `pawn.torch_loader`, the JAX adapter trainer with all 8 strategies,
> the JAX eval port (move accuracy / probes / generation /
> Elo-stratified Lichess) — is unchanged. What changes is the
> *re-implementation cadence* (§13): a fresh section-by-section land,
> driven by `/review-driven-development`, with backward compatibility
> as a first-class deliverable.

## Invocation

- Feature slug: `jax-migration`
- Master feature branch: `jax_migration` (long-lived integration
  branch, hard-reset to `origin/main` on 2026-05-21; the prior HEAD
  is preserved on `jax_migration_backup_2026-05-21`)
- Effective `/review-driven-development` flags (defaults):
  - `--plan-path`: `docs/jax-migration.md`
  - `--no-review-plan`: `false`
  - `--loop-chunks`: `false` (one review wave per chunk)
  - `--loop-sections`: `true`
  - `--loop-final`: `true`
  - `--pr-after-fixes`: `false` (final framework-swap PR opened
    manually after the section loop converges)

## 1. Motivation

PAWN-base is small: ~35.8M parameters, 512-token sequences, a
1,980-token vocabulary. A training step is only ~30 TFLOP, yet the
v1 PyTorch stack runs at roughly 15% MFU — most of the machine is
idle, and the idle time is not matmuls; it goes into kernel-launch
latency, framework dispatch, host round-trips, and the CPU/PCIe data
path. For a model this small the design should optimize *those*
costs and treat the matmuls as nearly free.

The redesign does four things:

1. **Whole training loop as one compiled program** — eliminates
   per-step launch and dispatch overhead.
2. **Data resident on (or streamed to) the device** — eliminates the
   per-step CPU/PCIe path and live data generation.
3. **A single framework** — JAX everywhere, so there is one model
   definition, one optimizer stack, and no PyTorch/JAX boundary to
   maintain.
4. **One supernet instead of three variants** — small, base, and
   large are extracted as nested slices of a single shared-weight
   model (§5).

### Decision: JAX-only, eval included

JAX-on-ROCm locally is accepted, and a single framework is strongly
preferred over two. If pretraining **and** adapter training are JAX,
keeping `eval_suite` in PyTorch is still maintaining two frameworks,
so the end state is **JAX-only — PyTorch removed from the training
and eval surface, eval included.** The eval model is the same
Equinox model; probes, accuracy, generation, and Elo-stratified
Lichess eval are forward passes plus small Optax fits.

PyTorch survives only in two narrow, well-marked places:

- `pawn.torch_loader` — a thin, dependency-light loader for external
  non-JAX consumers (§3.5).
- `pawn._torch_legacy_fixture` — a frozen reference architecture used
  by the legacy-converter parity tests.

Neither path is exercised by `pawn.{model,trainer,adapter_trainer,
adapters,eval,probes,generation,lichess_eval,checkpoint}`, so the
JAX surface stays torch-free under a CPU-jax install.

### Precision

v2 keeps precision unchanged from v1: **bf16 compute** (fp32
accumulate), **fp32 master parameters**, **fp32 Adam state**. fp8
compute is a documented future lever (analysis suggests ~1.6×
throughput left on the table) but is explicitly out of v2 to de-risk
convergence.

## 2. Scope

| Migrates to JAX | Unchanged (framework-agnostic) | GONE BY DESIGN |
|---|---|---|
| `pawn/model.py` — Equinox `PAWNModel` supernet | Rust `engine/` — all chess logic, tokenization, legal-mask replay | `pawn/cotrain.py` — supernet replaces it (§5) |
| `pawn/trainer.py` — fused training loop | `deploy/`, Docker images | `pawn/gpu.py` — JAX manages its own GPU/SDPA backend |
| `pawn/corpus.py` — corpus generation/loading | `pawn/lichess_cache.py`, `pawn/lichess_data.py` — Lichess parquet pipeline shared with eval | `cotrain` run-type in `scripts/train.py` |
| `pawn/checkpoint.py` — PyTree serialization | | RoSA `retro-sparse`/`retro-bottleneck` modes |
| `pawn/adapter_trainer.py` + `pawn/adapters/` — all 8 strategies | | `mask_samples` / `grad_alpha` RoSA mask-gen (replaced by one-shot grad-magnitude) |
| `pawn/eval.py`, `pawn/probes.py`, `pawn/generation.py`, `pawn/lichess_eval.py` | | `bucket_size` (JAX trainer is shape-static) |

`pawn/config.py` is the canonical config module post-migration; it
defines `ModelConfig` plus the `SUPERNET` / `TINY_SUPERNET` /
`VARIANTS` / `TINY_VARIANTS` constants (§5.1). The legacy v1
`CLMConfig` / `TrainingConfig` dataclasses do **not** survive — they
were strictly architecture descriptors, and `ModelConfig` plus the
supernet/variant tables replace them with cleaner semantics.

`pawn/run_config.py` (the pydantic `BaseRunConfig` / `PretrainConfig`
/ `AdapterConfig` hierarchy) **does** survive. It is the single
source of truth for every parameter that training scripts, sweep
tooling, and the lab MCP server care about — `CotrainConfig` /
`CotrainVariant` are removed (supernet replaces cotrain), but
everything else is forward-ported with the v2 fields added (see §8).
This was a load-bearing piece of v1 that the first attempt
incorrectly discarded.

## 3. Shared core

These components are used identically by pretraining, adapter
training, and evaluation.

### 3.1 Model — Equinox

`PAWNModel` (the JAX-side counterpart to v1's PyTorch `PAWNCLM`) is
an Equinox module (a PyTree of arrays): RMSNorm, SwiGLU, RoPE, and
factored embeddings (`src_embed[s] + dst_embed[d] +
promo_embed[p]`). The model is stored at the **supernet**
dimensions; the three variants are nested slices of it (§5).

- The transformer layers are applied with `lax.scan` over weights
  stacked on a leading axis — one compiled layer body, fast compile.
- **Plain attention** (`softmax(QK^T)V`, scores materialized) rather
  than a fused flash kernel. At seq 512 attention is ~12% of step
  FLOPs, and plain attention sidesteps the maturity of fused
  attention kernels under JAX-on-ROCm.

Equinox is the chosen framework: PyTree-native, plays cleanly with
`lax.scan`, `lax.fori_loop`, and buffer donation, with minimal
framework magic.

### 3.2 Optimizer — Optax

Optax `adamw`. Forward casts parameters to bf16 with fp32
accumulation; the master copy and Adam moments stay fp32 (see §1,
Precision). Gradient clipping is
`optax.chain(optax.clip_by_global_norm(1.0), adamw)` — matches the
v1 PyTorch contract.

### 3.3 Whole loop as one program

The training loop is a `lax.scan` over K steps inside a single
`@eqx.filter_jit` function, with buffers donated. The Python host
drives the loop in chunks of K steps; between chunks it flushes
metrics, writes checkpoints, and (for adapters) runs validation.
The per-step body never returns to the host.

K is chosen to amortize host overhead; for adapters it is
additionally bounded by the validation cadence (`K ≤ val_every`).

### 3.4 Two-tier parameter tree

The model PyTree is partitioned into `frozen` and `trainable` via
`eqx.partition(model, adapter_filter(model))`. `jax.grad` /
`eqx.filter_grad` differentiates only `trainable`; XLA
dead-code-eliminates the gradient computations for `frozen`
parameters (~33% FLOP cut on the backward pass).

- **Pretraining:** everything is `trainable`.
- **Adapter training:** the backbone is `frozen`, adapter parameters
  are `trainable` (for `unfreeze`, the top-N backbone layers are
  also `trainable`).

One mechanism, two configurations.

### 3.5 Checkpoints and interop

The atomic write contract is preserved: write to a `.tmp` directory,
rename, and emit a `.complete` sentinel containing SHA-256 hashes of
every file. Hashes are always verified on load (sentinel logic
lives in stdlib-only `pawn/_sentinel.py`; `IncompleteCheckpointError`
+ `CheckpointIntegrityError`). Async HF push is unchanged in spirit.

The JAX PyTree is serialized to **safetensors** (a framework-neutral
file format) under a documented canonical parameter-name schema.

- **Publishing.** The supernet is sliced into three standalone
  variant checkpoints at publish time (§5.5); downstream consumers
  see ordinary independent checkpoints. The supernet itself is
  optionally published as a separate `pawn-supernet` artifact for
  research / ablation use.
- **Thin PyTorch loader.** A small, dependency-light PyTorch module
  is shipped alongside the published checkpoints (`pawn.torch_loader`,
  installable via the `torch-loader` extra). It reads the
  safetensors weights via the canonical name schema into a minimal
  `nn.Module`, so external non-JAX users can load PAWN weights
  without taking a JAX dependency. **Loader-only — not a training
  path.**
- **Legacy converter.** The three already-published checkpoints
  (`pawn-{small,base,large}`) were trained independently and cannot
  be retro-fitted into a supernet. A one-time converter
  (`pawn.legacy.convert_legacy_checkpoint`) produces JAX-loadable
  versions of them for backward compatibility; the new
  supernet-derived variants are fresh training artifacts that will
  be published to **new** HF repos (the existing `pawn-{small,base,
  large}` repos are not overwritten). The converter rejects
  pre-vocab-transition checkpoints (the current `decomp_table` uses
  the 1,968-action vocabulary; older ~60k-token checkpoints would
  silently embed every move incorrectly).

## 4. Pretraining path

### 4.1 Corpus

The corpus is **pre-generated offline** by the existing Rust engine
(`generate_random_games()`). The engine returns int16 tokens, int16
game lengths, and uint8 termination codes; `pawn.corpus` widens
them at the boundary to a packed `Corpus` of int32 `tokens [N, T]`,
bool `attn_mask [N, T]`, int32 `targets [N, T]` (input shifted left
by one), bool `loss_mask [N, T]`, and uint8 `outcome_offset [N]`.
`N = total_steps × batch_size`, so every step sees a fresh game in a
single pass.

### 4.2 No shuffling

Random games are i.i.d. by construction — each game's seed is a
hash of its index, so consecutive indices are statistically
independent. Under single-pass, no-reuse consumption, sequential
order is statistically identical to random sampling and strictly
simpler:

- **Trivial resume:** the offset is `step × batch_size`; no RNG to
  seed or log.
- **Faster I/O:** each chunk is a contiguous slab, not a scattered
  gather — sequential reads are ~3–5× faster and benefit from OS
  readahead.
- **Auditable:** batch `s` is a known, fixed range of game indices.

Conditions (all easy to honor): single pass; homogeneous on-disk
order — do **not** sort the corpus by game length (length
correlates with outcome); store in generation/index order. If the
corpus ever concatenates distinct distributions, interleave them
round-robin instead of consuming sequentially.

### 4.3 Data path

The corpus lives in host RAM (or an NVMe memmap). A double-buffered
prefetch thread stages the next chunk (`K · batch_size · T · 10`
bytes; ~262 MB at K=200, B=256, T=512 — int32 tokens + bool attn_mask
+ int32 targets + bool loss_mask = 4+1+4+1 bytes per position) to
the device while the current chunk trains.

The required ingest bandwidth has a closed form, independent of
batch and sequence length:

```
BW_required = peak_FLOP/s · MFU / (3 · N)
```

For PAWN-base on a B200 (peak ≈ 2.2 PFLOP/s, MFU ≈ 0.5, N = 35.8M)
this is **~10 MB/s** — 500–2500× below PCIe or sequential-NVMe
bandwidth. The prefetch is provably hidden behind compute; a model
would need to shrink below ~10–15K parameters before I/O could bind.
Instrument a data-wait timer and a prefetch-starvation counter in
`metrics.jsonl` to confirm this empirically.

### 4.4 Loop

A fixed `total_steps` → a single clean `lax.scan` of K-step chunks.
The host loop flushes metrics and checkpoints between chunks.
`schedule_health.json` (planned vs actual steps) carries over from
v1 unchanged.

## 5. Variant scheme: shared-weight supernet

PAWN ships three variants — small, base, large. Rather than three
independent training runs, the redesign trains **one supernet** from
which the variants are extracted as nested slices — a MatFormer- /
Matryoshka-style scheme (cf. Devvrit et al., *MatFormer*, 2023).
This replaces v1's `cotrain` run-type and
`configs/cotrain_three_variants.json`: "cotrain" becomes "train the
supernet."

### 5.1 Nested variant dimensions

For the variants to be exact slices, their dimensions must nest.
This requires redefining them around a fixed head dimension:

| Variant | d_model | layers | heads (= d_model / 64) |
|---|---|---|---|
| small | 256 | 8 | 4 |
| base | 512 | 8 | 8 |
| large (= supernet) | 640 | 10 | 10 |

`head_dim` is fixed at 64, so for supernet-nested models `n_heads`
is pinned to `d_model / 64`. (`n_heads` stays an explicit
`ModelConfig` field — standalone models such as converted legacy
checkpoints keep their own head count: legacy `pawn-large` uses
`head_dim = 80` and is not a nested slice.) The only material
redefinition vs. v1 is large's head count (8 → 10); this is
**parameter-count-neutral** (the q/k/v/o matrices are `d_model ×
d_model` regardless of how `d_model` is partitioned into heads) and
it keeps RoPE identical across variants. `d_ff` is sliced alongside
`d_model` (each variant keeps its SwiGLU ratio; the ratios must
nest).

The variants are `ModelConfig` instances in `pawn/config.py` — a
single `SUPERNET` config plus a `VARIANTS` dict. Per existing
project practice, model cards and published-checkpoint docs derive
parameter counts from `config.json` — they are not hardcoded, so
the slight redefinition does not require manual edits.

### 5.2 What gets sliced

The supernet is stored at large's dimensions (d=640, 10 layers).
Variant V with width `d_V` and depth `L_V` uses:

- Every weight matrix: the `[:d_V, :d_V]` (or `[:d_V, :d_ff_V]`)
  prefix.
- Factored embeddings (`src/dst/promo`): the `[:, :d_V]` column
  prefix.
- `lm_head` and the final RMSNorm: the `[:d_V]` prefix, applied
  after layer `L_V`.
- The layer stack: layers `[0:L_V]`.

RMSNorm normalizes over the *active* width at run time. Because each
variant's forward during training uses exactly its slice, the
sliced weights are trained to function at that width — the
slimmable-network / MatFormer property.

`pawn.config.validate_nested(variant, supernet)` enforces the
nesting invariant — equal `head_dim`, identical vocab / context /
outcome layout, and no axis exceeding the supernet — and is called
by `pawn.model.sliced()` before extraction.

### 5.3 Joint training

Each step, on the same batch, the jitted step computes the
next-token loss for each variant at its slice and **sums the
losses** (with per-variant weights); gradients accumulate into the
one shared weight tensor. The variants have different widths, so
they do not `vmap` into a single batched matmul — the step is a
static unroll of three forward calls.

The gradient structure is naturally Matryoshka-coherent: the inner
`[:256]` channels receive gradient from all three variants,
`[256:512]` from base and large, `[512:640]` from large alone — the
inner channels carry the most signal, exactly as the nesting
intends.

**Cost.** Training all three each step costs the sum of their FLOPs
(≈ 1.66× a large-only step), which is FLOP-identical to three
separate runs of the same length — the supernet does not save raw
training FLOPs by default.

**Cost knob.** Every variant's weights are a *subset* of large's,
and large is the full set, so training large every step already
supplies gradient to every weight. The explicit small/base losses
only *shape* those slices into good reduced-width standalone models.
A sandwich-rule schedule — large every step, small and base every
Nth step — therefore trades a little variant quality for a genuine
FLOP reduction below the three-separate-runs baseline. v2 default:
train all three every step; the schedule is a tunable, not a design
open question.

### 5.4 Coupling constraint

The supernet shares one `lm_head` and final RMSNorm across both
widths and depths — so that head must decode layer-8 hidden states
(small, base) *and* layer-10 hidden states (large). This is the
early-exit / layer-dropping training regime; it is trainable but it
couples the variants, and the loss-summation introduces per-variant
loss-weighting hyperparameters. These are accepted costs of the
scheme.

### 5.5 What it wins, and slicing out checkpoints

Relative to three independent runs:

- **One run** — one schedule, one data pipeline, one checkpoint
  stream.
- **Shared weights and optimizer state** — store ~68.4M parameters'
  worth, not ~113.7M (≈ 1.66× less).
- **Guaranteed-consistent variants** — same data, same step count,
  nested representations; directly comparable, which is exactly what
  a finetuning testbed wants.
- **Intermediate widths** are extractable without extra training
  ("Mix'n'Match"); the smaller variants double as draft / early-exit
  models.

At publish time the supernet is **sliced into three standalone
safetensors checkpoints** (`pawn-small-v2`, `pawn-base-v2`,
`pawn-large-v2` — *new* HF repos; the existing v1 repos stay frozen
and remain reachable via the legacy converter for backward
compatibility), each with its own `config.json`. Downstream
consumers — adapters, eval, external users — see ordinary
independent checkpoints and never need the supernet.

## 6. Adapter path

Adapter training reuses the shared core; it adds a frozen backbone
and a finite dataset. The frozen backbone is an ordinary **sliced
variant checkpoint** (§5.5) or a **converted legacy checkpoint** —
adapter training needs no awareness of the supernet. Design deltas
relative to pretraining:

| Aspect | Pretraining | Adapters |
|---|---|---|
| PyTree | all trainable | **frozen backbone / trainable adapter** |
| Dataset | infinite i.i.d. stream | **finite** Elo-filtered Lichess cache |
| Shuffling | none | **per-epoch index permutation** (finite + multi-epoch) |
| Residency | chunk-streamed | whole cache VRAM-resident (~512 MB typical) — no prefetcher |
| Termination | fixed `total_steps` | **adaptive** — host checks patience / best-val between chunks |
| Validation | optional | **required** — a second jitted, forward-only function |

The cache-first model is unchanged in spirit: the first run with a
given `(Elo, min_ply)` builds the tokenized cache; the cache key
logic is untouched. `pawn.lichess_cache` + `pawn.lichess_data`
survive the migration — the Lichess Elo-stratified pipeline is
framework-agnostic (it's a polars + Rust-engine pipeline that
produces token arrays), and only the consumer side (the JAX
adapter trainer) needs to read them.

### 6.1 Performance optimizations

**1. Frozen-backbone autodiff → automatic ~33% compute cut.**
Training is ~6N FLOPs/token = 2N forward + 2N activation-gradient +
2N weight-gradient. Differentiating only the adapter parameters
lets XLA dead-code-eliminate the backbone weight-gradients (~2N),
leaving **4N**. Activation gradients still flow through every
layer (needed to reach the lowest adapter), so per-layer adapters
land at exactly 4N. This is free — it follows from the two-tier
PyTree split.

**2. `vmap`'d sweep populations — the testbed's headline win.**
PAWN is a testbed for finetuning methods; the dominant adapter
workload is sweeps. A population of P adapters shares the *frozen
backbone weights*. `vmap`-ing the forward over the population folds
P into the batch (`M`) dimension of every backbone matmul,
producing one larger, **higher-MFU** matmul instead of P small ones
— this directly attacks the small-matmul inefficiency that caps
this model's throughput.

- Sweepable at fixed graph shape via `vmap`: LR, weight decay,
  dropout, seed, LR schedule, illegal-penalty λ. Use
  `optax.inject_hyperparams` so the optimizer carries per-member
  hyperparameter *arrays*.
- Shape-changing knobs (LoRA rank, bottleneck dim, target layers)
  bucket into separate `vmap` groups.
- Aggressive Hyperband-style pruning loses most of its value: the
  dominant cost (the shared backbone) is paid once regardless of
  how many population members survive. Optuna can still drive the
  outer search — each suggested batch of P configs is one
  `vmap`'d run.
- P is bounded by activation memory (each member has distinct
  activations); trade per-member batch size for population size.

**3. Precompute everything frozen-and-input-only.**
For a fixed dataset, anything that is a pure function of (input
tokens, frozen weights) is computed once at cache-build time:

- **Legal masks** — replay all games in Rust once, store sparse
  legal indices device-resident alongside the tokens.
  `--disable-legal-mask` simply skips applying them.
- **Factored embeddings** — embed the corpus once (small, free).
- **`unfreeze` frozen prefix** — the bottom `8 − N` layers are frozen
  *and* adapter-free, so their output is a pure function of the
  input. Cache it and run only the top-N layers (forward and
  backward). Memory-gated: the cached activations are
  `d_model`-wide, ~512× the token corpus, so this is viable for
  smaller Elo-band caches.

### 6.2 Per-strategy notes

| Strategy | Notes |
|---|---|
| `lora`, `bottleneck`, `film`, `hybrid` | Per-layer inserts; backbone runs fully; benefit from optimizations 1 and 2. Adapter compute itself is negligible — the cost *is* the frozen backbone forward + activation-backward. |
| `unfreeze` | Two-tier split with top-N backbone layers trainable; best case for prefix-activation caching (run forward and backward only over the top-N). |
| `specialized_clm` | No backbone — it *is* the pretraining pipeline pointed at the finite Lichess corpus. Inherits the finite-dataset machinery (epoch shuffle, validation, early stopping). |
| `rosa` | 3-phase; phase transitions are host control flow (re-jit at phase boundaries). |
| `sparse` | Sparse delta on frozen weights; treated as a dense delta at this scale (unstructured sparsity yields no tensor-core speedup here). |

### 6.3 Precision

Adapter training uses **bf16 compute throughout** — never fp16.
bf16's fp32-range exponent eliminates the fp16-overflow failure
mode that ceiling-scale adapters previously hit.

## 7. Evaluation path

`eval_suite` ports to JAX:

- **Accuracy / generation** — forward passes plus argmax / sampling
  on the Equinox model. Argmax is restricted to `[0, NUM_ACTIONS)`
  so PAD + outcome tokens cannot leak.
- **Linear probes** — a small Optax fit on frozen hidden states.
- **Diagnostics** — forward passes. The `outcome_prefix_trained`
  gate that v1 added to `impossible_task_test` /
  `improbable_task_test` extends to the three other diagnostics
  that condition on outcome tokens (`outcome_signal_test`,
  `prefix_continuation_test`, `poisoned_prefix_test`), so all five
  return the same `{"_skipped": ...}` sentinel when the model
  wasn't trained with the outcome prefix.
- **Elo-stratified Lichess eval** — Maia-style move-prediction
  accuracy over Lichess Elo bins, hot-path identical to v1.
- **Legacy position-parquet pipeline** — `pawn/eval_suite/bounds.py`
  (theoretical accuracy bounds) and `pawn/eval_suite/viz.py`
  (matplotlib/seaborn plotting) survive under the `data-tools`
  extra, both wired against the restored
  `pawn/eval_suite/corpus.py` polars iterator. They are
  off-the-hot-path tooling; restoring them costs nothing the JAX
  surface notices.

Legal-mask logic already lives in Rust. This is the largest single
chunk of porting work but conceptually the simplest — it is mostly
forward evaluation.

## 8. Backward compatibility

The first attempt broke compatibility on three surfaces that v1
users depend on. v2 restores them.

### 8.1 Pydantic `run_config` — RESTORED

v1's `pawn/run_config.py` (`BaseRunConfig`, `PretrainConfig`,
`AdapterConfig`, `extra="forbid"`, 6+ cross-field `model_validator`s,
`model_json_schema()` for lab introspection, `--config <json>` flag
on the trainers) is the single source of truth for every training
parameter. The first attempt deleted this and scattered 37
individual `raise SystemExit(...)` guards across `scripts/train_jax.
py` + `scripts/train_jax_adapter.py`; that is reverted.

Forward-port deltas:

- `CotrainConfig` / `CotrainVariant` — **removed**. Supernet replaces
  cotrain (§5).
- New fields: `supernet: Literal["tiny","supernet"]`,
  `variant: Literal["small","base","large"]`, `rosa_warmup_frac:
  float`, `rosa_top_k_frac: float`, etc.
- v2-shape literal types where v1 used scalars: `lora_targets:
  list[str]` (subset of `{q,k,v,o}`) replaces `Literal["qkvo","qv",
  "qkv"]`; same for `sparse_targets` / `rosa_targets`.
- `PretrainConfig` / `AdapterConfig` expose
  `model_json_schema()` — the lab `lab_schema` returns that directly
  instead of a hand-maintained dict.
- Both `scripts/train_jax.py` and `scripts/train_jax_adapter.py`
  accept `--config <json>`; the parser feeds the JSON through
  `TypeAdapter(...).validate_python(...).model_dump()` and then
  drives the existing argparse code path. Bare-CLI invocation still
  works.

### 8.2 `MetricsLogger` — RESTORED

v1's `pawn/logging.py` (`MetricsLogger`, ~327 LoC) ported forward.
JSONL records carry:

- `type: "config" | "train" | "val"` discriminator (the
  `pawn/dashboard/metrics.py` train/val split depends on this).
- `timestamp` (absolute), `elapsed`, `slug`, `hostname`, `git_hash`
  on every record.
- `mem/system_rss_gb`, `mem/system_used_gb`, `mem/cpu_percent` via
  `psutil` (already a dep).
- GPU stats via shell-out to `nvidia-smi` / `rocm-smi` (the same
  approach `pawn/lab/runner._discover_gpus` already uses); the v1
  `torch.cuda.*` branch is dropped.
- NaN / Inf sanitised to `None` so records stay RFC-7159 valid.
- Per-record flush (SIGKILL-durable).
- Slug-based run-dir naming (`run_20260520_140000_zesty-osprey`),
  using a small built-in adjective/animal list — no extra dep.

Both `pawn/trainer.py` and `pawn/adapter_trainer.py` go through
`MetricsLogger.log_config / log_train / log_val` rather than
opening `metrics.jsonl` directly.

### 8.3 Cosmetic renames — REVERTED

These v1 → v2 renames broke backward compatibility with no
documented reason. They are reverted; v1 names become the
canonical `dest=` in argparse and the field name in the pydantic
config. (Where transition matters, the v2 alias may be retained as
an `add_argument` alias for one release — but the v2 alias is **not**
the canonical name.)

| v1 name (canonical, restored) | Discarded v2 rename |
|---|---|
| `lora_rank` | ~~`rank`~~ — ambiguous now that rank is shared across LoRA / Hybrid / RoSA. |
| `density` | ~~`sparse_density`~~ — `density` was already sparse-only in v1; the prefix is redundant. |
| `use_output_film` (default `True`) | ~~`film_output` (polarity-flipped via `--no-film-output`)~~ — polarity flip is gratuitous. |
| `no_adapt_attn` | ~~`bottleneck_no_attn`~~ — cosmetic prefixing. |
| `no_adapt_ffn` | ~~`bottleneck_no_ffn`~~ — cosmetic prefixing. |
| `d_model` / `n_layers` / `n_heads` / `d_ff` (inside `SpecializedCLMConfig`) | ~~`specialized_d_model` / `specialized_n_layers` / `specialized_n_heads` / `specialized_d_ff`~~ — the prefix isn't needed inside a per-strategy config. |

### 8.4 Substantive changes — KEPT (document; do not revert)

These changed v1 semantics on purpose. They are documented under
"v2 release notes" but the v2 names stay.

| v1 | v2 | Rationale |
|---|---|---|
| `lora_targets: Literal["qkvo","qv","qkv"]` | `lora_targets: list[str]` | More flexible; arbitrary subsets of `{q,k,v,o}`. |
| `sparse_targets: Literal[...]` | `sparse_targets: list[str]` | Same. |
| `rosa_warmup_steps: int` | `rosa_warmup_frac: float` | Scales with `--total-steps`. |
| `rosa_mode: "rosa" \| "retro-sparse" \| "retro-bottleneck"` | only `"rosa"` | Retro-ablation modes weren't ported. |
| `mask_samples`, `grad_alpha` | removed | RoSA mask-gen algorithm changed (v1 averaged grad magnitudes over `mask_samples` batches with `grad_alpha`-power weighting; v2 uses single forward+backward with all-True mask). |
| `unfreeze_layers: "5,6,7"` (explicit picks) | `n_unfreeze: 3` (top-N count) | **Regressive — keep on the radar.** v1 let you pick specific layers; v2 only picks the top. Worth restoring v1 flexibility if anyone actually used it; punt on this round unless someone speaks up. |
| `epochs: int` | step-based | The Rust-engine corpus is infinite for pretrain; epochs are meaningless. For adapter training over a finite Lichess corpus, step-based still works but loses the epoch-aligned val-loss curve. |
| `bucket_size: int` | removed | The JAX trainer is shape-static; bucketed padding doesn't apply. |
| `lora_ffn: bool`, `sparse_ffn: bool` | removed | FFN adaptation was deleted from LoRA / Sparse. Could come back. |

### 8.5 `metrics.jsonl` schema — RESTORED

The v1 schema is the canonical schema. Each row carries `type`,
`timestamp`, `elapsed`, `slug`, `hostname`, `git_hash`, and the
`mem/*` keys; NaN/Inf is `null`; adapter val rows are emitted as a
separate `type: "val"` record rather than as a "train row with a
val column."

The legacy v1 `metrics.jsonl` files remain readable by the
dashboard — that is the entire reason the schema is restored
verbatim. Any v2-only ad-hoc files written by the first attempt
are not migrated; they were always self-described as "v2-style."

## 9. ROCm specifics

- **Base install:** `jax`, `equinox`, `optax` are declared as base
  dependencies in `pyproject.toml`, pulling CPU `jaxlib` by default.
  This is what makes `uv run pyright pawn/` resolve on CI without
  an extra and keeps unit tests (logit-parity, checkpoint
  round-trip) running on CPU JAX.
- **GPU extras** mirror v1's pattern: `jax[rocm]` vs `jax[cuda12]`
  via the `rocm` / `cu128` extras. uv cannot co-resolve them from
  one lockfile, so the `--extra rocm` / `--extra cu128` split is
  retained.
- The thin PyTorch loader + the legacy-converter parity fixture
  also need torch, so the `rocm` / `cu128` extras pull torch
  alongside JAX. The `torch-loader` extra adds only torch (CPU jax
  + torch) for users who want the thin loader without GPU jaxlib.
- `jax.jit` replaces `torch.compile`; there is no SDPA-backend
  selection to manage (plain attention).
- Port v1's `configure_gpu()` CPU guard and the `PAWN_ALLOW_CPU=1`
  escape hatch into the JAX entry points — same contract: every
  training/eval script raises `RuntimeError` if no GPU is detected
  unless the override is set.
- If something fails under ROCm, suspect our code first —
  historically every "ROCm bug" in this project turned out to be a
  build-artifact or dependency issue.

## 10. Module layout

The end state is a flat `pawn/` package with optional extras for
non-core surfaces. The first attempt briefly used a `pawn/jax/*`
namespace; that has been flattened.

```
pawn/
├── _sentinel.py             # stdlib-only sentinel + integrity helpers
├── _torch_legacy_fixture.py # frozen torch arch for converter parity tests
├── adapters/                # 8 strategies
│   ├── __init__.py
│   ├── lora.py
│   ├── film.py
│   ├── bottleneck.py
│   ├── hybrid.py
│   ├── sparse.py
│   ├── rosa.py
│   ├── specialized_clm.py
│   └── unfreeze.py
├── adapter_trainer.py       # two-tier frozen/trainable training; scan + eval
├── checkpoint.py            # atomic safetensors save/load + HF push
├── config.py                # ModelConfig, SUPERNET, TINY_SUPERNET, VARIANTS, TINY_VARIANTS, validate_nested
├── corpus.py                # Rust-engine corpus → trainer-shaped arrays
├── dashboard/               # OPTIONAL EXTRA: solara dashboard
├── eval.py                  # move-accuracy + per-phase breakdown
├── eval_suite/              # OPTIONAL EXTRA: legacy position-parquet pipeline
│   ├── bounds.py
│   ├── corpus.py
│   └── viz.py
├── generation.py            # generation diagnostics + KV-cache
├── lab/                     # OPTIONAL EXTRA: pawn-lab MCP server
├── legacy.py                # one-time PyTorch → JAX converter
├── lichess_cache.py         # Lichess Elo-stratified tokenized cache
├── lichess_data.py          # Lichess parquet pipeline (polars)
├── lichess_eval.py          # Elo-stratified Maia-style accuracy eval
├── logging.py               # MetricsLogger (RESTORED)
├── model.py                 # Equinox PAWNModel
├── probes.py                # linear probes on hidden states
├── run_config.py            # pydantic config models (RESTORED)
├── sweep.py                 # standalone Optuna driver (RESTORED)
├── torch_loader.py          # thin PyTorch loader for external consumers
├── trainer.py               # pretraining loop + supernet joint loss
└── wandb_utils.py           # OPTIONAL EXTRA: W&B metrics integration
```

### Optional extras

- `--extra rocm` / `--extra cu128` — GPU jaxlib (+ torch for the
  loader and legacy-converter parity tests).
- `--extra torch-loader` — torch only, for users of
  `pawn.torch_loader` who don't need GPU jaxlib.
- `--extra dashboard` — solara + plotly + anywidget for
  `pawn.dashboard`.
- `--extra lab` — fastmcp + optuna for `pawn.lab`.
- `--extra wandb` — wandb client for `pawn.wandb_utils`.
- `--extra data-tools` — polars + jinja2 + zstandard for the data
  scripts (`extract_lichess_parquet`, `compute_theoretical_ceiling`,
  `generate_model_cards`, etc.) and for the legacy
  `pawn.eval_suite` position-parquet pipeline.

## 11. Integration branch and merge strategy

The migration uses a long-lived integration branch
**`jax_migration`**. All sections (§13) land as their own PRs
targeted at `jax_migration`, never at `main`. When all sections
have merged into `jax_migration`, a single final PR merges
`jax_migration` into `main` — that PR is the project's only
"framework swap" event.

The invariant: **`main` never carries the half-migrated state.**
While JAX matures, `main` stays single-framework (the existing
PyTorch implementation). `jax_migration` is the only branch where
JAX and PyTorch coexist (transiently — the final section deletes
the PyTorch surface), and the coexistence is bounded. The repo
never directly supports both implementations at once on `main`.

Merge mechanics: per-section PRs are **squash-merged** into
`jax_migration` (the repo's policy disallows merge commits). Each
section becomes one commit on `jax_migration`. The final
`jax_migration → main` PR likewise squash-merges, so `main`
ultimately gains one commit per section plus one final
consolidation commit.

## 12. Per-section review and verification process

Each section decomposes into **chunks** — logical,
independently-reviewable units of work. The cadence within a
section is fixed:

1. **Implement one chunk.** Commit + push to the section branch
   (or to `jax_migration` directly for chunks small enough that the
   section is the whole branch).
2. **Single full subagent review wave on that chunk.** Spawn the
   six review lanes in parallel — `review-bug-detector`,
   `review-performance-analyzer`, `review-type-correctness`,
   `review-test-risk`, `review-simplification`, `review-doc-accuracy`
   — plus `codex review` via Bash (background). Synthesize
   findings, apply fixes that meet the "would a careful reviewer
   block the PR" bar (Critical / Important / SIGNIFICANT), skip
   nits. Commit + push the fixes as `fix(jax): round-1 review
   fixes (<chunk>)`. **One review wave per chunk** — no within-chunk
   iteration.
3. **Repeat for the next chunk.**
4. **Full-section `--loop` review.** Once every chunk has landed,
   re-run the six lanes + Codex across the whole section diff.
   Apply fixes, commit + push, re-run only the lanes that flagged
   significant issues last round; iterate until every running lane
   returns clean. Convergence typically takes 3–6 rounds.
5. **Open the section PR against `jax_migration`** (if a separate
   branch was used) or just confirm the section is complete on
   `jax_migration`.

### Section-level smoke test

Each section that touches a runnable surface includes a small
verification run, executed before the section's full `--loop`
review:

| Section | Smoke run |
|---|---|
| S3 (JAX core) | Convert each of `pawn-{small,base,large}` end-to-end and verify forward parity against `pawn._torch_legacy_fixture` on a real batch (`pawn.legacy` parity test). |
| S6 (Pretrain) | Pretrain `TINY_SUPERNET` for ≥1000 steps on Rust-generated random games. Verify loss decreases, no NaNs, all sliced variants forward-evaluate cleanly. |
| S7 (Adapters) | Train one adapter strategy (LoRA rank 4) for one epoch on a small Lichess Elo slice using `TINY_SUPERNET`. Verify val loss decreases, no NaNs. Smoke-run RoSA's 3-phase schedule on the same slice. |
| S8 (Eval) | Run move-accuracy + a probe + the generation diagnostic suite on a converted JAX checkpoint. Numbers within tolerance of the v1 PyTorch reference. |
| S10 (Sweep) | 3-trial subprocess Optuna sweep over LoRA rank ∈ {2,4,8} on the tiny variant; confirm metric extraction works and best-trial selection picks the lowest `val_loss`. |
| S16 (Final loop) | End-to-end smoke: pretrain tiny supernet, train an adapter on the resulting variant slice, evaluate. |

### Data caching

Data and weights downloaded for one section's verification are
cached and reused by later sections. Cache locations:

- **HuggingFace artifacts** — `$HF_HOME` (default
  `~/.cache/huggingface`). Always use
  `huggingface_hub.hf_hub_download` / `snapshot_download`; both
  cache by content hash and never re-download an unchanged
  artifact.
- **Tokenized Lichess cache** — `$HF_HOME/pawn-lichess-cache/<key>/`
  (v1 convention).
- **Rust-engine random games** — generated on-the-fly per section;
  not downloaded, not cached.

## 13. Re-implementation plan: sections and chunks

The migration re-implements as 15 sections on `jax_migration`,
each landed via the per-section review process in §12. Order is
chosen so each later section can lean on the earlier ones; the
ordering also fixes one foundational missing piece (the pydantic
config + MetricsLogger restoration in S4) before anything that
would otherwise re-introduce the same drift.

### S1 — Master plan (this document)

Reconcile the original plan, the v2-parity-gap audit, and the
actual state of the codebase into a single master document at
`docs/jax-migration.md`. Update CLAUDE.md to point to it. The
parity-gap audit (`docs/v2-parity-gaps.md`) is folded into §8 of
this doc and deleted as a separate file at the end of S14.

### S2 — Foundation

Chunks:

- **S2.1 Repo metadata.** `pyproject.toml`: add JAX core deps
  (`jax`, `equinox`, `optax`) to `[project.dependencies]`; add
  extras (`rocm`, `cu128`, `torch-loader`, `dashboard`, `lab`,
  `wandb`, `data-tools`). Adjust `tool.uv.sources` for the dual
  GPU index. Update `requires-python` if needed.
- **S2.2 CLAUDE.md JAX rewrite.** Replace v1 CLAUDE.md with a JAX
  layout map; preserve the v1 disclaimer for pre-existing metrics
  (a one-line note that the published `pawn-{small,base,large}`
  metrics on HF come from v1 PyTorch runs).
- **S2.3 .gitignore + minor infra.** `rust_out`, JAX cache dirs,
  etc.

### S3 — JAX core

Chunks:

- **S3.1 `pawn/config.py`.** `ModelConfig` + `SUPERNET` +
  `TINY_SUPERNET` + `VARIANTS` + `TINY_VARIANTS` + `validate_nested`.
- **S3.2 `pawn/model.py`.** Equinox `PAWNModel`: RMSNorm, SwiGLU,
  RoPE, factored embeddings, `lax.scan` over stacked layers, plain
  attention, slice extraction.
- **S3.3 `pawn/_sentinel.py` + `pawn/checkpoint.py`.** Atomic
  safetensors save/load, HF push, `.complete` sentinel +
  SHA-256 verification.
- **S3.4 `pawn/legacy.py` + `pawn/_torch_legacy_fixture.py`.**
  One-time PyTorch → JAX converter for the published v1
  checkpoints, plus the frozen torch reference architecture
  used by parity tests.
- **S3.5 `pawn/torch_loader.py`.** Thin PyTorch loader for
  external non-JAX consumers.
- **S3.6 Tests.** `tests/test_jax_model.py`, `tests/test_jax_checkpoint.py`,
  `tests/test_jax_legacy.py`, `tests/test_jax_torch_loader.py`.

Verifiable: instantiate at supernet dims, slice cleanly to each
variant; legacy converter's output matches `_torch_legacy_fixture`
logits within tolerance (`1e-3` on fp32 for `pawn-large`, tighter
for toy configs).

### S4 — `run_config` + `MetricsLogger` (THE KEYSTONE)

This section lands the load-bearing infrastructure both trainers
sit on. **It must precede S6 and S7** so we never re-create the
scattered-`SystemExit` and ad-hoc-`metrics.jsonl` drift.

Chunks:

- **S4.1 `pawn/run_config.py`.** Port v1 pydantic models forward:
  `BaseRunConfig` + `PretrainConfig` + `AdapterConfig` +
  `SpecializedCLMConfig`. Drop `CotrainConfig` /
  `CotrainVariant`. Add v2 fields (`supernet`, `variant`,
  `rosa_warmup_frac`, `rosa_top_k_frac`, etc.). **Restore v1
  field names** for the cosmetic renames (§8.3). Keep all v1
  `model_validator(mode="after")` cross-field invariants
  (LR-schedule shape, RoSA phase fractions, bottleneck no-attn AND
  no-ffn no-op, etc.). Expose `model_json_schema()` for the lab.
- **S4.2 `pawn/logging.py`.** Port v1 `MetricsLogger` forward:
  `log_config`, `log_train`, `log_val`, slug-based run-dir, NaN/Inf
  sanitisation, per-record flush, baseline metadata. Drop
  `torch.cuda.*` memory branch; replace with `psutil` (host) +
  `nvidia-smi`/`rocm-smi` shell-out (GPU).
- **S4.3 Tests.** `tests/core/test_run_config.py` (the v1 tests,
  ~930 LoC) + `tests/core/test_logging.py` (the v1 tests, ~493 LoC).
  Adjust for v2 field names where they legitimately differ.

Verifiable: `PretrainConfig.model_json_schema()` returns valid
JSON Schema; `MetricsLogger` round-trips a synthetic run through
JSONL with the v1 record shape; NaN loss writes `null`.

### S5 — Corpus

Chunks:

- **S5.1 `pawn/corpus.py`.** Rust-engine corpus →
  `Corpus(tokens, attn_mask, targets, loss_mask, outcome_offset)`,
  with the int16 → int32 boundary widening.
- **S5.2 `pawn/lichess_cache.py` + `pawn/lichess_data.py`.** v1
  Lichess Elo-stratified pipeline kept as-is (framework-agnostic
  polars + Rust-engine pipeline). Only consumer-side changes are
  in S7 / S8.
- **S5.3 Tests.** `tests/test_jax_corpus.py` +
  preserve `tests/model/test_lichess_data.py` /
  `test_lichess_cache.py` (relocated under `tests/data/`).

### S6 — Pretraining trainer

Chunks:

- **S6.1 `pawn/trainer.py`.** JAX trainer: AdamW + warmup-cosine,
  `optax.clip_by_global_norm(1.0)`, K-step `lax.scan` chunks,
  joint multi-variant supernet loss (§5.3), checkpoint cadence.
  All metric writes go through `MetricsLogger`. All config
  validation is via `PretrainConfig`.
- **S6.2 Tests.** `tests/test_jax_trainer.py`. Cover the `state.step`
  scalar pin, the `optax.warmup_cosine_decay_schedule` `decay_steps`
  contract, the padded-batch AdamW weight-decay guard, the
  gradient clipping, the supernet joint loss.

Smoke: `TINY_SUPERNET` × 1000 steps on Rust-generated random games;
loss decreases monotonically; all sliced variants
forward-evaluate.

### S7 — Adapter trainer + 8 strategies

Chunks:

- **S7.1 `pawn/adapters/__init__.py` + per-strategy modules.**
  All 8 strategies (`lora`, `film`, `unfreeze`, `bottleneck`,
  `hybrid`, `sparse`, `rosa`, `specialized_clm`). v1 names
  restored (`lora_rank`, `density`, `use_output_film`,
  `no_adapt_attn`, `no_adapt_ffn`, `specialized_clm.{d_model,n_layers,
  n_heads}`).
- **S7.2 `pawn/adapter_trainer.py`.** Two-tier frozen/trainable
  PyTree, K-step `lax.scan`, jitted forward-only val function,
  per-strategy gradient mask (only `unfreeze` passes a mask
  today). RoSA three-phase schedule: Phase 1 (LoRA warmup) →
  Phase 2 (one-shot grad-magnitude mask gen via
  `compute_phase2_mask`) → Phase 3 (joint training under fixed
  mask). All metrics through `MetricsLogger`; all config
  validation through `AdapterConfig`.
- **S7.3 Tests.** `tests/test_jax_adapters.py`,
  `tests/test_jax_adapters_bottleneck_hybrid_sparse.py`,
  `tests/test_jax_adapters_extra.py`,
  `tests/test_jax_adapters_rosa_specialized.py`,
  `tests/test_jax_adapter_trainer.py`. Pin the two-tier
  partition invariant (every array field of
  `state.trainable.backbone` is `None` after partitioning).

### S8 — JAX eval surface

Chunks:

- **S8.1 `pawn/eval.py` (move-accuracy + per-phase breakdown).**
- **S8.2 `pawn/probes.py` (linear probes via Optax fit).**
- **S8.3 `pawn/generation.py` (generation diagnostics + KV-cache
  + variable-prefix-length grouping).** Extend the
  `outcome_prefix_trained` gate to `outcome_signal_test`,
  `prefix_continuation_test`, `poisoned_prefix_test` (§5.1 from
  v2-parity-gaps).
- **S8.4 `pawn/lichess_eval.py` (Elo-stratified Maia-style eval).**
- **S8.5 Tests.** `tests/test_jax_eval.py`,
  `tests/test_jax_probes.py`, `tests/test_jax_generation.py`,
  `tests/test_jax_lichess_eval.py`.

### S9 — Eval-suite legacy (`bounds` + `viz` + `corpus`)

Restore v1's `pawn/eval_suite/{bounds,viz,corpus}.py` under the
`data-tools` extra. `bounds.py` computes theoretical accuracy
bounds from a position-level parquet corpus; `viz.py` is the
matplotlib/seaborn plotting helpers; `corpus.py` is the polars
parquet iterator they depend on. All three are off-the-hot-path
tooling; the JAX surface (S3–S8) is unaffected.

Chunks:

- **S9.1 `pawn/eval_suite/corpus.py` (polars iterator).**
- **S9.2 `pawn/eval_suite/bounds.py` (theoretical accuracy bounds).**
- **S9.3 `pawn/eval_suite/viz.py` (matplotlib/seaborn plots).**
- **S9.4 Tests.** `tests/eval/test_bounds.py`,
  `tests/eval/test_corpus.py`, `tests/eval/test_viz.py` (v1
  tests, gated on the `data-tools` extra).

### S10 — Standalone sweep driver

Chunks:

- **S10.1 `pawn/sweep.py`.** Port v1's `AdapterObjective`
  (subprocess) + the metric-parsing logic. Now that S4 has
  restored the `type` field, parsing flips back to
  `record["type"] == "val"`. Pruning hookup
  (`trial.report(val_loss, step) + should_prune()`).
- **S10.2 `scripts/sweep.py` CLI driver.** `--strategy lora
  --n-trials 50 --storage sqlite:///sweeps/lora.db`. Persistent
  study state.
- **S10.3 Tests.** `tests/training/test_sweep.py` (v1) +
  `tests/training/test_sweep_rosa_ratio.py`. Defer
  `InProcessRoSAObjective` to a follow-up.

### S11 — Lab + dashboard

Chunks:

- **S11.1 `pawn/lab/*`.** Bring over from backup; rewire
  `lab_schema` to return `PretrainConfig.model_json_schema()` +
  `AdapterConfig.model_json_schema()` directly (delete the
  hand-rolled dict).
- **S11.2 `pawn/dashboard/*`.** Bring over from backup; with
  S4's restored `type` field the train-vs-val chart split
  becomes automatic again.
- **S11.3 Tests.** `tests/lab/*` + `tests/lab/test_dashboard_*`.

### S12 — Scripts

Chunks:

- **S12.1 `scripts/train_jax.py`.** Both bare CLI flags and
  `--config <json>` (parsed through `PretrainConfig`). The
  scattered `raise SystemExit` guards (37 of them in the first
  attempt) collapse into pydantic-validator failures.
- **S12.2 `scripts/train_jax_adapter.py`.** Same shape with
  `AdapterConfig`. Strategy dispatch via `--strategy` (8
  strategies); RoSA three-phase schedule controlled by
  `--rosa-warmup-frac` / `--rosa-top-k-frac`.
- **S12.3 `scripts/eval_jax.py` + `scripts/eval_probes_jax.py` +
  `scripts/eval_generation_jax.py`.**
- **S12.4 `scripts/convert_published_checkpoints.py`.** One-shot
  converter driving `pawn.legacy.convert_legacy_checkpoint`.
- **S12.5 Data scripts.** Bring over the 6 swept-up data /
  eval / publishing scripts (`compute_theoretical_ceiling.py`,
  `extract_lichess_parquet.py`, `generate_lc0_data.py`,
  `generate_model_cards.py`, `eval_vs_stockfish.py`,
  `export_hf_repo.py`). Each gated on its needed extras
  (typically `data-tools`).
- **S12.6 Tests.** `tests/scripts/test_train_jax.py`,
  `tests/scripts/test_train_jax_adapter.py`,
  `tests/scripts/test_eval_jax.py`,
  `tests/scripts/test_convert_published_checkpoints.py`,
  `tests/scripts/test_script_smoke.py`.

### S13 — Engine fixes + diagnostics gate

Chunks:

- **S13.1 PAD-token init.** Extend the
  `engine/src/lib.rs:parse_pgn_lichess` PAD_TOKEN seed fix to
  `parse_pgn_enriched` (line ~879),
  `parse_pgn_lichess_filtered` (line ~1203), and
  `uci_moves_to_tokens` (line ~773). None are called from v2
  Python today but they share the same bug.
- **S13.2 Engine tests.** Restore / extend `tests/test_enriched_pgn.py`
  if any Python consumer still calls into those paths;
  otherwise add Rust-side unit tests.

(The `outcome_prefix_trained` gate extension lands in S8.3, not
here.)

### S14 — Docs + Docker + deploy

Chunks:

- **S14.1 docs/jax-migration.md.** Final pass — this document.
- **S14.2 docs/ADAPTERS.md, docs/TRAINING.md, docs/ARCHITECTURE.md,
  docs/ACCURACY_CEILING.md.** Update to JAX state. Mark v1
  metrics with the "v1 PyTorch run; new JAX runs pending"
  disclaimer.
- **S14.3 README.md.** Top-of-fold mention of JAX. v1 metrics
  preserved with disclaimer.
- **S14.4 docs/LEGACY.md.** Keep as historical record of the v1
  PyTorch surface; update to note that v1 is reachable via the
  `pre-jax-migration` git tag.
- **S14.5 Delete docs/v2-parity-gaps.md.** Folded into §8 of this
  doc. No longer a standalone source of truth.
- **S14.6 Dockerfile.** Bake `--extra data-tools` into the
  runtime image's `uv sync` lines so the published image can run
  the data scripts without manual sync. Decide
  dashboard/lab/wandb inclusion per-image-tag (e.g. `:dev`
  includes them, `:runtime` doesn't).
- **S14.7 deploy/pod.sh + deploy/vast.sh.** Confirm `--logs-dir`
  flag, JAX entry points, image tags.

### S15 — Comprehensive tests

Chunks:

- **S15.1 Test layout.** Re-organize `tests/` so the JAX surface
  is under `tests/` (the v1 `tests/{model,core,eval,adapters,lab,
  training}` subdirs are renamed or merged as appropriate).
- **S15.2 Full suite green.** `uv run --extra rocm pytest tests/`
  passes end-to-end. CI mirrors this.

### S16 — Final full-loop review

Chunks:

- **S16.1 Full-tree subagent review.** All 6 lanes + codex
  against the entire `origin/main..jax_migration` diff. Iterate
  to clean.
- **S16.2 Final smoke run.** Pretrain `TINY_SUPERNET`, train a
  LoRA adapter on a tiny Lichess slice, evaluate. Attach the
  output to the framework-swap PR.
- **S16.3 Open framework-swap PR.** `jax_migration → main`.

## 14. Resolved decisions

| Question | Decision | Reference |
|---|---|---|
| Framework | All-JAX, PyTorch removed from training/eval surface; thin loader + legacy converter survive | §1, §3.5 |
| Local GPU | JAX-on-ROCm accepted, preferred over two pipelines | §9 |
| Model framework | Equinox | §3.1 |
| Precision (v2) | bf16 compute, fp32 master + Adam state; fp8 deferred | §1 |
| External interop | Thin, dependency-light PyTorch loader (`pawn.torch_loader`, `--extra torch-loader`) | §3.5 |
| Multi-variant training | One shared-weight supernet; variants are nested slices; replaces `cotrain` | §5 |
| Per-variant loss weighting / sandwich schedule | Tunable, not a design blocker; v2 trains all three every step | §5.3 |
| Pydantic `run_config` | **RESTORED** with v1 field names + v2 fields added | §8.1, S4 |
| `MetricsLogger` | **RESTORED** with v1 schema; psutil + shell-out replaces torch.cuda | §8.2, S4 |
| Cosmetic v1 → v2 renames | **REVERTED**: `lora_rank`, `density`, `use_output_film`, `no_adapt_attn`, `no_adapt_ffn`, specialized-clm names | §8.3 |
| Substantive v1 → v2 changes | KEPT (`lora_targets: list[str]`, `rosa_warmup_frac`, etc.) | §8.4 |
| `pawn/cotrain.py` | GONE BY DESIGN — supernet replaces it | §2 |
| `pawn/gpu.py` | GONE BY DESIGN — JAX manages its own GPU config; port the `PAWN_ALLOW_CPU=1` escape hatch into the JAX entry points | §2, §9 |
| Published v1 HF repos (`pawn-{small,base,large}`) | Stay frozen; v2 publishes to new repos; legacy converter handles backward compatibility | §3.5, §5.5 |

## Appendix: performance reasoning

- **Required data-ingest bandwidth** is `peak · MFU / (3N)`,
  derived from `bytes_per_step = B · seq · 2` and
  `t_step = 6N · B · seq / (peak · MFU)`; `B` and `seq` cancel.
  For PAWN-base this is ~10 MB/s.
- **Throughput today (v1 PyTorch):** ~15% MFU on a B200, ~11
  steps/s for base.
- **JAX + fused loop + resident corpus (bf16):** estimated ~45–55%
  MFU, ~3× over today. fp8 compute would add a further ~1.5–1.8×
  — deliberately deferred past v2.
- **Frozen-backbone adapter training:** ~4N FLOPs/token vs ~6N for
  full finetuning — a ~33% cut, obtained automatically from the
  two-tier PyTree.
- **Supernet training:** ≈ 1.66× a large-only step (sum of variant
  FLOPs), FLOP-identical to three separate runs; the win is
  operational and storage (~1.66× less weight + optimizer state),
  not raw training FLOPs.
