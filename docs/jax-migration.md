# PAWN → JAX migration

> **Status:** **feature-complete on `jax_migration`; the final
> `jax_migration → main` framework-swap PR (#111) is open for human
> review.** Phases 1–4 (model + checkpoint + converter + pretraining
> trainer + adapter trainer + the full ported eval suite + PyTorch
> removal) have all squash-merged into the `jax_migration` integration
> branch. The remaining trainer-side follow-ups (adapter trainer
> dispatch glue, KV-cache generation, variable-prefix-length grouping,
> stale-doc cleanup, deploy-script log-dir flag) land on the
> integration branch as additional commits before the framework swap.
> `main` never carries the half-migrated state. See §10.
>
> This document describes a from-scratch redesign of PAWN's training stack
> (pretraining, adapter training, evaluation) onto a single all-JAX pipeline,
> motivated by compute efficiency and by eliminating the dual-framework
> maintenance burden.

## Invocation

This document predates the `/review-driven-development` skill, so the
original phase-by-phase implementation followed an earlier ad-hoc
cadence. From Phase 4-followup onward the work follows the skill's
contract; `--resume docs/jax-migration.md` re-enters that workflow.

- Feature slug: `jax-migration`
- Master feature branch: `jax_migration` (long-lived integration
  branch, cut once from `main`)
- Effective flags (after defaults applied):
  - `--plan-path`: `docs/jax-migration.md`
  - `--no-review-plan`: `false` (the plan was reviewed inline as
    Phases were authored)
  - `--loop-chunks`: `false` (single review wave per chunk; matches the
    cadence used through Phase 4)
  - `--loop-sections`: `true`
  - `--loop-final`: `true`
  - `--pr-after-fixes`: `false` — the framework-swap PR (#111) was
    opened manually and stays open per user directive ("don't merge
    the full PR yet; summarize the work and I'll review the full-fat
    PR before merging").

## 1. Motivation

PAWN-base is small: ~35.8M parameters, 512-token sequences, a 1,980-token
vocabulary. A training step is only ~30 TFLOP, yet the current PyTorch stack
runs at roughly 15% MFU — most of the machine is idle. That idle time is not
going into matmuls; it goes into kernel-launch latency, framework dispatch,
host round-trips, and the CPU/PCIe data path.

For a model this small the design should optimize *those* costs and treat the
matmuls as nearly free. The redesign does four things:

1. **Whole training loop as one compiled program** — eliminates per-step launch
   and dispatch overhead.
2. **Data resident on (or streamed to) the device** — eliminates the per-step
   CPU/PCIe path and live data generation.
3. **A single framework** — JAX everywhere, so there is one model definition,
   one optimizer stack, and no PyTorch/JAX boundary to maintain.
4. **One supernet instead of three variants** — small, base, and large are
   extracted as nested slices of a single shared-weight model (§5).

### Decision: JAX-only, eval included

JAX-on-ROCm locally is accepted, and a single framework is strongly preferred
over two. That has a scope consequence: if pretraining **and** adapter training
are JAX, keeping `eval_suite` in PyTorch *is* maintaining two frameworks. So the
end state is **JAX-only — PyTorch removed entirely, eval included.** The eval
model is the same Equinox model; probes, accuracy, and generation are forward
passes plus a small linear fit.

### Precision

v1 keeps precision unchanged from today: **bf16 compute** (fp32 accumulate),
**fp32 master parameters**, **fp32 Adam state**. fp8 compute is a documented
future lever (analysis suggests ~1.6× throughput left on the table) but is
explicitly out of v1 to de-risk convergence.

## 2. Scope

| Migrates to JAX | Unchanged (framework-agnostic) |
|---|---|
| `pawn/model.py` — Equinox `PAWNModel` supernet | Rust `engine/` — all chess logic, tokenization, legal-mask replay |
| `pawn/trainer.py` — fused training loop | `deploy/`, Docker images |
| `pawn/corpus.py` — corpus generation/loading | |
| `pawn/checkpoint.py` — PyTree serialization | |
| `pawn/adapters/` — all 8 strategies (lora / film / unfreeze / bottleneck / hybrid / sparse / rosa / specialized_clm) | |
| `pawn/eval.py` (move-accuracy), `pawn/probes.py` (linear probes), `pawn/generation.py` (generation diagnostics), `pawn/lichess_eval.py` (Elo-stratified eval) — ported from `pawn/eval_suite/` | |

`pawn/config.py` is the canonical config module post-migration; it
defines `ModelConfig` plus the `SUPERNET` / `TINY_SUPERNET` / `VARIANTS`
/ `TINY_VARIANTS` constants (§5.1). The legacy `CLMConfig` /
`TrainingConfig` dataclasses were removed alongside the PyTorch surface
in Phase 4.

## 3. Shared core

These components are used identically by pretraining, adapter training, and
evaluation.

### 3.1 Model — Equinox

`PAWNModel` (the JAX-side counterpart to the legacy PyTorch `PAWNCLM`) is an
Equinox module (a PyTree of arrays): RMSNorm, SwiGLU, RoPE,
and factored embeddings (`src_embed[s] + dst_embed[d] + promo_embed[p]`). The
model is stored at the **supernet** dimensions; the three variants are nested
slices of it (§5).

- The transformer layers are applied with `lax.scan` over weights stacked on a
  leading axis — one compiled layer body, fast compile.
- **Plain attention** (`softmax(QK^T)V`, scores materialized) rather than a
  fused flash kernel. At seq 512 attention is ~12% of step FLOPs, and plain
  attention sidesteps the maturity of fused attention kernels under JAX-on-ROCm.

Equinox is the chosen framework: PyTree-native, plays cleanly with `lax.scan`,
`lax.fori_loop`, and buffer donation, with minimal framework magic.

### 3.2 Optimizer — Optax

Optax `adamw`. Forward casts parameters to bf16 with fp32 accumulation; the
master copy and Adam moments stay fp32 (see §1, Precision).

### 3.3 Whole loop as one program

The training loop is a `lax.scan` over K steps inside a single
`@eqx.filter_jit` function, with buffers donated. The Python host drives the
loop in chunks of K steps; between chunks it flushes metrics, writes
checkpoints, and (for adapters) runs validation. The per-step body never
returns to the host.

K is chosen to amortize host overhead; for adapters it is additionally bounded
by the validation cadence (`K ≤ val_every`).

### 3.4 Two-tier parameter tree

The model PyTree is partitioned into `frozen` and `trainable`. `jax.grad` /
`eqx.filter_grad` differentiates only `trainable`; XLA dead-code-eliminates the
gradient computations for `frozen` parameters.

- **Pretraining:** everything is `trainable`.
- **Adapter training:** the backbone is `frozen`, adapter parameters are
  `trainable` (for `unfreeze`, the top-N backbone layers are also `trainable`).

One mechanism, two configurations.

### 3.5 Checkpoints and interop

The atomic write contract is preserved: write to a `.tmp` directory, rename,
and emit a `.complete` sentinel containing SHA-256 hashes of every file.
Hashes are always verified on load. Async HF push is unchanged.

The JAX PyTree is serialized to **safetensors** (a framework-neutral file
format) under a documented canonical parameter-name schema.

- **Publishing.** The supernet is sliced into three standalone variant
  checkpoints at publish time (§5.5); downstream consumers see ordinary
  independent checkpoints.
- **Thin PyTorch loader.** A small, dependency-light PyTorch module is shipped
  alongside the published checkpoints. It reads the safetensors weights via the
  canonical name schema into a minimal `nn.Module`, so external non-JAX users
  can load PAWN weights without taking a JAX dependency. This is loader-only —
  not a training path.
- **Legacy converter.** The three already-published checkpoints
  (`pawn-{small,base,large}`) were trained independently and cannot be
  retro-fitted into a supernet. A one-time converter produces JAX-loadable
  versions of them for backward compatibility; the new supernet is a fresh
  training artifact (Phase 2).

## 4. Pretraining path

### 4.1 Corpus

The corpus is **pre-generated offline** by the existing Rust engine
(`generate_random_games()`). The engine returns int16 tokens, int16
game lengths, and uint8 termination codes; `pawn.corpus` widens
them at the boundary to a packed `Corpus` of int32 `tokens [N, T]`,
bool `attn_mask [N, T]`, int32 `targets [N, T]` (input shifted left by
one), bool `loss_mask [N, T]`, and uint8 `outcome_offset [N]`. `N =
total_steps × batch_size`, so every step sees a fresh game in a
single pass.

### 4.2 No shuffling

Random games are i.i.d. by construction — each game's seed is a hash of its
index, so consecutive indices are statistically independent. Under single-pass,
no-reuse consumption, sequential order is statistically identical to random
sampling and strictly simpler:

- **Trivial resume:** the offset is `step × batch_size`; no RNG to seed or log.
- **Faster I/O:** each chunk is a contiguous slab, not a scattered gather —
  sequential reads are ~3–5× faster and benefit from OS readahead.
- **Auditable:** batch `s` is a known, fixed range of game indices.

Conditions (all easy to honor): single pass; homogeneous on-disk order — do
**not** sort the corpus by game length (length correlates with outcome); store
in generation/index order. If the corpus ever concatenates distinct
distributions, interleave them round-robin instead of consuming sequentially.

### 4.3 Data path

The corpus lives in host RAM (or an NVMe memmap). A double-buffered prefetch
thread stages the next chunk (`K · batch_size · T · 10` bytes;
~262 MB at K=200, B=256, T=512 — int32 tokens + bool attn_mask +
int32 targets + bool loss_mask = 4+1+4+1 bytes per position)
to the device while the current chunk trains.

The required ingest bandwidth has a closed form, independent of batch and
sequence length:

```
BW_required = peak_FLOP/s · MFU / (3 · N)
```

For PAWN-base on a B200 (peak ≈ 2.2 PFLOP/s, MFU ≈ 0.5, N = 35.8M) this is
**~10 MB/s** — 500–2500× below PCIe or sequential-NVMe bandwidth. The prefetch
is provably hidden behind compute; a model would need to shrink below
~10–15K parameters before I/O could bind. Instrument a data-wait timer and a
prefetch-starvation counter in `metrics.jsonl` to confirm this empirically.

### 4.4 Loop

A fixed `total_steps` → a single clean `lax.scan` of K-step chunks. The host
loop flushes metrics and checkpoints between chunks. `schedule_health.json`
(planned vs actual steps) carries over unchanged.

## 5. Variant scheme: shared-weight supernet

PAWN ships three variants — small, base, large. Rather than three independent
training runs, the redesign trains **one supernet** from which the variants are
extracted as nested slices — a MatFormer- / Matryoshka-style scheme (cf.
Devvrit et al., *MatFormer*, 2023). This replaces the current `cotrain`
run-type and `configs/cotrain_three_variants.json`: "cotrain" becomes "train
the supernet."

### 5.1 Nested variant dimensions

For the variants to be exact slices, their dimensions must nest. This requires
redefining them around a fixed head dimension:

| Variant | d_model | layers | heads (= d_model / 64) |
|---|---|---|---|
| small | 256 | 8 | 4 |
| base | 512 | 8 | 8 |
| large (= supernet) | 640 | 10 | 10 |

`head_dim` is fixed at 64, so for supernet-nested models `n_heads` is pinned to
`d_model / 64`. (`n_heads` stays an explicit `ModelConfig` field — standalone
models such as converted legacy checkpoints keep their own head count.) The only
material redefinition is large's head count (8 → 10); this is
**parameter-count-neutral** — the q/k/v/o matrices are
`d_model × d_model` regardless of how `d_model` is partitioned into heads — and
it keeps RoPE identical across variants. `d_ff` is sliced alongside `d_model`
(each variant keeps its SwiGLU ratio; the ratios must nest).

The variants are `ModelConfig` instances in `pawn/config.py` — a single
`SUPERNET` config plus a `VARIANTS` dict.
Per existing project practice, model cards and
published-checkpoint docs derive parameter counts from `config.json` — they are
not hardcoded, so the slight redefinition does not require manual edits.

### 5.2 What gets sliced

The supernet is stored at large's dimensions (d=640, 10 layers). Variant V with
width `d_V` and depth `L_V` uses:

- Every weight matrix: the `[:d_V, :d_V]` (or `[:d_V, :d_ff_V]`) prefix.
- Factored embeddings (`src/dst/promo`): the `[:, :d_V]` column prefix.
- `lm_head` and the final RMSNorm: the `[:d_V]` prefix, applied after layer
  `L_V`.
- The layer stack: layers `[0:L_V]`.

RMSNorm normalizes over the *active* width at run time. Because each variant's
forward during training uses exactly its slice, the sliced weights are trained
to function at that width — the slimmable-network / MatFormer property.

`pawn.config.validate_nested(variant, supernet)` enforces the nesting
invariant — equal `head_dim`, identical vocab / context / outcome
layout, and no axis exceeding the supernet — and is called by
`pawn.model.sliced()` before extraction.

### 5.3 Joint training

Each step, on the same batch, the jitted step computes the next-token loss for
each variant at its slice and **sums the losses** (with per-variant weights);
gradients accumulate into the one shared weight tensor. The variants have
different widths, so they do not `vmap` into a single batched matmul — the step
is a static unroll of three forward calls.

The gradient structure is naturally Matryoshka-coherent: the inner `[:256]`
channels receive gradient from all three variants, `[256:512]` from base and
large, `[512:640]` from large alone — the inner channels carry the most signal,
exactly as the nesting intends.

**Cost.** Training all three each step costs the sum of their FLOPs (≈ 1.66× a
large-only step), which is FLOP-identical to three separate runs of the same
length — the supernet does not save raw training FLOPs by default.

**Cost knob.** Every variant's weights are a *subset* of large's, and large is
the full set, so training large every step already supplies gradient to every
weight. The explicit small/base losses only *shape* those slices into good
reduced-width standalone models. A sandwich-rule schedule — large every step,
small and base every Nth step — therefore trades a little variant quality for a
genuine FLOP reduction below the three-separate-runs baseline. v1 default:
train all three every step; the schedule is a tunable, not a design open
question.

### 5.4 Coupling constraint

The supernet shares one `lm_head` and final RMSNorm across both widths and
depths — so that head must decode layer-8 hidden states (small, base) *and*
layer-10 hidden states (large). This is the early-exit / layer-dropping
training regime; it is trainable but it couples the variants, and the
loss-summation introduces per-variant loss-weighting hyperparameters. These are
accepted costs of the scheme.

### 5.5 What it wins, and slicing out checkpoints

Relative to three independent runs:

- **One run** — one schedule, one data pipeline, one checkpoint stream.
- **Shared weights and optimizer state** — store ~68.4M parameters' worth, not
  ~113.7M (≈ 1.66× less).
- **Guaranteed-consistent variants** — same data, same step count, nested
  representations; directly comparable, which is exactly what a finetuning
  testbed wants.
- **Intermediate widths** are extractable without extra training
  ("Mix'n'Match"); the smaller variants double as draft / early-exit models.

At publish time the supernet is **sliced into three standalone safetensors
checkpoints** (`pawn-small`, `pawn-base`, `pawn-large`), each with its own
`config.json`. Downstream consumers — adapters, eval, external users — see
ordinary independent checkpoints and never need the supernet.

## 6. Adapter path

Adapter training reuses the shared core; it adds a frozen backbone and a finite
dataset. The frozen backbone is an ordinary **sliced variant checkpoint** (§5.5)
— adapter training needs no awareness of the supernet. Design deltas relative
to pretraining:

| Aspect | Pretraining | Adapters |
|---|---|---|
| PyTree | all trainable | **frozen backbone / trainable adapter** |
| Dataset | infinite i.i.d. stream | **finite** Elo-filtered Lichess cache |
| Shuffling | none | **per-epoch index permutation** (finite + multi-epoch) |
| Residency | chunk-streamed | whole cache VRAM-resident (~512 MB typical) — no prefetcher |
| Termination | fixed `total_steps` | **adaptive** — host checks patience / best-val between chunks |
| Validation | optional | **required** — a second jitted, forward-only function |

The cache-first model is unchanged in spirit: the first run with a given
`(Elo, min_ply)` builds the tokenized cache; the cache key logic is untouched.

### 6.1 Performance optimizations

**1. Frozen-backbone autodiff → automatic ~33% compute cut.**
Training is ~6N FLOPs/token = 2N forward + 2N activation-gradient + 2N
weight-gradient. Differentiating only the adapter parameters lets XLA
dead-code-eliminate the backbone weight-gradients (~2N), leaving **4N**.
Activation gradients still flow through every layer (needed to reach the lowest
adapter), so per-layer adapters land at exactly 4N. This is free — it follows
from the two-tier PyTree split.

**2. `vmap`'d sweep populations — the testbed's headline win.**
PAWN is a testbed for finetuning methods; the dominant adapter workload is
sweeps. A population of P adapters shares the *frozen backbone weights*.
`vmap`-ing the forward over the population folds P into the batch (`M`)
dimension of every backbone matmul, producing one larger, **higher-MFU** matmul
instead of P small ones — this directly attacks the small-matmul inefficiency
that caps this model's throughput.

- Sweepable at fixed graph shape via `vmap`: LR, weight decay, dropout, seed,
  LR schedule, illegal-penalty λ. Use `optax.inject_hyperparams` so the
  optimizer carries per-member hyperparameter *arrays*.
- Shape-changing knobs (LoRA rank, bottleneck dim, target layers) bucket into
  separate `vmap` groups.
- Aggressive Hyperband-style pruning loses most of its value: the dominant cost
  (the shared backbone) is paid once regardless of how many population members
  survive. Optuna can still drive the outer search — each suggested batch of P
  configs is one `vmap`'d run.
- P is bounded by activation memory (each member has distinct activations);
  trade per-member batch size for population size.

**3. Precompute everything frozen-and-input-only.**
For a fixed dataset, anything that is a pure function of (input tokens, frozen
weights) is computed once at cache-build time:

- **Legal masks** — replay all games in Rust once, store sparse legal indices
  device-resident alongside the tokens. `--disable-legal-mask` simply skips
  applying them.
- **Factored embeddings** — embed the corpus once (small, free).
- **`unfreeze` frozen prefix** — the bottom `8 − N` layers are frozen *and*
  adapter-free, so their output is a pure function of the input. Cache it and
  run only the top-N layers (forward and backward). Memory-gated: the cached
  activations are `d_model`-wide, ~512× the token corpus, so this is viable for
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

Adapter training uses **bf16 compute throughout** — never fp16. bf16's
fp32-range exponent eliminates the fp16-overflow failure mode that ceiling-scale
adapters previously hit.

## 7. Evaluation path

`eval_suite` ports to JAX:

- **Accuracy / generation** — forward passes plus argmax / sampling on the
  Equinox model.
- **Linear probes** — a small Optax fit on frozen hidden states.
- **Diagnostics** — forward passes.

Legal-mask logic already lives in Rust. This is the largest single chunk of
porting work but conceptually the simplest — it is mostly forward evaluation.

## 8. ROCm specifics

- **Phase 1 (current):** `jax`, `equinox`, `optax` are declared as base
  dependencies in `pyproject.toml`, pulling CPU `jaxlib` by default. This is
  what makes `uv run pyright pawn/` resolve on CI without an extra and keeps
  Phase-1 verification (logit-parity tests, checkpoint round-trip) running
  on CPU JAX.
- **Phase 2 (deferred):** GPU-backend extras will mirror today's torch
  pattern: `jax[rocm]` vs `jax[cuda12]`. uv cannot co-resolve them from one
  lockfile, so the `--extra rocm` / `--extra cu128` split will be retained.
- `jax.jit` replaces `torch.compile`; there is no SDPA-backend selection to
  manage (plain attention).
- Port the `configure_gpu()` CPU guard and the `PAWN_ALLOW_CPU=1` escape hatch.
- If something fails under ROCm, suspect our code first — historically every
  "ROCm bug" in this project turned out to be a build-artifact or dependency
  issue.

## 9. Module layout and migration phasing

The end state replaces the PyTorch modules in place. During the migration
branch a transient dual-framework state is unavoidable and acceptable; the
phasing keeps the repo coherent at each phase boundary.

1. **Core.** Equinox `PAWNModel` supernet, `ModelConfig` + `SUPERNET`/`VARIANTS`
   specs, checkpoint serialization, the thin PyTorch loader, and the legacy converter
   for the three already-published checkpoints.
   *Verifiable:* the model instantiates at supernet dims and slices cleanly to
   each variant; the legacy converter's output matches the PyTorch model's
   logits within tolerance.
2. **Pretraining = supernet training.** Corpus generator, prefetcher, fused
   training loop, joint multi-variant loss, slice-and-publish.
   *Verifiable:* loss curve tracks a PyTorch baseline run; each sliced variant
   runs.
3. **Adapters.** Two-tier PyTree, finite-dataset data path, validation
   function, `vmap`'d sweeps.
4. **Eval.** Port `eval_suite`; **remove PyTorch.** Single framework achieved.

PyTorch remains a dependency transiently through Phases 2–3 (eval still loads
converted checkpoints) and is removed only at the end of Phase 4. The phases
never appear on `main`; they accumulate on a long-lived integration branch
(§10), and the per-phase work cadence is fixed (§11).

## 10. Integration branch and merge strategy

The migration uses a long-lived integration branch **`jax_migration`**, cut
once from `main`. All four phases land as their own PRs targeted at
`jax_migration`, never at `main`. When all four phases have merged into
`jax_migration`, a single final PR merges `jax_migration` into `main` — that
PR is the project's only "framework swap" event.

The invariant: **`main` never carries the half-migrated state.** While JAX
matures, `main` stays single-framework (the existing PyTorch implementation).
`jax_migration` is the only branch where JAX and PyTorch coexist, and the
coexistence is bounded — Phase 4 deletes PyTorch and flattens what had
been the transient `pawn/jax/*` namespace up into `pawn/` before the
final merge. The repo never directly supports both implementations at
once on `main`. (The flatten landed in PR #106; on the integration
branch everything is now reachable from the top-level `pawn` package.)

Merge mechanics: per-phase PRs are **squash-merged** into `jax_migration`
(the repo's policy disallows merge commits). Each phase becomes one commit
on `jax_migration`; the squash body preserves the per-chunk + per-round
iteration table the PR description carries. The final `jax_migration → main`
PR likewise squash-merges, so `main` ultimately gains one commit per phase
plus one final consolidation commit.

## 11. Per-phase review and verification process

Each phase decomposes into **chunks** — logical, independently-reviewable
units of work (Phase 1's chunks were: ModelConfig + Equinox `PAWNModel`,
checkpoint serializer, legacy converter, thin PyTorch loader, pytest
suite). The cadence within a phase is fixed:

1. **Implement one chunk.** Commit + push to the phase branch.
2. **Single full subagent review wave on that chunk.** Spawn the six
   review lanes in parallel — `review-bug-detector`,
   `review-performance-analyzer`, `review-type-correctness`,
   `review-test-risk`, `review-simplification`, `review-doc-accuracy` —
   plus `codex review` via Bash. All backgrounded; the lanes operate
   read-only. Synthesize findings, apply fixes that meet the "would a
   careful reviewer block the PR" bar (Critical / Important /
   SIGNIFICANT), skip nits. Commit + push the fixes as
   `fix(jax): round-1 review fixes (<chunk>)`. **One review wave per
   chunk** — no within-chunk iteration.
3. **Repeat for the next chunk.**
4. **Full-phase `--loop` review.** Once every chunk has landed, re-run
   the six lanes + Codex across the whole phase diff. Apply fixes,
   commit + push, re-run only the lanes that flagged significant issues
   last round; iterate until every running lane returns clean.
   Convergence typically takes 3–6 rounds. (Phase 1 converged on round
   5; see PR #101's iteration table.)
5. **Open the PR against `jax_migration`.** The body includes the
   per-chunk + per-loop-round commit history and the training-run
   output (below).

### Training run on PR open

Every phase PR includes a **small training run** appropriate to the
branch's current state, executed when the PR is opened. The point is to
exercise the new code on a real (if tiny) workload beyond the
unit-level parity already in the test suite:

| Phase | Run |
|---|---|
| 1 | Convert each of `pawn-{small,base,large}` end-to-end and verify forward parity against the PyTorch reference on a real batch. Phase 1 has no trainer; this is the closest analogue. |
| 2 | Pretrain the tiny **nested** supernet ``TINY_SUPERNET`` for ≥1000 steps on Rust-generated random games. Verify loss decreases, no NaNs, all sliced variants forward-evaluate cleanly. ``TINY_SUPERNET`` (defined in ``pawn.config``) is ``(d_model=192, n_layers=4, n_heads=3, d_ff=768)`` with ``head_dim=64``, paired with ``TINY_VARIANTS`` at ``d_model ∈ {64, 128, 192}`` — three nested variants. (``pawn.config.TOY`` is *not* a nested supernet — it has ``head_dim=16`` and would fail ``validate_nested`` against ``TINY_SUPERNET``/``SUPERNET``; it is test-internal only.) The production ``SUPERNET`` is too large for a smoke run on commodity hardware. |
| 3 | Train one adapter strategy (e.g. LoRA rank 4) for one epoch on a small Lichess Elo slice. Verify val loss decreases, no NaNs. |
| 4 | Run a probe + move-accuracy eval (the ported JAX `eval_suite`) on a converted / published checkpoint. Numbers within tolerance of the PyTorch reference. |

The run's output — loss curves, final metrics, wall-time — is attached
to the PR body.

### Data caching

Data and weights downloaded for one phase's verification are cached and
reused by later phases. Cache locations:

- **HuggingFace artifacts** — published checkpoints
  (`pawn-{small,base,large}`), Stockfish / Lichess data repos, and any
  JAX-converted variants once published: `$HF_HOME` (default
  `~/.cache/huggingface`). Always use
  `huggingface_hub.hf_hub_download` / `snapshot_download`; both cache by
  content hash and never re-download an unchanged artifact.
- **Tokenized Lichess cache** — `$HF_HOME/pawn-lichess-cache/<key>/`
  (existing convention from the legacy pipeline).
- **Rust-engine random games** — generated on-the-fly per phase; not
  downloaded, not cached.

Phase verification scripts must not re-download an artifact a previous
phase already pulled. If a fresh download is genuinely needed (cache
eviction, format change), do it explicitly and one-time and note it in
the PR body.

## 12. Phase-4 followup chunks

After Phase 4 squash-merged into `jax_migration`, a handful of
trainer-side parity gaps remained — the framework swap was
feature-complete on the **evaluation** side but the adapter trainer
driver still only dispatched LoRA, and the generation path lacked a
KV-cache variant. The follow-ups land as additional chunks on the
integration branch under `/review-driven-development --resume
docs/jax-migration.md`. Status is per-chunk:

- **Deploy script log-dir flag (landed).** `deploy/pod.sh` and
  `deploy/vast.sh` `cmd_launch` now pass `--logs-dir` rather than the
  legacy `--log-dir` (which matched the now-deleted PyTorch
  `scripts/train.py`). CLAUDE.md's launch examples updated in lockstep
  — they no longer mention `--logs-dir` because the wrapper injects it
  silently; users **should not** pass `--logs-dir` in the command they
  hand to `pod.sh launch <name> …` or `vast.sh launch <name> …` or
  argparse will reject the duplicate.
- **Adapter dispatch glue (pending).** `scripts/train_jax_adapter.py`
  will gain a `--strategy {lora,film,unfreeze,bottleneck,hybrid,sparse,
  rosa,specialized_clm}` flag plus per-strategy hyperparameter args.
  The PyTree contract is the same across strategies —
  `adapter_filter(model)` returns a Python-bool spec consumed by
  `eqx.partition`; the `Unfreeze` strategy adds a companion per-layer
  `unfreeze_gradient_mask` plugged into `optax.masked`. `RoSA` will
  carry a three-phase host-driven training schedule (LoRA warmup →
  gradient-magnitude mask → joint training); the phase transitions
  re-jit naturally because the trainable subtree's bool spec changes
  between phases.
- **KV-cache generation (pending).** `pawn.generation.autoregressive_generate`
  will gain a KV-cached variant alongside the O(N²) recompute path
  the Phase-4 port currently ships. A bitwise-equivalence regression
  test against the recompute path pins parity; the KV-cache path is
  the default for long generations.
- **Variable-prefix-length grouping (pending).** The `poisoned_prefix`
  and `impossible_task` §6 generation tests currently inherit a legacy
  port bug — variable-length prefixes within a batch align under a
  batch-wide max, leaving PAD gaps in the attended context for rows
  with shorter prefixes (the engine state is still correct, but the
  model is conditioned on a PAD-padded context). The fix groups games
  by prefix length and runs each group as its own generation batch.

These follow-ups do not change any design decision in §1–§11; they
close out the work the integration branch had left as trainer-side
TODOs. Each chunk gets its own section branch off `jax_migration`,
squash-merges into the integration branch on close, and updates this
section's chunk status to "landed".

## 13. Resolved decisions

| Question | Decision |
|---|---|
| Framework | All-JAX, PyTorch removed; eval included (§1). |
| Local GPU | JAX-on-ROCm accepted, strongly preferred over two pipelines (§8). |
| Model framework | Equinox (§3.1). |
| Precision (v1) | bf16 compute, fp32 master + Adam state; fp8 deferred (§1). |
| External interop | Ship a thin, dependency-light PyTorch loader (§3.5). |
| Multi-variant training | One shared-weight supernet; variants are nested slices; replaces `cotrain` (§5). |
| Per-variant loss weighting / sandwich schedule | Tunable, not a design blocker; v1 trains all three every step (§5.3). |

## Appendix: performance reasoning

- **Required data-ingest bandwidth** is `peak · MFU / (3N)`, derived from
  `bytes_per_step = B · seq · 2` and `t_step = 6N · B · seq / (peak · MFU)`;
  `B` and `seq` cancel. For PAWN-base this is ~10 MB/s.
- **Throughput today:** ~15% MFU on a B200, ~11 steps/s for base.
- **JAX + fused loop + resident corpus (bf16):** estimated ~45–55% MFU,
  ~3× over today. fp8 compute would add a further ~1.5–1.8× — deliberately
  deferred past v1.
- **Frozen-backbone adapter training:** ~4N FLOPs/token vs ~6N for full
  finetuning — a ~33% cut, obtained automatically from the two-tier PyTree.
- **Supernet training:** ≈ 1.66× a large-only step (sum of variant FLOPs),
  FLOP-identical to three separate runs; the win is operational and storage
  (~1.66× less weight + optimizer state), not raw training FLOPs.
