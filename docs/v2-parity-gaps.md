# v2.0.0 → v1 parity gaps

What's still missing or changed on the `jax_migration` branch relative
to v1.x. Compiled from the multi-round review + the post-review audit
of what `7a6f3cd` (the Phase-4 "remove PyTorch" commit) actually swept
up.

## Status legend

- **MISSING** — was in v1, deleted in `7a6f3cd`, not yet restored on
  this branch.
- **RENAMED** — same semantics in v2, different name. Backward
  compatibility broken with no documented reason.
- **CHANGED** — v2 semantics differ from v1; document but don't revert.
- **GONE BY DESIGN** — explicitly out of scope for v2 per the
  migration plan (e.g. `cotrain` because supernet replaces it).
- **RESTORED** — was missing, brought back on this branch.

---

## 1. Modules removed without justification

### 1.1 `pawn/logging.py` — `MetricsLogger` — MISSING

**What it provided (v1, 327 LoC).** A JSONL metrics logger every
training entry point funnelled through. Each record carried:

- `type: "config" | "train" | "val"` (discriminator)
- `timestamp`, `elapsed`, `slug`, hostname, git hash
- Per-record system memory (`mem/system_rss_gb`, `mem/system_used_gb`,
  `mem/cpu_percent`), and GPU memory (`mem/gpu_peak_gb`,
  `mem/gpu_reserved_gb`, `mem/gpu_current_gb`)
- NaN/Inf sanitised to `None` (records stay valid JSON)
- Per-record flush (SIGKILL-durable)
- Slug-based run-dir naming (`run_20260520_140000_zesty-osprey`)

**v2 today.** Each training driver opens `metrics.jsonl` directly and
writes a flat-dict row per chunk. **No `type` field**, no baseline
metadata, no NaN guard, flush every 10 chunks. Config goes to a
sibling `config.json`. Slug is `YYYYMMDD_HHMMSS_microseconds_pid`.

**Knock-on effects.**

- `pawn/dashboard/metrics.py:127` reads `rec.get("type", "train")`,
  so **every v2 row falls into the "train" bucket**. The dashboard's
  train-vs-val chart split is broken against v2 runs.
- A NaN loss writes `"NaN"` literal into `metrics.jsonl`. `json.loads`
  rejects it. Any downstream tool that re-reads the file fails.
- Memory/CPU stats useful for diagnosing OOMs and slow leaks are
  simply not captured. Operator has to read `dmesg` after the fact.

**Restoration scope (~half a day).** Port `MetricsLogger`:

- Drop the `torch.cuda.*` memory branch; replace with `psutil` host
  memory (already a dep) and either no GPU stats (clean) or shell-out
  to `nvidia-smi` / `rocm-smi` like `pawn/lab/runner._discover_gpus`.
- Refactor `scripts/train_jax.py` and `scripts/train_jax_adapter.py`
  to use `MetricsLogger.log_train / log_val` instead of inline writes.
- Restore the `type` field and the dashboard's train/val split.
- Restore NaN sanitisation.

### 1.2 `pawn/run_config.py` — pydantic schemas — MISSING

**What it provided (v1, 455 LoC).**

- `BaseRunConfig`, `PretrainConfig`, `AdapterConfig`, `CotrainConfig`,
  `CotrainVariant` — pydantic BaseModels with `extra="forbid"` so
  unknown fields fail loud. Docstring specifically called out the
  legacy-flag-drift problem the schema was guarding against
  ("`legacy_vocab` silently ignored", "misspelled hybrid args
  silently ignored").
- 6+ `@model_validator(mode="after")` blocks for cross-field invariants
  argparse can't express (LR schedule shape, RoSA phase fractions,
  bottleneck `--no-attn` AND `--no-ffn` no-op, cotrain variant
  uniqueness).
- `PretrainConfig.model_json_schema()` — auto-generated JSON Schema
  for tooling (lab `lab_schema`, sweep configs, IDE introspection).
- `--config <json>` interface in `scripts/train.py` that parsed via
  `TypeAdapter(PretrainConfig).validate_python(...)`. The lab
  depended on this — wrote `run_config.json` per trial and passed
  `--config <path>` to the trainer.

**v2 today.** **37 individual `raise SystemExit(...)` calls scattered
across `scripts/train_jax.py` + `scripts/train_jax_adapter.py`**, some
duplicated between them. Lab's restored `_validate_config` is a
hand-rolled ~30-LoC dict validator. `lab_schema` returns a
hand-maintained string-keyed dict that **has already drifted** — I
added `--wandb` / `--wandb-project` to the trainers this session and
forgot `lab_schema`; nothing caught it.

**Knock-on effects.**

- Drift between the two trainers' guards is now possible and undetected.
- No serialisable single-source-of-truth for a training run — `config.json`
  is a write-only log of what was used, not a spec you can re-feed.
- Sweep tools (current + future) have to re-implement the dict → argv
  translation.

**Restoration scope (~half a day).** Port the v1 pydantic classes:

- Drop `CotrainConfig` (supernet replaces it).
- Add v2 fields (`supernet`, `variant`, `rosa_warmup_frac`,
  `rosa_top_k_frac`, etc.).
- **Restore v1 field names where the rename was cosmetic** (see §3).
- Add `--config <json>` input to both v2 trainers that calls
  `TypeAdapter(...).validate_python(...).model_dump()` then drives
  the existing argparse code path. Keep individual CLI flags so the
  bare-CLI invocation still works.
- Replace the 37 scattered `raise SystemExit` guards with their
  pydantic-validator equivalents.
- Restore `lab_schema` to call `model_json_schema()`.

### 1.3 `pawn/sweep.py` — standalone Optuna driver — MISSING

**What it provided (v1, 871 LoC).**

- `AdapterObjective` — the Optuna-callable: build CLI argv from
  trial params, run `scripts/train.py` as a subprocess, parse the
  resulting `metrics.jsonl`, return val_loss to Optuna.
- `InProcessRoSAObjective` — same shape but runs the trainer
  in-process (skips ~30 s JAX startup per trial; matters for RoSA
  sweeps that run dozens of trials).
- Pruning hookup via `trial.report(val_loss, step) +
  should_prune()` — `MedianPruner` / `HyperbandPruner` were wired.
- `scripts/sweep.py` CLI driver: `--strategy lora --n-trials 50
  --storage sqlite:///sweeps/lora.db`. Persistent study state.

**Lab vs. standalone driver — what people thought was the same thing
but wasn't.** Both spawned `scripts/train.py` as subprocesses. The
*lab* never called `study.optimize()`; it just used Optuna's `ask()`
to produce candidate suggestions on demand via `lab_results`. The
*standalone driver* gave Optuna full control of trial selection +
pruning. The lab restore covers the suggestion path; the optimize
loop is still gone.

**Restoration scope (~half a day, subprocess-only).** Port:

- `AdapterObjective` to call `scripts/train_jax_adapter.py` with
  v2 flag shapes (same translation `pawn.lab.runner._build_command`
  already does). Metric parsing flips from `record["type"] == "val"`
  to `record["val_loss"] is not None`.
- Pruning hookup.
- `scripts/sweep.py` CLI.
- Defer `InProcessRoSAObjective` — more involved; can land in 2.1.

### 1.4 `pawn/eval_suite/bounds.py` + `pawn/eval_suite/viz.py` — MISSING

**What they provided (v1, 108 + 433 LoC).** `bounds.py` computed
theoretical accuracy bounds (overall, per-phase, k-stats, perplexity)
from a position-level parquet corpus. `viz.py` was the matplotlib /
seaborn plotting helpers — per-layer probe accuracy curves, eval
breakdowns, etc.

**Coupling problem.** Both depend on `pawn/eval_suite/corpus.py`
(deleted, 457 LoC) — a polars-based parquet iterator
(`_iter_position_parts` + accumulator pattern). Restoring `bounds.py`
usefully requires the corpus reader back too.

**Restoration scope (~1 day).** Decide whether to keep the v1
position-parquet pipeline (`pawn/eval_suite/corpus.py` restored) or
port `bounds.py` to compute the same statistics over the v2 Rust-engine
corpus on the fly. The polars dep is now optional (`data-tools`
extra); restoring corpus.py is the lower-friction call.

### 1.5 `scripts/benchmark.py` — MISSING

**What it provided (v1, 1593 LoC).** Standalone perf benchmark: per-op
throughput, attention micro-benchmarks, forward/backward latency,
batch-size sweep. Useful for catching regressions on a new GPU.

**Restoration scope (~2 days, full JAX rewrite).** The script is
torch-API-heavy (`torch.cuda.Event`, `torch.compile`, `torch.profiler`).
A v2-flavoured version would use `jax.block_until_ready`, the JAX
profiler, and `jax.devices()`. Big enough to be its own PR.

### 1.6 Other tests removed alongside their producers

- `tests/test_enriched_pgn.py` — covered `parse_pgn_enriched` (still
  exists in `engine/src/lib.rs`, no Python consumer).
- `tests/training/test_wandb_{integration,utils}.py` — RESTORED as
  `tests/test_jax_wandb_utils.py` against the new wandb_utils
  signature.

---

## 2. Schema changes in `metrics.jsonl`

| v1 row shape | v2 row shape | Implication |
|---|---|---|
| `type: "config" | "train" | "val"` discriminator | no `type` field | dashboard groups everything as "train" |
| `mem/system_rss_gb`, `mem/cpu_percent`, `mem/gpu_*` | absent | OOM/leak diagnosis post-mortem only via `dmesg` |
| `timestamp` (absolute) | absent (only `wall_s` cumulative) | hard to correlate against external observability (Grafana, etc.) |
| NaN/Inf → `null` (RFC-7159 valid) | NaN literal written | downstream `json.loads` rejects malformed rows |
| Flush per-record | Flush every 10 chunks | finer-grained recovery vs fewer syscalls; minor |
| `slug` (zesty-osprey) | not in rows; only in run-dir name | mostly cosmetic |
| Adapter val: `type="val"` separate row | Same row as train, `val_loss` column null on non-val chunks | dashboard val/train split silently broken |

**The right fix:** restore `MetricsLogger` and have both trainers go
through it (§1.1). Keeps the v2 trainers' per-chunk metric extraction
but gets the schema back.

---

## 3. Cosmetic renames to revert

These broke backward compatibility with no documented reason. v1
field name should be restored; v2 argparse can accept the v1 name
as the canonical `dest=` with the v2 name as a `--alias` for one
release if we care about transition.

| v1 name | v2 name | Revert to | Note |
|---|---|---|---|
| `lora_rank` | `rank` | `lora_rank` | v2's `rank` is ambiguous now that it's shared across LoRA / Hybrid / RoSA. |
| `density` | `sparse_density` | `density` | `density` was already sparse-only in v1; the prefix is redundant. |
| `use_output_film` | `film_output` (polarity-flipped via `--no-film-output`) | `use_output_film` | Polarity flip is gratuitous. |
| `no_adapt_attn` | `bottleneck_no_attn` | `no_adapt_attn` | Cosmetic prefixing. |
| `no_adapt_ffn` | `bottleneck_no_ffn` | `no_adapt_ffn` | Cosmetic prefixing. |
| `d_model` / `n_layers` / `n_heads` (under `specialized_clm`) | `specialized_d_model` / `specialized_n_layers` / `specialized_n_heads` | v1 names | These live in `SpecializedCLMConfig`; the prefix isn't needed inside a per-strategy config. |

---

## 4. Substantive changes (document; do **not** revert)

These changed v1 semantics on purpose. Document them under "v2 release
notes" but leave the v2 names in place.

| v1 | v2 | Rationale (where there is one) |
|---|---|---|
| `lora_targets: Literal["qkvo","qv","qkv"]` | `lora_targets: list[str]` | More flexible; arbitrary subsets of `{q,k,v,o}`. |
| `sparse_targets: Literal[...]` | `sparse_targets: list[str]` | Same. |
| `rosa_warmup_steps: int` | `rosa_warmup_frac: float` | Scales with `--total-steps`. |
| `rosa_mode: "rosa" | "retro-sparse" | "retro-bottleneck"` | only `"rosa"` | Retro-ablation modes weren't ported. |
| `mask_samples`, `grad_alpha` | removed | RoSA mask-gen algorithm changed (v1 averaged grad magnitudes over `mask_samples` batches with `grad_alpha`-power weighting; v2 uses single forward+backward with all-True mask). |
| `unfreeze_layers: "5,6,7"` (explicit picks) | `n_unfreeze: 3` (top-N count) | **Regressive** — v1 let you pick specific layers. v2 only picks the top. Worth restoring v1 flexibility if anyone actually used it. |
| `epochs: int` | step-based | The Rust-engine corpus is infinite for pretrain; epochs are meaningless. For adapter training over a finite Lichess corpus, step-based still works but loses the epoch-aligned val-loss curve. |
| `bucket_size: int` | removed | The JAX trainer is shape-static; bucketed padding doesn't apply. |
| `lora_ffn: bool`, `sparse_ffn: bool` | removed | FFN adaptation was deleted from LoRA / Sparse. Could come back. |

---

## 5. Other gaps caught during review

### 5.1 `outcome_prefix_trained` gating only on impossible/improbable

`outcome_signal_test`, `prefix_continuation_test`, and
`poisoned_prefix_test` all condition on outcome tokens too. They
should accept the same `outcome_prefix_trained: bool` arg
`impossible_task_test` / `improbable_task_test` do, and skip with the
same `{"_skipped": ...}` sentinel when the model wasn't trained
that way. ~30 LoC change in `pawn/generation.py`.

### 5.2 Engine PAD-token init bug remaining in 3 paths

`engine/src/lib.rs:parse_pgn_lichess` was fixed to seed `flat_tokens`
with `PAD_TOKEN` instead of `0i16`. The same 0-init pattern is still
present in `parse_pgn_enriched` (line 879), `parse_pgn_lichess_filtered`
(line 1203), and `uci_moves_to_tokens` (line 773). None are called
from v2's Python surface today, but they share the same bug and
should get the same fix.

### 5.3 `MetricsLogger` schema bridge for dashboard

Until §1.1 is done, the restored dashboard's val/train split is
broken against v2 runs. Minimal hot-fix: change
`pawn/dashboard/metrics.py:127` from `rec.get("type", "train")` to
`"val" if rec.get("val_loss") is not None else "train"`. Real fix
restores `MetricsLogger`.

### 5.4 `Dockerfile` `compute_theoretical_ceiling.py` / `extract_lichess_parquet.py` etc.

The restored scripts pull in `data-tools` (polars / jinja2 /
zstandard). The current `Dockerfile` `uv sync` lines for runtime
don't include `--extra data-tools`, so the published image can't run
those scripts unless someone runs `uv sync --extra data-tools`
manually inside the container. Decide whether to bake the extra into
the image (cleaner) or document the manual sync.

### 5.5 `pawn/gpu.py` — GONE BY DESIGN

The v1 `pawn/gpu.py` configured PyTorch's `torch.compile` + SDPA
backend. JAX handles its own GPU configuration; no v2 equivalent
needed. **Do not restore.**

### 5.6 `pawn/cotrain.py` — GONE BY DESIGN

The supernet's multi-variant joint loss (every pretrain step computes
loss on the supernet plus each nested slice) is what cotrain used to
provide. **Do not restore.**

---

## 6. Suggested ordering of remaining work

In rough order of value / dependency:

1. **§1.2 + §3 together** — restore `pawn/run_config.py` with v1 field
   names (revert the cosmetic renames). Adds `--config <json>` to both
   trainers, fixes lab `lab_schema` to auto-generate. ~half a day.
2. **§1.1** — restore `MetricsLogger`. Fixes the dashboard's val/train
   split (§5.3 becomes a no-op), NaN handling, memory stats. ~half a day.
3. **§1.3** — restore `pawn/sweep.py` standalone Optuna driver
   (subprocess-only; defer in-process). ~half a day.
4. **§5.4** — bake `--extra data-tools` into the production Docker images.
   ~1 hour.
5. **§5.2** — extend the Rust `parse_pgn_lichess` PAD-token fix to the
   other 3 PGN-buffer paths. ~1 hour.
6. **§5.1** — extend the `outcome_prefix_trained` gate to the other 3
   generation diagnostics. ~30 minutes.
7. **§1.4** — restore `pawn/eval_suite/bounds.py` + `viz.py` (plus
   `corpus.py` to support them). ~1 day; needs an architectural
   decision on whether to keep the v1 position-parquet pipeline.
8. **§1.5** — restore `scripts/benchmark.py`. ~2 days, full JAX rewrite.
   Deferrable.

Total before §1.5: ~3 days. The first three items cover everything that
silently broke functionality the user expected to keep.
