# PAWN → JAX migration

## 0. How to read this document

This plan is written for the `/review-driven-development` skill. Each `### S<N>` section below is structured with four blocks the skill's `review-spec-alignment` agent will read and grade: `**Goal:**` (the section's intent in one sentence), `**Deliverables:**` (the concrete items the agent will check off), `**Verification:**` (the commands to run on the local GPU), and `**Definition of done:**` (the acceptance criterion or invariant the section advances). The structure is uniform across all sixteen sections. Don't rephrase or split these blocks — the spec-alignment agent grades by reading them.

The skill enforces a hard gate at every chunk close, every section close, and the final review: `review-spec-alignment` runs **before** the other review lanes. If the change doesn't meet the section's deliverables, the gate returns FAIL and the implementer goes back to finish the work. The other reviewers (bug-detector / type-correctness / etc.) never see code that doesn't yet do what the spec asks for.

## 1. The bar

**One sentence:** every command that works on `origin/main` works after this migration, plus the new JAX training stack runs in place of the PyTorch one.

**Operationally:** when the framework-swap PR merges, a user who had a v1 workflow on `main` can run the same workflow on the new code and get a comparable result. "Comparable" means: same CLI surface (or a documented rename), same on-disk schemas where they're durable artifacts, same eval numbers within published-checkpoint tolerance, same dashboard rendering.

This is a parity migration. The framework is changing. Almost nothing else should.

## 2. Your reference is `origin/main`

When you don't know what something should do, look at `origin/main`. When you've changed something and want to know if it's still right, diff against `origin/main`. When you finish a section, the question is "would a v1 user notice anything missing or broken?" — and the only way to answer it is by going back to `origin/main` and looking.

Concrete procedure:

- `git show origin/main:<path>` reads the v1 source without checking it out.
- `git checkout origin/main -- <path>` materialises a v1 file in your work tree (don't forget to revert).
- `git diff origin/main -- <path>` shows what you changed.
- `git log origin/main --oneline -- <path>` shows the v1 history (useful for "why is it this way").
- Switch to `main` and *actually run* the v1 command if you need to observe behaviour: `git switch main && uv run --extra <gpu> <cmd>`, then switch back to your branch.

Keep `origin/main` reachable for the entire migration. Don't delete it. Don't merge over it. Don't squash so aggressively that you can't `git diff origin/main` cleanly.

## 3. Acceptance criteria

These are the workflows the framework-swap PR has to preserve. Each one has a verification command you can run locally on a GPU machine. **Every section close re-grades the relevant subset.** No criterion is optional.

| # | Workflow | Verification |
|---|---|---|
| 1 | `uv sync --extra rocm` (or `--extra cu128`) installs cleanly. | `uv sync --extra rocm` exits 0. |
| 2 | The Rust engine builds. | `cd engine && uv run --with maturin maturin develop --release` exits 0. |
| 3 | The full test suite runs and passes on the work branch. | `uv run --extra rocm pytest tests/` — all green. |
| 4 | A v1 published checkpoint converts and matches v1 logits within tolerance. | `uv run python scripts/convert_published_checkpoints.py` runs through `pawn-{small,base,large}` and reports mean ‖Δlogit‖ ≤ 1e-3 (max ≤ 1e-4) on a real batch vs. the v1 torch reference. |
| 5 | A v1 published checkpoint (HF repo or local) loads through the compatibility layer into the JAX model. | `python -c "from pawn.legacy import convert_legacy_checkpoint; convert_legacy_checkpoint('thomas-schweich/pawn-base')"` produces a JAX checkpoint that `pawn.checkpoint.load_model` reads cleanly. |
| 6 | Pretrain the tiny supernet for ≥1000 steps. | `uv run --extra rocm python scripts/train_jax.py --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 --k 50 --local-checkpoints` — loss decreases monotonically over the run, no NaNs, all three sliced variants forward-evaluate after training. |
| 7 | Fine-tune a LoRA adapter on a real Lichess Elo band. | `uv run --extra rocm python scripts/train_jax_adapter.py --strategy lora --supernet tiny --variant base --lora-rank 4 --total-steps 200 --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 2000 --local-checkpoints` — val loss decreases, val games come from the held-out `validation` split (not carved from train). |
| 8 | All 8 adapter strategies dispatch and train at least one chunk. | Parameterised script test (`tests/scripts/test_train_jax_adapter.py::test_each_strategy_dispatch_runs`) runs all 8 strategies end-to-end. |
| 9 | Move-prediction accuracy + per-phase eval. | `uv run --extra rocm python scripts/eval_jax.py --checkpoint <converted>` produces overall + per-phase accuracy that matches v1 numbers within ±0.5 percentage points (run the v1 equivalent on `main` to get the baseline). |
| 10 | Linear probes. | `uv run --extra rocm python scripts/eval_probes_jax.py --checkpoint <converted>` runs all per-layer probes; output schema is the same as v1's `eval_probes.py`. |
| 11 | Generation diagnostics — all five, with the outcome-prefix gate. | `uv run --extra rocm python scripts/eval_generation_jax.py --checkpoint <converted> --outcome-prefix-trained` runs `outcome_signal_test` / `prefix_continuation_test` / `poisoned_prefix_test` / `impossible_task_test` / `improbable_task_test`. With `--no-outcome-prefix-trained` every one of them returns the `{"_skipped": ...}` sentinel. |
| 12 | Edge-case diagnostics. | `uv run --extra rocm python scripts/eval_generation_jax.py --checkpoint <converted> --edge-cases` (or however you name it) reports accuracy on `in_check` / `double_check` / `pin_restricts` / `ep_available` / `castle_legal_*` positions — same structure as v1's `pawn.eval_suite.diagnostics`. |
| 13 | Elo-stratified Lichess move-prediction. | `uv run --extra rocm python scripts/eval_vs_stockfish.py --checkpoint <converted> --pgn thomas-schweich/pawn-lichess-full` — Maia-style per-Elo-bin accuracy, schema matches v1. |
| 14 | A 3-trial Optuna sweep picks the best LoRA rank by val_loss. | `uv run --extra rocm python scripts/sweep.py --strategy lora --n-trials 3 --supernet tiny --variant base --storage sqlite:///<tmp>/lora.db --logs-dir <tmp>/sweep` exits 0 and prints a best trial with a finite `val_loss`. |
| 15 | `metrics.jsonl` carries the v1 schema. | Every row in `metrics.jsonl` has `type ∈ {"config","train","val"}`, `timestamp`, `slug`, `hostname`, `git_hash`, and (on train/val) `mem/system_*` + GPU memory keys. Validate with the v1 dashboard: `uv run --extra dashboard python -m pawn.dashboard --log-dir <run_dir>` renders the train/val split correctly. |
| 16 | `--resume <ckpt>` continues training with monotonic step counts. | Train for 500 steps, kill, then `--resume <last_ckpt>` — final `state.step` reaches the original `--total-steps` exactly. No step-counter rollback at the resume point. |
| 17 | SIGTERM during training triggers a final save before exit. | `train_jax.py --total-steps 100000 ...` in the background, `kill -TERM <pid>` mid-step. The process exits 0 (or non-zero with a graceful-shutdown log message), and the latest `step_*` directory on disk has a `.complete` sentinel + a step count past the last interval. |
| 18 | HF-backed checkpoint push. | `train_jax.py --hf-repo thomas-schweich/scratch-test ...` pushes `step_*` directories to the HF repo branch async (`git lfs ls-files` on the repo shows the safetensors). |
| 19 | `pawn-lab` MCP server accepts a trial config validated through pydantic. | `uv run --extra lab python -m pawn.lab` starts; `lab_launch(config={...})` rejects an unknown field (`extra="forbid"`); `lab_schema` returns valid JSON Schema generated from `PretrainConfig.model_json_schema()` / `AdapterConfig.model_json_schema()`. |
| 20 | Published v1 HF checkpoints remain usable; v2 publishes to new repos. | `pawn.legacy.convert_legacy_checkpoint("thomas-schweich/pawn-base", ...)` produces a JAX checkpoint that loads + forward-evaluates. v2 training publishes to `pawn-{small,base,large}-v2` (or similar); the v1 repos are not modified. |

If a workflow on `main` works that isn't on this list — and it's not in §6 ("GONE BY DESIGN") — add it to the list before you ship. Don't skip it because it's not listed.

## 4. Motivation

PAWN-base is small: ~35.8M parameters, 512-token sequences, a 1,980-token vocabulary. A training step is ~30 TFLOP, yet the v1 stack runs at roughly 15% MFU on a B200 — most of the machine is idle. The idle time isn't matmuls; it's framework dispatch, kernel-launch latency, host round-trips, and the CPU/PCIe data path.

For a model this small, the design should optimise *those* costs:

1. **Whole training loop as one compiled program** — eliminate per-step launch and dispatch overhead.
2. **Data resident on (or streamed to) the device** — eliminate the per-step CPU/PCIe path.
3. **A single framework** — JAX everywhere. One model definition, one optimiser stack, no PyTorch/JAX boundary.
4. **One supernet** — small / base / large extracted as nested slices of a single shared-weight model (MatFormer / slimmable-network scheme).

Estimated improvement: ~3× throughput on B200 from JAX + fused loop + resident corpus + bf16. A separate ~1.5–1.8× from fp8 compute is deliberately out of scope for v1.

## 5. Design

**Framework.** JAX/Equinox + Optax. PyTorch is removed from the training and eval surface. Two narrow torch touchpoints survive: a thin loader for external non-JAX consumers (§5.6) and a frozen reference architecture used by the legacy-converter parity tests.

**Model.** A single `PAWNModel` Equinox module shapes the supernet, every sliced variant, and any standalone (converted-legacy) model. Stacked transformer layers applied with `jax.lax.scan` over a leading `n_layers` axis. Plain attention (materialised `QK^T` scores) — at seq 512 attention is ~12% of step FLOPs and plain attention sidesteps fused-kernel maturity under JAX-on-ROCm. RMSNorm + RoPE + SwiGLU + factored input embeddings (`src_embed[s] + dst_embed[d] + promo_embed[p]`).

**Supernet + variants.** The supernet is large's dimensions (d=640, 10 layers, 10 heads, head_dim=64). small (d=256, 4 heads) and base (d=512, 8 heads) are nested slices of it — the inner `[:d_V, :d_V]` of every weight matrix. Joint training sums per-variant cross-entropies on the same batch; gradients accumulate into the one shared weight tensor. The supernet is sliced into three standalone safetensors checkpoints at publish time; downstream consumers see ordinary independent checkpoints. `head_dim = 64` is fixed so width slices align to whole heads and RoPE is variant-invariant.

A small `TINY_SUPERNET` (d=192, 4 layers, 3 heads) + nested `TINY_VARIANTS` exists for verification runs that don't need the production scale.

**Optimiser.** `optax.chain(optax.clip_by_global_norm(1.0), adamw)`. Forward casts parameters to bf16 with fp32 accumulation; the master copy and Adam moments stay fp32.

**Training loop.** `lax.scan` over K inner steps inside one `@eqx.filter_jit` function with buffers donated. Host loop runs N chunks of K steps each. Per-step body never returns to the host. Per-chunk metrics flush to disk between chunks. K is chosen to amortise host overhead; for adapter training K is additionally bounded by validation cadence (`K ≤ val_every`).

**Two-tier PyTree.** Adapter training partitions the model into `frozen` + `trainable` via `eqx.partition(model, adapter_filter(model))`. `jax.grad` only differentiates `trainable`; XLA DCEs the backbone weight-gradients (~33% FLOP cut on the backward pass). Pretraining is the same machinery with everything `trainable`.

**Checkpoints.** Every save creates a new `step_<N>` directory next to the existing ones — **old checkpoints are never overwritten or deleted by the trainer.** Within a single save, the write is atomic: payload files land in `step_<N>.tmp`, then a `.complete` SHA-256 sentinel, then `os.rename(step_<N>.tmp, step_<N>)`. If something kills the process mid-save, the next run sees either a complete `step_<N>` or just the orphaned `.tmp` (which gets cleaned up at the next save's start). Across runs, `step_*` directories accumulate; pruning old ones is the user's call, not the trainer's. safetensors format. Every load verifies the sentinel. Sentinel helpers live in a stdlib-only module (`pawn/_sentinel.py`) so they can be imported from anywhere without pulling JAX.

**Legacy / compatibility loader.** A single bridge: `pawn.legacy.convert_legacy_checkpoint(repo_id_or_path)` reads a v1 torch checkpoint (from a HuggingFace repo or a local path), reshapes the weights into JAX layout, and writes a JAX-format safetensors checkpoint under `$HF_HOME/pawn-jax-converted/<variant>/` (reused across runs via content hashing). Pre-vocab-transition checkpoints (older ~60k-token vocab) are rejected loudly. This is the only compatibility surface — there is no separate "thin torch loader for external consumers." If, post-merge, a real use case emerges for loading v2 JAX checkpoints in pure PyTorch code (an external consumer who doesn't want JAX as a dependency), that's a future addition; we don't build it speculatively.

**Configuration.** Pydantic models (`BaseRunConfig` / `PretrainConfig` / `AdapterConfig` / `SpecializedCLMConfig`) with `extra="forbid"`. Every training driver, sweep tool, and the lab MCP server reads through them. JSON Schema is derived automatically (`model_json_schema()`); the lab uses that for client-side validation. Training scripts accept `--config <json>` and merge into argparse defaults; bare CLI flags still work and take precedence.

**Metrics.** `MetricsLogger.log_config / log_train / log_val`. Every row in `metrics.jsonl` carries a `type ∈ {"config","train","val"}` discriminator + baseline metadata (`timestamp`, `slug`, `hostname`, `git_hash`, `elapsed`) + host CPU/memory stats (psutil) + GPU memory stats (shell-out to `nvidia-smi` / `rocm-smi`). NaN / Inf sanitised to `null`. Per-record flush. The training drivers go through this — they never open `metrics.jsonl` directly.

**Data.**
- *Pretraining:* Rust-engine random games, generated offline per run. Sequential consumption (i.i.d. by construction so no shuffle is needed). Double-buffered prefetch.
- *Adapter training:* Lichess Elo-stratified parquet (HF dataset). The slice is filtered by `(elo_min, elo_max, min_ply)` through polars, packed into the same `Corpus` shape pretraining uses, and cached on disk under `$HF_HOME/pawn-lichess-cache/<key>/` with a `.complete` sentinel. Train uses the dataset's `train` split; val uses the dataset's held-out `validation` split (no carve-from-train leakage by default). A flag exists to opt back into carve-from-train for single-file local sources without split structure. The finite train slice is tiled across epochs with a per-epoch permutation.

**Eval.** Move-accuracy + per-phase breakdown; linear probes (Optax fit over hidden states); five generation diagnostics (`outcome_signal_test`, `prefix_continuation_test`, `poisoned_prefix_test`, `impossible_task_test`, `improbable_task_test`) — **all five** condition on outcome at position 0, so **all five** take an `outcome_prefix_trained: bool` and return a `{"_skipped": ...}` sentinel when False; edge-case diagnostics using the Rust engine's `edge_case_bits()` for guaranteed coverage of `in_check` / `double_check` / `pin_restricts` / `ep_available` / `castle_legal_*` / etc.; Elo-stratified Maia-style accuracy.

**Sweeps.** Standalone Optuna driver. Per-strategy suggester functions (LoRA / Bottleneck / FiLM / Hybrid / Sparse / RoSA / Unfreeze / SpecializedCLM). `AdapterObjective` runs `scripts/train_jax_adapter.py` as a subprocess per trial, parses `metrics.jsonl` for the best `val_loss`. Persistent study state via SQLite.

**Operational extras.** HF-backed checkpoint push (async). SIGTERM handler: save → push → exit. `--resume <ckpt>` loads `TrainState` mid-training, preserves `state.step` across the resume so metrics step counts stay monotonic. wandb integration (optional).

**Lab + dashboard.** `pawn.lab` (FastMCP daemon) drives Optuna's `ask()` to suggest trials, spawns `scripts/train_jax_adapter.py` subprocesses, polls each trial's `metrics.jsonl`. `pawn.dashboard` (Solara UI) reads `metrics.jsonl` and renders per-run / per-trial charts. Both go through the pydantic config layer and the structured metrics schema.

## 6. Scope

**In scope** (everything below has an acceptance criterion in §3):

- `pawn/model.py` — Equinox PAWNModel supernet
- `pawn/config.py` — `ModelConfig`, `SUPERNET`, `VARIANTS`, `validate_nested`
- `pawn/run_config.py` — pydantic config models
- `pawn/logging.py` — `MetricsLogger`
- `pawn/checkpoint.py` — atomic JAX/safetensors save/load + HF push
- `pawn/_sentinel.py` — stdlib-only sentinel helpers
- `pawn/corpus.py` — Rust-engine corpus → JAX arrays
- `pawn/lichess_data.py` — Lichess parquet → JAX `Corpus`, on-disk cache, multi-epoch tiling
- `pawn/trainer.py` — pretraining loop + supernet joint loss
- `pawn/adapter_trainer.py` — two-tier frozen/trainable training
- `pawn/adapters/` — 8 strategies (lora / film / unfreeze / bottleneck / hybrid / sparse / rosa / specialized_clm)
- `pawn/eval.py` — move accuracy + per-phase
- `pawn/probes.py` — linear probes
- `pawn/generation.py` — 5 generation diagnostics (all 5 gated on `outcome_prefix_trained`) + KV-cached decoder
- `pawn/lichess_eval.py` — Elo-stratified Maia-style accuracy
- `pawn/eval_suite/` — edge-case diagnostics + theoretical accuracy bounds + plotting helpers (polars / matplotlib / seaborn)
- `pawn/sweep.py` — standalone Optuna driver
- `pawn/legacy.py` — the single compatibility bridge: v1 torch HF checkpoints → JAX safetensors
- `pawn/lab/`, `pawn/dashboard/`, `pawn/wandb_utils.py`
- `scripts/train_jax.py`, `scripts/train_jax_adapter.py`, `scripts/eval_jax.py`, `scripts/eval_probes_jax.py`, `scripts/eval_generation_jax.py`, `scripts/sweep.py`, `scripts/convert_published_checkpoints.py`, `scripts/run_evals_backbone.py` (batch-eval orchestrator)
- The data tools: `scripts/extract_lichess_parquet.py`, `scripts/compute_theoretical_ceiling.py`, `scripts/generate_model_cards.py`
- Engine: extend the `PAD_TOKEN` initialisation fix to every PGN-token-init site in `engine/src/lib.rs` (search the file for `vec![0i16;` patterns over token buffers and audit each one)
- Tests: a JAX-side test per module, parity tests for the converter, smoke tests for the scripts

**Every adapter on `main` ports forward.** Read `pawn/adapters/__init__.py` on `main` for the canonical list. Each adapter has at least one tested workflow on `main`; each one keeps a tested workflow in v2. **Specifically — and non-negotiably — RoSA's `retro-sparse` and `retro-bottleneck` modes are in scope alongside the standard `rosa` mode.** A future round of empirical evaluation may decide which modes to keep; that's not the migration's call to make. Port everything; cull only by explicit design decision later.

**GONE BY DESIGN** (each one has a justification you can quote):

- `pawn/cotrain.py` — the supernet's multi-variant joint loss is what cotrain provided. Cotrain is replaced by "pretrain the supernet."
- `pawn/gpu.py` — JAX handles its own GPU configuration. Port only the `PAWN_ALLOW_CPU=1` escape hatch into the JAX entry points.
- `pawn/data.py` + `pawn/data_utils.py` — replaced by `pawn/corpus.py` (random games) and `pawn/lichess_data.py` (parquet).
- `pawn/eval_suite/worker.py` — multi-process eval wrapper for the v1 torch path; the JAX eval is single-process.
- `bucket_size` (bucketed dynamic padding) — the JAX trainer is shape-static. The Lichess data path emits fixed-width packed arrays; no per-batch bucketing.

**Things that *look* like they could be out and aren't:**

- **`scripts/benchmark.py`.** Port it. A perf-microbench harness is part of the v1 surface; users have it on `main`. Don't drop it because it's "secondary" or "deferrable."
- **RoSA `mask_samples` / `grad_alpha`.** Port forward — these are part of v1's tested RoSA contract. If a later round of empirical work decides a single-forward-backward Phase-2 mask is enough, fine, but the migration doesn't get to make that call.
- **The v1 `unfreeze_layers: "5,6,7"` form** — explicit per-layer picks. That's the v1 contract. Keep it.
- Anything else you find on `main` and don't see in v2: don't drop it silently. Port it, or — only if implementing it is nonsensical, impossible, or actively detrimental — write a `DEFERRALS.md` entry (see §9.4).

## 7. Backward-compatibility contract

The names users type and the schemas they grep are durable. Look at `main` to find them.

**CLI flag names.** `scripts/train.py` on `main` is the canonical source. The same `--lora-rank`, `--density`, `--use-output-film`, `--no-adapt-attn`, `--no-adapt-ffn` etc. that work on `main` work in v2 (with the appropriate `dest` so the namespace is sensible — `lora_rank` not `rank`). If you find yourself renaming a flag, stop and ask whether the user types it on `main`.

**`metrics.jsonl` schema.** `pawn/logging.py` on `main` is the spec. The v2 `MetricsLogger` produces records with the same fields. The dashboard's train/val split depends on the `type` discriminator — don't drop it. If you find yourself wanting to bolt a new column onto a train row to mean "val also happened here," don't; emit a separate `type: "val"` record like v1.

**Published HF checkpoints.** `thomas-schweich/pawn-{small,base,large}` are v1 PyTorch artifacts. They stay frozen. v2 supernet-derived checkpoints publish to new repos (`pawn-{small,base,large}-v2` or whatever naming you settle on). The legacy converter is what makes the v1 repos remain usable from v2 code.

**The compatibility loader (`pawn.legacy.convert_legacy_checkpoint`).** This is the only bridge between v1 HF checkpoints and the v2 JAX surface. Don't ship without it; the published `pawn-{small,base,large}` weights have to remain usable from v2 code.

**Run-dir layout, `.complete` sentinel format, `config.json` schema.** All on `main`. Match the durable parts.

## 8. Pre-flight

Before you write a line of code:

1. Clone the repo and check out `origin/main`.
2. `uv sync --extra rocm` (or `--extra cu128`).
3. `cd engine && uv run --with maturin maturin develop --release && cd ..`
4. `uv run --extra rocm pytest tests/` — confirm the v1 test suite passes on your machine. If it doesn't, fix the env before doing anything else; you'll need this as your reference oracle.
5. Pick a small v1 workflow and run it end-to-end. Suggested: `uv run python scripts/train.py --variant toy --total-steps 100 --batch-size 4 --local-checkpoints` then `uv run python scripts/eval_accuracy.py ...`. The point is to see the v1 output shape before you start producing the v2 equivalent.
6. Take a checkpoint or two. `git tag pre-jax-baseline` so you can always come back.

Now create your feature branch:

```bash
git checkout -b jax-migration
```

You will work on this branch through every section. Don't merge `main` into it during the migration unless `main` itself gets fixes you need. Don't merge it into `main` until §13.

## 9. Discipline rules

These exist because the failure modes they prevent are easy and silent. Treat each one as a precondition for any commit.

### 9.1 The commit message describes the diff, not your intent

Before you `git commit`, run `git diff --staged` and read the body you're about to write against the diff. If the body says "X is now wired" and the diff doesn't wire X, fix one of them. Never both rationalise and ship.

### 9.2 No claim without verification

For any commit body that asserts a behaviour ("loads cleanly", "preserves monotonic step counts", "matches v1 within tolerance"), paste the command you ran and an excerpt of the output that proves it. If you didn't run it, don't claim it. "Tests pass" means you ran them; "tests collect" means you didn't.

### 9.3 Deletions justify their replacement

For any file you delete, the commit body answers: **what capability did this provide, and where does that capability live in v2?** If the answer is "in `pawn/<other_file>.py`," cite the function. If the answer is "nowhere," that's a GONE-BY-DESIGN deletion — the design document (§6 of this plan) must already list it as GONE BY DESIGN with a justification you can quote. If the design doc doesn't list it, you don't get to delete it; the `review-spec-alignment` agent will catch this. A `git rm` without that paragraph is a bug.

### 9.4 Deferrals are rare and documented

**The default is: implement everything.** A deferral is only legitimate when actually doing the work is *nonsensical, impossible, or actively detrimental*. Not "harder than I thought." Not "I'm running out of context." Not "I'd rather not." Not "the design says it'll change later."

When you genuinely hit one of the three allowed reasons:

- Append an entry to `DEFERRALS.md` at the repo root. Format:

  ```
  ## <short title>
  **Section:** S<N>.<chunk> (<commit sha>)
  **Reason:** <nonsensical | impossible | detrimental> — <one paragraph>
  **What was supposed to happen:** <what the plan / spec asked for>
  **What I did instead:** <what shipped + why it's acceptable>
  **To revisit when:** <the condition under which this should be reopened>
  ```

- Name the entry in the section's squash commit body under a `## Deferred` heading.

Things you cannot defer under any of the three reasons: anything on the §3 acceptance criteria list. The acceptance criteria are the contract; "I deferred criterion N" is the same sentence as "I didn't finish the migration." Don't write it.

This is a discipline, not an approval gate. There's no user ACK to wait for — you don't pause the workflow on a deferral. You file the entry and keep moving. The discipline is that the entry exists, is honest about which of the three reasons applies, and is visible in the diff for anyone (including future you) to review.

### 9.5 "Look at `main`" before inventing

When you're about to decide what `lora_targets` looks like, what fields go on a metric row, what the run-dir naming convention is, what a script's exit code means: don't decide. `git show origin/main:<path>` and follow what's there. Decisions are for things `main` doesn't answer.

### 9.6 Run it before you call it done

Each section in §10 lists verification commands. Run them. With a local GPU this is practical for every section. "I ran type-check and tests collect" is not "I ran the smoke." Don't substitute.

### 9.6a The spec-alignment agent is a hard gate

After every chunk commit, every section close, and the final review, the **first** review you run is `review-spec-alignment`. It reads this plan + the diff and answers one question: *does this change actually accomplish what the section/chunk says it should?* It runs **before** the other review lanes — if the work doesn't meet the spec, you don't burn token budget having bug-detector / type-correctness / etc. review code that isn't doing the right thing yet.

If `review-spec-alignment` returns FAIL with a list of unmet requirements: go back, finish the work, then re-run `review-spec-alignment`. Only when it returns PASS do you proceed to the other review lanes. The `review-driven-development` skill enforces this order; the spec-alignment check is non-skippable.

### 9.7 The workflow is autonomous

You do not stop for user approval between sections. After each section's verification commands pass, run the section review, squash to the integration branch, and move on. The user expects you to drive the whole migration end-to-end. Section close = run the smokes, paste the output into the squash commit body, run the reviews, fix what they catch, squash, push, start the next section. The only legitimate places to surface to the user are: (a) you're genuinely blocked and one of the three deferral reasons applies, and `DEFERRALS.md` was a worse fit than asking; (b) the final framework-swap PR is open and waiting for human review.

## 10. Section plan

The sections are ordered so that the load-bearing infrastructure exists before anything depends on it. Specifically:

- **Pydantic config (S2) and `MetricsLogger` (S3) come before any trainer or script.** When you write the trainer in S6, you can't "decide later" whether to use them — they're the only API there is.
- **Lichess data (S5) comes before the adapter trainer (S7).** The adapter trainer can't be written against random games "as a verification proxy" and then have Lichess bolted on — the Lichess path is what defines the adapter trainer's data contract.
- **Engine fixes (S11) come before the eval surface (S9) is finalised**, because the JAX eval may catch new edge cases against PAD-token sentinel handling.

Section by section:

### S1 — Foundation

**Goal:** stand up the dependency configuration and orientation docs that everything else builds on. Nothing runs yet.

**Reference on main:** `pyproject.toml`, `CLAUDE.md`, `.gitignore`, `pyproject.toml`'s extras (`rocm`, `cu128`).

**Deliverables:**
- `pyproject.toml` — JAX/Equinox/Optax/pydantic/polars/optuna/etc. all in `[project.dependencies]`. Extras are minimised: GPU (`rocm` / `cu128`) — these add torch (the legacy converter parity tests need it) alongside the GPU jax backend — plus `dashboard`, `lab`, `wandb`. *Polars, matplotlib, seaborn, jinja2, zstandard are in base — they're used by the Lichess data path which is core.*
- A new top-level `CLAUDE.md` mapping the post-swap repo. Reference the upcoming modules; mark this doc as the canonical orientation. **Add a top-of-file note that the published v1 metrics on HF are v1 PyTorch numbers — v2 republishes to new repos.**
- `.gitignore` cleanups as needed.
- `docs/jax_migration_plan.md` — this document, committed verbatim so future you can grep it.

**Verification:**
- `uv sync --extra rocm` exits 0; `uv pip list | grep -E "jax|equinox|optax|polars|pydantic"` shows the expected packages.
- `python -c "import jax, equinox, optax, polars, pydantic; print('ok')"` runs.
- Engine still builds.

**Definition of done:** acceptance criteria 1 and 2 pass on the work branch.

### S2 — JAX core: config, supernet, model, checkpoint

**Goal:** the model exists and a checkpoint round-trips. No training yet.

**Reference on main:** `pawn/config.py` for the vocab constants (NUM_ACTIONS, PAD_TOKEN, OUTCOME_TOKEN_BASE, N_OUTCOMES, VOCAB_SIZE, MAX_SEQ_LEN). `pawn/checkpoint.py` for the atomic-write contract and `.complete` sentinel. `pawn/model.py` for the architecture (RMSNorm + RoPE + SwiGLU + factored embeddings).

**Deliverables:**
- `pawn/config.py` — `ModelConfig` dataclass, `SUPERNET`, `VARIANTS`, `TINY_SUPERNET`, `TINY_VARIANTS`, `validate_nested(variant, supernet)`. **`head_dim = 64` is fixed across nested variants.**
- `pawn/model.py` — Equinox `PAWNModel`. Stacked-layer `lax.scan`. Plain attention. RMSNorm with the `weight-multiply-in-fp32-then-downcast` cast order. Slice extraction via `pawn.model.sliced(supernet, variant_cfg)`.
- `pawn/_sentinel.py` — stdlib-only. `sha256_file`, `write_sentinel`, `verify_sentinel`, `IncompleteCheckpointError`, `CheckpointIntegrityError`. Validate the on-disk shape, not just the file's presence.
- `pawn/checkpoint.py` — atomic safetensors save/load. Canonical schema = one tensor per `PAWNModel` array field (~16 fields), declaration order. `.complete` sentinel verified on every load. Asserts the 16-field count at import.
- `pawn/__init__.py` — package docstring only. Don't eager-import `pawn.model` or anything else that drags JAX into `import pawn`. Other modules (`pawn.sweep`, `pawn.legacy`) must remain importable without dragging in the full JAX dependency graph unnecessarily.

**Verification:**
- `uv run python -c "from pawn.model import init_model; from pawn.config import TINY_SUPERNET; m = init_model(TINY_SUPERNET, key=0); print(m)"` runs.
- `uv run python -c "from pawn.model import init_model, sliced; from pawn.config import TINY_SUPERNET, TINY_VARIANTS; m = init_model(TINY_SUPERNET, 0); s = sliced(m, TINY_VARIANTS['small']); ..."` produces a forward pass at the small variant's shape.
- A checkpoint save+load round-trips and verifies the sentinel.
- All tests for this surface pass.

**Definition of done:** all acceptance criteria depending only on the model + checkpoint pass. (None of §3's criteria are fully done yet — most depend on later sections.)

### S3 — Pydantic run_config (KEYSTONE)

**Goal:** every training parameter has a single home; `extra="forbid"` rejects stale or misspelled fields; `model_json_schema()` is the public surface for sweep / lab tooling.

**Reference on main:** `pawn/run_config.py`. Read it cover to cover. Every `model_validator`, every cross-field invariant, every field default. The v2 version differs (drop `CotrainConfig`, drop torch-specific fields like `amp_dtype` / `device` / `num_workers`, add `supernet` / `variant` / `k` / `max_corpus_gb` / `rosa_warmup_frac` / `rosa_top_k_frac`) but the *shape* — pydantic-typed, `extra="forbid"`, JSON-Schema-derivable, cross-field validators — is preserved verbatim.

**Deliverables:**
- `pawn/run_config.py` — `BaseRunConfig`, `PretrainConfig`, `AdapterConfig`, `SpecializedCLMConfig`, discriminated-union `RunConfig`. **Every field name matches `main`.** Specifically: `lora_rank` (not `rank`), `density` (not `sparse_density`), `use_output_film` (default `True`, not polarity-flipped), `no_adapt_attn` / `no_adapt_ffn` (not `bottleneck_no_attn` / `bottleneck_no_ffn`), bare `d_model` / `n_layers` / `n_heads` / `d_ff` inside `SpecializedCLMConfig` (no `specialized_` prefix), `unfreeze_layers: str | None` (explicit comma-separated layer picks, e.g. `"5,6,7"` — not a top-N count), `rosa_mode: Literal["rosa", "retro-sparse", "retro-bottleneck"]` (all three modes), `rosa_warmup_steps: int`, `mask_samples: int`, `grad_alpha: Literal[1, 2]` (all v1 RoSA hyperparameters preserved verbatim).

When in doubt: open `pawn/run_config.py` on `main` and copy the field set. New fields needed for the JAX path (`supernet`, `variant`, `k`, `max_corpus_gb`, `seq_len`) are additions, not replacements.

**Verification:**
- `PretrainConfig.model_json_schema()` returns valid JSON Schema; same for `AdapterConfig`.
- A `--config <json>` smoke round-trip: dump → load → `model_dump()` equals the original.
- Every cross-field validator has at least one happy-path and one failure-path test.

**Definition of done:** the load-bearing surface is locked. Any later module that needs configuration goes through this; there is no other way to configure a run.

### S4 — `MetricsLogger`

**Goal:** there is exactly one path metrics take to disk. The `type` discriminator and baseline metadata are non-optional.

**Reference on main:** `pawn/logging.py`. The schema there is the schema. Read it. Don't redesign.

**Deliverables:**
- `pawn/logging.py` — `MetricsLogger` with `log_config`, `log_train`, `log_val`. Per-record flush. NaN/Inf → `null`. Slug-based run-dir naming (`<prefix>_<YYYYMMDD>_<HHMMSS>_<microseconds>_<suffix?>_<slug>`). `get_git_info()` / `random_slug()` helpers.
- GPU memory stats via shell-out to `nvidia-smi` / `rocm-smi` (the choice cached at process start). v1's `torch.cuda.*` branch becomes the shell-out.
- The module is torch-free. `pawn.logging` does not `import torch`.

**Verification:**
- 16+ unit tests covering: record-type discriminator, baseline fields, NaN sanitisation, per-record flush (read back before close), `nvidia-smi` mock, `rocm-smi` mock, `device='cpu'` skips the shell-out.
- `pawn.logging` is importable in a CPU-only env (no torch, no GPU).

**Definition of done:** there is no other way for a JAX training driver to write `metrics.jsonl`. (Future trainers go through this.)

### S5 — Corpus + Lichess data

**Goal:** the data layer the trainers consume — both Rust-engine random games and Elo-stratified Lichess parquet.

**Reference on main:** `pawn/data.py` (Rust-engine wrapper). `pawn/lichess_data.py` (parquet + filter — note the **torch.utils.data Dataset** shape; v2 emits JAX arrays, not torch tensors, but the parquet schema + filter logic is the same). `pawn/lichess_cache.py` (on-disk cache + SHA-256 sentinel).

**Deliverables:**
- `pawn/corpus.py` — `Corpus` dataclass (`tokens [N,T] int32`, `attn_mask [N,T] bool`, `targets [N,T] int32`, `loss_mask [N,T] bool`, `outcome_offset [N] uint8`). `generate_corpus(n_games, max_ply, seq_len, seed) → Corpus` calling `engine.generate_random_games`. `pack_corpus(move_ids, game_lengths, outcome_offset, *, seq_len) → Corpus` for pre-tokenised games. `_pack_clm` shared helper.
- `pawn/lichess_data.py` — `load_lichess_corpus(source, *, split, elo_min, elo_max, min_ply, seq_len, max_games, cache_dir) → Corpus`. Polars scan + Elo filter (both players, `elo_max` exclusive) + min_ply filter. Caches packed arrays under `$HF_HOME/pawn-lichess-cache/<sha-of-filter-params>/` with a `.complete` sentinel. **Split-aware for both HF repos and local directories with `train-*.parquet` / `validation-*.parquet` shards.** `make_epoch_schedule(n_pool, n_needed, seed) → NDArray[int64]` for multi-epoch tiling.

**Verification:**
- Synthetic-parquet tests for the Elo filter, min_ply filter, cache round-trip (write → read → verify byte-equal arrays), cache-key sensitivity (different filter → different cache entry), local-dir split-prefixed scan.
- Tests against the real HF dataset are not required, but a single `load_lichess_corpus("thomas-schweich/pawn-lichess-full", elo_min=1800, elo_max=1900, max_games=100)` on the local GPU box must run end-to-end and produce a non-empty Corpus.

**Definition of done:** acceptance criterion 7 (Lichess adapter training) becomes implementable. The trainer's only data source for adapters is this module — no random-game fallback at the trainer level. Random games stay available as a verification proxy via the `--no-pgn` script-level flag.

### S6 — Pretraining trainer

**Goal:** train the supernet end-to-end.

**Reference on main:** `pawn/trainer.py` (LR schedules, gradient clipping, AdamW), `pawn/cotrain.py` (multi-variant joint loss — this is what the supernet replaces).

**Deliverables:**
- `pawn/trainer.py` — `Batch`, `TrainState` (with `state.step` as a JAX scalar so JIT doesn't recompile per step), `VariantSpec`, `cross_entropy_loss`, `make_lr_schedule` (warmup-cosine + WSD + constant + one_cycle + infinite, with sum-constraint validation that matches `BaseRunConfig._check_lr_schedule_fractions`), `make_optimizer` (`optax.chain(clip_by_global_norm(1.0), adamw)` with a `lax.cond` guard against padded-batch weight-decay drift), `make_train_step` (jitted single step), `make_scan_step` (K-step `lax.scan`). Supernet joint loss is a static unroll over the variants — sum the per-variant cross-entropies on the same batch.

**Verification:**
- Acceptance criterion 6: train `TINY_SUPERNET` for 1000 steps, no NaNs, loss decreases monotonically, all sliced variants forward-evaluate.
- Tests pin: `state.step` is a JAX scalar inside JIT; `optax.warmup_cosine_decay_schedule` is built with `decay_steps=total_steps` (not `total_steps - warmup`); padded-batch weight-decay guard works; gradient clipping caps the global norm at 1.0; scan doesn't recompile per call.

**Definition of done:** criterion 6 passes locally with the verification command pasted into the section's squash commit body.

### S7 — Adapter trainer + 8 strategies

**Goal:** each of 8 strategies trains end-to-end, on both random games and Lichess data.

**Reference on main:** `pawn/adapter_training.py` (the v1 trainer; the two-tier-PyTree concept is new in v2 but the per-strategy contracts are the same). `pawn/adapters/*` (each strategy's hyperparameters + the per-strategy build/filter contract). `pawn/specialized_clm.py` (the from-scratch path).

**Deliverables:**
- `pawn/adapter_trainer.py` — two-tier PyTree partition via `eqx.partition(model, adapter_filter(model))`. K-step `lax.scan`. Jitted forward-only eval function. Per-strategy gradient mask (only `unfreeze` needs one; the others use the partition alone). RoSA's three-phase schedule (Phase 1 LoRA warmup → Phase 2 gradient-magnitude mask gen → Phase 3 joint training under fixed mask) preserves `state.step` across the Phase 2→3 re-init so the metrics log stays monotonic; Optax-internal step resets by design (Phase 3 gets its own warmup ramp).
- `pawn/adapters/__init__.py` — re-export surface (`LoRAConfig`, `init_lora_model`, `adapter_filter`, etc.) per strategy. **The export list on `main` is the canonical list of adapters; port every name on it.**
- One per-strategy module per adapter on `main`. Each has a `Config` dataclass, an `init_<name>_model(backbone, cfg, key)` constructor, an `<name>_adapter_filter(model)` partition function, and (where it needs one) a gradient mask helper. **Specifically including:** `lora`, `film`, `unfreeze` (with the v1 `unfreeze_layers="5,6,7"` explicit-pick form), `bottleneck`, `hybrid`, `sparse`, `rosa` (the standard mode), `rosa-retro-sparse`, `rosa-retro-bottleneck`, `specialized_clm`. All three RoSA modes are in scope; the mode is selected by `AdapterConfig.rosa_mode`. RoSA's v1 hyperparameters (`mask_samples`, `grad_alpha`, `rosa_warmup_steps`) are preserved.

**Verification:**
- Acceptance criterion 8: each strategy dispatches and trains at least one chunk without crashing.
- Acceptance criterion 7: a real LoRA fine-tune on a Lichess Elo band converges (val loss decreases).
- The structural invariant — every array field of `state.trainable.backbone` is `None` after partitioning — is pinned by a test.

**Definition of done:** criteria 7 and 8 pass locally, output pasted.

### S8 — Eval surface

**Goal:** every v1 eval workflow (move accuracy, probes, generation diagnostics, edge-case diagnostics, Lichess Elo-stratified) has a JAX equivalent.

**Reference on main:** `pawn/eval_suite/*` covers: `probes.py` (linear probes), `diagnostics.py` (edge-case eval using `engine.edge_case_bits()`), `generation.py` (the 5 generation tests), `lichess.py` (Elo-stratified accuracy). Each has its own contract; read each one before porting.

**Deliverables:**
- `pawn/eval.py` — move-accuracy + per-phase breakdown. Argmax restricted to `[0, NUM_ACTIONS)` so PAD + outcome tokens can't be sampled.
- `pawn/probes.py` — linear probes via Optax fit on frozen hidden states.
- `pawn/generation.py` — the 5 diagnostics. **Every one of the five takes `outcome_prefix_trained: bool` as a keyword-only argument and returns `{"_skipped": ...}` if False.** KV-cached decoder for performance; variable-prefix-length grouping (sort + stable un-permute) so the batched generator doesn't see ragged prefixes.
- `pawn/lichess_eval.py` — Maia-style Elo-stratified accuracy on held-out Lichess games.
- `pawn/eval_suite/bounds.py` + `viz.py` + `corpus.py` — the polars position-parquet pipeline (theoretical accuracy bounds + plotting). Polars / matplotlib / seaborn are base deps.
- `pawn/eval_suite/diagnostics.py` — edge-case diagnostic eval. Uses `engine.edge_case_bits()` for guaranteed coverage of `in_check` / `double_check` / `pin_restricts` / `ep_available` / `castle_legal_*` / etc. This is the v1 `pawn/eval_suite/diagnostics.py` ported to the JAX model. **Do not omit this.**

**Verification:**
- Acceptance criteria 9 (move-accuracy), 10 (probes), 11 (generation diagnostics + gate), 12 (edge-case diagnostics), 13 (Elo-stratified Lichess). Run each command locally; paste output in the section's commit body.
- For 9: compare the v2 output number against v1's `scripts/eval_accuracy.py` output on `main` for the same checkpoint. Should match within ±0.5 percentage points.

**Definition of done:** criteria 9–13 pass.

### S9 — Sweep + lab + dashboard + wandb

**Goal:** the operational surfaces around training work — sweeps run, the lab MCP server accepts trials, the dashboard renders runs, wandb integration is wired.

**Reference on main:** `pawn/sweep.py`, `pawn/lab/*`, `pawn/dashboard/*`, `pawn/wandb_utils.py`. The per-strategy suggester functions in `sweep.py` are the canonical search-space specs.

**Deliverables:**
- `pawn/sweep.py` — Optuna driver. `AdapterObjective` (subprocess — runs `scripts/train_jax_adapter.py`, parses `metrics.jsonl` for the best `val_loss`, returns to Optuna). Per-strategy `suggest_*` functions matching the suggester surface on `main` — every adapter gets one, including `suggest_rosa` (standard), `suggest_rosa_retro_sparse`, `suggest_rosa_retro_bottleneck`, and the `suggest_rosa_ratio` sweep over the bottleneck-vs-sparse parameter split. Pruning hookup via `trial.report(val_loss, step) + should_prune()`. Persistent study state via SQLite. Field names match the v1 names in `BaseRunConfig` / `AdapterConfig` — `lora_rank` not `rank`, `density` not `sparse_density`, RoSA's `rosa_mode` / `rosa_warmup_steps` / `mask_samples` / `grad_alpha` all preserved. Both `AdapterObjective` (subprocess) and `InProcessRoSAObjective` (the v1 in-process objective that skips per-trial JAX startup for big RoSA sweeps) are in scope; port both.
- `pawn/lab/*` — port forward. `lab_schema` returns `PretrainConfig.model_json_schema() + AdapterConfig.model_json_schema()` directly (no hand-rolled dict). `_validate_config` validates the incoming trial dict through pydantic — `extra="forbid"` rejects stale field names at the lab boundary.
- `pawn/dashboard/*` — Solara UI; reads `metrics.jsonl`, splits train/val by `rec["type"]`.
- `pawn/wandb_utils.py` — optional. `init_wandb`, `log_metrics`, `finish_wandb`. The training drivers conditional-import it on `args.wandb`.

**Verification:**
- Acceptance criterion 14: 3-trial sweep over LoRA rank picks the best `val_loss`.
- Acceptance criterion 19: `pawn-lab` rejects an unknown field through pydantic; `lab_schema` returns valid JSON Schema.
- Dashboard renders a real run dir (no errors in `python -m pawn.dashboard --log-dir <run>`).
- Tests for `pawn.sweep` (the suggester contracts, `_params_to_argv`, `_read_best_val_loss`); tests for `pawn.lab` (the v1 test surface is the spec — port forward, don't reinvent); tests for `pawn.dashboard.metrics` (the JSONL reader).

**Definition of done:** criteria 14 and 19 pass; lab + dashboard + sweep tests are green.

### S10 — Compatibility loader (v1 HF → JAX)

**Goal:** every v1 published HF checkpoint loads through the compatibility bridge into the JAX model with forward parity. There is no second compatibility surface; this is the only bridge.

**Reference on main:** `pawn/model.py` (the v1 torch architecture is the parity oracle). `pawn/checkpoint.py` (the v1 load path).

**Deliverables:**
- `pawn/legacy.py` — `convert_legacy_checkpoint(repo_id_or_path)` reads a v1 torch `.safetensors` checkpoint (from `huggingface_hub.snapshot_download` or a local path), transposes linear weights from `(out, in)` to `(in, out)` JAX convention, builds a `ModelConfig`, writes a JAX checkpoint under `$HF_HOME/pawn-jax-converted/<variant>/`. Rejects pre-vocab-transition checkpoints (older ~60k-token vocab) loudly. Cached by content hash so the second call with the same source is a no-op.
- `pawn/_torch_legacy_fixture.py` — a frozen v1 architecture (private, leading underscore) used only by the converter's parity tests. Not imported anywhere in the JAX surface (`pawn.{model,trainer,...}`).

**Verification:**
- Acceptance criterion 4: convert each of `pawn-{small,base,large}` end-to-end. Mean `‖Δlogit‖ ≤ 1e-3`, max `≤ 1e-4` on a real batch vs. the v1 torch reference. The toy-config converter test gets `~2.4e-7` — don't tighten this for published checkpoints.
- Acceptance criterion 5: `python -c "from pawn.legacy import convert_legacy_checkpoint; convert_legacy_checkpoint('thomas-schweich/pawn-base')"` produces a JAX checkpoint that `pawn.checkpoint.load_model` reads cleanly and forward-evaluates.

**Definition of done:** criteria 4 and 5 pass.

### S11 — Engine fixes

**Goal:** every PGN-to-tokens path in `engine/src/lib.rs` initialises its token buffer with `PAD_TOKEN`, not `0`. The vocab assigns the token `0` to a legal move, so a `0`-initialised tail looks like real moves to downstream consumers.

**Reference on main:** `engine/src/lib.rs`. Grep for `let mut flat_tokens = vec![0i16;` and `let mut flat = vec![0i16;` over token buffers. Audit each: if the buffer ever gets written to with real tokens and read past `game_length`, the init must be `vocab::PAD_TOKEN as i16`.

**Deliverables:**
- Audit + fix every relevant site. (At least four on `main`: `uci_to_tokens`, `parse_pgn_enriched`, `parse_pgn_lichess`, `parse_pgn_sampled`. Confirm against the live file.)
- Comment each fix referencing this section.

**Verification:**
- `cd engine && cargo test` passes.
- Build the extension: `cd engine && uv run --with maturin maturin develop --release`.
- A Rust-side unit test asserts `parse_pgn_*`'s output rows have `tokens[game_length] == PAD_TOKEN` (or sentinel equivalent), not `0`.

**Definition of done:** the engine builds, every PGN-init site is `PAD_TOKEN`-seeded, tests pass.

### S12 — HF-backed checkpoint push + SIGTERM + `--resume`

**Goal:** long runs survive pod restarts and don't lose work.

**Reference on main:** `pawn/checkpoint.py` (the v1 async-push helpers), `pawn/trainer.py` (where the v1 SIGTERM handler is installed), `scripts/train.py` (the v1 `--resume` plumbing).

**Deliverables:**
- HF push: when `--hf-repo` is set, `pawn.checkpoint.save_model(... hf_repo=...)` enqueues an async upload of the new `step_*` directory to the repo's branch. Failures don't block training. Same `hf_bucket` semantics as v1.
- SIGTERM: training drivers install a handler that, on `SIGTERM`, finishes the current chunk, calls `save_model` + the async-push (with a join), prints a graceful-shutdown line, and exits 0.
- `--resume <ckpt>`: the training drivers accept the flag, load `TrainState` from the checkpoint, splice `state.step` from the loaded value (so the metrics log stays monotonic across the resume), and continue.

**Verification:**
- Acceptance criterion 16: train 500 steps, kill, `--resume`, training continues to the original `--total-steps`. Final `state.step` is exact. Metrics rows after the resume have `step > step_at_resume`.
- Acceptance criterion 17: launch a long-ish training run, `kill -TERM <pid>`, process exits 0. The last `step_*` on disk has a `.complete` sentinel.
- Acceptance criterion 18: train with `--hf-repo <scratch-repo>`; `gh api repos/<repo>/git/trees/main` shows the new `step_*` directory in the repo.

**Definition of done:** criteria 16, 17, 18 pass.

### S13 — Scripts + data tools

**Goal:** every script users invoke on `main` has a JAX equivalent.

**Reference on main:** `scripts/`. Every file.

**Deliverables:**
- `scripts/train_jax.py` — pretrain the supernet. Accepts `--config <json>` (validated through `PretrainConfig`). All v1 flags preserved by name.
- `scripts/train_jax_adapter.py` — adapter training. Accepts `--config <json>` (`AdapterConfig`). All 8 strategies via `--strategy`. Lichess data via `--pgn` (default uses `pawn-lichess-full`); `--pgn-val-split` defaults to `validation` (the HF held-out split); pass `--pgn-val-split ""` to carve from train.
- `scripts/eval_jax.py` — move accuracy + per-phase.
- `scripts/eval_probes_jax.py` — probes.
- `scripts/eval_generation_jax.py` — 5 generation diagnostics + edge-case diagnostics. `--outcome-prefix-trained` / `--no-outcome-prefix-trained` flag (default True; matches `BaseRunConfig.prepend_outcome` semantics).
- `scripts/eval_vs_stockfish.py` — Elo-stratified Maia-style accuracy.
- `scripts/sweep.py` — Optuna sweep CLI.
- `scripts/convert_published_checkpoints.py` — one-shot conversion of `pawn-{small,base,large}` to JAX format, with parity check.
- `scripts/run_evals_backbone.py` — batch-eval orchestrator. Takes a list of checkpoints, runs `eval_jax` + `eval_probes_jax` + `eval_generation_jax` + `eval_vs_stockfish` per checkpoint, writes `<output_dir>/<name>/eval_results.json` per checkpoint. Same schema as v1.
- Data tools (unchanged surface): `scripts/extract_lichess_parquet.py`, `scripts/compute_theoretical_ceiling.py`, `scripts/generate_model_cards.py`.

Every training script uses `MetricsLogger` (no inline `json.dumps`). Every script validates its args through pydantic. **All v1 flag names work** — when in doubt, `grep --include='*.py' "argparse" main` and copy.

**Verification:**
- A script-level smoke test per script — `tests/scripts/test_<script>.py` covers the happy path + a couple of validation-failure paths + no-orphan-run-dir on validation failure.
- Acceptance criteria 6, 7, 9, 10, 11, 12, 13, 14 are all driven by these scripts. Run each one locally on a small input; paste a few lines of output into the section commit body.

**Definition of done:** every acceptance criterion that calls a script passes.

### S14 — Docs + Docker + deploy

**Goal:** the user-visible documentation, container images, and deploy scripts reflect the v2 surface.

**Reference on main:** `docs/*.md`, `Dockerfile`, `deploy/*.sh`, `README.md`.

**Deliverables:**
- `docs/jax_migration_plan.md` — this document.
- `CLAUDE.md` — the orientation map.
- `docs/ADAPTERS.md` / `docs/TRAINING.md` / `docs/ARCHITECTURE.md` / `docs/ACCURACY_CEILING.md` — add a top-of-file note that the v1 metrics shown are from v1 PyTorch runs; v2 republishes to new HF repos. Otherwise content survives unchanged.
- `docs/LEGACY.md` — retain as historical record.
- `README.md` — replace v1 train/eval command examples with v2 equivalents; preserve the published-checkpoint disclaimer.
- `Dockerfile` — runtime images install base deps + GPU torch only. Dev images add the optional extras (`dashboard`, `lab`, `wandb`). All v2 entry points.
- `deploy/pod.sh` + `deploy/vast.sh` — example commands point at the JAX scripts. `--logs-dir` flag is used in launch wrappers (matches the trainers' arg name).

**Verification:**
- `docker build --target runtime .` exits 0 (run a couple of commands inside the image).
- `bash deploy/pod.sh --help` and `bash deploy/vast.sh --help` show the JAX example commands.
- Every doc link in `README.md` resolves (`scripts/...`, `docs/...`).

**Definition of done:** docs / images / deploy reference v2. No leftover v1 command examples.

### S15 — Comprehensive tests

**Goal:** every JAX-side surface has tests; the whole tree collects + passes under `uv run --extra rocm pytest tests/`.

**Reference on main:** every `tests/*` file. Each one was written for a reason — the JAX equivalent should cover the same invariants (different code, same behaviours).

**Deliverables:**
- Per-module JAX-side tests: `tests/test_jax_<module>.py` for every `pawn/<module>` from S2–S12.
- Per-script smoke tests in `tests/scripts/`.
- Engine tests pass.
- `tests/test_public_api.py` pins the post-swap public surface.

**Verification:**
- `uv run --extra rocm pytest tests/ -q` exits 0. Paste the final summary line into the section commit body.

**Definition of done:** acceptance criterion 3 passes. Full green.

### S16 — Final review + framework-swap PR

**Goal:** the framework swap merges. Every §3 acceptance criterion has been re-graded on the work branch, every silent deletion has been caught, the multi-perspective final review is clean, and the PR is open with the verification evidence attached.

**Reference on main:** every acceptance criterion in §3; the project's `DEFERRALS.md` (if any entries exist).

**Deliverables:**
- `final_smoke.md` at the repo root (or under `docs/`) — for every §3 acceptance criterion, the verification command + a pasted output excerpt proving it passed. **Don't summarise; paste actual output.**
- A passing `review-spec-alignment` run scoped to the entire feature branch (`--base main`) against this plan.
- A passing multi-perspective final review (bug-detector / type-correctness / test-risk / simplification / doc-accuracy / codex) iterated to clean.
- An open PR against `main`, body containing: the final_smoke.md artifact (or link), the §3 criteria grade, the contents of `DEFERRALS.md` (each entry one line in the PR summary), and the squash-merge history of the migration.

**Verification:**
- `uv run --extra rocm pytest tests/` — green.
- Re-run each §3 criterion command; every one passes. Paste the outputs into `final_smoke.md`.
- `review-spec-alignment` returns VERDICT: PASS against the feature branch.
- The multi-lane review (run separately) returns clean — no Critical / Important findings unaddressed.

**Definition of done:** every §3 criterion passes on the work branch with output pasted into `final_smoke.md`; PR is open with the full artifact.

## 11. Things that are easy to drop and shouldn't be

These showed up as "obvious to skip" during planning. They aren't.

- **The Lichess data path.** PAWN is a finetuning testbed. The realistic adapter task is Elo-stratified human-move prediction. Random games are a verification proxy, not a substitute. The Lichess adapter path is in §3 (criterion 7) for a reason.
- **The held-out validation split.** The `thomas-schweich/pawn-lichess-full` dataset ships proper `train` / `validation` / `test` splits. Carving val out of train silently leaks. Default to the held-out split; carve only when the source doesn't have one (single-file local parquets).
- **The edge-case diagnostic eval.** `pawn/eval_suite/diagnostics.py` on `main` is a real eval surface — guaranteed-coverage edge cases via `engine.edge_case_bits()`. It's distinct from the generation diagnostics. If you find yourself merging these two concepts, stop; they cover different things.
- **`MetricsLogger` wiring inside the trainers.** The module existing isn't the work. The work is the trainers calling `log_config` / `log_train` / `log_val` instead of `json.dumps`. If `grep -n "json.dump" scripts/train_jax*.py` finds anything, you're not done.
- **The `outcome_prefix_trained` gate on all 5 generation diagnostics.** `impossible_task_test` and `improbable_task_test` are obviously gated. So are `outcome_signal_test`, `prefix_continuation_test`, `poisoned_prefix_test` — all five condition on outcome at position 0.
- **HF-backed checkpoint push and SIGTERM.** These exist on `main` and users rely on them for long runs. Don't ship without them.
- **`--resume`.** Long runs get interrupted. `--resume` is the recovery path. Wire it.
- **All v1 CLI flag names.** When users type `--lora-rank 4`, that has to keep working. When they type `--use-output-film`, that has to keep working. When they type `--no-adapt-attn`, ditto. Don't rename for taste.

## 12. Polars, matplotlib, seaborn, jinja2, zstandard belong in base

These are used by the realistic adapter path (Lichess parquet), the legacy eval_suite (bounds + viz), and the data/publishing scripts. They are not "optional tooling"; they're load-bearing for workflows users actually run. Move them to `[project.dependencies]`. Don't ship them behind an extra.

Extras live for things genuinely optional: the GPU backend choice (`rocm` / `cu128`), `dashboard`, `lab`, `wandb`.

## 13. When you're unsure

The default move is: **read `main`.** Failing that, **run `main`** to observe behaviour. Failing that, pick the most conservative interpretation that preserves observable v1 behaviour and file the decision in the section's commit body so it's visible. Don't invent.

If something looks like it can be deferred, it probably can't. The acceptance criteria in §3 are the bar. Everything else is supporting work for them.

If you're at the end of a section and the section's verification command doesn't pass, the section is not done. "Mostly" doesn't ship.

If you find yourself writing a commit body that says "we'll wire this later," check §9.4. Either write the `DEFERRALS.md` entry (with the genuine reason — nonsensical / impossible / actively detrimental) or wire it now. There is no third option.
