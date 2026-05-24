# JAX Parity Shortfalls

This document records an audit of the `jax_migration` branch against `main`
and against the acceptance criteria in `docs/jax_migration_plan.md`.

The migration goal is 100% parity: every workflow and durable surface that
worked on `main` should either work on the JAX branch or have an explicit,
defensible, documented replacement. The current branch does not meet that bar.

No code changes are proposed here. This is an audit artifact.

## Summary

The branch contains a broad PyTorch to JAX rewrite, but it also removes or
weakens large portions of the v1 surface. The highest-risk gaps are:

- bf16 mixed precision is not implemented; the branch uses fp32 throughout.
- `final_smoke.md` claims acceptance criteria are satisfied, but many criteria
  were verified with weaker substitutes such as help output, schema tests,
  random-game runs, or unit tests.
- The top-level public API is intentionally changed instead of preserved.
- Several user-facing scripts from `main` were deleted rather than preserved as
  compatibility wrappers.
- Adapter implementations are not feature-equivalent ports.
- RoSA modes and the RoSA three-phase schedule are mostly nominal.
- Adapter training does not use the planned K-step scan and lacks resume
  plumbing.
- Legacy checkpoint conversion lacks the required real published-checkpoint
  logit parity check and caches by source string rather than content hash.
- Generation and edge-case diagnostics are skeletal compared with v1.
- Metrics memory keys do not match the v1 schema exactly.
- Much of the v1 test suite was deleted, so the new tests do not prove parity.

## 1. bf16 Mixed Precision Is Missing

`docs/jax_migration_plan.md` explicitly requires bf16 mixed precision:

- Plan section 5 says forward casts parameters to bf16 with fp32 accumulation,
  while the master copy and Adam moments stay fp32.
- The stated performance motivation depends partly on JAX plus resident corpus
  plus bf16.

Current code initializes and carries model parameters in fp32:

- `pawn/model.py` initializes normal weights with `dtype=jnp.float32`.
- RMSNorm weights, `embed_pad`, and other trainable arrays are also fp32.
- The converted legacy model path uses `jnp.asarray(...)` without bf16 forward
  casting.

Relevant files:

- `pawn/model.py`
- `pawn/trainer.py`
- `pawn/adapter_trainer.py`
- `pawn/run_config.py`

The config layer also explicitly drops the old precision fields:

```python
Torch-only fields (``amp_dtype``, ``device``, ``num_workers``,
``no_compile``, ``sdpa_math``) are dropped — JAX manages its own
backend.
```

That removal is not equivalent to implementing the planned bf16 behavior.
There is no obvious replacement field or unconditional bf16 execution path.

Impact:

- The branch does not meet the migration plan's precision contract.
- Throughput and memory expectations are likely wrong.
- Numerical behavior differs from v1's bf16 AMP training path.

## 2. `final_smoke.md` Overstates Acceptance-Criteria Coverage

`final_smoke.md` claims every acceptance criterion has a passing verification.
Several entries do not run the required workflow.

### Criterion 4: Published Checkpoint Logit Parity

The plan requires converting published v1 checkpoints and reporting logit
parity against the v1 torch reference:

- mean logit delta <= 1e-3
- max logit delta <= 1e-4
- run on real published `pawn-{small,base,large}` checkpoints

`final_smoke.md` instead says the real v1-v2 logit comparison is part of a
future v2-publish flow, and uses synthetic fixture tests plus non-trivial
accuracy as evidence.

That is not the required verification.

Files:

- `final_smoke.md`
- `scripts/convert_published_checkpoints.py`
- `pawn/legacy.py`
- `tests/test_jax_legacy.py`

### Criterion 7: Real Lichess LoRA Fine-Tune

The plan requires:

```bash
uv run --extra rocm python scripts/train_jax_adapter.py \
  --strategy lora --supernet tiny --variant base --lora-rank 4 \
  --total-steps 200 --pgn thomas-schweich/pawn-lichess-full \
  --elo-min 1800 --elo-max 2000 --local-checkpoints
```

with validation coming from the held-out `validation` split.

`final_smoke.md` uses `--no-pgn`, which trains on random generated games
instead of Lichess parquet.

That is a different task and does not verify Lichess data loading,
split handling, Elo filtering, or held-out validation.

Files:

- `final_smoke.md`
- `scripts/train_jax_adapter.py`
- `pawn/lichess_data.py`

### Criterion 10: Linear Probes

The smoke file records `eval_probes_jax.py --help` and says schema tests cover
the output. It does not show a real probe run against a converted checkpoint.

Files:

- `final_smoke.md`
- `scripts/eval_probes_jax.py`
- `pawn/probes.py`

### Criterion 11: Generation Diagnostics

The smoke file verifies the existence of the five diagnostic names and the
skip gate behavior. It does not run the actual diagnostics on a converted
checkpoint and does not demonstrate equivalent generation behavior.

Files:

- `final_smoke.md`
- `scripts/eval_generation_jax.py`
- `pawn/generation.py`

### Criterion 12: Edge-Case Diagnostics

The smoke file describes script wiring and tests, but it does not show the
required edge-case diagnostic command producing the expected structure.

The implementation also samples 64 random games in the script path instead of
using the quota-controlled diagnostic corpus from v1.

Files:

- `final_smoke.md`
- `scripts/eval_generation_jax.py`
- `pawn/eval_suite/diagnostics.py`

### Criterion 13: Elo-Stratified Lichess Accuracy

The smoke file records `eval_vs_stockfish.py --help` and schema assertions, not
a live Elo-stratified Lichess run.

Files:

- `final_smoke.md`
- `scripts/eval_vs_stockfish.py`
- `pawn/lichess_eval.py`

### Criteria 16-18: Resume, SIGTERM, HF Push

The smoke file uses unit tests for helper functions instead of live process
checks:

- train, kill, resume to exact `total_steps`
- launch long run, send SIGTERM, verify final complete checkpoint
- train with `--hf-repo`, verify uploaded `step_*` in HF repo

Files:

- `final_smoke.md`
- `pawn/lifecycle.py`
- `scripts/train_jax.py`
- `scripts/train_jax_adapter.py`

Impact:

- The migration's final verification artifact is unreliable.
- Multiple required workflows remain unproven.
- The "no deferrals" claim in `DEFERRALS.md` is not supported.

## 3. Top-Level Public API Is Not Preserved

On `main`, `pawn/__init__.py` exposes:

```python
from pawn.config import CLMConfig, TrainingConfig
from pawn.model import PAWNCLM

__all__ = ["CLMConfig", "TrainingConfig", "PAWNCLM"]
```

The old public API test explicitly pins:

```python
from pawn import CLMConfig, TrainingConfig, PAWNCLM
```

On the JAX branch, `pawn/__init__.py` is intentionally docstring-only. The new
test asserts that this is expected.

Files:

- `pawn/__init__.py`
- `tests/test_public_api.py`

Impact:

- Existing users importing `CLMConfig`, `TrainingConfig`, or `PAWNCLM` from
  `pawn` break immediately.
- This is an intentional API change, not parity.
- If the migration wants a lightweight `import pawn`, it still needs a
  compatibility answer for existing import sites.

## 4. User-Facing Scripts Were Deleted Instead of Preserved

The branch deletes several scripts that existed on `main`, including:

- `scripts/train.py`
- `scripts/eval_accuracy.py`
- `scripts/eval_probes.py`
- `scripts/export_hf_repo.py`

New scripts exist, such as:

- `scripts/train_jax.py`
- `scripts/train_jax_adapter.py`
- `scripts/eval_jax.py`
- `scripts/eval_probes_jax.py`

However, the migration plan says the same CLI surface should work, or a rename
must be documented. It also says names users type are durable.

The new argparse surfaces are much narrower than the old dynamic config-driven
`scripts/train.py` path. For example:

- `scripts/train_jax.py` only exposes a small subset of pretraining flags.
- `scripts/train_jax_adapter.py` exposes a subset of adapter flags and omits
  several fields that remain in `AdapterConfig`.
- `scripts/train_jax_adapter.py` has no `--resume` flag.

Files:

- `scripts/train_jax.py`
- `scripts/train_jax_adapter.py`
- `scripts/eval_jax.py`
- `scripts/eval_probes_jax.py`
- `scripts/eval_generation_jax.py`
- `scripts/eval_vs_stockfish.py`

Impact:

- Existing scripts and workflows invoking v1 entry points break.
- The new names may be reasonable, but compatibility wrappers or documented
  migrations are missing.
- The branch does not satisfy "every command that works on origin/main works
  after this migration."

## 5. Adapter Implementations Are Not Feature-Equivalent Ports

The migration plan explicitly says every adapter on `main` must port forward,
including tested workflows and hyperparameter contracts.

The current adapter code preserves some names but often not behavior.

### Bottleneck

`pawn/adapters/bottleneck.py` says:

```python
For simplicity we attach to FFN only
```

The v1 flags `no_adapt_attn` and `no_adapt_ffn` exist, but attention-side
adaptation is not actually implemented. `bottleneck_n_hidden` is accepted in
config and dataclass fields, but the computation does not implement extra
hidden stages.

Files:

- `pawn/adapters/bottleneck.py`
- `pawn/run_config.py`
- `scripts/train_jax_adapter.py`

Impact:

- v1 bottleneck configurations with attention adapters do not have equivalent
  behavior.
- Search/sweep results using `bottleneck_n_hidden` are misleading because the
  knob is effectively inert.

### Sparse

`SparseConfig` has:

```python
ffn: bool = False
```

But `SparseAdapter` only contains Q/K/V/O fields, and `apply_sparse` only
modifies attention projections. FFN sparse adaptation is not implemented.

Files:

- `pawn/adapters/sparse.py`
- `pawn/run_config.py`
- `scripts/train_jax_adapter.py`

Impact:

- `sparse_ffn` from `AdapterConfig` is not wired into the strategy config.
- v1 sparse configurations that adapted FFN are not equivalent.

### LoRA

`LoRAConfig` supports `ffn`, but `scripts/train_jax_adapter.py` passes
`ffn=cfg.lora_ffn` only for the plain LoRA strategy. Hybrid constructs
`LoRAConfig(rank=cfg.lora_rank or 4)` without forwarding targets or FFN
settings.

Files:

- `pawn/adapters/lora.py`
- `pawn/adapters/hybrid.py`
- `scripts/train_jax_adapter.py`

Impact:

- Some v1 LoRA/hybrid knobs do not survive through the CLI dispatch.

### Unfreeze

The explicit `unfreeze_layers="5,6,7"` form is present, which is good.
However, `unfreeze_filter` marks all floating adapter-layer leaves trainable.
Masked layers receive zero gradients through the forward path, but AdamW
weight decay behavior for masked slots depends on optimizer/update semantics.
The comment says no drift occurs because gradients are zero, but decoupled
AdamW can still apply parameter decay even with zero gradients unless the
update is explicitly masked.

Files:

- `pawn/adapters/unfreeze.py`
- `pawn/adapter_trainer.py`

Impact:

- Masked unfreeze slots may drift under weight decay.
- A focused test should confirm byte-stability of masked slots across updates
  when `weight_decay > 0`.

## 6. RoSA Modes Are Mostly Nominal

The migration plan explicitly calls out RoSA as non-negotiable:

- `rosa`
- `retro-sparse`
- `retro-bottleneck`
- `mask_samples`
- `grad_alpha`
- `rosa_warmup_steps`
- three-phase schedule:
  1. LoRA warmup
  2. gradient-magnitude mask generation
  3. joint training under fixed mask

Current code has `RoSAConfig` fields and strategy names, but the three modes
share the same init/apply/filter functions:

- `rosa`
- `rosa-retro-sparse`
- `rosa-retro-bottleneck`

`pawn/adapter_trainer.py` mentions a `run_rosa_schedule` helper in the
docstring, but no such function exists. There is no implementation of the
three-phase schedule, no mask generation using `mask_samples`, and no
different behavior for retro-sparse or retro-bottleneck beyond a static mode
field.

Files:

- `pawn/adapters/rosa.py`
- `pawn/adapter_trainer.py`
- `scripts/train_jax_adapter.py`
- `pawn/run_config.py`
- `pawn/sweep.py`

Impact:

- RoSA parity is not implemented.
- Tests currently verify dispatch and shape-level behavior, not the v1 RoSA
  algorithm.
- Sweeps over RoSA modes will produce results for an implementation that does
  not match the named methods.

## 7. Adapter Training Does Not Use the Planned K-Step Scan

`pawn/adapter_trainer.py` implements `make_adapter_scan_step`, but
`scripts/train_jax_adapter.py` does not use it. The script loops in Python:

```python
for step in range(cfg.total_steps):
    ...
    state, loss = train_step(state, batch)
```

The `--k` argument is parsed into config but does not drive a K-step scan in
the adapter script.

Files:

- `pawn/adapter_trainer.py`
- `scripts/train_jax_adapter.py`

Impact:

- The adapter trainer does not meet the migration plan's compiled K-step
  training-loop design.
- Throughput measurements and acceptance criteria that assume scan-based
  amortization are not representative.
- `K <= val_every` constraints are not meaningful in the current adapter
  script.

## 8. Adapter Resume Is Missing

`AdapterConfig` inherits `resume`, but `scripts/train_jax_adapter.py` has no
`--resume` argument and no adapter-state resume path.

Additionally, adapter checkpoints save the effective merged model:

```python
effective = apply_fn(state.backbone, state.adapter)
save_model(effective, out, ...)
```

This makes the checkpoint usable as a normal model, but it does not preserve
the adapter/backbone split needed to resume adapter training as the same
strategy. The saved optimizer state is for adapter parameters, but the saved
model is not the adapter PyTree those optimizer leaves correspond to.

Files:

- `scripts/train_jax_adapter.py`
- `pawn/lifecycle.py`
- `pawn/checkpoint.py`

Impact:

- Long adapter runs cannot be resumed with strategy state intact.
- Criterion 16 is only partially addressed for pretraining, not adapter
  training.

## 9. Legacy Converter Does Not Meet the Full Contract

The plan says `convert_legacy_checkpoint` should cache by content hash. The
implementation caches by source identifier:

```python
SHA-256 of the source *identifier* string
```

The docstring explicitly says this is not a content hash and that republished
weights under the same source do not invalidate the cache.

Files:

- `pawn/legacy.py`

Impact:

- Stale conversions are possible.
- The behavior does not match the migration plan.

The converter script also does not run the required real logit parity check.
It converts and loads the model, then prints metadata:

```python
model = load_model(out)
results.append({... "status": "ok"})
```

Files:

- `scripts/convert_published_checkpoints.py`
- `tests/test_jax_legacy.py`

Impact:

- Published checkpoint parity remains unproven.
- Synthetic fixture parity is useful but not a substitute for real HF
  checkpoint parity.

## 10. Generation Diagnostics Are Skeletal

The v1 generation path includes autoregressive generation, optional legal
masking, engine-managed state, prefix handling, and KV-cache-aware model
generation.

The JAX branch's `pawn/generation.py` says:

```python
For the trainer-side smoke check the gate is the load-bearing behavior;
the actual diagnostic values are computed at S13 via
`scripts/eval_generation_jax.py`. This module ships the gate +
skeleton scoring loop.
```

The implementation computes simple distribution probes and argmax next-token
checks. It does not implement an equivalent autoregressive generator or
KV-cached decoder.

Files:

- `pawn/generation.py`
- `scripts/eval_generation_jax.py`
- deleted `pawn/eval_suite/generation.py`

Impact:

- Criterion 11 is not actually satisfied.
- v1 generation diagnostic behavior is not available.
- The output may have the right names but not the same semantics.

## 11. Edge-Case Diagnostics Dropped Categories and Quota-Controlled Generation

On `main`, edge diagnostics include categories such as:

- `in_check`
- `double_check`
- `pin_restricts`
- `ep_available`
- `castle_legal_k`
- `castle_legal_q`
- `castle_blocked_check`
- `promotion_available`
- `checkmate`
- `stalemate`

The JAX branch only keeps:

- `in_check`
- `double_check`
- `pin_restricts`
- `ep_available`
- `castle_legal_kingside`
- `castle_legal_queenside`

The v1 path uses `engine.generate_diagnostic_sets` with quotas for guaranteed
coverage. The JAX script path uses `engine.generate_random_games(64, 64, 42)`.

Files:

- `pawn/eval_suite/diagnostics.py`
- `scripts/eval_generation_jax.py`
- deleted v1 `pawn/eval_suite/diagnostics.py` content from `main`

Impact:

- Rare edge cases may have zero coverage.
- Several v1 diagnostic categories are missing.
- Criterion 12 is only superficially implemented.

## 12. Move Accuracy Eval Uses Random Games Only

`scripts/eval_jax.py` always builds a random generated corpus:

```python
corpus = generate_corpus(...)
```

The acceptance criterion asks for move-prediction accuracy and per-phase eval
matching v1 numbers for the same checkpoint. The v1 workflows include
published evaluation datasets and evaluation scripts beyond random generated
games.

Files:

- `scripts/eval_jax.py`
- `pawn/eval.py`

Impact:

- Accuracy parity is only checked on a random-game distribution.
- Reported accuracy may not correspond to the v1 evaluation workflow users
  relied on.

## 13. Metrics Schema Is Not Exact v1 Parity

The plan says `pawn/logging.py` on `main` is the metrics schema spec.

On `main`, GPU stats from torch use fields:

- `mem/gpu_peak_gb`
- `mem/gpu_reserved_gb`
- `mem/gpu_current_gb`

The JAX branch emits:

- `mem/gpu_used_gb`
- `mem/gpu_total_gb`

Files:

- `pawn/logging.py`
- `pawn/dashboard/metrics.py`

Impact:

- Existing dashboards or scripts expecting v1 GPU keys may break.
- `final_smoke.md` accepts the new fields, but the plan required v1 schema
  parity.

## 14. Lichess Acceptance Is Not Proven End-to-End

`pawn/lichess_data.py` has a real parquet/cache implementation and tests, but
the branch does not show the required live run against
`thomas-schweich/pawn-lichess-full`.

`final_smoke.md` says the tokenized dataset requires a large download and uses
random games instead.

Files:

- `pawn/lichess_data.py`
- `scripts/train_jax_adapter.py`
- `scripts/eval_vs_stockfish.py`
- `final_smoke.md`

Impact:

- The most important adapter workflow remains unverified.
- Held-out validation split behavior is tested synthetically but not proven on
  the real dataset.

## 15. Tests Were Replaced, Not Used as Parity Guards

The diff deletes large test suites from `main`, including tests for:

- adapters
- checkpoint
- config
- logging
- run config
- eval corpus/generation/diagnostics/probes/viz/worker
- lab runner/server/sweep/dashboard metrics
- model/data/lichess cache/specialized CLM
- training scheduler/trainer/adapter training/cotrain/resume/wandb
- script smoke tests

The branch adds many JAX-specific tests. Those are useful, but they mostly pin
the new implementation rather than proving old behavior survived.

Files:

- `tests/`
- `tests/test_public_api.py`
- `tests/test_jax_*.py`

Impact:

- Regressions against v1 behavior can be hidden by changing the tests to match
  the new code.
- The deleted tests should be triaged one by one: either port the behavioral
  assertion, document a gone-by-design removal, or retain a compatibility test.

## 16. `DEFERRALS.md` Is Incorrect

`DEFERRALS.md` says:

```markdown
No deferrals.

The framework swap implements the full §3 acceptance-criteria contract.
Every v1 surface that existed on `main` has a v2 counterpart on this
branch
```

That statement is inconsistent with the shortfalls above.

Files:

- `DEFERRALS.md`

Impact:

- Reviewers may incorrectly assume missing behavior was audited and found
  complete.
- This file should either be corrected or replaced with explicit deferrals,
  though the plan states acceptance criteria cannot be deferred.

## Suggested Remediation Order

1. Restore or wrap the deleted user-facing CLI entry points so `main` commands
   continue to work, or document every rename explicitly.
2. Implement real bf16 mixed precision: bf16 forward compute with fp32 master
   weights and optimizer state.
3. Re-run the acceptance criteria exactly as written, especially published
   checkpoint logit parity, real Lichess LoRA training, real probes,
   generation diagnostics, edge-case diagnostics, Elo-stratified eval, resume,
   SIGTERM, and HF push.
4. Port adapter behavior fully, with special focus on bottleneck placement,
   sparse FFN support, LoRA/hybrid target propagation, and RoSA three-phase
   mode-specific behavior.
5. Rebuild generation and edge-case diagnostics from the v1 semantics, not just
   the names.
6. Restore public API compatibility or provide backward-compatible aliases.
7. Reconcile metrics field names with the v1 schema.
8. Port deleted v1 tests as behavioral parity tests where the behavior is still
   in scope.
9. Correct `final_smoke.md` and `DEFERRALS.md` so they reflect actual verified
   status.

## Current Conclusion

The branch is a substantial JAX rewrite, but it is not yet a parity migration.
It should not be treated as feature-complete until the missing workflows above
are either implemented and verified or explicitly removed from the parity
contract by a human design decision.
