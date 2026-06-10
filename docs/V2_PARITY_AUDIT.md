# v1→v2 Parity, Correctness & Performance Audit — PAWN JAX Overhaul

> **Provenance.** Generated 2026-05-30 by a gated multi-agent audit (`.wf_audit.mjs`,
> run `wf_63a992b2-e6d`): 21 feature areas × independent sonnet verifier →
> adversarial opus gate (false-positive + false-negative hunt) → opus tiebreak on
> 48 contested items → opus synthesis. 91 agents, 266 raw findings. Static
> code-level analysis only (no training/eval was run); empirical confirmation is
> the §7 runtime checklist. Precedence on conflict: tiebreak > gate > verifier.

## 1. Verdict

**v2 is NOT yet a strict functional superset of v1.** The architecture, training loops, adapters, distillation, checkpointing, and eval *cores* are at parity-or-better, but a concrete set of v1 workflows are broken or impossible in v2. The exact short list of what's missing/broken:

- **Lab trial-orchestration daemon is gone** — `TrialRunner` + 8 of 10 MCP tools (`lab_status/kill/resume/results/events/log/notes/set_cost/audit`), GPU-aware scheduling, crash recovery, event bus, and interactive HP-suggestion all have no v2 path (blocker).
- **`hf_bucket` storage mode is silently dead** — config validates a bucket-only run, then a pretrain run saves/pushes nothing (blocker).
- **Solara dashboard crashes on import** — `sol.py` imports 7 symbols removed from `metrics.py` (blocker).
- **Adapter-on-Lichess accuracy eval is gone** — `eval_accuracy.py` wrapper hard-fails on `--adapter-checkpoint`/`--pgn`; bottleneck/FiLM adapters silently eval the bare backbone.
- **No backbone-pretrain validation loop** — zero `val/*` records emitted; model-card generator KeyErrors; dashboard val charts dead; no early-stopping/patience.
- **Per-layer adapter placement (`--adapter-layers`), legal-move masking + `illegal_penalty` in adapter loss, regression (MSE/R²/MAE) probes** — all v1 capabilities with no v2 path.
- **HF push lands on `main`, not per-run `run/{run_id}` branch** — per-run isolation + squash-merge workflow broken.
- Plus a cluster of generation-diagnostic scope reductions and eval-schema/metric drops (top-5, loss, perplexity, legal_move_rate, per-ply, control arms).

The supernet/embedding/precision redesign and the distillation/KV-cache/conditioning additions are genuine, well-tested superset improvements. The blocker count is driven by the lab subsystem, the bucket-mode trap, and the dashboard import break.

---

## 2. Confirmed blockers & regressions

Only items surviving gate/tiebreak as REAL. Severity reflects final tiebreak ruling where one exists.

| Area | Issue | Status | Severity | Evidence (file:line) | Suggested fix |
|---|---|---|---|---|---|
| Lab MCP | `TrialRunner` daemon entirely absent (GPU discovery, state persistence, spawn/monitor/kill/resume, event bus, recovery) | gap | **blocker** | `pawn/lab/runner.py:1-196` (no class); `git show main:pawn/lab/runner.py:55,237,522,990` | Port `TrialRunner` forward or write a DEFERRALS.md entry; plan §6:144 lists `pawn/lab/*` as IN SCOPE |
| Lab MCP | 8 of 10 MCP tools missing (`lab_status/kill/resume/results/events/log/notes/set_cost/audit`) | gap | **blocker** | `pawn/lab/server.py:26-36` (only `lab_schema`, `lab_launch`) | Re-expose tools backed by the ported runner |
| Lab MCP | v1 test surface (`test_runner.py` 68, `test_server.py` 26, `test_sweep.py` 16) not ported | gap | **blocker** | `ls tests/lab/` (only monitor/state/charts); plan:427 "v1 test surface is the spec" | Port the tests with the runner |
| Lab MCP | `pawn/lab/sweep.py` `builtin_distributions`/`parse_distribution` (interactive HP suggestion via `study.ask`) gone | gap | major | `git show main:pawn/lab/sweep.py:10-98`; `runner.py:699-726` | Port the Distribution-dict API + suggester |
| Checkpoint/HF | `hf_bucket`-only run validates but pretrain saves/pushes nothing (gated on `local_checkpoints or hf_repo`) | regression | **blocker** | `pawn/run_config.py:328-335`; `scripts/train_jax.py:684`; `pawn/lifecycle.py:159` (repo_id only); plan:471 "Same hf_bucket semantics as v1" | Wire `submit_bucket`/`push_checkpoint_to_bucket` or reject `hf_bucket` at config time |
| Dashboard | `sol.py` imports 7 symbols removed from `metrics.py` `__all__` → ImportError on any UI launch | regression | **blocker** | `pawn/dashboard/sol.py:19-28,815`; `pawn/dashboard/metrics.py:15-19`; `__init__.py:18-21` routes `Dashboard`→`sol` | Update `sol.py` imports + `load_metrics` call sites to the new `MetricsBundle` API |
| Eval | Adapter-on-Lichess accuracy eval has no v2 path; bottleneck/FiLM adapter checkpoints silently eval bare backbone | partial | major | `scripts/eval_accuracy.py:16-25` (execvp); `scripts/eval_jax.py:19-26`; `pawn/checkpoint.py:496-540` (loads only model.safetensors); `train_jax_adapter.py:240-255` (FiLM/bottleneck sidecar) | Add an adapter-aware Lichess accuracy eval that reapplies typed sidecars |
| Pretrain | No backbone-pretrain validation loop — zero `val/*` records; model-card gen KeyErrors; dashboard val charts dead; no patience/early-stop | gap | major | `scripts/train_jax.py:631-694` (no `log_val`); `git show main:pawn/trainer.py:1159-1221`; `generate_model_cards.py:288-302` (direct subscripts); plan:173 "emit a separate type:'val' record like v1" | Add periodic held-out eval + `log_val` emitting v1's `val/*` keys |
| Adapters | Per-layer adapter placement (`--adapter-layers`) inert; `adapter_layers` accepted via `--config` then silently ignored | gap | major | `pawn/run_config.py:528` (dead field); `scripts/train_jax_adapter.py:384-464` (never read); v1 `pawn/adapters/lora.py:88-97`, `adapter_training.py:832` | Thread a layer-subset param into Lora/Sparse/Bottleneck/Hybrid/RoSA configs, or DEFERRALS entry |
| Adapters | Legal-move masking in adapter loss (default ON in v1) inert in v2; loss is full-vocab, not legality-restricted | gap | major | v1 `adapter_training.py:378-380,1111`; v2 `pawn/trainer.py:236-320` (only reserved-col mask); `run_config.py:589` dead `disable_legal_mask` | Port legal-mask into the adapter CE loss path, or DEFERRALS entry |
| Adapters | `illegal_penalty` loss term inert (field+validator+docs present, no consumer) | gap | major | v1 `adapter_training.py:382-387,548-549`; v2 `run_config.py:590,714-728`; docs claim it live `docs/ADAPTERS.md:159-170` | Wire the term or remove field+docs+validator |
| Adapters | Patience / val-plateau early stopping inert in adapter loop (and pretrain) | gap | major | v1 `adapter_training.py:1984-2087`, `trainer.py:1183-1243`; v2 `run_config.py:123` dead; `scripts/train_jax_adapter.py:967` step-budget only; charts/lifecycle reference a `patience` reason never emitted | Implement best-val tracking + patience break, or DEFERRALS entry |
| Probes | Regression probes (MSE / R²/ MAE) entirely absent — `fit_probe` hardcodes cross-entropy; v1 material_count/legal_move_count/halfmove_clock gone | gap | major | v1 `eval_suite/probes.py:34-37,456-484`; v2 `pawn/probes.py:123-127` | Add a regression head/loss path + R²/MAE metrics to `fit_probe`/`ProbeResult` |
| Eval | `eval_jax.py`/`lichess_eval.py` report no top-5, cross-entropy loss, or per-ply breakdown | partial | major | v1 `eval_accuracy.py:394-403,445`; v2 `pawn/eval.py:56-79` (top-1 only) | Add top-k + CE-loss + `--per-ply` to the eval API |
| Eval | `legal_move_rate`/`late_legal_move_rate` per-band metric absent; orphaned consumers KeyError (`generate_model_cards.py:291`) | gap | major | v1 `eval_suite/lichess.py:275`, `trainer.py:1045-1057`; v2 grep empty; engine prim still exists `engine/src/labels.rs:12` | Re-emit legality rate (engine prim available) |
| Edge-case | Model-card edge-case diagnostics section silently empty end-to-end (`run_evals_backbone` no `--edge-cases`; writes `edge_cases` not `diagnostics`; stale castle keys) | regression | major | `scripts/run_evals_backbone.py:76-96`; `eval_generation_jax.py:128-135`; `generate_model_cards.py:138-139,330-334` | Wire edge data into `diagnostics` key; update label names |
| Edge-case | Per-category sampled legal-rate / PAD-prob / entropy diagnostic replaced by argmax-accuracy only — distributional signal dropped | partial | major | v1 `eval_suite/diagnostics.py:331-351`; v2 `diagnostics.py:113-117` | Restore sampled per-category legality/entropy (or document the substitution) |
| Elo eval | Per-bin schema is `{elo_bin, accuracy, n_games}` — drops v1's `n_tokens/loss/perplexity/top5/legal_move_rate`; plan §3:62 says "schema matches v1" | partial | major | v1 `eval_suite/lichess.py:268-276`; v2 `pawn/lichess_eval.py:39-43`; `final_smoke.md:428` false claim | Restore dropped fields or amend the contract |
| Checkpoint/HF | All HF pushes land on `main`, not per-run `run/{run_id}` branch | regression | major | `pawn/lifecycle.py:159`; `scripts/train_jax.py:338`; `train_jax_adapter.py:824`; CLAUDE.md:322-326 contradicts | Pass `branch=f"run/{slug}"` into `HFPushTracker`; restore squash-merge workflow |
| Sweeps | No mid-trial pruning (`trial.report`+`should_prune`) anywhere; plan §10 S9 requires it | partial | major | plan:418; v1 `sweep.py:808-811,840-871`; v2 `sweep.py:341-347` (failure-path only) | Wire intermediate `trial.report`/`should_prune` + pruner CLI |
| Sweeps | `InProcessRoSAObjective` reduced to a 14-line callable shim — no in-process 3-phase trainer, no pruning; plan S9 "port both" | partial | major | v1 `sweep.py:421-833`; v2 `sweep.py:357-370` | Port the in-process trainer or DEFERRALS entry |
| Distillation | `train_jax_distill.py` lacks `--resume`; interrupted run loses all progress (saves resumable artifacts it never reloads) | gap | info* | `scripts/train_jax_distill.py:90-132,203-207,280-281` | Add `--resume` (parity with other v2 trainers). *Tiebreak: not a v1-parity gap (no v1 distillation); internal-consistency note |
| Eval gen | `prefix_continuation_test` cross-conditioning over 5 outcomes dropped; only single hardcoded `prefix=[5,10]`+1 outcome, `n_continuations=8` | partial | major | v1 `eval_suite/generation.py:402-522`; v2 `generation.py:619-672,806,813-818` | Restore corpus-driven multi-bucket cross-conditioning |
| Eval gen | `poisoned_prefix_test` reduced to single pair; `original_outcome_match_rate` (capitulation signal) gone | partial | major | v1 `generation.py:543-611`; v2 `generation.py:675-699,819-823` | Restore POISONING_PAIRS + original-outcome tracking |
| Eval gen | `improbable_task_test` corpus scenarios + control arms (`control_few_ply/control_early`) gone | partial | major | v1 `generation.py:700-774`; v2 `generation.py:736-779` | Restore control arms (the scientific baseline) |
| Eval gen | Within-test CONTROL arms in impossible/improbable tests absent (no matched-prefix baseline) | partial | major | v1 `generation.py:675-692,734-742,764-771`; v2 `generation.py:702-779` | Add control arm conditioned on the natural outcome |
| Lichess data | Carve-from-train tail-holdout has no v2 implementation; `--pgn-val-split ""` → `"validation"` (`or` short-circuit) → single-file source leaks full dataset into val | regression | major | `scripts/train_jax_adapter.py:798`; `pawn/lichess_data.py:147-149`; v1 `train.py:425-441`; `split_has_files` gone | Restore disjoint-index carve + split probe |
| Data tools | `generate_model_cards.py` reads v1 config schema (`model_config`/`training_config`) → KeyError on v2 checkpoints; also hardcodes v1 repo IDs | regression | major | `scripts/generate_model_cards.py:259,270,138-139`; v2 `checkpoint.py:380-386` (`model`/`run`) | Migrate to v2 config keys + `-v2` repos; auto-detect from config.json |

\* Distillation `--resume`: tiebreak ruled this is not a v1-parity violation (distillation is net-new), but it IS a real internal-consistency/operability gap vs every other v2 trainer — worth fixing, info severity against the superset bar.

---

## 3. Confirmed partials (works but incomplete vs v1)

| Area | Issue | Status | Severity | Evidence (file:line) | Suggested fix |
|---|---|---|---|---|---|
| Eval gen | `impossible_task_test` lost corpus scenarios + control arm (single synthetic probe) | partial | minor | `pawn/generation.py:702-733`; v1 `generation.py:614-698` | Restore zero_remaining_ply/insufficient_material + control (capability reachable via primitives) |
| Adapters | `--lora-ffn`/`--sparse-ffn`/`--sparse-targets`/`--bottleneck-n-hidden` not registered as CLI flags (config-JSON-only) | partial | minor | `scripts/train_jax_adapter.py:479-491,540-555`; v1 generic `_parse_cli` | Register the flags; values reachable via `--config` today |
| Config/CLI | `amp_dtype` / `--prepend-outcome` / `max_seq_len` reachable only via `--config` JSON, not direct CLI flag (verbatim v1 JSON with `amp_dtype:'none'` rejected) | partial | minor | `run_config.py:156,242-269`; `train_jax.py:99-174` | Optional `_migrate_amp_dtype` + register flags |
| Lichess data | Adapter trainer samples WITH replacement; v1 used epoch-permuted WITHOUT replacement (`make_epoch_schedule` exists but unwired) | partial | minor | `train_jax_adapter.py:930`; `pawn/lichess_data.py:438-465` (no callers); plan §5:350 | Wire `make_epoch_schedule` into the adapter loop |
| Sweeps | RoSA/hybrid suggesters drop `lora_targets`; common hparams (batch_size/wd/warmup/patience) + FFN axes dropped from all suggesters | partial | minor | v1 `sweep.py:49-56,159`; v2 `sweep.py:48-143` | Re-add to search spaces |
| Sweeps | `suggest_rosa_retro_bottleneck` doesn't sweep `bottleneck_dim`; `suggest_architecture`/`suggest_pretrain` non-LR axes gone | partial | minor | `sweep.py:112-113`; v1 `sweep.py:180-188,120-152` | Re-add bottleneck_dim; arch search largely moot (fixed supernet) |
| RoSA | No test that Phase-3 retro-bottleneck → `BottleneckEffective`; RoSA phase-transition metrics (`rosa/phase`, `mask_density`) not logged | partial | minor | `tests/test_jax_adapters.py:670-687`; `train_jax_adapter.py:1022-1075` | Add Phase-3 type test + phase-boundary logging |
| RoSA | `bottleneck_n_hidden` not threaded into `RoSAConfig` (retro-bottleneck locked to n_hidden=0); mask-gen legal-conditioning dropped | partial | minor | `pawn/adapters/rosa.py:75-85`; v1 `adapter_training.py:1322` | Add field + constructor arg |
| Eval | `--min-eval-ply` (MAIA opening-skip) and `--elo-min/--elo-max` single-band adapter filter absent | partial | minor | v1 `eval_accuracy.py:60-79`; v2 eval scripts | Add flags |
| Probes | piece_type probe single-square + color-agnostic (7-class) vs v1 all-64×13-class; no best-epoch tracking; intra-game val leakage (positions split, not games); 16× smaller default corpus | partial | minor | `pawn/probes.py:107-112,215-234`; `scripts/eval_probes_jax.py:41,49` | Add all-squares/color mode; disjoint-game val |
| Probes | Output schema diverged from v1 (`layers` int-keyed, no run/step/variant/model_config/loss/best_accuracy); criterion 10 "same schema" unmet; `--log-dir`/`--top-layer-only`/`--prepend-outcome` flags gone | partial/regression | major | `scripts/eval_probes_jax.py:82-107,38-56`; v1 `eval_probes.py`; plan:59 | Restore schema + log-dir scan |
| Sweeps | `scripts/sweep.py` hardcodes `--no-pgn`; no multi-GPU trial pinning | partial | minor | `scripts/sweep.py:39-52`; v1 `n_gpus`/`CUDA_VISIBLE_DEVICES` | Expose `--pgn`; JAX-appropriate device pinning |
| Metrics | v2 emits `loss` key; pawn-run dashboard expects `train/loss`/`train/accuracy` → blank loss chart/KPI | resolved | major | `scripts/train_jax.py` (emits `train/loss`+`train/accuracy` via in-scan `top1_accuracy` on the supernet variant); `dashboard/charts.py:344,373`; `tests/test_jax_logging.py::test_pretrain_emits_train_loss_dashboard_key`; `tests/test_jax_trainer.py::test_top1_accuracy_*` | DONE — both keys emitted with bare aliases; supernet top-1 = v1 parity (`main:pawn/trainer.py:1145`) |
| Metrics | `load_metrics` API change (single-arg `MetricsBundle`) breaks `sol.py:1349` two-arg call; per-run notes / HF-sync / trial-grouping helpers deleted | regression | major | `pawn/dashboard/metrics.py:41`; `sol.py:1349`; v1 `metrics.py:113,145,186` | Fix call site; restore or relocate notes/sync/trials |
| Metrics | wandb `init_wandb` narrowed (no `job_type`/group/`run_dir` name; no `finish_wandb(exit_code)`); per-step grad_norm gated behind `--emit-grad-norms` | partial | minor | `pawn/wandb_utils.py:61-132`; `train_jax.py:660-663` | Restore lost wandb knobs if needed |
| Model | No unit test exercises `__call__` with `compute_dtype=bfloat16` (activation/logit dtype); no train-step-level bf16 test | partial | minor | `tests/test_jax_model.py` (no bf16); `tests/test_jax_trainer.py` (no compute_dtype) | Add bf16 forward + train-step dtype assertions |
| Model | `__call__` docstring falsely says logits always fp32 (bf16 path returns bf16; trainer compensates); `model.py:999` cites nonexistent `test_jax_kv_cache.py` | partial | info | `pawn/model.py:537-539,631-633,999` | Fix docstrings |
| Checkpoint | SIGTERM/HF-push covered only by unit + manual smoke, no subprocess kill-and-check test; live HF push not in CI; `best_val_loss`/best-checkpoint persistence dropped | partial/needs_runtime | minor | `tests/test_jax_lifecycle.py:414-449`; v1 `checkpoint.py:671`,`trainer.py:1320` | Add subprocess SIGTERM test; restore best-checkpoint sidecar |
| Pretrain | `pause_after_steps` dead config; padded-batch no-drift test passes for wrong reason (LR=0 at step 0); one_cycle ramp linear vs v1 cosine | partial/regression | minor | `run_config.py:126`; `tests/test_jax_trainer.py:561-594`; `trainer.py:541-551` | Fix test; document ramp change |
| Pretrain | Stochastic sandwich-sampling (default mode) has no unbiased-expectation test | partial | major | `pawn/trainer.py:392-426`; `run_config.py:197`; tests use `stochastic_variants=False` | Add E[loss_stoch]==sum test (see §7) |
| Engine | 5 lower-level token-init sites still zero-init (`batch.rs:45,99`; `pgn.rs:618`; `uci.rs:179`; `diagnostic.rs:148`); `test_pad_after_game_end` has contradictory assertions; `test_token_buffer_pad_init_contract` is a tautology | partial/regression | minor | `engine/src/batch.rs:45,386-407`; `lib.rs:1749-1765`; masked by Python `_pack_clm:442` | PAD-init all sites; fix the two tests |
| Bench | `benchmark.py` adapter section always fp32 (ignores `--amp-dtype`); drops 2 engine calls + GPU hardware metadata; can't bench flash attention | partial | minor | `scripts/benchmark.py:921-928,1155-1175,211-274` | Thread `compute_dtype`; restore engine calls |

---

## 4. Design-superseded (intentional v1 removals — justification check)

Each confirmed via §6/CLAUDE with a quotable rationale and is internally consistent **except where flagged**:

| Removed v1 item | v2 replacement | Justification | Consistent? |
|---|---|---|---|
| **cotrain** (`cotrain.py`, `CotrainConfig`) | Supernet joint loss (`trainer.py:324-426`) | plan §6:154 "the supernet's multi-variant joint loss is what cotrain provided. Cotrain is replaced by pretrain the supernet"; v2_redesign §7 (RAM analysis); CLAUDE.md | ✅ Confirmed absent; tests pin rejection |
| **`pawn/gpu.py`** (torch device init) | `jax_setup.require_accelerator` + `PAWN_ALLOW_CPU=1` | plan §6; JAX manages its own backend | ✅ Only the escape hatch kept, as specified |
| **`pawn/data.py` / `data_utils.py`** | `corpus.py` + `lichess_data.py` | plan §6:158 | ✅ |
| **`bucket_size`** (dynamic padding) | `Corpus.by_bucket()` static pre-bucketing | plan §6:161 "the JAX trainer is shape-static" | ✅ (note: `--hf-bucket` is a *different* concept and is NOT design-superseded — it's the broken blocker in §2) |
| **Legacy v1→JAX converter** (`pawn/legacy.py`) | v1.0.0 git tag | H.2 commit `1af5737` body + `checkpoint.py:613-617` | ⚠️ **DOC-INCONSISTENT** (see below) |
| **Torch-only fields** (device/num_workers/no_compile/sdpa_math) | dropped | plan §6; `extra='forbid'` | ✅ Test pins absence |
| **`variant='toy'/'custom'`** | `supernet='tiny'` / `SpecializedCLMConfig` | plan §3:311 | ✅ (verbatim-v1-JSON loading breaks — minor compat nit) |
| **Stockfish self-play harness** | Maia-style per-Elo accuracy (criterion 13) | plan §13:494 "Elo-stratified Maia-style accuracy" | ✅ Tiebreak: §13 deliverable documents the repurposing (the gate's "undocumented" concern was overturned) |

**Converter inconsistency (criteria 4/5/20) — flagged per the brief:** The converter was built, verified working (`pawn-small` mean Δ=5.94e-6, argmax 100%), then removed in H.2. Removal is a defensible design decision (Phase-A `VOCAB_SIZE 1980→2000` + un-factored embeddings make v1 checkpoints architecture-incompatible anyway). **But the retirement is recorded inconsistently:**
- `DEFERRALS.md` "Gone-by-design" and plan §6 GONE-BY-DESIGN list **do not** mention the converter or criteria 4/5/20 retirement (violates plan §9.3 "if the design doc doesn't list it, you don't get to delete it").
- Plan §6:143 still lists `pawn/legacy.py` as IN-SCOPE and §7:177 still says "Don't ship without it."
- `final_smoke.md` criteria 4/5/20 still cite `pawn.legacy.convert_legacy_checkpoint` as runnable (last touched pre-H.2); `final_smoke.md:628` also falsely says "DEFERRALS.md … is empty."
- Stale present-tense converter references in `README.md:58`, `docs/{ADAPTERS,ARCHITECTURE,TRAINING,ACCURACY_CEILING}.md:7`, `CLAUDE.md:445-447` (claims torch touchpoints exist), and inline comments in `model.py:27,37,232,288`, `trainer.py:256`, `_sentinel.py:11`, `scripts/train_jax.py:295`.
- `CLAUDE.md:112,451-453` still describe **factored embeddings** that Phase A un-factored.

**Recommended fix:** add a DEFERRALS.md "Gone-by-design" entry for the converter + criteria 4/5/20; amend plan §6/§7; sweep the stale doc/comment references.

---

## 5. Superset improvements (v2 adds beyond v1)

- **Distillation trainer** (`pawn/distill.py` + `train_jax_distill.py` + `DistillConfig`): injectable frozen teacher (`stop_gradient`), pluggable CE/KL/mix objective with T²-rescaled KL, KL masked to `[0, PAD_TOKEN)`, C-mismatch guard, K-step scan. Six spec-mandated unit gates all behavioral; end-to-end smoke verifies teacher byte-immutability.
- **Conditioning prefix** `[BOS][cond...]` generalizes v1's single `prepend_outcome`, with migration validator and extensible `CONDITIONING_KINDS`; MASK_VERSION + C-width enforced at load/resume.
- **KV-cached decoder** (`PAWNModel.forward_with_cache` + `KVCache` eqx.Module + OOB `eqx.error_if`); bit-identical-to-full-forward parity tests; auto-detected in `autoregressive_generate`.
- **Pallas flash attention** default on GPU (~6× over plain QK^T on RDNA3); three-way attention dispatch (flash/sdpa/plain).
- **Supernet nesting**: `sliced()` `[:d_V,:d_V]` extraction, fixed `HEAD_DIM=64`, `validate_nested` invariant + import-time checks; `--variants` subset training.
- **Precision improvements**: RMSNorm weight-multiply-in-fp32-then-downcast; RoPE rotates in compute dtype; bf16 AdamW first-moment (`mu_dtype=bfloat16`, half optimizer-state bandwidth) with round-trip test; activation remat (`PAWN_USE_REMAT`).
- **Checkpoint hardening**: stdlib-only `_sentinel.py` (jax-free import, subprocess-tested), sentinel verified on **every** load incl. config-only, immutable `FileExistsError` policy, daemon-thread push executor with stuck-upload abandon.
- **Two-tier adapter freeze** via `eqx.filter_value_and_grad` (backbone grad DCE, ~33% backward FLOP cut); unfreeze decay-drift snap-back; `_chunk_bound` boundary-safe scan; RoSA CPU top-k (ROCm workaround).
- **New harnesses**: supernet parity harness (`pawn/parity.py` + `eval_parity.py`), `scripts/bench/run.py` K-step scan matrix + cost(T) curve, `scripts/sweep.py` CLI, `scripts/sweep_pretrain_lr.py`, lab `dry_run` + `distill` run_type + `read/audit_schedule_health`.
- **Cache key superset** (9 fields incl. seq_len/conditioning/mask_version, preventing v1 cross-run collisions); `PAWN_ALLOW_BULK_DOWNLOAD` gate (v1 silently downloaded 262 GB); split-aware validation default; sweep `_read_best_val_loss` fixes v1's `0.0`-falls-through `or`-bug; reserved-kwarg defense in MetricsLogger.

---

## 6. Prior-audit (JAX_PARITY_SHORTFALLS.md) corrections

The prior audit's 16 claims, re-adjudicated against the current tree (most likely to mislead a reader):

| # | Prior claim | Current status | Evidence |
|---|---|---|---|
| §1 | bf16 AMP forward cast missing | **FIXED** — wired end-to-end (`amp_dtype` default bfloat16 → `compute_dtype`, cast before CE, `mu_dtype=bf16`) | `model.py:589-590`, `trainer.py:240,314,714`, `train_jax.py:293-302` |
| §3 | `__init__.py` docstring-only / v1 names not redirected | **STALE/FALSE** — PEP 562 `__getattr__` raises precise ImportError; tested | `pawn/__init__.py:54-93`, `test_public_api.py:46-55` |
| §6 | RoSA "mostly nominal", no three-phase schedule | **FALSE** — full 3-phase (warmup/mask-gen/joint), all 3 modes distinct, Algorithm-1 mask gen, behaviorally tested | `adapter_trainer.py:401,629`; `train_jax_adapter.py:1022-1075` |
| (adapters) | Skeletal FiLM / missing attn-bottleneck / hybrid drops targets-ffn / sparse-FFN inert / unfreeze drift / Python loop not K-scan | **ALL FIXED** — real residual-stream FiLM, both bottleneck placements, hybrid forwards LoRAConfig, sparse FFN wired, unfreeze snap-back, K-step `lax.scan` | `adapters/film.py:147-162`, `bottleneck.py:60-90`, `train_jax_adapter.py:402-410`, `sparse.py:40-43`, `adapter_trainer.py:197-228,305-331` |
| §10 | KV-cached decoder is a gap | **FIXED** — real eqx.Module cache, bit-identical parity tests; `DEFERRALS.md` marks Resolved | `model.py:390-445,964`; `tests/test_jax_eval.py:262-284` |
| §11 | Edge-case six-category gap; `generate_random_games` instead of quotas; terminal-label off-by-one | **FIXED** — all 10 categories restored, `generate_diagnostic_sets` quotas, H5 alignment fixed with discriminating tests | `diagnostics.py:56-67,288-329,228-232`; `tests/test_jax_eval.py:802-913` |
| §13 | GPU memory key mismatch | **FIXED/RESOLVED** — emits v1-parity `mem/gpu_{peak,reserved,current}_gb` + new smi fields | `logging.py:344-346,555-566`; `charts.py:409-411` |
| §12 | eval on random games only | **PARTIALLY STALE** — criterion-9 random-game baseline is correct (v1 8.57% IS a random-game val number, `ACCURACY_CEILING.md:78-83`); the *residual* (no adapter-on-Lichess accuracy) is REAL and is a §2 blocker, but not for the reason the prior audit gave | `eval_jax.py:44-47`; `final_smoke.md:283-301` |
| §14 | criterion-7 not run on real `pawn-lichess-full` HF repo | **STALE as a parity gap** — capability fully present (`_scan_parquet` hf:// path, defaults to `pawn-lichess-full`, bulk gate), unit-tested via mocks; real-network e2e unproven but v1 never proved it either | `lichess_data.py:130-202`; `train_jax_adapter.py:787-804` |

**Net:** the prior audit is now substantially stale — the headline gaps (bf16, RoSA, KV-cache, edge-case categories, GPU-mem keys, FiLM/sparse/hybrid/unfreeze) are all fixed. A reader trusting JAX_PARITY_SHORTFALLS.md would chase resolved issues and miss the *current* blockers (lab daemon, hf_bucket, dashboard import, adapter-eval, pretrain-val), none of which it flagged.

---

## 7. Needs-runtime checklist (run serially on the GPU)

Deduplicated, ordered by value. All use `--extra rocm`; the GPU resolves to `RocmDevice(id=0)`.

1. **Criterion 3 — full test suite** (catches the dashboard import crash via a new test you should add; also covers stochastic-path gap):
   `uv run --extra rocm pytest tests/ -m "not gpu"` then `... -m gpu`
2. **Stochastic sandwich unbiasedness (criterion 6 correctness, default mode untested):**
   `uv run --extra rocm python -c "<200-draw E[loss_stochastic] vs sum-of-variants, expect ratio≈1>"` (verifier command in `stochastic-variant-sampling-untested`)
3. **Criterion 6 — tiny pretrain loss decrease:**
   `PAWN_ALLOW_CPU=1 uv run --extra rocm python scripts/train_jax.py --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 --k 50 --local-checkpoints --logs-dir /tmp/c6 2>&1 | tail -30`
4. **Criterion 7 — real Lichess LoRA on `pawn-lichess-full`:**
   `uv run --extra rocm python scripts/train_jax_adapter.py --strategy lora --supernet tiny --variant base --lora-rank 4 --total-steps 200 --pgn thomas-schweich/pawn-lichess-full --elo-min 1800 --elo-max 2000 --local-checkpoints --logs-dir /tmp/v2_lora_hf`
5. **Criterion 9 — accuracy vs v1 baseline (same random-game distribution, ±0.5pp):**
   `uv run --extra rocm python scripts/eval_jax.py --checkpoint <converted-base> --n-games 512 --batch-size 8` → compare overall to v1 8.57%
6. **Criteria 10/11/12/13 — probes / generation / edge-case / elo runs:**
   `eval_probes_jax.py --checkpoint <ckpt> --n-games 64 --max-ply 20 --n-epochs 5`; `eval_generation_jax.py --checkpoint <ckpt> --outcome-prefix-trained --edge-cases`; multi-bin `eval_vs_stockfish.py --checkpoint <ckpt> --pgn thomas-schweich/pawn-lichess-full` (criterion 13 never run on the real multi-bin repo)
7. **Criterion 14 — 3-trial LoRA sweep:**
   `uv run --extra rocm python scripts/sweep.py --strategy lora --n-trials 3 --supernet tiny --variant base --storage sqlite:////tmp/lora.db --logs-dir /tmp/sweep --total-steps 25`
8. **Criteria 16/17 — resume + SIGTERM (subprocess kill-and-check, no automated test today):**
   `... train_jax.py --supernet tiny --total-steps 100000 ... & PID=$!; sleep 60; kill -TERM $PID; wait $PID; echo $?; ls .../step_*/` then `--resume` the saved step and check metric monotonicity
9. **Criterion 18 — live HF push (not in CI; verify `run/{slug}` branch once §2 fix lands):**
   `... --hf-repo thomas-schweich/scratch-test-v2 ...` then list repo siblings/branches
10. **Perf benchmark — Pallas vs plain + cost(T):**
    `uv run --extra rocm python scripts/bench/run.py --tiny --k 10 --warmup-outers 1 --timed-outers 5 --label pallas-verify`
11. **`hf_bucket` blocker reproduction** (confirm the silent no-save): launch a tiny run with ONLY `--hf-bucket <url>` and verify nothing is written/pushed.

---

## 8. Open contests (unresolved after tiebreak)

None remain genuinely unresolved — every contested ID received a tiebreak ruling. The rulings that **moved severity against the gate/verifier** (flagged here so a reader knows where judgment was applied):

- **`hf_bucket` dead path**: verifier missed it; gate found it (major); tiebreak **escalated to blocker** (pretrain loses all work).
- **Lab `TrialRunner` / MCP tools / tests**: gate said major; tiebreak **escalated to blocker** ×3 (entire post-launch orchestration impossible).
- **`lab-sweep-module-missing`**: tiebreak **escalated minor→major** (false-equivalence with `pawn/sweep.py`).
- **`eval-accuracy-wrapper`**: tiebreak **de-escalated blocker→major** (folding adapters partially reachable) but surfaced a NEW silent-correctness hole (bottleneck/FiLM eval bare backbone).
- **`pretrain-no-inline-validation-loop`**: confirmed gap/major; tiebreak broadened to "**no backbone-pretrain validation loop at all**" (root cause of multiple metric gaps + model-card KeyErrors).
- **Distillation `--resume`**: gate said gap/major; tiebreak **ruled not a v1-parity gap (info)** since distillation is net-new — but it remains a real operability shortfall vs other v2 trainers.
- **`elo-stockfish-gameplay-removed`**: gate said partial/minor; tiebreak **ruled design_superseded/info** (§13:494 documents it).
- **Numeric/runtime criteria (6 loss-decrease, 9 accuracy, 7 real-HF)**: tiebreak ruled these **parity/info** (capability present, baseline-comparison premises corrected) — they remain in §7 as verification-rigor items, not gaps.

The one item a reviewer might still debate is whether the generation-diagnostic scope reductions (prefix/poisoned/impossible/improbable control arms) are collectively one redesign or several independent regressions — tiebreak confirmed each as REAL/major individually; they share a root cause (corpus-driven multi-scenario harness not ported) and could be closed by one fix.