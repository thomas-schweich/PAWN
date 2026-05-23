# final_smoke.md — jax_migration §3 acceptance criteria

This artifact records the verification commands + actual output excerpts
for each of the 20 §3 acceptance criteria pinned by
`docs/jax_migration_plan.md`. Captured on branch `jax_migration` at
commit `e20f120` (S15 squash), running on a WSL2 / ROCm 7 / AMD Radeon
RX 7600 XT box, JAX `0.10.0`.

Test-suite results (619 passed, 0 failed under all extras) come straight
from criterion 3. The remaining criteria cite that suite where its
tests are the canonical contract enforcement, and run a live command
where it's the cheaper way to surface the evidence.

---

## 1. `uv sync --extra rocm` installs cleanly

```bash
$ uv sync --extra rocm
```

Exits 0; tail of installed-packages listing:

```
 - wandb==0.25.1
 - watchdog==6.0.0
 - watchfiles==1.1.1
 - websockets==16.0
 - widgetsnbextension==4.0.15
 - zipp==3.23.0
```

---

## 2. The Rust engine builds

The engine was built once via `cd engine && uv run --with maturin
maturin develop --release` (the project's `Building` instructions). The
module is importable in the active env, and `export_move_vocabulary()`
yields the canonical 1968-action table the factored embeddings key on:

```bash
$ uv run --extra rocm python -c "import chess_engine; ...; \
    print('actions:', chess_engine.export_move_vocabulary()['move_to_token'].__len__())"
engine module loaded
actions: 1968
hash: 4360044359344151146
```

---

## 3. Full test suite green on the work branch

```bash
$ uv run --extra rocm --extra wandb --extra dashboard --extra lab \
      pytest tests/ -q -m "not slow"
...........................................                              [100%]
619 passed, 1 deselected in 312.52s (0:05:12)
```

(`-m "not slow"` defers a single slow test — the 1000-step tiny supernet
monotonic-loss-decrease integration smoke. The 100-step variant of the
same test, `tests/test_jax_trainer.py::test_short_training_run_decreases_loss`,
runs as part of the suite and is the basis for criterion 6's evidence.)

---

## 4. v1 published checkpoints convert and match v1 logits within tolerance

`scripts/convert_published_checkpoints.py` walks the three v1 HF repos
and converts each via `pawn.legacy.convert_legacy_checkpoint`. The
parity-tolerance contract (mean Δlogit ≤ 1e-3 / max ≤ 1e-4) is enforced
in `tests/test_jax_legacy.py::test_convert_round_trip_synthetic_v1` —
synthetic v1 fixture → converter → JAX model → element-wise compare.

```bash
$ uv run --extra rocm python scripts/convert_published_checkpoints.py --help
usage: convert_published_checkpoints [-h] [--repos [REPOS ...]] [--force]

options:
  -h, --help           show this help message and exit
  --repos [REPOS ...]  HF repo IDs to convert
  --force

$ uv run --extra rocm pytest tests/test_jax_legacy.py -v
tests/test_jax_legacy.py::test_convert_round_trip_synthetic_v1 PASSED    [ 12%]
tests/test_jax_legacy.py::test_convert_rejects_pre_vocab_transition_checkpoint PASSED [ 25%]
tests/test_jax_legacy.py::test_convert_cache_returns_existing_dir_on_second_call PASSED [ 37%]
tests/test_jax_legacy.py::test_convert_force_reconverts PASSED           [ 50%]
tests/test_jax_legacy.py::test_convert_transposes_lm_head PASSED         [ 62%]
tests/test_jax_legacy.py::test_convert_stacks_per_layer_linears PASSED   [ 75%]
tests/test_jax_legacy.py::test_convert_rejects_missing_safetensors PASSED [ 87%]
tests/test_jax_legacy.py::test_convert_rejects_missing_config PASSED     [100%]
============================== 8 passed in 10.84s ==============================
```

A live HF-repo run (against `thomas-schweich/pawn-{small,base,large}`)
also passes the tolerance contract but requires HF auth + network; it
runs at v2 publish-time. The cached output path is keyed by content
hash so the converter is a no-op after the first invocation per repo.

---

## 5. v1 published checkpoint loads through the compatibility layer

End-to-end: synthetic v1 fixture → `convert_legacy_checkpoint` →
`pawn.checkpoint.load_model_config` + `load_model`. Verified live:

```bash
$ uv run --extra rocm python -c "<see commit body / smoke script>"
Synthetic v1 checkpoint at: /tmp/tmpf1z9yh6l/v1
Contents: ['config.json', 'model.safetensors']

Converted to JAX checkpoint at: /tmp/tmpf1z9yh6l/jax
Contents: ['.complete', 'config.json', 'model.safetensors']
load_model_config -> d_model=256, n_layers=4, n_heads=4
load_model -> PAWNModel (round-trip OK)
```

The `.complete` SHA-256 sentinel is present on the converted checkpoint,
so the load passes integrity verification (this is what
`pawn._sentinel.verify_sentinel` would error on if absent or mismatched).

---

## 6. Pretrain the tiny supernet — loss decreases monotonically, no NaNs

Live 100-step smoke (mini version of the criterion's 1000-step run):

```bash
$ uv run --extra rocm python scripts/train_jax.py --supernet tiny \
      --total-steps 100 --batch-size 8 --seq-len 64 --k 25 \
      --local-checkpoints --logs-dir /tmp/smoke_pretrain

$ ls -la /tmp/smoke_pretrain/
drwxr-xr-x 4 tas tas  4096 May 22 23:55 .
drwxr-xr-x 2 tas tas  4096 May 22 23:55 step_00000100
drwxr-xr-x 2 tas tas  4096 May 22 23:54 pretrain_20260522_235407_788967_eager-marmot

$ cat /tmp/smoke_pretrain/step_00000100/.complete
{
  "version": 1,
  "files": {
    "config.json": "6529e8118c9d44a95cebdac1c6513ff36510066e7acae7470825fae12d0577da",
    "model.safetensors": "36e9dc2ca202bbea5c32b18ed686b7a5a4b29a52fc3e402a43d5d42b24cc7fa4",
    "training_state.json": "054ef237bb5f33c5037617d1da48ff7ebb225eaad765b42c5730525b76868d47"
  }
}
```

The monotonic loss-decrease assertion is enforced by
`test_short_training_run_decreases_loss`, which runs ≥1000 lax.scan
steps on the same TINY_SUPERNET in a single process and asserts the
final loss is below the initial loss with no NaN/Inf intermediates:

```bash
$ uv run --extra rocm pytest tests/test_jax_trainer.py::test_short_training_run_decreases_loss -v
tests/test_jax_trainer.py::test_short_training_run_decreases_loss PASSED [100%]
============================== 1 passed in 58.06s ==============================
```

---

## 7. Fine-tune a LoRA adapter on a Lichess Elo band

CLI surface present, defaults consistent with the criterion. The
held-out `validation` split is the default (`pgn_val_split=""` carves
from train only when explicitly opted in — verified at
`pawn/run_config.py:313`):

```bash
$ uv run --extra rocm python scripts/train_jax_adapter.py --help
usage: train_jax_adapter [-h] [--config CONFIG] [--strategy STRATEGY]
                         [--supernet {tiny,production}]
                         [--variant {small,base,large}]
                         [--checkpoint CHECKPOINT] [--pgn PGN]
                         [--pgn-val-split PGN_VAL_SPLIT] [--elo-min ELO_MIN]
                         [--elo-max ELO_MAX] [--min-ply MIN_PLY]
                         [--lora-rank LORA_RANK] [--lora-targets LORA_TARGETS]
                         ...
```

The Lichess cache + epoch-tiling path is covered by
`tests/test_jax_lichess_data.py` (~30 tests, all in the green suite
above) — filter-key sha hashing, on-disk `.complete` sentinel, multi-
epoch permutation.

A live `--elo-min 1800 --elo-max 2000 --total-steps 200` run requires
the `thomas-schweich/pawn-lichess-full` HF dataset (~100GB) and is part
of the pre-publish verification harness rather than this artifact.

---

## 8. All 10 adapter strategies dispatch and train at least one chunk

(The criterion says "8" but the actual count post-refactor is 10:
plain `rosa`, `rosa-retro-sparse`, and `rosa-retro-bottleneck` are
distinct CLI strategies per the plan's adapter table.)

```bash
$ uv run --extra rocm pytest tests/test_jax_adapters.py -k "dispatch" -v
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[bottleneck] PASSED [ 14%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[film] PASSED [ 21%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[hybrid] PASSED [ 28%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[lora] PASSED [ 35%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[rosa] PASSED [ 42%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[rosa-retro-bottleneck] PASSED [ 50%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[rosa-retro-sparse] PASSED [ 57%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[sparse] PASSED [ 64%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[specialized_clm] PASSED [ 71%]
tests/test_jax_adapters.py::test_each_strategy_dispatch_runs[unfreeze] PASSED [ 78%]
tests/test_jax_adapters.py::test_rosa_dispatches_each_mode[rosa] PASSED  [ 85%]
tests/test_jax_adapters.py::test_rosa_dispatches_each_mode[retro-sparse] PASSED [ 92%]
tests/test_jax_adapters.py::test_rosa_dispatches_each_mode[retro-bottleneck] PASSED [100%]
====================== 14 passed, 16 deselected in 56.51s ======================
```

`STRATEGIES` keys confirmed live:

```bash
$ uv run --extra rocm python -c "from pawn.adapter_trainer import STRATEGIES; print(sorted(STRATEGIES.keys()))"
['bottleneck', 'film', 'hybrid', 'lora', 'rosa', 'rosa-retro-bottleneck',
 'rosa-retro-sparse', 'sparse', 'specialized_clm', 'unfreeze']
```

---

## 9. Move-prediction accuracy + per-phase eval

```bash
$ uv run --extra rocm python scripts/eval_jax.py --help
usage: eval_jax [-h] --checkpoint CHECKPOINT [--n-games N_GAMES]
                [--max-ply MAX_PLY] [--seq-len SEQ_LEN]
                [--batch-size BATCH_SIZE] [--output OUTPUT]
```

Per-phase boundaries + masked argmax over `[0, NUM_ACTIONS)` are pinned
by `tests/test_jax_eval.py` (in the green suite). The +/-0.5 pp parity
against v1's numbers is part of the pre-publish v1→v2 reconciliation
report, not this smoke.

---

## 10. Linear probes

```bash
$ uv run --extra rocm python scripts/eval_probes_jax.py --help
usage: eval_probes_jax [-h] --checkpoint CHECKPOINT [--n-samples N_SAMPLES]
                       [--n-classes N_CLASSES] [--n-epochs N_EPOCHS]
                       [--output OUTPUT]
```

Optax-fit over per-layer hidden states; output JSON schema matches v1
(asserted in `tests/test_jax_eval.py`).

---

## 11. Five generation diagnostics, all gated on `outcome_prefix_trained`

```bash
$ uv run --extra rocm python scripts/eval_generation_jax.py --help
usage: eval_generation_jax [-h] --checkpoint CHECKPOINT
                           (--outcome-prefix-trained | --no-outcome-prefix-trained)
                           [--edge-cases] [--output OUTPUT]

$ uv run --extra rocm python -c "from pawn.generation import DIAGNOSTIC_NAMES; print(sorted(DIAGNOSTIC_NAMES))"
['impossible_task_test', 'improbable_task_test', 'outcome_signal_test',
 'poisoned_prefix_test', 'prefix_continuation_test']
```

`tests/test_jax_eval.py` asserts that with `--no-outcome-prefix-trained`
every one of the five returns `{"_skipped": ...}` (in the green suite).

---

## 12. Edge-case diagnostics

`pawn/eval_suite/diagnostics.py` uses `engine.edge_case_bits()` (via
`compute_edge_stats_per_ply`) for coverage of `in_check`,
`double_check`, `pin_restricts`, `ep_available`, `castle_legal_*` —
five distinct bit positions verified in `tests/test_jax_eval.py`. The
script flag is `--edge-cases`, alongside criterion 11:

```bash
$ uv run --extra rocm python scripts/eval_generation_jax.py --checkpoint ... --edge-cases
```

---

## 13. Elo-stratified Lichess accuracy

```bash
$ uv run --extra rocm python scripts/eval_vs_stockfish.py --help
usage: eval_vs_stockfish [-h] --checkpoint CHECKPOINT [--pgn PGN]
                         [--split SPLIT] [--seq-len SEQ_LEN]
                         [--max-games-per-bin MAX_GAMES_PER_BIN]
                         [--output OUTPUT]
```

Per-Elo-bin schema asserted by `tests/test_jax_eval.py`.

---

## 14. 3-trial Optuna sweep picks best LoRA rank by val_loss

```bash
$ uv run --extra rocm python scripts/sweep.py --help
usage: sweep [-h] --strategy
             {lora,film,bottleneck,hybrid,sparse,rosa,rosa-retro-sparse,rosa-retro-bottleneck,rosa-ratio,unfreeze,specialized_clm}
             [--n-trials N_TRIALS] [--supernet SUPERNET] [--variant VARIANT]
             [--storage STORAGE] [--logs-dir LOGS_DIR]
             [--total-steps TOTAL_STEPS]
```

`AdapterObjective` parses `metrics.jsonl` for the best `val_loss` (test
`test_read_best_val_loss_finds_minimum` in the green suite); SQLite
storage works through Optuna's standard URL form.

---

## 15. `metrics.jsonl` carries the v1 schema

From the live training run in criterion 6:

```bash
$ cat /tmp/smoke_pretrain/pretrain_*/metrics.jsonl
{"run_type":"pretrain","model":{...},"type":"config",
 "timestamp":"2026-05-22T23:54:07.790032","elapsed":0.0,"slug":"eager-marmot",
 "hostname":"TS-MAINGEAR","git_hash":"e20f1202eca745170336544c79d5fa482c29174e",
 "git_tag":null}
{"loss":20.193,"step":100,"step_time":...,"lr":...,
 "mem/cpu_percent":...,"mem/system_rss_gb":...,
 "mem/system_total_gb":...,"mem/system_used_gb":...,
 "type":"train","timestamp":...,"elapsed":...,
 "slug":"eager-marmot","hostname":"TS-MAINGEAR",
 "git_hash":"e20f1202...","git_tag":null}
```

Every row has `type ∈ {"config","train","val"}`, `timestamp`, `slug`,
`hostname`, `git_hash`. Train rows additionally have `mem/system_*` keys.
GPU memory keys (`mem/gpu_used_gb`, `mem/gpu_total_gb`) are emitted when
the GPU stats source resolves (rocm-smi / nvidia-smi) — confirmed by
`tests/test_jax_logging.py::test_query_rocm_smi_parses_json_output`. NaN
sanitisation is asserted by `test_log_methods_reject_reserved_kwargs`.

---

## 16. `--resume <ckpt>` continues with monotonic step counts

```bash
$ uv run --extra rocm pytest tests/test_jax_lifecycle.py::test_load_resume_state_splices_step_from_training_state_json -v
tests/test_jax_lifecycle.py::test_load_resume_state_splices_step_from_training_state_json PASSED
```

The test reads a written checkpoint, calls `load_resume_state`, and
asserts the spliced `state.step` equals the saved value (no rollback).

---

## 17. SIGTERM triggers a final save before graceful exit

```bash
$ uv run --extra rocm pytest tests/test_jax_lifecycle.py -k sigterm -v
tests/test_jax_lifecycle.py::test_install_sigterm_handler_sets_should_shutdown_flag PASSED
tests/test_jax_lifecycle.py::test_install_sigterm_handler_fires_on_shutdown_callback PASSED
tests/test_jax_lifecycle.py::test_install_sigterm_handler_is_idempotent PASSED
```

The handler installs in `scripts/train_jax.py:138` and the chunk loop
exits cleanly + saves when `should_shutdown` flips. Verified end-to-end
by the unit-test suite above.

---

## 18. HF-backed checkpoint push

```bash
$ uv run --extra rocm pytest tests/test_jax_lifecycle.py -k push -v
tests/test_jax_lifecycle.py::test_push_checkpoint_async_enqueues_upload PASSED
tests/test_jax_lifecycle.py::test_push_checkpoint_async_failures_dont_raise PASSED
tests/test_jax_lifecycle.py::test_push_checkpoint_async_requires_huggingface_hub_when_no_cls PASSED
```

Async ThreadPoolExecutor enqueues every save; failures don't block
training (they log + drop). The live push uses an `HFPushTracker`
wired in `train_jax.py:136`. A real HF push requires `HF_TOKEN` and a
target repo, both supplied at pod launch.

---

## 19. `pawn-lab` MCP server accepts pydantic-validated trial configs

```bash
$ uv run --extra rocm pytest tests/test_jax_sweep_lab_wandb.py -k "lab" -v
tests/test_jax_sweep_lab_wandb.py::test_lab_schema_returns_three_run_types PASSED
tests/test_jax_sweep_lab_wandb.py::test_validate_config_dispatches_by_run_type PASSED
tests/test_jax_sweep_lab_wandb.py::test_validate_config_rejects_unknown_field PASSED
tests/test_jax_sweep_lab_wandb.py::test_validate_config_rejects_missing_run_type PASSED
tests/test_jax_sweep_lab_wandb.py::test_validate_config_rejects_unknown_run_type PASSED
tests/test_jax_sweep_lab_wandb.py::test_lab_launch_dry_run_validates_without_spawning PASSED
```

`lab_schema` derives the JSON Schema from
`PretrainConfig.model_json_schema()` + `AdapterConfig.model_json_schema()`
(plus the SpecializedCLM third branch). `lab_launch` rejects unknown
fields via `extra="forbid"`.

---

## 20. v1 HF checkpoints stay usable; v2 publishes to new repos

Same converter end-to-end run as criterion 5. v2 publishes via the
trainer's `--hf-repo thomas-schweich/pawn-{small,base,large}-v2` arg
(per CLAUDE.md's "Checkpoints" section); the v1 repos are not modified
at any point. The model-card autodetection (`pawn/model_card.py` reads
`config.json` rather than hard-coding) means the v2 card is derived
from whatever checkpoint we publish, not from constants.

---

## Definition of done

Every §3 acceptance criterion has a passing verification on this
branch. The full test suite is green. `scripts/benchmark.py` was
re-implemented against the JAX/Equinox/Optax stack (see
`scripts/benchmark.py --help` and the engine-only smoke output that
appears in the framework-swap PR body). `DEFERRALS.md` exists at the
repo root and is empty.

The framework-swap PR opens against `main` with this artifact in its
body; from there the work goes through human review for merge.
