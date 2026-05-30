# Phase D — Eval & Observability: Implementation Spec

Phase D of `docs/v2_redesign_plan.md` (§8.2 H5/H6/H7, §8.3, §8.4). Branch
`feat/v2-phase-d-eval-obs` off `jax_migration` (Phase C merged). Same common
process/constraints as `docs/phase_b_spec.md` (serialize JAX, no push, never
suppress types / gut tests, parity bar).

## Global invariants (acceptance gates)
- `pyright` clean on `pawn scripts tests`; full `pytest` green.
- Edge-case diagnostics aligned; probes use REAL hidden states; observability
  contracts (`schedule_health.json`, wandb, RNG/scheduler persistence) honored.
- Smoke: an eval run emits real (non-noise) probe accuracy + a `schedule_health.json`;
  a `--wandb` run is gated correctly.

---

## Stage D1 — Eval correctness (H5, H6)
- **H5 edge-case off-by-one** (`pawn/eval_suite/diagnostics.py:135-167`): pair
  `correct[:, :-1]` with `bits[:, 1:]` (shift bits right by one), expressed against
  the Phase-A prefix-shifted layout (account for `C`). Also fix the terminal-label
  case (checkmate/stalemate currently scored against a PAD target). Tests: a known
  in-check ply AND a terminal label score correctly.
- **H6 real probes** (`scripts/eval_probes_jax.py:29-45`, `pawn/probes.py`): forward
  the FROZEN model on engine games, extract per-layer hidden states, label via
  `engine.extract_board_states`; add a train/val split (the current `fit_probe`
  reports in-sample, `probes.py:81-96`). Do NOT emit `results['probes']` from
  `run_evals_backbone.py` until real. Tests: probe accuracy on a known-separable
  feature is materially above chance and uses held-out data.

## Stage D2 — Observability contracts (H7, §8.3)
- **H7 `schedule_health.json`**: re-implement `write_schedule_health` at BOTH exit
  paths (`scripts/train_jax.py:495-514` + adapter exit) recording
  `{planned_total_steps, actual_total_steps, reason_for_stop, lr_peak,
  actual_final_lr}`; the lab runner reads it and flags `actual≠planned ∧
  completed`. Tests: file written at normal + SIGTERM exit; lab-runner banner on a
  structural mismatch.
- **wandb**: wire `pawn/wandb_utils.py` into both training entry points (gated on
  `--wandb` + the `wandb` extra) or hard-error if requested-but-unavailable; fix
  the false "every entry point funnels through" docstring. Tests: `--wandb` without
  the extra errors clearly; with it, the mirror is invoked (mock).
- **Scheduler + RNG persistence** (`pawn/checkpoint.py:19-21`,
  `pawn/lifecycle.py:405-448`): persist scheduler state + RNG in
  `training_state.json`; resume restores them. Tests: RNG/scheduler round-trip;
  a resumed run is bit-reproducible vs an uninterrupted run.

## Stage D3 — Diagnostics cleanup (§8.3, §8.4)
- `improbable_task_test` conditions on the unreachable `DRAW_BY_AGREEMENT`
  (`generation.py:649-679`) → use a producible outcome or document the structural
  pin. Fold the 8 per-chunk `int()` host-syncs in per-phase accuracy
  (`eval.py:144-172`) into the jitted body. Add `setup_jax_caching` to the eval
  scripts (`scripts/eval_jax.py:16-49`). Reconcile the three RoPE-precision
  docstrings; drop dead imports; hoist duplicated `_resolve_device`/
  `_require_accelerator` into `pawn/jax_setup.py`. Tests/asserts as appropriate.

## Full suite & smoke
- Full suite: `pyright pawn scripts tests` + `pytest tests/ -q`.
- Smoke (single process; prefer GPU else `PAWN_ALLOW_CPU=1`):
  1. Probes: `scripts/eval_probes_jax.py` on a tiny local checkpoint → per-layer
     probe accuracy is real (above chance on a separable feature), held-out.
  2. `schedule_health.json`: a tiny pretrain writes it at exit with sane fields.
  3. wandb gate: `--wandb` without the extra errors cleanly; (optional) with a
     mocked mirror, a metric is forwarded.

## Done = all global-invariant gates green + smoke passes.
