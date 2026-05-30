# Phase E — Sweeps: Implementation Spec

Phase E of `docs/v2_redesign_plan.md` (§8.2 H8/H9). Branch
`feat/v2-phase-e-sweeps` off `jax_migration` (Phase D merged). Same common
process/constraints as `docs/phase_b_spec.md` (serialize JAX, no push, never
suppress types / gut tests, parity bar). Final v2 parity phase.

## Global invariants (acceptance gates)
- `pyright` clean on `pawn scripts tests`; full `pytest` green.
- Every sweep strategy actually produces non-pruned trials; the adapter argparse
  accepts every suggester's params.
- Smoke: a tiny multi-trial sweep for previously-broken strategies completes with
  a valid `study.best_value` (not 100% pruned).

---

## Stage E1 — Sweep fixes (H8, H9)
- **H8** (`pawn/sweep.py:64-280`): `AdapterObjective` must pass suggested params via
  a temp-JSON `--config` (round-trips through pydantic) instead of kebab-cased CLI
  flags the adapter argparse doesn't register. **Migrate the `suggest_*` functions
  off `prepend_outcome` to the Phase-A `conditioning` field** (else `extra="forbid"`
  re-breaks them). Add an **argv/config-acceptance integration test** per suggester:
  every strategy's suggested params, serialized to `--config`, are accepted by
  `train_jax_adapter.py`'s argparse + pydantic without exit-2/`TrialPruned`.
- **H9** (`scripts/sweep.py:17,53-59`): `rosa-ratio` is a selectable sweep strategy
  but not a valid adapter `--strategy` → map it onto `rosa` with `bottleneck_ratio`
  passed through a consumed flag/config, or drop it from `STRATEGY_SUGGESTERS`.
- Use `sys.executable` (not bare `'python'`) for the subprocess
  (`sweep.py:249-270`, §8.4-low) while here.
- Tests: an in-process tiny sweep (2–3 trials) for `bottleneck`, `sparse`, a `rosa`
  sub-mode, and `rosa-ratio` yields a real `best_value` (no all-prune); the
  argv/config-acceptance test covers every `STRATEGY_SUGGESTERS` entry.

## Full suite & smoke
- Full suite: `pyright pawn scripts tests` + `pytest tests/ -q`.
- Smoke (single process; prefer GPU else `PAWN_ALLOW_CPU=1`; tiny supernet, local
  backbone; minimal trials/steps to keep it short):
  1. `scripts/sweep.py --strategy bottleneck --n-trials 2 --supernet tiny
     --storage sqlite:///<tmp>/e.db --logs-dir <tmp>` completes with a finite
     `best_value` (previously 100% pruned).
  2. Repeat for `sparse` and one `rosa` sub-mode (and `rosa-ratio` if retained).

## Done = all global-invariant gates green + smoke passes.
## On completion: this closes the v2 parity work — final report should assess the
## whole v2 migration (Phases A–E) and any residual §8.4-low cleanups deferred.
