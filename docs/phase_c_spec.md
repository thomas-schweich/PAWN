# Phase C — Adapters: Implementation Spec

Phase C of `docs/v2_redesign_plan.md` (§8.2 H2/H3/H11, §8.3). Branch
`feat/v2-phase-c-adapters` off `jax_migration` (Phase B merged). Same common
process/constraints as `docs/phase_b_spec.md` §"Process & constraints"
(serialize JAX, no push, never suppress types / gut tests, parity bar).

## Global invariants (acceptance gates)
- `pyright` clean on `pawn scripts tests`; full `pytest` green.
- FiLM actually implements FiLM; adapter resume is correct for every strategy;
  the adapter loop uses the K-step scan.
- Smoke: a FiLM run with non-trivial γ/β changes outputs (not identity); a LoRA
  resume preserves optimizer correctness; the adapter K-step loop runs.

---

## Stage C1 — Real FiLM (H2)
- Implement true FiLM via the post-residual `attn_hook`/`ffn_hook` injection points
  (`pawn/model.py`): residual-stream shift `h = γ⊙h + β` (NOT folding γ/β into
  RMSNorm input weights, the current `adapters/film.py:71-128` bug). **Output FiLM
  modulates logits** at the uniform `V`-wide head (post-Phase-A vocab), not
  `final_norm_w` at `d_model`. `hybrid` inherits the corrected FiLM.
- Fix the stale `(d_model,)` "v1 parity" assertion in `tests/test_jax_adapters.py`
  (do not just add a new test).
- Tests: numeric test vs an explicit `γ⊙h+β` reference with **non-trivial** γ/β
  (identity-at-init still holds; a non-identity γ/β provably changes the residual
  stream and the logits); output-FiLM dimensionality is `V`.

## Stage C2 — Adapter resume opt-state (H3)
- On `--resume` (`scripts/train_jax_adapter.py:377-416`), a warm Adam state must
  never be applied to cold-started adapter params. Fix for ALL strategies:
  persist + reload the adapter PyTree (sidecar, as bottleneck already does) **or**
  cold-start `opt_state` when the adapter is cold-started. `unfreeze` is largely
  fine (re-clones the folded backbone) — verify.
- Tests: per-strategy resume round-trip — after save→resume, the adapter params and
  optimizer moments correspond (no first-step corruption); a resumed step equals
  the equivalent uninterrupted step within fp32 noise.

## Stage C3 — Adapter K-step scan + knobs (H11, §8.3)
- Drive the adapter loop through `make_adapter_scan_step` (K-step `lax.scan` with
  K-chunked pre-gathered batches), not one `eqx.filter_jit` dispatch per step
  (`scripts/train_jax_adapter.py:559-564`); single-step only for eval. The wrapper
  exists; it's currently dead code.
- Fix the adapter `step_time` resume divisor (divide by `(step - start)`, not the
  absolute step — `train_jax_adapter.py:544-570`).
- Consume-or-reject any adapter inert knobs surfaced (mirror Phase B's discipline).
- Tests: the scan loop produces the same trajectory as the single-step loop within
  fp32 noise over K steps; `step_time` correct after resume.

## Full suite & smoke
- Full suite: `pyright pawn scripts tests` + `pytest tests/ -q`.
- Smoke (single process; prefer GPU else `PAWN_ALLOW_CPU=1`; build a local tiny
  backbone since `pawn-base-v2` is unpublished — Phase-A carry-in):
  1. FiLM: a tiny FiLM adapter run; then with manually-set non-identity γ/β, assert
     logits differ from the un-adapted backbone (proves FiLM modulates).
  2. LoRA resume: tiny run → checkpoint → `--resume` → the resumed adapter+opt
     state is correct (no NaN/blowup on the first post-resume step).
  3. K-step adapter loop: a tiny `--strategy lora` run uses the scan path and
     backbone stays bit-identical (frozen).

## Done = all global-invariant gates green + smoke passes.
