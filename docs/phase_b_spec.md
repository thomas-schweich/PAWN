# Phase B — Pretraining Substrate: Implementation Spec

Phase B of `docs/v2_redesign_plan.md` (§7, §7.1, §8.3). Branch
`feat/v2-phase-b-distillation` off `jax_migration` (Phase A merged). The plan has
the *why*; this has the *what* + exact names. Plan intent wins on conflict.

## Process & constraints (common to all v2 phases)
- Tests / `pyright` / small smokes allowed; **NEVER run JAX/training in
  parallel** (RAM can't hold multiple model copies) — serialize.
- **Never push.** Commit per stage. **Never suppress a type error; never gut a
  test** (fix the impl, or replace a genuinely-obsolete test with an equivalent).
- Verify: `uv run --extra rocm --extra dashboard --extra lab --extra wandb pyright
  <paths>` and `uv run --extra rocm pytest <targeted>`.
- v1 compat intentionally dropped; inert/missing v2 features are **bugs** (parity
  bar). Match surrounding style; flag (don't silently make) spec deviations.

## Global invariants (phase acceptance gates)
- `pyright` clean on `pawn scripts tests`; full `pytest` green.
- Distillation trainer is the canonical-ladder mechanism (plan §7) and is built
  for generality per §7.1 (injectable teacher, pluggable loss, unified with the
  adapter trainer).
- Smoke: tiny teacher→student distillation runs (KL finite + decreasing, student
  checkpoint round-trips), grad-accum reachable, `max_grad_norm` honored.

---

## Stage B1 — Distillation trainer (plan §7, §7.1)
New module `pawn/distill.py` (+ `DistillConfig` in `pawn/run_config.py`, CLI
`scripts/train_jax_distill.py`). **Design for the adapted/conditioned-teacher
generality (§7.1) — do not hardcode teacher=large or loss=KL-only.**

- **Teacher = an injectable `logits_fn`**, not a fixed checkpoint. Signature:
  `TeacherFn = Callable[[Int[Array,"B T"], Bool[Array,"B T"]], Float[Array,"B T V"]]`.
  Provide a constructor `frozen_teacher(model) -> TeacherFn` that wraps any frozen
  `PAWNModel` (incl. an `eqx.combine`d adapter or a conditioned forward) via
  `jax.lax.stop_gradient` on the teacher params. The teacher is **frozen** — it
  must receive zero gradient (assert in a test).
- **Pluggable objective** `DistillLoss`:
  `kl` = `KL(softmax(student/T) ‖ softmax(teacher/T)) * T^2`,
  `ce` = ground-truth CE (reuse `cross_entropy_loss`),
  `mix` = `alpha*ce + (1-alpha)*kl`. Select via `DistillConfig.objective ∈
  {"kl","ce","mix"}`, `temperature: float = 2.0`, `alpha: float = 0.5`.
- **Mask KL to the supervised vocab support** (plan §7): reserved/NULL/PAD columns
  (IDs ≥ `PAD_TOKEN`, and the reserved block) are excluded from BOTH the student
  and teacher softmax (set to `-inf` before softmax) so teacher mass never spreads
  onto dead columns. Reuse the §2 column-mask helper from Phase A. Also apply the
  time-axis `loss_mask` (`build_loss_mask`, Phase A) so only supervised positions
  contribute.
- **Unify with the adapter trainer** (§7.1): the student is a trainable PyTree
  (full model OR an adapter on a frozen backbone). Reuse
  `adapter_trainer`'s partition (`grad` over the trainable leaves only). Concretely
  factor a shared `make_distill_scan_step` mirroring `make_adapter_scan_step`
  (K-step `lax.scan`, donated buffers) whose loss is `DistillLoss` against the
  teacher. Students reuse `specialized_clm` shapes; **one student at a time**.
- **No hidden-state matching** (logit KL only) — students aren't slices.
- CLI `scripts/train_jax_distill.py`: `--distill-from <teacher-ckpt>` (required),
  `--student-supernet {tiny,production}` or explicit student dims, `--objective`,
  `--distill-temp`, `--distill-alpha`, plus the standard
  `--hf-repo`/`--local-checkpoints` XOR. Teacher loaded frozen via
  `load_model`; conditioning/`C` inherited from the teacher checkpoint (reuse the
  Phase-A load-time C-assert).
- Tests `tests/test_jax_distill.py`: (a) `kl` loss is 0 when student==teacher;
  (b) teacher params bit-identical before/after a step (zero teacher grad);
  (c) grad flows only to student leaves; (d) reserved/NULL columns get no KL mass
  / no gradient; (e) `mix` reduces to `ce` at alpha=1 and `kl` at alpha=0;
  (f) the scan step matches a single-step reference within fp32 noise.

## Stage B2 — Trainer parity fixes (plan §8.2 H10, §8.3)
- **H10:** thread `cfg.max_grad_norm` into `make_optimizer` as the clip threshold
  (currently hardcoded `_CLIP_NORM=1.0`, `trainer.py:576-617`); the `did_clip`
  metric (`train_jax.py:471`) must compare the pre-clip norm against the **actual**
  threshold. Test: a run with `max_grad_norm=0.5` clips at 0.5 and `did_clip` is
  consistent.
- **Grad accumulation reachable:** the prefetcher must emit `(K,N,B,T)` so
  `accumulation_steps≠1` runs instead of raising `NotImplementedError`
  (`train_jax.py:259`); the K-step accumulation kernel (`trainer.py:760-858`) is
  done+tested — wire the data loop. Test: `accumulation_steps=2` runs; grad ==
  mean of 2 micro-batches within fp32 noise.
- **Inert knobs consumed-or-rejected:** `mate_boost` (`run_config.py:118` — thread
  into corpus gen or reject), `epochs`/`steps_per_epoch`/`data_seed`/`val_every`
  (`run_config.py:446-456` — wire into the loop/sampling or reject at parse time).
  Every field is consumed or `extra="forbid"`-rejected. Tests: each knob changes
  behavior or raises.

## Stage B3 — Supernet quality-parity harness (plan §7, Alt #1)
- New `pawn/parity.py` + `scripts/eval_parity.py`. Given a **supernet** checkpoint
  and a **canonical** (distilled/independent) checkpoint at the same width,
  compute and emit JSON: per-phase move accuracy delta, linear-probe decodability
  delta (reuse `pawn/probes.py`), and a reference-LoRA val-loss delta
  (supernet-slice vs canonical at that width). The supernet stays behind its
  existing flag as the **contrast arm**.
- This is a measurement tool — implement + unit-test the harness mechanics; full
  cross-model comparison runs are later analysis, **not** a Phase-B gate.
- Tests `tests/test_jax_parity.py`: harness runs on two tiny checkpoints and emits
  a well-formed gap report; metric signs/shapes correct.

## Full suite & smoke
- Full suite: `pyright pawn scripts tests` + `pytest tests/ -q`.
- Smoke (single process; prefer GPU else `PAWN_ALLOW_CPU=1`):
  1. Tiny teacher: `train_jax.py --supernet tiny --total-steps 20 --batch-size 8
     --seq-len 64 --k 10 --local-checkpoints` → checkpoint.
  2. Tiny distill: `train_jax_distill.py --distill-from <teacher-ckpt>
     --student-supernet tiny --total-steps 20 --batch-size 8 --seq-len 64
     --objective mix --local-checkpoints` → KL/total loss finite & trending down,
     student checkpoint written + round-trips; **teacher bit-identical** after.
  3. Grad-accum: `train_jax.py --supernet tiny --total-steps 10 --accumulation-steps
     2 --batch-size 4 --seq-len 64 --k 5 --local-checkpoints` → runs (no
     NotImplementedError), loss finite.
  4. `max_grad_norm`: a tiny run with `--max-grad-norm 0.5` → `did_clip` consistent
     with a 0.5 threshold.

## Done = all global-invariant gates green + smoke passes.
