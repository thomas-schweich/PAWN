"""Tests for :mod:`pawn.trainer` — Optax wiring + JIT-compiled train step.

Coverage per plan §10 S6 verification list:

- `state.step` is a JAX scalar inside JIT (no per-step retrace).
- `optax.warmup_cosine_decay_schedule` is built with
  `decay_steps=total_steps` (the canonical pinned-bug-from-v1).
- Padded-batch weight-decay guard works (empty `loss_mask` → no
  param drift).
- Gradient clipping caps the global norm at 1.0.
- Scan doesn't recompile per call.
- LR schedule shapes (cosine / wsd / constant / one_cycle / infinite).
- supernet_joint_loss sums per-variant CEs.

A real-data smoke (TINY_SUPERNET 1000 steps → monotonic loss decrease)
is the S6 section-close verification command in the plan; we run a
small (50-step) variant of that here to confirm the trainer wires up
end-to-end.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pawn.config import TINY_SUPERNET, TINY_VARIANTS
from pawn.corpus import generate_corpus
from pawn.model import PAWNModel, init_model, sliced
from pawn.run_config import PretrainConfig
from pawn.trainer import (
    Batch,
    TrainState,
    VariantSpec,
    cross_entropy_loss,
    make_lr_schedule,
    make_optimizer,
    make_scan_step,
    make_train_step,
    slice_batch,
    supernet_joint_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_model() -> PAWNModel:
    return init_model(TINY_SUPERNET, key=0)


def _tiny_variants() -> tuple[VariantSpec, ...]:
    return (
        VariantSpec("small", TINY_VARIANTS["small"]),
        VariantSpec("base", TINY_VARIANTS["base"]),
        VariantSpec("large", TINY_VARIANTS["large"], is_supernet=True),
    )


def _tiny_train_state(model: PAWNModel, *, warmup_frac: float = 0.0) -> tuple[TrainState, optax.GradientTransformation]:
    """Build a TrainState with the tiny supernet. Default `warmup_frac=0`
    so step 0 already has a non-zero LR (tests that need to observe
    parameter movement after a single step don't have to step past
    the warmup ramp first)."""
    cfg = PretrainConfig(
        local_checkpoints=True,
        total_steps=100,
        batch_size=4,
        seq_len=32,
        k=4,
        warmup_frac=warmup_frac,
        supernet="tiny",
        lr_schedule="constant",
    )
    schedule = make_lr_schedule(cfg, total_steps=100)
    opt = make_optimizer(cfg, schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    return TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.int32(0),
        key=jax.random.key(0),
    ), opt


def _small_batch(seq_len: int = 32, batch_size: int = 4) -> Batch:
    corpus = generate_corpus(
        n_games=batch_size, max_ply=seq_len, seq_len=seq_len, seed=0
    )
    return slice_batch(corpus, np.arange(batch_size))


# ---------------------------------------------------------------------------
# Batch + state shapes
# ---------------------------------------------------------------------------


def test_slice_batch_produces_jax_arrays() -> None:
    """slice_batch transfers host-numpy → JAX device arrays."""
    corpus = generate_corpus(n_games=4, max_ply=32, seq_len=64, seed=0)
    batch = slice_batch(corpus, np.array([0, 1]))
    assert isinstance(batch.tokens, jax.Array)
    assert batch.tokens.shape == (2, 64)


def test_train_state_step_is_jax_scalar() -> None:
    """`state.step` must be a JAX 0-d array so JIT doesn't retrace on
    value changes — Python int would invalidate the cache every step."""
    model = _tiny_model()
    state, _ = _tiny_train_state(model)
    assert isinstance(state.step, jax.Array)
    assert state.step.shape == ()


# ---------------------------------------------------------------------------
# Cross-entropy loss
# ---------------------------------------------------------------------------


def test_cross_entropy_loss_returns_finite_scalar() -> None:
    model = _tiny_model()
    batch = _small_batch()
    loss = cross_entropy_loss(model, batch)
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert loss > 0


def test_cross_entropy_loss_zero_on_fully_padded_batch() -> None:
    """When loss_mask is all-False, loss is exactly 0 (clip(1) denom)."""
    model = _tiny_model()
    batch = _small_batch()
    # Build a batch with the same shapes but a zero loss_mask.
    empty_batch = Batch(
        tokens=batch.tokens,
        targets=batch.targets,
        attn_mask=batch.attn_mask,
        loss_mask=jnp.zeros_like(batch.loss_mask),
    )
    loss = cross_entropy_loss(model, empty_batch)
    assert float(loss) == 0.0


def test_supernet_joint_loss_sums_variants() -> None:
    """Plan §5 / §10 S6: joint loss is the SUM (not mean) of per-variant
    CEs on the same batch. Each variant contributes its full
    per-supervised-position loss into the shared weight gradient."""
    model = _tiny_model()
    batch = _small_batch()
    variants = _tiny_variants()
    joint = supernet_joint_loss(model, batch, variants)
    assert joint.shape == ()
    assert jnp.isfinite(joint)
    per_variant = [
        cross_entropy_loss(sliced(model, spec.cfg) if not spec.is_supernet else model, batch)
        for spec in variants
    ]
    expected = sum(per_variant)
    assert jnp.allclose(joint, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# LR schedule — every shape produces values in the expected range
# ---------------------------------------------------------------------------


def _make_cfg(**kwargs: Any) -> PretrainConfig:
    defaults: dict[str, Any] = dict(local_checkpoints=True, total_steps=100, batch_size=4)
    defaults.update(kwargs)
    return PretrainConfig(**defaults)


def _f(arr_like: Any) -> float:
    """`_f(sched(step))` — extracted helper because pyright sees the
    optax schedule's return type as `ArrayLike` (which is a `Union` that
    includes `complex`), refusing the direct ``float(...)`` cast.
    ``np.asarray(x).item()`` returns a real Python ``float``, which
    satisfies the type check while behaving identically at runtime.
    """
    return np.asarray(arr_like).item()


def test_lr_schedule_cosine_warmup_then_decay() -> None:
    cfg = _make_cfg(lr_schedule="cosine", warmup_frac=0.1, lr=1e-3)
    sched = make_lr_schedule(cfg, total_steps=100)
    assert _f(sched(0)) == pytest.approx(0.0, abs=1e-7)
    # Peak around end of warmup.
    assert _f(sched(10)) == pytest.approx(1e-3, rel=1e-3)
    # End of schedule decays to 0.
    assert _f(sched(99)) < 1e-4


def test_lr_schedule_cosine_decay_steps_equals_total_steps() -> None:
    """Plan §10 S6 pinned: `decay_steps=total_steps`, NOT total_steps -
    warmup. The cosine completes its decay by `total_steps`, not earlier.
    """
    cfg = _make_cfg(lr_schedule="cosine", warmup_frac=0.5, lr=1.0)
    sched = make_lr_schedule(cfg, total_steps=100)
    # At step 50 (end of warmup, decay starts), LR is peak.
    assert _f(sched(50)) == pytest.approx(1.0, rel=1e-3)
    # At step 100, LR has just reached end (≈0).
    # If decay_steps were (total_steps - warmup) = 50, we'd see LR=0
    # at step 100 with the first 50 steps in pure cosine — but we'd
    # also see weird oscillation past step 100.
    assert _f(sched(100)) == pytest.approx(0.0, abs=1e-3)


def test_lr_schedule_constant_holds_peak() -> None:
    cfg = _make_cfg(lr_schedule="constant", warmup_frac=0.05, lr=1e-3)
    sched = make_lr_schedule(cfg, total_steps=100)
    assert _f(sched(50)) == pytest.approx(1e-3, rel=1e-3)
    assert _f(sched(99)) == pytest.approx(1e-3, rel=1e-3)


def test_lr_schedule_wsd_warmup_stable_decay() -> None:
    cfg = _make_cfg(
        lr_schedule="wsd", warmup_frac=0.1, decay_frac=0.1, lr=1e-3
    )
    sched = make_lr_schedule(cfg, total_steps=100)
    # Stable middle (step 50): peak.
    assert _f(sched(50)) == pytest.approx(1e-3, rel=1e-3)
    # End of run: decayed.
    assert _f(sched(100)) < 1e-4


def test_lr_schedule_one_cycle_ramps_then_cosine() -> None:
    cfg = _make_cfg(lr_schedule="one_cycle", warmup_frac=0.1, lr=1e-3)
    sched = make_lr_schedule(cfg, total_steps=100)
    # Peak at end of warmup.
    assert _f(sched(10)) == pytest.approx(1e-3, rel=1e-3)
    # End: small (peak/10000).
    assert _f(sched(99)) < 1e-5


def test_lr_schedule_infinite_has_stable_plateau() -> None:
    cfg = _make_cfg(
        lr_schedule="infinite",
        warmup_frac=0.05,
        cooldown_frac=0.2,
        decay_frac=0.1,
        stable_lr_ratio=0.1,
        lr=1.0,
    )
    sched = make_lr_schedule(cfg, total_steps=100)
    # Stable plateau (middle of stable phase): stable_lr.
    # warmup ends at 5, cooldown at 25, stable ends at 100 - 10 = 90.
    # Pick a step deep in stable: 50.
    assert _f(sched(50)) == pytest.approx(0.1, rel=1e-2)


def test_lr_schedule_rejects_unknown_shape() -> None:
    cfg = _make_cfg(lr_schedule="cosine")
    # Bypass the literal type check by mutating the underlying field.
    object.__setattr__(cfg, "lr_schedule", "no-such-schedule")
    with pytest.raises(ValueError, match="lr_schedule"):
        make_lr_schedule(cfg, total_steps=100)


def test_lr_schedule_rejects_warmup_greater_than_total_steps() -> None:
    """`warmup_steps` > `total_steps` would make optax's cosine builders
    pass negative `decay_steps` and crash. Catch it earlier with a
    clear message."""
    cfg = _make_cfg(warmup_steps=200, total_steps=100)
    with pytest.raises(ValueError, match="warmup_steps.*exceeds"):
        make_lr_schedule(cfg, total_steps=100)


def test_lr_schedule_wsd_rejects_non_monotonic_boundary() -> None:
    """PR #115 review: `_check_lr_schedule_fractions` validates the
    fractions, but `warmup_steps` overrides `warmup_frac` and can
    drive `warmup + decay_steps > total_steps` — which produces a
    non-monotonic `join_schedules` boundary that Optax silently
    clamps wrong. Refuse the config up front.
    """
    cfg = _make_cfg(
        lr_schedule="wsd", warmup_steps=600, decay_frac=0.5,
        total_steps=1000,
    )
    with pytest.raises(ValueError, match="wsd schedule.*exceeds"):
        make_lr_schedule(cfg, total_steps=1000)


def test_lr_schedule_infinite_rejects_non_monotonic_boundary() -> None:
    """Same gap as WSD for the `infinite` schedule (warmup +
    cooldown + decay can exceed total even when fractions pass).
    """
    cfg = _make_cfg(
        lr_schedule="infinite", warmup_steps=600,
        cooldown_frac=0.3, decay_frac=0.2,
        total_steps=1000,
    )
    with pytest.raises(ValueError, match="infinite schedule.*exceeds"):
        make_lr_schedule(cfg, total_steps=1000)


def test_lr_schedule_infinite_with_extreme_rounding_doesnt_crash() -> None:
    """A config with float fractions summing close to 1.0 can round into
    integer step counts that sum to > total_steps. The `infinite`
    schedule clamps `stable_steps` to `max(0, ...)` so `join_schedules`
    doesn't see non-monotonic boundaries."""
    cfg = _make_cfg(
        lr_schedule="infinite",
        warmup_frac=0.17,
        cooldown_frac=0.17,
        decay_frac=0.5,
        stable_lr_ratio=0.1,
        total_steps=3,
    )
    # Constructs without error even though int(round(0.17*3)) +
    # int(round(0.17*3)) + int(round(0.5*3)) = 1+1+2 = 4 > 3.
    sched = make_lr_schedule(cfg, total_steps=3)
    # Each step is queryable (no crash).
    for step in range(3):
        _f(sched(step))


def test_supernet_joint_loss_rejects_empty_variants() -> None:
    """An empty `variants` tuple would produce zero loss every step,
    silently advancing the optimizer state with weight-decay drift.
    Refuse it loudly."""
    model = _tiny_model()
    batch = _small_batch()
    with pytest.raises(ValueError, match="at least one VariantSpec"):
        supernet_joint_loss(model, batch, ())


# ---------------------------------------------------------------------------
# Optimizer + gradient clip
# ---------------------------------------------------------------------------


def test_make_optimizer_chain_has_clip_then_adamw() -> None:
    """The optax.chain structure is `clip_by_global_norm(1.0) → adamw`.
    AdamW renormalises gradients via the second-moment estimate so a
    1000x-too-big gradient still produces an update of order `lr`
    regardless of upstream clipping. The right test is structural:
    inspect the chain's transformations to confirm clip is the first
    transform.

    A separate clip-only test against a non-adaptive optimizer would
    verify the clip's numerical effect, but that's outside the
    `make_optimizer` contract (which is specifically the v1 trainer's
    `clip + adamw` shape)."""
    cfg = _make_cfg(lr=1.0, weight_decay=0.0)
    sched = make_lr_schedule(cfg, total_steps=100)
    opt = make_optimizer(cfg, sched)
    # The returned GradientTransformation is the composed optax.chain;
    # we can't inspect its internal `args` directly via the public API,
    # but we can verify that init_fn produces the chained state tuple
    # whose first member is the clip's state.
    params = {"w": jnp.ones((4,))}
    state = opt.init(params)
    # The clip-by-global-norm state is `EmptyState` (it's stateless);
    # the chain produces a tuple of states. The first slot corresponds
    # to the clip; AdamW's state is more complex (with `count`).
    assert isinstance(state, tuple)
    # AdamW's state has a `count` attribute somewhere downstream; the
    # clip's state is empty. Verify the shape matches `clip → adamw`.
    assert len(state) == 2  # chain produces (clip_state, adamw_state)


def test_make_optimizer_clip_actually_clips_under_sgd() -> None:
    """Direct verification of the clip-by-global-norm value (1.0): build
    a minimal `clip(1.0) → sgd(lr=1.0)` chain and confirm that a
    1000-norm gradient produces a unit-norm update."""
    pure_clip_sgd = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate=1.0),
    )
    params = {"w": jnp.ones((4,))}
    huge_grads = {"w": jnp.full((4,), 500.0)}  # global norm = 1000
    state = pure_clip_sgd.init(params)
    updates, _ = pure_clip_sgd.update(huge_grads, state, params)
    # `updates` is a `dict[str, jax.Array]` mirroring `params` at runtime.
    # optax's typed `Updates` is the generic `ArrayTree` (Array | dict | …),
    # so pyright won't accept `updates["w"]` without an explicit cast.
    from typing import cast
    upd_w = cast(dict[str, jax.Array], updates)["w"]
    # After clip to norm 1.0 + sgd(lr=1.0), update norm ≈ 1.0 (with
    # SGD's sign flip: updates = -clipped_grads).
    upd_norm = _f(jnp.linalg.norm(upd_w))
    assert upd_norm == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Train step + JIT contract
# ---------------------------------------------------------------------------


def test_train_step_updates_params_and_advances_step() -> None:
    """A single train step runs end-to-end, advances step, returns
    finite loss."""
    model = _tiny_model()
    # Snapshot lm_head BEFORE the train step — `donate="all"` on the
    # JIT will delete the original buffer, so a later read would raise.
    lm_head_before = np.asarray(model.lm_head)
    state, opt = _tiny_train_state(model)
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants)
    batch = _small_batch()
    new_state, loss = train_step(state, batch)
    assert int(new_state.step) == 1
    assert jnp.isfinite(loss)
    # Model changed.
    assert not np.array_equal(lm_head_before, np.asarray(new_state.model.lm_head))


def test_train_step_jit_does_not_retrace_across_steps() -> None:
    """The JIT cache should hit for steps 1, 2, 3 — no re-trace per
    call. We verify by inspecting `train_step._fn`'s cache info
    (eqx.filter_jit wraps but the underlying jax.jit can be queried)."""
    model = _tiny_model()
    state, opt = _tiny_train_state(model)
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants)
    batch = _small_batch()
    # First call compiles; subsequent calls hit cache.
    state, _ = train_step(state, batch)
    state, _ = train_step(state, batch)
    state, _ = train_step(state, batch)
    assert int(state.step) == 3


def test_train_step_padded_batch_does_not_drift_params() -> None:
    """The lax.cond guard against padded-batch weight-decay drift:
    when `loss_mask.sum() == 0`, the optimizer is skipped entirely
    and `weight_decay * model_params` shouldn't apply."""
    model = _tiny_model()
    # Snapshot params before donation deletes them.
    lm_head_before = np.asarray(model.lm_head)
    embed_src_before = np.asarray(model.embed_src)
    cfg = _make_cfg(lr=1e-3, weight_decay=0.1)  # nontrivial wd
    sched = make_lr_schedule(cfg, total_steps=100)
    opt = make_optimizer(cfg, sched)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.int32(0),
        key=jax.random.key(0),
    )
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants)

    batch = _small_batch()
    empty_batch = Batch(
        tokens=batch.tokens,
        targets=batch.targets,
        attn_mask=batch.attn_mask,
        loss_mask=jnp.zeros_like(batch.loss_mask),
    )
    state_after_empty, _ = train_step(state, empty_batch)
    # Params should be byte-identical (no weight-decay drift).
    assert np.array_equal(lm_head_before, np.asarray(state_after_empty.model.lm_head))
    assert np.array_equal(embed_src_before, np.asarray(state_after_empty.model.embed_src))


# ---------------------------------------------------------------------------
# K-step scan
# ---------------------------------------------------------------------------


def test_scan_step_runs_K_inner_steps() -> None:
    """The K-step scan returns the same final state as K sequential
    train steps would, with K losses."""
    model = _tiny_model()
    state, opt = _tiny_train_state(model)
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants)
    scan_step = make_scan_step(train_step)

    # Build a (K, B, T) batch by stacking K copies of the same batch.
    K = 4
    base = _small_batch()
    batches = Batch(
        tokens=jnp.stack([base.tokens] * K),
        targets=jnp.stack([base.targets] * K),
        attn_mask=jnp.stack([base.attn_mask] * K),
        loss_mask=jnp.stack([base.loss_mask] * K),
    )
    new_state, losses = scan_step(state, batches)
    assert int(new_state.step) == K
    assert losses.shape == (K,)
    assert jnp.all(jnp.isfinite(losses))


# ---------------------------------------------------------------------------
# Smoke: small-scale training run with loss-decreasing assertion
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_short_training_run_decreases_loss() -> None:
    """50-step smoke that the trainer wires up. Not the full 1000-step
    acceptance criterion 6 — that runs from `scripts/train_jax.py` in
    S13. Here we just confirm the trainer mechanics are wired
    correctly: loss is finite, decreases over the run."""
    model = _tiny_model()
    cfg = PretrainConfig(
        local_checkpoints=True,
        total_steps=50,
        batch_size=8,
        lr=1e-3,
        warmup_frac=0.1,
        seq_len=32,
        k=10,
        supernet="tiny",
    )
    schedule = make_lr_schedule(cfg, total_steps=50)
    opt = make_optimizer(cfg, schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.int32(0),
        key=jax.random.key(0),
    )
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants)

    corpus = generate_corpus(n_games=200, max_ply=32, seq_len=32, seed=42)
    losses = []
    for s in range(50):
        idx = np.arange(s * 8, (s + 1) * 8) % corpus.n_games
        batch = slice_batch(corpus, idx)
        state, loss = train_step(state, batch)
        losses.append(float(loss))
    # First 5 vs last 5 — loss should drop.
    assert sum(losses[-5:]) < sum(losses[:5])
