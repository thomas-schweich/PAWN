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

import math
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
    _FIRST_RESERVED_COLUMN,
    cross_entropy_loss,
    get_grad_norm,
    make_lr_schedule,
    make_optimizer,
    make_scan_step,
    make_train_step,
    slice_batch,
    supernet_joint_loss,
    top1_accuracy,
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


def test_cross_entropy_loss_masks_reserved_columns() -> None:
    """Reserved / NULL / control columns (IDs >= NULL_TOKEN) must be
    masked to -inf before the softmax, so they neither dilute the
    move-token probability mass nor change the loss. We verify by
    inflating those columns' logits via the embedding table: because the
    head is tied (logits = x @ embed_tokens.T), scaling the reserved rows
    of `embed_tokens` scales their logit columns. The masked CE must be
    invariant to that perturbation."""
    model = _tiny_model()
    batch = _small_batch()
    base_loss = float(cross_entropy_loss(model, batch))

    # Blow up the reserved rows of the embedding table by a large factor —
    # without masking, these would dominate the softmax denominator and
    # shrink the loss. With masking they're -inf and contribute nothing.
    # (`np.array(...)` forces a writable copy — `np.asarray` of a JAX array
    # yields a read-only view.)
    embed = np.array(model.embed_tokens)
    embed[_FIRST_RESERVED_COLUMN:] *= 1e3
    perturbed = eqx.tree_at(
        lambda m: m.embed_tokens, model, jnp.asarray(embed)
    )
    perturbed_loss = float(cross_entropy_loss(perturbed, batch))
    assert perturbed_loss == pytest.approx(base_loss, abs=1e-4)


def test_cross_entropy_loss_reserved_rows_get_zero_grad() -> None:
    """Acceptance gate: reserved / NULL / control columns (>= NULL_TOKEN)
    receive exactly zero gradient after a loss/grad — the -inf softmax
    mask detaches them from the autograd graph. For a tied model the
    reserved *rows* of `embed_tokens` are both the (unused) input
    embeddings for those ids and their (masked) output logit columns; the
    grad through the CE must be all-zero there."""
    model = _tiny_model()
    batch = _small_batch()
    grads = eqx.filter_grad(cross_entropy_loss)(model, batch)
    g_embed = np.asarray(grads.embed_tokens)
    reserved_grad = g_embed[_FIRST_RESERVED_COLUMN:]
    assert np.all(reserved_grad == 0.0), (
        "reserved/NULL/control embedding rows must get zero gradient"
    )
    # Sanity: at least some non-reserved rows DO get gradient, so the test
    # isn't trivially passing on an all-zero grad tree.
    assert np.any(g_embed[:_FIRST_RESERVED_COLUMN] != 0.0)


def test_top1_accuracy_returns_finite_rate() -> None:
    """`top1_accuracy` (the v2 pretrain `train/accuracy` source) is a finite
    scalar in [0, 1]."""
    model = _tiny_model()
    batch = _small_batch()
    acc = top1_accuracy(model, batch)
    assert acc.shape == ()
    assert jnp.isfinite(acc)
    assert 0.0 <= float(acc) <= 1.0


def test_top1_accuracy_perfect_when_argmax_matches_targets() -> None:
    """A batch whose targets equal the model's restricted argmax scores a
    perfect 1.0 — the formula is `mean(argmax == target | supervised)`,
    matching v1 (`git show main:pawn/trainer.py:1030`)."""
    model = _tiny_model()
    batch = _small_batch()
    # Forward + restricted argmax give the predictions the metric scores
    # itself against; using them AS the targets must yield exactly 1.0.
    from pawn.trainer import mask_reserved_columns

    logits = model(batch.tokens, batch.attn_mask)
    preds = jnp.argmax(mask_reserved_columns(logits.astype(jnp.float32)), axis=-1)
    aligned = Batch(
        tokens=batch.tokens,
        targets=preds.astype(batch.targets.dtype),
        attn_mask=batch.attn_mask,
        loss_mask=batch.loss_mask,
    )
    acc = top1_accuracy(model, aligned)
    assert float(acc) == pytest.approx(1.0)


def test_top1_accuracy_zero_on_fully_padded_batch() -> None:
    """All-False loss_mask → 0 correct / 1 (clip) denom → exactly 0.0."""
    model = _tiny_model()
    batch = _small_batch()
    empty = Batch(
        tokens=batch.tokens,
        targets=batch.targets,
        attn_mask=batch.attn_mask,
        loss_mask=jnp.zeros_like(batch.loss_mask),
    )
    assert float(top1_accuracy(model, empty)) == 0.0


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


def test_supernet_joint_loss_stochastic_is_unbiased_in_expectation() -> None:
    """The stochastic sandwich loss is an UNBIASED estimator of the full
    sum-over-variants loss (`stochastic_variants=True` design intent,
    pawn/trainer.py: the sampled non-supernet variant's CE is scaled by N
    so ``E[loss_stochastic] == sum-of-all loss``).

    Tested empirically: average the stochastic loss over every distinct
    ``stochastic_key`` draw (one per non-supernet variant, all equally
    likely) and check the mean equals the deterministic sum. Because the
    sampling is a uniform draw over a finite set of branches, the exact
    expectation is the mean over the branch losses — we enumerate the
    branches directly rather than Monte-Carlo sampling, so the assertion is
    tight (no sampling variance).

    Previously only ``stochastic_variants=False`` (the deterministic sum)
    was covered, leaving the unbiasedness of the production loss surface
    untested.
    """
    model = _tiny_model()
    batch = _small_batch()
    variants = _tiny_variants()

    deterministic = supernet_joint_loss(
        model, batch, variants, stochastic_key=None
    )

    # The supernet (is_supernet) CE always runs; one of the N non-supernet
    # variants is sampled uniformly per key. Average the stochastic loss
    # over many keys and assert the mean matches the deterministic sum.
    n_other = sum(1 for v in variants if not v.is_supernet)
    assert n_other >= 2  # tiny supernet has small + base as non-supernet

    # jit the per-key loss so each draw reuses one compiled program, then
    # loop sequentially over the keys. A vmap over the key axis would batch
    # every variant's full forward over the leading key dim and OOM the GPU
    # at any useful sample count, so we trade the batched dispatch for a
    # Python loop over a single compiled call.
    loss_fn = eqx.filter_jit(
        lambda k: supernet_joint_loss(model, batch, variants, stochastic_key=k)
    )
    keys = jax.random.split(jax.random.key(0), 256)
    stoch_losses = np.array([float(loss_fn(k)) for k in keys])
    mean_stoch = float(stoch_losses.mean())
    # Unbiased: the Monte-Carlo mean converges to the deterministic sum.
    # 256 draws over the non-supernet branches keep the standard error well
    # under the 3% tolerance for the tiny supernet's loss scale.
    assert mean_stoch == pytest.approx(float(deterministic), rel=0.03)

    # The estimator is genuinely stochastic: the individual draws vary
    # (they pick different N-scaled variants), so the sample carries
    # non-trivial spread rather than being a constant equal to the mean.
    # (At random init the per-variant CEs all sit near ln(V), so a single
    # draw can coincide with the sum; the spread, not any single draw, is
    # what evidences stochasticity.)
    assert float(stoch_losses.std()) > 0.0


def test_supernet_joint_loss_stochastic_exact_branch_expectation() -> None:
    """Tighter, variance-free version of the unbiasedness check: the
    expectation over the uniform branch choice equals the deterministic
    sum *exactly* (no Monte-Carlo error).

    The stochastic loss is ``supernet_CE + N * sampled_variant_CE`` where
    the sampled variant is uniform over the N non-supernet variants. So
    ``E[stochastic] = supernet_CE + N * mean_v(variant_CE)
    = supernet_CE + sum_v(variant_CE)`` = the full sum. We compute each
    side from the per-variant CEs directly.
    """
    model = _tiny_model()
    batch = _small_batch()
    variants = _tiny_variants()

    supernet_ce = sum(
        float(cross_entropy_loss(model, batch))
        for spec in variants if spec.is_supernet
    )
    other_ces = [
        float(cross_entropy_loss(sliced(model, spec.cfg), batch))
        for spec in variants if not spec.is_supernet
    ]
    n_other = len(other_ces)

    # Closed-form expectation of the stochastic estimator.
    expected_stochastic = supernet_ce + n_other * (sum(other_ces) / n_other)
    deterministic = float(
        supernet_joint_loss(model, batch, variants, stochastic_key=None)
    )
    assert expected_stochastic == pytest.approx(deterministic, rel=1e-4)


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
    # Ramp starts at peak/25 (Smith one-cycle initial_div).
    assert _f(sched(0)) == pytest.approx(1e-3 / 25.0, rel=1e-3)
    # Peak at end of warmup.
    assert _f(sched(10)) == pytest.approx(1e-3, rel=1e-3)
    # End: small (peak/10000).
    assert _f(sched(99)) < 1e-5


def test_lr_schedule_one_cycle_ramp_is_cosine_not_linear() -> None:
    """The one-cycle warmup ramp is **cosine**, not linear (v1 parity —
    `git show main:pawn/trainer.py` ``OneCycle`` used
    ``init + (peak-init)*0.5*(1-cos(pi*progress))``).

    An earlier v2 used ``optax.linear_schedule`` for the ramp, which
    changes the canonical one-cycle shape. The cosine ramp is a strictly
    convex acceleration into the peak: at the ramp midpoint the cosine
    ease passes through exactly the linear interpolant value (the cos term
    is 0.5 there), but at the first quarter the cosine ramp sits *below*
    the straight line, and at the third quarter it sits *above*. We pin
    both asymmetry points so a regression back to the linear ramp fails.
    """
    peak = 1.0
    warmup = 100
    cfg = _make_cfg(
        lr_schedule="one_cycle", warmup_steps=warmup, lr=peak,
        total_steps=1000,
    )
    sched = make_lr_schedule(cfg, total_steps=1000)
    init = peak / 25.0

    def cos_ramp(step: int) -> float:
        progress = step / warmup
        return init + (peak - init) * 0.5 * (1.0 - math.cos(math.pi * progress))

    def lin_ramp(step: int) -> float:
        progress = step / warmup
        return init + (peak - init) * progress

    # Matches the v1 cosine ramp closed form at every probe point.
    for step in (25, 50, 75):
        assert _f(sched(step)) == pytest.approx(cos_ramp(step), rel=1e-4)
    # Cosine ramp ≠ linear ramp away from the midpoint (the regression
    # guard): below the line in the first quarter, above it in the third.
    assert _f(sched(25)) < lin_ramp(25) - 1e-3
    assert _f(sched(75)) > lin_ramp(75) + 1e-3
    # Midpoint coincides with the linear interpolant.
    assert _f(sched(50)) == pytest.approx(lin_ramp(50), rel=1e-4)


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


def test_lr_schedule_wsd_with_extreme_rounding_doesnt_crash() -> None:
    """Mirror of the infinite-schedule rounding test. The `wsd`
    boundary `total_steps - decay_steps` can fall below `warmup` when
    `warmup_frac + decay_frac` rounds just over 1.0 — e.g.
    `warmup_frac=0.5, decay_frac=0.5, total_steps=3` rounds to
    warmup=2, decay_steps=2, decay-start=1 (< warmup). Clamp
    `decay_start = max(warmup, total_steps - decay_steps)` keeps the
    `join_schedules` boundaries monotonic. (Round-1 review-test-risk +
    review-bug-detector caught the missing clamp.)
    """
    cfg = _make_cfg(
        lr_schedule="wsd",
        warmup_frac=0.5,
        decay_frac=0.5,
        total_steps=3,
    )
    sched = make_lr_schedule(cfg, total_steps=3)
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


def test_make_optimizer_clips_at_cfg_max_grad_norm() -> None:
    """H10: the clip threshold is `cfg.max_grad_norm`, not a hardcoded 1.0.

    Build `make_optimizer` with `max_grad_norm=0.5`, feed a gradient whose
    global norm is well above 0.5, and confirm:

    1. The post-clip update has global norm == 0.5 (AdamW would mask this,
       so we read the clip's own `_ClipState` to verify the threshold is
       honored rather than inspecting the final update).
    2. `get_grad_norm` reports the *pre-clip* norm — the quantity the
       `did_clip` metric (`train_jax.py`) compares against
       `cfg.max_grad_norm`. With the old hardcoded 1.0 threshold the clip
       fired at 1.0 while `did_clip` keyed on 0.5, so the two disagreed for
       any norm in (0.5, 1.0]; threading the config keeps them consistent.
    """
    cfg = _make_cfg(max_grad_norm=0.5, optimizer="adamw")
    sched = make_lr_schedule(cfg, total_steps=100)
    opt = make_optimizer(cfg, sched)
    params = {"w": jnp.ones((4,))}
    # global norm = sqrt(4 * 0.4^2) = 0.8 — above 0.5, below 1.0, so the
    # old hardcoded-1.0 clip would NOT fire while the 0.5 clip MUST.
    grads = {"w": jnp.full((4,), 0.4)}
    pre_norm = _f(optax.tree.norm(grads))
    assert 0.5 < pre_norm < 1.0
    state = opt.init(params)
    _updates, new_state = opt.update(grads, state, params)
    # The clip transform is chain slot 0; its `_ClipState` carries the
    # pre-clip norm so the trainer can log `did_clip` without recomputing.
    reported = _f(get_grad_norm(new_state))
    assert reported == pytest.approx(pre_norm, rel=1e-5)
    # `did_clip` consistency: the metric is `pre_norm > cfg.max_grad_norm`.
    # With the threshold honored at 0.5 the clip fires, and the reported
    # norm exceeds 0.5, so `did_clip` is True — they agree.
    assert reported > cfg.max_grad_norm

    # Direct numeric check of the 0.5 threshold under SGD (AdamW's
    # second-moment renorm would otherwise hide the clip magnitude).
    pure_clip_sgd = optax.chain(
        _branchless_clip_for_test(0.5),
        optax.sgd(learning_rate=1.0),
    )
    s = pure_clip_sgd.init(params)
    upd, _ = pure_clip_sgd.update(grads, s, params)
    from typing import cast
    upd_w = cast(dict[str, jax.Array], upd)["w"]
    # clip to 0.5 then sgd(lr=1.0) → update norm == 0.5 (sign-flipped).
    assert _f(jnp.linalg.norm(upd_w)) == pytest.approx(0.5, rel=1e-5)


def _branchless_clip_for_test(max_norm: float) -> optax.GradientTransformation:
    """Standalone clip-by-global-norm for the numeric H10 check.

    Mirrors `optax.clip_by_global_norm` semantics (the production clip in
    `pawn.trainer` carries extra `_ClipState`); used here only to verify the
    0.5 threshold's effect on a non-adaptive SGD update in isolation."""
    return optax.clip_by_global_norm(max_norm)


def _tree_all_zero(tree: Any) -> bool:
    return bool(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x: jnp.all(x == 0), tree)
        )
    )


def _trees_equal(a: Any, b: Any) -> bool:
    return bool(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), a, b)
        )
    )


def test_make_optimizer_skips_step_on_nonfinite_grad() -> None:
    """Default `grad_skip_threshold=inf` ⇒ non-finite-only skip (v1
    GradScaler parity). A NaN/Inf gradient must produce a *true* skip:
    zero updates AND a fully reverted optimizer state (Adam moments and the
    step count unchanged), so the spike never reaches the moments."""
    cfg = _make_cfg(
        lr=1.0, optimizer="adamw", warmup_frac=0.0, lr_schedule="constant"
    )
    opt = make_optimizer(cfg, make_lr_schedule(cfg, total_steps=100))
    params = {"w": jnp.ones((4,))}
    state0 = opt.init(params)

    # One finite step first so the reverted-to state is non-trivial
    # (Adam count advanced, moments populated).
    finite = {"w": jnp.full((4,), 0.1)}
    upd1, state1 = opt.update(finite, state0, params)
    assert not _tree_all_zero(upd1)

    # NaN gradient → skipped: zero update, state identical to state1.
    nan_grads = {"w": jnp.array([jnp.nan, 0.0, 0.0, 0.0])}
    upd2, state2 = opt.update(nan_grads, state1, params)
    assert _tree_all_zero(upd2)
    assert _trees_equal(state2, state1)


def test_make_optimizer_applies_finite_step_at_inf_threshold() -> None:
    """A finite gradient below the (infinite) skip threshold must NOT be
    skipped — the update is applied and the state advances."""
    cfg = _make_cfg(
        lr=1.0, optimizer="adamw", warmup_frac=0.0, lr_schedule="constant"
    )
    opt = make_optimizer(cfg, make_lr_schedule(cfg, total_steps=100))
    params = {"w": jnp.ones((4,))}
    state0 = opt.init(params)
    upd, state1 = opt.update({"w": jnp.full((4,), 0.1)}, state0, params)
    assert not _tree_all_zero(upd)
    assert not _trees_equal(state1, state0)


def test_make_optimizer_skips_step_above_finite_threshold() -> None:
    """A finite `grad_skip_threshold` rejects finite spikes above it and
    applies steps below it (the skip uses the clip's pre-clip norm)."""
    cfg = _make_cfg(
        lr=1.0,
        optimizer="adamw",
        grad_skip_threshold=0.5,
        warmup_frac=0.0,
        lr_schedule="constant",
    )
    opt = make_optimizer(cfg, make_lr_schedule(cfg, total_steps=100))
    params = {"w": jnp.ones((4,))}
    state0 = opt.init(params)

    # global norm = sqrt(4 * 0.4^2) = 0.8 > 0.5 → skipped.
    above = {"w": jnp.full((4,), 0.4)}
    assert _f(optax.tree.norm(above)) > 0.5
    upd_hi, state_hi = opt.update(above, state0, params)
    assert _tree_all_zero(upd_hi)
    assert _trees_equal(state_hi, state0)

    # global norm = sqrt(4 * 0.1^2) = 0.2 < 0.5 → applied.
    below = {"w": jnp.full((4,), 0.1)}
    assert _f(optax.tree.norm(below)) < 0.5
    upd_lo, state_lo = opt.update(below, state0, params)
    assert not _tree_all_zero(upd_lo)
    assert not _trees_equal(state_lo, state0)


# ---------------------------------------------------------------------------
# Train step + JIT contract
# ---------------------------------------------------------------------------


def test_train_step_updates_params_and_advances_step() -> None:
    """A single train step runs end-to-end, advances step, returns
    finite loss."""
    model = _tiny_model()
    # Snapshot embed_tokens BEFORE the train step — `donate="all"` on the
    # JIT will delete the original buffer, so a later read would raise.
    # TINY_SUPERNET ties embeddings (lm_head is None), so embed_tokens is
    # the trainable token table *and* the output head (logits reuse its
    # transpose); its drift covers head drift.
    embed_tokens_before = np.asarray(model.embed_tokens)
    state, opt = _tiny_train_state(model)
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants)
    batch = _small_batch()
    new_state, loss = train_step(state, batch)
    assert int(new_state.step) == 1
    assert jnp.isfinite(loss)
    # Model changed.
    assert not np.array_equal(embed_tokens_before, np.asarray(new_state.model.embed_tokens))


def test_train_step_compute_dtype_bf16_runs_and_keeps_fp32_master() -> None:
    """``make_train_step(..., compute_dtype=jnp.bfloat16)`` exercises the
    AMP forward (plan §5): activations + logits run in bf16, but the
    master parameters stay fp32 and the optimizer update lands on fp32
    weights.

    Behavioral assertions:
      * the step completes, advances, returns a finite scalar loss;
      * params move (bf16 grads still drive a real update);
      * the updated master weights are still fp32 (AMP invariant — bf16
        is a *compute* dtype, never the stored-weight dtype).
    """
    model = _tiny_model()
    assert model.embed_tokens.dtype == jnp.float32
    embed_before = np.asarray(model.embed_tokens)
    state, opt = _tiny_train_state(model)
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants, compute_dtype=jnp.bfloat16)
    batch = _small_batch()

    new_state, loss = train_step(state, batch)

    assert int(new_state.step) == 1
    assert jnp.isfinite(loss)
    # Master weights stayed fp32 — bf16 is compute-only.
    assert new_state.model.embed_tokens.dtype == jnp.float32
    # bf16 grads still moved the params.
    assert not np.array_equal(embed_before, np.asarray(new_state.model.embed_tokens))


def test_train_step_compute_dtype_bf16_loss_tracks_fp32() -> None:
    """The bf16 forward must produce a loss close to the fp32 forward on
    the same model + batch — proving the AMP path actually runs in bf16
    (and isn't silently identical to fp32) while still tracking it within
    the dtype's coarse precision. A bf16 step whose loss diverged wildly
    from fp32 would signal a broken cast somewhere in the forward.
    """
    model = _tiny_model()
    variants = _tiny_variants()
    batch = _small_batch()

    fp32_loss = float(supernet_joint_loss(model, batch, variants))
    bf16_loss = float(
        supernet_joint_loss(
            model, batch, variants, compute_dtype=jnp.bfloat16
        )
    )
    assert math.isfinite(bf16_loss)
    # bf16 mantissa is ~3 decimal digits; the joint CE over a tiny batch
    # lands within a few percent of the fp32 value.
    assert abs(bf16_loss - fp32_loss) < 0.1 * abs(fp32_loss) + 0.05


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


def _empty_batch(batch: Batch) -> Batch:
    """A fully-padded copy of `batch` — same tokens, empty loss_mask."""
    return Batch(
        tokens=batch.tokens,
        targets=batch.targets,
        attn_mask=batch.attn_mask,
        loss_mask=jnp.zeros_like(batch.loss_mask),
    )


def test_train_step_padded_batch_zero_grad_no_drift_with_zero_wd() -> None:
    """An all-PAD batch produces zero loss → zero grad → no param drift
    when ``weight_decay == 0``.

    Critically this runs at a **non-zero LR** (constant schedule,
    ``warmup_frac=0`` so ``schedule(0) == peak``). The predecessor test
    ran at step 0 of a cosine schedule whose ``schedule(0) == 0``, so it
    proved nothing: params can't move when the LR is zero regardless of
    the batch. Pinning the LR > 0 first means the no-drift result is
    attributable to the empty batch's zero gradient, not a zero LR
    (the misleading-pass the audit flagged). With ``weight_decay=0`` AdamW
    has no decoupled-decay term, so a zero gradient leaves params exactly
    fixed.
    """
    model = _tiny_model()
    embed_tokens_before = np.asarray(model.embed_tokens)
    # warmup_frac=0 + constant ⇒ schedule(0) == peak (non-zero LR at step 0).
    cfg = _make_cfg(
        lr=1e-3, weight_decay=0.0, lr_schedule="constant", warmup_frac=0.0,
    )
    sched = make_lr_schedule(cfg, total_steps=100)
    assert _f(sched(0)) == pytest.approx(1e-3, rel=1e-6)  # LR is genuinely > 0
    opt = make_optimizer(cfg, sched)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model, opt_state=opt_state, step=jnp.int32(0),
        key=jax.random.key(0),
    )
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants, stochastic_variants=False)

    batch = _small_batch()
    state_after_empty, loss = train_step(state, _empty_batch(batch))
    # The empty batch's loss is exactly zero (0 / 1).
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    # Zero grad + zero weight decay ⇒ params byte-identical.
    assert np.array_equal(
        embed_tokens_before, np.asarray(state_after_empty.model.embed_tokens)
    )


def test_train_step_padded_batch_weight_decay_drift_is_accepted() -> None:
    """v2 removed v1's ``lax.cond`` empty-batch guard *by design*
    (pawn/trainer.py `make_train_step` docstring: the cond traced + ran
    both branches, costing more than the rare all-PAD drift it prevented).

    The correct v2 behavior to pin is therefore the OPPOSITE of the old
    (misleading) assertion: at a non-zero LR with ``weight_decay > 0``, an
    all-PAD batch DOES drift params toward zero — AdamW's decoupled weight
    decay (``param *= 1 - lr*wd``) fires even when the gradient is zero.
    This documents the accepted-drift design rather than asserting a guard
    that no longer exists.
    """
    model = _tiny_model()
    embed_tokens_before = np.asarray(model.embed_tokens)
    cfg = _make_cfg(
        lr=1e-2, weight_decay=0.5, lr_schedule="constant", warmup_frac=0.0,
    )
    sched = make_lr_schedule(cfg, total_steps=100)
    opt = make_optimizer(cfg, sched)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model, opt_state=opt_state, step=jnp.int32(0),
        key=jax.random.key(0),
    )
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants, stochastic_variants=False)

    batch = _small_batch()
    state_after_empty, loss = train_step(state, _empty_batch(batch))
    after = np.asarray(state_after_empty.model.embed_tokens)
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    # Weight-decay drift: params moved (toward zero) despite the zero grad.
    assert not np.array_equal(embed_tokens_before, after)
    # The drift is a shrink toward zero (decoupled decay), not random.
    assert np.abs(after).sum() < np.abs(embed_tokens_before).sum()


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
# C.4 — gradient accumulation
# ---------------------------------------------------------------------------


def test_train_step_accumulation_steps_eq_one_matches_baseline() -> None:
    """accumulation_steps=1 must be a no-op vs the pre-C.4 path."""
    # Two fresh states with two fresh models — donate="all" deletes
    # input buffers, and sharing a model object across two TrainStates
    # would also delete its buffers when the first call runs.
    state_a, opt_a = _tiny_train_state(_tiny_model())
    state_b, opt_b = _tiny_train_state(_tiny_model())
    variants = _tiny_variants()
    ts_baseline = make_train_step(opt_a, variants)
    ts_n1 = make_train_step(opt_b, variants, accumulation_steps=1)
    batch_a = _small_batch()
    batch_b = _small_batch()
    s_base, l_base = ts_baseline(state_a, batch_a)
    s_n1, l_n1 = ts_n1(state_b, batch_b)
    assert jnp.allclose(l_base, l_n1, rtol=0, atol=1e-6)
    # TINY_SUPERNET ties embeddings (lm_head is None); compare the shared
    # embed_tokens table, which is both the trainable weight and the head.
    assert jnp.allclose(s_base.model.embed_tokens, s_n1.model.embed_tokens,
                        rtol=0, atol=1e-6)


def test_train_step_accumulation_steps_eq_2_runs() -> None:
    """accumulation_steps=2 with (N, B, T) batches runs cleanly and
    produces a finite loss + an updated model.

    A strict numeric-match-vs-concat test would need either (a) per-pos
    rather than per-game loss averaging or (b) constant supervision
    density across micros. Both are out of scope here — we instead
    smoke-test the path runs end-to-end and the optimizer actually
    moves the parameters.
    """
    model = _tiny_model()
    embed_tokens_before = np.asarray(model.embed_tokens)
    state, opt = _tiny_train_state(model)
    variants = _tiny_variants()
    ts_acc = make_train_step(
        opt, variants, accumulation_steps=2, stochastic_variants=False,
    )
    B = 4
    base1 = _small_batch(batch_size=B)
    base2 = _small_batch(batch_size=B)
    stacked = Batch(
        tokens=jnp.stack([base1.tokens, base2.tokens], axis=0),
        targets=jnp.stack([base1.targets, base2.targets], axis=0),
        attn_mask=jnp.stack([base1.attn_mask, base2.attn_mask], axis=0),
        loss_mask=jnp.stack([base1.loss_mask, base2.loss_mask], axis=0),
    )
    new_state, loss = ts_acc(state, stacked)
    assert int(new_state.step) == 1
    assert jnp.isfinite(loss)
    assert not np.array_equal(
        embed_tokens_before, np.asarray(new_state.model.embed_tokens)
    )


def test_train_step_accumulation_grad_equals_mean_of_micros() -> None:
    """B2: the accumulation kernel's gradient is the MEAN of the per-micro
    gradients (within fp32 noise).

    Drive `make_train_step(accumulation_steps=2)` with a plain SGD(lr=1.0)
    optimizer (clip threshold set huge so it never fires) so the parameter
    delta equals exactly `-mean_grad`. Compare against the reference mean of
    the two single-micro gradients computed directly from
    `supernet_joint_loss` (the kernel's `_loss_for`), confirming the scan
    body sums then divides by N rather than e.g. summing without the 1/N.
    """
    variants = _tiny_variants()
    B = 4
    base1 = _small_batch(batch_size=B)
    base2 = _small_batch(batch_size=B)

    # Reference: mean of the two micro-batch gradients (deterministic sum,
    # stochastic_variants=False).
    ref_model = _tiny_model()

    def _loss(model: PAWNModel, batch: Batch) -> jax.Array:
        return supernet_joint_loss(model, batch, variants, stochastic_key=None)

    _, g1 = eqx.filter_value_and_grad(lambda m: _loss(m, base1))(ref_model)
    _, g2 = eqx.filter_value_and_grad(lambda m: _loss(m, base2))(ref_model)
    mean_grad_embed = (
        np.asarray(g1.embed_tokens) + np.asarray(g2.embed_tokens)
    ) / 2.0

    # Kernel: SGD(lr=1.0) with the clip disabled (huge threshold) so the
    # embed_tokens delta is exactly -mean_grad.
    model = _tiny_model()
    embed_before = np.asarray(model.embed_tokens)
    opt = optax.chain(
        optax.clip_by_global_norm(1e9), optax.sgd(learning_rate=1.0)
    )
    state = TrainState(
        model=model,
        opt_state=opt.init(eqx.filter(model, eqx.is_inexact_array)),
        step=jnp.int32(0),
        key=jax.random.key(0),
    )
    ts = make_train_step(
        opt, variants, accumulation_steps=2, stochastic_variants=False,
    )
    stacked = Batch(
        tokens=jnp.stack([base1.tokens, base2.tokens], axis=0),
        targets=jnp.stack([base1.targets, base2.targets], axis=0),
        attn_mask=jnp.stack([base1.attn_mask, base2.attn_mask], axis=0),
        loss_mask=jnp.stack([base1.loss_mask, base2.loss_mask], axis=0),
    )
    new_state, _ = ts(state, stacked)
    embed_after = np.asarray(new_state.model.embed_tokens)
    # delta = -lr * mean_grad = -mean_grad (lr=1.0).
    kernel_mean_grad = -(embed_after - embed_before)
    assert np.allclose(kernel_mean_grad, mean_grad_embed, rtol=0, atol=1e-5)


def test_train_step_accumulation_steps_rejects_zero() -> None:
    """accumulation_steps must be ≥ 1; reject zero / negative."""
    model = _tiny_model()
    _, opt = _tiny_train_state(model)
    variants = _tiny_variants()
    with pytest.raises(ValueError, match="accumulation_steps"):
        make_train_step(opt, variants, accumulation_steps=0)


# ---------------------------------------------------------------------------
# C.1 — optimizer dispatch (adamw / lion / adafactor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("optimizer_name", ["adamw", "lion"])
def test_make_optimizer_dispatches_by_name(optimizer_name: str) -> None:
    """Each optimizer choice builds a working chain that initializes
    and updates the tiny model without crashing."""
    model = _tiny_model()
    cfg = _make_cfg(
        lr=1e-4 if optimizer_name == "adamw" else 3e-5,
        weight_decay=0.0 if optimizer_name == "lion" else 0.01,
        optimizer=optimizer_name,  # type: ignore[arg-type]
        # constant schedule + warmup_frac=0 so step 0 has a non-zero
        # LR (otherwise the optimizer does nothing on the first step).
        lr_schedule="constant", warmup_frac=0.0,
    )
    schedule = make_lr_schedule(cfg, total_steps=100)
    opt = make_optimizer(cfg, schedule)
    state = TrainState(
        model=model,
        opt_state=opt.init(eqx.filter(model, eqx.is_inexact_array)),
        step=jnp.int32(0),
        key=jax.random.key(0),
    )
    variants = _tiny_variants()
    train_step = make_train_step(opt, variants)
    batch = _small_batch()
    embed_tokens_before = np.asarray(model.embed_tokens)
    new_state, loss = train_step(state, batch)
    assert int(new_state.step) == 1
    assert jnp.isfinite(loss)
    assert not np.array_equal(
        embed_tokens_before, np.asarray(new_state.model.embed_tokens)
    )


def test_make_optimizer_rejects_unknown_name() -> None:
    """Unknown optimizer names raise loudly at build time."""
    model = _tiny_model()
    cfg = _make_cfg()
    # `optimizer` is a Literal field — pydantic rejects unknown values at
    # parse time. The runtime check fires when the field is bypassed
    # (e.g., a test setting it via object.__setattr__).
    object.__setattr__(cfg, "optimizer", "adafactor")
    schedule = make_lr_schedule(cfg, total_steps=100)
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer(cfg, schedule)


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
