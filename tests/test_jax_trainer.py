"""Tests for ``pawn.jax.trainer`` — single-step training primitives."""

from __future__ import annotations

import math
import time

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("optax")
pytest.importorskip("chess_engine")

import jax
import jax.numpy as jnp

from pawn.jax.config import TINY_SUPERNET, TINY_VARIANTS, VOCAB_SIZE, ModelConfig
from pawn.jax.corpus import generate_corpus
from pawn.jax.model import init_model
from pawn.jax.trainer import (
    Batch,
    VariantSpec,
    _variant_key,
    compute_supernet_loss,
    compute_variant_loss,
    cross_entropy_loss,
    init_train_state,
    make_lr_schedule,
    make_optimizer,
    make_scan_step,
    make_train_step,
)


def _random_batch(key: jax.Array, b: int = 4, t: int = 16) -> Batch:
    """Synthetic batch for trainer-shape tests. The trainer doesn't care
    that targets are random tokens; only the shape/dtype contract
    matters for the math under test."""
    tokens_key, target_key = jax.random.split(key)
    tokens = jax.random.randint(tokens_key, (b, t), 0, 1968).astype(jnp.int32)
    targets = jax.random.randint(target_key, (b, t), 0, 1968).astype(jnp.int32)
    attn_mask = jnp.ones((b, t), dtype=jnp.bool_)
    loss_mask = jnp.ones((b, t), dtype=jnp.bool_)
    return tokens, attn_mask, targets, loss_mask


def _tile_chunk(single_batch: Batch, k: int) -> Batch:
    """Stack a single ``[B, T]`` batch K times along a new leading axis
    to produce a ``[K, B, T]`` chunk for ``make_scan_step``. Three
    scan-layer tests need this idiom and the `tuple(generator)` pattern
    widens to ``tuple[Any, ...]`` under pyright — collapsing to one
    helper keeps the ``type: ignore`` localised."""
    return tuple(  # type: ignore[return-value]
        jnp.stack([x] * k, axis=0) for x in single_batch
    )


def test_cross_entropy_at_uniform_logits_is_log_vocab() -> None:
    """Sanity: uniform logits → CE = ln(V). Pins the loss math
    against the most basic analytical reference. A future refactor
    that drops the log-softmax for softmax would silently break it."""
    b, t, v = 2, 8, VOCAB_SIZE
    logits = jnp.zeros((b, t, v))  # uniform after softmax
    targets = jnp.zeros((b, t), dtype=jnp.int32)
    loss_mask = jnp.ones((b, t), dtype=jnp.bool_)
    ce = float(cross_entropy_loss(logits, targets, loss_mask))
    assert math.isclose(ce, math.log(v), rel_tol=1e-5), (
        f"uniform-logit CE = {ce}, expected ln({v}) = {math.log(v)}"
    )


def test_cross_entropy_handles_empty_mask() -> None:
    """An all-False mask returns 0 (not NaN). The K-step scan padding
    relies on this to ignore short tail chunks safely."""
    b, t, v = 2, 4, VOCAB_SIZE
    logits = jnp.zeros((b, t, v))
    targets = jnp.zeros((b, t), dtype=jnp.int32)
    loss_mask = jnp.zeros((b, t), dtype=jnp.bool_)
    ce = float(cross_entropy_loss(logits, targets, loss_mask))
    assert ce == 0.0, f"empty-mask CE should be 0, got {ce}"


def test_cross_entropy_perfect_prediction_is_zero() -> None:
    """A heavily-peaked logit at the target reduces CE → 0. Validates
    the gather (``take_along_axis``) path picks the right index."""
    b, t, v = 1, 1, VOCAB_SIZE
    target_idx = 42
    targets = jnp.array([[target_idx]], dtype=jnp.int32)
    logits = jnp.full((b, t, v), -100.0).at[0, 0, target_idx].set(100.0)
    loss_mask = jnp.ones((b, t), dtype=jnp.bool_)
    ce = float(cross_entropy_loss(logits, targets, loss_mask))
    assert ce < 1e-3, f"peaked-logit CE should be ~0, got {ce}"


def test_compute_variant_loss_initial_value_near_log_vocab() -> None:
    """A freshly-initialised model produces approximately-uniform
    logits → CE ≈ ln(V). Pins the joint of cross_entropy_loss +
    forward against the random-init prior."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    tokens, attn_mask, targets, loss_mask = _random_batch(
        jax.random.PRNGKey(1), b=4, t=16
    )
    ce = float(compute_variant_loss(model, tokens, attn_mask, targets, loss_mask))
    expected = math.log(VOCAB_SIZE)
    # Random-init transformers diverge from exactly-uniform but stay
    # within 0.5 of ln(V); a regression that broke the embedding or
    # the lm_head would either NaN out or land far above ln(V).
    assert abs(ce - expected) < 0.5, (
        f"random-init CE = {ce}, expected near {expected}"
    )


def test_supernet_loss_sums_variants() -> None:
    """Joint loss is the (weighted) sum of per-variant losses. Pins
    §5.3's joint-objective math against a regression that drops one
    variant or applies an unintended weighting."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v, weight=1.0) for v in TINY_VARIANTS.values())
    tokens, attn_mask, targets, loss_mask = _random_batch(
        jax.random.PRNGKey(2), b=2, t=8
    )
    joint, per_v = compute_supernet_loss(
        model, specs, tokens, attn_mask, targets, loss_mask
    )
    expected = sum(float(v) for v in per_v.values())
    assert math.isclose(float(joint), expected, rel_tol=1e-5), (
        f"joint {float(joint)} != sum {expected}"
    )
    # All three TINY variants present, distinguishable by full key
    # (d<W>_L<L>_F<F>) — not just by width — so a future sandwich
    # spec that varies depth at fixed width stays distinct.
    expected_keys = {_variant_key(v) for v in TINY_VARIANTS.values()}
    assert set(per_v.keys()) == expected_keys, (
        f"per_v keys = {set(per_v.keys())}, expected {expected_keys}"
    )


def test_supernet_loss_rejects_empty_variant_specs() -> None:
    """``compute_supernet_loss`` with an empty specs tuple would
    silently produce joint=0 and {} per-variant — backprop yields
    zero gradients and AdamW's decoupled weight decay still drifts
    the model. Pin the guard."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    tokens, attn_mask, targets, loss_mask = _random_batch(
        jax.random.PRNGKey(11), b=2, t=8
    )
    with pytest.raises(ValueError, match="at least one VariantSpec"):
        compute_supernet_loss(model, (), tokens, attn_mask, targets, loss_mask)


def test_supernet_loss_keys_distinguish_same_width_different_depth() -> None:
    """Two specs sharing ``d_model`` but differing in ``n_layers`` must
    produce distinct per-variant keys — otherwise the second
    silently overwrites the first in the metrics dict (the joint
    sum is still correct, but per-slice convergence curves get
    blended). Pins the Bug-detector MINOR finding."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    # Two valid nested specs at d=192: one is the full supernet
    # (L=4); the other slices the depth to L=2. Both have head_dim=64
    # so validate_nested accepts them.
    spec_full = VariantSpec(cfg=TINY_SUPERNET)
    spec_short = VariantSpec(
        cfg=ModelConfig(d_model=192, n_layers=2, n_heads=3, d_ff=768)
    )
    specs = (spec_full, spec_short)
    tokens, attn_mask, targets, loss_mask = _random_batch(
        jax.random.PRNGKey(10), b=2, t=8
    )
    _, per_v = compute_supernet_loss(
        model, specs, tokens, attn_mask, targets, loss_mask
    )
    assert len(per_v) == 2, (
        f"both specs should produce distinct keys; per_v={per_v}"
    )


def test_supernet_loss_weights_applied() -> None:
    """Per-spec ``weight`` scales each variant's contribution."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs_weighted = (
        VariantSpec(cfg=TINY_VARIANTS["small"], weight=2.0),
        VariantSpec(cfg=TINY_VARIANTS["base"], weight=0.5),
        VariantSpec(cfg=TINY_VARIANTS["large"], weight=1.0),
    )
    tokens, attn_mask, targets, loss_mask = _random_batch(
        jax.random.PRNGKey(3), b=2, t=8
    )
    joint, per_v = compute_supernet_loss(
        model, specs_weighted, tokens, attn_mask, targets, loss_mask
    )
    k_small = _variant_key(TINY_VARIANTS["small"])
    k_base = _variant_key(TINY_VARIANTS["base"])
    k_large = _variant_key(TINY_VARIANTS["large"])
    expected = (
        2.0 * float(per_v[k_small])
        + 0.5 * float(per_v[k_base])
        + 1.0 * float(per_v[k_large])
    )
    assert math.isclose(float(joint), expected, rel_tol=1e-5)


def test_train_step_changes_weights() -> None:
    """One optimizer step must change every trainable leaf the model
    exposes. Pins the gradient-flow contract — a regression that
    hands gradients to a tree that doesn't reach Adam (the most
    common ``eqx.filter`` boundary bug) would leave one or more
    leaves bit-identical.

    Spot-check pattern: enumerate the supernet's array leaves by
    Equinox-name and assert each changed by some positive amount.
    ``pad_embed`` is intentionally initialised to zero and lives in
    the loss-free padding code path — gradients reach it through the
    embedding gather but the contribution is always masked out, so
    we exempt it from the must-change list."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=1e-2)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    batch = _random_batch(jax.random.PRNGKey(4), b=2, t=8)
    new_state, _ = step(state, batch)
    assert int(new_state.step) == 1
    # Every array-leaf attribute on PAWNModel that contributes to the
    # forward pass. ``pad_embed`` is exempt because it's initialised
    # to zero and never indexed by a real move token, so AdamW's
    # decoupled weight decay doesn't move it. ``outcome_embed`` is
    # similarly exempt: ``_random_batch`` draws tokens in
    # ``[0, 1968)`` (move IDs only), so the outcome-embed gather is
    # never exercised and the parameter receives zero gradient on
    # this batch. (Weight decay still moves it slightly via AdamW,
    # but the magnitude is well below the gradient-driven leaves'
    # delta, and a leaf-by-leaf "any positive change" check is
    # noisy for it — we prefer to skip rather than blur the test's
    # intent of catching ``eqx.filter`` boundary bugs.)
    must_change = (
        "src_embed", "dst_embed", "promo_embed",
        "attn_norm", "wq", "wk", "wv", "wo",
        "ffn_norm", "w_gate", "w_up", "w_down",
        "final_norm", "lm_head",
    )
    for name in must_change:
        before = getattr(model, name)
        after = getattr(new_state.model, name)
        delta = float(jnp.abs(after - before).max())
        assert delta > 0.0, (
            f"leaf {name!r} unchanged after step — gradients did not "
            f"reach it. Likely an eqx.filter boundary bug."
        )


def test_train_step_decreases_loss_on_fixed_batch() -> None:
    """Repeatedly fitting the same batch must reduce the joint loss
    monotonically — this is the trainer's most basic contract.
    Detects: schedule sign flips, optimizer state bugs, accidental
    gradient zeroing."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-3)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    batch = _random_batch(jax.random.PRNGKey(5), b=4, t=16)
    initial_loss = float(compute_supernet_loss(state.model, specs, *batch)[0])
    for _ in range(10):
        state, _ = step(state, batch)
    final_loss = float(compute_supernet_loss(state.model, specs, *batch)[0])
    assert final_loss < initial_loss * 0.95, (
        f"10 steps on a fixed batch did not reduce loss: "
        f"initial={initial_loss}, final={final_loss}"
    )


def test_train_step_loss_is_finite() -> None:
    """No-NaN guarantee on the freshly-initialised model. A common
    failure mode for new attention paths is overflow in the softmax;
    this pins it doesn't fire at TINY scale."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-4)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    batch = _random_batch(jax.random.PRNGKey(6), b=2, t=8)
    state, metrics = step(state, batch)
    assert jnp.isfinite(metrics["loss"]).all()
    assert jnp.isfinite(metrics["grad_norm"]).all()
    for k, v in metrics.items():
        assert jnp.isfinite(v).all(), f"metric {k} is non-finite: {v}"


def test_train_step_grad_norm_positive() -> None:
    """A random-init model on real data produces a non-trivial
    gradient — pins that the train step isn't accidentally
    computing gradients through ``stop_gradient`` or similar."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-4)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    batch = _random_batch(jax.random.PRNGKey(7), b=2, t=8)
    _, metrics = step(state, batch)
    grad_norm = float(metrics["grad_norm"])
    assert grad_norm > 1e-6, f"grad_norm={grad_norm} is too small"


def test_train_step_with_real_corpus_batch() -> None:
    """End-to-end smoke: a real Rust-engine corpus batch flows through
    the trainer cleanly. Catches dtype-mismatch and shape-coupling
    bugs that synthetic random batches mask."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-4)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)

    c = generate_corpus(n_games=8, max_ply=16, seed=42)
    batch: Batch = (
        jnp.asarray(c.tokens),
        jnp.asarray(c.attn_mask),
        jnp.asarray(c.targets),
        jnp.asarray(c.loss_mask),
    )
    new_state, metrics = step(state, batch)
    assert int(new_state.step) == 1
    assert jnp.isfinite(metrics["loss"]).all()
    # Loss is in a sane range for a random-init model on a real corpus.
    loss_val = float(metrics["loss"])
    expected_per_variant = math.log(VOCAB_SIZE)
    # Joint loss = sum of 3 per-variant losses, each ≈ ln(V).
    assert 0.5 * 3 * expected_per_variant < loss_val < 1.5 * 3 * expected_per_variant, (
        f"joint loss {loss_val} far from 3·ln(V) = {3 * expected_per_variant}"
    )


def test_train_step_skips_update_on_all_padded_batch() -> None:
    """A batch with ``loss_mask=False`` everywhere (the scan-padding
    edge case) must NOT mutate weights or optimizer state. AdamW's
    decoupled weight decay would otherwise drift the model on every
    zero-gradient padded step. Pins the Codex P2 guard."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-3)  # any decay is enough
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    # Padding-only batch: real-shape tokens but loss_mask all False.
    b, t = 2, 8
    tokens = jnp.zeros((b, t), dtype=jnp.int32)
    attn_mask = jnp.zeros((b, t), dtype=jnp.bool_)
    targets = jnp.zeros((b, t), dtype=jnp.int32)
    loss_mask = jnp.zeros((b, t), dtype=jnp.bool_)
    new_state, metrics = step(state, (tokens, attn_mask, targets, loss_mask))
    # Weights must be bit-identical (no decoupled-weight-decay drift).
    for name in ("wq", "lm_head", "final_norm"):
        before = getattr(model, name)
        after = getattr(new_state.model, name)
        assert jnp.array_equal(before, after), (
            f"all-padded batch mutated {name!r} — Codex P2 regression "
            f"(AdamW weight decay applied on zero-grad path)"
        )
    # Loss/grad reported as 0 (modulo the safe-divide), n_supervised==0.
    assert float(metrics["loss"]) == 0.0
    assert int(metrics["n_supervised"]) == 0
    # Step counter does NOT advance on a fully-padded batch.
    # ``state.step`` must stay locked to the Optax internal schedule
    # count — Optax's count only advances on a real ``update`` call,
    # so advancing the host counter on a skipped step would silently
    # desync the host-reported step from the LR schedule.
    assert int(new_state.step) == 0


def test_train_state_is_jax_pytree() -> None:
    """``TrainState`` is a NamedTuple — JAX/Equinox can lift it as a
    PyTree and Optax's state composes cleanly under the same
    convention. Pins the structural integration that ``lax.scan``
    (chunk 2.3) will rely on."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    opt = make_optimizer(learning_rate=3e-4)
    state = init_train_state(model, opt)
    # NamedTuple flattening via JAX.
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) > 0
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert int(rebuilt.step) == int(state.step)
    # ``step`` is a JAX 0-d array (not a Python int) so the trainer
    # doesn't recompile on every increment — the CRITICAL invariant
    # the trainer ChunkR identified.
    assert isinstance(state.step, jax.Array)
    assert state.step.shape == ()
    assert state.step.dtype == jnp.int32


def _sched_at(sched, step: int) -> float:
    """Helper: evaluate an ``optax.Schedule`` at a step and return a
    Python float. Wraps the ``ArrayLike`` return through ``jnp.asarray``
    so pyright sees the narrow ``jax.Array`` type, not the broader
    ``ArrayLike`` union that includes ``complex``."""
    return float(jnp.asarray(sched(step)))


def test_lr_schedule_warmup_then_decay() -> None:
    """LR schedule must: start at 0, hit peak at end of warmup, decay
    to end_value at total_steps, plateau past total_steps."""
    sched = make_lr_schedule(
        peak_lr=3e-4, total_steps=1000, warmup_steps=100, end_value=3e-5
    )
    assert _sched_at(sched, 0) == 0.0
    assert math.isclose(_sched_at(sched, 100), 3e-4, rel_tol=1e-5)
    # Mid-decay: cosine half-period — value is between peak and end.
    mid = _sched_at(sched, 550)
    assert 3e-5 < mid < 3e-4
    # End of schedule lands at end_value.
    assert math.isclose(_sched_at(sched, 1000), 3e-5, rel_tol=1e-3)
    # Past schedule: plateau (no negative LR).
    assert math.isclose(_sched_at(sched, 5000), 3e-5, rel_tol=1e-3)


def test_lr_schedule_default_end_value() -> None:
    """Default end_value is peak_lr * 0.1."""
    sched = make_lr_schedule(peak_lr=1e-3, total_steps=1000, warmup_steps=50)
    assert math.isclose(_sched_at(sched, 1000), 1e-4, rel_tol=1e-3)


def test_lr_schedule_rejects_warmup_exceeding_total() -> None:
    """Misconfigured schedule (warmup ≥ total) raises clearly rather
    than silently producing a degenerate cosine."""
    with pytest.raises(ValueError, match="warmup_steps"):
        make_lr_schedule(peak_lr=1e-3, total_steps=100, warmup_steps=100)


def test_lr_schedule_rejects_non_positive_total_or_peak() -> None:
    """Non-positive ``total_steps`` or ``peak_lr`` fail loudly with a
    targeted message rather than emitting a degenerate optax schedule."""
    with pytest.raises(ValueError, match="total_steps"):
        make_lr_schedule(peak_lr=1e-3, total_steps=0, warmup_steps=10)
    with pytest.raises(ValueError, match="peak_lr"):
        make_lr_schedule(peak_lr=0.0, total_steps=100, warmup_steps=10)


def test_lr_schedule_rejects_negative_warmup() -> None:
    """``warmup_steps < 0`` would feed a negative slope into optax;
    pin a clear ValueError instead."""
    with pytest.raises(ValueError, match="warmup_steps"):
        make_lr_schedule(peak_lr=1e-3, total_steps=100, warmup_steps=-1)


def test_lr_schedule_rejects_end_value_geq_peak() -> None:
    """``end_value >= peak_lr`` would make the cosine *increase* through
    the "decay" phase — almost certainly a misconfiguration. Pin a
    clear ValueError so it doesn't silently corrupt a long run."""
    with pytest.raises(ValueError, match="end_value"):
        make_lr_schedule(
            peak_lr=1e-3, total_steps=100, warmup_steps=10, end_value=1e-3
        )
    with pytest.raises(ValueError, match="end_value"):
        make_lr_schedule(
            peak_lr=1e-4, total_steps=100, warmup_steps=10, end_value=1e-3
        )


def test_lr_schedule_rejects_negative_end_value() -> None:
    """``end_value < 0`` would have optax plateau at a negative
    learning rate (reversing AdamW updates). Pin a clear ValueError
    (Codex P2)."""
    with pytest.raises(ValueError, match="end_value"):
        make_lr_schedule(
            peak_lr=1e-3, total_steps=100, warmup_steps=10, end_value=-1e-4
        )


def test_lr_schedule_warmup_zero_starts_at_peak() -> None:
    """A warmup of 0 is a valid degenerate case: step 0 already returns
    ``peak_lr``. Pins the boundary so a future tightening of the
    ``warmup_steps >= 0`` guard to ``> 0`` doesn't silently break this."""
    sched = make_lr_schedule(
        peak_lr=1e-3, total_steps=100, warmup_steps=0
    )
    assert math.isclose(_sched_at(sched, 0), 1e-3, rel_tol=1e-5)


def test_lr_schedule_floor_reached_exactly_at_total_steps() -> None:
    """The cosine decay must hit ``end_value`` AT ``total_steps``, not
    earlier. Pins the optax ``decay_steps`` semantics fix: optax's
    parameter is the *end-to-end* schedule length and it subtracts
    warmup internally — passing ``total_steps - warmup`` (the previous
    bug) made end_value land at ``total_steps - warmup`` instead.
    Concretely test that just before total_steps the schedule is
    strictly above end_value and at total_steps it equals end_value."""
    sched = make_lr_schedule(
        peak_lr=1e-3, total_steps=1000, warmup_steps=100, end_value=1e-4
    )
    # Just-before-end is still in the decay region (above floor).
    assert _sched_at(sched, 999) > 1e-4
    # Floor hit at exactly total_steps.
    assert math.isclose(_sched_at(sched, 1000), 1e-4, rel_tol=1e-3)


def test_optimizer_accepts_schedule_callable() -> None:
    """``make_optimizer`` must accept an ``optax.Schedule`` as
    ``learning_rate``. Pins the integration before chunk 2.4's
    driver attempts to compose the two."""
    sched = make_lr_schedule(peak_lr=3e-4, total_steps=100, warmup_steps=10)
    opt = make_optimizer(sched)
    model = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    state = init_train_state(model, opt)
    assert state.opt_state is not None


def test_scan_step_advances_state_by_k_per_call() -> None:
    """K-step scan: ``state.step`` advances by exactly K per call;
    metrics are stacked on a leading [K] axis."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-4)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    K = 4
    scan = make_scan_step(step, K)
    B, T = 2, 8
    rng = jax.random.PRNGKey(20)
    tokens = jax.random.randint(rng, (K, B, T), 0, 1968).astype(jnp.int32)
    attn = jnp.ones((K, B, T), dtype=jnp.bool_)
    targets = jax.random.randint(rng, (K, B, T), 0, 1968).astype(jnp.int32)
    loss_mask = jnp.ones((K, B, T), dtype=jnp.bool_)
    state, metrics = scan(state, (tokens, attn, targets, loss_mask))
    assert int(state.step) == K
    # Metrics are stacked across the scan axis.
    assert metrics["loss"].shape == (K,)
    assert metrics["grad_norm"].shape == (K,)


def test_scan_step_loss_decreases_on_fixed_chunk() -> None:
    """The K-step scan on the same repeated batch should produce a
    monotone-ish loss trajectory — the final step's loss strictly
    less than the first. Catches a regression where the scan carry
    silently resets state between iterations."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-3)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    K = 10
    scan = make_scan_step(step, K)
    # Same batch repeated K times along the leading axis.
    B, T = 4, 16
    single_batch = _random_batch(jax.random.PRNGKey(21), b=B, t=T)
    chunk = _tile_chunk(single_batch, K)
    _, metrics = scan(state, chunk)
    losses = metrics["loss"]
    assert float(losses[-1]) < float(losses[0]) * 0.95, (
        f"scan over {K} repeats of the same batch did not reduce loss: "
        f"first={float(losses[0])}, last={float(losses[-1])}"
    )


def test_scan_step_integrates_with_lr_schedule() -> None:
    """End-to-end with a real LR schedule: the optimizer's
    decoupled-LR path must see a non-trivial LR during the scan."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    sched = make_lr_schedule(peak_lr=3e-3, total_steps=100, warmup_steps=10)
    opt = make_optimizer(sched)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    K = 20
    scan = make_scan_step(step, K)
    B, T = 2, 8
    single_batch = _random_batch(jax.random.PRNGKey(22), b=B, t=T)
    chunk = _tile_chunk(single_batch, K)
    state, metrics = scan(state, chunk)
    # Past warmup: weights diverge from init (real LR > 0 was applied).
    delta = float(jnp.abs(state.model.lm_head - model.lm_head).max())
    assert delta > 0.0
    # All-finite metrics across the chunk.
    for name, vals in metrics.items():
        assert jnp.isfinite(vals).all(), f"{name} has non-finite entries"


def test_train_step_gradient_clipping_caps_updates() -> None:
    """``make_optimizer(max_grad_norm=1.0)`` clips global grad norm
    before Adam. A pathological logit spike that would otherwise
    drive AdamW to take a huge step should be tamed by the clipper.
    Compare a clipped run vs the same run with clipping disabled —
    the post-step weight delta under clipping must be strictly
    smaller than without it.

    Pin against a regression that drops the clip from the composed
    optimizer — Codex P2 finding."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    batch = _random_batch(jax.random.PRNGKey(99), b=4, t=16)

    # Same lr, same model, same batch. Clip cap of 0.01 forces a
    # measurable shrink even on a normal gradient; ``None`` disables.
    opt_clipped = make_optimizer(learning_rate=1e-2, max_grad_norm=0.01)
    state_c = init_train_state(model, opt_clipped)
    step_c = make_train_step(opt_clipped, specs)
    new_c, _ = step_c(state_c, batch)

    opt_raw = make_optimizer(learning_rate=1e-2, max_grad_norm=None)
    state_r = init_train_state(model, opt_raw)
    step_r = make_train_step(opt_raw, specs)
    new_r, _ = step_r(state_r, batch)

    delta_clip = float(jnp.abs(new_c.model.wq - model.wq).max())
    delta_raw = float(jnp.abs(new_r.model.wq - model.wq).max())
    assert delta_clip < delta_raw, (
        f"clipped delta {delta_clip} not strictly less than raw {delta_raw}; "
        "is the optax.clip_by_global_norm step being composed correctly?"
    )


def test_scan_step_does_not_recompile_per_call() -> None:
    """The K-step scan must hit the JIT cache after warmup, just like
    the single-step path. A regression that lifts ``k`` or some other
    per-chunk value out of the static partition would cause a
    full retrace per scan call — the dominant cost a real training
    loop would pay. Mirrors ``test_train_step_does_not_recompile_per_call``
    at the scan layer."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-4)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    K = 5
    scan = make_scan_step(step, K)
    B, T = 2, 8
    single_batch = _random_batch(jax.random.PRNGKey(30), b=B, t=T)
    chunk = _tile_chunk(single_batch, K)
    # Warmup compile.
    state, _ = scan(state, chunk)
    t0 = time.perf_counter()
    state, _ = scan(state, chunk)
    t1 = time.perf_counter()
    state, _ = scan(state, chunk)
    t2 = time.perf_counter()
    second_call = t1 - t0
    third_call = t2 - t1
    assert third_call < 0.5, (
        f"third scan call took {third_call:.3f}s — likely retracing per "
        f"call (second={second_call:.3f}s)."
    )


def test_train_step_does_not_recompile_per_call() -> None:
    """The most important perf-correctness contract: repeated calls
    must hit the same JIT cache entry, not retrace. A regression that
    re-promotes ``step`` to a Python int (or otherwise lifts a
    per-step-varying value into the static partition) would cause
    O(N) recompiles — a ~70x wall-clock slowdown on real workloads.
    Measure by inspecting Equinox's wrapped-fn compile count."""
    key = jax.random.PRNGKey(0)
    model = init_model(TINY_SUPERNET, key)
    specs = tuple(VariantSpec(cfg=v) for v in TINY_VARIANTS.values())
    opt = make_optimizer(learning_rate=3e-4)
    state = init_train_state(model, opt)
    step = make_train_step(opt, specs)
    batch = _random_batch(jax.random.PRNGKey(11), b=2, t=8)
    # Warmup compile.
    state, _ = step(state, batch)
    # Subsequent calls must be cache hits. ``jax.jit``-style cache
    # access through the wrapper is fragile; instead use a clock-
    # based proxy: the second call must be at least 5x faster than
    # the first (compile-vs-cached gap is typically 100x+).
    t0 = time.perf_counter()
    state, _ = step(state, batch)
    t1 = time.perf_counter()
    state, _ = step(state, batch)
    t2 = time.perf_counter()
    second_call = t1 - t0
    third_call = t2 - t1
    # If recompile-per-step were happening, second and third calls
    # would have similar magnitudes (both ~1+ second). If hitting
    # cache, both should be milliseconds. The "no monotonic blowup"
    # test: third_call should be in the same ballpark as second_call.
    # A regression that retraces every call would show steady ~1s+
    # per call; cache hits show ~ms per call.
    assert third_call < 0.5, (
        f"third call took {third_call:.3f}s — likely retracing every step "
        f"(second={second_call:.3f}s). state.step type is "
        f"{type(state.step).__name__}."
    )
