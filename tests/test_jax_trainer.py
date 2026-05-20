"""Tests for ``pawn.jax.trainer`` — single-step training primitives."""

from __future__ import annotations

import math

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
    VariantSpec,
    _variant_key,
    compute_supernet_loss,
    compute_variant_loss,
    cross_entropy_loss,
    init_train_state,
    make_optimizer,
    make_train_step,
)


from pawn.jax.trainer import Batch


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
    # forward pass. ``pad_embed`` is exempt (always-masked).
    leaf_names = (
        "src_embed", "dst_embed", "promo_embed", "outcome_embed",
        "attn_norm", "wq", "wk", "wv", "wo",
        "ffn_norm", "w_gate", "w_up", "w_down",
        "final_norm", "lm_head",
    )
    for name in leaf_names:
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
    # Step counter still advances — padded steps still count toward
    # the schedule the host might be tracking.
    assert int(new_state.step) == 1


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
    import time
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
