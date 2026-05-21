"""Tests for ``pawn.model`` — supernet, variant slicing, forward."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("chess_engine")

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.config import SUPERNET, TOY, VARIANTS, ModelConfig, validate_nested
from pawn.model import PAWNModel, init_model, sliced

pytestmark = pytest.mark.unit


def _n_params(model: PAWNModel) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(x.size for x in leaves))


def _toy_batch(b: int = 2, t: int = 16) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(0)
    tokens = jax.random.randint(key, (b, t), 0, 1968)
    mask = jnp.ones((b, t), dtype=bool)
    return tokens, mask


def test_supernet_forward_shape_and_finite() -> None:
    model = init_model(SUPERNET, jax.random.PRNGKey(0))
    tokens, mask = _toy_batch()
    out = eqx.filter_jit(lambda m, t, a: m(t, a))(model, tokens, mask)
    assert out.shape == (2, 16, 1980)
    assert bool(jnp.all(jnp.isfinite(out)))


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_slice_matches_fresh_init_param_count(name: str) -> None:
    cfg = VARIANTS[name]
    supernet = init_model(SUPERNET, jax.random.PRNGKey(0))
    variant = sliced(supernet, cfg)
    fresh = init_model(cfg, jax.random.PRNGKey(1))
    assert _n_params(variant) == _n_params(fresh)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_slice_forward(name: str) -> None:
    cfg = VARIANTS[name]
    supernet = init_model(SUPERNET, jax.random.PRNGKey(0))
    variant = sliced(supernet, cfg)
    tokens, mask = _toy_batch()
    out = eqx.filter_jit(lambda m, t, a: m(t, a))(variant, tokens, mask)
    assert out.shape == (2, 16, 1980)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_large_slice_is_identity() -> None:
    """Slicing the supernet at large's dimensions must equal the supernet."""
    supernet = init_model(SUPERNET, jax.random.PRNGKey(0))
    large = sliced(supernet, VARIANTS["large"])
    tokens, mask = _toy_batch()
    fwd = eqx.filter_jit(lambda m, t, a: m(t, a))
    assert bool(jnp.array_equal(fwd(large, tokens, mask), fwd(supernet, tokens, mask)))


def test_toy_forward() -> None:
    model = init_model(TOY, jax.random.PRNGKey(0))
    tokens, mask = _toy_batch()
    out = eqx.filter_jit(lambda m, t, a: m(t, a))(model, tokens, mask)
    assert out.shape == (2, 16, 1980)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_pad_and_outcome_embedding_paths() -> None:
    """PAD and outcome tokens must produce finite, non-NaN embeddings."""
    model = init_model(TOY, jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (1, 8), 0, 1968)
    tokens = tokens.at[0, -1].set(1968)        # PAD
    tokens = tokens.at[0, 0].set(1969 + 3)     # outcome
    mask = jnp.ones((1, 8), dtype=bool)
    out = eqx.filter_jit(lambda m, t, a: m(t, a))(model, tokens, mask)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_max_seq_len_guard_raises() -> None:
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=8)
    model = init_model(cfg, jax.random.PRNGKey(0))
    tokens = jnp.zeros((1, 16), dtype=jnp.int32)
    mask = jnp.ones((1, 16), dtype=bool)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        eqx.filter_jit(lambda m, t, a: m(t, a))(model, tokens, mask)


def test_validate_nested_accepts_each_variant() -> None:
    for cfg in VARIANTS.values():
        validate_nested(cfg, SUPERNET)


def test_validate_nested_rejects_head_dim_mismatch() -> None:
    # SUPERNET head_dim = 64; build a variant with head_dim 32.
    bad = ModelConfig(d_model=256, n_layers=8, n_heads=8, d_ff=1024)
    with pytest.raises(ValueError, match=r"head_dim=\d+ != supernet"):
        validate_nested(bad, SUPERNET)


def test_validate_nested_rejects_too_wide() -> None:
    too_wide = ModelConfig(d_model=768, n_layers=8, n_heads=12, d_ff=2048)
    with pytest.raises(ValueError, match=r"d_model=\d+ exceeds supernet"):
        validate_nested(too_wide, SUPERNET)


def test_validate_nested_rejects_rope_base_drift() -> None:
    drifted = ModelConfig(
        d_model=256, n_layers=8, n_heads=4, d_ff=1024, rope_base=50000.0
    )
    with pytest.raises(ValueError, match=r"rope_base=.* must equal supernet"):
        validate_nested(drifted, SUPERNET)


def test_modelconfig_post_init_divisibility() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        ModelConfig(d_model=257, n_layers=2, n_heads=4, d_ff=256)


def test_modelconfig_post_init_rejects_odd_head_dim() -> None:
    # d=60, n_heads=4 -> head_dim=15 (odd); RoPE would mis-rotate the last channel.
    with pytest.raises(ValueError, match="head_dim=15 must be even"):
        ModelConfig(d_model=60, n_layers=2, n_heads=4, d_ff=256)


def test_grad_through_scan_remat_is_finite() -> None:
    """``jax.checkpoint(run_layer)`` (the Phase-2 memory-saving wrap on the
    scan body) must compose with autodiff and produce finite gradients —
    including a batch element with an all-padding mask, which exercises
    the all-masked-query softmax-NaN-avoidance path under grad."""
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256)
    model = init_model(cfg, jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, 1968)
    # Two batch elements: one all-real, one all-padding (the degenerate
    # case that previously would have NaN'd softmax forward and poisoned
    # gradients backward).
    mask = jnp.ones((2, 8), dtype=bool).at[1].set(False)

    def loss(m: PAWNModel, tk: jax.Array, am: jax.Array) -> jax.Array:
        return m(tk, am).sum()

    grads = eqx.filter_grad(loss)(model, tokens, mask)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves, "expected at least one gradient leaf"
    for leaf in leaves:
        assert bool(jnp.all(jnp.isfinite(leaf))), "non-finite gradient"


def test_partial_padding_mask_path() -> None:
    """Real-world inputs have trailing PAD; verify logits stay finite and the
    real-token region is independent of the padded tail (causal + key-masking
    invariant). Previously every test passed an all-True mask."""
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256)
    model = init_model(cfg, jax.random.PRNGKey(0))
    fwd = eqx.filter_jit(lambda m, t, a: m(t, a))

    real_tokens = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, 1968)
    pad_mask = jnp.ones((1, 16), dtype=bool).at[0, 12:].set(False)

    # Tail beyond position 11 is masked; tokens beyond 11 must not influence
    # logits at positions <= 11. Compare against a run with different tail tokens.
    tail_swapped = real_tokens.at[0, 12:].set(0)
    out_a = fwd(model, real_tokens, pad_mask)
    out_b = fwd(model, tail_swapped, pad_mask)
    assert bool(jnp.all(jnp.isfinite(out_a)))
    assert bool(jnp.array_equal(out_a[:, :12], out_b[:, :12])), (
        "padded tail tokens leak into earlier positions"
    )
