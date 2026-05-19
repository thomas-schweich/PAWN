"""Tests for ``pawn.jax.model`` — supernet, variant slicing, forward."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.jax.config import SUPERNET, TOY, VARIANTS, ModelConfig, validate_nested
from pawn.jax.model import PAWNModel, init_model, sliced

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
    with pytest.raises(ValueError, match="head_dim"):
        validate_nested(bad, SUPERNET)


def test_validate_nested_rejects_too_wide() -> None:
    too_wide = ModelConfig(d_model=768, n_layers=8, n_heads=12, d_ff=2048)
    with pytest.raises(ValueError, match="d_model"):
        validate_nested(too_wide, SUPERNET)


def test_validate_nested_rejects_rope_base_drift() -> None:
    drifted = ModelConfig(
        d_model=256, n_layers=8, n_heads=4, d_ff=1024, rope_base=50000.0
    )
    with pytest.raises(ValueError, match="rope_base"):
        validate_nested(drifted, SUPERNET)


def test_modelconfig_post_init_divisibility() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        ModelConfig(d_model=257, n_layers=2, n_heads=4, d_ff=256)
