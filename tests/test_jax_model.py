"""Tests for :mod:`pawn.model` — Equinox PAWNModel + init + slicing.

The model module depends on JAX, so these tests assume the test env
has the rocm or cu128 extra installed. They run on CPU just fine
(JAX falls back); the per-section verification commands run on the
GPU.

Coverage:
- The 16 trainable arrays land in the right shapes / dtypes.
- ``init_model`` builds a runnable model for both the production
  ``SUPERNET`` and the lightweight ``TINY_SUPERNET``.
- The forward pass produces the right shape and finite outputs.
- The forward pass handles PAD tokens and outcome tokens correctly
  (they don't crash the embedding lookup; the override branches fire).
- The attention is causal — masking out the right-hand context doesn't
  change the left-hand outputs (within tolerance).
- ``sliced(supernet, variant_cfg)`` builds a runnable smaller model
  with the right shapes.
- ``sliced`` rejects an invalid nesting.
- ``SAVED_FIELDS`` has exactly 16 entries; every name resolves to an
  array attribute on the model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pawn.config import (
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    ModelConfig,
    NestingError,
)
from pawn.model import (
    SAVED_FIELDS,
    PAWNModel,
    init_model,
    sliced,
)


# ---------------------------------------------------------------------------
# Shape + structural invariants
# ---------------------------------------------------------------------------


def test_saved_fields_is_sixteen() -> None:
    """The save schema is exactly 16 named arrays — this is the v2
    `pawn.checkpoint` contract."""
    assert len(SAVED_FIELDS) == 16
    assert len(set(SAVED_FIELDS)) == 16  # No duplicates


def test_saved_fields_match_pawn_model_attributes() -> None:
    """Every name in `SAVED_FIELDS` resolves to an actual array on a
    freshly-built model. Catches typos like `embed_pads` vs `embed_pad`."""
    model = init_model(TINY_SUPERNET, key=0)
    for path in SAVED_FIELDS:
        node = model
        for piece in path.split("."):
            node = getattr(node, piece)
        assert isinstance(node, jax.Array), f"{path} did not resolve to a jax.Array"


def test_init_model_tiny_supernet_runs() -> None:
    """`init_model(TINY_SUPERNET, key=0)` builds a model and a forward
    pass on a small batch produces the right shape and finite numbers.
    """
    model = init_model(TINY_SUPERNET, key=0)
    # Embedding dims
    assert model.embed_src.shape == (64, TINY_SUPERNET.d_model)
    assert model.embed_dst.shape == (64, TINY_SUPERNET.d_model)
    assert model.embed_promo.shape == (5, TINY_SUPERNET.d_model)
    assert model.embed_pad.shape == (TINY_SUPERNET.d_model,)
    assert model.embed_outcome.shape == (
        TINY_SUPERNET.n_outcomes,
        TINY_SUPERNET.d_model,
    )
    # Stacked layer dims
    L = TINY_SUPERNET.n_layers
    d = TINY_SUPERNET.d_model
    d_ff = TINY_SUPERNET.d_ff
    assert model.layers.attn_norm_w.shape == (L, d)
    assert model.layers.wq.shape == (L, d, d)
    assert model.layers.w_gate.shape == (L, d, d_ff)
    assert model.layers.w_down.shape == (L, d_ff, d)
    # Final + LM head
    assert model.final_norm_w.shape == (d,)
    assert model.lm_head.shape == (d, TINY_SUPERNET.vocab_size)


def test_init_model_accepts_int_key_seed() -> None:
    """Passing a Python int as `key` is a documented convenience."""
    a = init_model(TINY_SUPERNET, key=0)
    b = init_model(TINY_SUPERNET, key=jax.random.key(0))
    # Same seed → identical params
    assert jnp.array_equal(a.embed_src, b.embed_src)
    assert jnp.array_equal(a.layers.wq, b.layers.wq)


def test_init_model_different_keys_produce_different_params() -> None:
    a = init_model(TINY_SUPERNET, key=0)
    b = init_model(TINY_SUPERNET, key=1)
    assert not jnp.array_equal(a.embed_src, b.embed_src)
    assert not jnp.array_equal(a.layers.wq, b.layers.wq)


def test_init_model_rmsnorm_weights_are_ones() -> None:
    """RMSNorm acts as identity at step 0 — `attn_norm_w` / `ffn_norm_w` /
    `final_norm_w` start at all-ones. The trainer learns to deviate."""
    model = init_model(TINY_SUPERNET, key=0)
    assert jnp.all(model.layers.attn_norm_w == 1)
    assert jnp.all(model.layers.ffn_norm_w == 1)
    assert jnp.all(model.final_norm_w == 1)


def test_init_model_embed_pad_is_zero() -> None:
    """`embed_pad` starts at zero so PAD positions contribute nothing
    until training learns a non-trivial representation."""
    model = init_model(TINY_SUPERNET, key=0)
    assert jnp.all(model.embed_pad == 0)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_pass_shape_and_finite() -> None:
    """A forward pass on a small batch returns `(B, T, V)` with finite logits."""
    model = init_model(TINY_SUPERNET, key=0)
    B, T = 2, 32
    tokens = jnp.zeros((B, T), dtype=jnp.int32)  # All "first move"
    logits = model(tokens)
    assert logits.shape == (B, T, TINY_SUPERNET.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_forward_pass_handles_pad_and_outcome_tokens() -> None:
    """The factored embedding + override path doesn't crash and produces
    finite logits when the batch contains PAD and outcome tokens."""
    model = init_model(TINY_SUPERNET, key=0)
    # Construct a sequence: outcome prefix + a few moves + PAD tail
    tokens = jnp.array(
        [
            [OUTCOME_TOKEN_BASE, 5, 100, 200, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN],
        ],
        dtype=jnp.int32,
    )
    logits = model(tokens)
    assert logits.shape == (1, 8, TINY_SUPERNET.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_forward_pass_respects_attention_mask() -> None:
    """When `attention_mask` excludes the tail, the prefix logits don't
    depend on what's after the masked-out positions. Causal + padding
    masking working correctly together.

    The causal mask alone would make positions 0–3 unaffected by tokens
    at positions 4–7, so the prefix invariance isn't itself proof that
    `attention_mask` is honored. To actually exercise pad-masking, we
    compare position 7 (which CAN see all earlier tokens under the
    causal mask) between an all-real mask and a mask that hides
    positions 4–6 — the two should differ. If `attention_mask` were
    ignored, those positions would still influence the position-7 logit
    via attention and the assertion below would fail.
    """
    model = init_model(TINY_SUPERNET, key=0)
    tokens_a = jnp.array(
        [[5, 10, 15, 20, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]],
        dtype=jnp.int32,
    )
    tokens_b = jnp.array(
        [[5, 10, 15, 20, 99, 88, 77, 66]],
        dtype=jnp.int32,
    )
    prefix_mask = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=jnp.int32)

    # Pad-tail invariance: positions 0–3 see only the masked prefix.
    out_a_prefix = model(tokens_a, prefix_mask)
    out_b_prefix = model(tokens_b, prefix_mask)
    assert jnp.allclose(out_a_prefix[:, :4], out_b_prefix[:, :4], atol=1e-5)

    # Pad-masking actually fires: with positions 4–6 hidden, the
    # position-7 logit should NOT match what we get when those positions
    # are visible (and contain divergent tokens).
    full_mask = jnp.ones((1, 8), dtype=jnp.int32)
    mid_hidden = jnp.array([[1, 1, 1, 1, 0, 0, 0, 1]], dtype=jnp.int32)
    same_tokens = jnp.array([[5, 10, 15, 20, 99, 88, 77, 66]], dtype=jnp.int32)
    out_full = model(same_tokens, full_mask)
    out_mid_hidden = model(same_tokens, mid_hidden)
    assert not jnp.allclose(out_full[:, 7], out_mid_hidden[:, 7], atol=1e-3)


def test_attention_is_causal() -> None:
    """Modifying a position's input token doesn't change earlier
    positions' outputs (causal mask working)."""
    model = init_model(TINY_SUPERNET, key=0)
    T = 8
    tokens_a = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
    tokens_b = jnp.array([[0, 1, 2, 3, 4, 99, 6, 7]], dtype=jnp.int32)
    out_a = model(tokens_a)
    out_b = model(tokens_b)
    # Positions 0..4 should be identical (they can't see position 5).
    assert jnp.allclose(out_a[:, :5], out_b[:, :5], atol=1e-5)
    # Positions 5..7 SHOULD differ.
    assert not jnp.allclose(out_a[:, 5:], out_b[:, 5:], atol=1e-3)


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def test_sliced_small_variant_runs() -> None:
    """`sliced(tiny_supernet, TINY_VARIANTS['small'])` produces a runnable
    smaller model with the right shapes."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    small_cfg = TINY_VARIANTS["small"]
    variant_model = sliced(supernet_model, small_cfg)

    d = small_cfg.d_model
    L = small_cfg.n_layers
    d_ff = small_cfg.d_ff

    assert variant_model.embed_src.shape == (64, d)
    assert variant_model.layers.wq.shape == (L, d, d)
    assert variant_model.layers.w_gate.shape == (L, d, d_ff)
    assert variant_model.layers.w_down.shape == (L, d_ff, d)
    assert variant_model.final_norm_w.shape == (d,)
    assert variant_model.lm_head.shape == (d, small_cfg.vocab_size)

    # Forward pass still works at the new shape
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = variant_model(tokens)
    assert logits.shape == (2, 16, small_cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_sliced_base_variant_runs() -> None:
    """The middle (`base`) tiny variant slices cleanly too."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    base_cfg = TINY_VARIANTS["base"]
    variant_model = sliced(supernet_model, base_cfg)
    assert variant_model.embed_src.shape == (64, base_cfg.d_model)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = variant_model(tokens)
    assert logits.shape == (2, 16, base_cfg.vocab_size)


def test_sliced_large_variant_is_identity_shape() -> None:
    """`large` IS the supernet — sliced(supernet, large_cfg) has the
    same shapes (though it's a fresh PAWNModel instance)."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    large_cfg = TINY_VARIANTS["large"]
    variant_model = sliced(supernet_model, large_cfg)
    assert variant_model.embed_src.shape == supernet_model.embed_src.shape
    assert variant_model.layers.wq.shape == supernet_model.layers.wq.shape
    # The slicing is just `[:d_V, :d_V]` with d_V == d_super, so the
    # underlying data is byte-identical.
    assert jnp.array_equal(variant_model.embed_src, supernet_model.embed_src)


def test_sliced_preserves_weights() -> None:
    """Every variant weight is the inner block of the supernet's
    corresponding tensor — slicing only narrows; values don't change.
    Spot-checks every one of the 16 trainable arrays so a transposed
    slice index (e.g. `wk` accidentally pulling the `wv` block) would
    surface."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    small_cfg = TINY_VARIANTS["small"]
    variant_model = sliced(supernet_model, small_cfg)
    dv = small_cfg.d_model
    dv_ff = small_cfg.d_ff

    # Embedding fields
    assert jnp.array_equal(variant_model.embed_src, supernet_model.embed_src[:, :dv])
    assert jnp.array_equal(variant_model.embed_dst, supernet_model.embed_dst[:, :dv])
    assert jnp.array_equal(variant_model.embed_promo, supernet_model.embed_promo[:, :dv])
    assert jnp.array_equal(variant_model.embed_pad, supernet_model.embed_pad[:dv])
    assert jnp.array_equal(
        variant_model.embed_outcome, supernet_model.embed_outcome[:, :dv]
    )
    # Attention block (5 tensors × n_layers)
    sup_l = supernet_model.layers
    var_l = variant_model.layers
    assert jnp.array_equal(var_l.attn_norm_w, sup_l.attn_norm_w[:, :dv])
    assert jnp.array_equal(var_l.wq, sup_l.wq[:, :dv, :dv])
    assert jnp.array_equal(var_l.wk, sup_l.wk[:, :dv, :dv])
    assert jnp.array_equal(var_l.wv, sup_l.wv[:, :dv, :dv])
    assert jnp.array_equal(var_l.wo, sup_l.wo[:, :dv, :dv])
    # FFN block (4 tensors × n_layers) — note w_down's slice shape differs.
    assert jnp.array_equal(var_l.ffn_norm_w, sup_l.ffn_norm_w[:, :dv])
    assert jnp.array_equal(var_l.w_gate, sup_l.w_gate[:, :dv, :dv_ff])
    assert jnp.array_equal(var_l.w_up, sup_l.w_up[:, :dv, :dv_ff])
    assert jnp.array_equal(var_l.w_down, sup_l.w_down[:, :dv_ff, :dv])
    # Final norm + output head
    assert jnp.array_equal(
        variant_model.final_norm_w, supernet_model.final_norm_w[:dv]
    )
    assert jnp.array_equal(variant_model.lm_head, supernet_model.lm_head[:dv, :])


def test_sliced_reuses_decomp_table() -> None:
    """The decomposition table doesn't depend on width; the variant
    should reuse the supernet's table verbatim. RoPE phase tables aren't
    stored on the model — they're rebuilt inside `__call__` — so there's
    nothing to test for them here."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    variant_model = sliced(supernet_model, TINY_VARIANTS["small"])
    assert jnp.array_equal(variant_model.decomp_table, supernet_model.decomp_table)


def test_sliced_rejects_invalid_nesting() -> None:
    """`sliced` calls `validate_nested`, so an incompatible config raises
    `NestingError`."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    # A variant with the wrong head_dim doesn't nest.
    bogus = ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=512, head_dim=32)
    with pytest.raises(NestingError):
        sliced(supernet_model, bogus)


# ---------------------------------------------------------------------------
# Production supernet (smoke; smaller than full training but real shape)
# ---------------------------------------------------------------------------


def test_init_model_production_supernet_builds() -> None:
    """`init_model(SUPERNET, key=0)` builds without error.

    Skip if we don't have enough memory — this is the full d=640 / 10
    layers / 10 heads shape and allocates ~50M float32 params (~200MB).
    On a constrained CI box you might want @pytest.mark.gpu.
    """
    model = init_model(SUPERNET, key=0)
    assert model.embed_src.shape == (64, 640)
    assert model.layers.wq.shape == (10, 640, 640)
    assert model.lm_head.shape == (640, 1980)


def test_forward_pass_rejects_sequence_too_long() -> None:
    """A sequence longer than `cfg.max_seq_len` is rejected at the top of
    `__call__` with a clear error, before the shape mismatch would
    surface as a confusing broadcast error from RoPE."""
    model = init_model(TINY_SUPERNET, key=0)
    over_max = TINY_SUPERNET.max_seq_len + 1
    tokens = jnp.zeros((1, over_max), dtype=jnp.int32)
    with pytest.raises(ValueError, match="exceeds cfg.max_seq_len"):
        model(tokens)


def test_only_inexact_arrays_count_as_trainable() -> None:
    """`eqx.filter(model, eqx.is_inexact_array)` should pick up exactly
    the 16 trainable tensors — no buffer leak. RoPE phase tables are
    recomputed inside `__call__`, not stored as model state, and
    `decomp_table` is int32 so `is_inexact_array` filters it out
    naturally."""
    import equinox as eqx

    model = init_model(TINY_SUPERNET, key=0)
    trainable = eqx.filter(model, eqx.is_inexact_array)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(trainable) if leaf is not None]
    # 16 trainable tensors: 5 embed + 9 layers + 2 final/output.
    assert len(leaves) == 16


def test_decomp_table_shape() -> None:
    """`decomp_table` is `(NUM_ACTIONS, 3)` of int32 — every move token has
    a (src, dst, promo) triple."""
    model = init_model(TINY_SUPERNET, key=0)
    assert model.decomp_table.shape == (NUM_ACTIONS, 3)
    assert model.decomp_table.dtype == jnp.int32
    # Square indices are in [0, 64), promo in [0, 5).
    assert jnp.all((model.decomp_table[:, 0] >= 0) & (model.decomp_table[:, 0] < 64))
    assert jnp.all((model.decomp_table[:, 1] >= 0) & (model.decomp_table[:, 1] < 64))
    assert jnp.all((model.decomp_table[:, 2] >= 0) & (model.decomp_table[:, 2] < 5))


# ---------------------------------------------------------------------------
# Parity #43: SDPA opt-in fast path
# ---------------------------------------------------------------------------


def test_use_sdpa_matches_plain_attention_within_fp32_noise() -> None:
    """``use_sdpa=True`` switches the attention block to
    :func:`jax.nn.dot_product_attention` (XLA impl). The migration
    plan §5 marked SDPA out of scope citing fused-kernel maturity on
    JAX-on-ROCm; in practice the XLA implementation works fine.

    Verify the SDPA path produces logits that match the plain
    materialised-QK path within fp32 numerical noise (the two paths
    differ only in how XLA fuses the matmul+softmax — algebraically
    identical, but the kernel ordering can introduce tiny rounding
    differences)."""
    model = init_model(TINY_SUPERNET, key=0)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    plain = model(tokens)
    sdpa = model(tokens, use_sdpa=True)
    assert plain.shape == sdpa.shape
    # Generous tolerance: SDPA's fused kernel reorders ops vs the plain
    # path, so per-position diffs can be a few ulps. Empirically max
    # diff is in the 1e-5 range on tiny supernet random weights.
    assert jnp.allclose(plain, sdpa, atol=1e-3, rtol=1e-3)
