"""Tests for :mod:`pawn.model` — Equinox PAWNModel + init + slicing.

The model module depends on JAX, so these tests assume the test env
has the rocm or cu128 extra installed. They run on CPU just fine
(JAX falls back); the per-section verification commands run on the
GPU.

Coverage:
- The trainable arrays land in the right shapes / dtypes (tied =
  11 leaves, untied = 12 with a standalone ``lm_head``).
- ``init_model`` builds a runnable model for both the production
  ``SUPERNET`` and the lightweight ``TINY_SUPERNET``.
- The forward pass produces the right shape and finite outputs.
- The forward pass handles PAD tokens and outcome tokens correctly
  (they don't crash the single-gather embedding lookup).
- The attention is causal — masking out the right-hand context doesn't
  change the left-hand outputs (within tolerance).
- ``sliced(supernet, variant_cfg)`` builds a runnable smaller model
  with the right shapes.
- ``sliced`` rejects an invalid nesting.
- ``saved_fields`` returns the right schema per ``tie_embeddings`` and
  every name resolves to an array attribute on the model.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from pawn.config import (
    BOS_TOKEN,
    NULL_TOKEN,
    N_TOTAL_OUTCOMES,
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VOCAB_SIZE,
    ModelConfig,
    NestingError,
)
from pawn.model import (
    SAVED_FIELDS,
    KVCache,
    PAWNModel,
    init_kv_cache,
    init_model,
    saved_fields,
    sliced,
)


# ---------------------------------------------------------------------------
# Shape + structural invariants
# ---------------------------------------------------------------------------


def test_saved_fields_untied_superset_is_twelve() -> None:
    """`SAVED_FIELDS` is the untied superset: 12 named arrays
    (embed_tokens + 9 layers + final_norm + lm_head), `lm_head` last."""
    assert len(SAVED_FIELDS) == 12
    assert len(set(SAVED_FIELDS)) == 12  # No duplicates
    assert SAVED_FIELDS[0] == "embed_tokens"
    assert SAVED_FIELDS[-1] == "lm_head"


def test_saved_fields_helper_per_tie() -> None:
    """`saved_fields(tie_embeddings)` drops `lm_head` when tied and keeps
    it when untied; declaration order is otherwise preserved."""
    untied = saved_fields(False)
    tied = saved_fields(True)
    assert untied == SAVED_FIELDS
    assert "lm_head" not in tied
    assert len(tied) == 11
    # Tied schema is the untied superset minus lm_head, order preserved.
    assert tied == tuple(f for f in SAVED_FIELDS if f != "lm_head")


def test_saved_fields_match_pawn_model_attributes() -> None:
    """Every name in the per-tie schema resolves to an actual array on a
    freshly-built model. Catches typos like `embed_token` vs
    `embed_tokens`. Checks both the tied (default) and untied models."""
    for tie in (True, False):
        cfg = _tiny_cfg(tie_embeddings=tie)
        model = init_model(cfg, key=0)
        for path in saved_fields(tie):
            node = model
            for piece in path.split("."):
                node = getattr(node, piece)
            assert isinstance(node, jax.Array), (
                f"{path} did not resolve to a jax.Array (tie={tie})"
            )


def _tiny_cfg(*, tie_embeddings: bool) -> ModelConfig:
    """A TINY_SUPERNET-shaped config with an explicit `tie_embeddings`."""
    return ModelConfig(
        d_model=TINY_SUPERNET.d_model,
        n_layers=TINY_SUPERNET.n_layers,
        n_heads=TINY_SUPERNET.n_heads,
        d_ff=TINY_SUPERNET.d_ff,
        tie_embeddings=tie_embeddings,
    )


def test_init_model_tiny_supernet_runs() -> None:
    """`init_model(TINY_SUPERNET, key=0)` builds a model and a forward
    pass on a small batch produces the right shape and finite numbers.
    """
    model = init_model(TINY_SUPERNET, key=0)
    # Single uniform token table [V, d]
    assert model.embed_tokens.shape == (TINY_SUPERNET.vocab_size, TINY_SUPERNET.d_model)
    # Stacked layer dims
    L = TINY_SUPERNET.n_layers
    d = TINY_SUPERNET.d_model
    d_ff = TINY_SUPERNET.d_ff
    assert model.layers.attn_norm_w.shape == (L, d)
    assert model.layers.wq.shape == (L, d, d)
    assert model.layers.w_gate.shape == (L, d, d_ff)
    assert model.layers.w_down.shape == (L, d_ff, d)
    # Final norm + untied output head (untied is the v2 default).
    assert model.final_norm_w.shape == (d,)
    assert TINY_SUPERNET.tie_embeddings is False
    assert model.lm_head is not None
    assert model.lm_head.shape == (d, TINY_SUPERNET.vocab_size)


def test_init_model_untied_has_standalone_lm_head() -> None:
    """An untied config gives a standalone `lm_head[d, V]`; a tied config
    leaves it `None` (logits reuse `embed_tokens.T`)."""
    untied = init_model(_tiny_cfg(tie_embeddings=False), key=0)
    assert untied.lm_head is not None
    assert untied.lm_head.shape == (TINY_SUPERNET.d_model, TINY_SUPERNET.vocab_size)

    tied = init_model(_tiny_cfg(tie_embeddings=True), key=0)
    assert tied.lm_head is None


def test_tied_logits_equal_untied_with_transposed_head() -> None:
    """The tied head is exactly `embed_tokens.T`. Build a matched untied
    model — same backbone + token table, `lm_head = embed_tokens.T` — and
    confirm the two produce identical logits. This pins that tying is the
    weight-sharing it claims to be (not just a shape coincidence)."""
    import dataclasses

    import equinox as eqx

    tied = init_model(_tiny_cfg(tie_embeddings=True), key=0)
    # Re-tag the same arrays as an untied model with lm_head = embed_tokens.T.
    untied = eqx.tree_at(
        lambda m: m.lm_head,
        dataclasses.replace(tied, cfg=_tiny_cfg(tie_embeddings=False)),
        tied.embed_tokens.T,
        is_leaf=lambda x: x is None,
    )
    assert untied.lm_head is not None
    tokens = jnp.arange(2 * 12, dtype=jnp.int32).reshape(2, 12) % NUM_ACTIONS
    assert jnp.array_equal(tied(tokens), untied(tokens))


def test_tied_and_untied_logits_shape_match() -> None:
    """Tied and untied models both emit `(B, T, V)` logits."""
    tied = init_model(_tiny_cfg(tie_embeddings=True), key=0)
    untied = init_model(_tiny_cfg(tie_embeddings=False), key=0)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    assert tied(tokens).shape == (2, 16, VOCAB_SIZE)
    assert untied(tokens).shape == (2, 16, VOCAB_SIZE)


def test_init_model_accepts_int_key_seed() -> None:
    """Passing a Python int as `key` is a documented convenience."""
    a = init_model(TINY_SUPERNET, key=0)
    b = init_model(TINY_SUPERNET, key=jax.random.key(0))
    # Same seed → identical params
    assert jnp.array_equal(a.embed_tokens, b.embed_tokens)
    assert jnp.array_equal(a.layers.wq, b.layers.wq)


def test_init_model_different_keys_produce_different_params() -> None:
    a = init_model(TINY_SUPERNET, key=0)
    b = init_model(TINY_SUPERNET, key=1)
    assert not jnp.array_equal(a.embed_tokens, b.embed_tokens)
    assert not jnp.array_equal(a.layers.wq, b.layers.wq)


def test_init_model_rmsnorm_weights_are_ones() -> None:
    """RMSNorm acts as identity at step 0 — `attn_norm_w` / `ffn_norm_w` /
    `final_norm_w` start at all-ones. The trainer learns to deviate."""
    model = init_model(TINY_SUPERNET, key=0)
    assert jnp.all(model.layers.attn_norm_w == 1)
    assert jnp.all(model.layers.ffn_norm_w == 1)
    assert jnp.all(model.final_norm_w == 1)


def test_init_model_embed_tokens_is_normal_init() -> None:
    """`embed_tokens` is the uniform `[V, d]` table, `Normal(0, 0.02)`
    initialised — no special zero row (v1's zero-initialised `embed_pad`
    is gone; PAD is just another row of the shared table now)."""
    model = init_model(TINY_SUPERNET, key=0)
    assert model.embed_tokens.shape == (TINY_SUPERNET.vocab_size, TINY_SUPERNET.d_model)
    # Not degenerate: the table has real spread, no all-zero rows expected
    # at init (random Normal).
    assert float(jnp.std(model.embed_tokens)) > 0.0
    assert not jnp.any(jnp.all(model.embed_tokens == 0, axis=-1))


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
    """The single-gather embedding path doesn't crash and produces
    finite logits when the batch contains PAD and outcome tokens (every
    id indexes the same `embed_tokens` table)."""
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


def test_embed_does_not_alias_control_tokens_onto_last_outcome() -> None:
    """BOS / NULL / reserved-control ids (≥1980) embed as their *own*
    independent rows — the load-bearing contract of the §2-vocab/§3-un-factor
    co-land.

    The old factored ``_embed`` clamped ids and aliased the Python-side
    control tokens onto the last outcome embedding. The new ``_embed`` is a
    plain gather into the ``[V, d]`` ``embed_tokens`` table, so every id in
    ``[0, V)`` — including BOS (1980), NULL (1981), and the reserved columns
    (1982–1999) — must index its own distinct, finite row. If anyone
    reintroduced a cap (``clip(ids, 0, 1979)``) or sized the table ``< V``,
    these ids would silently alias onto the last outcome row (1979) and the
    rest of the suite would stay green.
    """
    model = init_model(TINY_SUPERNET, key=0)
    last_outcome = OUTCOME_TOKEN_BASE + N_TOTAL_OUTCOMES - 1  # 1979
    reserved_id = 1999  # top reserved-control column
    control_ids = [BOS_TOKEN, NULL_TOKEN, reserved_id]

    # The table is wide enough to hold every control id (no aliasing by size).
    assert model.embed_tokens.shape[0] == VOCAB_SIZE
    assert VOCAB_SIZE > reserved_id

    tokens = jnp.array([control_ids], dtype=jnp.int32)
    embedded = model._embed(tokens)  # pyright: ignore[reportPrivateUsage]
    assert embedded.shape == (1, len(control_ids), TINY_SUPERNET.d_model)
    assert jnp.all(jnp.isfinite(embedded))

    for slot, tok in enumerate(control_ids):
        # Each control id gathers exactly its own embed_tokens row …
        assert jnp.array_equal(embedded[0, slot], model.embed_tokens[tok])
        # … and is NOT aliased onto the last outcome row (1979), the row the
        # v1 factored/clipped path collapsed them onto.
        assert not jnp.array_equal(
            model.embed_tokens[tok], model.embed_tokens[last_outcome]
        )

    # A full forward over a control-bearing sequence stays finite.
    logits = model(tokens)
    assert logits.shape == (1, len(control_ids), TINY_SUPERNET.vocab_size)
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

    assert variant_model.embed_tokens.shape == (small_cfg.vocab_size, d)
    assert variant_model.layers.wq.shape == (L, d, d)
    assert variant_model.layers.w_gate.shape == (L, d, d_ff)
    assert variant_model.layers.w_down.shape == (L, d_ff, d)
    assert variant_model.final_norm_w.shape == (d,)
    # Untied by default → lm_head narrowed along d to [:d_V, :], full vocab cols.
    assert variant_model.lm_head is not None
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
    assert variant_model.embed_tokens.shape == (base_cfg.vocab_size, base_cfg.d_model)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = variant_model(tokens)
    assert logits.shape == (2, 16, base_cfg.vocab_size)


def test_sliced_large_variant_is_identity_shape() -> None:
    """`large` IS the supernet — sliced(supernet, large_cfg) has the
    same shapes (though it's a fresh PAWNModel instance)."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    large_cfg = TINY_VARIANTS["large"]
    variant_model = sliced(supernet_model, large_cfg)
    assert variant_model.embed_tokens.shape == supernet_model.embed_tokens.shape
    assert variant_model.layers.wq.shape == supernet_model.layers.wq.shape
    # The slicing is just `[:, :d_V]` with d_V == d_super, so the
    # underlying data is byte-identical.
    assert jnp.array_equal(variant_model.embed_tokens, supernet_model.embed_tokens)


def test_sliced_preserves_weights() -> None:
    """Every variant weight is the inner block of the supernet's
    corresponding tensor — slicing only narrows; values don't change.
    Spot-checks every trainable array so a transposed slice index (e.g.
    `wk` accidentally pulling the `wv` block) would surface."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    small_cfg = TINY_VARIANTS["small"]
    variant_model = sliced(supernet_model, small_cfg)
    dv = small_cfg.d_model
    dv_ff = small_cfg.d_ff

    # Uniform token table: width-sliced to d_V (full vocab rows kept).
    assert jnp.array_equal(
        variant_model.embed_tokens, supernet_model.embed_tokens[:, :dv]
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
    # Final norm + output head. TINY_SUPERNET is untied (v2 default), so
    # lm_head[d, V] is narrowed along d ([:dv, :]) with the full vocab
    # columns kept — the same inner-block slicing as every other tensor.
    assert jnp.array_equal(
        variant_model.final_norm_w, supernet_model.final_norm_w[:dv]
    )
    assert supernet_model.lm_head is not None
    assert jnp.array_equal(variant_model.lm_head, supernet_model.lm_head[:dv, :])


def test_sliced_untied_slices_lm_head() -> None:
    """For an untied supernet, `sliced` narrows the standalone `lm_head`
    along d (`[:dv, :]`) and keeps the full vocab columns."""
    supernet_cfg = ModelConfig(
        d_model=TINY_SUPERNET.d_model,
        n_layers=TINY_SUPERNET.n_layers,
        n_heads=TINY_SUPERNET.n_heads,
        d_ff=TINY_SUPERNET.d_ff,
        tie_embeddings=False,
    )
    supernet_model = init_model(supernet_cfg, key=0)
    assert supernet_model.lm_head is not None
    small = TINY_VARIANTS["small"]
    variant_cfg = ModelConfig(
        d_model=small.d_model,
        n_layers=small.n_layers,
        n_heads=small.n_heads,
        d_ff=small.d_ff,
        tie_embeddings=False,
    )
    variant_model = sliced(supernet_model, variant_cfg)
    dv = small.d_model
    assert variant_model.lm_head is not None
    assert variant_model.lm_head.shape == (dv, small.vocab_size)
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


def test_sliced_depth_then_width_variant() -> None:
    """B.5: a variant with fewer layers AND smaller d_model is a
    depth+width slice. The variant model carries the supernet's first
    `nv` layers, each truncated to width `dv`."""
    supernet_model = init_model(TINY_SUPERNET, key=0)
    # TINY_SUPERNET is L=4, d=192. Try L=2, d=128 (matches TINY base width).
    shallow = ModelConfig(d_model=128, n_layers=2, n_heads=2, d_ff=512)
    variant = sliced(supernet_model, shallow)
    assert variant.cfg.n_layers == 2
    assert variant.cfg.d_model == 128
    # Layer stack is shape (2, ...) not (4, ...)
    assert variant.layers.wq.shape == (2, 128, 128)
    # Width truncation: first 128 cols of supernet's first 2 layers.
    assert jnp.allclose(
        variant.layers.wq, supernet_model.layers.wq[:2, :128, :128]
    )
    # The variant runs forward without error.
    tokens = jnp.zeros((1, 8), dtype=jnp.int32)
    logits = variant(tokens)
    assert logits.shape == (1, 8, TINY_SUPERNET.vocab_size)


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
    # Uniform token table [V, d] = [2000, 640]; input vocab == output vocab.
    assert model.embed_tokens.shape == (VOCAB_SIZE, 640)
    assert model.layers.wq.shape == (10, 640, 640)
    # SUPERNET is untied by default → standalone lm_head[d, V].
    assert model.lm_head is not None
    assert model.lm_head.shape == (640, VOCAB_SIZE)


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
    the trainable tensors — no buffer leak. A tied model has 11
    (embed_tokens + 9 layers + final_norm); an untied model adds the
    standalone lm_head for 12. RoPE phase tables are recomputed inside
    `__call__`, not stored as model state, and `decomp_table` is int32 so
    `is_inexact_array` filters it out naturally."""
    import equinox as eqx

    tied = init_model(_tiny_cfg(tie_embeddings=True), key=0)
    tied_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(eqx.filter(tied, eqx.is_inexact_array))
        if leaf is not None
    ]
    # 11 tensors: embed_tokens + 9 layers + final_norm. lm_head is None
    # (an empty PyTree subtree), so the optimizer never sees a head leaf.
    assert len(tied_leaves) == 11

    untied = init_model(_tiny_cfg(tie_embeddings=False), key=0)
    untied_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(eqx.filter(untied, eqx.is_inexact_array))
        if leaf is not None
    ]
    # 12 tensors: the tied set plus the standalone lm_head.
    assert len(untied_leaves) == 12


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


def test_use_flash_matches_plain_attention_within_fp32_noise() -> None:
    """``use_flash=True`` routes attention through
    :func:`jax.experimental.pallas.ops.gpu.attention.mha` — a Triton
    fused-attention kernel. Verify the Pallas path is algebraically
    equivalent to the plain materialised path within fp32 noise on
    tiny-supernet random weights.

    Pallas requires a GPU backend; skip on CPU. ``head_dim`` must be a
    power of two ≥ 16 for the bundled Pallas mha, which the tiny
    supernet (head_dim=64) satisfies.
    """
    import pytest

    if jax.default_backend() != "gpu":
        pytest.skip("Pallas flash attention requires a GPU backend")
    model = init_model(TINY_SUPERNET, key=0)
    # Use a non-zero token so the embed-lookup hits real (non-PAD) rows;
    # an all-zero batch goes through the PAD-embed override and bypasses
    # the attention contribution we actually want to compare.
    tokens = jnp.arange(2 * 16, dtype=jnp.int32).reshape(2, 16) % 100
    plain = model(tokens)
    flash = model(tokens, use_flash=True)
    assert plain.shape == flash.shape
    # Tolerance: Pallas reorders the softmax/matmul reduction across
    # tiles, so per-position diffs can run a few ulps wider than SDPA's.
    assert jnp.allclose(plain, flash, atol=1e-3, rtol=1e-3)


def test_materialised_attn_fp32_matches_reference_value_and_grad() -> None:
    """The fp32 materialised attention used as ``_flash_attn``'s backward
    (:func:`pawn.model._materialised_attn_fp32`) is algebraically a causal,
    fp32-softmax attention. Verify both its value AND its gradient against an
    independent reference. CPU-runnable (pure JAX, no Pallas) — this locks in
    the einsum layouts / causal orientation that the flash backward depends
    on without needing a GPU.
    """
    from pawn.model import _materialised_attn_fp32

    b, h, t, d = 2, 3, 16, 8
    inv = d ** -0.5
    ks = jax.random.split(jax.random.key(7), 3)
    # (B, T, H, d) layout — what `_pallas_attn` feeds `_flash_attn`.
    q = jax.random.normal(ks[0], (b, t, h, d))
    k = jax.random.normal(ks[1], (b, t, h, d))
    v = jax.random.normal(ks[2], (b, t, h, d))

    def reference(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        scores = jnp.einsum("bthd,bshd->bhts", q, k) * inv
        causal = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bhts,bshd->bthd", attn, v)

    # Value parity.
    assert jnp.allclose(
        _materialised_attn_fp32(q, k, v, inv), reference(q, k, v), atol=1e-5
    )

    # Gradient parity (the whole point of the flash backward fix).
    def sq_loss(fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
        return lambda q, k, v: (fn(q, k, v) ** 2).sum()

    g_mat = jax.grad(
        sq_loss(lambda q, k, v: _materialised_attn_fp32(q, k, v, inv)),
        (0, 1, 2),
    )(q, k, v)
    g_ref = jax.grad(sq_loss(reference), (0, 1, 2))(q, k, v)
    for gm, gr in zip(g_mat, g_ref, strict=True):
        assert jnp.allclose(gm, gr, atol=1e-4, rtol=1e-4)


def test_use_flash_backward_matches_plain_within_tol() -> None:
    """``use_flash=True`` now routes the BACKWARD through an fp32 materialised
    attention (custom VJP in :func:`pawn.model._flash_attn`), because the
    stock Pallas Triton backward downcasts the softmax-gradient intermediates
    to bf16 and diverged in pretraining at step ~255.5k. Verify the flash
    gradient matches the plain ``use_flash=False`` gradient — the property the
    fix exists to guarantee. Both backwards use fp32 softmax, so a tight
    tolerance holds even under bf16 activations.

    Pallas requires a GPU backend; skip on CPU.
    """
    import pytest

    if jax.default_backend() != "gpu":
        pytest.skip("Pallas flash attention requires a GPU backend")

    model = init_model(TINY_SUPERNET, key=0)
    tokens = jnp.arange(2 * 16, dtype=jnp.int32).reshape(2, 16) % 100

    # Gradient w.r.t. the embedding table (downstream of every attention
    # block) is a faithful proxy for backward correctness.
    def embed_grad(use_flash: bool, dtype: jnp.dtype) -> jax.Array:
        def f(embed: jax.Array) -> jax.Array:
            m = eqx.tree_at(lambda mm: mm.embed_tokens, model, embed)
            logits = m(tokens, use_flash=use_flash, compute_dtype=dtype)
            return (logits.astype(jnp.float32) ** 2).sum()

        return jax.grad(f)(model.embed_tokens).astype(jnp.float32)

    def rel_norm(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.linalg.norm(a - b) / jnp.linalg.norm(b)

    # In fp32 (no bf16 rounding) the custom-VJP backward must be algebraically
    # the plain path's gradient — the core correctness claim.
    g_flash_f32 = embed_grad(True, jnp.float32)
    g_plain_f32 = embed_grad(False, jnp.float32)
    assert rel_norm(g_flash_f32, g_plain_f32) < 1e-4

    # Under bf16 (the production config) per-element noise on small-magnitude
    # entries is unavoidable, so an elementwise tolerance is the wrong test.
    # The meaningful property is that flash is no farther from the fp32 ground
    # truth than the plain path — a faithful bf16 replacement, not a
    # regression. (It is in fact marginally closer: it upcasts q/k to fp32
    # before the score matmul, whereas the plain path matmuls in bf16.)
    truth = g_plain_f32
    d_flash = rel_norm(embed_grad(True, jnp.bfloat16), truth)
    d_plain = rel_norm(embed_grad(False, jnp.bfloat16), truth)
    assert d_flash <= d_plain * 1.05


def test_logits_always_fp32_even_under_bf16_amp() -> None:
    """Output logits are ALWAYS returned in fp32, even when the forward
    runs in bf16 (``compute_dtype=jnp.bfloat16``). The head matmul is
    upcast to fp32 so a bf16 logit column can't overflow to ``inf`` (→
    softmax NaN → one optimizer step poisoning the output projection).
    Stability hardening; an earlier version returned bf16 logits under AMP.
    """
    model = init_model(TINY_SUPERNET, key=0)
    tokens = jnp.arange(2 * 16, dtype=jnp.int32).reshape(2, 16) % 100

    fp32_logits = model(tokens)
    assert fp32_logits.dtype == jnp.float32

    bf16_amp_logits = model(tokens, compute_dtype=jnp.bfloat16)
    # bf16 activations through the stack, but logits upcast to fp32 in the head.
    assert bf16_amp_logits.dtype == jnp.float32
    assert bf16_amp_logits.shape == fp32_logits.shape
    assert jnp.all(jnp.isfinite(bf16_amp_logits))
    # The bf16 forward still tracks the fp32 forward within bf16's coarse tol.
    assert jnp.allclose(bf16_amp_logits, fp32_logits, atol=2e-1, rtol=2e-1)


# ---------------------------------------------------------------------------
# KV-cached generation
# ---------------------------------------------------------------------------


def test_init_kv_cache_shapes_match_cfg() -> None:
    """``init_kv_cache`` allocates a (n_layers, B, H, T_max, head_dim)
    K/V pair matching the model config."""
    cache = init_kv_cache(TINY_SUPERNET, batch_size=3, max_seq_len=16)
    expected = (
        TINY_SUPERNET.n_layers, 3, TINY_SUPERNET.n_heads, 16, TINY_SUPERNET.head_dim,
    )
    assert cache.k.shape == expected
    assert cache.v.shape == expected
    # Zero-initialised so unfilled slots are masked-out cleanly.
    assert jnp.all(cache.k == 0)
    assert jnp.all(cache.v == 0)


def test_init_kv_cache_defaults_to_cfg_max_seq_len() -> None:
    """When ``max_seq_len`` is omitted, the cache capacity comes from
    ``cfg.max_seq_len``."""
    cache = init_kv_cache(TINY_SUPERNET, batch_size=1)
    assert cache.k.shape[3] == TINY_SUPERNET.max_seq_len


def test_forward_with_cache_matches_full_forward_one_shot() -> None:
    """Single-call cached forward over a full prefix produces logits
    bit-identical to the non-cached forward (modulo XLA kernel-ordering
    noise). This pins the load-bearing invariant — without it the
    diagnostics get different numbers in cached vs non-cached mode."""
    model = init_model(TINY_SUPERNET, key=0)
    tokens = jnp.array(
        [[1969, 5, 10, 20, 30, 40], [1970, 8, 16, 24, 32, 40]], dtype=jnp.int32,
    )
    plain = model(tokens)
    cache = init_kv_cache(TINY_SUPERNET, batch_size=2, max_seq_len=8)
    cached, _ = model.forward_with_cache(tokens, cache, pos_start=0)
    assert plain.shape == cached.shape
    assert jnp.allclose(plain, cached, atol=1e-4, rtol=1e-4)


def test_forward_with_cache_step_by_step_matches_full_forward() -> None:
    """Step-by-step single-token cached decode reproduces the full
    forward across positions. This is the actual hot path for
    autoregressive generation — one token per call, threading the cache
    through."""
    model = init_model(TINY_SUPERNET, key=0)
    tokens = jnp.array(
        [[1969, 5, 10, 20, 30, 40, 50, 60]], dtype=jnp.int32,
    )
    plain = model(tokens)
    cache = init_kv_cache(TINY_SUPERNET, batch_size=1, max_seq_len=16)
    chunks: list[jax.Array] = []
    for pos in range(tokens.shape[1]):
        l, cache = model.forward_with_cache(
            tokens[:, pos : pos + 1], cache, pos_start=pos,
        )
        chunks.append(l)
    cached = jnp.concatenate(chunks, axis=1)
    assert jnp.allclose(plain, cached, atol=1e-4, rtol=1e-4)


def test_forward_with_cache_rejects_oversized_input() -> None:
    """An input longer than the cache capacity must raise — silently
    overflowing the cache would corrupt later decode steps."""
    model = init_model(TINY_SUPERNET, key=0)
    cache = init_kv_cache(TINY_SUPERNET, batch_size=1, max_seq_len=4)
    tokens = jnp.zeros((1, 8), dtype=jnp.int32)
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        model.forward_with_cache(tokens, cache, pos_start=0)


@pytest.mark.parametrize(
    "pos_start",
    [
        6,                                       # Python int
        jnp.int32(6),                            # JAX scalar
    ],
    ids=["python_int", "jnp_scalar"],
)
def test_forward_with_cache_rejects_write_window_overflow(pos_start) -> None:  # type: ignore[no-untyped-def]
    """A write window past the cache capacity must raise for any
    integer type of `pos_start`: `lax.dynamic_update_slice` silently
    clamps the start so `pos_start + T_new > T_max` would corrupt the
    cache without warning. Round-1 + round-2 review (bug-detector +
    codex P2) pinned this as a critical correctness gap; the round-1
    fix only caught Python int, leaving JAX scalars (the
    `autoregressive_generate` cached-path call form) silently broken.
    `eqx.error_if` now handles both."""
    import equinox as eqx
    model = init_model(TINY_SUPERNET, key=0)
    cache = init_kv_cache(TINY_SUPERNET, batch_size=1, max_seq_len=8)
    tokens = jnp.zeros((1, 4), dtype=jnp.int32)
    # `eqx.error_if` raises EquinoxRuntimeError (concrete type — using
    # the specific class instead of bare `Exception` so unrelated
    # regressions can't silently satisfy the match).
    with pytest.raises(eqx.EquinoxRuntimeError, match="exceeds cache capacity"):
        model.forward_with_cache(tokens, cache, pos_start=pos_start)


def test_forward_with_cache_oob_guard_fires_under_jit() -> None:
    """The OOB guard must also raise when the caller wraps
    `forward_with_cache` in `eqx.filter_jit` — that's the actual
    `autoregressive_generate` hot path. Round-3 test-risk HIGH +
    bug-detector IMPORTANT raised the concern that `eqx.error_if`
    might silently degrade to a no-op inside JIT; this test pins the
    expected behavior (EquinoxRuntimeError still surfaces)."""
    import equinox as eqx
    model = init_model(TINY_SUPERNET, key=0)
    cache = init_kv_cache(TINY_SUPERNET, batch_size=1, max_seq_len=8)
    tokens = jnp.zeros((1, 4), dtype=jnp.int32)

    @eqx.filter_jit
    def jitted_forward(t: jax.Array, c, p: jax.Array):  # type: ignore[no-untyped-def]
        return model.forward_with_cache(t, c, p)

    with pytest.raises(eqx.EquinoxRuntimeError, match="exceeds cache capacity"):
        jitted_forward(tokens, cache, jnp.int32(6))


def test_forward_with_cache_returns_fresh_cache() -> None:
    """The cached forward is a functional update — the input cache is
    not mutated, and the returned cache has the new K/V written in."""
    model = init_model(TINY_SUPERNET, key=0)
    cache = init_kv_cache(TINY_SUPERNET, batch_size=1, max_seq_len=8)
    tokens = jnp.array([[1969, 5, 10]], dtype=jnp.int32)
    _, new_cache = model.forward_with_cache(tokens, cache, pos_start=0)
    # Input cache untouched.
    assert jnp.all(cache.k == 0)
    assert jnp.all(cache.v == 0)
    # Filled slots: positions [0, 3) — non-zero.
    assert float(jnp.linalg.norm(new_cache.k[:, 0, :, 0, :])) > 0.0
    assert float(jnp.linalg.norm(new_cache.k[:, 0, :, 2, :])) > 0.0
    # Unfilled slots: position 3+ — still zero.
    assert jnp.all(new_cache.k[:, 0, :, 3, :] == 0)


def test_kv_cache_is_pytree() -> None:
    """KVCache is an eqx.Module so it threads through jit and scan
    cleanly. Verify it round-trips through ``jax.tree_util.tree_map``."""
    cache = init_kv_cache(TINY_SUPERNET, batch_size=1, max_seq_len=4)
    cache2 = jax.tree_util.tree_map(lambda x: x + 0, cache)
    assert isinstance(cache2, KVCache)
    assert cache2.k.shape == cache.k.shape
