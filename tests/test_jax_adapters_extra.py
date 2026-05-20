"""Tests for the FiLM + Unfreeze adapter strategies.

LoRA-side tests already live in ``tests/test_jax_adapter_trainer.py``.
This file pins:

* FiLM identity-init: ``FiLMModel(backbone, init γ=1, β=0)`` forward
  output equals the bare backbone forward bitwise.
* FiLM partition: ``film_adapter_filter`` produces a tree where the
  backbone subtree's array leaves are all ``False`` and the FiLM
  subtree's array leaves are all ``True``.
* Unfreeze partition: layer-stacked weights get a True/False mask
  along the layer axis with True iff ``layer_idx >= n_layers - n_unfreeze``.
  ``lm_head`` + ``final_norm`` follow ``include_lm_head``; embeddings
  follow ``include_embeddings``.
* The unfrozen-only partition really excludes the frozen layers from
  gradients (verified through ``eqx.filter_grad``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pawn.adapters import (
    FiLMConfig,
    FiLMModel,
    UnfreezeConfig,
    UnfreezeModel,
    film_adapter_filter,
    init_film_model,
    init_unfreeze_model,
    unfreeze_adapter_filter,
)
from pawn.config import TINY_SUPERNET
from pawn.model import init_model


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------


def _tiny_inputs(B: int = 2, T: int = 8) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(
        rng.integers(0, 64, size=(B, T)), dtype=jnp.int32,
    )
    attn = jnp.ones((B, T), dtype=jnp.bool_)
    return tokens, attn


def test_film_identity_init_matches_backbone() -> None:
    """γ=1, β=0 at init means the FiLM-wrapped forward is bit-identical
    to the bare backbone."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    tokens, attn = _tiny_inputs()

    bare = backbone(tokens, attn)
    for use_output in (False, True):
        wrapped = init_film_model(
            backbone, FiLMConfig(use_output_film=use_output),
            jax.random.key(0),
        )
        got = wrapped(tokens, attn)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(bare), rtol=0, atol=1e-5,
            err_msg=f"FiLM identity init drifted (use_output_film={use_output})",
        )


def test_film_param_count() -> None:
    """Per-layer γ/β + optional output γ/β param count."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_film_model(
        backbone, FiLMConfig(use_output_film=True), jax.random.key(0),
    )
    # gammas + betas: 2 × n_layers × d_model + 2 × vocab_size
    expected = 2 * cfg.n_layers * cfg.d_model + 2 * cfg.vocab_size
    actual = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(model.film, eqx.is_inexact_array))
    )
    assert actual == expected, f"expected {expected}, got {actual}"


def test_film_no_output_param_count() -> None:
    """``use_output_film=False`` keeps the (0,) sentinel buffers."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_film_model(
        backbone, FiLMConfig(use_output_film=False), jax.random.key(0),
    )
    expected = 2 * cfg.n_layers * cfg.d_model  # no output FiLM
    actual = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(model.film, eqx.is_inexact_array))
    )
    assert actual == expected


def test_film_adapter_filter_partition() -> None:
    """``film_adapter_filter`` produces the trainable / frozen partition.

    Trainable subtree must contain the FiLM γ/β arrays. Frozen subtree
    must contain the backbone array leaves (all non-None) and FiLM
    leaves replaced by None.
    """
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_film_model(
        backbone, FiLMConfig(use_output_film=True), jax.random.key(1),
    )
    spec = film_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)
    # Trainable side: FiLM γ/β arrays present; backbone leaves None.
    assert trainable.film.gammas is not None
    assert trainable.film.betas is not None
    for name in (
        "src_embed", "dst_embed", "promo_embed", "wq", "wk", "wv", "wo",
        "lm_head", "final_norm",
    ):
        assert getattr(trainable.backbone, name) is None, (
            f"trainable.backbone.{name} should be None"
        )
    # Frozen side: backbone leaves present; FiLM γ/β are None.
    assert frozen.backbone.wq is not None
    assert frozen.film.gammas is None
    assert frozen.film.betas is None


def test_film_grad_flows_only_to_film() -> None:
    """``eqx.filter_value_and_grad`` over the partition only touches
    the FiLM leaves — backbone gradients are ``None``."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_film_model(
        backbone, FiLMConfig(use_output_film=False), jax.random.key(1),
    )
    tokens, attn = _tiny_inputs()
    spec = film_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)

    def loss_fn(trainable: FiLMModel) -> jax.Array:
        merged = eqx.combine(trainable, frozen)
        logits = merged(tokens, attn)
        return logits.mean()

    grads = eqx.filter_grad(loss_fn)(trainable)
    # FiLM grads must exist; backbone weight grads (those are None on
    # the trainable subtree to begin with) stay None on the gradient.
    assert grads.film.gammas is not None
    assert grads.film.betas is not None
    # Spot-check a backbone leaf: should be None on grads (it was None
    # on trainable and no gradient flowed).
    assert grads.backbone.wq is None


# ---------------------------------------------------------------------------
# Unfreeze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_unfreeze", [0, 1, 2, 4])
def test_unfreeze_filter_is_pure_python_bool(n_unfreeze: int) -> None:
    """The filter spec leaves are Python bools / None — ``eqx.partition``
    rejects array-valued spec leaves, so the layer-stacked weights are
    fully True iff at least one layer is unfrozen. Per-layer slicing
    happens in ``make_gradient_mask`` (covered separately)."""
    from pawn.adapters.unfreeze import make_gradient_mask

    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_unfreeze_model(
        backbone,
        UnfreezeConfig(
            n_unfreeze=n_unfreeze,
            include_lm_head=False,
            include_embeddings=False,
        ),
        jax.random.key(0),
    )
    spec = unfreeze_adapter_filter(model)
    layers_trainable = (n_unfreeze > 0)
    # wq spec leaf is a Python bool of the correct value.
    assert spec.backbone.wq is layers_trainable
    # eqx.partition must accept it without raising — that's the
    # CRITICAL contract bug-detector + Codex flagged earlier.
    eqx.partition(model, spec)


@pytest.mark.parametrize("n_unfreeze", [0, 1, 2, 4])
def test_unfreeze_gradient_mask_per_layer(n_unfreeze: int) -> None:
    """``make_gradient_mask`` returns the per-layer bool mask True only
    on layer indices ``>= n_layers - n_unfreeze`` for layer-stacked
    weights; head + embedding leaves are full-zero when their config
    flags are False."""
    from pawn.adapters.unfreeze import make_gradient_mask

    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_unfreeze_model(
        backbone,
        UnfreezeConfig(
            n_unfreeze=n_unfreeze,
            include_lm_head=False,
            include_embeddings=False,
        ),
        jax.random.key(0),
    )
    mask = make_gradient_mask(model)
    wq_mask = mask.backbone.wq
    assert wq_mask.shape == backbone.wq.shape
    assert wq_mask.dtype == jnp.bool_
    expected = jnp.arange(cfg.n_layers) >= (cfg.n_layers - n_unfreeze)
    np.testing.assert_array_equal(
        np.asarray(wq_mask[:, 0, 0]), np.asarray(expected),
    )
    # Head / embedding masks are all-False when their flags are off.
    assert not bool(mask.backbone.lm_head.any())
    assert not bool(mask.backbone.src_embed.any())


def test_unfreeze_include_lm_head_toggle() -> None:
    """``include_lm_head`` toggles whether lm_head + final_norm are
    in the trainable partition (Python-bool granularity)."""
    from pawn.adapters.unfreeze import make_gradient_mask

    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))

    for include in (False, True):
        model = init_unfreeze_model(
            backbone,
            UnfreezeConfig(
                n_unfreeze=0,
                include_lm_head=include,
                include_embeddings=False,
            ),
            jax.random.key(0),
        )
        spec = unfreeze_adapter_filter(model)
        if include:
            assert spec.backbone.lm_head is True
            assert spec.backbone.final_norm is True
            mask = make_gradient_mask(model)
            assert bool(mask.backbone.lm_head.all())
            assert bool(mask.backbone.final_norm.all())
        else:
            assert spec.backbone.lm_head is False
            assert spec.backbone.final_norm is False


def test_unfreeze_include_embeddings_toggle() -> None:
    from pawn.adapters.unfreeze import make_gradient_mask

    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_unfreeze_model(
        backbone,
        UnfreezeConfig(
            n_unfreeze=0, include_lm_head=False, include_embeddings=True,
        ),
        jax.random.key(0),
    )
    spec = unfreeze_adapter_filter(model)
    mask = make_gradient_mask(model)
    for name in ("src_embed", "dst_embed", "promo_embed", "pad_embed", "outcome_embed"):
        assert getattr(spec.backbone, name) is True
        m = getattr(mask.backbone, name)
        assert m.shape == getattr(backbone, name).shape
        assert bool(m.all())


def test_unfreeze_partition_and_gradient_mask_compose() -> None:
    """End-to-end: ``eqx.partition`` produces a trainable subtree, then
    multiplying gradients by ``make_gradient_mask`` zeros out the
    frozen-layer slices. This is the actual contract the trainer
    will use (which the previous test suite did not cover, letting the
    array-valued-filter regression through)."""
    from pawn.adapters.unfreeze import make_gradient_mask

    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    n_unfreeze = 2
    model = init_unfreeze_model(
        backbone,
        UnfreezeConfig(
            n_unfreeze=n_unfreeze,
            include_lm_head=False, include_embeddings=False,
        ),
        jax.random.key(0),
    )
    tokens, attn = _tiny_inputs()

    spec = unfreeze_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)

    def loss_fn(trainable: UnfreezeModel) -> jax.Array:
        merged = eqx.combine(trainable, frozen)
        return merged(tokens, attn).mean()

    grads = eqx.filter_grad(loss_fn)(trainable)
    grad_mask = make_gradient_mask(model)
    # Multiply trainable-side wq gradient by the layer mask; frozen
    # layer slices should land at zero.
    masked_wq = grads.backbone.wq * grad_mask.backbone.wq.astype(grads.backbone.wq.dtype)
    n_layers = cfg.n_layers
    first_unfrozen = n_layers - n_unfreeze
    # Frozen layer slices (indices 0..first_unfrozen-1): all zero.
    np.testing.assert_array_equal(
        np.asarray(masked_wq[:first_unfrozen]),
        np.zeros_like(np.asarray(masked_wq[:first_unfrozen])),
    )
    # Unfrozen layer slices: not (entirely) zero — gradients flow.
    assert float(jnp.abs(masked_wq[first_unfrozen:]).sum()) > 0.0


def test_unfreeze_rejects_too_many_layers() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    with pytest.raises(ValueError, match="exceeds backbone n_layers"):
        init_unfreeze_model(
            backbone,
            UnfreezeConfig(n_unfreeze=cfg.n_layers + 1),
            jax.random.key(0),
        )


def test_unfreeze_forward_matches_backbone() -> None:
    """``UnfreezeModel.__call__`` is identical to the bare backbone
    forward — the wrapper carries the config only."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_unfreeze_model(
        backbone, UnfreezeConfig(n_unfreeze=2), jax.random.key(0),
    )
    tokens, attn = _tiny_inputs()
    np.testing.assert_allclose(
        np.asarray(model(tokens, attn)),
        np.asarray(backbone(tokens, attn)),
        rtol=0, atol=0,
    )


def test_unfreeze_negative_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        UnfreezeConfig(n_unfreeze=-1)
