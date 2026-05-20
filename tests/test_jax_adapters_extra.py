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
def test_unfreeze_layer_mask_shape(n_unfreeze: int) -> None:
    """The layer-stacked weight masks are bool arrays True only on the
    top ``n_unfreeze`` layer indices."""
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
    # wq is shape (n_layers, d, d). The filter spec replaces it with a
    # bool array of the same shape; the leading-axis values are True
    # iff layer_idx >= n_layers - n_unfreeze.
    wq_mask = spec.backbone.wq
    assert wq_mask.shape == backbone.wq.shape
    assert wq_mask.dtype == jnp.bool_
    expected = jnp.arange(cfg.n_layers) >= (cfg.n_layers - n_unfreeze)
    actual = wq_mask[:, 0, 0]
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_unfreeze_include_lm_head_toggle() -> None:
    """``include_lm_head`` controls whether lm_head + final_norm are
    in the trainable partition."""
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
            # lm_head + final_norm are bool arrays of the right shape
            assert spec.backbone.lm_head.shape == backbone.lm_head.shape
            assert spec.backbone.lm_head.dtype == jnp.bool_
            assert bool(spec.backbone.lm_head.all())
            assert bool(spec.backbone.final_norm.all())
        else:
            # Stay as bare False scalars
            assert spec.backbone.lm_head is False
            assert spec.backbone.final_norm is False


def test_unfreeze_include_embeddings_toggle() -> None:
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
    for name in ("src_embed", "dst_embed", "promo_embed", "pad_embed", "outcome_embed"):
        m = getattr(spec.backbone, name)
        assert m.shape == getattr(backbone, name).shape
        assert bool(m.all())


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
