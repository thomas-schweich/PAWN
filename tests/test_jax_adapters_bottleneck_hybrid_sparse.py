"""Tests for the Bottleneck / Hybrid / Sparse adapter strategies.

Each strategy gets the canonical adapter-contract battery:

* identity-(approximate-)at-init: wrapped forward matches the bare
  backbone bit-identically (or within a small tolerance for the
  Sparse soft mask which sits at ``σ(5) ≈ 0.993``);
* ``eqx.partition`` accepts the strategy's ``adapter_filter`` spec
  and produces a trainable subtree shaped as expected;
* ``eqx.filter_grad`` flows gradients to the adapter leaves only —
  backbone leaves stay None;
* config rejects invalid hyperparameters.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pawn.adapters import (
    BottleneckConfig,
    BottleneckModel,
    FiLMConfig,
    HybridConfig,
    HybridModel,
    LoRAConfig,
    SparseConfig,
    SparseModel,
    bottleneck_adapter_filter,
    hybrid_adapter_filter,
    init_bottleneck_model,
    init_hybrid_model,
    init_sparse_model,
    sparse_adapter_filter,
)
from pawn.config import TINY_SUPERNET
from pawn.model import init_model


def _tiny_inputs(B: int = 2, T: int = 8) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(
        rng.integers(0, 64, size=(B, T)), dtype=jnp.int32,
    )
    attn = jnp.ones((B, T), dtype=jnp.bool_)
    return tokens, attn


# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------


def test_bottleneck_identity_init_matches_backbone() -> None:
    """``up.weight = 0`` at init → wrapped forward is bit-identical
    to the bare backbone."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    tokens, attn = _tiny_inputs()
    bare = backbone(tokens, attn)
    for n_hidden in (0, 2):
        model = init_bottleneck_model(
            backbone,
            BottleneckConfig(bottleneck_dim=4, n_hidden=n_hidden),
            jax.random.key(0),
        )
        got = model(tokens, attn)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(bare), rtol=0, atol=1e-5,
            err_msg=f"bottleneck identity-init drifted (n_hidden={n_hidden})",
        )


def test_bottleneck_param_count() -> None:
    """Per-adapter params (no bias): 2·d·bn + n_hidden·bn².
    Over n_layers × 2 positions (attn+ffn)."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    bn = 4
    n_hidden = 2
    model = init_bottleneck_model(
        backbone,
        BottleneckConfig(bottleneck_dim=bn, n_hidden=n_hidden),
        jax.random.key(0),
    )
    per_pos = 2 * cfg.d_model * bn + n_hidden * bn * bn
    expected = cfg.n_layers * 2 * per_pos
    actual = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(
            eqx.filter(model.bot, eqx.is_inexact_array)
        )
    )
    assert actual == expected, f"expected {expected}, got {actual}"


def test_bottleneck_partition_and_grad_flow() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_bottleneck_model(
        backbone, BottleneckConfig(bottleneck_dim=4), jax.random.key(1),
    )
    spec = bottleneck_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)
    # Adapter leaves are present on the trainable side; backbone weight
    # leaves are None.
    assert trainable.bot.attn_down is not None
    assert trainable.backbone.wq is None
    tokens, attn = _tiny_inputs()

    def loss_fn(trainable: BottleneckModel) -> jax.Array:
        merged = eqx.combine(trainable, frozen)
        return merged(tokens, attn).mean()

    grads = eqx.filter_grad(loss_fn)(trainable)
    # Adapter grads exist (some non-zero); backbone wq grad is None.
    # Note: at identity-init the bottleneck adapter output is zero,
    # so the ``down`` and ``hidden`` grads are zero too — only the
    # ``up`` grad can be non-zero on the very first step.
    assert grads.bot.attn_up is not None
    assert float(jnp.abs(grads.bot.attn_up).sum()) > 0.0
    assert grads.backbone.wq is None


def test_bottleneck_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="bottleneck_dim"):
        BottleneckConfig(bottleneck_dim=0)
    with pytest.raises(ValueError, match="n_hidden"):
        BottleneckConfig(bottleneck_dim=4, n_hidden=-1)
    with pytest.raises(ValueError, match="at least one"):
        BottleneckConfig(bottleneck_dim=4, adapt_attn=False, adapt_ffn=False)


def test_bottleneck_layers_filter_zeros_inactive() -> None:
    """When ``cfg.layers`` excludes a layer, that layer's adapter
    weights all start at zero — the forward is still bit-identical
    to the backbone since the inactive slice contributes nothing."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    # Adapt only the top layer (n_layers - 1).
    model = init_bottleneck_model(
        backbone,
        BottleneckConfig(
            bottleneck_dim=4, layers=(cfg.n_layers - 1,),
        ),
        jax.random.key(0),
    )
    # Inactive layer slice (index 0) of attn_down is zero.
    np.testing.assert_array_equal(
        np.asarray(model.bot.attn_down[0]),
        np.zeros_like(np.asarray(model.bot.attn_down[0])),
    )


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------


def test_hybrid_identity_init_matches_backbone() -> None:
    """LoRA's B = 0 and FiLM's γ=1 / β=0 both yield identity at init —
    composed, the hybrid forward is also bit-identical to the
    backbone."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    tokens, attn = _tiny_inputs()
    bare = backbone(tokens, attn)
    model = init_hybrid_model(
        backbone,
        HybridConfig(
            lora=LoRAConfig(rank=2, targets=("q", "v")),
            film=FiLMConfig(use_output_film=True),
        ),
        jax.random.key(0),
    )
    got = model(tokens, attn)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(bare), rtol=0, atol=1e-5,
        err_msg="hybrid identity-init drifted",
    )


def test_hybrid_partition_covers_both_adapters() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_hybrid_model(
        backbone,
        HybridConfig(
            lora=LoRAConfig(rank=2, targets=("q", "v")),
            film=FiLMConfig(use_output_film=False),
        ),
        jax.random.key(0),
    )
    spec = hybrid_adapter_filter(model)
    trainable, _ = eqx.partition(model, spec)
    # Both adapter subtrees contain trainable arrays.
    assert trainable.lora.a_q is not None
    assert trainable.film.gammas is not None
    # Backbone is filtered out.
    assert trainable.backbone.wq is None


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------


def test_sparse_approx_identity_at_init() -> None:
    """``init_logit=+5`` gives ``σ(5) ≈ 0.993``; the wrapped forward
    drifts ≤ ~1% from the backbone in fp32. Use a slightly looser
    tolerance than the bit-identical adapters above."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    tokens, attn = _tiny_inputs()
    bare = backbone(tokens, attn)
    model = init_sparse_model(
        backbone,
        SparseConfig(targets=("q", "k", "v", "o"), init_logit=5.0, hard=False),
        jax.random.key(0),
    )
    got = model(tokens, attn)
    bare_np = np.asarray(bare)
    got_np = np.asarray(got)
    # σ(5) ≈ 0.9933 → about 0.7% deviation per matmul element. The
    # actual logit-level drift accumulates over multiple layers; cap
    # at 5% of the bare logit's max magnitude.
    max_drift = float(np.abs(got_np - bare_np).max())
    max_bare = float(np.abs(bare_np).max())
    assert max_drift <= 0.05 * max_bare + 1e-3, (
        f"sparse soft-init drift too large: {max_drift} vs "
        f"5% of {max_bare}"
    )


def test_sparse_hard_mode_is_thresholded() -> None:
    """With ``hard=True``, the forward mask is exactly 0/1 (not a
    sigmoid). At ``init_logit=+5`` every element is above the
    default threshold 0.5, so the mask is all-ones and the forward
    *is* bit-identical to the backbone (which is the cleaner identity
    check than soft mode)."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    tokens, attn = _tiny_inputs()
    bare = backbone(tokens, attn)
    model = init_sparse_model(
        backbone,
        SparseConfig(
            targets=("q", "k", "v", "o"), init_logit=5.0, hard=True,
        ),
        jax.random.key(0),
    )
    got = model(tokens, attn)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(bare), rtol=0, atol=1e-5,
        err_msg="sparse hard-init at +5 should be bit-identical",
    )


def test_sparse_partition_and_grad_flow() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_sparse_model(
        backbone,
        SparseConfig(targets=("q", "v"), init_logit=5.0),
        jax.random.key(0),
    )
    spec = sparse_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)
    assert trainable.sparse.logits_q is not None
    assert trainable.backbone.wq is None
    tokens, attn = _tiny_inputs()

    def loss_fn(trainable: SparseModel) -> jax.Array:
        merged = eqx.combine(trainable, frozen)
        return merged(tokens, attn).mean()

    grads = eqx.filter_grad(loss_fn)(trainable)
    # Active-target grads exist; inactive (k, o) targets are
    # zero-width sentinels so their grads are also zero-width.
    assert grads.sparse.logits_q is not None
    assert grads.sparse.logits_q.shape == (cfg.n_layers, cfg.d_model, cfg.d_model)
    assert grads.sparse.logits_k is not None
    assert grads.sparse.logits_k.shape == (cfg.n_layers, cfg.d_model, 0)
    assert grads.backbone.wq is None


def test_sparse_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="density"):
        SparseConfig(density=-0.1)
    with pytest.raises(ValueError, match="threshold"):
        SparseConfig(threshold=0.0)
    with pytest.raises(ValueError, match="bare str"):
        SparseConfig(targets="qv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        SparseConfig(targets=())
    with pytest.raises(ValueError, match="unknown sparse target"):
        SparseConfig(targets=("x",))
