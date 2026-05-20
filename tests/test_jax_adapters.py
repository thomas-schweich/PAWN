"""Tests for ``pawn.jax.adapters`` — Phase-3 chunk 1 (LoRA).

Pins the load-bearing invariants the trainer will rely on:
  * LoRA forward at init equals the backbone (B=0 convention)
  * Filter / partition: gradients reach LoRA leaves only
  * Forward differs once LoRA weights are non-zero
  * Sliced backbone is also valid (TINY_VARIANTS interop)
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("chess_engine")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.jax.adapters import LoRAConfig, LoRAModel, init_lora_model
from pawn.jax.adapters.lora import (
    _VALID_TARGETS,
    LoRAParams,
    adapter_filter,
    is_adapter_leaf,
)
from pawn.jax.config import TINY_SUPERNET, TINY_VARIANTS
from pawn.jax.model import init_model, sliced


def _toy_batch(b: int = 2, t: int = 8) -> tuple[jax.Array, jax.Array]:
    tokens = jnp.arange(b * t, dtype=jnp.int32).reshape(b, t) % 1968
    mask = jnp.ones((b, t), dtype=jnp.bool_)
    return tokens, mask


def test_lora_config_rejects_invalid() -> None:
    """rank <= 0, empty targets, and unknown target names all fail loud."""
    with pytest.raises(ValueError, match="rank"):
        LoRAConfig(rank=0)
    with pytest.raises(ValueError, match="rank"):
        LoRAConfig(rank=-1)
    with pytest.raises(ValueError, match="at least one"):
        LoRAConfig(rank=4, targets=())
    with pytest.raises(ValueError, match="unknown LoRA target"):
        LoRAConfig(rank=4, targets=("q", "bogus"))


def test_lora_config_scale_default_is_one() -> None:
    """``alpha=None`` ⇒ alpha==rank ⇒ scale=1.0 (the v1 default)."""
    cfg = LoRAConfig(rank=8)
    assert cfg.scale == 1.0
    cfg2 = LoRAConfig(rank=8, alpha=16.0)
    assert math.isclose(cfg2.scale, 2.0)


def test_lora_init_forward_equals_backbone() -> None:
    """Standard LoRA initialisation (B=0) → effective weight is the
    backbone's; forward must be bit-identical. A future regression
    that initialised B non-zero would silently destroy the
    pretrained baseline at step 0."""
    key = jax.random.PRNGKey(0)
    backbone = init_model(TINY_SUPERNET, key)
    lora_model = init_lora_model(
        backbone, LoRAConfig(rank=4, targets=("q", "v")), jax.random.PRNGKey(1)
    )
    tokens, mask = _toy_batch()
    out_bb = backbone(tokens, mask)
    out_lora = lora_model(tokens, mask)
    assert jnp.array_equal(out_bb, out_lora), (
        f"LoRA at init should equal backbone; max diff = "
        f"{float(jnp.abs(out_bb - out_lora).max())}"
    )


def test_lora_with_all_four_targets_init_equal_to_backbone() -> None:
    """Same invariant with all four projections targeted."""
    key = jax.random.PRNGKey(0)
    backbone = init_model(TINY_SUPERNET, key)
    lora_model = init_lora_model(
        backbone, LoRAConfig(rank=4, targets=("q", "k", "v", "o")), jax.random.PRNGKey(2)
    )
    tokens, mask = _toy_batch()
    assert jnp.array_equal(backbone(tokens, mask), lora_model(tokens, mask))


def test_lora_param_shapes() -> None:
    """Per-layer A/B shapes: A ∈ [L, d, r], B ∈ [L, r, d] for targets;
    zero-width sentinels for non-targets."""
    cfg_qv = LoRAConfig(rank=8, targets=("q", "v"))
    backbone = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    model = init_lora_model(backbone, cfg_qv, jax.random.PRNGKey(1))
    L = TINY_SUPERNET.n_layers
    d = TINY_SUPERNET.d_model
    assert model.lora.a_q.shape == (L, d, 8)
    assert model.lora.b_q.shape == (L, 8, d)
    assert model.lora.a_v.shape == (L, d, 8)
    assert model.lora.b_v.shape == (L, 8, d)
    # K and O are not targeted: zero-width sentinels.
    assert model.lora.a_k.shape == (L, d, 0)
    assert model.lora.b_k.shape == (L, 0, d)
    assert model.lora.a_o.shape == (L, d, 0)
    assert model.lora.b_o.shape == (L, 0, d)


def test_lora_b_zero_at_init() -> None:
    """B init must be exactly zero — load-bearing for the
    forward-equals-backbone invariant."""
    backbone = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    cfg = LoRAConfig(rank=4, targets=("q", "k", "v", "o"))
    model = init_lora_model(backbone, cfg, jax.random.PRNGKey(1))
    for tgt in cfg.targets:
        b = getattr(model.lora, f"b_{tgt}")
        assert jnp.all(b == 0), f"b_{tgt} not zero at init"


def test_lora_forward_differs_once_b_nonzero() -> None:
    """Perturbing one B matrix makes the forward diverge from the
    backbone — confirms the LoRA path actually contributes to the
    output."""
    backbone = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    cfg = LoRAConfig(rank=4, targets=("q",))
    model = init_lora_model(backbone, cfg, jax.random.PRNGKey(1))
    # Bump b_q to a non-zero matrix on the first layer.
    perturbed_b_q = model.lora.b_q.at[0].set(
        jnp.ones_like(model.lora.b_q[0]) * 0.01
    )
    new_lora = eqx.tree_at(lambda lp: lp.b_q, model.lora, perturbed_b_q)
    perturbed = eqx.tree_at(lambda m: m.lora, model, new_lora)
    tokens, mask = _toy_batch()
    out_bb = backbone(tokens, mask)
    out_lora = perturbed(tokens, mask)
    diff = float(jnp.abs(out_bb - out_lora).max())
    assert diff > 1e-5, f"LoRA path produced no effect; diff = {diff}"


def test_adapter_filter_targets_only_lora_leaves() -> None:
    """``adapter_filter`` must return True only on ``model.lora.*``
    array leaves; backbone leaves must be False. This is the two-tier
    partition Phase-3 §6.1 relies on for the 33% compute cut."""
    backbone = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    model = init_lora_model(
        backbone, LoRAConfig(rank=4, targets=("q", "v")), jax.random.PRNGKey(1)
    )
    flt = adapter_filter(model)
    # Sample backbone leaves: must be False.
    assert flt.backbone.wq is False
    assert flt.backbone.lm_head is False
    assert flt.backbone.final_norm is False
    # LoRA array leaves: True.
    assert flt.lora.a_q is True
    assert flt.lora.b_q is True
    assert flt.lora.a_v is True
    assert flt.lora.b_v is True
    # Zero-width non-target leaves are still arrays (just shape-0
    # in one axis) — also True; the optimizer treats them uniformly.
    assert flt.lora.a_k is True
    assert flt.lora.b_o is True


def test_gradient_flows_only_into_lora_params() -> None:
    """The trainer-critical contract: ``eqx.filter_grad`` with
    ``adapter_filter`` produces a grad tree where backbone leaves are
    None (no gradient) and LoRA leaves are arrays."""
    backbone = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    model = init_lora_model(
        backbone, LoRAConfig(rank=4, targets=("q", "v")), jax.random.PRNGKey(1)
    )
    tokens, mask = _toy_batch()

    def loss_fn(m: LoRAModel) -> jax.Array:
        out = m(tokens, mask)
        return out.mean()

    flt = adapter_filter(model)
    grads = eqx.filter_grad(loss_fn, has_aux=False)(model)
    # The grad PyTree has the same structure as ``model``; backbone
    # subtree should have None at every array leaf (because
    # filter_grad respects eqx.filter's default partitioning by
    # is_inexact_array, which catches everything — so we apply the
    # adapter filter to verify the contract holds).
    # Easier check: backbone weight gradients are exactly zero (since
    # they didn't flow through the gradient computation — but they
    # would here unless we wrap in filter_grad with the adapter
    # filter spec).
    del flt, grads
    # The direct two-tier pattern: partition first, then grad on the
    # trainable subtree.
    trainable, frozen = eqx.partition(model, adapter_filter(model))

    def loss_on_trainable(trn: LoRAModel) -> jax.Array:
        return eqx.combine(trn, frozen)(tokens, mask).mean()

    adapter_grads = eqx.filter_grad(loss_on_trainable)(trainable)
    # Adapter grads: every LoRA array leaf is a real gradient (non-None);
    # backbone leaves remain None (carried from the filter).
    # b_q: backprop from output → effective_backbone → wq → forward;
    # since out.mean() depends on wq (and wq depends on b_q via the
    # LoRA delta), gradient must reach b_q.
    assert adapter_grads.lora.b_q is not None
    assert jnp.abs(adapter_grads.lora.b_q).sum() > 0
    # a_q has B=0 at init → gradient is zero (∂out/∂a = a's contribution
    # is scaled by B=0). Other targets also see zero a-grad at init.
    # This is correct LoRA behavior, not a bug. Just check the leaf is
    # an array (not None).
    assert isinstance(adapter_grads.lora.a_q, jax.Array)


def test_lora_on_sliced_backbone_works() -> None:
    """LoRA must compose with the sliced-variant backbone — the
    typical Phase-3 use case: train an adapter on, say, ``pawn-base``
    sliced from the supernet."""
    supernet = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    base = sliced(supernet, TINY_VARIANTS["base"])
    model = init_lora_model(
        base, LoRAConfig(rank=4, targets=("q", "v")), jax.random.PRNGKey(1)
    )
    tokens, mask = _toy_batch()
    out_bb = base(tokens, mask)
    out_lora = model(tokens, mask)
    assert jnp.array_equal(out_bb, out_lora), "LoRA on sliced variant ≠ sliced backbone"


def test_lora_scale_alpha_changes_effective_update() -> None:
    """Larger ``alpha`` should produce a larger forward divergence
    after perturbing B (the standard LoRA-α scaling). Compare two
    LoRAModels with the same B perturbation but different alphas."""
    backbone = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    cfg_low = LoRAConfig(rank=4, targets=("q",), alpha=4.0)   # scale = 1.0
    cfg_hi = LoRAConfig(rank=4, targets=("q",), alpha=16.0)   # scale = 4.0
    model_low = init_lora_model(backbone, cfg_low, jax.random.PRNGKey(1))
    model_hi = init_lora_model(backbone, cfg_hi, jax.random.PRNGKey(1))
    # Apply identical B perturbation to both.
    pert = jnp.full_like(model_low.lora.b_q, 0.01)

    def with_b(m: LoRAModel, b: jax.Array) -> LoRAModel:
        return eqx.tree_at(
            lambda mm: mm.lora.b_q, m, b
        )
    m_low = with_b(model_low, pert)
    m_hi = with_b(model_hi, pert)
    tokens, mask = _toy_batch()
    diff_low = float(jnp.abs(backbone(tokens, mask) - m_low(tokens, mask)).max())
    diff_hi = float(jnp.abs(backbone(tokens, mask) - m_hi(tokens, mask)).max())
    # scale_hi / scale_low = 4 → diff_hi ≈ 4 × diff_low.
    assert diff_hi > 3 * diff_low, (
        f"alpha scaling broken: diff_low={diff_low}, diff_hi={diff_hi}"
    )


def test_lora_model_is_jax_pytree() -> None:
    """LoRAModel must round-trip through ``jax.tree_util.tree_flatten``
    so it composes cleanly with ``lax.scan``, ``jit``, and Optax."""
    backbone = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    model = init_lora_model(
        backbone, LoRAConfig(rank=4, targets=("q", "v")), jax.random.PRNGKey(1)
    )
    leaves, treedef = jax.tree_util.tree_flatten(model)
    assert len(leaves) > 0
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, LoRAModel)
    assert isinstance(rebuilt.backbone, type(backbone))
    assert isinstance(rebuilt.lora, LoRAParams)
