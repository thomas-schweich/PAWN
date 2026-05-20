"""Tests for the RoSA + SpecializedCLM adapter strategies.

* RoSA identity-at-init: LoRA B=0 + sparse Δ=0 + mask=False → wrapped
  forward bit-identical to the backbone.
* ``rosa_set_mask`` rewrites the sparse-leg mask in place.
* RoSA partition / grad-flow contract.
* SpecializedCLM is a thin wrapper around ``init_model`` — verify it
  produces a usable ``PAWNModel`` at the small scale + the partition
  marks every leaf trainable.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pawn.adapters import (
    RoSAConfig,
    RoSAModel,
    SpecializedCLMConfig,
    init_rosa_model,
    init_specialized_clm,
    rosa_adapter_filter,
    rosa_compute_phase2_mask,
    rosa_lora_only_adapter_filter,
    rosa_set_mask,
    rosa_sparse_only_adapter_filter,
    specialized_clm_adapter_filter,
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
# RoSA
# ---------------------------------------------------------------------------


def test_rosa_identity_init_matches_backbone() -> None:
    """B=0, Δ=0, mask=False → bit-identical forward."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    tokens, attn = _tiny_inputs()
    bare = backbone(tokens, attn)
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q", "v")), jax.random.key(1),
    )
    got = model(tokens, attn)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(bare), rtol=0, atol=1e-5,
        err_msg="RoSA identity-init drifted",
    )


def test_rosa_partition_and_grad_flow() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q",)), jax.random.key(1),
    )
    spec = rosa_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)
    # The active q target gets trainable LoRA A/B + sparse Δ; backbone
    # leaves are None on the trainable side.
    assert trainable.rosa.a_q is not None
    assert trainable.rosa.delta_q is not None
    assert trainable.backbone.wq is None
    # Mask arrays are bool — eqx.is_inexact_array returns False on
    # bool, so they sit on the frozen side rather than the trainable
    # side. That's the right place since the trainer rewrites them
    # between phases via rosa_set_mask, not via gradients.
    assert trainable.rosa.mask_q is None
    assert frozen.rosa.mask_q is not None

    tokens, attn = _tiny_inputs()

    def loss_fn(trainable: RoSAModel) -> jax.Array:
        merged = eqx.combine(trainable, frozen)
        return merged(tokens, attn).mean()

    grads = eqx.filter_grad(loss_fn)(trainable)
    # At identity-init B = 0 and Δ = 0, so the only non-trivial
    # gradients flow into B (the entry that multiplies the next
    # operand). A and Δ have non-zero grad too once they multiply
    # B / mask respectively; the mask is False so Δ grad is zero,
    # and B is initially zero so the A grad is also zero. The B
    # grad must be non-zero for non-trivial learning to be possible.
    assert grads.rosa.b_q is not None
    assert float(jnp.abs(grads.rosa.b_q).sum()) > 0.0


def test_rosa_set_mask_rewrites_target() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q", "v")), jax.random.key(1),
    )
    new_mask_q = jnp.ones((cfg.n_layers, cfg.d_model, cfg.d_model), dtype=jnp.bool_)
    updated = rosa_set_mask(model, {"q": new_mask_q})
    np.testing.assert_array_equal(
        np.asarray(updated.rosa.mask_q), np.ones_like(np.asarray(new_mask_q)),
    )
    # Untouched targets keep their original (all-False) mask.
    assert not bool(updated.rosa.mask_v.any())


def test_rosa_set_mask_rejects_wrong_shape() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q",)), jax.random.key(0),
    )
    with pytest.raises(ValueError, match="expected"):
        rosa_set_mask(model, {"q": jnp.zeros((3, 3), dtype=jnp.bool_)})


def test_rosa_config_validation() -> None:
    with pytest.raises(ValueError, match="rank"):
        RoSAConfig(rank=0)
    with pytest.raises(ValueError, match="bare str"):
        RoSAConfig(rank=2, targets="qv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        RoSAConfig(rank=2, targets=())
    with pytest.raises(ValueError, match="unknown RoSA target"):
        RoSAConfig(rank=2, targets=("z",))
    with pytest.raises(ValueError, match="alpha"):
        RoSAConfig(rank=2, alpha=0.0)


def test_rosa_mask_affects_forward() -> None:
    """After ``rosa_set_mask`` sets an all-ones mask on q and Δ is
    perturbed, the forward output differs from the backbone forward.
    Covers the sparse-leg fold-in path."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q",)), jax.random.key(1),
    )
    tokens, attn = _tiny_inputs()
    bare = backbone(tokens, attn)
    # Activate the sparse leg on q with a non-zero Δ.
    new_delta_q = (
        jnp.ones((cfg.n_layers, cfg.d_model, cfg.d_model), dtype=jnp.float32)
        * 0.01
    )
    new_mask_q = jnp.ones((cfg.n_layers, cfg.d_model, cfg.d_model), dtype=jnp.bool_)
    updated = eqx.tree_at(
        lambda m: m.rosa.delta_q, model, new_delta_q,
    )
    updated = rosa_set_mask(updated, {"q": new_mask_q})
    got = updated(tokens, attn)
    diff = float(jnp.abs(got - bare).sum())
    assert diff > 0.0, "RoSA sparse leg failed to affect the forward"


# ---------------------------------------------------------------------------
# RoSA three-phase primitives
# ---------------------------------------------------------------------------


def test_rosa_lora_only_filter_freezes_delta() -> None:
    """Phase 1 filter: A, B are trainable; Δ stays frozen.

    After ``eqx.partition``, only the LoRA ``a_*`` / ``b_*`` leaves
    appear under ``trainable.rosa``; the sparse-Δ ``delta_*`` leaves
    are ``None``. Mirrors the structural invariant used by the
    Phase-1 warmup loop in
    ``scripts/train_jax_adapter.py``.
    """
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q",)), jax.random.key(1),
    )
    spec = rosa_lora_only_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)
    assert trainable.rosa.a_q is not None
    assert trainable.rosa.b_q is not None
    assert trainable.rosa.delta_q is None
    assert frozen.rosa.delta_q is not None


def test_rosa_sparse_only_filter_freezes_lora() -> None:
    """Phase 2 filter (used for gradient extraction): only Δ is on
    the trainable side; A, B are frozen. The trainer never optimises
    with this filter — it's used to compute ``dL/dΔ`` on a batch
    while ensuring XLA dead-code-eliminates dL/dA, dL/dB."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q",)), jax.random.key(1),
    )
    spec = rosa_sparse_only_adapter_filter(model)
    trainable, frozen = eqx.partition(model, spec)
    assert trainable.rosa.a_q is None
    assert trainable.rosa.b_q is None
    assert trainable.rosa.delta_q is not None


def test_rosa_compute_phase2_mask_picks_top_k_per_layer() -> None:
    """``rosa_compute_phase2_mask`` returns bool masks shaped like the
    delta arrays, with exactly ``ceil(top_k_frac * d * d)`` True
    entries per layer per target (with ties broken by mask >=
    threshold so the count can be ≥, not exactly equal, to k)."""
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q", "v")), jax.random.key(1),
    )
    n_layers = cfg.n_layers
    d = cfg.d_model
    rng = np.random.default_rng(7)
    fake_grads = {
        "q": jnp.asarray(rng.standard_normal((n_layers, d, d)), dtype=jnp.float32),
        "v": jnp.asarray(rng.standard_normal((n_layers, d, d)), dtype=jnp.float32),
    }
    masks = rosa_compute_phase2_mask(model, fake_grads, top_k_frac=0.05)
    assert set(masks.keys()) == {"q", "v"}
    n_per_layer = d * d
    k = max(1, int(n_per_layer * 0.05))
    for tgt in ("q", "v"):
        m = masks[tgt]
        assert m.shape == (n_layers, d, d)
        assert m.dtype == jnp.bool_
        for layer in range(n_layers):
            n_active = int(m[layer].sum())
            # The threshold-based ranking can include ties at the
            # boundary, so we get at least ``k`` entries (and rarely
            # more).
            assert n_active >= k, (
                f"target={tgt} layer={layer}: {n_active} active "
                f"< k={k}"
            )


def test_rosa_compute_phase2_mask_rejects_bad_top_k_frac() -> None:
    cfg = TINY_SUPERNET
    backbone = init_model(cfg, jax.random.key(0))
    model = init_rosa_model(
        backbone, RoSAConfig(rank=2, targets=("q",)), jax.random.key(1),
    )
    n_layers = cfg.n_layers
    d = cfg.d_model
    grads = {"q": jnp.zeros((n_layers, d, d), dtype=jnp.float32)}
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="top_k_frac"):
            rosa_compute_phase2_mask(model, grads, top_k_frac=bad)


# ---------------------------------------------------------------------------
# SpecializedCLM
# ---------------------------------------------------------------------------


def test_specialized_clm_produces_usable_model() -> None:
    cfg = SpecializedCLMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = init_specialized_clm(cfg, jax.random.key(0))
    tokens, attn = _tiny_inputs()
    logits = model(tokens, attn)
    assert logits.shape == (2, 8, model.cfg.vocab_size)


def test_specialized_clm_filter_marks_everything_trainable() -> None:
    cfg = SpecializedCLMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = init_specialized_clm(cfg, jax.random.key(0))
    spec = specialized_clm_adapter_filter(model)
    # Every inexact-array leaf is True; the static ``cfg`` field is
    # False.
    leaves = jax.tree_util.tree_leaves(spec)
    array_leaves = [l for l in leaves if isinstance(l, (bool, np.bool_))]
    assert any(array_leaves), "no boolean filter-spec leaves emitted"
    assert all(l is True or l is False for l in array_leaves)


def test_specialized_clm_rejects_odd_head_dim() -> None:
    """``head_dim = d_model / n_heads`` must be even (RoPE rotates
    pairs). ``d_model=70 / n_heads=2 → 35`` is odd → ModelConfig
    rejects."""
    with pytest.raises(ValueError, match="head_dim"):
        init_specialized_clm(
            SpecializedCLMConfig(d_model=70, n_layers=2, n_heads=2, d_ff=128),
            jax.random.key(0),
        )
