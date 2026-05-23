"""LoRA — Low-Rank Adaptation (Hu et al. 2021).

Injects rank-r adapters into Q, K, V, O attention projections (and
optionally the SwiGLU FFN gate/up/down) of all transformer layers:

    effective_weight = frozen_weight + (A @ B) * (alpha / rank)

B is zero-initialised so the model starts identical to the frozen
backbone. The effective weights flow through the standard
:class:`pawn.model.PAWNModel` forward pass — the trainer never sees
A/B individually; they live in a separate :class:`LoRAAdapter` PyTree
that the apply function uses to build the effective model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pawn.model import PAWNModel, TransformerLayer

__all__ = [
    "LoRAConfig",
    "LoRAAdapter",
    "init_lora_adapter",
    "apply_lora",
    "lora_filter",
]


LoRATargets = Literal["qkvo", "qv", "qkv"]


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA hyperparameters.

    ``rank`` is the bottleneck dimension; ``alpha`` is the LoRA scaling
    factor (defaults to ``rank`` if None per Hu et al. convention).
    ``targets`` selects which attention projections get LoRA; ``ffn``
    toggles LoRA on the SwiGLU gate/up/down.
    """

    rank: int
    alpha: float | None = None
    targets: LoRATargets = "qkvo"
    ffn: bool = False

    @property
    def scaling(self) -> float:
        return (self.alpha if self.alpha is not None else float(self.rank)) / self.rank


class LoRAAdapter(eqx.Module):
    """LoRA A/B parameter pairs for each targeted projection.

    Each pair is ``(A[n_layers, d, r], B[n_layers, r, d])`` so the
    effective ``[n_layers, d, d]`` correction is
    ``jnp.einsum("ldr,lre->lde", A, B) * scaling``.

    Fields are ``None`` when the corresponding projection isn't
    LoRA-targeted (e.g. ``targets="qv"`` leaves ``A_k`` / ``A_o`` as
    ``None``). Equinox treats ``None`` leaves as no-ops in the
    filter / partition machinery.
    """

    A_q: Float[Array, "n_layers d r"] | None
    B_q: Float[Array, "n_layers r d"] | None
    A_k: Float[Array, "n_layers d r"] | None
    B_k: Float[Array, "n_layers r d"] | None
    A_v: Float[Array, "n_layers d r"] | None
    B_v: Float[Array, "n_layers r d"] | None
    A_o: Float[Array, "n_layers d r"] | None
    B_o: Float[Array, "n_layers r d"] | None
    # FFN LoRA (when cfg.ffn=True; None otherwise).
    A_gate: Float[Array, "n_layers d r"] | None
    B_gate: Float[Array, "n_layers r d_ff"] | None
    A_up: Float[Array, "n_layers d r"] | None
    B_up: Float[Array, "n_layers r d_ff"] | None
    A_down: Float[Array, "n_layers d_ff r"] | None
    B_down: Float[Array, "n_layers r d"] | None
    # Static config for scaling factor.
    cfg: LoRAConfig = eqx.field(static=True)


def _targets_to_flags(targets: LoRATargets) -> dict[str, bool]:
    return {
        "q": "q" in targets,
        "k": "k" in targets,
        "v": "v" in targets,
        "o": "o" in targets,
    }


def init_lora_adapter(
    backbone: PAWNModel, cfg: LoRAConfig, key: jax.Array | int
) -> LoRAAdapter:
    """Build a fresh LoRA adapter against ``backbone``.

    A matrices: kaiming-uniform (sqrt(5)) — matches the v1 init.
    B matrices: zero — model starts identical to the frozen backbone.
    """
    if isinstance(key, int):
        key = jax.random.key(key)

    d = backbone.cfg.d_model
    d_ff = backbone.cfg.d_ff
    n_layers = backbone.cfg.n_layers
    r = cfg.rank
    flags = _targets_to_flags(cfg.targets)

    def kaiming(k: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        # Kaiming-uniform matching PyTorch `kaiming_uniform_(a=sqrt(5))`
        # — the v1 init this adapter is meant to mirror. For
        # `a=sqrt(5)`, `gain = sqrt(2 / (1 + 5)) = sqrt(1/3)`, and
        # `bound = gain * sqrt(3 / fan_in) = sqrt(1/fan_in)`.
        # An earlier version computed `sqrt(2/fan_in)` (gain=1.0
        # default), inflating LoRA A-matrices by sqrt(2) relative to
        # v1; this restored the v1 contract.
        if len(shape) == 0:
            raise ValueError("kaiming requires at least a 1-D shape")
        fan_in = shape[-1] if len(shape) >= 2 else shape[0]
        bound = math.sqrt(1.0 / fan_in)
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    keys = jax.random.split(key, 16)

    def maybe_A(active: bool, k: jax.Array, fan_in: int) -> jax.Array | None:
        if not active:
            return None
        return kaiming(k, (n_layers, fan_in, r))

    def maybe_B(active: bool, fan_out: int) -> jax.Array | None:
        if not active:
            return None
        return jnp.zeros((n_layers, r, fan_out), dtype=jnp.float32)

    return LoRAAdapter(
        A_q=maybe_A(flags["q"], keys[0], d), B_q=maybe_B(flags["q"], d),
        A_k=maybe_A(flags["k"], keys[1], d), B_k=maybe_B(flags["k"], d),
        A_v=maybe_A(flags["v"], keys[2], d), B_v=maybe_B(flags["v"], d),
        A_o=maybe_A(flags["o"], keys[3], d), B_o=maybe_B(flags["o"], d),
        A_gate=maybe_A(cfg.ffn, keys[4], d), B_gate=maybe_B(cfg.ffn, d_ff),
        A_up=maybe_A(cfg.ffn, keys[5], d), B_up=maybe_B(cfg.ffn, d_ff),
        A_down=maybe_A(cfg.ffn, keys[6], d_ff), B_down=maybe_B(cfg.ffn, d),
        cfg=cfg,
    )


def _add_lora_correction(
    weight: jax.Array, A: jax.Array | None, B: jax.Array | None, scaling: float
) -> jax.Array:
    """Return ``weight + (A @ B) * scaling`` (per-layer leading axis)."""
    if A is None or B is None:
        return weight
    correction = jnp.einsum("ldr,lre->lde", A, B) * scaling
    return weight + correction


def apply_lora(backbone: PAWNModel, adapter: LoRAAdapter) -> PAWNModel:
    """Return a new :class:`PAWNModel` with effective LoRA-corrected weights.

    Reuses every backbone field except the targeted projection
    weights, which become ``frozen + LoRA(A, B)``. The autograd graph
    runs only through the LoRA A/B params — the backbone arrays are
    untouched.
    """
    s = adapter.cfg.scaling
    layers = backbone.layers
    new_layers = TransformerLayer(
        attn_norm_w=layers.attn_norm_w,
        wq=_add_lora_correction(layers.wq, adapter.A_q, adapter.B_q, s),
        wk=_add_lora_correction(layers.wk, adapter.A_k, adapter.B_k, s),
        wv=_add_lora_correction(layers.wv, adapter.A_v, adapter.B_v, s),
        wo=_add_lora_correction(layers.wo, adapter.A_o, adapter.B_o, s),
        ffn_norm_w=layers.ffn_norm_w,
        w_gate=_add_lora_correction(layers.w_gate, adapter.A_gate, adapter.B_gate, s),
        w_up=_add_lora_correction(layers.w_up, adapter.A_up, adapter.B_up, s),
        w_down=_add_lora_correction(layers.w_down, adapter.A_down, adapter.B_down, s),
    )
    return PAWNModel(
        embed_src=backbone.embed_src,
        embed_dst=backbone.embed_dst,
        embed_promo=backbone.embed_promo,
        embed_pad=backbone.embed_pad,
        embed_outcome=backbone.embed_outcome,
        layers=new_layers,
        final_norm_w=backbone.final_norm_w,
        lm_head=backbone.lm_head,
        decomp_table=backbone.decomp_table,
        cfg=backbone.cfg,
    )


def lora_filter(adapter: LoRAAdapter) -> LoRAAdapter:
    """All inexact-array leaves on LoRAAdapter are trainable; cfg is
    static. `eqx.partition` keys off this filter."""
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
