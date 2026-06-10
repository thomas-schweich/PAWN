"""LoRA + FiLM hybrid adapter — both methods composed on the same backbone.

Apply LoRA's effective-weight corrections first (folded into the
backbone's weight tensors), then wrap the LoRA-corrected backbone with
true FiLM (residual-stream ``gamma * h + beta`` + optional output-logit
modulation). LoRA returns a folded :class:`PAWNModel`; FiLM wraps it in
a :class:`FiLMEffective`, so the composition is
``FiLMEffective(backbone=lora_corrected_backbone)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax

from pawn.adapters.film import (
    FiLMAdapter,
    FiLMConfig,
    FiLMEffective,
    apply_film,
    init_film_adapter,
)
from pawn.adapters.lora import (
    LoRAAdapter,
    LoRAConfig,
    apply_lora,
    init_lora_adapter,
)
from pawn.model import PAWNModel

__all__ = [
    "HybridConfig",
    "HybridAdapter",
    "init_hybrid_adapter",
    "apply_hybrid",
    "hybrid_filter",
]


@dataclass(frozen=True)
class HybridConfig:
    """Hybrid = LoRA + FiLM. Both sub-configs are exposed verbatim."""

    lora: LoRAConfig
    film: FiLMConfig = field(default_factory=FiLMConfig)


class HybridAdapter(eqx.Module):
    """Holds both sub-adapters; apply_hybrid composes them."""

    lora: LoRAAdapter
    film: FiLMAdapter
    cfg: HybridConfig = eqx.field(static=True)


def init_hybrid_adapter(
    backbone: PAWNModel, cfg: HybridConfig, key: jax.Array | int
) -> HybridAdapter:
    if isinstance(key, int):
        key = jax.random.key(key)
    k1, k2 = jax.random.split(key, 2)
    return HybridAdapter(
        lora=init_lora_adapter(backbone, cfg.lora, k1),
        film=init_film_adapter(backbone, cfg.film, k2),
        cfg=cfg,
    )


def apply_hybrid(backbone: PAWNModel, adapter: HybridAdapter) -> FiLMEffective:
    """LoRA then FiLM — the v1 layering convention.

    LoRA folds into the backbone weights (returns a :class:`PAWNModel`);
    FiLM then wraps that LoRA-corrected backbone in a
    :class:`FiLMEffective` so its residual-stream / output-logit
    modulation rides on top of the LoRA corrections.
    """
    return apply_film(apply_lora(backbone, adapter.lora), adapter.film)


def hybrid_filter(adapter: HybridAdapter) -> HybridAdapter:
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
