"""Adapter strategies for JAX PAWN models.

Each adapter is an ``eqx.Module`` that wraps a frozen ``PAWNModel``
backbone and adds a small set of trainable parameters. The two-tier
partition (frozen backbone / trainable adapter) lets
``jax.grad`` / ``eqx.filter_grad`` differentiate only the adapter
parameters; XLA dead-code-eliminates the backbone weight-gradients
for the ~33% FLOP cut described in ``docs/jax-migration.md`` §6.1.

The canonical adapter-only partition is:

    trainable, frozen = eqx.partition(model, adapter_filter(model))

Phase-3 chunk-1 ships LoRA only. Bottleneck/FiLM/Hybrid land in
subsequent chunks.
"""

from pawn.adapters.lora import (
    LoRAConfig,
    LoRAModel,
    adapter_filter,
    init_lora_model,
)

__all__ = [
    "LoRAConfig",
    "LoRAModel",
    "adapter_filter",
    "init_lora_model",
]
