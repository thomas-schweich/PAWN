"""Adapter strategies for JAX PAWN models.

Each adapter is an ``eqx.Module`` that wraps a frozen ``PAWNModel``
backbone and adds a small set of trainable parameters (or, for the
``unfreeze`` strategy, marks an existing subset of backbone params
as trainable). The two-tier partition (frozen backbone / trainable
adapter) lets ``jax.grad`` / ``eqx.filter_grad`` differentiate only
the trainable subtree; XLA dead-code-eliminates the frozen-leaf
gradients for the ~33% FLOP cut described in
``docs/jax-migration.md`` §6.1.

The canonical adapter-only partition is::

    trainable, frozen = eqx.partition(model, adapter_filter(model))

Each strategy ships its own ``adapter_filter`` that ``True``-marks
the leaves the optimizer should update.

Ported strategies (``docs/jax-migration.md`` §9 Phase 3):

* ``lora`` — Low-rank update on attention projections
  (Hu et al. 2021).
* ``film`` — Per-channel post-layer affine modulation
  (Perez et al. 2017).
* ``unfreeze`` — Partial fine-tune of the top ``n_unfreeze`` backbone
  layers (no new parameters; trainable subtree is a per-layer mask
  applied to the existing backbone leaves).

Not yet ported (follow-up PRs; see the final migration PR for the
list and ``docs/jax-migration.md`` for status):

* ``bottleneck`` — Houlsby residual MLP (Houlsby et al. 2019).
* ``hybrid`` — LoRA + FiLM stacked.
* ``sparse`` — Learnable mask on backbone weights with
  straight-through estimator.
* ``rosa`` — Gradient-informed sparse + LoRA, 3-phase training.
* ``specialized_clm`` — From-scratch standalone transformer (not
  really an adapter — separate from the backbone entirely).
"""

from pawn.adapters.film import (
    FiLMConfig,
    FiLMModel,
    FiLMParams,
)
from pawn.adapters.film import (
    adapter_filter as film_adapter_filter,
)
from pawn.adapters.film import (
    init_film_model,
)
from pawn.adapters.lora import (
    LoRAConfig,
    LoRAModel,
    adapter_filter,
    init_lora_model,
)
from pawn.adapters.unfreeze import (
    UnfreezeConfig,
    UnfreezeModel,
)
from pawn.adapters.unfreeze import (
    adapter_filter as unfreeze_adapter_filter,
)
from pawn.adapters.unfreeze import (
    init_unfreeze_model,
)

__all__ = [
    "FiLMConfig",
    "FiLMModel",
    "FiLMParams",
    "LoRAConfig",
    "LoRAModel",
    "UnfreezeConfig",
    "UnfreezeModel",
    "adapter_filter",
    "film_adapter_filter",
    "init_film_model",
    "init_lora_model",
    "init_unfreeze_model",
    "unfreeze_adapter_filter",
]
