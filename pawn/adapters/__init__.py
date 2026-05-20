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
* ``bottleneck`` — Houlsby residual MLP after attention + FFN
  sublayers (Houlsby et al. 2019).
* ``hybrid`` — LoRA + FiLM stacked.
* ``sparse`` — Learnable continuous (or hard, via straight-through
  estimator) mask on attention projections.
* ``unfreeze`` — Partial fine-tune of the top ``n_unfreeze`` backbone
  layers. The Python-bool ``adapter_filter`` marks each backbone
  field fully True or fully False at coarse granularity; per-layer
  slicing (top-N only) is layered on top via the companion
  ``unfreeze_gradient_mask`` which returns the per-element bool
  mask the trainer plugs into ``optax.masked``.

* ``rosa`` — Gradient-informed sparse + LoRA (Nikdan et al. 2024).
  This module ships the parameterisation + ``set_mask`` helper; the
  legacy three-phase training schedule itself is a trainer-side
  concern that lands when the adapter trainer driver picks up RoSA
  dispatch.
* ``specialized_clm`` — From-scratch standalone transformer (not
  really an adapter — separate from the backbone entirely; the
  module is a thin wrapper around ``pawn.model.init_model`` for
  dispatch parity with the legacy ``--strategy`` table).
"""

from pawn.adapters.bottleneck import (
    BottleneckConfig,
    BottleneckModel,
    BottleneckParams,
)
from pawn.adapters.bottleneck import (
    adapter_filter as bottleneck_adapter_filter,
)
from pawn.adapters.bottleneck import (
    init_bottleneck_model,
)
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
from pawn.adapters.hybrid import (
    HybridConfig,
    HybridModel,
)
from pawn.adapters.hybrid import (
    adapter_filter as hybrid_adapter_filter,
)
from pawn.adapters.hybrid import (
    init_hybrid_model,
)
from pawn.adapters.lora import (
    LoRAConfig,
    LoRAModel,
    adapter_filter,
    init_lora_model,
)
from pawn.adapters.rosa import (
    RoSAConfig,
    RoSAModel,
    RoSAParams,
)
from pawn.adapters.rosa import (
    adapter_filter as rosa_adapter_filter,
)
from pawn.adapters.rosa import (
    init_rosa_model,
    set_mask as rosa_set_mask,
)
from pawn.adapters.specialized_clm import (
    SpecializedCLMConfig,
)
from pawn.adapters.specialized_clm import (
    adapter_filter as specialized_clm_adapter_filter,
)
from pawn.adapters.specialized_clm import (
    init_specialized_clm,
)
from pawn.adapters.sparse import (
    SparseConfig,
    SparseModel,
    SparseParams,
)
from pawn.adapters.sparse import (
    adapter_filter as sparse_adapter_filter,
)
from pawn.adapters.sparse import (
    init_sparse_model,
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
    make_gradient_mask as unfreeze_gradient_mask,
)

__all__ = [
    "BottleneckConfig",
    "BottleneckModel",
    "BottleneckParams",
    "FiLMConfig",
    "FiLMModel",
    "FiLMParams",
    "HybridConfig",
    "HybridModel",
    "LoRAConfig",
    "LoRAModel",
    "RoSAConfig",
    "RoSAModel",
    "RoSAParams",
    "SparseConfig",
    "SparseModel",
    "SparseParams",
    "SpecializedCLMConfig",
    "UnfreezeConfig",
    "UnfreezeModel",
    "adapter_filter",
    "bottleneck_adapter_filter",
    "film_adapter_filter",
    "hybrid_adapter_filter",
    "init_bottleneck_model",
    "init_film_model",
    "init_hybrid_model",
    "init_lora_model",
    "init_rosa_model",
    "init_specialized_clm",
    "init_sparse_model",
    "init_unfreeze_model",
    "rosa_adapter_filter",
    "rosa_set_mask",
    "sparse_adapter_filter",
    "specialized_clm_adapter_filter",
    "unfreeze_adapter_filter",
    "unfreeze_gradient_mask",
]
