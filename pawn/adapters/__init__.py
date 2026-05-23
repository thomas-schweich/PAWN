"""PAWN adapters — strategies for fine-tuning a frozen supernet backbone.

8 strategies dispatch through this package per plan §10 S7:

- ``lora`` — Low-rank attention (Hu et al. 2021).
- ``film`` — channel-wise affine on hidden states (Perez et al. 2018).
- ``bottleneck`` — Houlsby et al. 2019 MLP bottleneck adapter.
- ``hybrid`` — LoRA + FiLM jointly.
- ``sparse`` — Sparse binary mask on the attention projections.
- ``rosa`` — Gradient-informed sparse + LoRA, three modes
  (``rosa`` / ``retro-sparse`` / ``retro-bottleneck``).
- ``unfreeze`` — Explicit per-layer unfreeze (v1 ``"5,6,7"`` form).
- ``specialized_clm`` — From-scratch standalone transformer.

Each module exports:

- A ``<Name>Config`` dataclass with the strategy's hyperparameters.
- ``init_<name>_adapter(backbone, cfg, key)`` constructor returning
  an Adapter PyTree.
- ``apply_<name>(backbone, adapter)`` that returns the effective
  :class:`pawn.model.PAWNModel` (backbone + adapter correction).
- ``<name>_filter(adapter)`` — PyTree-of-bool for
  :func:`eqx.partition` (selects the adapter's trainable arrays).

The trainer in :mod:`pawn.adapter_trainer` dispatches through a
unified table keyed by strategy name.
"""

from pawn.adapters import (
    bottleneck,
    film,
    hybrid,
    lora,
    rosa,
    sparse,
    specialized_clm,
    unfreeze,
)
from pawn.adapters.bottleneck import (
    BottleneckAdapter,
    BottleneckConfig,
    init_bottleneck_adapter,
)
from pawn.adapters.film import FiLMAdapter, FiLMConfig, init_film_adapter
from pawn.adapters.hybrid import HybridAdapter, HybridConfig, init_hybrid_adapter
from pawn.adapters.lora import LoRAAdapter, LoRAConfig, init_lora_adapter
from pawn.adapters.rosa import RoSAAdapter, RoSAConfig, init_rosa_adapter
from pawn.adapters.sparse import SparseAdapter, SparseConfig, init_sparse_adapter
from pawn.adapters.specialized_clm import (
    SpecializedCLMAdapter,
    SpecializedCLMConfig,
    init_specialized_clm_adapter,
)
from pawn.adapters.unfreeze import (
    UnfreezeAdapter,
    UnfreezeConfig,
    init_unfreeze_adapter,
)

STRATEGIES = (
    "lora",
    "film",
    "bottleneck",
    "hybrid",
    "sparse",
    "rosa",
    "rosa-retro-sparse",
    "rosa-retro-bottleneck",
    "unfreeze",
    "specialized_clm",
)

__all__ = [
    "STRATEGIES",
    "LoRAAdapter",
    "LoRAConfig",
    "init_lora_adapter",
    "FiLMAdapter",
    "FiLMConfig",
    "init_film_adapter",
    "BottleneckAdapter",
    "BottleneckConfig",
    "init_bottleneck_adapter",
    "HybridAdapter",
    "HybridConfig",
    "init_hybrid_adapter",
    "SparseAdapter",
    "SparseConfig",
    "init_sparse_adapter",
    "RoSAAdapter",
    "RoSAConfig",
    "init_rosa_adapter",
    "UnfreezeAdapter",
    "UnfreezeConfig",
    "init_unfreeze_adapter",
    "SpecializedCLMAdapter",
    "SpecializedCLMConfig",
    "init_specialized_clm_adapter",
    "bottleneck",
    "film",
    "hybrid",
    "lora",
    "rosa",
    "sparse",
    "specialized_clm",
    "unfreeze",
]
