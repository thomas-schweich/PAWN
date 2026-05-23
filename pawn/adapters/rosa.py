"""RoSA — Robust Sparse Adaptation (Nikdan et al. 2024).

Three modes (all in scope per plan §6 "non-negotiable"):

- ``rosa`` (standard) — sparse delta + low-rank LoRA correction.
- ``retro-sparse`` — RoSA's "retroactive sparse" variant.
- ``retro-bottleneck`` — RoSA's "retroactive bottleneck" variant.

The full three-phase schedule (Phase 1 LoRA warmup → Phase 2
gradient-magnitude mask gen → Phase 3 joint training under fixed mask)
is owned by the adapter trainer, not this module. Here we provide the
PyTree shape and the `apply` function that composes sparse + LoRA
contributions on the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import equinox as eqx
import jax

from pawn.adapters.lora import LoRAAdapter, LoRAConfig, apply_lora, init_lora_adapter
from pawn.adapters.sparse import (
    SparseAdapter,
    SparseConfig,
    apply_sparse,
    init_sparse_adapter,
)
from pawn.model import PAWNModel

__all__ = [
    "RoSAConfig",
    "RoSAAdapter",
    "RoSAMode",
    "init_rosa_adapter",
    "apply_rosa",
    "rosa_filter",
]


RoSAMode = Literal["rosa", "retro-sparse", "retro-bottleneck"]


@dataclass(frozen=True)
class RoSAConfig:
    """RoSA hyperparameters, v1 names preserved.

    ``mode`` selects the variant. ``rosa_warmup_steps`` controls the
    Phase 1 LoRA warmup duration. ``mask_samples`` is the number of
    mini-batches sampled in Phase 2 to estimate gradient magnitudes.
    ``grad_alpha`` is the moment order (1 = absolute, 2 = squared).
    """

    mode: RoSAMode = "rosa"
    rosa_warmup_steps: int = 128
    mask_samples: int = 32
    grad_alpha: Literal[1, 2] = 2
    lora_rank: int = 4
    density: float = 0.01


class RoSAAdapter(eqx.Module):
    """Composite of a LoRA sub-adapter (Phase 1 + Phase 3) and a Sparse
    sub-adapter (Phase 2 mask + Phase 3 trainable delta).

    The two-phase init starts both with their identity-at-step-0
    layouts; the trainer's Phase 2 freezes LoRA and regenerates the
    sparse mask off gradient magnitudes before unfreezing both for
    Phase 3.
    """

    lora: LoRAAdapter
    sparse: SparseAdapter
    cfg: RoSAConfig = eqx.field(static=True)


def init_rosa_adapter(
    backbone: PAWNModel, cfg: RoSAConfig, key: jax.Array | int
) -> RoSAAdapter:
    if isinstance(key, int):
        key = jax.random.key(key)
    k1, k2 = jax.random.split(key, 2)
    lora_cfg = LoRAConfig(rank=cfg.lora_rank, targets="qkvo")
    sparse_cfg = SparseConfig(density=cfg.density, targets="qkvo")
    return RoSAAdapter(
        lora=init_lora_adapter(backbone, lora_cfg, k1),
        sparse=init_sparse_adapter(backbone, sparse_cfg, k2),
        cfg=cfg,
    )


def apply_rosa(backbone: PAWNModel, adapter: RoSAAdapter) -> PAWNModel:
    """Compose: backbone → +sparse delta → +LoRA correction."""
    return apply_lora(apply_sparse(backbone, adapter.sparse), adapter.lora)


def rosa_filter(adapter: RoSAAdapter) -> RoSAAdapter:
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
