"""RoSA — Robust Sparse Adaptation (Nikdan et al. 2024).

Three modes (all in scope per plan §6 "non-negotiable"):

- ``rosa`` (standard) — sparse delta + low-rank LoRA correction.
- ``retro-sparse`` — sparse-only after LoRA warmup drops the LoRA branch.
- ``retro-bottleneck`` — sparse delta + Houlsby bottleneck residual
  (no LoRA in the final phase; bottleneck replaces it).

All three modes share the three-phase schedule (Phase 1 LoRA warmup →
Phase 2 gradient-magnitude mask gen → Phase 3 joint training under
fixed mask); the orchestrator lives in
:mod:`pawn.adapter_trainer` (:func:`run_rosa_schedule`). The trio of
strategies (``rosa``, ``rosa-retro-sparse``, ``rosa-retro-bottleneck``)
dispatch through the same init + filter; ``apply_rosa`` branches on
``cfg.mode`` to compose the right effective model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax

from pawn.adapters.bottleneck import (
    BottleneckAdapter,
    BottleneckConfig,
    BottleneckEffective,
    apply_bottleneck,
    init_bottleneck_adapter,
)
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

    ``lora_targets`` / ``lora_ffn`` / ``sparse_targets`` / ``sparse_ffn``
    forward the v1 LoRA+sparse target knobs into RoSA's sub-adapters, so
    a user can RoSA-adapt a non-default projection set (e.g.
    ``targets="qv"`` per v1 ATTN_PRESETS) or include FFN projections.

    ``bottleneck_dim`` controls the Houlsby bottleneck width when
    ``mode="retro-bottleneck"`` (the bottleneck is unused for the other
    two modes).
    """

    mode: RoSAMode = "rosa"
    rosa_warmup_steps: int = 128
    mask_samples: int = 32
    grad_alpha: Literal[1, 2] = 2
    lora_rank: int = 4
    density: float = 0.01
    lora_targets: Literal["qkvo", "qv", "qkv"] = "qkvo"
    lora_ffn: bool = False
    sparse_targets: Literal["qkvo", "qv", "qkv"] = "qkvo"
    sparse_ffn: bool = False
    bottleneck_dim: int = 8


class RoSAAdapter(eqx.Module):
    """Composite of a LoRA, a Sparse, and (for retro-bottleneck) a
    Houlsby Bottleneck sub-adapter.

    The two-phase init starts every branch in its identity-at-step-0
    layout. ``cfg.mode`` selects which branches contribute to the
    effective forward; the trainer's three-phase schedule enables /
    re-initialises the right branches per phase.

    ``active`` toggles whether the LoRA and sparse branches contribute
    to the effective forward at each step. Phase 1 sets
    ``(lora_active=True, sparse_active=False)``; Phase 2 generates the
    mask off LoRA-only gradients; Phase 3 sets
    ``(lora_active=mode == "rosa", sparse_active=True)`` so retro-sparse
    and retro-bottleneck drop the LoRA contribution after warmup.
    """

    lora: LoRAAdapter
    sparse: SparseAdapter
    # `bottleneck` is only populated when cfg.mode == "retro-bottleneck";
    # the field carries None otherwise so eqx.is_inexact_array filters
    # treat it as a no-op.
    bottleneck: BottleneckAdapter | None
    cfg: RoSAConfig = eqx.field(static=True)
    # Per-phase toggles. Equinox `eqx.field(static=True)` keeps them out
    # of the optimizer's update path; the trainer rebuilds the adapter
    # with new toggles between phases.
    lora_active: bool = eqx.field(static=True, default=True)
    sparse_active: bool = eqx.field(static=True, default=False)


def init_rosa_adapter(
    backbone: PAWNModel, cfg: RoSAConfig, key: jax.Array | int
) -> RoSAAdapter:
    if isinstance(key, int):
        key = jax.random.key(key)
    k1, k2, k3 = jax.random.split(key, 3)
    lora_cfg = LoRAConfig(
        rank=cfg.lora_rank, targets=cfg.lora_targets, ffn=cfg.lora_ffn,
    )
    sparse_cfg = SparseConfig(
        density=cfg.density, targets=cfg.sparse_targets, ffn=cfg.sparse_ffn,
    )
    bottleneck = (
        init_bottleneck_adapter(
            backbone, BottleneckConfig(dim=cfg.bottleneck_dim), k3,
        )
        if cfg.mode == "retro-bottleneck"
        else None
    )
    # Initial phase: LoRA warmup, sparse disabled. The trainer flips
    # `sparse_active` on entering Phase 3 (and disables `lora_active` for
    # retro-sparse / retro-bottleneck).
    return RoSAAdapter(
        lora=init_lora_adapter(backbone, lora_cfg, k1),
        sparse=init_sparse_adapter(backbone, sparse_cfg, k2),
        bottleneck=bottleneck,
        cfg=cfg,
        lora_active=True,
        sparse_active=False,
    )


def apply_rosa(
    backbone: PAWNModel, adapter: RoSAAdapter
) -> PAWNModel | BottleneckEffective:
    """Compose backbone with the active RoSA branches.

    - ``rosa`` (standard): backbone + LoRA + sparse delta (both
      branches active throughout Phase 3).
    - ``retro-sparse``: backbone + sparse delta (LoRA dropped after
      Phase 2; the LoRA sub-adapter still lives in the PyTree but its
      contribution is masked).
    - ``retro-bottleneck``: backbone + sparse delta + Houlsby
      bottleneck (LoRA dropped, bottleneck replaces it).

    The ``*_active`` toggles let the same ``apply_rosa`` function serve
    every phase of the schedule without re-initialising the PyTree
    shape.
    """
    effective: PAWNModel | BottleneckEffective = backbone
    if adapter.sparse_active:
        effective = apply_sparse(effective, adapter.sparse)
    if adapter.lora_active:
        effective = apply_lora(effective, adapter.lora)
    if adapter.bottleneck is not None and adapter.sparse_active:
        # Bottleneck only contributes after the sparse phase starts —
        # during LoRA warmup the bottleneck would be a second
        # parameter-efficient branch competing with LoRA, which isn't
        # the v1 retro-bottleneck schedule.
        effective = apply_bottleneck(effective, adapter.bottleneck)
    return effective


def rosa_filter(adapter: RoSAAdapter) -> RoSAAdapter:
    """Mark every inexact-array leaf trainable.

    Inactive branches (per ``lora_active`` / ``sparse_active``) are
    skipped by :func:`apply_rosa`, so their gradients are
    zero-by-construction. Optax-level weight-decay drift on inactive
    LoRA/bottleneck leaves is forward-invisible because the next call
    to :func:`apply_rosa` simply ignores those leaves; once the
    schedule flips a branch back on, the small drift is dominated by
    Phase 3's gradient updates.
    """
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
