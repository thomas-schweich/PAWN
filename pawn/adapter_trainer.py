"""Adapter trainer — two-tier PyTree partition + K-step lax.scan.

The trainer takes a frozen backbone + an adapter PyTree and trains
only the adapter's parameters. The two-tier partition uses
:func:`eqx.partition` to separate trainable (adapter) from frozen
(backbone) — :func:`jax.grad` then differentiates only the trainable
PyTree, and XLA dead-code-eliminates the backbone weight-gradients
(~33% backward-pass FLOP cut per plan §5).

Public surface:

- :class:`AdapterTrainState` — (backbone, adapter, opt_state, step,
  key). The backbone is held but never updated.
- :func:`make_adapter_train_step` — JIT'd single training step with
  the strategy's ``apply`` function baked in.
- :func:`make_adapter_scan_step` — K-step :func:`jax.lax.scan` wrapper.
- :data:`STRATEGIES` — the dispatch table mapping ``--strategy`` value
  to ``(init, apply, filter)``.
- :func:`forward_eval` — jitted forward-only eval (no gradients).

RoSA's three-phase schedule (LoRA warmup → mask-gen → joint training
under fixed mask) preserves ``state.step`` across the Phase 2→3
re-init so the metrics log stays monotonic; Optax-internal step resets
by design per plan §10 S7. The full schedule lives in a separate
helper (``run_rosa_schedule``) — adapter trainers for the other 7
strategies use the simple `make_adapter_train_step` path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

from pawn.adapters import (
    BottleneckAdapter,
    FiLMAdapter,
    HybridAdapter,
    LoRAAdapter,
    RoSAAdapter,
    SparseAdapter,
    SpecializedCLMAdapter,
    UnfreezeAdapter,
    bottleneck,
    film,
    hybrid,
    lora,
    rosa,
    sparse,
    specialized_clm,
    unfreeze,
)
from pawn.model import PAWNModel
from pawn.trainer import Batch, cross_entropy_loss

__all__ = [
    "AdapterTrainState",
    "STRATEGIES",
    "dispatch_init",
    "dispatch_apply",
    "dispatch_filter",
    "make_adapter_train_step",
    "make_adapter_scan_step",
    "forward_eval",
]


# ---------------------------------------------------------------------------
# Strategy dispatch table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyEntry:
    init: Callable[..., Any]
    apply: Callable[[PAWNModel, Any], PAWNModel]
    filter: Callable[[Any], Any]


STRATEGIES: dict[str, StrategyEntry] = {
    "lora": StrategyEntry(lora.init_lora_adapter, lora.apply_lora, lora.lora_filter),
    "film": StrategyEntry(film.init_film_adapter, film.apply_film, film.film_filter),
    "bottleneck": StrategyEntry(
        bottleneck.init_bottleneck_adapter,
        bottleneck.apply_bottleneck,
        bottleneck.bottleneck_filter,
    ),
    "hybrid": StrategyEntry(
        hybrid.init_hybrid_adapter, hybrid.apply_hybrid, hybrid.hybrid_filter
    ),
    "sparse": StrategyEntry(
        sparse.init_sparse_adapter, sparse.apply_sparse, sparse.sparse_filter
    ),
    "rosa": StrategyEntry(
        rosa.init_rosa_adapter, rosa.apply_rosa, rosa.rosa_filter
    ),
    # Plan §10 S7 + CLAUDE.md adapter table treat the three RoSA modes as
    # distinct `--strategy` values. They share init/apply/filter — the
    # `mode` field on the config selects the variant. Whoever holds the
    # CLI surface (S13's scripts/train_jax_adapter.py) is expected to
    # default `rosa_mode` to match the strategy name when the user passes
    # `--strategy rosa-retro-sparse` etc.
    "rosa-retro-sparse": StrategyEntry(
        rosa.init_rosa_adapter, rosa.apply_rosa, rosa.rosa_filter
    ),
    "rosa-retro-bottleneck": StrategyEntry(
        rosa.init_rosa_adapter, rosa.apply_rosa, rosa.rosa_filter
    ),
    "unfreeze": StrategyEntry(
        unfreeze.init_unfreeze_adapter,
        unfreeze.apply_unfreeze,
        unfreeze.unfreeze_filter,
    ),
    "specialized_clm": StrategyEntry(
        specialized_clm.init_specialized_clm_adapter,
        specialized_clm.apply_specialized_clm,
        specialized_clm.specialized_clm_filter,
    ),
}


def dispatch_init(strategy: str) -> Callable[..., Any]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy].init


def dispatch_apply(strategy: str) -> Callable[[PAWNModel, Any], PAWNModel]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy].apply


def dispatch_filter(strategy: str) -> Callable[[Any], Any]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy].filter


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


class AdapterTrainState(eqx.Module):
    """State for adapter training.

    ``backbone`` is frozen (never updated by the optimizer);
    ``adapter`` holds the trainable parameters. ``opt_state`` tracks
    AdamW moments + clip + lr schedule. ``step`` is the JAX scalar
    counter.
    """

    backbone: PAWNModel
    adapter: Any  # one of the *Adapter types — eqx.Module
    opt_state: optax.OptState
    step: Int[Array, ""]
    key: jax.Array


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def make_adapter_train_step(
    strategy: str,
    optimizer: optax.GradientTransformation,
) -> Callable[
    [AdapterTrainState, Batch], tuple[AdapterTrainState, Float[Array, ""]]
]:
    """Build the JIT'd train step for ``strategy``.

    The closure captures the strategy's ``apply`` function so XLA can
    inline it. Gradients flow only through the adapter PyTree;
    ``backbone`` is held outside the autograd path via
    :func:`eqx.partition`.
    """

    apply_fn = dispatch_apply(strategy)

    @eqx.filter_jit(donate="all")
    def step(
        state: AdapterTrainState, batch: Batch
    ) -> tuple[AdapterTrainState, Float[Array, ""]]:
        def loss_fn(adapter: Any) -> Float[Array, ""]:
            effective = apply_fn(state.backbone, adapter)
            return cross_entropy_loss(effective, batch)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.adapter)

        # Empty-batch guard (mirror of pretrain trainer).
        do_update = batch.loss_mask.sum() > 0

        def apply_update(args):
            grads_, opt_state_, adapter_ = args
            updates, new_opt = optimizer.update(grads_, opt_state_, adapter_)
            new_adapter = eqx.apply_updates(adapter_, updates)
            return new_adapter, new_opt

        def skip_update(args):
            _, opt_state_, adapter_ = args
            return adapter_, opt_state_

        new_adapter, new_opt_state = jax.lax.cond(
            do_update,
            apply_update,
            skip_update,
            (grads, state.opt_state, state.adapter),
        )

        new_state = AdapterTrainState(
            backbone=state.backbone,
            adapter=new_adapter,
            opt_state=new_opt_state,
            step=state.step + jnp.int32(1),
            key=state.key,
        )
        return new_state, loss

    return step


def make_adapter_scan_step(
    train_step: Callable[
        [AdapterTrainState, Batch],
        tuple[AdapterTrainState, Float[Array, ""]],
    ],
) -> Callable[
    [AdapterTrainState, Batch], tuple[AdapterTrainState, Float[Array, "K"]]
]:
    """Wrap a single adapter train step in a K-step :func:`jax.lax.scan`.

    Input ``batches`` has a leading K axis on every Batch field.
    """

    @eqx.filter_jit(donate="all")
    def scan_step(
        state: AdapterTrainState, batches: Batch
    ) -> tuple[AdapterTrainState, Float[Array, "K"]]:
        def body(
            carry: AdapterTrainState, batch: Batch
        ) -> tuple[AdapterTrainState, Float[Array, ""]]:
            new_carry, loss = train_step(carry, batch)
            return new_carry, loss

        final_state, losses = jax.lax.scan(body, state, batches)
        return final_state, losses

    return scan_step


# ---------------------------------------------------------------------------
# Forward-only eval
# ---------------------------------------------------------------------------


@eqx.filter_jit
def forward_eval(
    backbone: PAWNModel, adapter: Any, batch: Batch, strategy: str
) -> Float[Array, "B T V"]:
    """JIT'd forward pass for eval — backbone + adapter → logits.

    Doesn't compute gradients; calls the strategy's ``apply`` function
    to build the effective model, then PAWNModel.__call__.
    """
    apply_fn = dispatch_apply(strategy)
    effective = apply_fn(backbone, adapter)
    return effective(batch.tokens, batch.attn_mask)
