"""JAX pretraining trainer — single-step primitives.

This module provides the *primitives* the pretraining loop needs:

* ``cross_entropy_loss`` — masked next-token cross-entropy on logits.
* ``compute_variant_loss`` — forward a single ``PAWNModel`` (the supernet
  or a sliced variant) on one batch and return the mean masked CE.
* ``compute_supernet_loss`` — sum the per-variant losses on a shared
  batch (the joint-supernet objective from
  ``docs/jax-migration.md`` §5.3).
* ``make_train_step`` — jitted ``(state, batch) -> (state, metrics)``
  that applies one optimizer step to the supernet against the joint
  objective.

The K-step ``lax.scan`` outer loop, prefetcher, LR schedule, and metrics
sink land in subsequent Phase-2 chunks. This chunk is verifiably-correct
single-step machinery only — a chunked review can scrutinize loss math
+ gradient flow without dragging in scheduler / metrics / driver
complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pawn.jax.config import ModelConfig
from pawn.jax.model import PAWNModel, sliced


class TrainState(NamedTuple):
    """Per-step training state.

    ``model`` is the full supernet; ``opt_state`` is Optax's state for
    the same PyTree. ``step`` is a plain Python int that JIT closes over
    via the metrics return — it is *not* a JAX array so the host can
    decide when to log / checkpoint / decay without recompiling.
    """

    model: PAWNModel
    opt_state: optax.OptState
    step: int


@dataclass(frozen=True)
class VariantSpec:
    """One slice of the supernet contributing to the joint loss.

    ``cfg`` is the variant's ``ModelConfig`` (must satisfy
    ``validate_nested`` against the supernet); ``weight`` scales its
    contribution in the sum. Default weights are 1.0; future
    sandwich-rule schedules (§5.3) live at the call site, not here.
    """

    cfg: ModelConfig
    weight: float = 1.0


def cross_entropy_loss(
    logits: jax.Array,        # bf16/fp32  [B, T, V]
    targets: jax.Array,       # int32      [B, T]
    loss_mask: jax.Array,     # bool       [B, T]
) -> jax.Array:
    """Mean masked next-token cross-entropy.

    Returns a scalar fp32 loss. ``loss_mask=False`` positions
    contribute zero and do not increment the denominator. An
    all-False mask returns ``0`` (vs. raising) so the trainer can
    safely tolerate fully-padded batches in scan padding.
    """
    # Cast to fp32 for the log-softmax — bf16 logsumexp loses too much
    # mantissa near the maximum logit. The forward stays bf16; the
    # loss reduction is fp32.
    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    # Gather log-prob of the target token at each position.
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    )[..., 0]
    mask_f32 = loss_mask.astype(jnp.float32)
    neg_log_lik = -target_log_probs * mask_f32
    total = neg_log_lik.sum()
    n = mask_f32.sum()
    # Safe divide: when n == 0, total is also 0 (the masked-zero
    # multiplications are exactly 0), so total / max(n, 1) = 0.
    # No ``jnp.where`` branch needed and pyright sees a single
    # ``Array`` rather than the multi-overload union of ``jnp.where``.
    return total / jnp.maximum(n, 1.0)


def compute_variant_loss(
    model: PAWNModel,
    tokens: jax.Array,        # int32  [B, T]
    attn_mask: jax.Array,     # bool   [B, T]
    targets: jax.Array,       # int32  [B, T]
    loss_mask: jax.Array,     # bool   [B, T]
) -> jax.Array:
    """Forward ``model`` and return masked CE on the supplied batch."""
    logits = model(tokens, attn_mask)
    return cross_entropy_loss(logits, targets, loss_mask)


def compute_supernet_loss(
    supernet: PAWNModel,
    variant_specs: tuple[VariantSpec, ...],
    tokens: jax.Array,
    attn_mask: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Sum per-variant losses on the shared batch (§5.3).

    Returns ``(joint_loss, per_variant_dict)`` — the joint loss is the
    weighted sum used for backprop; the dict has one entry per variant
    keyed by ``str(cfg.d_model)`` so the host can log per-slice
    convergence.

    Static unroll: each variant has a different width, so they can't
    ``vmap`` into one batched matmul. Three forward calls per step is
    the documented cost.
    """
    per_variant: dict[str, jax.Array] = {}
    joint: jax.Array = jnp.float32(0.0)
    for spec in variant_specs:
        if spec.cfg == supernet.cfg:
            # No-op slice — call the supernet directly to avoid the
            # PyTree shape-copy and keep the trace pure.
            sub = supernet
        else:
            sub = sliced(supernet, spec.cfg)
        ce = compute_variant_loss(sub, tokens, attn_mask, targets, loss_mask)
        joint = joint + spec.weight * ce
        per_variant[f"d{spec.cfg.d_model}"] = ce
    return joint, per_variant


def make_optimizer(
    learning_rate: float, *, weight_decay: float = 0.01, b1: float = 0.9, b2: float = 0.95
) -> optax.GradientTransformation:
    """Default AdamW. Hyperparameters mirror the legacy pretraining
    defaults (``b2=0.95`` from the legacy trainer config) so a
    sane Phase-2 verification run doesn't need any tuning to get
    well-conditioned gradients.

    ``learning_rate`` is a fixed scalar here; the LR-schedule integration
    lands in the next Phase-2 chunk (``optax.chain(optax.adamw(...),
    optax.scale_by_schedule(...))`` is a one-line swap).
    """
    return optax.adamw(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        weight_decay=weight_decay,
    )


def init_train_state(
    model: PAWNModel, optimizer: optax.GradientTransformation
) -> TrainState:
    """Build the initial ``TrainState`` for ``model`` under ``optimizer``."""
    # Equinox separates trainable arrays from static attrs via
    # ``eqx.filter`` (anything that's not a JAX array is treated as
    # static under JIT). The optimizer state only mirrors the
    # array-leaf subtree.
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    return TrainState(model=model, opt_state=opt_state, step=0)


def make_train_step(
    optimizer: optax.GradientTransformation,
    variant_specs: tuple[VariantSpec, ...],
):
    """Build a jitted ``train_step(state, batch) -> (state, metrics)``.

    ``batch`` is a 4-tuple ``(tokens, attn_mask, targets, loss_mask)``
    sized ``[B, T]`` each. ``metrics`` is a dict with keys ``loss``
    (joint), ``loss_d<N>`` per variant, and ``grad_norm`` (global).
    """

    def loss_fn(
        model: PAWNModel,
        batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        tokens, attn_mask, targets, loss_mask = batch
        joint, per_variant = compute_supernet_loss(
            model, variant_specs, tokens, attn_mask, targets, loss_mask
        )
        return joint, per_variant

    @eqx.filter_jit
    def train_step(
        state: TrainState,
        batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        (joint_loss, per_variant), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(state.model, batch)
        # Pass only the array-leaf PyTree to Optax. Optax's stubs type
        # ``params`` as ``Array | None`` but in practice accept any
        # PyTree; ``eqx.filter`` produces a tree of arrays + None where
        # the static-attribute leaves were, which matches the stub.
        params = eqx.filter(state.model, eqx.is_inexact_array)
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, params
        )
        new_model = eqx.apply_updates(state.model, updates)

        # Global grad norm — only over the array leaves Optax saw.
        grad_arrays = jax.tree_util.tree_leaves(
            eqx.filter(grads, eqx.is_inexact_array)
        )
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in grad_arrays)
        )

        metrics: dict[str, jax.Array] = {
            "loss": joint_loss,
            "grad_norm": grad_norm,
            **{f"loss_{k}": v for k, v in per_variant.items()},
        }
        new_state = TrainState(
            model=new_model, opt_state=new_opt_state, step=state.step + 1
        )
        return new_state, metrics

    return train_step
