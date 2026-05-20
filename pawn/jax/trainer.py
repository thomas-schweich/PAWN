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

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pawn.jax.config import ModelConfig
from pawn.jax.model import PAWNModel, sliced

# A pretraining batch — the four parallel ``[B, T]`` arrays the trainer
# consumes. Aliased so the trainer signature and any test helpers can
# share a single shape contract; ``*batch`` unpacking thereby stays
# type-correct without ``# type: ignore``.
Batch = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
TrainStep = Callable[["TrainState", Batch], tuple["TrainState", dict[str, jax.Array]]]


class TrainState(NamedTuple):
    """Per-step training state.

    ``model`` is the full supernet; ``opt_state`` is Optax's state for
    the same PyTree. ``step`` is a 0-d ``jnp.int32`` traced into the
    state PyTree — it MUST be a JAX array, not a Python int, because
    ``eqx.filter_jit`` would treat a Python int as a static cache key
    and recompile on every increment (a ~70× per-call slowdown). The
    host can still read it via ``int(state.step)`` after a step.
    The same convention works seamlessly when ``train_step`` is later
    folded into a ``jax.lax.scan`` body (chunk 2.3).
    """

    model: PAWNModel
    opt_state: optax.OptState
    step: jax.Array


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


def _variant_key(cfg: ModelConfig) -> str:
    """Stable, unique key for a variant in the per-variant metrics dict.

    Includes width, depth, and FFN size so two specs that happen to
    share ``d_model`` but differ in ``n_layers`` or ``d_ff`` (a valid
    sandwich-rule configuration) don't silently overwrite each other.
    """
    return f"d{cfg.d_model}_L{cfg.n_layers}_F{cfg.d_ff}"


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
    weighted sum used for backprop; the per-variant dict is keyed
    ``d<W>_L<L>_F<F>`` (e.g. ``d192_L4_F768``) so the host can log
    per-slice convergence and two specs sharing only ``d_model`` stay
    distinguishable.

    Static unroll: each variant has a different width, so they can't
    ``vmap`` into one batched matmul. Three forward calls per step is
    the documented cost. The identity-slice shortcut (when
    ``spec.cfg == supernet.cfg``) saves a trace-time Python pass —
    no measurable execution-time win, since XLA constant-folds
    static slicing — but it keeps the trace tree clean.
    """
    per_variant: dict[str, jax.Array] = {}
    joint: jax.Array = jnp.float32(0.0)
    for spec in variant_specs:
        if spec.cfg == supernet.cfg:
            # No-op slice — direct supernet forward.
            sub = supernet
        else:
            sub = sliced(supernet, spec.cfg)
        ce = compute_variant_loss(sub, tokens, attn_mask, targets, loss_mask)
        joint = joint + spec.weight * ce
        per_variant[_variant_key(spec.cfg)] = ce
    return joint, per_variant


def make_lr_schedule(
    peak_lr: float,
    *,
    total_steps: int,
    warmup_steps: int = 100,
    end_value: float | None = None,
) -> optax.Schedule:
    """Linear warmup → cosine decay (Optax canonical).

    Args:
        peak_lr: LR reached at the end of warmup.
        total_steps: full schedule length; decay_steps = total - warmup.
        warmup_steps: linear ramp from 0 to ``peak_lr``.
        end_value: floor at ``total_steps``; defaults to ``peak_lr * 0.1``.

    Past ``total_steps`` the schedule plateaus at ``end_value``, so
    going over budget (e.g. a continuation run) won't drive LR
    negative.
    """
    if end_value is None:
        end_value = peak_lr * 0.1
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps={warmup_steps} must be < total_steps={total_steps}"
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=end_value,
    )


def make_optimizer(
    learning_rate: float | optax.Schedule,
    *,
    weight_decay: float = 0.01,
    b1: float = 0.9,
    b2: float = 0.95,
) -> optax.GradientTransformation:
    """AdamW with optional LR schedule.

    Hyperparameters mirror the legacy pretraining defaults
    (``b2=0.95`` from the legacy trainer config). ``learning_rate``
    accepts either a scalar (constant LR — convenient for tests) or
    an ``optax.Schedule`` (callable ``step -> lr``). For Phase-2
    pretraining pair this with ``make_lr_schedule`` for the
    canonical warmup→cosine path.
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
    """Build the initial ``TrainState`` for ``model`` under ``optimizer``.

    ``step`` is a 0-d ``jnp.int32`` so it traces as a dynamic array
    rather than a static Python int (the latter would cause
    ``eqx.filter_jit`` to recompile on every increment).
    """
    # Equinox separates trainable arrays from static attrs via
    # ``eqx.filter`` (anything that's not a JAX array is treated as
    # static under JIT). The optimizer state only mirrors the
    # array-leaf subtree.
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    return TrainState(
        model=model, opt_state=opt_state, step=jnp.zeros((), dtype=jnp.int32)
    )


def make_scan_step(
    train_step: TrainStep, k: int
) -> Callable[
    [TrainState, Batch], tuple[TrainState, dict[str, jax.Array]]
]:
    """Fuse ``train_step`` into a K-step ``lax.scan`` loop (§4.4).

    The returned callable consumes a *chunk* batch of shape ``[K, B, T]``
    (per-step batches stacked on a leading axis) and applies
    ``train_step`` K times, returning ``(final_state, stacked_metrics)``
    where ``stacked_metrics[k]`` is the metrics dict produced at step
    ``k``. The host loop unstacks for logging.

    K should be chosen to amortise host overhead — for PAWN-base on a
    B200 the docs target K=200 (~52 MB per chunk staged to device while
    the previous chunk trains).

    Designed for use in chunk 2.4's driver: the driver pre-stages each
    ``[K, ...]`` chunk to device, calls this, then logs / checkpoints
    based on the host-side step counter ``int(state.step)``.
    """

    def scan_step(state: TrainState, chunk: Batch) -> tuple[TrainState, dict[str, jax.Array]]:
        tokens_chunk, attn_chunk, target_chunk, loss_chunk = chunk

        # Each [K, B, T] slice along axis 0 is one batch.
        def body(
            inner_state: TrainState, slice_idx: jax.Array
        ) -> tuple[TrainState, dict[str, jax.Array]]:
            batch: Batch = (
                tokens_chunk[slice_idx],
                attn_chunk[slice_idx],
                target_chunk[slice_idx],
                loss_chunk[slice_idx],
            )
            new_state, metrics = train_step(inner_state, batch)
            return new_state, metrics

        return jax.lax.scan(body, state, jnp.arange(k, dtype=jnp.int32))

    return eqx.filter_jit(scan_step)


def make_train_step(
    optimizer: optax.GradientTransformation,
    variant_specs: tuple[VariantSpec, ...],
) -> TrainStep:
    """Build a jitted ``train_step(state, batch) -> (state, metrics)``.

    ``batch`` is a 4-tuple ``(tokens, attn_mask, targets, loss_mask)``
    sized ``[B, T]`` each. ``metrics`` is a dict with keys ``loss``
    (joint), ``loss_<key>`` per variant (key is ``d<W>_L<L>_F<F>``),
    ``grad_norm`` (global L2), and ``n_supervised`` (count of
    True entries in ``loss_mask``).

    A batch whose ``loss_mask`` is entirely False (the scan-padding
    edge case) bypasses ``optimizer.update`` so neither weights nor
    optimizer state mutate — AdamW's decoupled weight decay would
    otherwise still drift weights on zero-gradient padded steps.
    Returned ``loss`` and ``grad_norm`` are 0 in that case.
    """

    @eqx.filter_jit
    def train_step(
        state: TrainState, batch: Batch
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        tokens, attn_mask, targets, loss_mask = batch
        (joint_loss, per_variant), grads = eqx.filter_value_and_grad(
            lambda m: compute_supernet_loss(
                m, variant_specs, tokens, attn_mask, targets, loss_mask
            ),
            has_aux=True,
        )(state.model)

        # ``eqx.filter_value_and_grad`` returns grads as a PyTree with
        # ``None`` at every non-inexact-array leaf. Treat that as the
        # canonical array subtree — no second ``eqx.filter`` needed.
        # ``jax.tree_util.tree_leaves`` skips ``None`` by the standard
        # JAX pytree convention.
        grad_leaves = jax.tree_util.tree_leaves(grads)

        # Global grad norm via flatten + single dot (one BLAS kernel
        # vs 16 separate reductions + 15 adds). Cast each leaf to fp32
        # once during flatten; the dot stays fp32.
        if grad_leaves:
            flat = jnp.concatenate(
                [g.astype(jnp.float32).ravel() for g in grad_leaves]
            )
            grad_norm = jnp.sqrt(jnp.dot(flat, flat))
        else:
            # Frozen-everywhere edge: no array leaves to backprop into.
            # Shouldn't fire for the supernet but keeps the metric
            # well-typed for the future adapter path.
            grad_norm = jnp.float32(0.0)

        n_supervised = loss_mask.sum().astype(jnp.int32)
        has_supervision = n_supervised > 0

        # Padding-only batch (all ``loss_mask=False``) → joint_loss is
        # 0 and grads are 0, but AdamW's decoupled weight decay would
        # still mutate weights and advance optimizer state. Skip the
        # update entirely in that case; ``jax.lax.cond`` is the
        # cheapest way under JIT.
        def _apply_update(
            args: tuple[PAWNModel, optax.OptState, PAWNModel],
        ) -> tuple[PAWNModel, optax.OptState]:
            model_in, opt_state_in, grads_in = args
            # Filter both the params (``model``) and updates (``grads``)
            # down to the array-leaf PyTree. Optax's stubs type
            # ``updates`` / ``params`` as ``Array``-flavoured Updates;
            # in practice optax accepts any matching PyTree. Filtering
            # explicitly keeps both pyright happy and the contract
            # symmetric (params and grads have identical shape).
            params_in = eqx.filter(model_in, eqx.is_inexact_array)
            grads_filt = eqx.filter(grads_in, eqx.is_inexact_array)
            updates_in, new_opt_state_in = optimizer.update(
                grads_filt, opt_state_in, params_in
            )
            new_model_in = eqx.apply_updates(model_in, updates_in)
            return new_model_in, new_opt_state_in

        def _skip_update(
            args: tuple[PAWNModel, optax.OptState, PAWNModel],
        ) -> tuple[PAWNModel, optax.OptState]:
            model_in, opt_state_in, _ = args
            return model_in, opt_state_in

        # NOTE: optimizer.update + apply_updates use the same array
        # subtree shape as grads / state.model, so the two branches of
        # ``lax.cond`` agree on PyTree structure.
        new_model, new_opt_state = jax.lax.cond(
            has_supervision,
            _apply_update,
            _skip_update,
            (state.model, state.opt_state, grads),
        )

        metrics: dict[str, jax.Array] = {
            "loss": joint_loss,
            "grad_norm": grad_norm,
            "n_supervised": n_supervised,
            **{f"loss_{k}": v for k, v in per_variant.items()},
        }
        new_state = TrainState(
            model=new_model,
            opt_state=new_opt_state,
            step=state.step + jnp.int32(1),
        )
        return new_state, metrics

    return train_step
