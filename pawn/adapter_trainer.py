"""Adapter training loop — two-tier frozen/trainable partition.

Builds on the pretraining trainer primitives but with three design
deltas per ``docs/jax-migration.md`` §6:

  * **Two-tier PyTree.** The model is partitioned into a trainable
    subtree (adapter parameters) and a frozen subtree (backbone).
    Only the trainable subtree receives gradients; XLA DCE removes
    backbone weight-gradient computations (~33% FLOP cut, §6.1).
  * **Finite dataset.** Adapter training consumes a fixed corpus
    (typically a Lichess Elo-band slice). Per-epoch index
    permutations replace the infinite-stream sequential scan.
  * **Validation.** A second jitted forward-only callable evaluates
    the (eqx.combined) model on a held-out split; the host can use
    val-loss to gate checkpointing / early-stop.

The single-step machinery (cross-entropy, gradient clipping, AdamW)
is reused from ``pawn.trainer`` so the same fixes (Codex P2 grad
clip, padded-batch lockstep, etc.) apply unchanged. This module adds:
  * ``make_adapter_train_step`` — single-step that consumes a
    partitioned ``(trainable, frozen)`` carry.
  * ``make_eval_step`` — forward-only loss over a single batch on
    the combined model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pawn.adapters import LoRAModel, adapter_filter
from pawn.trainer import Batch, compute_grad_norm, cross_entropy_loss


class AdapterTrainState(NamedTuple):
    """Per-step state for adapter training.

    Carries the *trainable* subtree of the partitioned ``LoRAModel``
    (the adapter parameters) plus the optimizer state and a 0-d
    ``jnp.int32`` step counter. The frozen backbone subtree lives
    outside the state — captured in the JIT closure of
    ``make_adapter_train_step``. Under ``eqx.filter_jit`` the
    closed-over JAX arrays are traced as dynamic constants (not
    truly static), but they no longer ride through any ``lax.scan``
    carry — that's the structural win. The real §6.1 saving is that
    the backbone arrays are not in ``trainable``, so gradient
    computations for them are dead-code-eliminated by XLA.
    """

    trainable: LoRAModel  # partition: lora.* are real arrays, backbone is None
    opt_state: optax.OptState
    step: jax.Array


AdapterTrainStep = Callable[
    [AdapterTrainState, Batch],
    tuple[AdapterTrainState, dict[str, jax.Array]],
]


def init_adapter_state(
    model: LoRAModel, optimizer: optax.GradientTransformation
) -> tuple[AdapterTrainState, LoRAModel]:
    """Partition ``model`` and build the initial training state.

    Returns ``(state, frozen)``. The caller passes ``frozen`` as a
    closure to ``make_adapter_train_step`` and ``make_eval_step``;
    ``state.trainable`` is what gets gradient-updated.
    """
    trainable, frozen = eqx.partition(model, adapter_filter(model))
    # Optax sees only the array leaves of the trainable subtree —
    # everything in ``frozen`` is None and is silently ignored.
    params = eqx.filter(trainable, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    state = AdapterTrainState(
        trainable=trainable,
        opt_state=opt_state,
        step=jnp.zeros((), dtype=jnp.int32),
    )
    return state, frozen


def make_adapter_train_step(
    optimizer: optax.GradientTransformation,
    frozen: LoRAModel,
) -> AdapterTrainStep:
    """Build a jitted ``(state, batch) -> (state, metrics)`` callable.

    The ``frozen`` PyTree is captured in the JIT closure rather than
    threaded through a scan carry. Under ``eqx.filter_jit`` its array
    leaves are traced as dynamic constants and are re-traced if the
    closure object identity changes; the §6.1 win is that they are
    NOT in ``trainable`` and so XLA dead-code-eliminates their
    weight-gradient computations. Mirrors the pretraining trainer's
    behaviour around padded batches (``has_supervision`` gates the
    optimizer update) and uses the same gradient-clipping
    composition that ``make_optimizer`` already wires in.
    """

    @eqx.filter_jit
    def adapter_train_step(
        state: AdapterTrainState, batch: Batch
    ) -> tuple[AdapterTrainState, dict[str, jax.Array]]:
        tokens, attn_mask, targets, loss_mask = batch

        def loss_fn(trn: LoRAModel) -> jax.Array:
            model = eqx.combine(trn, frozen)
            logits = model(tokens, attn_mask)
            return cross_entropy_loss(logits, targets, loss_mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.trainable)

        # Optax accepts a PyTree of grads + matching params; both are
        # the trainable subtree (with backbone leaves as None).
        params = eqx.filter(state.trainable, eqx.is_inexact_array)
        grads_filt = eqx.filter(grads, eqx.is_inexact_array)

        # Global grad norm via the shared helper (same pattern as the
        # pretrain trainer; centralised so a future change lands in
        # one place).
        grad_norm = compute_grad_norm(grads)

        n_supervised = loss_mask.sum().astype(jnp.int32)
        has_supervision = n_supervised > 0

        def _apply(args: tuple[LoRAModel, optax.OptState]) -> tuple[
            LoRAModel, optax.OptState
        ]:
            trn_in, opt_in = args
            updates, new_opt = optimizer.update(grads_filt, opt_in, params)
            new_trn = eqx.apply_updates(trn_in, updates)
            return new_trn, new_opt

        def _skip(args: tuple[LoRAModel, optax.OptState]) -> tuple[
            LoRAModel, optax.OptState
        ]:
            return args

        new_trainable, new_opt_state = jax.lax.cond(
            has_supervision, _apply, _skip, (state.trainable, state.opt_state)
        )

        new_step = state.step + jnp.where(
            has_supervision, jnp.int32(1), jnp.int32(0)
        )
        new_state = AdapterTrainState(
            trainable=new_trainable,
            opt_state=new_opt_state,
            step=new_step,
        )
        metrics: dict[str, jax.Array] = {
            "loss": loss,
            "grad_norm": grad_norm,
            "n_supervised": n_supervised,
        }
        return new_state, metrics

    return adapter_train_step


AdapterScanStep = Callable[
    [AdapterTrainState, Batch],
    tuple[AdapterTrainState, dict[str, jax.Array]],
]


def make_adapter_scan_step(
    train_step: AdapterTrainStep, k: int
) -> AdapterScanStep:
    """Fuse the adapter ``train_step`` into a K-step ``lax.scan``.

    The driver previously called ``train_step`` K times from a Python
    loop, paying one XLA dispatch per call (~K × dispatch overhead
    per chunk). Wrapping in ``lax.scan`` reduces that to one
    dispatch per chunk and gives XLA the full K-iteration body for
    fusion. Chunk argument shape: ``[K, B, T]`` per field.
    """

    def scan_step(
        state: AdapterTrainState, chunk: Batch
    ) -> tuple[AdapterTrainState, dict[str, jax.Array]]:
        return jax.lax.scan(train_step, state, chunk, length=k)

    return eqx.filter_jit(scan_step)


EvalStep = Callable[[LoRAModel, Batch], dict[str, jax.Array]]


def make_eval_step(frozen: LoRAModel) -> EvalStep:
    """Build a jitted forward-only ``(trainable, batch) -> metrics``
    callable.

    Used by the driver to compute held-out val loss. No gradient
    flow, no optimizer state, no weight updates — purely a forward
    + cross-entropy on the eqx-combined model.
    """

    @eqx.filter_jit
    def eval_step(trn: LoRAModel, batch: Batch) -> dict[str, jax.Array]:
        tokens, attn_mask, targets, loss_mask = batch
        model = eqx.combine(trn, frozen)
        logits = model(tokens, attn_mask)
        loss = cross_entropy_loss(logits, targets, loss_mask)
        n_supervised = loss_mask.sum().astype(jnp.int32)
        return {"loss": loss, "n_supervised": n_supervised}

    return eval_step
