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

The trainer is *strategy-agnostic*: it works for any adapter type
that exposes an ``adapter_filter(model)`` returning a Python-bool
partition spec for ``eqx.partition``. Strategies that need
per-element gradient masking on top of the partition (today only
``unfreeze``'s per-layer slicing) pass an optional ``gradient_mask``
PyTree to ``make_adapter_train_step``; the trainer multiplies
gradients by the mask element-wise before the optimizer step. RoSA's
phase-dependent gating works by swapping the ``adapter_filter``
itself at the Phase 2 → 3 boundary, not via ``gradient_mask``.
``make_adapter_train_step`` and ``make_eval_step`` close over
``frozen`` (a captured JIT closure) rather than threading it through
a ``lax.scan`` carry — the structural win documented in
``AdapterTrainState``.

The single-step machinery (cross-entropy, gradient clipping, AdamW)
is reused from ``pawn.trainer`` so the same fixes (Codex P2 grad
clip, padded-batch lockstep, etc.) apply unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pawn.trainer import Batch, compute_grad_norm, cross_entropy_loss


class AdapterModelProto(Protocol):
    """Structural protocol for any adapter ``eqx.Module``.

    Every concrete adapter type (``LoRAModel``, ``FiLMModel``, etc.)
    defines ``__call__(tokens, attn_mask) -> logits``. The base
    ``eqx.Module`` doesn't declare a callable signature, so pyright
    rejects ``model(tokens, attn_mask)``. This protocol expresses the
    runtime contract — every value flowing through here implements
    it — without introducing a unifying base class. Public so that
    ``scripts/train_jax_adapter.py`` (and any other downstream caller
    that does ``cast(...)`` around an eqx-combined model) shares the
    one canonical shape.
    """

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array: ...

# Type alias: a filter-spec function for any adapter ``eqx.Module``.
# Returns a tree shaped like ``model`` whose leaves are Python ``bool``
# values (``eqx.partition`` consumes these directly) or callables.
# The argument is intentionally typed as ``...`` (variadic) so that
# per-strategy filters typed against their concrete model (e.g.
# ``adapter_filter(model: LoRAModel) -> LoRAModel``) assign cleanly —
# pyright treats ``Callable[[Module], Module]`` as contravariant in
# its argument, rejecting narrower-typed callables that are actually
# safe to call here (the dispatcher in
# ``scripts/train_jax_adapter.py`` always pairs a strategy's filter
# with its own model type).
AdapterFilterFn = Callable[..., eqx.Module]


class AdapterTrainState(NamedTuple):
    """Per-step state for adapter training.

    Carries the *trainable* subtree of the partitioned adapter model
    plus the optimizer state and a 0-d ``jnp.int32`` step counter.
    The frozen backbone subtree lives outside the state — captured
    in the JIT closure of ``make_adapter_train_step``. Under
    ``eqx.filter_jit`` the closed-over JAX arrays are traced as
    dynamic constants (not truly static), but they no longer ride
    through any ``lax.scan`` carry — that's the structural win. The
    real §6.1 saving is that the backbone arrays are not in
    ``trainable``, so gradient computations for them are dead-code-
    eliminated by XLA.

    The ``trainable`` field is annotated ``eqx.Module`` — concretely,
    it can be a ``LoRAModel``, ``FiLMModel``, ``BottleneckModel``,
    ``HybridModel``, ``SparseModel``, ``RoSAModel``,
    ``UnfreezeModel``, or a from-scratch ``PAWNModel``
    (``specialized_clm``). All eight strategy types post-partition
    carry the same PyTree shape contract (``None`` at frozen leaves,
    real arrays at trainable leaves).
    """

    trainable: eqx.Module
    opt_state: optax.OptState
    step: jax.Array


AdapterTrainStep = Callable[
    [AdapterTrainState, Batch],
    tuple[AdapterTrainState, dict[str, jax.Array]],
]


def init_adapter_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    *,
    adapter_filter_fn: AdapterFilterFn,
) -> tuple[AdapterTrainState, eqx.Module]:
    """Partition ``model`` via ``adapter_filter_fn`` and build the
    initial training state.

    Args:
        model: any adapter ``eqx.Module`` whose strategy module
            provides an ``adapter_filter`` function.
        optimizer: an Optax gradient transformation. Typically built
            via ``pawn.trainer.make_optimizer(lr_schedule)`` so that
            gradient clipping + AdamW + the LR schedule are wired
            consistently with pretraining.
        adapter_filter_fn: the strategy's ``adapter_filter`` —
            ``pawn.adapters.adapter_filter`` for LoRA,
            ``pawn.adapters.bottleneck_adapter_filter`` for
            bottleneck, etc. Must be passed explicitly so the caller
            (``scripts/train_jax_adapter.py``) is the single dispatch
            point that knows which strategy is in use.

    Returns:
        ``(state, frozen)``. The caller passes ``frozen`` as a
        closure to ``make_adapter_train_step`` and ``make_eval_step``;
        ``state.trainable`` is what gets gradient-updated.
    """
    trainable, frozen = eqx.partition(model, adapter_filter_fn(model))
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
    frozen: eqx.Module,
    *,
    gradient_mask: eqx.Module | None = None,
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

    Args:
        optimizer: the Optax gradient transformation that drives the
            update step.
        frozen: the frozen partition of the original model — what
            ``eqx.combine`` joins back with ``state.trainable`` to
            reconstitute the full model for the forward pass.
        gradient_mask: optional per-element bool-mask PyTree shaped
            like the trainable subtree's array leaves. When provided,
            gradients are multiplied by the mask element-wise before
            the optimizer step — used by ``unfreeze`` for per-layer
            slicing of layer-stacked weights. RoSA's phase gating
            uses an ``adapter_filter`` swap instead, not this mask.
    """

    # Pre-filter the gradient mask once (outside the JIT) so the inner
    # tree_map traverses identical structures. ``eqx.filter`` with
    # ``eqx.is_array`` keeps the bool-array leaves and drops scalar /
    # config-object leaves of the mask tree.
    if gradient_mask is not None:
        mask_filt = eqx.filter(gradient_mask, eqx.is_array)
    else:
        mask_filt = None

    @eqx.filter_jit
    def adapter_train_step(
        state: AdapterTrainState, batch: Batch
    ) -> tuple[AdapterTrainState, dict[str, jax.Array]]:
        tokens, attn_mask, targets, loss_mask = batch

        def loss_fn(trn: eqx.Module) -> jax.Array:
            model = cast(AdapterModelProto, eqx.combine(trn, frozen))
            logits = model(tokens, attn_mask)
            return cross_entropy_loss(logits, targets, loss_mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.trainable)

        # Optax accepts a PyTree of grads + matching params; both are
        # the trainable subtree (with backbone leaves as None).
        params = eqx.filter(state.trainable, eqx.is_inexact_array)
        grads_filt = eqx.filter(grads, eqx.is_inexact_array)

        # Element-wise gradient mask applied in TWO places:
        #
        # 1. PRE-optimizer (here): zero gradients at frozen positions
        #    so they don't pollute ``clip_by_global_norm`` (which
        #    operates on the global L2 norm across all leaves —
        #    large frozen-layer grads would inflate the norm and
        #    cause the actually-trainable grads to be clipped too
        #    aggressively) and Adam's moment estimates.
        # 2. POST-optimizer (inside ``_apply``): zero the final
        #    update at frozen positions so AdamW's decoupled
        #    weight-decay term (which is independent of the
        #    gradient) cannot move frozen params either.
        #
        # The two masks are redundant only at trainable positions
        # (where multiplying by 1 is a no-op); at frozen positions
        # the pre-mask handles gradient-norm leakage and the
        # post-mask handles weight-decay leakage. Removing either
        # leaves the corresponding leak path open (Codex P2 + test-
        # risk HIGH on round-1: post-only let weight decay through;
        # pre-only let weight decay through).
        if mask_filt is not None:
            grads_filt = jax.tree.map(
                lambda g, m: (
                    g if g is None or m is None
                    else g * m.astype(g.dtype)
                ),
                grads_filt, mask_filt,
                is_leaf=lambda x: x is None,
            )

        # Global grad norm via the shared helper (same pattern as the
        # pretrain trainer; centralised so a future change lands in
        # one place). Computed after the pre-mask so the reported
        # norm reflects the actually-applied gradients.
        grad_norm = compute_grad_norm(grads_filt)

        n_supervised = loss_mask.sum().astype(jnp.int32)
        has_supervision = n_supervised > 0

        # Thread ``grads_filt`` through the ``lax.cond`` operand tuple
        # rather than capturing it by closure. ``grads_filt`` is
        # reassigned by the pre-mask branch above (lines 231-239); a
        # future refactor that moved the masking below the ``_apply``
        # definition would silently feed unmasked grads to the
        # optimizer if the closure pattern stayed. ``params`` is
        # read-only inside ``_apply`` (no later mutation), so the
        # closure capture is safe.
        def _apply(args: tuple[
            eqx.Module, optax.OptState, optax.Updates,
        ]) -> tuple[eqx.Module, optax.OptState]:
            trn_in, opt_in, grads_in = args
            updates, new_opt = optimizer.update(grads_in, opt_in, params)
            if mask_filt is not None:
                # Post-optimizer mask zeros the final update at
                # frozen positions so AdamW's decoupled weight decay
                # cannot move them. The pre-mask above already
                # zeroed the gradient signal feeding Adam's moments
                # and the global-norm clipper.
                updates = jax.tree.map(
                    lambda u, m: (
                        u if u is None or m is None
                        else u * m.astype(u.dtype)
                    ),
                    updates, mask_filt,
                    is_leaf=lambda x: x is None,
                )
            new_trn = eqx.apply_updates(trn_in, updates)
            return new_trn, new_opt

        def _skip(args: tuple[
            eqx.Module, optax.OptState, optax.Updates,
        ]) -> tuple[eqx.Module, optax.OptState]:
            trn_in, opt_in, _grads_in = args
            return trn_in, opt_in

        new_trainable, new_opt_state = jax.lax.cond(
            has_supervision, _apply, _skip,
            (state.trainable, state.opt_state, grads_filt),
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


EvalStep = Callable[[eqx.Module, Batch], dict[str, jax.Array]]


def make_eval_step(frozen: eqx.Module) -> EvalStep:
    """Build a jitted forward-only ``(trainable, batch) -> metrics``
    callable.

    Used by the driver to compute held-out val loss. No gradient
    flow, no optimizer state, no weight updates — purely a forward
    + cross-entropy on the eqx-combined model.
    """

    @eqx.filter_jit
    def eval_step(trn: eqx.Module, batch: Batch) -> dict[str, jax.Array]:
        tokens, attn_mask, targets, loss_mask = batch
        model = cast(AdapterModelProto, eqx.combine(trn, frozen))
        logits = model(tokens, attn_mask)
        loss = cross_entropy_loss(logits, targets, loss_mask)
        n_supervised = loss_mask.sum().astype(jnp.int32)
        return {"loss": loss, "n_supervised": n_supervised}

    return eval_step
