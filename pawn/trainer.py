"""Pretraining trainer: JAX/Optax + lax.scan K-step inner loop + supernet joint loss.

The v2 pretrain trainer compiles the K-step inner loop into one
:func:`jax.lax.scan` so the per-step body never returns to the host;
per-chunk metrics flush to disk between scans. The supernet joint
loss sums per-variant cross-entropies on the same batch — that's what
v1's ``pawn.cotrain.py`` provided and is now the only pretrain path.

Public surface:

- :class:`Batch` — per-step input (tokens / targets / attn_mask /
  loss_mask) sliced from a :class:`pawn.corpus.Corpus`.
- :class:`TrainState` — eqx.Module wrapping (model, opt_state, step,
  key). ``step`` is a JAX scalar so JIT doesn't recompile each call.
- :class:`VariantSpec` — name + ModelConfig + ``is_supernet`` flag;
  the supernet trainer iterates over these and sums per-variant CEs.
- :func:`cross_entropy_loss` — masked CE over the full vocab.
- :func:`make_lr_schedule` — warmup + (cosine / wsd / constant /
  one_cycle / infinite) Optax schedule. The cross-field validators
  on :class:`pawn.run_config.BaseRunConfig` are what bound the
  fraction values; the trainer just stitches the Optax pieces.
- :func:`make_optimizer` — ``optax.chain(clip_by_global_norm(1.0),
  adamw(lr_schedule, weight_decay=wd))`` with a `lax.cond` guard
  against padded-batch weight-decay drift (skip update when the
  loss mask is empty).
- :func:`make_train_step` — `@eqx.filter_jit` single training step
  with the supernet joint loss baked in.
- :func:`make_scan_step` — wraps a single train step into a K-step
  :func:`jax.lax.scan` for amortised host overhead.
- :func:`supernet_joint_loss` — the variants-summed CE referenced
  inside :func:`make_train_step` (exposed so tests + sweeps can call
  it directly without going through the JIT wrapper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Bool, Float, Int

from pawn.config import ModelConfig
from pawn.corpus import Corpus
from pawn.model import PAWNModel, sliced
from pawn.run_config import BaseRunConfig

__all__ = [
    "Batch",
    "TrainState",
    "VariantSpec",
    "cross_entropy_loss",
    "supernet_joint_loss",
    "make_lr_schedule",
    "make_optimizer",
    "make_train_step",
    "make_scan_step",
    "slice_batch",
]


_CLIP_NORM: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Per-step input
# ---------------------------------------------------------------------------


class Batch(eqx.Module):
    """A single training batch — slice of a :class:`pawn.corpus.Corpus`.

    Fields:
        tokens: ``(B, T)`` int32 input IDs.
        targets: ``(B, T)`` int32 left-shifted targets.
        attn_mask: ``(B, T)`` bool — True for real tokens.
        loss_mask: ``(B, T)`` bool — True at supervised positions.
    """

    tokens: Int[Array, "B T"]
    targets: Int[Array, "B T"]
    attn_mask: Bool[Array, "B T"]
    loss_mask: Bool[Array, "B T"]


def slice_batch(corpus: Corpus, indices: np.ndarray) -> Batch:
    """Materialise a :class:`Batch` from a host-side Corpus + a numpy
    array of game indices. Performs the host → device transfer at the
    boundary."""
    return Batch(
        tokens=jnp.asarray(corpus.tokens[indices]),
        targets=jnp.asarray(corpus.targets[indices]),
        attn_mask=jnp.asarray(corpus.attn_mask[indices]),
        loss_mask=jnp.asarray(corpus.loss_mask[indices]),
    )


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


class TrainState(eqx.Module):
    """Wraps the model, optimizer state, step counter, and RNG.

    ``step`` is held as a JAX scalar (``jnp.int32(...)``) so the JIT
    cache keys on shape, not value. A Python-int step would re-trace
    every iteration.
    """

    model: PAWNModel
    opt_state: optax.OptState
    step: Int[Array, ""]
    key: jax.Array


# ---------------------------------------------------------------------------
# Per-variant spec for supernet joint loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    """One variant in the supernet joint loss.

    ``name`` is for logging; ``cfg`` is the variant's :class:`ModelConfig`;
    ``is_supernet`` is True for the "large" variant (= the supernet
    itself), in which case ``pawn.model.sliced`` is a no-op and the
    full model goes through the forward pass directly.
    """

    name: str
    cfg: ModelConfig
    is_supernet: bool = False


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    model: PAWNModel, batch: Batch
) -> Float[Array, ""]:
    """Masked cross-entropy on a single variant + batch.

    Returns the mean per-supervised-position loss. PAD positions
    (``loss_mask`` False) don't contribute. The output is a 0-d scalar
    JAX array.

    The denominator is ``loss_mask.sum().clip(min=1)`` — a fully-padded
    batch returns 0 / 1 = 0 (the optimizer should be a no-op then,
    which is what the `lax.cond` guard in :func:`make_optimizer` is
    for).
    """
    logits = model(batch.tokens, batch.attn_mask)  # (B, T, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_lp = jnp.take_along_axis(
        log_probs, batch.targets[..., None], axis=-1
    ).squeeze(-1)  # (B, T)
    neg_lp = -target_lp * batch.loss_mask
    n_real = jnp.maximum(batch.loss_mask.sum(), 1)
    return neg_lp.sum() / n_real


def supernet_joint_loss(
    model: PAWNModel,
    batch: Batch,
    variants: tuple[VariantSpec, ...],
) -> Float[Array, ""]:
    """The supernet joint loss: **sum** per-variant cross-entropies on
    the same batch.

    Refuses an empty `variants` tuple — an empty list would produce
    zero loss and gradients every step, advancing the optimizer state
    silently and applying weight-decay drift indefinitely. Better to
    crash at trainer init than to log a clean-looking training curve
    that isn't learning anything.

    Per plan §5 ("Joint training sums per-variant cross-entropies on
    the same batch") and §10 S6 ("sum the per-variant
    cross-entropies"). Sum (not mean) is what cotrain did in v1 and
    is what the supernet's gradient mathematics expect — each variant
    contributes its full per-supervised-position loss into the shared
    weight gradient.

    Variants are unrolled statically — the tuple is treated as a
    Python-static list so `lax.scan` over it would be wrong (variants
    have different shapes, so they can't be scan-stacked).
    """
    if not variants:
        raise ValueError(
            "supernet_joint_loss requires at least one VariantSpec; "
            "got an empty tuple"
        )
    total = jnp.array(0.0, dtype=jnp.float32)
    for spec in variants:
        if spec.is_supernet:
            sub_model = model
        else:
            sub_model = sliced(model, spec.cfg)
        total = total + cross_entropy_loss(sub_model, batch)
    return total


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _warmup_steps(cfg: BaseRunConfig, total_steps: int) -> int:
    """Resolve warmup_steps from the explicit override or the fraction.

    Refuses warmup > total_steps: optax's cosine builders compute
    `decay_steps - warmup_steps` and pass it to schedule constructors
    that reject negative values, so an out-of-range override would
    crash at trainer init rather than produce a sensible schedule.
    """
    warmup = cfg.warmup_steps if cfg.warmup_steps is not None else int(
        round(cfg.warmup_frac * total_steps)
    )
    if warmup > total_steps:
        raise ValueError(
            f"warmup_steps ({warmup}) exceeds total_steps ({total_steps}); "
            f"reduce warmup_frac or warmup_steps in the run config"
        )
    return warmup


def make_lr_schedule(
    cfg: BaseRunConfig, total_steps: int
) -> optax.Schedule:
    """Build the Optax schedule for ``cfg.lr_schedule`` over ``total_steps``.

    Five shapes:

    - ``cosine`` — :func:`optax.warmup_cosine_decay_schedule` with
      ``decay_steps=total_steps`` (the plan §10 S6 pinned contract: NOT
      ``total_steps - warmup``; the optax convention is that
      decay_steps is the full timeline length, not the post-warmup
      remainder).
    - ``constant`` — warmup ramp then flat at peak.
    - ``wsd`` — warmup → stable plateau → final decay (linear or
      cosine shape per ``wsd_decay_shape``).
    - ``one_cycle`` — peak/25 → peak over ``warmup_frac``, then
      cosine to peak/10000 over the rest.
    - ``infinite`` — warmup → cosine cooldown to
      ``stable_lr_ratio*peak`` → flat stable plateau → final decay
      to 0.
    """
    warmup = _warmup_steps(cfg, total_steps)
    peak = cfg.lr

    if cfg.lr_schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak,
            warmup_steps=warmup,
            decay_steps=total_steps,
            end_value=0.0,
        )
    if cfg.lr_schedule == "constant":
        return optax.join_schedules(
            [
                optax.linear_schedule(0.0, peak, warmup),
                optax.constant_schedule(peak),
            ],
            [warmup],
        )
    if cfg.lr_schedule == "wsd":
        decay_steps = int(round(cfg.decay_frac * total_steps))
        decay = (
            optax.linear_schedule(peak, 0.0, decay_steps)
            if cfg.wsd_decay_shape == "linear"
            else optax.cosine_decay_schedule(peak, decay_steps, 0.0)
        )
        return optax.join_schedules(
            [
                optax.linear_schedule(0.0, peak, warmup),
                optax.constant_schedule(peak),
                decay,
            ],
            [warmup, total_steps - decay_steps],
        )
    if cfg.lr_schedule == "one_cycle":
        init = peak / 25.0
        end = peak / 10000.0
        # Ramp warmup steps init → peak, then cosine for the remainder.
        remaining = total_steps - warmup
        return optax.join_schedules(
            [
                optax.linear_schedule(init, peak, warmup),
                optax.cosine_decay_schedule(peak, remaining, end / peak),
            ],
            [warmup],
        )
    if cfg.lr_schedule == "infinite":
        cooldown_steps = int(round(cfg.cooldown_frac * total_steps))
        decay_steps = int(round(cfg.decay_frac * total_steps))
        stable_lr = peak * cfg.stable_lr_ratio
        # Integer rounding of three fractions can sum to more than
        # total_steps even when the float fractions sum to ≤1
        # (BaseRunConfig validates floats, not rounded ints). Clamp to
        # avoid `join_schedules` getting non-monotonic boundaries.
        stable_steps = max(0, total_steps - warmup - cooldown_steps - decay_steps)
        cooldown = optax.cosine_decay_schedule(
            peak, cooldown_steps, cfg.stable_lr_ratio
        )
        decay = (
            optax.linear_schedule(stable_lr, 0.0, decay_steps)
            if cfg.wsd_decay_shape == "linear"
            else optax.cosine_decay_schedule(stable_lr, decay_steps, 0.0)
        )
        return optax.join_schedules(
            [
                optax.linear_schedule(0.0, peak, warmup),
                cooldown,
                optax.constant_schedule(stable_lr),
                decay,
            ],
            [warmup, warmup + cooldown_steps, warmup + cooldown_steps + stable_steps],
        )
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def make_optimizer(
    cfg: BaseRunConfig,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    """Build the v2 optimizer: gradient clip + AdamW.

    ``optax.chain(clip_by_global_norm(1.0), adamw(lr_schedule, weight_decay))``
    is the plan-pinned shape. Weight-decay is applied through Optax's
    AdamW (decoupled, scaled by lr). The padded-batch weight-decay
    drift guard isn't here at the optimizer level — it lives in
    :func:`make_train_step` where we can see whether the batch was
    empty.
    """
    return optax.chain(
        optax.clip_by_global_norm(_CLIP_NORM),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=cfg.weight_decay,
        ),
    )


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


def make_train_step(
    optimizer: optax.GradientTransformation,
    variants: tuple[VariantSpec, ...],
) -> Callable[[TrainState, Batch], tuple[TrainState, Float[Array, ""]]]:
    """Return a JIT-compiled single training step closing over the
    optimizer + variant list.

    The returned function has signature
    ``(state, batch) -> (new_state, loss)``. ``state.step`` is a JAX
    scalar so the JIT trace is value-independent — same compiled
    program for step 0 and step 999.

    Padded-batch weight-decay drift guard: when ``batch.loss_mask`` is
    all-False (an empty batch), the gradient is zero everywhere and
    the loss is 0, so the optimizer's `clip + adamw` would still apply
    `weight_decay * model_params` to every parameter — drifting the
    model toward zero across many padded batches.

    The guard uses ``jax.lax.cond`` to select between an "apply update"
    branch and a "skip update" branch. Under XLA both branches are
    traced and lowered into the executable, then `select` picks the
    correct outputs — so the cost saving is in the *output*
    (params unchanged), not the compute. The model + opt_state remain
    byte-identical when the batch is empty; ``state.step`` still
    advances (wall-clock counter, decoupled from optimizer progress).
    """

    @eqx.filter_jit(donate="all")
    def train_step(
        state: TrainState, batch: Batch
    ) -> tuple[TrainState, Float[Array, ""]]:
        def loss_fn(model: PAWNModel) -> Float[Array, ""]:
            return supernet_joint_loss(model, batch, variants)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model)

        # Empty-batch guard: if loss_mask sums to 0, the batch supervised
        # nothing — skip the optimizer update to avoid weight_decay
        # drifting the params toward zero.
        n_supervised = batch.loss_mask.sum()
        do_update = n_supervised > 0

        def apply_update(args):
            grads_, opt_state_, model_ = args
            updates, new_opt = optimizer.update(grads_, opt_state_, model_)
            new_model = eqx.apply_updates(model_, updates)
            return new_model, new_opt

        def skip_update(args):
            _, opt_state_, model_ = args
            return model_, opt_state_

        new_model, new_opt_state = jax.lax.cond(
            do_update,
            apply_update,
            skip_update,
            (grads, state.opt_state, state.model),
        )

        new_state = TrainState(
            model=new_model,
            opt_state=new_opt_state,
            step=state.step + jnp.int32(1),
            key=state.key,
        )
        return new_state, loss

    return train_step


# ---------------------------------------------------------------------------
# K-step scan
# ---------------------------------------------------------------------------


def make_scan_step(
    train_step: Callable[
        [TrainState, Batch], tuple[TrainState, Float[Array, ""]]
    ],
) -> Callable[[TrainState, Batch], tuple[TrainState, Float[Array, "K"]]]:
    """Wrap a single train step into a K-step :func:`jax.lax.scan`.

    Input is a ``Batch`` whose leaves have a leading K axis (so
    ``batch.tokens`` is ``(K, B, T)`` etc.). Output is the final state
    and a ``(K,)`` array of per-step losses.

    The body never returns to the host — that's the v2 amortisation.
    Per-chunk metrics flush between calls (the trainer loop drives the
    K-step boundaries from Python).
    """

    @eqx.filter_jit(donate="all")
    def scan_step(
        state: TrainState, batches: Batch
    ) -> tuple[TrainState, Float[Array, "K"]]:
        def body(
            carry: TrainState, batch: Batch
        ) -> tuple[TrainState, Float[Array, ""]]:
            new_carry, loss = train_step(carry, batch)
            return new_carry, loss

        final_state, losses = jax.lax.scan(body, state, batches)
        return final_state, losses

    return scan_step
