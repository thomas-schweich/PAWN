"""Distillation trainer — logit-only teacher→student KL/CE distillation.

The canonical-ladder mechanism (plan §7): a frozen *teacher* supplies
soft targets that a from-scratch *student* matches via a temperature-
scaled KL divergence (optionally mixed with ground-truth cross-entropy).
This is the v2 substitute for the supernet's nested-slice trick — students
are independent models (specialized_clm shapes), not width slices, so
there is **no hidden-state matching**, only logit distillation.

Built for the adapted/conditioned-teacher generality of plan §7.1:

- **The teacher is an injectable** :data:`TeacherFn` — a plain
  ``(tokens, attn_mask) -> logits`` callable, not a hardcoded checkpoint.
  :func:`frozen_teacher` wraps any frozen :class:`pawn.model.PAWNModel`
  (including an ``eqx.combine``'d adapter or a conditioned forward) and
  runs it under :func:`jax.lax.stop_gradient` so the teacher receives
  exactly zero gradient.
- **The objective is pluggable** — :func:`distill_loss` selects between
  ``kl`` (temperature-scaled KL), ``ce`` (ground-truth cross-entropy), and
  ``mix`` (``alpha·ce + (1-alpha)·kl``) via the run config.
- **Unified with the adapter trainer** — the student is a trainable PyTree
  (a standalone :class:`pawn.model.PAWNModel` here). :func:`make_distill_train_step`
  mirrors :func:`pawn.adapter_trainer.make_adapter_train_step` (grad over
  the trainable leaves only; donated buffers), and
  :func:`make_distill_scan_step` mirrors
  :func:`pawn.adapter_trainer.make_adapter_scan_step` (K-step ``lax.scan``).

The KL is masked to the supervised move-token support (plan §7 / Phase-B
spec, Stage B1): columns ``[PAD_TOKEN .. V)`` — PAD, the outcome tokens,
BOS, NULL, and the reserved control block — are forced to ``-inf`` in
**both** the student and teacher logits before either softmax, so teacher
probability mass never spreads onto those columns and they accrue no
student gradient. This is a strictly larger mask than the integer-label CE
reserved block (``≥ NULL_TOKEN``, :func:`pawn.trainer.mask_reserved_columns`):
PAD / outcome / BOS are never CE targets, but as *soft* KL targets the
teacher would otherwise distil their mass into the student, so the
distillation path uses :func:`pawn.trainer.mask_distill_columns`. The
time-axis ``loss_mask`` (Phase-A :func:`pawn.corpus.build_loss_mask`)
restricts every objective to the supervised positions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Int

from pawn.model import EffectiveCallable, PAWNModel
from pawn.trainer import (
    Batch,
    cross_entropy_loss,
    mask_distill_columns,
    mask_reserved_columns,
)

__all__ = [
    "TeacherFn",
    "DistillObjective",
    "frozen_teacher",
    "kl_divergence_loss",
    "distill_loss",
    "DistillTrainState",
    "make_distill_train_step",
    "make_distill_scan_step",
]


# A teacher is any callable producing per-position logits for a batch.
# `frozen_teacher` is the canonical constructor, but the type is open so a
# conditioned / adapter-composed forward can be injected in its place
# (plan §7.1).
TeacherFn = Callable[
    [Int[Array, "B T"], Bool[Array, "B T"]], Float[Array, "B T V"]
]

DistillObjective = Literal["kl", "ce", "mix"]


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------


def frozen_teacher(
    model: "PAWNModel | EffectiveCallable",
    *,
    compute_dtype: jnp.dtype | None = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
) -> TeacherFn:
    """Wrap a frozen model as a :data:`TeacherFn`.

    The returned closure runs ``model`` forward and applies
    :func:`jax.lax.stop_gradient` to its **output logits**. Because the
    teacher's parameters only influence the loss through those logits,
    stopping the gradient at the output severs every backward path into the
    teacher's weights — the teacher receives exactly zero gradient even if
    the caller accidentally exposes it to ``jax.grad`` (asserted in
    ``tests/test_jax_distill.py::test_teacher_receives_zero_gradient``).

    Any frozen :class:`pawn.model.PAWNModel` works, as does an
    ``eqx.combine``'d adapter model or a conditioned forward — anything
    satisfying :class:`pawn.model.EffectiveCallable` (plan §7.1's
    adapted/conditioned-teacher generality).

    ``compute_dtype`` / ``use_sdpa`` / ``use_flash`` pick the teacher's
    forward precision + attention kernel; they default to the bit-stable
    fp32 plain-attention path so the soft targets are deterministic.
    """

    def teacher_fn(
        tokens: Int[Array, "B T"], attn_mask: Bool[Array, "B T"]
    ) -> Float[Array, "B T V"]:
        logits = model(
            tokens, attn_mask,
            compute_dtype=compute_dtype,
            use_sdpa=use_sdpa, use_flash=use_flash,
        )
        return jax.lax.stop_gradient(logits)

    return teacher_fn


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _masked_log_softmax(
    logits: Float[Array, "B T V"], temperature: float
) -> Float[Array, "B T V"]:
    """Temperature-scaled, distill-column-masked log-softmax.

    Columns ``[PAD_TOKEN .. V)`` — PAD, the outcome tokens, BOS, NULL, and
    the reserved control block — are set to ``-inf`` *before* the softmax so
    the normaliser runs over the supervised move-token support only and the
    teacher's soft target never spreads mass onto those dead columns
    (plan §7 / Phase-B spec, Stage B1). This is a strictly larger mask than
    :func:`pawn.trainer.mask_reserved_columns` (which the integer-label CE
    can use, since PAD / outcome / BOS are never targets): for *soft-target*
    KL the teacher would otherwise distil real PAD / outcome / BOS mass into
    the student. Computed in fp32 for numerical stability irrespective of
    the forward compute dtype.
    """
    scaled = mask_distill_columns(logits.astype(jnp.float32)) / jnp.float32(
        temperature
    )
    return jax.nn.log_softmax(scaled, axis=-1)


def kl_divergence_loss(
    student_logits: Float[Array, "B T V"],
    teacher_logits: Float[Array, "B T V"],
    loss_mask: Bool[Array, "B T"],
    *,
    temperature: float = 2.0,
) -> Float[Array, ""]:
    """Temperature-scaled ``KL(softmax(t/T) ‖ softmax(s/T)) · T²``.

    Both student and teacher logits have their PAD / outcome / BOS / NULL /
    reserved columns (``[PAD_TOKEN .. V)``) masked to ``-inf`` before the
    softmax, so the divergence is measured purely over the supervised
    move-token support and the teacher's soft target carries zero mass on
    the dead columns (plan §7 / Phase-B spec, Stage B1). The
    ``T²`` factor restores the gradient scale that the ``1/T`` logit
    scaling would otherwise shrink (Hinton et al. 2015) so the distillation
    term stays comparable to the CE term inside ``mix``.

    The mean is over **supervised positions only** (``loss_mask`` True);
    PAD / prefix positions contribute nothing. A fully-masked batch divides
    by ``max(count, 1) = 1`` and returns 0 — the optimizer step is then a
    no-op (mirrors :func:`pawn.trainer.cross_entropy_loss`).
    """
    student_log_p = _masked_log_softmax(student_logits, temperature)
    teacher_log_p = _masked_log_softmax(teacher_logits, temperature)
    teacher_p = jnp.exp(teacher_log_p)
    # KL(teacher ‖ student) = Σ p_t · (log p_t − log p_s). The masked
    # columns contribute ``p_t = 0`` and (post-`-inf`) ``log p_t = -inf``;
    # ``0 · -inf`` is NaN, so guard with `where` to zero those terms before
    # the sum. The surviving columns carry the real divergence.
    per_col = teacher_p * (teacher_log_p - student_log_p)
    per_col = jnp.where(teacher_p > 0.0, per_col, 0.0)
    per_pos = per_col.sum(axis=-1)  # (B, T)
    mask = loss_mask.astype(jnp.bool_)
    per_pos = jnp.where(mask, per_pos, 0.0)
    n_real = jnp.maximum(mask.sum(), 1)
    kl = per_pos.sum() / n_real
    return kl * jnp.float32(temperature) ** 2


def distill_loss(
    student: "PAWNModel | EffectiveCallable",
    teacher_fn: TeacherFn,
    batch: Batch,
    *,
    objective: DistillObjective = "mix",
    temperature: float = 2.0,
    alpha: float = 0.5,
    compute_dtype: jnp.dtype | None = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
) -> Float[Array, ""]:
    """Pluggable distillation objective for one student + batch.

    ``objective``:

    - ``"kl"`` — :func:`kl_divergence_loss` against the teacher's soft
      targets only.
    - ``"ce"`` — ground-truth cross-entropy only (reuses
      :func:`pawn.trainer.cross_entropy_loss`); the teacher is not
      consulted.
    - ``"mix"`` — ``alpha · ce + (1 - alpha) · kl``. ``alpha=1`` reduces to
      ``ce``; ``alpha=0`` reduces to ``kl`` (guarded in
      ``tests/test_jax_distill.py``).

    The student forward runs once and its logits feed both terms when
    ``mix`` is selected — no redundant second forward. The teacher forward
    is skipped entirely for the ``ce`` objective.
    """
    if objective == "ce":
        return cross_entropy_loss(
            student, batch,
            compute_dtype=compute_dtype,
            use_sdpa=use_sdpa, use_flash=use_flash,
        )

    student_logits = student(
        batch.tokens, batch.attn_mask,
        compute_dtype=compute_dtype,
        use_sdpa=use_sdpa, use_flash=use_flash,
    )
    teacher_logits = teacher_fn(batch.tokens, batch.attn_mask)
    kl = kl_divergence_loss(
        student_logits, teacher_logits, batch.loss_mask,
        temperature=temperature,
    )
    if objective == "kl":
        return kl

    # mix: alpha·ce + (1-alpha)·kl. Reuse the already-computed student
    # logits for the CE term so the student forward isn't run twice.
    ce = _cross_entropy_from_logits(student_logits, batch)
    return jnp.float32(alpha) * ce + (jnp.float32(1.0) - jnp.float32(alpha)) * kl


def _cross_entropy_from_logits(
    logits: Float[Array, "B T V"], batch: Batch
) -> Float[Array, ""]:
    """Masked CE straight from precomputed logits.

    Mirrors the loss-side of :func:`pawn.trainer.cross_entropy_loss` (same
    reserved-column ``-inf`` mask, same ``loss_mask`` denominator) but takes
    logits rather than a model, so the ``mix`` objective can share the
    single student forward between its CE and KL terms.
    """
    logits_f32 = logits.astype(jnp.float32)
    masked_logits = mask_reserved_columns(logits_f32)
    per_pos_loss = optax.softmax_cross_entropy_with_integer_labels(
        masked_logits, batch.targets,
        where=batch.loss_mask.astype(jnp.bool_)[..., None],
    )
    n_real = jnp.maximum(batch.loss_mask.sum(), 1)
    return per_pos_loss.sum() / n_real


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


class DistillTrainState(eqx.Module):
    """State for distillation training.

    ``student`` is the trainable PyTree (a standalone
    :class:`pawn.model.PAWNModel` at specialized_clm shapes). ``teacher`` is
    the frozen source model — held in the state so it is a JIT-traced
    constant, but never updated (its gradient is severed by
    :func:`frozen_teacher`'s :func:`jax.lax.stop_gradient`). ``opt_state``
    tracks the optimizer moments + clip + lr schedule; ``step`` is the JAX
    scalar counter; ``key`` is the run RNG.
    """

    student: PAWNModel
    teacher: PAWNModel
    opt_state: optax.OptState
    step: Int[Array, ""]
    key: jax.Array


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def make_distill_train_step(
    optimizer: optax.GradientTransformation,
    *,
    objective: DistillObjective = "mix",
    temperature: float = 2.0,
    alpha: float = 0.5,
    compute_dtype: jnp.dtype | None = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
) -> Callable[
    [DistillTrainState, Batch], tuple[DistillTrainState, Float[Array, ""]]
]:
    """Build the JIT'd single distillation step.

    Mirrors :func:`pawn.adapter_trainer.make_adapter_train_step`: gradients
    flow only through the trainable ``student`` PyTree (the partition is
    achieved by differentiating ``loss_fn`` w.r.t. ``state.student`` alone
    via :func:`equinox.filter_value_and_grad`). The teacher is held in the
    state as a frozen JIT constant — its contribution to the loss is wrapped
    in :func:`jax.lax.stop_gradient` by :func:`frozen_teacher`, so XLA
    dead-code-eliminates every teacher weight-gradient.

    The optimizer update is unconditional (same rationale as the adapter /
    pretrain trainers — the ``lax.cond`` empty-batch guard cost more than
    the hypothetical all-PAD drift it prevented).
    """

    @eqx.filter_jit(donate="all")
    def step(
        state: DistillTrainState, batch: Batch
    ) -> tuple[DistillTrainState, Float[Array, ""]]:
        teacher_fn = frozen_teacher(
            state.teacher,
            compute_dtype=compute_dtype,
            use_sdpa=use_sdpa, use_flash=use_flash,
        )

        def loss_fn(student: PAWNModel) -> Float[Array, ""]:
            return distill_loss(
                student, teacher_fn, batch,
                objective=objective, temperature=temperature, alpha=alpha,
                compute_dtype=compute_dtype,
                use_sdpa=use_sdpa, use_flash=use_flash,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.student)

        params: Any = state.student
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, params
        )
        new_student = eqx.apply_updates(state.student, updates)

        new_state = DistillTrainState(
            student=new_student,
            teacher=state.teacher,
            opt_state=new_opt_state,
            step=state.step + jnp.int32(1),
            key=state.key,
        )
        return new_state, loss

    return step


def make_distill_scan_step(
    train_step: Callable[
        [DistillTrainState, Batch],
        tuple[DistillTrainState, Float[Array, ""]],
    ],
) -> Callable[
    [DistillTrainState, Batch], tuple[DistillTrainState, Float[Array, "K"]]
]:
    """Wrap a single distillation step in a K-step :func:`jax.lax.scan`.

    Input ``batches`` has a leading K axis on every :class:`Batch` field.
    Mirrors :func:`pawn.adapter_trainer.make_adapter_scan_step` so the
    per-step body never returns to the host across the chunk.
    """

    @eqx.filter_jit(donate="all")
    def scan_step(
        state: DistillTrainState, batches: Batch
    ) -> tuple[DistillTrainState, Float[Array, "K"]]:
        def body(
            carry: DistillTrainState, batch: Batch
        ) -> tuple[DistillTrainState, Float[Array, ""]]:
            new_carry, loss = train_step(carry, batch)
            return new_carry, loss

        final_state, losses = jax.lax.scan(body, state, batches)
        return final_state, losses

    return scan_step
