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
- :func:`make_optimizer` — ``optax.chain(clip_by_global_norm(
  cfg.max_grad_norm), adamw(lr_schedule, weight_decay=wd))`` with a
  `lax.cond` guard against padded-batch weight-decay drift (skip update
  when the loss mask is empty).
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
from typing import Any, Callable, Final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int

from pawn.config import ModelConfig, NULL_TOKEN, PAD_TOKEN
from pawn.corpus import Corpus
from pawn.model import EffectiveCallable, PAWNModel, sliced
from pawn.run_config import BaseRunConfig

__all__ = [
    "Batch",
    "TrainState",
    "VariantSpec",
    "cross_entropy_loss",
    "top1_accuracy",
    "supernet_joint_loss",
    "make_lr_schedule",
    "make_optimizer",
    "make_train_step",
    "make_scan_step",
    "slice_batch",
    "flatten_opt_state",
    "unflatten_opt_state",
    "get_grad_norm",
    "reserved_column_mask",
    "mask_reserved_columns",
    "distill_column_mask",
    "mask_distill_columns",
    "apply_legal_mask",
    "illegal_probability_mass",
]


# First logit column that must be masked to -inf before the softmax-CE.
# Columns ``[NULL_TOKEN .. V)`` are NULL + the reserved control IDs
# (1981–1999): they exist in the uniform ``V``-wide embedding / logit
# tables (Phase-A Chunk 2) but are never legitimate prediction targets,
# so masking them to -inf keeps them out of the softmax denominator
# (no dilution of move-token mass) and zeros their backward gradient.
# BOS (1980) is excluded — it sits one below ``NULL_TOKEN`` and, like
# PAD / outcome columns, is left in the softmax (it is never a target so
# it accrues only the benign softmax-denominator gradient). The
# threshold matches the plan's "reserved/NULL/control columns (≥1981)"
# acceptance gate; Phase B's distillation-KL will reuse the same mask.
_FIRST_RESERVED_COLUMN: Final[int] = NULL_TOKEN


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
        legal_mask: ``(B, T, V)`` bool — True at the legal move tokens
            for the position the target predicts. ``None`` (the pretrain
            default) means "no per-position legality information": the
            CE softmax then only masks the reserved / NULL / control
            columns (:func:`mask_reserved_columns`). The adapter loop
            (:mod:`pawn.adapter_trainer`) attaches this so the move-CE
            softmax normalises over legal moves only (v1 ``apply_legal_mask``
            default-ON parity) and the ``illegal_penalty`` term has a legality
            reference to penalise against.
    """

    tokens: Int[Array, "B T"]
    targets: Int[Array, "B T"]
    attn_mask: Bool[Array, "B T"]
    loss_mask: Bool[Array, "B T"]
    legal_mask: Bool[Array, "B T V"] | None = None


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
# Reserved / NULL / control column masking (Phase-A §2 column-mask helper)
# ---------------------------------------------------------------------------


def reserved_column_mask(v: int) -> Bool[Array, "v"]:
    """Boolean ``(V,)`` vector — True at columns that must be masked out.

    Columns ``[NULL_TOKEN .. V)`` are NULL + the reserved control IDs:
    they live in the uniform ``V``-wide logit table (Phase-A Chunk 2) but
    are never legitimate prediction targets. The :class:`cross_entropy_loss`
    softmax and the Phase-B distillation KL both feed this mask through
    :func:`mask_reserved_columns` so the dead columns neither dilute the
    move-token probability mass nor accrue backward gradient. Factored out
    of the inline ``cross_entropy_loss`` computation so the distillation
    path reuses the *same* threshold rather than re-deriving it.
    """
    return jnp.arange(v, dtype=jnp.int32) >= jnp.int32(_FIRST_RESERVED_COLUMN)


def mask_reserved_columns(
    logits: Float[Array, "... V"],
) -> Float[Array, "... V"]:
    """Set the reserved / NULL / control columns of ``logits`` to ``-inf``.

    ``-inf`` is representable, so ``exp(-inf) = 0`` keeps the softmax
    denominator finite as long as at least one move / PAD / outcome / BOS
    column survives (columns ``[0, NULL_TOKEN)`` are always untouched). The
    autograd graph through the constant ``-inf`` substitution is detached at
    those columns, so the corresponding embedding / ``lm_head`` rows receive
    exactly zero gradient.
    """
    v = logits.shape[-1]
    reserved = reserved_column_mask(v)
    return jnp.where(reserved, -jnp.inf, logits)


def distill_column_mask(v: int) -> Bool[Array, "v"]:
    """Boolean ``(V,)`` vector — True at columns excluded from the KL.

    Distillation masks a **strictly larger** set of columns than the
    integer-label CE softmax: columns ``[PAD_TOKEN .. V)`` — PAD, the 11
    outcome tokens, BOS, NULL, and the reserved control block — are all
    excluded from both the student and teacher softmax (plan §7 / Phase-B
    spec, Stage B1). The CE softmax can leave PAD / outcome / BOS columns
    in (their only contribution is the benign softmax-denominator gradient,
    since they are never integer targets), but for *soft-target* KL the
    teacher's softmax would otherwise place real probability mass on those
    columns and distil it into the student. Masking at ``PAD_TOKEN`` keeps
    the KL distribution on the supervised move-token support only.
    """
    return jnp.arange(v, dtype=jnp.int32) >= jnp.int32(PAD_TOKEN)


def mask_distill_columns(
    logits: Float[Array, "... V"],
) -> Float[Array, "... V"]:
    """Set the PAD / outcome / BOS / NULL / reserved columns to ``-inf``.

    The distillation counterpart of :func:`mask_reserved_columns`: it masks
    columns ``[PAD_TOKEN .. V)`` (a superset of the CE reserved block) so the
    KL softmax normalises over the supervised move-token support alone and
    the teacher's soft target carries zero mass on PAD / outcome / BOS / NULL
    columns (Phase-B spec, Stage B1). ``exp(-inf) = 0`` keeps the denominator
    finite as long as at least one move column survives, and the constant
    ``-inf`` substitution detaches the backward gradient on those columns.
    """
    v = logits.shape[-1]
    masked = distill_column_mask(v)
    return jnp.where(masked, -jnp.inf, logits)


# ---------------------------------------------------------------------------
# Legal-move masking (adapter-loss legality, shared with RoSA mask-gen)
# ---------------------------------------------------------------------------


def apply_legal_mask(
    logits: Float[Array, "... V"],
    legal_mask: Bool[Array, "... V"],
) -> Float[Array, "... V"]:
    """Set the *illegal* move columns of ``logits`` to ``-inf``.

    ``legal_mask`` is True at the tokens that are legal at each position;
    every False column is forced to ``-inf`` so the softmax never assigns
    it probability mass and its embedding / ``lm_head`` row receives zero
    gradient (the constant ``-inf`` substitution detaches the backward
    graph at those columns). The v2 parity of v1's
    ``valid_logits.masked_fill(~valid_legal, -inf)`` (``apply_legal_mask``
    default-ON in the adapter loss; ``git show
    main:pawn/adapter_training.py:378-380``).

    Reused by the RoSA mask-generation path (:func:`pawn.adapter_trainer.\
generate_rosa_masks`) so the gradient signal that selects sparse positions
    is taken under the same legality regime as training.
    """
    return jnp.where(legal_mask, logits, -jnp.inf)


def illegal_probability_mass(
    logits: Float[Array, "... V"],
    legal_mask: Bool[Array, "... V"],
    *,
    where: Bool[Array, "..."] | None = None,
) -> Float[Array, ""]:
    """Mean softmax mass landing on illegal moves, over supervised positions.

    The v2 parity of v1's ``illegal_probability_mass``
    (``git show main:pawn/adapter_training.py:513-530``): softmax the
    *unmasked* logits (over the move-token support — reserved columns are
    still pushed to ``-inf`` so they never carry mass), zero the legal
    columns, and sum the remaining (illegal) mass per position. ``where``
    restricts the mean to supervised (non-PAD) positions; when ``None`` the
    mean is over every position.

    Only meaningful when the hard legal mask is *off* (``apply_legal_mask``
    not applied to ``logits``): under hard masking the illegal mass is
    analytically zero, so the caller short-circuits this term.
    """
    logits_f32 = mask_reserved_columns(logits.astype(jnp.float32))
    probs = jax.nn.softmax(logits_f32, axis=-1)
    illegal = jnp.where(legal_mask, 0.0, probs).sum(axis=-1)  # (...,)
    if where is None:
        return illegal.mean()
    w = where.astype(jnp.float32)
    denom = jnp.maximum(w.sum(), 1.0)
    return (illegal * w).sum() / denom


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    model: "PAWNModel | EffectiveCallable",
    batch: Batch,
    *,
    compute_dtype: jnp.dtype | None = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
    apply_legal: bool = True,
    illegal_penalty: float = 0.0,
) -> Float[Array, ""]:
    """Masked cross-entropy on a single variant + batch.

    Returns the mean per-supervised-position loss. PAD positions
    (``loss_mask`` False) don't contribute. The output is a 0-d scalar
    JAX array.

    The denominator is ``loss_mask.sum().clip(min=1)`` — a fully-padded
    batch returns 0 / 1 = 0 (the optimizer should be a no-op then,
    which is what the `lax.cond` guard in :func:`make_optimizer` is
    for).

    ``compute_dtype`` is the AMP forward dtype (plan §5). ``None`` (the
    default) runs the model in fp32 — the fp32 parity tests and any
    consumer that needs bit-stable logits pass ``None``.

    ``use_sdpa`` opts the attention block into
    :func:`jax.nn.dot_product_attention` (parity #43). Bare
    :class:`PAWNModel` accepts the kwarg; wrappers that satisfy
    :class:`EffectiveCallable` (e.g. ``BottleneckEffective``) do not —
    the trainer falls back to the plain path automatically. See
    :data:`pawn.run_config.BaseRunConfig.use_sdpa` for the operator
    surface.

    **Legality (adapter-loss parity).** When ``batch.legal_mask`` is set
    *and* ``apply_legal`` is True (v1 ``apply_legal_mask`` default-ON),
    the illegal move columns are forced to ``-inf`` before the softmax so
    the move-CE normalises over legal moves only. When ``apply_legal`` is
    False the raw move logits survive and, if ``illegal_penalty > 0``, the
    loss gains ``illegal_penalty · E[P_illegal]`` — the mean softmax mass
    on illegal moves over supervised positions (v1
    ``compute_adapter_loss``; ``git show
    main:pawn/adapter_training.py:382-387,548-549``). Both terms are no-ops
    when ``batch.legal_mask is None`` (the pretrain path), so pretrain CE
    is bit-identical to before.
    """
    # ``EffectiveCallable`` now mandates the ``use_sdpa`` / ``use_flash``
    # kwargs (Protocol updated alongside the bottleneck wrapper's
    # ``__call__`` signature) — both :class:`PAWNModel` and
    # :class:`BottleneckEffective` accept the same surface, so the
    # dispatch is uniform and bottleneck-style adapters get the
    # Pallas-flash win too.
    with jax.named_scope("forward"):
        logits = model(
            batch.tokens, batch.attn_mask,
            compute_dtype=compute_dtype, use_sdpa=use_sdpa, use_flash=use_flash,
        )
    # Fused logsumexp + integer-label gather. Replaces the old
    # ``log_softmax(logits) → take_along_axis(...)`` pair which
    # materialised the full ``(B, T, V)`` log-probs tensor in fp32
    # (~520 MB/step at BASE shape). The fused kernel runs in fp32 for
    # numerical stability — the explicit ``.astype(jnp.float32)`` here
    # is the one-time upcast that used to live inside
    # :meth:`PAWNModel.__call__`.
    #
    # ``where=loss_mask[..., None]`` is the **backward-only** PAD
    # optimisation, not a forward bandwidth saving. Reading optax
    # source: the ``where=`` path inside
    # ``optax.softmax_cross_entropy_with_integer_labels`` calls
    # ``jax.nn.logsumexp(logits, axis=-1, where=where)``, whose
    # implementation does ``a = jnp.where(where, a, 0)`` and then
    # reduces — i.e. it materialises *another* full ``(B, T, V)``
    # fp32 copy. The actual saving comes from optax's outer
    # ``jnp.where(jnp.any(where, axis), out, 0.0)`` (loss is forced
    # to 0 at PAD positions): the autograd graph through that
    # ``jnp.where`` zeros the upstream gradient at PAD positions,
    # so the lm_head, final-norm, and pre-norm backward never
    # accumulate PAD-position contributions. Measured supervision
    # density on bench data is 71% (mean game length ~364 plies of
    # 512), so we skip ~29% of the loss-side backward compute. Round-3
    # review (Opus conv) caught the previous comment overstating the
    # forward saving here.
    with jax.named_scope("cross_entropy"):
        # Mask the reserved / NULL / control columns (IDs ≥ NULL_TOKEN)
        # to -inf before the softmax so they neither dilute the move-token
        # probability mass nor accrue gradient. The autograd graph through
        # the constant -inf substitution is detached at those columns, so
        # the corresponding rows of ``embed_tokens`` (and, when untied,
        # ``lm_head``) receive exactly zero gradient. Done in fp32 (the CE
        # runs in fp32 for numerical stability); -inf is representable so
        # ``exp(-inf) = 0`` keeps the denominator finite as long as at
        # least one move/PAD/outcome/BOS column survives — which it always
        # does (columns ``[0, NULL_TOKEN)`` are untouched).
        logits_f32 = logits.astype(jnp.float32)
        masked_logits = mask_reserved_columns(logits_f32)
        # Adapter-loss legality (v1 parity). With a per-position
        # ``legal_mask`` and hard masking on, push illegal move columns to
        # -inf so the softmax denominator (and the gradient) only sees legal
        # moves. ``None`` legal_mask leaves pretrain CE untouched.
        if batch.legal_mask is not None and apply_legal:
            masked_logits = apply_legal_mask(masked_logits, batch.legal_mask)
        loss_mask_bool = batch.loss_mask.astype(jnp.bool_)
        per_pos_loss = optax.softmax_cross_entropy_with_integer_labels(
            masked_logits, batch.targets,
            where=loss_mask_bool[..., None],
        )  # (B, T), zero at PAD positions thanks to ``where=``
        n_real = jnp.maximum(batch.loss_mask.sum(), 1)
        loss = per_pos_loss.sum() / n_real
        # Illegal-mass penalty (v1 ``compute_adapter_loss``). Only
        # meaningful with the hard mask off: under hard masking the
        # illegal mass is analytically zero. We add it on the *un*-legal-
        # masked logits so the term has a non-trivial gradient.
        if (
            batch.legal_mask is not None
            and not apply_legal
            and illegal_penalty > 0.0
        ):
            penalty = illegal_probability_mass(
                logits_f32, batch.legal_mask, where=loss_mask_bool,
            )
            loss = loss + illegal_penalty * penalty
        return loss


def top1_accuracy(
    model: "PAWNModel | EffectiveCallable",
    batch: Batch,
    *,
    compute_dtype: jnp.dtype | None = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
) -> Float[Array, ""]:
    """Mean next-token top-1 accuracy on a single variant + batch.

    The v2 parity of v1's pretrain ``train/accuracy`` metric
    (``git show main:pawn/trainer.py:1030`` —
    ``(valid_logits.argmax(-1) == valid_targets).float().mean()``):
    the fraction of supervised positions whose argmax prediction matches
    the target token. PAD positions (``loss_mask`` False) are excluded
    from both numerator and denominator.

    The argmax is taken over the move-token support only — reserved /
    NULL / control columns are masked to ``-inf`` exactly as the loss
    softmax does (:func:`mask_reserved_columns`), so a reserved column
    can never win the argmax and inflate the "wrong" count. This mirrors
    the eval-time restriction to ``[0, NUM_ACTIONS)`` documented in the
    project eval contract.

    Returns a 0-d fp32 JAX array. A fully-padded batch returns 0 / 1 = 0.
    """
    with jax.named_scope("forward_accuracy"):
        logits = model(
            batch.tokens, batch.attn_mask,
            compute_dtype=compute_dtype, use_sdpa=use_sdpa, use_flash=use_flash,
        )
    masked_logits = mask_reserved_columns(logits.astype(jnp.float32))
    preds = jnp.argmax(masked_logits, axis=-1)  # (B, T)
    mask = batch.loss_mask.astype(jnp.bool_)
    correct = jnp.where(mask, preds == batch.targets, False)
    n_real = jnp.maximum(mask.sum(), 1)
    return correct.sum().astype(jnp.float32) / n_real.astype(jnp.float32)


def supernet_joint_loss(
    model: PAWNModel,
    batch: Batch,
    variants: tuple[VariantSpec, ...],
    *,
    compute_dtype: jnp.dtype | None = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
    stochastic_key: jax.Array | None = None,
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

    ``stochastic_key`` switches into a sandwich-sampling regime: when
    given, the supernet (``is_supernet=True``) variant is run every
    step as normal, and exactly *one* of the remaining non-supernet
    variants is sampled uniformly each step. The sampled variant's
    CE is scaled by ``N`` (the number of non-supernet variants) so
    that ``E[loss_stochastic] == sum-of-all loss``: the gradient
    update is unbiased in expectation. Saves ~50-67% of the per-step
    forward+backward FLOPs when the variant count grows (the small
    and base variants of the production supernet each contribute a
    full forward+backward pass through their respective dims). The
    correctness story is the MatFormer / matryoshka-supernet
    literature: stochastic width sampling has been shown to converge
    to the same quality as full joint training, with a small variance
    increase in the small-variant gradient signal.

    Pass ``stochastic_key=None`` (default) to keep the exhaustive
    sum-of-all path for tests, parity checks, and any caller that
    needs a deterministic loss surface.
    """
    if not variants:
        raise ValueError(
            "supernet_joint_loss requires at least one VariantSpec; "
            "got an empty tuple"
        )
    if stochastic_key is None:
        total = jnp.array(0.0, dtype=jnp.float32)
        for spec in variants:
            if spec.is_supernet:
                sub_model = model
            else:
                sub_model = sliced(model, spec.cfg)
            total = total + cross_entropy_loss(
                sub_model, batch,
                compute_dtype=compute_dtype,
                use_sdpa=use_sdpa, use_flash=use_flash,
            )
        return total

    # Stochastic sandwich: supernet always + one sampled non-supernet
    # variant, scaled by N to keep the expectation equal to the full
    # sum.
    supernet_variants = tuple(v for v in variants if v.is_supernet)
    other_variants = tuple(v for v in variants if not v.is_supernet)
    total = jnp.array(0.0, dtype=jnp.float32)
    for spec in supernet_variants:
        total = total + cross_entropy_loss(
            model, batch,
            compute_dtype=compute_dtype,
            use_sdpa=use_sdpa, use_flash=use_flash,
        )

    if other_variants:
        n_other = len(other_variants)

        def _make_branch(spec: VariantSpec):
            def _branch(_: Any) -> Float[Array, ""]:
                sub = sliced(model, spec.cfg)
                return cross_entropy_loss(
                    sub, batch,
                    compute_dtype=compute_dtype,
                    use_sdpa=use_sdpa, use_flash=use_flash,
                ) * jnp.float32(n_other)
            return _branch

        idx = jax.random.randint(
            stochastic_key, (), 0, n_other, dtype=jnp.int32
        )
        sampled = jax.lax.switch(
            idx, [_make_branch(s) for s in other_variants], operand=None
        )
        total = total + sampled

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


def _cosine_ramp_schedule(
    init_value: float, peak_value: float, ramp_steps: int
) -> optax.Schedule:
    """Cosine ramp-up from ``init_value`` to ``peak_value`` over
    ``ramp_steps`` (v1 ``OneCycle`` warmup shape).

    v1's one-cycle ramp was
    ``init + (peak - init) * 0.5 * (1 - cos(pi * progress))`` where
    ``progress = step / ramp_steps``: a cosine-eased acceleration into the
    peak rather than a straight line. Optax ships no cosine **ramp-up**
    builder (``cosine_decay_schedule`` only decays), so this reproduces the
    v1 curve directly. ``ramp_steps <= 0`` collapses to a constant at the
    peak (the warmup is zero-width — :func:`optax.join_schedules` clamps
    the boundary to 0 in that case).
    """
    if ramp_steps <= 0:
        return optax.constant_schedule(peak_value)

    def schedule(count: ArrayLike) -> Array:
        progress = jnp.clip(jnp.asarray(count) / ramp_steps, 0.0, 1.0)
        cos_eased = 0.5 * (1.0 - jnp.cos(jnp.pi * progress))
        return init_value + (peak_value - init_value) * cos_eased

    return schedule


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
        # `_check_lr_schedule_fractions` validates `warmup_frac +
        # decay_frac ≤ 1.0` in fractions, but `warmup` may have been
        # resolved from an explicit `warmup_steps` override that
        # ignores `warmup_frac`. When the override drives the actual
        # sum over total_steps, refuse — Optax `join_schedules` would
        # see a non-monotonic boundary (PR #115 review #2).
        #
        # Pure float-fraction rounding overflow (no explicit
        # `warmup_steps` override) is tolerated by the
        # `total_steps - decay_steps` boundary, which is at least
        # `warmup` here when fractions are within 1.0.
        if (
            cfg.warmup_steps is not None
            and warmup + decay_steps > total_steps
        ):
            raise ValueError(
                f"wsd schedule: warmup ({warmup}) + decay_steps "
                f"({decay_steps}) exceeds total_steps ({total_steps}). "
                f"`decay_frac={cfg.decay_frac}` was validated against "
                f"warmup_frac={cfg.warmup_frac} but warmup_steps "
                f"({cfg.warmup_steps}) overrode the fraction. Reduce "
                f"warmup_steps or decay_frac."
            )
        decay = (
            optax.linear_schedule(peak, 0.0, decay_steps)
            if cfg.wsd_decay_shape == "linear"
            else optax.cosine_decay_schedule(peak, decay_steps, 0.0)
        )
        # Pure float-fraction rounding can drive `total_steps -
        # decay_steps < warmup` (e.g. `warmup_frac=0.5, decay_frac=0.5,
        # total_steps=3` rounds to warmup=2, decay=2, decay-start=1).
        # Clamp to the warmup boundary so `join_schedules` boundaries
        # stay monotonic; the stable phase has zero width then, which is
        # the right behavior — the schedule transitions straight from
        # warmup into decay. Mirrors the `max(0, ...)` clamp the
        # `infinite` branch has at line 379. (Round-1 review-test-risk
        # + review-bug-detector flagged the missing clamp.)
        decay_start = max(warmup, total_steps - decay_steps)
        return optax.join_schedules(
            [
                optax.linear_schedule(0.0, peak, warmup),
                optax.constant_schedule(peak),
                decay,
            ],
            [warmup, decay_start],
        )
    if cfg.lr_schedule == "one_cycle":
        init = peak / 25.0
        end = peak / 10000.0
        # Ramp warmup steps init → peak with a **cosine** shape (Smith
        # 2018, v1 parity — `git show main:pawn/trainer.py` ``OneCycle``).
        # v1's ramp was `init + (peak - init) * 0.5 * (1 - cos(pi *
        # progress))`, NOT a straight line. An earlier v2 used
        # ``optax.linear_schedule(init, peak, warmup)`` here, changing the
        # canonical one-cycle ramp shape (linear vs cosine) — restored to
        # the cosine ramp so the schedule matches v1 step-for-step. Then
        # cosine-decay from peak to ``end`` over the remainder.
        remaining = total_steps - warmup
        return optax.join_schedules(
            [
                _cosine_ramp_schedule(init, peak, warmup),
                optax.cosine_decay_schedule(peak, remaining, end / peak),
            ],
            [warmup],
        )
    if cfg.lr_schedule == "infinite":
        cooldown_steps = int(round(cfg.cooldown_frac * total_steps))
        decay_steps = int(round(cfg.decay_frac * total_steps))
        stable_lr = peak * cfg.stable_lr_ratio
        # `_check_lr_schedule_fractions` validates the float fractions;
        # the same `warmup_steps` override path that breaks WSD breaks
        # infinite too. Refuse `warmup + cooldown + decay > total` only
        # when warmup was explicitly overridden — pure-rounding
        # overflow is clamped to a zero-width stable plateau below
        # (PR #115 review #2).
        if (
            cfg.warmup_steps is not None
            and warmup + cooldown_steps + decay_steps > total_steps
        ):
            raise ValueError(
                f"infinite schedule: warmup ({warmup}) + cooldown "
                f"({cooldown_steps}) + decay ({decay_steps}) exceeds "
                f"total_steps ({total_steps}). cooldown_frac="
                f"{cfg.cooldown_frac}, decay_frac={cfg.decay_frac} "
                f"were validated against warmup_frac={cfg.warmup_frac}, "
                f"but warmup_steps ({cfg.warmup_steps}) overrode the "
                f"fraction. Reduce warmup_steps, cooldown_frac, or "
                f"decay_frac so the stable plateau has non-negative "
                f"width."
            )
        # Integer rounding of three fractions can still sum to slightly
        # more than total_steps even when the floats sum to ≤1; clamp
        # to keep `join_schedules` boundaries monotonic.
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


class _ClipState(eqx.Module):
    """State for :func:`_branchless_clip_by_global_norm` — exposes the
    most-recent unscaled grad norm so the trainer can log clip
    triggers (C.5).
    """
    g_norm: Float[Array, ""]


def _branchless_clip_by_global_norm(max_norm: float) -> optax.GradientTransformation:
    """Branchless replacement for :func:`optax.clip_by_global_norm`.

    Optax's stock implementation
    (``optax/transforms/_clipping.py:94``) is:

    .. code-block:: python

        trigger = g_norm < max_norm
        clip_fn = lambda t: jax.lax.select(trigger, t, (t / g_norm) * max_norm)
        updates = jax.tree.map(clip_fn, updates)

    ``jax.lax.select`` always evaluates both branches, so the divide-
    and-scale path runs for every parameter leaf on every step, even
    when the gradient norm is already below the clip threshold. The
    optax source has a TODO acknowledging this. The branchless form
    is mathematically equivalent:

        g_norm = max(g_norm, max_norm)
        updates = updates / g_norm * max_norm

    When ``g_norm <= max_norm`` the ``max`` selects ``max_norm`` and
    the rescale is a no-op (multiply-then-divide by the same scalar).
    When ``g_norm > max_norm`` the ``max`` selects ``g_norm`` and the
    rescale is the usual clip. One pass over ``updates`` instead of
    two, no ``select`` materialisation. ~0.2-0.3 ms saved per step at
    LARGE shape. Round-3 review surfaced this (Opus conv).

    The state now carries the most-recent unscaled `g_norm` so C.5's
    clip-trigger measurement can log it without altering scan_step's
    return signature. The state is a `eqx.Module` with one scalar
    field — donate-friendly and zero compile overhead.
    """

    def init_fn(params: Any) -> Any:
        del params
        return _ClipState(g_norm=jnp.float32(0.0))

    def update_fn(updates: Any, state: Any, params: Any = None) -> Any:
        del params, state
        g_norm = optax.tree.norm(updates)
        scale = max_norm / jnp.maximum(g_norm, max_norm)
        clipped = jax.tree_util.tree_map(lambda t: t * scale.astype(t.dtype), updates)
        return clipped, _ClipState(g_norm=g_norm.astype(jnp.float32))

    return optax.GradientTransformation(init_fn, update_fn)


def get_grad_norm(opt_state) -> Float[Array, ""]:
    """Read the most-recent unscaled grad norm out of the optimizer
    state. Returns 0 if the state doesn't contain a `_ClipState`
    (e.g., the optimizer was built without our clip transformation)."""
    # `opt_state` is a tuple from `optax.chain`. Walk the leaves until
    # we find a `_ClipState`. If none, return 0.
    for leaf in jax.tree_util.tree_leaves(opt_state, is_leaf=lambda x: isinstance(x, _ClipState)):
        if isinstance(leaf, _ClipState):
            return leaf.g_norm
    return jnp.float32(0.0)


def make_optimizer(
    cfg: BaseRunConfig,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    """Build the v2 optimizer: branchless gradient clip + (AdamW | Lion | Adafactor).

    Uses the local :func:`_branchless_clip_by_global_norm` instead of
    ``optax.clip_by_global_norm`` to avoid the latter's
    ``lax.select``-materialises-both-branches overhead. Weight-decay
    is applied through Optax's AdamW (decoupled, scaled by lr). The
    padded-batch weight-decay drift guard isn't here at the optimizer
    level — it lives in :func:`make_train_step` where we can see
    whether the batch was empty.

    C.1: ``cfg.optimizer`` selects between:
        ``adamw`` (default) — bf16 first moment, fp32 second moment.
        ``lion`` — sign-based; halves optimizer state.

    H10: the global-norm clip threshold is ``cfg.max_grad_norm`` (default
    1.0), threaded through to :func:`_branchless_clip_by_global_norm`. The
    old code hardcoded the threshold at a module-level ``1.0`` constant
    regardless of the config, so a run with ``max_grad_norm=0.5`` still
    clipped at 1.0
    while the ``did_clip`` metric (``train_jax.py``) compared the pre-clip
    norm against the *config* value — the two disagreed for any non-default
    threshold. Reading the config here keeps the actual clip and the
    ``did_clip`` measurement consistent.
    """
    name = getattr(cfg, "optimizer", "adamw")
    if name == "adamw":
        inner = optax.adamw(
            learning_rate=lr_schedule,
            # b2=0.95 (v1 value) NOT optax's 0.999 default — see
            # ``BaseRunConfig.adam_b2``: 0.999's ~700-step second-moment memory
            # fails to damp a gradient spike on the next step, which is the v2
            # collapse-to-uniform root cause. Passed explicitly so it can never
            # silently regress to the optax default again.
            b1=cfg.adam_b1,
            b2=cfg.adam_b2,
            weight_decay=cfg.weight_decay,
            # First moment (``mu``) in fp32 (optax defaults bf16). fp32 keeps
            # the small surviving gradient components from rounding to zero on
            # the step after a spike.
            mu_dtype=jnp.float32,
        )
    elif name == "lion":
        inner = optax.lion(
            learning_rate=lr_schedule,
            weight_decay=cfg.weight_decay,
            # Lion keeps its own b1/b2 defaults (different semantics from Adam's;
            # the b2=0.95 fix is adamw-specific). Single moment in fp32.
            mu_dtype=jnp.float32,
        )
    else:
        raise ValueError(f"unknown optimizer {name!r}; expected adamw/lion")
    chain = optax.chain(
        _branchless_clip_by_global_norm(cfg.max_grad_norm), inner
    )
    # Always wrap so the non-finite backstop (true GradScaler parity) is on
    # even at an infinite threshold. ``grad_skip_threshold = inf`` ⇒
    # non-finite-only skip; a finite threshold ALSO rejects finite spikes
    # above it — but note (see BaseRunConfig) a finite threshold near the
    # spike scale deadlocks/degrades, so the source of the spikes should be
    # removed (fp32 RoPE, fp32 softmax via use_flash=False) rather than
    # relying on a finite skip.
    threshold = getattr(cfg, "grad_skip_threshold", float("inf"))
    return _skip_on_grad_spike(chain, threshold)


def _skip_on_grad_spike(
    inner: optax.GradientTransformation, threshold: float
) -> optax.GradientTransformation:
    """Skip the optimizer step when the raw global grad norm is non-finite
    or exceeds ``threshold`` — a JAX/optax equivalent of v1's
    ``torch.amp.GradScaler`` step-skip on gradient overflow.

    On a skipped step the param updates are zeroed **and** the inner
    optimizer state is reverted, so a pathological batch's gradient spike is
    never applied and Adam's moments never absorb it (a *true* skip, not a
    damped one). v1 was stable in part because GradScaler rejected such
    steps; the JAX rewrite dropped that guard, so a bf16 gradient spike got
    applied and — combined with the (separately-fixed) b2=0.999 regression —
    drove the run into a uniform-output collapse.

    Branchless (``jnp.where`` select over both branches' outputs), matching
    the trainer's no-``lax.cond`` convention — XLA computes the inner update
    regardless, so a select is the cheap way to discard it.
    """

    def init_fn(params: optax.Params) -> optax.OptState:
        return inner.init(params)

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.OptState]:
        sq = sum(
            jnp.sum(jnp.square(leaf))
            for leaf in jax.tree_util.tree_leaves(updates)
        )
        g_norm = jnp.sqrt(sq)
        bad = jnp.logical_or(
            jnp.logical_not(jnp.isfinite(g_norm)), g_norm > threshold
        )
        new_updates, new_state = inner.update(updates, state, params)
        safe_updates = jax.tree_util.tree_map(
            lambda u: jnp.where(bad, jnp.zeros_like(u), u), new_updates
        )
        safe_state = jax.tree_util.tree_map(
            lambda n, o: jnp.where(bad, o, n), new_state, state
        )
        return safe_updates, safe_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Opt-state serialisation — preserve Adam moments + clip-state across resume
# ---------------------------------------------------------------------------


def flatten_opt_state(opt_state: optax.OptState) -> dict[str, np.ndarray]:
    """Flatten an Optax state PyTree into a ``{path: ndarray}`` dict
    that safetensors can persist.

    `jax.tree_util.tree_flatten_with_path` yields stable path keys
    (one per array leaf); we render those as a string key so the
    safetensors file is content-addressable. Non-array leaves (e.g.
    Optax's scalar counters that JAX serialises as 0-d arrays) are
    stored too. The companion :func:`unflatten_opt_state` rebuilds
    the PyTree using a freshly-initialised opt_state as the template.

    Issues ONE batched device→host transfer via :func:`jax.device_get`
    instead of one `np.asarray` round-trip per leaf — at SUPERNET
    scale the opt_state carries ~3× the model parameter count (Adam
    moments + clip), so the per-leaf path stalled the trainer at
    checkpoint boundaries waiting on N separate D→H copies. The
    batched form lets the XLA backend coalesce the transfer.
    (Round-1 review-performance-analyzer Critical.)
    """
    host_state = jax.device_get(opt_state)
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(host_state)
    out: dict[str, np.ndarray] = {}
    for path, leaf in leaves_with_paths:
        key = jax.tree_util.keystr(path)
        # After `jax.device_get`, leaves are already numpy / python
        # scalars; `np.asarray` is a no-op wrap for the array case
        # and a trivial 0-d allocation for the scalar case.
        out[key] = np.asarray(leaf)
    return out


def unflatten_opt_state(
    template: optax.OptState, flat: dict[str, np.ndarray]
) -> optax.OptState:
    """Rebuild an Optax state PyTree from the flat dict produced by
    :func:`flatten_opt_state`.

    `template` is a fresh `optimizer.init(...)` against the loaded
    model — it provides the PyTree structure (treedef + path map) but
    has empty / zero leaves. We overwrite each leaf with the loaded
    array, matching by `jax.tree_util.keystr(path)` so the mapping
    stays stable across Optax versions that may reorder leaves.

    Raises :class:`ValueError` if the loaded dict and the template
    have different leaf sets — that's a sign the saved checkpoint
    used a different optimiser configuration than the current one,
    which is unrecoverable.
    """
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    template_keys = {jax.tree_util.keystr(p) for p, _ in leaves_with_paths}
    if template_keys != set(flat):
        missing = sorted(template_keys - set(flat))
        extra = sorted(set(flat) - template_keys)
        raise ValueError(
            "opt_state checkpoint shape mismatch — likely the saved "
            "run used a different optimiser configuration than the "
            f"current init. Missing: {missing[:5]}{'…' if len(missing) > 5 else ''} "
            f"Extra: {extra[:5]}{'…' if len(extra) > 5 else ''}"
        )
    # Bind each restored leaf's dtype to the template leaf's dtype.
    # Without `dtype=`, `jnp.asarray(numpy_array)` infers from the
    # numpy dtype — fine for arrays whose dtype was preserved by the
    # safetensors round-trip, but a silent narrowing risk if Optax
    # ever stores a Python int leaf (which `np.asarray` widens to
    # int64 on Linux/x86_64) where the template expected int32.
    #
    # `getattr(leaf, "dtype", None)` covered the array case but
    # *re-introduced* the footgun for python-int template leaves
    # (round-2 bug-detector finding). Round-3 test-risk caught that
    # the fallback to `saved.dtype` is still risky: `np.asarray(42)`
    # on Linux defaults to int64, which JAX silently truncates to
    # int32 with a `UserWarning` (the JAX default int dtype).
    #
    # Resolution: when the template is a python scalar, normalise to
    # `np.int32` (for ints / bools) or `np.float32` (for floats),
    # which match JAX's defaults. Array-typed leaves still use the
    # template's dtype directly — that's the contract for the
    # production case where every Optax leaf is a JAX array.
    # Numpy scalars (`np.int64(5)` etc) and python scalars both
    # behave the same way semantically — a 0-d value carrying a
    # default-int-width dtype. Treat both as scalar inputs and
    # normalise to the JAX canonical width (int32 / float32 / bool).
    # Numpy *arrays* are NOT scalars — their dtype is the user's
    # intent and we preserve it verbatim. Round-4 bug-detector
    # Important caught that a `numpy.int64` scalar template leaf
    # (e.g. if a future Optax release stores `np.int64(step)` as a
    # counter) passed through the `hasattr("dtype")` branch and
    # returned int64, which JAX then truncated with a UserWarning.
    def _dtype_for(template_leaf: Any, saved: np.ndarray) -> Any:
        if isinstance(template_leaf, (bool, np.bool_)):
            return np.bool_
        # Numpy scalars are `np.generic` but not `np.ndarray`. They
        # mimic python scalars in semantics; route them through the
        # normalisation path rather than trusting their non-canonical
        # default dtype.
        if isinstance(template_leaf, np.ndarray) and template_leaf.ndim > 0:
            # Multi-dimensional numpy array — preserve the user's
            # dtype intent.
            return template_leaf.dtype
        if hasattr(template_leaf, "dtype") and not isinstance(
            template_leaf, np.generic
        ):
            # JAX array (or other array-like with a canonical dtype).
            return template_leaf.dtype
        # Python scalar OR numpy scalar — normalise to JAX defaults.
        if isinstance(template_leaf, int):
            return np.int32
        if isinstance(template_leaf, float):
            return np.float32
        # Unknown scalar type — fall back to saved.dtype as a
        # best-effort.
        return saved.dtype

    new_leaves = [
        jnp.asarray(
            flat[jax.tree_util.keystr(p)],
            dtype=_dtype_for(leaf, flat[jax.tree_util.keystr(p)]),
        )
        for p, leaf in leaves_with_paths
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


def make_train_step(
    optimizer: optax.GradientTransformation,
    variants: tuple[VariantSpec, ...],
    *,
    compute_dtype: jnp.dtype | None = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
    stochastic_variants: bool = False,
    accumulation_steps: int = 1,
) -> Callable[[TrainState, Batch], tuple[TrainState, Float[Array, ""]]]:
    """Return a JIT-compiled single training step closing over the
    optimizer + variant list.

    The returned function has signature
    ``(state, batch) -> (new_state, loss)``. ``state.step`` is a JAX
    scalar so the JIT trace is value-independent — same compiled
    program for step 0 and step 999.

    With ``accumulation_steps > 1`` the input ``batch`` gains a leading
    microbatch axis: each leaf's first dim becomes ``accumulation_steps``
    (so e.g. ``tokens`` is ``(N, B_micro, T)``). The body scans over
    that axis, sums grads, and issues one optimizer.update — equivalent
    to a single forward+backward on a batch of ``N * B_micro`` games
    but with memory cost ``B_micro × T`` per micro-pass instead of
    ``N × B_micro × T``. C.4 win: at LARGE on a 5090 a B=64 step beats
    B=128 by ~7% (smaller batch fits kernel-shape better); accumulation
    lets us train at effective batch 128 paying B=64's per-step price.

    The optimizer update is unconditional. An earlier version wrapped
    it in ``jax.lax.cond`` to skip the AdamW step on all-PAD batches —
    the concern being that ``weight_decay * model_params`` would still
    fire and drift params toward zero. In practice the cond hurt more
    than it helped: XLA traces and computes **both** branches of a
    ``lax.cond`` (the saving is in the selected output, not the
    compute), so the empty-batch guard cost a ~0.5-1 ms/step ``select``
    over every leaf of model + opt_state on every batch, including the
    >99% of batches with no empty positions. The Rust engine guarantees
    at least one supervised position per real game and Lichess
    filtering rejects pathological games, so we accept the tiny drift
    risk on the (hypothetical) all-PAD batch.
    """
    if accumulation_steps < 1:
        raise ValueError(
            f"accumulation_steps must be ≥ 1, got {accumulation_steps}"
        )

    def _loss_for(model: PAWNModel, batch: Batch, sub_key: jax.Array | None
                  ) -> Float[Array, ""]:
        return supernet_joint_loss(
            model, batch, variants,
            compute_dtype=compute_dtype,
            use_sdpa=use_sdpa, use_flash=use_flash,
            stochastic_key=sub_key,
        )

    @eqx.filter_jit(donate="all")
    def train_step(
        state: TrainState, batch: Batch
    ) -> tuple[TrainState, Float[Array, ""]]:
        sub_key = (
            jax.random.fold_in(state.key, state.step)
            if stochastic_variants
            else None
        )

        if accumulation_steps == 1:
            loss, grads = eqx.filter_value_and_grad(
                lambda model: _loss_for(model, batch, sub_key)
            )(state.model)
        else:
            # Accumulate grads + loss over the leading microbatch axis
            # via lax.scan. The body computes one micro's value+grad and
            # adds it into the carry. Memory: only one model-grad tree
            # resident at a time (NOT the per-micro stack a vmap would
            # produce). Compile cost: the body is traced once.
            params = eqx.filter(state.model, eqx.is_inexact_array)
            zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
            zero_loss = jnp.float32(0.0)
            micro_indices = jnp.arange(accumulation_steps, dtype=jnp.int32)

            def _acc_body(carry, micro_pair):
                accum_grads, accum_loss = carry
                micro_idx, micro_batch = micro_pair
                # Each micro gets a distinct fold of the sub_key so
                # stochastic-variant sampling sees different draws per
                # micro (otherwise N micros would all sample the same
                # variant and the accumulated grad would be biased).
                micro_key = (
                    jax.random.fold_in(sub_key, micro_idx)
                    if sub_key is not None
                    else None
                )
                micro_loss, micro_grads = eqx.filter_value_and_grad(
                    lambda model: _loss_for(model, micro_batch, micro_key)
                )(state.model)
                new_grads = jax.tree_util.tree_map(
                    lambda a, g: a + g, accum_grads, micro_grads
                )
                return (new_grads, accum_loss + micro_loss), None

            (acc_grads, acc_loss), _ = jax.lax.scan(
                _acc_body, (zero_grads, zero_loss), (micro_indices, batch)
            )
            scale = jnp.float32(1.0 / accumulation_steps)
            grads = jax.tree_util.tree_map(lambda g: g * scale, acc_grads)
            loss = acc_loss * scale

        # ``optimizer.update`` expects ``params`` to be a generic PyTree
        # — :class:`PAWNModel` is one (it's an ``eqx.Module``), but pyright
        # narrows the field type to the concrete class. Pass through a
        # local binding so the type widens to ``Any`` for the call.
        params_any: Any = state.model
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, params_any
        )
        new_model = eqx.apply_updates(state.model, updates)

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
    *,
    emit_grad_norms: bool = False,
    accuracy_fn: Callable[[PAWNModel, Batch], Float[Array, ""]] | None = None,
) -> Callable[..., tuple]:
    """Wrap a single train step into a K-step :func:`jax.lax.scan`.

    Input is a ``Batch`` whose leaves have a leading K axis (so
    ``batch.tokens`` is ``(K, B, T)`` etc.). Output is the final state
    and a ``(K,)`` array of per-step losses. With
    ``emit_grad_norms=True`` the return also includes ``(K,)`` of
    pre-clip grad norms drawn from the optimizer's ``_ClipState`` —
    used by C.5's clip-trigger measurement.

    With ``accuracy_fn`` supplied the return also includes a trailing
    ``(K,)`` array of per-step top-1 accuracies — the v2 parity of v1's
    pretrain ``train/accuracy`` metric. The function is evaluated on the
    **pre-step** model in the carry (the same weights that produced that
    step's loss), so the loss and accuracy describe the same forward. It
    runs inside the scan body, so the per-chunk D→H sync count is
    unchanged (one stacked array more per chunk, no extra host round
    trips). The output tuple ordering is
    ``(state, losses[, g_norms], accuracy)`` — grad norms, when present,
    precede accuracy.

    The body never returns to the host — that's the v2 amortisation.
    Per-chunk metrics flush between calls (the trainer loop drives the
    K-step boundaries from Python).
    """

    if emit_grad_norms and accuracy_fn is not None:
        acc_fn = accuracy_fn

        @eqx.filter_jit(donate="all")
        def scan_step_norms_acc(
            state: TrainState, batches: Batch
        ) -> tuple[
            TrainState, Float[Array, "K"], Float[Array, "K"], Float[Array, "K"]
        ]:
            def body(
                carry: TrainState, batch: Batch
            ) -> tuple[
                TrainState,
                tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]],
            ]:
                acc = acc_fn(carry.model, batch)
                new_carry, loss = train_step(carry, batch)
                g_norm = get_grad_norm(new_carry.opt_state)
                return new_carry, (loss, g_norm, acc)

            final_state, (losses, g_norms, accs) = jax.lax.scan(
                body, state, batches
            )
            return final_state, losses, g_norms, accs

        return scan_step_norms_acc

    if accuracy_fn is not None:
        acc_fn = accuracy_fn

        @eqx.filter_jit(donate="all")
        def scan_step_acc(
            state: TrainState, batches: Batch
        ) -> tuple[TrainState, Float[Array, "K"], Float[Array, "K"]]:
            def body(
                carry: TrainState, batch: Batch
            ) -> tuple[
                TrainState, tuple[Float[Array, ""], Float[Array, ""]]
            ]:
                acc = acc_fn(carry.model, batch)
                new_carry, loss = train_step(carry, batch)
                return new_carry, (loss, acc)

            final_state, (losses, accs) = jax.lax.scan(body, state, batches)
            return final_state, losses, accs

        return scan_step_acc

    if emit_grad_norms:
        @eqx.filter_jit(donate="all")
        def scan_step_with_norms(
            state: TrainState, batches: Batch
        ) -> tuple[TrainState, Float[Array, "K"], Float[Array, "K"]]:
            def body(
                carry: TrainState, batch: Batch
            ) -> tuple[TrainState, tuple[Float[Array, ""], Float[Array, ""]]]:
                new_carry, loss = train_step(carry, batch)
                g_norm = get_grad_norm(new_carry.opt_state)
                return new_carry, (loss, g_norm)

            final_state, (losses, g_norms) = jax.lax.scan(body, state, batches)
            return final_state, losses, g_norms

        return scan_step_with_norms

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
