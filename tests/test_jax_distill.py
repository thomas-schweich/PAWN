"""Tests for `pawn.distill` — logit-only teacher→student distillation.

Covers the Stage-B1 acceptance gates (`docs/phase_b_spec.md`):

(a) ``kl`` loss is 0 when student == teacher.
(b) teacher params bit-identical before/after a step (zero teacher grad).
(c) grad flows only to student leaves.
(d) reserved / NULL columns get no KL mass / no gradient.
(e) ``mix`` reduces to ``ce`` at alpha=1 and ``kl`` at alpha=0.
(f) the scan step matches a single-step reference within fp32 noise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pawn.config import (
    BOS_TOKEN,
    NULL_TOKEN,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    VOCAB_SIZE,
    ModelConfig,
)
from pawn.corpus import generate_corpus
from pawn.distill import (
    DistillTrainState,
    distill_loss,
    frozen_teacher,
    kl_divergence_loss,
    make_distill_scan_step,
    make_distill_train_step,
)
from pawn.model import PAWNModel, init_model
from pawn.trainer import (
    Batch,
    distill_column_mask,
    make_lr_schedule,
    make_optimizer,
    slice_batch,
)
from pawn.run_config import DistillConfig


# Tiny shapes keep these CPU-runnable. head_dim = d_model / n_heads.
_TEACHER_CFG = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, head_dim=16)
_STUDENT_CFG = ModelConfig(d_model=16, n_layers=2, n_heads=1, d_ff=64, head_dim=16)


def _batch(seq_len: int = 24, n_games: int = 6) -> Batch:
    corpus = generate_corpus(
        n_games=n_games, max_ply=seq_len, seq_len=seq_len, seed=3,
        conditioning=[],
    )
    idx = np.arange(min(n_games, corpus.n_games))
    return slice_batch(corpus, idx)


def _distill_cfg(**overrides: Any) -> DistillConfig:
    base: dict[str, Any] = dict(
        run_type="distill",
        distill_from="unused-in-test",
        student_supernet="tiny",
        total_steps=4,
        batch_size=4,
        seq_len=24,
        local_checkpoints=True,
    )
    base.update(overrides)
    return DistillConfig(**base)


_StepFn = Callable[
    [DistillTrainState, Batch], tuple[DistillTrainState, jax.Array]
]


# ---------------------------------------------------------------------------
# (a) KL is zero when student == teacher
# ---------------------------------------------------------------------------


def test_kl_zero_when_student_equals_teacher() -> None:
    model = init_model(_TEACHER_CFG, key=0)
    batch = _batch()
    logits = model(batch.tokens, batch.attn_mask)
    kl = kl_divergence_loss(logits, logits, batch.loss_mask, temperature=2.0)
    assert float(kl) == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_when_models_differ() -> None:
    teacher = init_model(_TEACHER_CFG, key=0)
    student = init_model(_TEACHER_CFG, key=1)
    batch = _batch()
    t_logits = teacher(batch.tokens, batch.attn_mask)
    s_logits = student(batch.tokens, batch.attn_mask)
    kl = kl_divergence_loss(s_logits, t_logits, batch.loss_mask, temperature=2.0)
    assert float(kl) > 0.0


def _reference_kl(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    loss_mask: np.ndarray,
    *,
    temperature: float,
) -> float:
    """Independent numpy reference for ``kl_divergence_loss``.

    Pins the three load-bearing pieces of the implementation that the
    self-comparison tests cannot constrain (Phase-B spec, Stage B1):

    * the *direction* — ``KL(teacher ‖ student) = Σ p_t·(log p_t − log p_s)``
      (swapping the two arguments must change the value);
    * the ``1/T`` logit scaling *inside* both softmaxes;
    * the outer ``T²`` gradient-rescale factor (Hinton et al. 2015);

    over the supervised move-token support only (columns ``[0, PAD_TOKEN)``),
    averaged over the real (``loss_mask`` True) positions.
    """
    support = slice(0, PAD_TOKEN)

    def _masked_log_softmax(logits: np.ndarray) -> np.ndarray:
        scaled = logits[..., support].astype(np.float64) / temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        log_z = np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
        return scaled - log_z

    s_log_p = _masked_log_softmax(student_logits)
    t_log_p = _masked_log_softmax(teacher_logits)
    t_p = np.exp(t_log_p)
    per_pos = (t_p * (t_log_p - s_log_p)).sum(axis=-1)  # (B, T)
    per_pos = np.where(loss_mask, per_pos, 0.0)
    n_real = max(int(loss_mask.sum()), 1)
    kl = per_pos.sum() / n_real
    return float(kl * temperature**2)


@pytest.mark.parametrize("temperature", [2.0, 4.0])
def test_kl_matches_closed_form_reference(temperature: float) -> None:
    """Pin ``kl_divergence_loss`` to a hand-checked numpy reference.

    Hand-chosen asymmetric student≠teacher logits over a (1, 2, V) array with
    an all-True ``loss_mask``. Because the reference is computed independently
    in fp64 numpy, this catches: a transposed-argument bug (``KL(student ‖
    teacher)``), a dropped/incorrect ``T²`` factor, and a wrong ``1/T`` logit
    scaling or masked-softmax normaliser — none of which the
    student==teacher / scan-vs-loop / mix-reduction tests can detect.
    Running at ``T ∈ {2, 4}`` exercises both the ``1/T`` and ``T²`` paths.
    """
    v = VOCAB_SIZE
    rng = np.random.default_rng(0)
    # Asymmetric logits: distinct, non-degenerate distributions on the support
    # so the teacher‖student direction is pinned (swapping args changes KL).
    student_logits = rng.standard_normal((1, 2, v)).astype(np.float32) * 1.7
    teacher_logits = rng.standard_normal((1, 2, v)).astype(np.float32) * 0.9
    # Put deliberate mass on reserved columns to prove they're masked out.
    student_logits[..., PAD_TOKEN:] = 5.0
    teacher_logits[..., PAD_TOKEN:] = -3.0
    loss_mask = np.ones((1, 2), dtype=bool)

    actual = float(kl_divergence_loss(
        jnp.asarray(student_logits),
        jnp.asarray(teacher_logits),
        jnp.asarray(loss_mask),
        temperature=temperature,
    ))
    expected = _reference_kl(
        student_logits, teacher_logits, loss_mask, temperature=temperature,
    )
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)

    # Direction guard: the reversed (student‖teacher) reference must differ,
    # so a transposed-argument implementation cannot pass the equality above.
    reversed_ref = _reference_kl(
        teacher_logits, student_logits, loss_mask, temperature=temperature,
    )
    assert not (actual == pytest.approx(reversed_ref, rel=1e-5, abs=1e-6))

    # T² guard: the same masked divergence without the T² factor must differ,
    # so dropping ``* temperature**2`` cannot pass the equality above.
    assert not (actual == pytest.approx(
        expected / temperature**2, rel=1e-5, abs=1e-6,
    ))


# ---------------------------------------------------------------------------
# (b) teacher is bit-identical after a step (zero teacher grad)
# ---------------------------------------------------------------------------


def _make_state(
    objective: str = "mix",
) -> tuple[DistillTrainState, _StepFn, optax.GradientTransformation]:
    cfg = _distill_cfg(objective=objective)
    teacher = init_model(_TEACHER_CFG, key=0)
    student = init_model(_STUDENT_CFG, key=1)
    schedule = make_lr_schedule(cfg, cfg.total_steps or 4)
    optimizer = make_optimizer(cfg, schedule)
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
    state = DistillTrainState(
        student=student, teacher=teacher, opt_state=opt_state,
        step=jnp.int32(0), key=jax.random.key(0),
    )
    train_step = make_distill_train_step(
        optimizer, objective=cfg.objective,
        temperature=cfg.temperature, alpha=cfg.alpha,
    )
    return state, train_step, optimizer


def test_teacher_bit_identical_after_step() -> None:
    state, train_step, _ = _make_state()
    teacher_before = jax.tree_util.tree_map(
        lambda x: np.asarray(x).copy(),
        eqx.filter(state.teacher, eqx.is_inexact_array),
    )
    batch = _batch(seq_len=24, n_games=4)
    new_state, _loss = train_step(state, batch)
    teacher_after = eqx.filter(new_state.teacher, eqx.is_inexact_array)
    for before, after in zip(
        jax.tree_util.tree_leaves(teacher_before),
        jax.tree_util.tree_leaves(teacher_after),
    ):
        np.testing.assert_array_equal(before, np.asarray(after))


def test_teacher_receives_zero_gradient() -> None:
    """The frozen teacher's stop_gradient means a grad w.r.t. the teacher
    PyTree is exactly zero, even when the teacher is exposed to jax.grad."""
    teacher = init_model(_TEACHER_CFG, key=0)
    student = init_model(_STUDENT_CFG, key=1)
    batch = _batch(seq_len=24, n_games=4)

    def loss_fn(teacher_m: PAWNModel) -> jnp.ndarray:
        teacher_fn = frozen_teacher(teacher_m)
        return distill_loss(
            student, teacher_fn, batch, objective="kl", temperature=2.0,
        )

    grads = eqx.filter_grad(loss_fn)(teacher)
    for leaf in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array)):
        np.testing.assert_array_equal(
            np.asarray(leaf), np.zeros_like(np.asarray(leaf))
        )


# ---------------------------------------------------------------------------
# (c) grad flows only to student leaves
# ---------------------------------------------------------------------------


def test_grad_flows_only_to_student() -> None:
    teacher = init_model(_TEACHER_CFG, key=0)
    student = init_model(_STUDENT_CFG, key=1)
    batch = _batch(seq_len=24, n_games=4)
    teacher_fn = frozen_teacher(teacher)

    def loss_fn(student_m: PAWNModel) -> jnp.ndarray:
        return distill_loss(
            student_m, teacher_fn, batch, objective="mix",
            temperature=2.0, alpha=0.5,
        )

    grads = eqx.filter_grad(loss_fn)(student)
    # At least one student leaf has a non-trivial gradient.
    norms = [
        float(jnp.linalg.norm(leaf))
        for leaf in jax.tree_util.tree_leaves(
            eqx.filter(grads, eqx.is_inexact_array)
        )
    ]
    assert any(n > 0.0 for n in norms), "student received no gradient"


# ---------------------------------------------------------------------------
# (d) reserved / NULL columns get no KL mass / no gradient
# ---------------------------------------------------------------------------


def test_reserved_columns_get_no_kl_mass() -> None:
    """The KL distribution is restricted to the supervised move-token
    support: PAD / outcome / BOS / NULL / reserved columns (IDs >=
    PAD_TOKEN) are masked out of *both* softmaxes, so the teacher places
    zero soft-target mass there and the student's logits on those columns
    get exactly zero gradient (Phase-B spec, Stage B1 — boundary at
    PAD_TOKEN, not NULL_TOKEN)."""
    teacher = init_model(_TEACHER_CFG, key=0)
    student = init_model(_TEACHER_CFG, key=1)
    batch = _batch(seq_len=24, n_games=4)
    t_logits = teacher(batch.tokens, batch.attn_mask)

    def loss_fn(s_logits):
        return kl_divergence_loss(
            s_logits, t_logits, batch.loss_mask, temperature=2.0,
        )

    s_logits = student(batch.tokens, batch.attn_mask)
    grad_logits = jax.grad(loss_fn)(s_logits)
    v = s_logits.shape[-1]
    masked = np.asarray(distill_column_mask(v))
    # The KL mask boundary is PAD_TOKEN, per spec — not NULL_TOKEN. PAD,
    # the outcome tokens, BOS, and NULL/reserved are all masked; the last
    # surviving column is the final move token (PAD_TOKEN - 1).
    assert masked[PAD_TOKEN] and not masked[PAD_TOKEN - 1]
    # PAD, every outcome token, and BOS are now inside the masked block —
    # the exact columns the spec requires the teacher mass not to spread to.
    assert masked[PAD_TOKEN]
    assert masked[OUTCOME_TOKEN_BASE] and masked[OUTCOME_TOKEN_BASE + 10]
    assert masked[BOS_TOKEN]
    assert masked[NULL_TOKEN]
    # Every masked column must have exactly zero gradient on the student
    # logits (no KL mass distilled into PAD / outcome / BOS / reserved).
    grad_masked = np.asarray(grad_logits)[..., masked]
    np.testing.assert_array_equal(grad_masked, np.zeros_like(grad_masked))
    # And the surviving move-token columns carry real gradient.
    grad_support = np.asarray(grad_logits)[..., ~masked]
    assert np.abs(grad_support).sum() > 0.0


# ---------------------------------------------------------------------------
# (e) mix reduces to ce at alpha=1 and kl at alpha=0
# ---------------------------------------------------------------------------


def test_mix_reduces_to_ce_and_kl_at_extremes() -> None:
    from pawn.trainer import cross_entropy_loss

    teacher = init_model(_TEACHER_CFG, key=0)
    student = init_model(_STUDENT_CFG, key=1)
    batch = _batch(seq_len=24, n_games=4)
    teacher_fn = frozen_teacher(teacher)

    mix_a1 = float(distill_loss(
        student, teacher_fn, batch, objective="mix", temperature=2.0, alpha=1.0,
    ))
    ce = float(cross_entropy_loss(student, batch))
    assert mix_a1 == pytest.approx(ce, rel=1e-5, abs=1e-5)

    mix_a0 = float(distill_loss(
        student, teacher_fn, batch, objective="mix", temperature=2.0, alpha=0.0,
    ))
    kl = float(distill_loss(
        student, teacher_fn, batch, objective="kl", temperature=2.0,
    ))
    assert mix_a0 == pytest.approx(kl, rel=1e-5, abs=1e-5)


def test_objective_ce_matches_cross_entropy_loss() -> None:
    from pawn.trainer import cross_entropy_loss

    teacher = init_model(_TEACHER_CFG, key=0)
    student = init_model(_STUDENT_CFG, key=1)
    batch = _batch(seq_len=24, n_games=4)
    teacher_fn = frozen_teacher(teacher)
    ce_via_distill = float(distill_loss(
        student, teacher_fn, batch, objective="ce",
    ))
    ce_direct = float(cross_entropy_loss(student, batch))
    assert ce_via_distill == pytest.approx(ce_direct, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# (f) scan step matches a single-step reference within fp32 noise
# ---------------------------------------------------------------------------


def test_scan_step_matches_single_step_reference() -> None:
    k = 3
    batches = [_batch(seq_len=24, n_games=4) for _ in range(k)]

    # The train step donates its input buffers, so the reference loop and
    # the scan each need their own (identical) starting state. Both states
    # are built from the same fixed keys, so they are bit-identical at
    # step 0.
    ref_state, train_step, _ = _make_state(objective="mix")
    scan_seed_state, _, _ = _make_state(objective="mix")

    # Reference: K sequential single-steps.
    for b in batches:
        ref_state, _ = train_step(ref_state, b)

    # Scan: stack the K batches along a leading axis and run one scan.
    stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *batches
    )
    scan_step = make_distill_scan_step(train_step)
    scan_state, losses = scan_step(scan_seed_state, stacked)

    assert losses.shape == (k,)
    ref_leaves = jax.tree_util.tree_leaves(
        eqx.filter(ref_state.student, eqx.is_inexact_array)
    )
    scan_leaves = jax.tree_util.tree_leaves(
        eqx.filter(scan_state.student, eqx.is_inexact_array)
    )
    for r, s in zip(ref_leaves, scan_leaves):
        np.testing.assert_allclose(
            np.asarray(r), np.asarray(s), rtol=1e-5, atol=1e-5
        )


def test_loss_decreases_over_a_few_steps() -> None:
    """Sanity: the student loss trends down over a handful of steps (KL
    finite + decreasing — smoke gate)."""
    state, train_step, _ = _make_state(objective="mix")
    batch = _batch(seq_len=24, n_games=4)
    losses = []
    for _ in range(8):
        state, loss = train_step(state, batch)
        losses.append(float(loss))
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
