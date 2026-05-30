"""Move-accuracy evaluation — overall + per-phase breakdown.

Argmax is restricted to ``[0, NUM_ACTIONS)`` so PAD and outcome tokens
can't be sampled (the v1 contract per plan §10 S8). Per-phase
breakdown bins moves by game phase (opening / midgame / endgame) on a
ply threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from pawn.config import NUM_ACTIONS
from pawn.corpus import Corpus
from pawn.model import PAWNModel

__all__ = [
    "PhaseBoundaries",
    "AccuracyResult",
    "compute_move_accuracy",
    "compute_per_phase_accuracy",
]


@dataclass(frozen=True)
class PhaseBoundaries:
    """Ply thresholds for opening / midgame / endgame split.

    Defaults: opening ≤ 20 ply (10 full moves); midgame 21..60; endgame > 60.
    Boundaries are inclusive on the lower bound, exclusive on the upper.
    """

    opening_end: int = 20
    midgame_end: int = 60


@dataclass(frozen=True)
class AccuracyResult:
    """Aggregate accuracy + per-phase breakdown."""

    overall: float
    opening: float
    midgame: float
    endgame: float
    n_total: int
    n_opening: int
    n_midgame: int
    n_endgame: int


@eqx.filter_jit
def _argmax_over_actions(
    logits: Float[Array, "B T V"],
) -> Int[Array, "B T"]:
    """Argmax restricted to ``[0, NUM_ACTIONS)``. PAD (token 1968) and
    outcome tokens (1969+) can't be sampled."""
    move_logits = logits[..., :NUM_ACTIONS]
    return jnp.argmax(move_logits, axis=-1)


@eqx.filter_jit
def _batch_correct(
    model: PAWNModel,
    tokens: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    attn_mask: Bool[Array, "B T"],
    loss_mask: Bool[Array, "B T"],
) -> tuple[Bool[Array, "B T"], Bool[Array, "B T"]]:
    """JIT'd inner: returns (correct, loss_mask) bool tensors so the
    Python loop can aggregate over chunks."""
    logits = model(tokens, attn_mask)
    pred = _argmax_over_actions(logits)
    correct = (pred == targets) & loss_mask
    return correct, loss_mask


def compute_move_accuracy(
    model: PAWNModel,
    corpus: Corpus,
    *,
    batch_size: int = 32,
) -> float:
    """Overall move-prediction accuracy on a Corpus.

    Iterates in chunks of ``batch_size``; returns the fraction of
    supervised positions where ``argmax(logits[:NUM_ACTIONS]) ==
    target``. Returns 0.0 if the corpus has no supervised positions.
    """
    n = corpus.n_games
    total_correct = 0
    total_supervised = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        correct, supervised = _batch_correct(model, tokens, targets, attn, loss)
        total_correct += int(correct.sum())
        total_supervised += int(supervised.sum())
    if total_supervised == 0:
        return 0.0
    return total_correct / total_supervised


def compute_per_phase_accuracy(
    model: PAWNModel,
    corpus: Corpus,
    *,
    batch_size: int = 32,
    phases: PhaseBoundaries = PhaseBoundaries(),
) -> AccuracyResult:
    """Move accuracy broken down by game phase.

    Phase membership is keyed on the **ply** a position predicts, not its
    raw sequence index. The conditioning prefix (Phase-A Chunk 4) shifts
    every move ``C`` slots to the right, where ``C`` is the prefix width
    persisted as ``corpus.outcome_offset`` (constant across the corpus).
    A position at sequence index ``t`` therefore predicts ply ``t - C``;
    binning on the raw ``t`` would smear the boundaries by ``C`` and
    mislabel the first ``C`` plies as opening padding (plan §8.1).

    A position at ply ``p = t - C`` belongs to:
    - opening if ``p < phases.opening_end``,
    - midgame if ``phases.opening_end <= p < phases.midgame_end``,
    - endgame otherwise.

    Within each phase, only supervised positions contribute (the prefix
    slots are never supervised, so the negative-ply prefix region drops
    out of every phase regardless of how it bins).
    """
    seq_len = corpus.seq_len
    # ``outcome_offset`` is the constant prefix width C for every game in
    # the corpus (the slot where the first move lives). Read it off the
    # corpus rather than hardcoding a default so the binning tracks the
    # checkpoint's own conditioning layout.
    C = int(corpus.outcome_offset[0]) if corpus.n_games > 0 else 1
    positions = np.arange(seq_len)
    ply = positions - C
    opening_pos = ply < phases.opening_end
    midgame_pos = (ply >= phases.opening_end) & (ply < phases.midgame_end)
    endgame_pos = ply >= phases.midgame_end

    n = corpus.n_games
    total_correct = total_sup = 0
    o_correct = o_sup = 0
    m_correct = m_sup = 0
    e_correct = e_sup = 0

    op_mask = jnp.asarray(opening_pos, dtype=jnp.bool_)
    mg_mask = jnp.asarray(midgame_pos, dtype=jnp.bool_)
    eg_mask = jnp.asarray(endgame_pos, dtype=jnp.bool_)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        correct, supervised = _batch_correct(model, tokens, targets, attn, loss)

        total_correct += int(correct.sum())
        total_sup += int(supervised.sum())

        # Broadcast per-position phase masks against (B, T) loss_mask.
        for phase_mask, c_acc, s_acc in (
            (op_mask, "o", "o"),
            (mg_mask, "m", "m"),
            (eg_mask, "e", "e"),
        ):
            in_phase = supervised & phase_mask[None, :]
            phase_correct = int((correct & phase_mask[None, :]).sum())
            phase_supervised = int(in_phase.sum())
            if c_acc == "o":
                o_correct += phase_correct
                o_sup += phase_supervised
            elif c_acc == "m":
                m_correct += phase_correct
                m_sup += phase_supervised
            else:
                e_correct += phase_correct
                e_sup += phase_supervised

    def safe_div(num: int, den: int) -> float:
        return num / den if den > 0 else 0.0

    return AccuracyResult(
        overall=safe_div(total_correct, total_sup),
        opening=safe_div(o_correct, o_sup),
        midgame=safe_div(m_correct, m_sup),
        endgame=safe_div(e_correct, e_sup),
        n_total=total_sup,
        n_opening=o_sup,
        n_midgame=m_sup,
        n_endgame=e_sup,
    )
