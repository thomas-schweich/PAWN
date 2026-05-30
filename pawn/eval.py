"""Move-accuracy evaluation — overall + per-phase breakdown.

Argmax is restricted to ``[0, NUM_ACTIONS)`` so PAD and outcome tokens
can't be sampled (the v1 contract per plan §10 S8). Per-phase
breakdown bins moves by game phase (opening / midgame / endgame) on a
ply threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
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
) -> tuple[Int[Array, ""], Int[Array, ""]]:
    """JIT'd inner: returns (n_correct, n_supervised) as device scalars
    so the Python loop accumulates on-device and syncs once at the end."""
    logits = model(tokens, attn_mask)
    pred = _argmax_over_actions(logits)
    correct = (pred == targets) & loss_mask
    return correct.sum(), loss_mask.sum()


@eqx.filter_jit
def _batch_phase_counts(
    model: PAWNModel,
    tokens: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    attn_mask: Bool[Array, "B T"],
    loss_mask: Bool[Array, "B T"],
    phase_masks: Bool[Array, "P T"],
) -> tuple[Int[Array, ""], Int[Array, ""], Int[Array, "P"], Int[Array, "P"]]:
    """JIT'd inner for the per-phase breakdown.

    Returns ``(n_correct, n_supervised, phase_correct, phase_supervised)``
    where the two ``P``-vectors carry the per-phase counts (one entry per
    row of ``phase_masks``). All reductions happen inside the jit so the
    Python loop syncs a single small tuple per chunk instead of eight
    separate ``int()`` host round-trips.
    """
    logits = model(tokens, attn_mask)
    pred = _argmax_over_actions(logits)
    correct = (pred == targets) & loss_mask  # (B, T)
    # Broadcast the (P, T) per-position phase masks against (B, T).
    in_phase = loss_mask[None, :, :] & phase_masks[:, None, :]  # (P, B, T)
    phase_correct = (correct[None, :, :] & phase_masks[:, None, :]).sum(axis=(1, 2))
    phase_supervised = in_phase.sum(axis=(1, 2))
    return correct.sum(), loss_mask.sum(), phase_correct, phase_supervised


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
    total_correct: Array = jnp.zeros((), dtype=jnp.int32)
    total_supervised: Array = jnp.zeros((), dtype=jnp.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        correct, supervised = _batch_correct(model, tokens, targets, attn, loss)
        total_correct = total_correct + correct
        total_supervised = total_supervised + supervised
    # Single host sync after the device-side accumulation.
    total_c = int(total_correct)
    total_s = int(total_supervised)
    if total_s == 0:
        return 0.0
    return total_c / total_s


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

    # Stack the three per-position phase masks into one (P=3, T) tensor —
    # row 0 opening, row 1 midgame, row 2 endgame. The jitted body reduces
    # over (B, T) per phase, so the host only sees small device scalars.
    phase_masks = jnp.asarray(
        np.stack([opening_pos, midgame_pos, endgame_pos], axis=0), dtype=jnp.bool_,
    )

    total_correct: Array = jnp.zeros((), dtype=jnp.int32)
    total_sup: Array = jnp.zeros((), dtype=jnp.int32)
    phase_correct: Array = jnp.zeros((3,), dtype=jnp.int32)
    phase_sup: Array = jnp.zeros((3,), dtype=jnp.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        c, s, pc, ps = _batch_phase_counts(
            model, tokens, targets, attn, loss, phase_masks,
        )
        total_correct = total_correct + c
        total_sup = total_sup + s
        phase_correct = phase_correct + pc
        phase_sup = phase_sup + ps

    # Single host sync after the device-side accumulation: one transfer of
    # the two scalars plus the two length-3 vectors, instead of eight
    # per-chunk ``int()`` round-trips.
    total_c = int(total_correct)
    total_s = int(total_sup)
    pc_host = np.asarray(phase_correct)
    ps_host = np.asarray(phase_sup)
    o_correct, m_correct, e_correct = (int(x) for x in pc_host)
    o_sup, m_sup, e_sup = (int(x) for x in ps_host)

    def safe_div(num: int, den: int) -> float:
        return num / den if den > 0 else 0.0

    return AccuracyResult(
        overall=safe_div(total_c, total_s),
        opening=safe_div(o_correct, o_sup),
        midgame=safe_div(m_correct, m_sup),
        endgame=safe_div(e_correct, e_sup),
        n_total=total_s,
        n_opening=o_sup,
        n_midgame=m_sup,
        n_endgame=e_sup,
    )
