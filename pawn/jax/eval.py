"""JAX move-prediction accuracy evaluation.

Phase-4 chunk 1: the minimum viable JAX eval that exercises the
full forward path end-to-end. Loads a JAX checkpoint, generates a
finite corpus of random games, computes per-position argmax accuracy
on supervised next-token positions, and reports overall + per-phase
(opening/midgame/endgame) breakdowns.

The remaining ``pawn.eval_suite`` surface — probes, generation
diagnostics, Lichess Elo-stratified eval — sits on the same forward
primitive and ports incrementally on top of this entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.jax.config import NUM_ACTIONS
from pawn.jax.corpus import Corpus
from pawn.jax.model import PAWNModel


@dataclass(frozen=True)
class AccuracyResult:
    """Move-prediction accuracy on a finite corpus.

    ``overall`` is the supervised-position-weighted mean accuracy.
    ``phase`` keys are ``"opening"`` (ply < 20), ``"midgame"``
    (20 <= ply < 60), ``"endgame"`` (ply >= 60). Each entry has
    ``(correct, total)`` where ``correct/total`` is the band's
    accuracy.
    """

    overall: float
    n_supervised: int
    phase: dict[str, tuple[int, int]]


def _ply_phase_indices(seq_len: int) -> tuple[int, int]:
    """Return ``(opening_end, midgame_end)`` boundaries in ply units."""
    return (20, 60)


def evaluate_accuracy(
    model: PAWNModel, corpus: Corpus, *, batch_size: int = 16
) -> AccuracyResult:
    """Compute move-prediction accuracy on ``corpus``.

    Argmax(logits) restricted to the legal action vocab ``[0,
    NUM_ACTIONS)`` is compared against the next-token target at every
    ``loss_mask=True`` position. Per-phase breakdown uses fixed ply
    boundaries (opening < 20, midgame 20-59, endgame ≥ 60).

    Args:
        model: forward callable taking ``(tokens, attn_mask) -> [B,
            T, V]`` logits.
        corpus: the finite eval corpus.
        batch_size: per-batch slice through the corpus.
    """

    @eqx.filter_jit
    def batch_correct(
        m: PAWNModel, tokens: jax.Array, attn_mask: jax.Array,
        targets: jax.Array, loss_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Per-batch reduction; returns (correct[B,T], target_in_band[B,T],
        position_grid[1,T])."""
        logits = m(tokens, attn_mask)
        # Restrict argmax to legal move tokens — the model's vocab
        # includes PAD + outcome tokens that should never be predicted
        # at a supervised position.
        action_logits = logits[..., :NUM_ACTIONS]
        pred = jnp.argmax(action_logits, axis=-1)
        # ``targets`` may contain PAD/outcome tokens at positions
        # outside loss_mask; the loss_mask gates whether we count
        # the comparison. We use ``equal`` directly — PAD targets
        # land outside the comparison via loss_mask.
        return pred == targets, loss_mask, jnp.arange(
            tokens.shape[1], dtype=jnp.int32
        )[None, :]

    n_games = corpus.n_games
    seq_len = corpus.seq_len
    # Phase breakdown counters (host-side accumulation).
    op_end, mg_end = _ply_phase_indices(seq_len)
    phase_counts: dict[str, list[int]] = {
        "opening": [0, 0],
        "midgame": [0, 0],
        "endgame": [0, 0],
    }
    total_correct = 0
    total_n = 0
    for start in range(0, n_games, batch_size):
        end = min(start + batch_size, n_games)
        # Trim to a multiple of (any) batch — JIT recompiles on
        # shape change, which is acceptable for a final-batch
        # remainder since eval runs once per checkpoint.
        tokens = jnp.asarray(corpus.tokens[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        loss_mask = jnp.asarray(corpus.loss_mask[start:end])
        correct, mask, pos_grid = batch_correct(
            model, tokens, attn, targets, loss_mask
        )
        # Pull to host as np arrays for the per-phase accumulation.
        correct_np = np.asarray(correct)
        mask_np = np.asarray(mask)
        pos_np = np.asarray(pos_grid)[0]  # [T]
        # Overall.
        valid = mask_np  # [B, T]
        total_correct += int((correct_np & valid).sum())
        total_n += int(valid.sum())
        # Per-phase: build [B, T] mask of (in-phase ∧ valid).
        op_mask_t = (pos_np < op_end)[None, :]
        mg_mask_t = ((pos_np >= op_end) & (pos_np < mg_end))[None, :]
        end_mask_t = (pos_np >= mg_end)[None, :]
        for name, phase_mask in (
            ("opening", op_mask_t),
            ("midgame", mg_mask_t),
            ("endgame", end_mask_t),
        ):
            band_valid = valid & phase_mask
            phase_counts[name][0] += int((correct_np & band_valid).sum())
            phase_counts[name][1] += int(band_valid.sum())

    overall = (total_correct / total_n) if total_n > 0 else 0.0
    return AccuracyResult(
        overall=overall,
        n_supervised=total_n,
        phase={
            name: (c, n) for name, (c, n) in phase_counts.items()
        },
    )
