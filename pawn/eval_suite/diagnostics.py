"""Edge-case diagnostics — guaranteed coverage via `engine.edge_case_bits()`.

The Rust engine returns per-position bits that flag edge cases
(in_check / double_check / pin_restricts / ep_available / castle_legal_*).
The diagnostic computes move accuracy on positions matching each bit
so the eval has guaranteed coverage of the rare cases.

This module ships a stub-but-functional implementation: it accepts a
batch of game move IDs + game lengths, calls into the engine to get
the edge-case bits, and reports per-bit accuracy. The engine call is
isolated behind a thin wrapper so the test suite can mock it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import chess_engine as engine
from pawn.config import NUM_ACTIONS
from pawn.model import PAWNModel

__all__ = [
    "EdgeCaseResult",
    "EDGE_CASE_LABELS",
    "compute_edge_case_accuracy",
]


# Canonical v2 edge-case labels. Each maps to a bit-mask name in the
# Rust engine's ``edge_case_bits()`` dict. The user-facing label keeps
# the v1 snake_case names per plan §10 S8.
EDGE_CASE_LABELS = (
    "in_check",
    "double_check",
    "pin_restricts",
    "ep_available",
    "castle_legal_kingside",
    "castle_legal_queenside",
)

# Mapping label → engine bit-mask name.
_LABEL_TO_BIT_NAME = {
    "in_check": "IN_CHECK",
    "double_check": "IN_DOUBLE_CHECK",
    "pin_restricts": "PIN_RESTRICTS_MOVEMENT",
    "ep_available": "EP_CAPTURE_AVAILABLE",
    "castle_legal_kingside": "CASTLE_LEGAL_KINGSIDE",
    "castle_legal_queenside": "CASTLE_LEGAL_QUEENSIDE",
}


@dataclass(frozen=True)
class EdgeCaseResult:
    label: str
    accuracy: float
    n_positions: int


def compute_edge_case_accuracy(
    model: PAWNModel,
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    batch_size: int = 16,
) -> list[EdgeCaseResult]:
    """Compute per-edge-case move accuracy.

    ``move_ids`` is ``(N, max_ply) int16`` from
    ``engine.generate_random_games`` (or the Lichess parquet path);
    ``game_lengths`` is ``(N,)``. The function calls
    ``engine.edge_case_bits`` to flag positions, runs the model's
    forward pass, and computes per-bit argmax accuracy.

    Returns a list of :class:`EdgeCaseResult`, one per label in
    :data:`EDGE_CASE_LABELS`.
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    # `compute_edge_stats_per_ply(moves, lengths)` returns a 3-tuple;
    # the first element is the (N, max_ply) uint64 of bit-packed flags.
    bits, _, _ = engine.compute_edge_stats_per_ply(move_ids, game_lengths)
    # Bit-mask constants come from `engine.edge_case_bits()` (a dict).
    bit_table = engine.edge_case_bits()
    n, max_ply = move_ids.shape
    seq_len = max_ply

    # Pack input ids for the model — same shape as a Corpus tokens row.
    # We don't pad to a fixed seq_len since edge-case eval is per-position.
    from pawn.config import PAD_TOKEN
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    positions = np.arange(max_ply, dtype=np.int32)[None]
    valid = positions < game_lengths[:, None]
    tokens = np.where(valid, move_ids.astype(np.int32), PAD_TOKEN)
    attn = tokens != PAD_TOKEN

    # Targets are tokens shifted left by 1.
    targets = np.full_like(tokens, PAD_TOKEN)
    targets[:, :-1] = tokens[:, 1:]

    # Forward pass, restricted to action vocab for argmax.
    pred_chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        t = jnp.asarray(tokens[start:end])
        a = jnp.asarray(attn[start:end])
        logits = model(t, a)
        pred = np.asarray(jnp.argmax(logits[..., :NUM_ACTIONS], axis=-1))
        pred_chunks.append(pred)
    pred_all = np.concatenate(pred_chunks, axis=0)

    correct = (pred_all == targets) & attn

    results: list[EdgeCaseResult] = []
    for label in EDGE_CASE_LABELS:
        bit_name = _LABEL_TO_BIT_NAME[label]
        mask_value = bit_table[bit_name]
        mask = (bits & mask_value).astype(bool)
        n_pos = int(mask.sum())
        if n_pos == 0:
            results.append(EdgeCaseResult(label=label, accuracy=0.0, n_positions=0))
            continue
        n_correct = int((correct & mask).sum())
        results.append(
            EdgeCaseResult(label=label, accuracy=n_correct / n_pos, n_positions=n_pos)
        )
    return results
