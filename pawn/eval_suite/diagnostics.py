"""Edge-case diagnostics — guaranteed coverage via `engine.edge_case_bits()`.

The Rust engine returns per-position bits that flag edge cases
(in_check / double_check / pin_restricts / ep_available / castle_legal_*
/ castle_blocked_check / promotion_available / checkmate / stalemate).
The diagnostic computes move accuracy on positions matching each bit so
the eval has guaranteed coverage of the rare cases.

Two corpus sources:

- :func:`compute_edge_case_accuracy` accepts a pre-existing
  ``(move_ids, game_lengths)`` corpus from
  ``engine.generate_random_games`` or the Lichess parquet path. Useful
  when the caller has already chosen the game pool.
- :func:`compute_edge_case_accuracy_quota` calls
  ``engine.generate_diagnostic_sets`` (v1 parity), which uses quota
  sampling to *guarantee* coverage for every label rather than hoping
  random games happen to surface them. This is the path
  ``scripts/eval_generation_jax.py`` uses when the operator passes
  ``--edge-cases``.

Both paths share the same per-bit accuracy computation and return a
list of :class:`EdgeCaseResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import chess_engine as engine
from pawn.config import NUM_ACTIONS, PAD_TOKEN
from pawn.corpus import (
    _map_termination_to_outcome,
    build_prefix,
    conditioning_to_C,
)
from pawn.model import EffectiveCallable, PAWNModel

__all__ = [
    "EdgeCaseResult",
    "EDGE_CASE_LABELS",
    "compute_edge_case_accuracy",
    "compute_edge_case_accuracy_quota",
    "default_diagnostic_quotas",
]


# Canonical v2 edge-case labels — full v1 parity per
# docs/JAX_PARITY_SHORTFALLS.md §11. Each maps to a bit-mask name in
# the Rust engine's ``edge_case_bits()`` dict. The user-facing label
# keeps the v1 snake_case names per plan §10 S8.
EDGE_CASE_LABELS = (
    "in_check",
    "double_check",
    "pin_restricts",
    "ep_available",
    "castle_legal_kingside",
    "castle_legal_queenside",
    "castle_blocked_check",
    "promotion_available",
    "checkmate",
    "stalemate",
)

# Mapping label → engine bit-mask name. The four labels added in
# parity #6 (`castle_blocked_check`, `promotion_available`, `checkmate`,
# `stalemate`) map to the matching bits already exported by the engine.
_LABEL_TO_BIT_NAME = {
    "in_check": "IN_CHECK",
    "double_check": "IN_DOUBLE_CHECK",
    "pin_restricts": "PIN_RESTRICTS_MOVEMENT",
    "ep_available": "EP_CAPTURE_AVAILABLE",
    "castle_legal_kingside": "CASTLE_LEGAL_KINGSIDE",
    "castle_legal_queenside": "CASTLE_LEGAL_QUEENSIDE",
    "castle_blocked_check": "CASTLE_BLOCKED_CHECK",
    "promotion_available": "PROMOTION_AVAILABLE",
    "checkmate": "CHECKMATE",
    "stalemate": "STALEMATE",
}

# `generate_diagnostic_sets` expects 64-element int32 quota arrays
# indexed by bit position. The engine's `edge_case_bits()` returns the
# bit *value* (1 << position); we convert.
def _quota_index_for_label(label: str, bit_table: dict[str, int]) -> int:
    bit_name = _LABEL_TO_BIT_NAME[label]
    val = int(bit_table[bit_name])
    # bit position = log2(value) since each edge-case constant is a
    # single set bit.
    return int(val.bit_length()) - 1


def default_diagnostic_quotas(
    per_label: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(quotas_white, quotas_black)`` arrays for
    :func:`compute_edge_case_accuracy_quota` from a per-label count.

    Same ``per_label`` applied to both colours so we get balanced
    coverage. Returns two int32 arrays of length 64 (the engine's
    contract; positions other than our 10 labels are zero).
    """
    bit_table = engine.edge_case_bits()
    quotas = np.zeros(64, dtype=np.int32)
    for label in EDGE_CASE_LABELS:
        quotas[_quota_index_for_label(label, bit_table)] = per_label
    return quotas, quotas.copy()


@dataclass(frozen=True)
class EdgeCaseResult:
    label: str
    accuracy: float
    n_positions: int


def _resolve_outcome_tokens(
    term_codes: np.ndarray | None,
    game_lengths: np.ndarray,
) -> np.ndarray:
    """Per-game outcome token IDs for the ``"outcome"`` conditioning slot.

    Maps the engine's termination codes through the same
    :func:`pawn.corpus._map_termination_to_outcome` the trainer uses, so
    the diagnostic prefix carries the exact outcome token the model was
    conditioned on. When ``term_codes`` is unavailable the games carry no
    resolvable outcome; we fall back to a ``< 0`` sentinel that
    :func:`pawn.corpus.build_prefix` resolves to ``NULL_TOKEN`` (only
    consumed when ``"outcome"`` is actually in the conditioning list).
    """
    n = int(np.asarray(game_lengths).shape[0])
    if term_codes is None:
        return np.full(n, -1, dtype=np.int32)
    return _map_termination_to_outcome(
        np.asarray(term_codes), np.asarray(game_lengths)
    )


def _compute_per_bit_accuracy(
    model: "PAWNModel | EffectiveCallable",
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    bits: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    conditioning: Sequence[str],
    batch_size: int,
) -> list[EdgeCaseResult]:
    """Shared per-bit accuracy core for both
    :func:`compute_edge_case_accuracy` and
    :func:`compute_edge_case_accuracy_quota`.

    ``bits`` is the ``(n, max_ply)`` uint64 from
    :func:`chess_engine.compute_edge_stats_per_ply`: ``bits[g, j]`` is
    the edge-case bitfield of the position the model plays ``move[j]``
    *from*, for ``j < game_length[g]``; ``bits[g, game_length[g]]`` is
    the *terminal* position reached after the last move (checkmate /
    stalemate live exclusively there). See ``engine/src/edgestats.rs``
    (``compute_edge_stats_per_ply`` writes ``length + 1`` plies).

    Layout (Phase-A): tokens are assembled with the run's conditioning
    prefix ``[BOS][cond…]`` of width ``C = 1 + len(conditioning)`` so the
    model sees the same absolute-RoPE layout it trained under; ``move[j]``
    lands at sequence slot ``C + j`` and is predicted at slot ``C + j - 1``
    (target ``tokens[:, C + j]``).

    Per-bit alignment (H5): the prediction made at slot ``s`` concerns
    ply ``j = s + 1 - C`` (the move it predicts is ``move[j]``), so each
    seq-slot correctness is matched against ``bits[:, j]`` — i.e. ``bits``
    shifted right by ``C`` along the sequence axis. The terminal ply
    ``j = game_length`` has a PAD target (no move follows the final move);
    we score whether the model predicts the game is over there
    (``argmax`` over the *full* vocab equals PAD) rather than scoring a
    move-argmax against PAD, which previously forced the checkmate /
    stalemate labels to ``accuracy=0`` (the attn mask zeroed that slot).
    """
    bit_table = engine.edge_case_bits()
    n, max_ply = move_ids.shape
    C = conditioning_to_C(conditioning)
    move_ids = move_ids.astype(np.int32)
    game_lengths = np.asarray(game_lengths, dtype=np.int32)

    # Sequence: [BOS][cond…][move_0 … move_{L-1}][PAD …]. One trailing slot
    # past the last move holds the predict-PAD (terminal) target, so the
    # terminal ply j = game_length is representable for every game whose
    # terminal bits fit (game_length < max_ply, the engine's own guard).
    seq_len = C + max_ply
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[:, :C] = build_prefix(conditioning, outcome_tokens, n)
    move_positions = np.arange(max_ply, dtype=np.int32)[None]
    valid_move = move_positions < game_lengths[:, None]
    tokens[:, C:] = np.where(valid_move, move_ids, PAD_TOKEN)
    attn = tokens != PAD_TOKEN

    # Target at slot s is tokens[:, s + 1]; the PAD trailing slot is benign
    # (the per-bit masks below decide which slots count).
    targets = np.full_like(tokens, PAD_TOKEN)
    targets[:, :-1] = tokens[:, 1:]

    action_pred_chunks: list[np.ndarray] = []
    full_pred_chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        t = jnp.asarray(tokens[start:end])
        a = jnp.asarray(attn[start:end])
        logits = model(t, a)
        action_pred = np.asarray(jnp.argmax(logits[..., :NUM_ACTIONS], axis=-1))
        full_pred = np.asarray(jnp.argmax(logits, axis=-1))
        action_pred_chunks.append(action_pred)
        full_pred_chunks.append(full_pred)
    action_pred = np.concatenate(action_pred_chunks, axis=0)
    full_pred = np.concatenate(full_pred_chunks, axis=0)

    # Per-slot correctness. For a move target (the common case) the model
    # must argmax the right action; for the terminal predict-PAD slot it
    # must argmax PAD over the full vocab (game-is-over). The terminal slot
    # is where targets == PAD inside the supervised window; everywhere else
    # uses the action argmax.
    is_pad_target = targets == PAD_TOKEN
    move_correct = (action_pred == targets) & ~is_pad_target
    terminal_correct = (full_pred == PAD_TOKEN) & is_pad_target
    correct_slot = move_correct | terminal_correct

    # Align bits onto sequence slots: slot s scores ply j = s + 1 - C, so
    # aligned_bits[:, s] = bits[:, s + 1 - C] for s + 1 - C in [0, max_ply).
    aligned_bits = np.zeros((n, seq_len), dtype=bits.dtype)
    # Valid source plies j map to slots s = C - 1 + j for j in [0, max_ply).
    aligned_bits[:, C - 1 : C - 1 + max_ply] = bits

    results: list[EdgeCaseResult] = []
    for label in EDGE_CASE_LABELS:
        bit_name = _LABEL_TO_BIT_NAME[label]
        mask_value = bit_table[bit_name]
        mask = (aligned_bits & mask_value).astype(bool)
        n_pos = int(mask.sum())
        if n_pos == 0:
            results.append(EdgeCaseResult(label=label, accuracy=0.0, n_positions=0))
            continue
        n_correct = int((correct_slot & mask).sum())
        results.append(
            EdgeCaseResult(label=label, accuracy=n_correct / n_pos, n_positions=n_pos)
        )
    return results


def compute_edge_case_accuracy(
    model: "PAWNModel | EffectiveCallable",
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    term_codes: np.ndarray | None = None,
    conditioning: Sequence[str] = (),
    batch_size: int = 16,
) -> list[EdgeCaseResult]:
    """Compute per-edge-case move accuracy on a pre-existing corpus.

    ``move_ids`` is ``(N, max_ply) int16`` from
    ``engine.generate_random_games`` (or the Lichess parquet path);
    ``game_lengths`` is ``(N,)``. The function calls
    ``engine.compute_edge_stats_per_ply`` to flag positions, runs the
    model's forward pass, and computes per-bit argmax accuracy.

    ``conditioning`` is the checkpoint's prefix-kind list (Phase-A); the
    tokens are assembled with that ``[BOS][cond…]`` prefix so the model
    runs on its in-distribution absolute-RoPE layout. ``term_codes`` (the
    engine's per-game termination codes) resolves the ``"outcome"``
    conditioning slot when present; with no ``"outcome"`` kind in
    ``conditioning`` it is unused and may be omitted.

    Returns a list of :class:`EdgeCaseResult`, one per label in
    :data:`EDGE_CASE_LABELS`. Rare edge cases (checkmate, stalemate)
    may have ``n_positions == 0`` on a small random corpus — use
    :func:`compute_edge_case_accuracy_quota` for guaranteed coverage.
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    bits, _, _ = engine.compute_edge_stats_per_ply(move_ids, game_lengths)
    outcome_tokens = _resolve_outcome_tokens(term_codes, game_lengths)
    return _compute_per_bit_accuracy(
        model, move_ids, game_lengths, bits, outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
    )


def compute_edge_case_accuracy_quota(
    model: "PAWNModel | EffectiveCallable",
    *,
    per_label: int = 10,
    max_ply: int = 256,
    seed: int = 42,
    max_simulated_factor: float = 500.0,
    conditioning: Sequence[str] = (),
    batch_size: int = 16,
) -> list[EdgeCaseResult]:
    """Compute per-edge-case move accuracy with quota-controlled coverage.

    Calls :func:`chess_engine.generate_diagnostic_sets` with a flat
    ``per_label`` quota per (colour, label), so every label is
    *guaranteed* coverage (subject to ``max_simulated_factor * total_games``
    being enough simulated games). The v1 parity path per
    docs/JAX_PARITY_SHORTFALLS.md §11; without quota control the rare
    labels (checkmate, stalemate) silently report ``accuracy=0,
    n_positions=0`` on a small random pool.

    The default ``max_simulated_factor=500`` is tuned so the rarest v1
    label — ``stalemate`` — reliably fills its quota at
    ``per_label=10`` (empirically the engine needs ~400 simulated
    games per stalemate game; smaller factors silently truncate the
    rare-label coverage).
    """
    quotas_w, quotas_b = default_diagnostic_quotas(per_label=per_label)
    # `total_games` is the number of distinct accepted games; the engine
    # pads the corpus to satisfy the per-label quotas.
    total_games = per_label * len(EDGE_CASE_LABELS) * 2  # heuristic upper bound
    output = engine.generate_diagnostic_sets(
        quotas_w, quotas_b, total_games, max_ply, seed, max_simulated_factor,
    )
    move_ids, game_lengths, term_codes, per_ply_stats, *_ = output
    move_ids_np = np.asarray(move_ids, dtype=np.int16)
    game_lengths_np = np.asarray(game_lengths, dtype=np.int16)
    bits = np.asarray(per_ply_stats, dtype=np.uint64)
    outcome_tokens = _resolve_outcome_tokens(np.asarray(term_codes), game_lengths_np)
    return _compute_per_bit_accuracy(
        model, move_ids_np, game_lengths_np, bits, outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
    )
