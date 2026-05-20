"""Pretraining corpus generator for the JAX trainer.

The Rust engine produces tokenized random games at ~43K games/sec. This
module wraps the engine's output into the shape the trainer expects:

* ``tokens``: ``int32[B, T]`` — move tokens left-aligned, PAD-filled.
* ``attn_mask``: ``bool[B, T]`` — True at real-move positions.
* ``targets``: ``int32[B, T]`` — input shifted left by one; PAD at the
  terminal slot.
* ``loss_mask``: ``bool[B, T]`` — True where ``targets`` should
  contribute to the loss.

The ``prepend_outcome=False`` mode of the legacy ``data.pack_clm_sequences``
is replicated here in NumPy (no torch dependency). Per ``docs/jax-migration.md``
§4.1 the corpus is pre-generated offline; ``generate_corpus`` takes a
seed + count and returns a NumPy array tuple ready to mmap or stage to the
device.

This module does **not** prepend the outcome token at position 0. The
outcome is returned separately so callers can choose to prepend it later
(useful for outcome-conditional eval probes). For Phase-2 verification
the simpler no-prepend path is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess_engine as engine
import numpy as np

from pawn.jax.config import (
    N_OUTCOMES,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
)

# Mirror of ``pawn.config`` — these are the outcome-token offsets the
# engine's term_codes get mapped to. Kept here so this module has no
# dependency on the legacy ``pawn.config``.
#
# Engine term codes (from ``engine/src/lib.rs``):
#   0 = Checkmate    1 = Stalemate     2 = SeventyFiveMoveRule
#   3 = FivefoldRep  4 = InsufficientMaterial   5 = PlyLimit
#
# Outcome token offsets (from ``pawn/config.py``):
#   0  WHITE_CHECKMATES
#   1  BLACK_CHECKMATES
#   2  STALEMATE
#   3  DRAW_BY_RULE    (engine codes 2, 3, 4)
#   4  PLY_LIMIT       (engine code 5)
_OFFSET_WHITE_CHECKMATES = 0
_OFFSET_BLACK_CHECKMATES = 1
_OFFSET_STALEMATE = 2
_OFFSET_DRAW_BY_RULE = 3
_OFFSET_PLY_LIMIT = 4


@dataclass(frozen=True)
class Corpus:
    """A packed corpus ready to feed the trainer.

    Shapes are uniform across all fields except ``outcome_offset``
    which is per-game. All arrays live in host RAM as plain NumPy
    arrays — the trainer stages chunks to the device.
    """

    tokens: np.ndarray            # int32[N, T]
    attn_mask: np.ndarray         # bool[N, T]
    targets: np.ndarray           # int32[N, T]
    loss_mask: np.ndarray         # bool[N, T]
    outcome_offset: np.ndarray    # uint8[N]  — 0..N_OUTCOMES-1

    @property
    def n_games(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])


def _map_term_to_outcome_offset(
    term_codes: np.ndarray, game_lengths: np.ndarray
) -> np.ndarray:
    """Map ``(term_code, game_length)`` → outcome-offset table.

    Returns a ``uint8[B]`` array in ``[0, N_OUTCOMES)``. Mirrors the
    legacy ``_map_termination_to_outcome`` but stays in NumPy.
    """
    if N_OUTCOMES < 5:
        raise AssertionError(
            f"N_OUTCOMES={N_OUTCOMES} but the outcome-offset table assumes 5 "
            "slots; update the offset constants in pawn/jax/corpus.py"
        )
    out = np.full_like(term_codes, _OFFSET_PLY_LIMIT, dtype=np.uint8)
    is_checkmate = term_codes == 0
    # Engine convention: an odd game length means white delivered mate
    # (white moves first, so game ends with black trying to respond);
    # an even length means black delivered mate.
    out[is_checkmate & (game_lengths % 2 == 1)] = _OFFSET_WHITE_CHECKMATES
    out[is_checkmate & (game_lengths % 2 == 0)] = _OFFSET_BLACK_CHECKMATES
    out[term_codes == 1] = _OFFSET_STALEMATE
    out[(term_codes == 2) | (term_codes == 3) | (term_codes == 4)] = (
        _OFFSET_DRAW_BY_RULE
    )
    return out


def _pack_clm(
    move_ids: np.ndarray, game_lengths: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack raw engine output into trainer tensors.

    Returns ``(tokens, attn_mask, targets, loss_mask)`` matching the
    canonical no-outcome-prepend mode of ``pawn.data.pack_clm_sequences``.

    * ``tokens``: ``[m_1, m_2, …, m_N, PAD, …]`` left-aligned to ``seq_len``.
    * ``attn_mask``: real-move positions only.
    * ``targets``: ``tokens`` shifted left by one; PAD at the last slot.
    * ``loss_mask``: positions ``0..game_length - 1`` (predict next move
      from current; the last predicted move is the terminal PAD,
      which is masked out).
    """
    B = int(game_lengths.shape[0])
    max_ply = int(move_ids.shape[1])
    if seq_len < max_ply:
        # Trim is a deliberate caller-bug guard; the corpus generator
        # always sets seq_len >= max_ply.
        raise ValueError(
            f"seq_len={seq_len} < max_ply={max_ply}; the corpus cannot "
            "hold all engine-produced moves"
        )

    # Mask out positions beyond per-game length (the engine pads with
    # garbage past game_lengths[i]).
    pos = np.arange(max_ply, dtype=np.int32)[None, :]  # (1, max_ply)
    real = pos < game_lengths.astype(np.int32)[:, None]
    move_ids_clean = np.where(
        real, move_ids.astype(np.int32), np.int32(PAD_TOKEN)
    )

    tokens = np.full((B, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[:, :max_ply] = move_ids_clean

    attn_mask = np.zeros((B, seq_len), dtype=np.bool_)
    capped_lengths = np.minimum(game_lengths, seq_len).astype(np.int32)
    seq_pos = np.arange(seq_len, dtype=np.int32)[None, :]
    attn_mask = seq_pos < capped_lengths[:, None]

    # Next-token target.
    targets = np.full((B, seq_len), PAD_TOKEN, dtype=np.int32)
    targets[:, :-1] = tokens[:, 1:]

    # Loss is over the supervised next-token transitions:
    # positions 0..game_length-1 (game_length items). The position at
    # game_length predicts PAD — excluded.
    loss_threshold = np.maximum(capped_lengths - 1, 0)
    loss_mask = seq_pos <= loss_threshold[:, None]
    # Edge case: a zero-move game produces ``loss_threshold = 0`` and
    # the above would still mark position 0 — zero it.
    loss_mask &= attn_mask
    return tokens, attn_mask, targets, loss_mask


def generate_corpus(
    n_games: int, *, max_ply: int = 256, seq_len: int | None = None, seed: int
) -> Corpus:
    """Generate a tokenized corpus from the Rust random-games engine.

    Args:
        n_games: number of games to generate.
        max_ply: per-game ply cap passed to the engine.
        seq_len: width of the packed tensors. Defaults to ``max_ply``;
            must be ``>= max_ply``.
        seed: RNG seed for the engine. Same seed → same corpus
            (the engine seeds each game from ``hash(seed, game_index)``,
            so two callers with the same seed get bit-identical output).

    Returns:
        ``Corpus`` ready to feed the trainer.
    """
    if seq_len is None:
        seq_len = max_ply
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        n_games=n_games, max_ply=max_ply, seed=seed
    )
    game_lengths_arr = np.asarray(game_lengths, dtype=np.int32)
    term_codes_arr = np.asarray(term_codes, dtype=np.int32)
    tokens, attn_mask, targets, loss_mask = _pack_clm(
        np.asarray(move_ids), game_lengths_arr, seq_len
    )
    outcome_offset = _map_term_to_outcome_offset(term_codes_arr, game_lengths_arr)
    return Corpus(
        tokens=tokens,
        attn_mask=attn_mask,
        targets=targets,
        loss_mask=loss_mask,
        outcome_offset=outcome_offset,
    )


def outcome_tokens(corpus: Corpus) -> np.ndarray:
    """Convert per-game outcome offsets to full ``int32`` outcome token
    IDs (1969..1979). Useful for callers that want to optionally
    prepend the outcome at sequence position 0."""
    return (
        corpus.outcome_offset.astype(np.int32) + np.int32(OUTCOME_TOKEN_BASE)
    )
