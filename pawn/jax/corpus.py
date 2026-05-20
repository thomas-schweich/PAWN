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
from numpy.typing import NDArray

from pawn.jax.config import (
    MAX_SEQ_LEN,
    N_OUTCOMES,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
)

# Mirror of ``pawn.config`` — these are the outcome-token offsets the
# engine's term_codes get mapped to. Kept here so this module has no
# dependency on the legacy ``pawn.config``.
#
# Engine term codes (from ``engine/src/types.rs`` ``Termination`` enum):
#   0 = Checkmate    1 = Stalemate     2 = SeventyFiveMoveRule
#   3 = FivefoldRep  4 = InsufficientMaterial   5 = PlyLimit
#
# Zero-based outcome-token offsets (full token IDs are these +
# ``OUTCOME_TOKEN_BASE=1969``; legacy absolute IDs live in ``pawn/config.py``):
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

    Note: ``targets`` and ``loss_mask`` are derived from ``tokens`` and
    ``attn_mask`` respectively; they are stored eagerly here for
    Phase-2 verification simplicity (the TINY corpus is small). A
    future production trainer with a mmap'd N=M-games corpus should
    derive these per-batch from a lean ``(tokens, game_lengths,
    outcome_offset)`` representation — see review-perf finding on
    Phase-2 chunk 1 for the rationale.
    """

    tokens: NDArray[np.int32]            # [N, T]
    attn_mask: NDArray[np.bool_]         # [N, T]
    targets: NDArray[np.int32]           # [N, T]
    loss_mask: NDArray[np.bool_]         # [N, T]
    outcome_offset: NDArray[np.uint8]    # [N]  — 0..N_OUTCOMES-1

    @property
    def n_games(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])


def _map_term_to_outcome_offset(
    term_codes: NDArray[np.int32], game_lengths: NDArray[np.int32]
) -> NDArray[np.uint8]:
    """Map ``(term_code, game_length)`` → outcome-offset table.

    Returns a ``uint8[B]`` array in ``[0, N_OUTCOMES)``. Mirrors the
    legacy ``_map_termination_to_outcome`` but stays in NumPy.
    """
    if N_OUTCOMES < 5:
        raise AssertionError(
            f"N_OUTCOMES={N_OUTCOMES} but the outcome-offset table assumes 5 "
            "slots; update the offset constants in pawn/jax/corpus.py"
        )
    # ``np.empty`` with the explicit shape communicates intent more
    # clearly than ``np.full_like`` (whose source array is used only as
    # a shape template here — the value and dtype are overridden).
    out = np.empty(term_codes.shape, dtype=np.uint8)
    out.fill(_OFFSET_PLY_LIMIT)
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
    move_ids: NDArray[np.int32],
    game_lengths: NDArray[np.int32],
    seq_len: int,
) -> tuple[
    NDArray[np.int32], NDArray[np.bool_], NDArray[np.int32], NDArray[np.bool_]
]:
    """Pack raw engine output into trainer tensors.

    Returns ``(tokens, attn_mask, targets, loss_mask)`` matching the
    canonical no-outcome-prepend mode of ``pawn.data.pack_clm_sequences``.

    * ``tokens``: ``[m_1, m_2, …, m_N, PAD, …]`` left-aligned to ``seq_len``.
    * ``attn_mask``: real-move positions only.
    * ``targets``: ``tokens`` shifted left by one; PAD at the last slot.
    * ``loss_mask``: positions ``0..game_length - 1`` (the last
      supervised position is ``game_length - 1``; its target is PAD —
      the game-over signal). Positions ``>= game_length`` are excluded
      from the loss.
    """
    B = int(game_lengths.shape[0])
    max_ply = int(move_ids.shape[1])
    if seq_len < max_ply:
        # Caller-bug guard; ``generate_corpus`` early-validates this
        # before any engine generation so the cost of the misuse stays
        # at zero. Inside _pack_clm this also catches direct callers.
        raise ValueError(
            f"seq_len={seq_len} < max_ply={max_ply}; the corpus cannot "
            "hold all engine-produced moves"
        )

    # Mask out positions beyond per-game length (the engine pads with
    # garbage past game_lengths[i]). ``move_ids`` is already int32 —
    # callers cast at the engine boundary so this loop avoids a
    # second full-corpus copy.
    pos = np.arange(max_ply, dtype=np.int32)[None, :]  # (1, max_ply)
    real = pos < game_lengths[:, None]

    tokens = np.full((B, seq_len), PAD_TOKEN, dtype=np.int32)
    # Bulk-copy the int32 moves in (includes engine garbage past
    # game_lengths), then scatter PAD into the garbage positions
    # in-place. Avoids the temporary ``np.where`` output buffer.
    tokens[:, :max_ply] = move_ids
    tokens_head = tokens[:, :max_ply]
    tokens_head[~real] = PAD_TOKEN

    capped_lengths = np.minimum(game_lengths, seq_len)
    seq_pos = np.arange(seq_len, dtype=np.int32)[None, :]
    attn_mask = seq_pos < capped_lengths[:, None]

    # Next-token target.
    targets = np.full((B, seq_len), PAD_TOKEN, dtype=np.int32)
    targets[:, :-1] = tokens[:, 1:]

    # Loss is over the supervised next-token transitions: positions
    # 0..game_length-1 (``game_length`` items). The position at
    # ``game_length`` would predict PAD with no model evidence (no
    # input at that position) — excluded.
    loss_threshold = np.maximum(capped_lengths - 1, 0)
    loss_mask = seq_pos <= loss_threshold[:, None]
    # Edge case: a zero-move game produces ``loss_threshold = 0`` and
    # the threshold check above would still mark position 0 True;
    # AND'ing with ``attn_mask`` (all False for that game) zeros it.
    loss_mask &= attn_mask
    return tokens, attn_mask, targets, loss_mask


def generate_corpus(
    n_games: int,
    *,
    max_ply: int = MAX_SEQ_LEN,
    seq_len: int | None = None,
    seed: int,
) -> Corpus:
    """Generate a tokenized corpus from the Rust random-games engine.

    Args:
        n_games: number of games to generate.
        max_ply: per-game ply cap passed to the engine. Defaults to
            ``MAX_SEQ_LEN`` (512) so the engine produces full
            model-context games unless the caller explicitly overrides.
        seq_len: width of the packed tensors. Defaults to ``max_ply``;
            must be ``>= max_ply``.
        seed: RNG seed for the engine. Same seed → same corpus (the
            engine derives per-game seeds from a ChaCha8 CSPRNG seeded
            by ``seed``; ``engine/src/random.rs`` ``derive_game_seeds``).

    Returns:
        ``Corpus`` ready to feed the trainer.
    """
    if seq_len is None:
        seq_len = max_ply
    if seq_len < max_ply:
        # Validate before the engine generates anything — for large
        # ``n_games`` the engine spend is non-trivial and bailing
        # afterwards wastes it. Same message as the inner _pack_clm
        # guard so the diagnostic is consistent regardless of which
        # check fires.
        raise ValueError(
            f"seq_len={seq_len} < max_ply={max_ply}; the corpus cannot "
            "hold all engine-produced moves"
        )
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        n_games=n_games, max_ply=max_ply, seed=seed
    )
    # Cast at the boundary so the rest of the pipeline operates on
    # consistent int32. Engine returns int16 for moves/lengths,
    # uint8 for term codes (see engine/src/lib.rs).
    move_ids_arr: NDArray[np.int32] = np.asarray(move_ids, dtype=np.int32)
    game_lengths_arr: NDArray[np.int32] = np.asarray(
        game_lengths, dtype=np.int32
    )
    term_codes_arr: NDArray[np.int32] = np.asarray(term_codes, dtype=np.int32)
    tokens, attn_mask, targets, loss_mask = _pack_clm(
        move_ids_arr, game_lengths_arr, seq_len
    )
    outcome_offset = _map_term_to_outcome_offset(term_codes_arr, game_lengths_arr)
    return Corpus(
        tokens=tokens,
        attn_mask=attn_mask,
        targets=targets,
        loss_mask=loss_mask,
        outcome_offset=outcome_offset,
    )


def outcome_tokens(corpus: Corpus) -> NDArray[np.int32]:
    """Convert per-game outcome offsets to full ``int32`` outcome token
    IDs (1969..1979). Useful for callers that want to optionally
    prepend the outcome at sequence position 0."""
    return corpus.outcome_offset.astype(np.int32) + OUTCOME_TOKEN_BASE
