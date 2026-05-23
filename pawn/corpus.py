"""Random-game corpus + the shared CLM packing helper.

A :class:`Corpus` is the v2 data unit consumed by every JAX-side
trainer (pretrain, adapter, specialized_clm). It is a tight bundle of
JAX-friendly arrays — fixed-width ``tokens`` / ``targets`` /
``loss_mask`` / ``attn_mask`` plus a per-game ``outcome_offset``
scalar — built once before training and reused across the
double-buffered streaming path.

Two entry points produce a :class:`Corpus`:

- :func:`generate_corpus` — calls into the Rust engine's
  ``generate_random_games`` to produce a fresh batch of random
  self-play games, then packs them with :func:`_pack_clm`.
- :func:`pack_corpus` — packs **pre-tokenised** game data (move IDs +
  game lengths + outcome tokens) into the same shape. The Lichess
  data path in :mod:`pawn.lichess_data` (S5.C2) uses this.

The outcome-prefixed layout (``prepend_outcome=True``) writes the
game's outcome token at slot 0 and shifts moves right by one;
acceptance criterion 11's five generation diagnostics gate on this
flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import chess_engine as engine
from pawn.config import (
    BLACK_CHECKMATES,
    DRAW_BY_RULE,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    WHITE_CHECKMATES,
)

__all__ = [
    "Corpus",
    "generate_corpus",
    "pack_corpus",
]


@dataclass(frozen=True, slots=True)
class Corpus:
    """A packed game corpus ready for batched JAX consumption.

    Every field has a leading game axis ``N`` and a trailing sequence
    axis ``T`` (= ``seq_len``). Storage dtype is int32 / bool — small
    enough to keep multi-million-game corpora resident-or-streamable
    on commodity hardware.

    **Arrays are stored as host numpy** (not JAX device arrays). For
    realistic Lichess slices the full corpus can be multi-GB; eagerly
    placing it on the default JAX device would exhaust accelerator
    memory before the double-buffered prefetch path can batch it. The
    trainer's prefetch loop slices per-batch and calls ``jnp.asarray``
    at the device-transfer boundary.

    Fields:
        tokens: ``(N, T)`` int32 — input token IDs. PAD where the game
            is shorter than seq_len.
        targets: ``(N, T)`` int32 — tokens shifted left by 1, with PAD
            in the trailing slot. Loss positions consume this.
        attn_mask: ``(N, T)`` bool — True where ``tokens`` is real
            (not PAD). Attention applies a causal × pad mask combo at
            forward time.
        loss_mask: ``(N, T)`` bool — True at positions where the loss
            is supervised. With ``prepend_outcome=False`` the model
            sees one fewer supervised position than the prefixed mode
            (it can't supervise position 0 from move m_2, only the
            outcome-prefixed mode places a target on slot 0).
        outcome_offset: ``(N,)`` int32 — 0 if pure moves, 1 if outcome
            prefixed. Tells the trainer where the first move lives in
            the sequence (slot 0 vs slot 1) and what the diagnostics
            should condition on.
    """

    tokens: NDArray[np.int32]
    targets: NDArray[np.int32]
    attn_mask: NDArray[np.bool_]
    loss_mask: NDArray[np.bool_]
    outcome_offset: NDArray[np.int32]

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_games(self) -> int:
        return len(self)

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])


def _map_termination_to_outcome(
    term_codes: np.ndarray, game_lengths: np.ndarray
) -> np.ndarray:
    """Map engine termination codes to outcome token IDs.

    Engine termination codes:
        0 = Checkmate
        1 = Stalemate
        2 = SeventyFiveMoveRule
        3 = FivefoldRepetition
        4 = InsufficientMaterial
        5 = PlyLimit

    For checkmate, the winning side is inferred from game length —
    odd ``game_length`` (white played the last move) means white
    delivered the mate; even means black.
    """
    term = np.asarray(term_codes, dtype=np.int32)
    gl = np.asarray(game_lengths, dtype=np.int32)

    outcomes = np.full(len(term), PLY_LIMIT, dtype=np.int32)
    is_checkmate = term == 0
    outcomes[is_checkmate & (gl % 2 == 1)] = WHITE_CHECKMATES
    outcomes[is_checkmate & (gl % 2 == 0)] = BLACK_CHECKMATES
    outcomes[term == 1] = STALEMATE
    outcomes[(term == 2) | (term == 3) | (term == 4)] = DRAW_BY_RULE
    # PlyLimit (code 5) keeps the default PLY_LIMIT.
    return outcomes


def _pack_clm(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    seq_len: int,
    prepend_outcome: bool = False,
) -> Corpus:
    """Shared packing helper — turns engine / parquet output into a Corpus.

    ``move_ids`` is ``(N, max_ply)`` int16 with PAD past ``game_length``;
    ``game_lengths`` is ``(N,)`` int (number of moves, never including
    an outcome slot); ``outcome_tokens`` is ``(N,)`` int (the outcome
    token for the game).

    Output sequence width is always ``seq_len``. With
    ``prepend_outcome=True``, slot 0 holds the outcome token and moves
    occupy slots 1..gl+1. With ``prepend_outcome=False`` (default),
    moves start at slot 0 and the outcome token isn't placed in the
    sequence at all.
    """
    move_ids = np.asarray(move_ids, dtype=np.int32)
    game_lengths = np.asarray(game_lengths, dtype=np.int32)
    outcome_tokens = np.asarray(outcome_tokens, dtype=np.int32)

    if move_ids.ndim != 2:
        raise ValueError(f"move_ids must be 2-D (N, max_ply), got shape {move_ids.shape}")
    if game_lengths.ndim != 1 or game_lengths.shape[0] != move_ids.shape[0]:
        raise ValueError(
            f"game_lengths must be (N,) matching move_ids' N={move_ids.shape[0]}, "
            f"got shape {game_lengths.shape}"
        )
    if outcome_tokens.shape != (move_ids.shape[0],):
        raise ValueError(
            f"outcome_tokens must be (N,) matching move_ids' N={move_ids.shape[0]}, "
            f"got shape {outcome_tokens.shape}"
        )
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    n, max_ply = move_ids.shape
    n_move_slots = seq_len - 1 if prepend_outcome else seq_len
    move_start = 1 if prepend_outcome else 0

    # 1. Initial tokens buffer — all PAD.
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    if prepend_outcome:
        tokens[:, 0] = outcome_tokens

    # 2. Clean move IDs (zero past game_length so trailing junk from the
    # engine doesn't leak into the sequence).
    positions = np.arange(max_ply, dtype=np.int32)[None, :]  # (1, max_ply)
    valid_move = positions < game_lengths[:, None]
    clean_moves = np.where(valid_move, move_ids, PAD_TOKEN)

    # 3. Copy the moves into the right slots.
    n_to_copy = min(max_ply, n_move_slots)
    tokens[:, move_start : move_start + n_to_copy] = clean_moves[:, :n_to_copy]

    # 4. attn_mask: True where tokens != PAD.
    attn_mask = tokens != PAD_TOKEN

    # 5. targets: tokens shifted left by 1, with PAD on the last slot.
    targets = np.full_like(tokens, PAD_TOKEN)
    targets[:, :-1] = tokens[:, 1:]

    # 6. loss_mask: positions where we supervise.
    # prepend_outcome=True: positions 0..gl  → gl+1 supervised positions.
    # prepend_outcome=False: positions 0..gl-1 → gl supervised positions.
    capped_lengths = np.minimum(game_lengths, n_move_slots)
    seq_positions = np.arange(seq_len, dtype=np.int32)[None, :]
    if prepend_outcome:
        # `prepend_outcome=True`: positions 0..gl supervised. A zero-
        # length game still gets position 0 supervised (predict the
        # outcome from… nothing, in the all-PAD case — but pack_corpus
        # is a packing helper, not a data validator, so we trust the
        # caller).
        threshold = capped_lengths[:, None]
    else:
        # `prepend_outcome=False`: positions 0..gl-1 supervised. For
        # `gl == 0` (zero-length game), the previous `(gl - 1).clip(0)
        # = 0` mistakenly supervised position 0 even though both the
        # input and target there are PAD. Use `-1` as the no-supervised-
        # positions sentinel so `seq_positions <= threshold` is False
        # everywhere in that row (PR #115 review #4).
        threshold = np.maximum(capped_lengths - 1, -1)[:, None]
    loss_mask = seq_positions <= threshold

    # outcome_offset is one int per game (0 or 1).
    outcome_offset = np.full(n, move_start, dtype=np.int32)

    # Stay on host. The trainer's prefetch loop is responsible for
    # batched device transfer; eagerly materialising a multi-GB Lichess
    # corpus on the JAX device would blow the accelerator's memory
    # budget before training even starts.
    return Corpus(
        tokens=tokens,
        targets=targets,
        attn_mask=attn_mask,
        loss_mask=loss_mask,
        outcome_offset=outcome_offset,
    )


def pack_corpus(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    seq_len: int,
    prepend_outcome: bool = False,
) -> Corpus:
    """Pack pre-tokenised games into a :class:`Corpus`.

    Use this for any source that already has tokenised moves and
    outcome tokens (e.g. Lichess parquet via
    :mod:`pawn.lichess_data`). ``outcome_tokens`` is the per-game
    outcome **token ID** (one of the :data:`pawn.config.WHITE_CHECKMATES`
    / etc. constants).
    """
    return _pack_clm(
        move_ids,
        game_lengths,
        outcome_tokens,
        seq_len=seq_len,
        prepend_outcome=prepend_outcome,
    )


def generate_corpus(
    n_games: int,
    max_ply: int,
    seq_len: int,
    seed: int,
    *,
    prepend_outcome: bool = False,
) -> Corpus:
    """Generate ``n_games`` random self-play games via the Rust engine and
    pack them into a :class:`Corpus`.

    ``max_ply`` is the per-game length cap inside the engine; the
    engine truncates with a ``PlyLimit`` termination code if a game
    runs past it. ``seq_len`` is the output sequence width; the
    packing helper truncates moves past ``seq_len - move_start``.

    ``seed`` is the engine RNG seed for reproducible runs.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games}")
    if max_ply <= 0:
        raise ValueError(f"max_ply must be positive, got {max_ply}")
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        n_games, max_ply, seed
    )
    outcome_tokens = _map_termination_to_outcome(term_codes, game_lengths)
    return _pack_clm(
        move_ids,
        game_lengths,
        outcome_tokens,
        seq_len=seq_len,
        prepend_outcome=prepend_outcome,
    )
