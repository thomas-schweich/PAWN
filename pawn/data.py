"""PAWN data pipeline: on-the-fly generation via Rust engine."""

import os
import threading
import time
from collections.abc import Iterator

import numpy as np
import torch
import torch.utils.data

import chess_engine as engine

from pawn.config import (
    PAD_TOKEN,
    WHITE_CHECKMATES,
    BLACK_CHECKMATES,
    STALEMATE,
    DRAW_BY_RULE,
    PLY_LIMIT,
)


_positions_cache: dict[tuple[str, int], torch.Tensor] = {}


def _map_termination_to_outcome(
    term_codes: np.ndarray, game_lengths: np.ndarray
) -> torch.Tensor:
    """Map engine termination codes to outcome token IDs.

    Engine codes: 0=Checkmate, 1=Stalemate, 2=SeventyFiveMoveRule,
                  3=FivefoldRepetition, 4=InsufficientMaterial, 5=PlyLimit

    For checkmate, who checkmated is determined by game length:
    - Odd game_length (last ply index even) -> white delivered checkmate
    - Even game_length (last ply index odd) -> black delivered checkmate
    """
    term = torch.from_numpy(term_codes).long()
    gl = torch.from_numpy(game_lengths).long()

    outcomes = torch.full((len(term_codes),), PLY_LIMIT, dtype=torch.long)

    is_checkmate = term == 0
    outcomes[is_checkmate & (gl % 2 == 1)] = WHITE_CHECKMATES
    outcomes[is_checkmate & (gl % 2 == 0)] = BLACK_CHECKMATES
    outcomes[term == 1] = STALEMATE
    outcomes[(term == 2) | (term == 3) | (term == 4)] = DRAW_BY_RULE
    # PlyLimit (code 5) is the default

    return outcomes


def pack_clm_sequences(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    outcome_tokens: torch.Tensor,
    seq_len: int,
    prepend_outcome: bool = False,
) -> dict[str, torch.Tensor]:
    """Pack move arrays into CLM training tensors.

    The output always has width ``seq_len`` regardless of mode; the
    difference is whether slot 0 holds the outcome or the first move.

    - ``prepend_outcome=True``: input_ids = [outcome, m_1, ..., m_N, PAD, ...],
      loss_mask covers positions ``0..game_length`` inclusive (``game_length + 1``
      positions — one prediction per move plus the game-over prediction).
    - ``prepend_outcome=False`` (default): input_ids = [m_1, m_2, ..., m_N, PAD, ...],
      loss_mask covers positions ``0..game_length - 1`` (``game_length`` positions —
      one fewer supervised position than the prefixed mode; this is the
      documented cost of running without the outcome prefix).

    The ``outcome_tokens`` argument is only read when ``prepend_outcome=True``;
    callers may pass any tensor of the right shape when they don't care.

    Args:
        move_ids: (B, max_ply) raw move token IDs
        game_lengths: (B,) actual game lengths (move counts, never including
            the outcome slot)
        outcome_tokens: (B,) pre-computed outcome token IDs
        seq_len: total CLM sequence length (the fixed tensor width)
        prepend_outcome: if True, write outcome at slot 0 and shift moves right
    """
    B = len(game_lengths)
    n_move_slots = seq_len - 1 if prepend_outcome else seq_len
    max_ply = move_ids.shape[1]

    game_lengths_t = torch.from_numpy(game_lengths).long()
    move_ids_t = torch.from_numpy(move_ids).long()  # (B, max_ply)

    input_ids = torch.full((B, seq_len), PAD_TOKEN, dtype=torch.long)
    move_start = 0
    if prepend_outcome:
        input_ids[:, 0] = outcome_tokens
        move_start = 1

    # Mask out any non-move tokens from engine output
    cache_key = ("engine", max_ply)
    engine_positions = _positions_cache.get(cache_key)
    if engine_positions is None:
        engine_positions = torch.arange(max_ply).unsqueeze(0)
        _positions_cache[cache_key] = engine_positions
    move_mask = engine_positions < game_lengths_t.unsqueeze(1)
    clean_moves = move_ids_t.where(move_mask, torch.tensor(PAD_TOKEN, dtype=torch.long))

    n_to_copy = min(max_ply, n_move_slots)
    input_ids[:, move_start : move_start + n_to_copy] = clean_moves[:, :n_to_copy]

    # Cap game_lengths to n_move_slots (handles edge case where engine
    # produces more moves than we have slots)
    capped_lengths = game_lengths_t.clamp(max=n_move_slots)

    # Targets: input shifted left by 1
    targets = torch.full((B, seq_len), PAD_TOKEN, dtype=torch.long)
    targets[:, :-1] = input_ids[:, 1:]

    # Loss mask:
    #   prepend_outcome=True  → positions 0..game_length inclusive (gl+1 True)
    #     position 0 predicts move_1 from outcome; position game_length predicts PAD
    #     (the game-over signal).
    #   prepend_outcome=False → positions 0..game_length-1 (gl True)
    #     position 0 predicts move_2 from move_1; position game_length-1 predicts PAD.
    cache_key_seq = ("seq", seq_len)
    seq_positions = _positions_cache.get(cache_key_seq)
    if seq_positions is None:
        seq_positions = torch.arange(seq_len).unsqueeze(0)
        _positions_cache[cache_key_seq] = seq_positions
    loss_threshold = capped_lengths if prepend_outcome else (capped_lengths - 1).clamp(min=0)
    loss_mask = seq_positions <= loss_threshold.unsqueeze(1)

    return {
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
    }


def _to_clm_batch(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    term_codes: np.ndarray,
    seq_len: int,
    prepend_outcome: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert Rust engine output to CLM training tensors.

    Convenience wrapper: computes outcome tokens from termination codes,
    then delegates to :func:`pack_clm_sequences`. ``prepend_outcome`` is
    forwarded — default is pure moves.
    """
    outcome_tokens = _map_termination_to_outcome(term_codes, game_lengths)
    return pack_clm_sequences(
        move_ids, game_lengths, outcome_tokens, seq_len,
        prepend_outcome=prepend_outcome,
    )


class CLMDataset(torch.utils.data.IterableDataset):
    """Generates CLM training data on-the-fly via the Rust engine.

    Each iteration yields a complete batch. Seeds are deterministic:
    base_seed + step * num_workers + worker_id.
    """

    def __init__(self, batch_size: int, max_ply: int, base_seed: int,
                 discard_ply_limit: bool = False,
                 mate_boost: float = 0.0, prepend_outcome: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.max_ply = max_ply
        self.base_seed = base_seed
        self.discard_ply_limit = discard_ply_limit
        self.mate_boost = mate_boost
        self.prepend_outcome = prepend_outcome
        self._start_step = 0
        self._main_pid = os.getpid()

    def set_start_step(self, step: int) -> None:
        self._start_step = step

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if worker_info is not None:
            main_pid = self._main_pid

            def _watchdog():
                while True:
                    time.sleep(2)
                    try:
                        os.kill(main_pid, 0)
                    except OSError:
                        os._exit(1)

            t = threading.Thread(target=_watchdog, daemon=True)
            t.start()

        step = self._start_step
        while True:
            seed = self.base_seed + step * num_workers + worker_id
            input_ids, targets, loss_mask, _move_ids, _gl, _tc = \
                engine.generate_clm_batch(
                    self.batch_size, self.max_ply, seed,
                    discard_ply_limit=self.discard_ply_limit,
                    mate_boost=self.mate_boost,
                    prepend_outcome=self.prepend_outcome,
                )
            batch = {
                "input_ids": torch.from_numpy(input_ids).long(),
                "targets": torch.from_numpy(targets).long(),
                "loss_mask": torch.from_numpy(loss_mask),
            }
            yield batch
            step += 1


def shift_legal_mask(mask: np.ndarray) -> np.ndarray:
    """Shift a legal move mask forward by one ply to align with CLM targets.

    The engine's legal masks are indexed by position *before* each move:
    mask[ply] = legal moves at position ply.  But CLM targets[ply] is the
    *next* move (= move_ids[ply+1]), so we need legal moves at position
    ply+1.  This rolls the mask by -1 along the ply axis and zeros the
    last entry (no next move at the final ply).

    Works for any mask shape (B, T, ...).
    """
    shifted = np.roll(mask, -1, axis=1)
    shifted[:, -1] = 0
    return shifted


def align_legal_to_preds(
    raw: np.ndarray, prepend_outcome: bool
) -> np.ndarray:
    """Align a raw per-game-ply legal mask with CLM predictions.

    The engine returns masks indexed by game-ply: ``raw[p]`` = legal moves
    at game-ply ``p`` (i.e. after ``p`` moves have been played).  CLM
    predictions are indexed by token position in the packed sequence:

      * ``prepend_outcome=False`` — ``preds[t]`` predicts ``targets[t]``
        = ``input_ids[t+1]``, which is game-ply ``t+1``.  Shift the raw
        mask by -1 so ``out[t] = raw[t+1]``.
      * ``prepend_outcome=True`` — position 0 is the outcome token, so
        ``preds[t]`` predicts game-ply ``t``.  We want ``out[t] = raw[t]``,
        plus one zero row appended at the end to keep the sequence length
        consistent with the token tensors.

    In both cases the returned tensor has shape ``(B, seq_len, ...)`` so
    it can be indexed in lockstep with ``preds`` / ``loss_mask``.
    """
    if prepend_outcome:
        pad = np.zeros_like(raw[:, :1])
        return np.concatenate([raw, pad], axis=1)
    return shift_legal_mask(raw)


def create_validation_set(
    n_games: int, max_ply: int, seed: int,
    discard_ply_limit: bool = False,
    mate_boost: float = 0.0,
    prepend_outcome: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate a fixed validation set.

    Also computes legal move masks for legal move rate evaluation.

    Args:
        max_ply: total CLM sequence length (512).
    """
    input_ids, targets, loss_mask, move_ids, game_lengths, _tc = \
        engine.generate_clm_batch(
            n_games, max_ply, seed, discard_ply_limit=discard_ply_limit,
            mate_boost=mate_boost,
            prepend_outcome=prepend_outcome,
        )
    batch = {
        "input_ids": torch.from_numpy(input_ids).long(),
        "targets": torch.from_numpy(targets).long(),
        "loss_mask": torch.from_numpy(loss_mask),
    }

    # Raw per-game-ply move ids (without the outcome prefix) — eval uses
    # these to recompute full-vocab legal token masks for game-completion
    # scoring, where ``input_ids`` can't be reused directly because it may
    # start with the outcome token.
    batch["move_ids"] = torch.from_numpy(move_ids).long()
    batch["prepend_outcome"] = torch.tensor(prepend_outcome, dtype=torch.bool)

    # Compute legal move masks for evaluating legal move rate.
    # ``align_legal_to_preds`` handles both modes: shift-by-one for pure
    # sequences (preds[t] predicts game-ply t+1), or append a zero row for
    # outcome-prefixed sequences (preds[t] predicts game-ply t).
    legal_grid, _legal_promo = engine.compute_legal_move_masks(move_ids, game_lengths)
    batch["legal_grid"] = torch.from_numpy(
        align_legal_to_preds(legal_grid, prepend_outcome)
    ).long()
    batch["game_lengths"] = torch.from_numpy(game_lengths).long()

    return batch
