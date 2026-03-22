"""PAWN data pipeline: on-the-fly generation via Rust engine."""

import os
import threading
import time

import numpy as np
import torch
import torch.utils.data

import chess_engine as engine

from pawn.config import (
    WHITE_CHECKMATES,
    BLACK_CHECKMATES,
    STALEMATE,
    DRAW_BY_RULE,
    PLY_LIMIT,
)


_positions_cache: dict = {}



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


def _to_clm_batch(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    term_codes: np.ndarray,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Convert Rust engine output to CLM training tensors.

    Constructs input_ids = [outcome, move_1, ..., move_N, PAD, ...]
    and targets shifted left by 1.

    Args:
        move_ids: (B, engine_max_ply) from generate_random_games
        game_lengths: (B,) actual game lengths (≤ engine_max_ply)
        term_codes: (B,) termination codes
        seq_len: total CLM sequence length (256)
    """
    B = len(game_lengths)
    n_move_slots = seq_len - 1  # 255 slots for moves (position 0 = outcome)
    engine_max_ply = move_ids.shape[1]

    game_lengths_t = torch.from_numpy(game_lengths).long()
    move_ids_t = torch.from_numpy(move_ids).long()  # (B, engine_max_ply)
    outcome_tokens = _map_termination_to_outcome(term_codes, game_lengths)

    # Build input_ids: [outcome, move_0, ..., move_{N-1}, PAD, ...]
    input_ids = torch.zeros(B, seq_len, dtype=torch.long)
    input_ids[:, 0] = outcome_tokens

    # Mask out any non-move tokens from engine output
    cache_key = ("engine", engine_max_ply)
    engine_positions = _positions_cache.get(cache_key)
    if engine_positions is None:
        engine_positions = torch.arange(engine_max_ply).unsqueeze(0)
        _positions_cache[cache_key] = engine_positions
    move_mask = engine_positions < game_lengths_t.unsqueeze(1)
    clean_moves = move_ids_t * move_mask

    # Place moves at positions 1..n_move_slots
    n_to_copy = min(engine_max_ply, n_move_slots)
    input_ids[:, 1 : n_to_copy + 1] = clean_moves[:, :n_to_copy]

    # Cap game_lengths to n_move_slots (handles edge case where engine
    # produces more moves than we have slots)
    capped_lengths = game_lengths_t.clamp(max=n_move_slots)

    # Targets: input shifted left by 1
    targets = torch.zeros(B, seq_len, dtype=torch.long)
    targets[:, :-1] = input_ids[:, 1:]

    # Loss mask: True for positions 0 through capped_lengths[b]
    cache_key_seq = ("seq", seq_len)
    seq_positions = _positions_cache.get(cache_key_seq)
    if seq_positions is None:
        seq_positions = torch.arange(seq_len).unsqueeze(0)
        _positions_cache[cache_key_seq] = seq_positions
    loss_mask = seq_positions <= capped_lengths.unsqueeze(1)

    return {
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
    }


class CLMDataset(torch.utils.data.IterableDataset):
    """Generates CLM training data on-the-fly via the Rust engine.

    Each iteration yields a complete batch. Seeds are deterministic:
    base_seed + step * num_workers + worker_id.
    """

    def __init__(self, batch_size: int, max_ply: int, base_seed: int):
        super().__init__()
        self.batch_size = batch_size
        self.max_ply = max_ply
        self.base_seed = base_seed
        self._start_step = 0
        self._main_pid = os.getpid()

    def set_start_step(self, step: int):
        self._start_step = step

    def __iter__(self):
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
        # Engine generates games of up to max_ply-1 moves, leaving 1 slot
        # for the outcome token in the seq_len=max_ply CLM sequence.
        engine_max_ply = self.max_ply - 1
        while True:
            seed = self.base_seed + step * num_workers + worker_id
            move_ids, game_lengths, term_codes = engine.generate_random_games(
                self.batch_size, engine_max_ply, seed
            )
            yield _to_clm_batch(move_ids, game_lengths, term_codes, self.max_ply)
            step += 1


def create_validation_set(
    n_games: int, max_ply: int, seed: int
) -> dict[str, torch.Tensor]:
    """Generate a fixed validation set.

    Also computes legal move masks for legal move rate evaluation.

    Args:
        max_ply: total CLM sequence length (256). Engine gets max_ply-1.
    """
    engine_max_ply = max_ply - 1
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        n_games, engine_max_ply, seed
    )
    batch = _to_clm_batch(move_ids, game_lengths, term_codes, max_ply)

    # Compute legal move masks for evaluating legal move rate
    legal_grid, _legal_promo = engine.compute_legal_move_masks(move_ids, game_lengths)
    batch["legal_grid"] = torch.from_numpy(legal_grid).long()  # (B, engine_max_ply, 64)
    batch["game_lengths"] = torch.from_numpy(game_lengths).long()

    return batch
