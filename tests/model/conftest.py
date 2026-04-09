"""Partition D (Model + Data) local fixtures.

Owned by Partition D. Synthetic sequences, legal-mask factories, KV-cache
harnesses, small-batch engine-output fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def tiny_clm_batch_np(rust_seed):
    """Tiny deterministic engine output (numpy arrays) — cheap reuse.

    Returns (move_ids, game_lengths, term_codes) straight from
    generate_random_games. Session-scoped because it's read-only.
    """
    import chess_engine as engine

    move_ids, game_lengths, term_codes = engine.generate_random_games(
        8, 63, seed=rust_seed
    )
    return move_ids, game_lengths, term_codes


@pytest.fixture
def synth_input_ids():
    """Deterministic synthetic token IDs for fast shape tests.

    Returns a helper factory (B, T, vocab_size) -> LongTensor.
    """

    def _make(B: int, T: int, vocab_size: int, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(1, vocab_size, (B, T), generator=g, dtype=torch.long)

    return _make


@pytest.fixture
def full_mask():
    """All-True attention mask factory."""

    def _make(B: int, T: int) -> torch.Tensor:
        return torch.ones(B, T, dtype=torch.bool)

    return _make


@pytest.fixture
def small_clm_sequences():
    """Build a small batch of CLM sequences by hand for unit tests.

    Deterministic, no engine dependency. 2 games, seq_len=8.
    Game 0: len=3 moves, Game 1: len=5 moves.
    """
    from pawn.config import WHITE_CHECKMATES, BLACK_CHECKMATES, PAD_TOKEN

    seq_len = 8
    input_ids = torch.full((2, seq_len), PAD_TOKEN, dtype=torch.long)
    # Game 0: [WHITE_CHECKMATES, m1, m2, m3, PAD, PAD, PAD, PAD]
    input_ids[0, 0] = WHITE_CHECKMATES
    input_ids[0, 1:4] = torch.tensor([10, 20, 30])
    # Game 1: [BLACK_CHECKMATES, m1, m2, m3, m4, m5, PAD, PAD]
    input_ids[1, 0] = BLACK_CHECKMATES
    input_ids[1, 1:6] = torch.tensor([11, 22, 33, 44, 55])

    targets = torch.full((2, seq_len), PAD_TOKEN, dtype=torch.long)
    targets[:, :-1] = input_ids[:, 1:]

    game_lengths = torch.tensor([3, 5], dtype=torch.long)
    positions = torch.arange(seq_len).unsqueeze(0)
    loss_mask = positions <= game_lengths.unsqueeze(1)

    return {
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
        "game_lengths": game_lengths,
    }
