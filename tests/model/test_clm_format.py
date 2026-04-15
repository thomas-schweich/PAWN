"""Tests for CLM sequence format, off-by-one boundaries, and engine/Python consistency.

The engine's ``generate_clm_batch()`` supports two modes:

- **Default** (``prepend_outcome=False``): sequences are pure moves
  ``[m1, m2, ..., mN, PAD, ...]`` with ``max_ply = seq_len``.
- **Outcome-prepended** (``prepend_outcome=True``): sequences are
  ``[outcome, m1, ..., mN, PAD, ...]`` with ``max_ply = seq_len - 1``.

Consistency tests between Rust and Python pass ``prepend_outcome=True`` to
both sides when they want the outcome-prefixed layout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import chess_engine as engine

from pawn.config import (
    BLACK_CHECKMATES,
    DRAW_BY_RULE,
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    WHITE_CHECKMATES,
)
from pawn.data import (
    _map_termination_to_outcome,
    _to_clm_batch,
    pack_clm_sequences,
)


# ---------------------------------------------------------------------------
# Vocabulary consistency
# ---------------------------------------------------------------------------


class TestVocabConsistency:
    @pytest.mark.unit
    def test_vocab_size(self):
        vocab = engine.export_move_vocabulary()
        n_moves = len(vocab["token_to_move"])
        assert n_moves == NUM_ACTIONS  # 1968 actions [0, 1967]
        from pawn.config import CLMConfig, N_TOTAL_OUTCOMES
        # Layout: 1968 actions + 1 PAD + 11 outcomes = 1980
        assert CLMConfig().vocab_size == n_moves + 1 + N_TOTAL_OUTCOMES

    @pytest.mark.unit
    def test_outcome_tokens_not_in_move_vocab(self):
        vocab = engine.export_move_vocabulary()
        for token_id in range(OUTCOME_TOKEN_BASE, PLY_LIMIT + 1):
            assert token_id not in vocab["token_to_move"]

    @pytest.mark.integration
    def test_no_eog_in_raw_move_ids(self):
        move_ids, game_lengths, _tc = engine.generate_random_games(64, 255, seed=42)
        for b in range(64):
            gl = int(game_lengths[b])
            for t in range(gl):
                tok = int(move_ids[b, t])
                assert 0 <= tok <= NUM_ACTIONS - 1


# ---------------------------------------------------------------------------
# CLM batch format (Rust engine)
# ---------------------------------------------------------------------------


class TestRustCLMBatchDefault:
    """Test the default (no outcome prefix) format: [m1, ..., mN, PAD, ...]."""

    @pytest.fixture(scope="class")
    def clm_batch(self):
        return engine.generate_clm_batch(32, 256, seed=42)

    @pytest.mark.integration
    def test_shapes(self, clm_batch):
        input_ids, targets, loss_mask, move_ids, game_lengths, term_codes = clm_batch
        assert input_ids.shape == (32, 256)
        assert targets.shape == (32, 256)
        assert loss_mask.shape == (32, 256)
        assert move_ids.shape == (32, 256)  # max_ply = seq_len (no outcome prefix)
        assert game_lengths.shape == (32,)
        assert term_codes.shape == (32,)

    @pytest.mark.integration
    def test_position_zero_is_move(self, clm_batch):
        input_ids, *_ = clm_batch
        for b in range(input_ids.shape[0]):
            tok = int(input_ids[b, 0])
            assert 0 <= tok <= NUM_ACTIONS - 1

    @pytest.mark.integration
    def test_moves_in_valid_range(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(gl):
                tok = int(input_ids[b, t])
                assert 0 <= tok <= NUM_ACTIONS - 1

    @pytest.mark.integration
    def test_padding_is_pad_token(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(gl, 256):
                assert input_ids[b, t] == PAD_TOKEN

    @pytest.mark.integration
    def test_target_shift_correct(self, clm_batch):
        input_ids, targets, *_ = clm_batch
        B, T = input_ids.shape
        assert np.array_equal(targets[:, :-1], input_ids[:, 1:])
        assert (targets[:, T - 1] == PAD_TOKEN).all()

    @pytest.mark.integration
    def test_target_at_game_end_is_pad(self, clm_batch):
        _, targets, _, _, game_lengths, _ = clm_batch
        for b in range(targets.shape[0]):
            gl = int(game_lengths[b])
            if gl < 256:
                assert targets[b, gl - 1] == PAD_TOKEN

    @pytest.mark.integration
    def test_loss_mask_boundary(self, clm_batch):
        _, _, loss_mask, _, game_lengths, _ = clm_batch
        for b in range(loss_mask.shape[0]):
            gl = int(game_lengths[b])
            # Loss mask is True for positions 0..gl-1 (the gl move positions)
            for t in range(gl):
                assert loss_mask[b, t]
            for t in range(gl, 256):
                assert not loss_mask[b, t]

    @pytest.mark.integration
    def test_raw_move_ids_replayable(self, clm_batch):
        _, _, _, move_ids, game_lengths, _ = clm_batch
        is_valid, _first_illegal = engine.validate_games(move_ids, game_lengths)
        assert all(is_valid)
        grid, _promo = engine.compute_legal_move_masks(move_ids, game_lengths)
        assert grid.shape[0] == 32


class TestRustCLMBatchWithOutcome:
    """Test the outcome-prepended format: [outcome, m1, ..., mN, PAD, ...]."""

    @pytest.fixture(scope="class")
    def clm_batch(self):
        return engine.generate_clm_batch(32, 256, seed=42, prepend_outcome=True)

    @pytest.mark.integration
    def test_shapes(self, clm_batch):
        input_ids, targets, loss_mask, move_ids, game_lengths, term_codes = clm_batch
        assert input_ids.shape == (32, 256)
        assert targets.shape == (32, 256)
        assert loss_mask.shape == (32, 256)
        assert move_ids.shape == (32, 255)  # max_ply = seq_len - 1
        assert game_lengths.shape == (32,)
        assert term_codes.shape == (32,)

    @pytest.mark.integration
    def test_position_zero_is_outcome(self, clm_batch):
        input_ids, *_ = clm_batch
        for b in range(input_ids.shape[0]):
            tok = int(input_ids[b, 0])
            assert OUTCOME_TOKEN_BASE <= tok <= PLY_LIMIT

    @pytest.mark.integration
    def test_moves_in_valid_range(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(1, gl + 1):
                tok = int(input_ids[b, t])
                assert 0 <= tok <= NUM_ACTIONS - 1

    @pytest.mark.integration
    def test_padding_is_pad_token(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(gl + 1, 256):
                assert input_ids[b, t] == PAD_TOKEN

    @pytest.mark.integration
    def test_target_shift_correct(self, clm_batch):
        input_ids, targets, *_ = clm_batch
        B, T = input_ids.shape
        assert np.array_equal(targets[:, :-1], input_ids[:, 1:])
        assert (targets[:, T - 1] == PAD_TOKEN).all()

    @pytest.mark.integration
    def test_target_at_game_end_is_pad(self, clm_batch):
        _, targets, _, _, game_lengths, _ = clm_batch
        for b in range(targets.shape[0]):
            gl = int(game_lengths[b])
            assert targets[b, gl] == PAD_TOKEN

    @pytest.mark.integration
    def test_loss_mask_boundary(self, clm_batch):
        _, _, loss_mask, _, game_lengths, _ = clm_batch
        for b in range(loss_mask.shape[0]):
            gl = int(game_lengths[b])
            for t in range(gl + 1):
                assert loss_mask[b, t]
            for t in range(gl + 1, 256):
                assert not loss_mask[b, t]

    @pytest.mark.integration
    def test_raw_move_ids_replayable(self, clm_batch):
        _, _, _, move_ids, game_lengths, _ = clm_batch
        is_valid, _first_illegal = engine.validate_games(move_ids, game_lengths)
        assert all(is_valid)
        grid, _promo = engine.compute_legal_move_masks(move_ids, game_lengths)
        assert grid.shape[0] == 32


# ---------------------------------------------------------------------------
# Rust CLM matches Python pack_clm_sequences
# ---------------------------------------------------------------------------


class TestRustPythonConsistency:
    @pytest.mark.integration
    def test_rust_clm_matches_python_pack(self):
        seq_len = 256
        seed = 42
        B = 16
        r_input_ids, r_targets, r_loss_mask, r_move_ids, r_gl, r_tc = \
            engine.generate_clm_batch(B, seq_len, seed, prepend_outcome=True)
        py_batch = _to_clm_batch(r_move_ids, r_gl, r_tc, seq_len, prepend_outcome=True)
        assert torch.equal(torch.from_numpy(r_input_ids).long(), py_batch["input_ids"])
        assert torch.equal(torch.from_numpy(r_targets).long(), py_batch["targets"])
        assert torch.equal(torch.from_numpy(r_loss_mask), py_batch["loss_mask"])


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------


class TestBoundaryCases:
    @pytest.mark.integration
    def test_boundary_max_length_game_with_outcome(self):
        """Boundary test with outcome-prepended format."""
        input_ids, targets, loss_mask, _, game_lengths, _ = engine.generate_clm_batch(
            256, 256, seed=123, prepend_outcome=True
        )
        for b in range(256):
            gl = int(game_lengths[b])
            assert OUTCOME_TOKEN_BASE <= input_ids[b, 0] <= PLY_LIMIT
            if gl + 1 < 256:
                assert input_ids[b, gl + 1] == PAD_TOKEN
            assert loss_mask[b, gl]
            if gl + 1 < 256:
                assert not loss_mask[b, gl + 1]

    @pytest.mark.integration
    def test_boundary_max_length_game_default(self):
        """Boundary test with default (no outcome) format."""
        input_ids, targets, loss_mask, _, game_lengths, _ = engine.generate_clm_batch(
            256, 256, seed=123
        )
        for b in range(256):
            gl = int(game_lengths[b])
            assert 0 <= input_ids[b, 0] <= NUM_ACTIONS - 1
            if gl < 256:
                assert input_ids[b, gl] == PAD_TOKEN
            if gl > 0:
                assert loss_mask[b, gl - 1]
            if gl < 256:
                assert not loss_mask[b, gl]

    @pytest.mark.integration
    def test_discard_ply_limit(self):
        input_ids, _, _, _, _, term_codes = engine.generate_clm_batch(
            32, 256, seed=42, discard_ply_limit=True, prepend_outcome=True
        )
        for b in range(32):
            assert term_codes[b] != 5
            assert input_ids[b, 0] != PLY_LIMIT

    @pytest.mark.integration
    def test_determinism(self):
        r1 = engine.generate_clm_batch(8, 256, seed=99)
        r2 = engine.generate_clm_batch(8, 256, seed=99)
        assert np.array_equal(r1[0], r2[0])
        assert np.array_equal(r1[1], r2[1])
        assert np.array_equal(r1[2], r2[2])

    @pytest.mark.integration
    def test_different_seeds_differ(self):
        r1 = engine.generate_clm_batch(8, 256, seed=1)
        r2 = engine.generate_clm_batch(8, 256, seed=2)
        assert not np.array_equal(r1[0], r2[0])


