"""Tests for CLM sequence format, off-by-one boundaries, and engine/Python consistency."""

import numpy as np
import torch
import pytest

import chess_engine as engine

from pawn.config import (
    PAD_TOKEN,
    OUTCOME_TOKEN_BASE,
    WHITE_CHECKMATES,
    BLACK_CHECKMATES,
    STALEMATE,
    DRAW_BY_RULE,
    PLY_LIMIT,
)
from pawn.data import (
    _map_termination_to_outcome,
    pack_clm_sequences,
    _to_clm_batch,
)


# ---------------------------------------------------------------------------
# Vocabulary consistency
# ---------------------------------------------------------------------------


class TestVocabConsistency:
    def test_vocab_size(self):
        """Engine move vocab + PAD + outcomes = model vocab_size."""
        vocab = engine.export_move_vocabulary()
        n_moves = len(vocab["token_to_move"])
        # 4096 grid + 176 promotions = 4272 move tokens
        assert n_moves == 4272
        # 1 PAD + 4272 moves + 5 outcomes = 4278
        from pawn.config import CLMConfig
        assert CLMConfig().vocab_size == 1 + n_moves + 5

    def test_outcome_tokens_not_in_move_vocab(self):
        """Outcome tokens (4273-4277) must not appear in the move vocabulary."""
        vocab = engine.export_move_vocabulary()
        for token_id in range(OUTCOME_TOKEN_BASE, PLY_LIMIT + 1):
            assert token_id not in vocab["token_to_move"], \
                f"Outcome token {token_id} should not be in move vocab"

    def test_no_eog_in_raw_move_ids(self):
        """Raw move_ids from generate_random_games should not contain
        tokens >= OUTCOME_BASE (no EOG or outcome tokens in move data)."""
        move_ids, game_lengths, _tc = engine.generate_random_games(64, 255, seed=42)
        for b in range(64):
            gl = int(game_lengths[b])
            # Moves should be in range 1-4272
            for t in range(gl):
                tok = int(move_ids[b, t])
                assert 1 <= tok <= 4272, \
                    f"Game {b}, ply {t}: expected move token, got {tok}"
            # Position game_length should be PAD (0), not EOG
            if gl < 255:
                assert move_ids[b, gl] == PAD_TOKEN, \
                    f"Game {b}: position {gl} should be PAD, got {move_ids[b, gl]}"


# ---------------------------------------------------------------------------
# CLM batch format (Rust engine)
# ---------------------------------------------------------------------------


class TestRustCLMBatch:
    @pytest.fixture
    def clm_batch(self):
        return engine.generate_clm_batch(32, 256, seed=42)

    def test_shapes(self, clm_batch):
        input_ids, targets, loss_mask, move_ids, game_lengths, term_codes = clm_batch
        assert input_ids.shape == (32, 256)
        assert targets.shape == (32, 256)
        assert loss_mask.shape == (32, 256)
        assert move_ids.shape == (32, 255)
        assert game_lengths.shape == (32,)
        assert term_codes.shape == (32,)

    def test_position_zero_is_outcome(self, clm_batch):
        input_ids, *_ = clm_batch
        for b in range(input_ids.shape[0]):
            tok = int(input_ids[b, 0])
            assert OUTCOME_TOKEN_BASE <= tok <= PLY_LIMIT, \
                f"Game {b}: position 0 should be outcome token, got {tok}"

    def test_moves_in_valid_range(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(1, gl + 1):
                tok = int(input_ids[b, t])
                assert 1 <= tok <= 4272, \
                    f"Game {b}, position {t}: expected move token, got {tok}"

    def test_padding_is_zero(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(gl + 1, 256):
                assert input_ids[b, t] == 0, \
                    f"Game {b}, position {t}: expected PAD, got {input_ids[b, t]}"

    def test_target_shift_correct(self, clm_batch):
        input_ids, targets, *_ = clm_batch
        B, T = input_ids.shape
        for b in range(B):
            for t in range(T - 1):
                assert targets[b, t] == input_ids[b, t + 1], \
                    f"Game {b}: targets[{t}]={targets[b, t]} != input_ids[{t+1}]={input_ids[b, t+1]}"
            assert targets[b, T - 1] == 0, "Last target should be PAD"

    def test_target_at_game_end_is_pad(self, clm_batch):
        _, targets, _, _, game_lengths, _ = clm_batch
        for b in range(targets.shape[0]):
            gl = int(game_lengths[b])
            assert targets[b, gl] == 0, \
                f"Game {b}: target at game_length={gl} should be PAD, got {targets[b, gl]}"

    def test_loss_mask_boundary(self, clm_batch):
        _, _, loss_mask, _, game_lengths, _ = clm_batch
        for b in range(loss_mask.shape[0]):
            gl = int(game_lengths[b])
            # True for positions 0..=gl
            for t in range(gl + 1):
                assert loss_mask[b, t], \
                    f"Game {b}: loss_mask[{t}] should be True (gl={gl})"
            # False after gl
            for t in range(gl + 1, 256):
                assert not loss_mask[b, t], \
                    f"Game {b}: loss_mask[{t}] should be False (gl={gl})"

    def test_raw_move_ids_replayable(self, clm_batch):
        """Raw move_ids from generate_clm_batch should work with replay functions."""
        _, _, _, move_ids, game_lengths, _ = clm_batch
        # validate_games should confirm all games are legal
        is_valid, first_illegal = engine.validate_games(move_ids, game_lengths)
        assert all(is_valid), "All generated games should be valid"
        # compute_legal_move_masks should not error
        grid, promo = engine.compute_legal_move_masks(move_ids, game_lengths)
        assert grid.shape[0] == 32


# ---------------------------------------------------------------------------
# Rust CLM matches Python pack_clm_sequences
# ---------------------------------------------------------------------------


class TestRustPythonConsistency:
    def test_rust_clm_matches_python_pack(self):
        """Rust generate_clm_batch should produce identical output to
        Python _to_clm_batch with the same seed."""
        seq_len = 256
        seed = 42
        B = 16

        # Rust path
        r_input_ids, r_targets, r_loss_mask, r_move_ids, r_gl, r_tc = \
            engine.generate_clm_batch(B, seq_len, seed)

        # Python path: generate raw + pack
        py_batch = _to_clm_batch(r_move_ids, r_gl, r_tc, seq_len)

        # Compare
        r_input_ids_t = torch.from_numpy(r_input_ids).long()
        r_targets_t = torch.from_numpy(r_targets).long()
        r_loss_mask_t = torch.from_numpy(r_loss_mask)

        assert torch.equal(r_input_ids_t, py_batch["input_ids"]), \
            "input_ids mismatch between Rust and Python"
        assert torch.equal(r_targets_t, py_batch["targets"]), \
            "targets mismatch between Rust and Python"
        assert torch.equal(r_loss_mask_t, py_batch["loss_mask"]), \
            "loss_mask mismatch between Rust and Python"


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------


class TestBoundaryCases:
    def test_boundary_max_length_game(self):
        """Test with games that may fill all 255 move slots.

        We generate many games and check that any near-max-length game
        is correctly formatted (no off-by-one at the boundary).
        """
        input_ids, targets, loss_mask, _, game_lengths, _ = \
            engine.generate_clm_batch(256, 256, seed=123)

        for b in range(256):
            gl = int(game_lengths[b])
            # Regardless of length, format invariants hold
            assert OUTCOME_TOKEN_BASE <= input_ids[b, 0] <= PLY_LIMIT
            if gl + 1 < 256:
                assert input_ids[b, gl + 1] == 0
            assert loss_mask[b, gl] == True
            if gl + 1 < 256:
                assert loss_mask[b, gl + 1] == False

    def test_discard_ply_limit(self):
        """generate_clm_batch with discard_ply_limit should only have
        naturally-terminated games (no PLY_LIMIT outcome)."""
        input_ids, _, _, _, _, term_codes = \
            engine.generate_clm_batch(32, 256, seed=42, discard_ply_limit=True)

        for b in range(32):
            assert term_codes[b] != 5, \
                f"Game {b} has PLY_LIMIT termination but discard_ply_limit=True"
            assert input_ids[b, 0] != PLY_LIMIT, \
                f"Game {b} has PLY_LIMIT outcome token but discard_ply_limit=True"

    def test_determinism(self):
        """Same seed should produce identical results."""
        r1 = engine.generate_clm_batch(8, 256, seed=99)
        r2 = engine.generate_clm_batch(8, 256, seed=99)
        assert np.array_equal(r1[0], r2[0]), "input_ids should be deterministic"
        assert np.array_equal(r1[1], r2[1]), "targets should be deterministic"
        assert np.array_equal(r1[2], r2[2]), "loss_mask should be deterministic"
