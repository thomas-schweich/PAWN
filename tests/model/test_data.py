"""Tests for pawn/data.py.

Covers _map_termination_to_outcome, pack_clm_sequences, _to_clm_batch, CLMDataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pawn.config import (
    BLACK_CHECKMATES,
    DRAW_BY_RULE,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    WHITE_CHECKMATES,
)
from pawn.data import (
    CLMDataset,
    _map_termination_to_outcome,
    _to_clm_batch,
    pack_clm_sequences,
)


# ---------------------------------------------------------------------------
# _map_termination_to_outcome
# ---------------------------------------------------------------------------


class TestMapTermination:
    @pytest.mark.unit
    def test_white_checkmates_odd_length(self):
        """Code 0 + odd length => WHITE_CHECKMATES."""
        term = np.array([0], dtype=np.uint8)
        gl = np.array([11], dtype=np.int16)
        out = _map_termination_to_outcome(term, gl)
        assert out[0].item() == WHITE_CHECKMATES

    @pytest.mark.unit
    def test_black_checkmates_even_length(self):
        term = np.array([0], dtype=np.uint8)
        gl = np.array([12], dtype=np.int16)
        out = _map_termination_to_outcome(term, gl)
        assert out[0].item() == BLACK_CHECKMATES

    @pytest.mark.unit
    def test_stalemate(self):
        term = np.array([1], dtype=np.uint8)
        gl = np.array([50], dtype=np.int16)
        out = _map_termination_to_outcome(term, gl)
        assert out[0].item() == STALEMATE

    @pytest.mark.unit
    def test_draw_by_rule_codes(self):
        # Codes 2, 3, 4 all map to DRAW_BY_RULE
        term = np.array([2, 3, 4], dtype=np.uint8)
        gl = np.array([100, 80, 60], dtype=np.int16)
        out = _map_termination_to_outcome(term, gl)
        assert out[0].item() == DRAW_BY_RULE
        assert out[1].item() == DRAW_BY_RULE
        assert out[2].item() == DRAW_BY_RULE

    @pytest.mark.unit
    def test_ply_limit_default(self):
        term = np.array([5], dtype=np.uint8)
        gl = np.array([255], dtype=np.int16)
        out = _map_termination_to_outcome(term, gl)
        assert out[0].item() == PLY_LIMIT

    @pytest.mark.unit
    def test_returns_long_tensor(self):
        term = np.array([0, 1, 2], dtype=np.uint8)
        gl = np.array([10, 20, 30], dtype=np.int16)
        out = _map_termination_to_outcome(term, gl)
        assert out.dtype == torch.long
        assert out.shape == (3,)

    @pytest.mark.unit
    def test_mixed_batch(self):
        """All 5 outcome codes mapped correctly in one batch."""
        term = np.array([0, 0, 1, 2, 3, 4, 5], dtype=np.uint8)
        gl = np.array([11, 12, 50, 100, 80, 60, 255], dtype=np.int16)
        out = _map_termination_to_outcome(term, gl)
        expected = [WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE,
                    DRAW_BY_RULE, DRAW_BY_RULE, DRAW_BY_RULE, PLY_LIMIT]
        assert out.tolist() == expected


# ---------------------------------------------------------------------------
# pack_clm_sequences — outcome-prepended mode
# ---------------------------------------------------------------------------


class TestPackCLMSequencesPrepended:
    """Pack in outcome-prefixed mode: input_ids[0] is the outcome, moves at 1.."""

    @pytest.mark.unit
    def test_output_keys(self):
        move_ids = np.array([[1, 2, 3, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8, prepend_outcome=True)
        assert set(batch.keys()) == {"input_ids", "targets", "loss_mask"}

    @pytest.mark.unit
    def test_position_zero_is_outcome(self):
        move_ids = np.array([[1, 2, 3, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8, prepend_outcome=True)
        assert batch["input_ids"][0, 0].item() == WHITE_CHECKMATES

    @pytest.mark.unit
    def test_moves_at_positions_1_to_n(self):
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8, prepend_outcome=True)
        assert batch["input_ids"][0, 1].item() == 10
        assert batch["input_ids"][0, 2].item() == 20
        assert batch["input_ids"][0, 3].item() == 30

    @pytest.mark.unit
    def test_padding_after_game_end(self):
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8, prepend_outcome=True)
        # Positions 4..7 should be PAD (1968)
        assert (batch["input_ids"][0, 4:] == PAD_TOKEN).all()

    @pytest.mark.unit
    def test_targets_shift(self):
        """targets[t] == input_ids[t+1]."""
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8, prepend_outcome=True)
        B, T = batch["targets"].shape
        for t in range(T - 1):
            assert batch["targets"][0, t] == batch["input_ids"][0, t + 1]
        assert batch["targets"][0, T - 1] == PAD_TOKEN

    @pytest.mark.unit
    def test_loss_mask_boundary(self):
        """loss_mask True for positions 0..game_length, False after."""
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8, prepend_outcome=True)
        # True for 0..3 (inclusive, outcome + 3 moves), False for 4..7
        assert (batch["loss_mask"][0, :4]).all()
        assert not (batch["loss_mask"][0, 4:]).any()

    @pytest.mark.unit
    def test_pollution_from_engine_masked_out(self):
        """Tokens past game_length in move_ids are zeroed even if nonzero."""
        move_ids = np.array([[10, 20, 30, 999, 888]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8, prepend_outcome=True)
        assert batch["input_ids"][0, 4].item() == PAD_TOKEN
        assert batch["input_ids"][0, 5].item() == PAD_TOKEN

    @pytest.mark.unit
    def test_output_dtypes(self):
        move_ids = np.array([[10, 20]], dtype=np.int16)
        gl = np.array([2], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=4, prepend_outcome=True)
        assert batch["input_ids"].dtype == torch.long
        assert batch["targets"].dtype == torch.long
        assert batch["loss_mask"].dtype == torch.bool

    @pytest.mark.unit
    def test_batch_multiple_games(self):
        move_ids = np.array(
            [[1, 2, 0, 0], [10, 20, 30, 0]], dtype=np.int16
        )
        gl = np.array([2, 3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES, BLACK_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=6, prepend_outcome=True)
        # Game 0: [WHITE_CM, 1, 2, PAD, PAD, PAD]
        assert batch["input_ids"][0, 0].item() == WHITE_CHECKMATES
        assert batch["input_ids"][0, 1].item() == 1
        assert batch["input_ids"][0, 2].item() == 2
        assert batch["input_ids"][0, 3].item() == PAD_TOKEN
        # Game 1: [BLACK_CM, 10, 20, 30, PAD, PAD]
        assert batch["input_ids"][1, 0].item() == BLACK_CHECKMATES
        assert batch["input_ids"][1, 3].item() == 30
        assert batch["input_ids"][1, 4].item() == PAD_TOKEN

    @pytest.mark.unit
    @given(n_moves=st.integers(min_value=1, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_property_loss_mask_count(self, n_moves):
        """Outcome-prefixed loss_mask has exactly game_length+1 True positions."""
        seq_len = 32
        max_ply = 24
        n_moves = min(n_moves, max_ply)
        move_ids = np.zeros((1, max_ply), dtype=np.int16)
        move_ids[0, :n_moves] = np.arange(1, n_moves + 1)
        gl = np.array([n_moves], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=seq_len, prepend_outcome=True)
        expected_true = min(n_moves + 1, seq_len)
        assert batch["loss_mask"][0].sum().item() == expected_true


# ---------------------------------------------------------------------------
# pack_clm_sequences — pure-moves mode (the default)
# ---------------------------------------------------------------------------


class TestPackCLMSequencesPureMoves:
    """Pack in pure-moves mode: input_ids[0] is the first move, no outcome slot.

    This is the default (``prepend_outcome=False``) and the canonical layout
    for models trained without outcome conditioning.
    """

    @pytest.mark.unit
    def test_position_zero_is_first_move(self):
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8)
        assert batch["input_ids"][0, 0].item() == 10
        assert batch["input_ids"][0, 1].item() == 20
        assert batch["input_ids"][0, 2].item() == 30

    @pytest.mark.unit
    def test_outcome_is_not_in_sequence(self):
        """The outcome token must not appear anywhere in pure-moves input_ids."""
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8)
        assert (batch["input_ids"] != WHITE_CHECKMATES).all()

    @pytest.mark.unit
    def test_padding_after_game_end(self):
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8)
        # Positions 3..7 should be PAD (one fewer move slot than prefixed mode)
        assert (batch["input_ids"][0, 3:] == PAD_TOKEN).all()

    @pytest.mark.unit
    def test_loss_mask_boundary(self):
        """Pure-moves loss_mask True for positions 0..game_length-1 (gl True)."""
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8)
        # game_length = 3 → loss positions 0, 1, 2 True, rest False
        assert batch["loss_mask"][0, :3].all()
        assert not batch["loss_mask"][0, 3:].any()

    @pytest.mark.unit
    def test_targets_shift(self):
        move_ids = np.array([[10, 20, 30, 0, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=8)
        B, T = batch["targets"].shape
        for t in range(T - 1):
            assert batch["targets"][0, t] == batch["input_ids"][0, t + 1]
        assert batch["targets"][0, T - 1] == PAD_TOKEN

    @pytest.mark.unit
    def test_default_is_pure_moves(self):
        """Omitting prepend_outcome defaults to pure-moves mode."""
        move_ids = np.array([[10, 20]], dtype=np.int16)
        gl = np.array([2], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        implicit = pack_clm_sequences(move_ids, gl, outcomes, seq_len=4)
        explicit = pack_clm_sequences(
            move_ids, gl, outcomes, seq_len=4, prepend_outcome=False,
        )
        assert torch.equal(implicit["input_ids"], explicit["input_ids"])
        assert torch.equal(implicit["loss_mask"], explicit["loss_mask"])

    @pytest.mark.unit
    @given(n_moves=st.integers(min_value=1, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_property_loss_mask_count(self, n_moves):
        """Pure-moves loss_mask has exactly game_length True positions."""
        seq_len = 32
        max_ply = 24
        n_moves = min(n_moves, max_ply)
        move_ids = np.zeros((1, max_ply), dtype=np.int16)
        move_ids[0, :n_moves] = np.arange(1, n_moves + 1)
        gl = np.array([n_moves], dtype=np.int16)
        outcomes = torch.tensor([WHITE_CHECKMATES], dtype=torch.long)
        batch = pack_clm_sequences(move_ids, gl, outcomes, seq_len=seq_len)
        expected_true = min(n_moves, seq_len)
        assert batch["loss_mask"][0].sum().item() == expected_true


# ---------------------------------------------------------------------------
# _to_clm_batch
# ---------------------------------------------------------------------------


class TestToCLMBatch:
    @pytest.mark.unit
    def test_basic_prepended(self):
        move_ids = np.array([[1, 2, 3, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        # Code 0 + odd gl = WHITE_CHECKMATES
        tc = np.array([0], dtype=np.uint8)
        batch = _to_clm_batch(move_ids, gl, tc, seq_len=8, prepend_outcome=True)
        assert batch["input_ids"][0, 0].item() == WHITE_CHECKMATES
        assert batch["input_ids"][0, 1].item() == 1
        assert batch["input_ids"][0, 3].item() == 3

    @pytest.mark.unit
    def test_basic_pure_moves(self):
        move_ids = np.array([[1, 2, 3, 0]], dtype=np.int16)
        gl = np.array([3], dtype=np.int16)
        tc = np.array([0], dtype=np.uint8)
        batch = _to_clm_batch(move_ids, gl, tc, seq_len=8)  # default pure moves
        assert batch["input_ids"][0, 0].item() == 1
        assert batch["input_ids"][0, 1].item() == 2
        assert batch["input_ids"][0, 2].item() == 3


# ---------------------------------------------------------------------------
# CLMDataset
# ---------------------------------------------------------------------------


class TestCLMDataset:
    @pytest.mark.unit
    def test_yields_batch_dicts(self):
        ds = CLMDataset(batch_size=4, max_ply=32, base_seed=42)
        it = iter(ds)
        batch = next(it)
        assert set(batch.keys()) == {"input_ids", "targets", "loss_mask"}
        assert batch["input_ids"].shape == (4, 32)

    @pytest.mark.unit
    def test_different_seeds_different_batches(self):
        ds1 = CLMDataset(batch_size=4, max_ply=32, base_seed=42)
        ds2 = CLMDataset(batch_size=4, max_ply=32, base_seed=99)
        b1 = next(iter(ds1))
        b2 = next(iter(ds2))
        assert not torch.equal(b1["input_ids"], b2["input_ids"])

    @pytest.mark.unit
    def test_set_start_step(self):
        ds = CLMDataset(batch_size=4, max_ply=32, base_seed=42)
        b_first = next(iter(ds))
        ds.set_start_step(10)
        b_later = next(iter(ds))
        assert not torch.equal(b_first["input_ids"], b_later["input_ids"])

    @pytest.mark.unit
    def test_pure_moves_default(self):
        """Default mode (no prepend_outcome flag) produces pure-moves sequences."""
        ds = CLMDataset(batch_size=4, max_ply=32, base_seed=42)
        batch = next(iter(ds))
        # No position should hold an outcome token — they live in their own
        # column in the new data pipeline, never in input_ids.
        assert (batch["input_ids"] < OUTCOME_TOKEN_BASE).all()

    @pytest.mark.unit
    def test_discard_ply_limit(self):
        """discard_ply_limit filters out games that hit max_ply without
        natural termination. In prepend_outcome=True mode, position 0 is
        the outcome, so verify it's never PLY_LIMIT."""
        ds = CLMDataset(
            batch_size=8, max_ply=32, base_seed=42,
            discard_ply_limit=True, prepend_outcome=True,
        )
        batch = next(iter(ds))
        assert (batch["input_ids"][:, 0] != PLY_LIMIT).all()

    @pytest.mark.unit
    def test_iterator_infinite(self):
        """Can pull multiple batches from the same iterator."""
        ds = CLMDataset(batch_size=2, max_ply=16, base_seed=42)
        it = iter(ds)
        batches = [next(it) for _ in range(3)]
        # Each batch should differ
        assert not torch.equal(batches[0]["input_ids"], batches[1]["input_ids"])


# ---------------------------------------------------------------------------
# End-to-end: Rust batch is consistent with Python pack
# ---------------------------------------------------------------------------


class TestRustPackConsistency:
    @pytest.mark.integration
    @pytest.mark.parametrize("prepend_outcome", [False, True])
    def test_pack_matches_rust(self, prepend_outcome):
        import chess_engine as engine

        B = 8
        seq_len = 64
        r_ids, r_tgt, r_mask, r_move_ids, r_gl, r_tc = engine.generate_clm_batch(
            B, seq_len, seed=42, prepend_outcome=prepend_outcome,
        )
        py = _to_clm_batch(
            r_move_ids, r_gl, r_tc, seq_len, prepend_outcome=prepend_outcome,
        )
        assert torch.equal(torch.from_numpy(r_ids).long(), py["input_ids"])
        assert torch.equal(torch.from_numpy(r_tgt).long(), py["targets"])
        assert torch.equal(torch.from_numpy(r_mask), py["loss_mask"])
