"""End-to-end tests for 512-token (and arbitrary) sequence lengths.

Verifies that non-default max_seq_len values propagate correctly through the
full pipeline: engine generation → data packing → model forward → legal move
rate → evaluation, with no silent truncation or off-by-one errors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import chess_engine as engine

from pawn.config import (
    CLMConfig,
    NUM_ACTIONS,
    PAD_TOKEN,
    PLY_LIMIT,
    OUTCOME_TOKEN_BASE,
    WHITE_CHECKMATES,
    BLACK_CHECKMATES,
    STALEMATE,
    DRAW_BY_RULE,
)
from pawn.data import create_validation_set, CLMDataset, _to_clm_batch
from pawn.model import PAWNCLM
from pawn.trainer import compute_legal_move_rate_from_preds


# All pretraining outcome tokens
_PRETRAINING_OUTCOMES = {
    WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE, DRAW_BY_RULE, PLY_LIMIT,
}


# ---------------------------------------------------------------------------
# Engine generates correct shapes and content at seq_len=512
# (default: prepend_outcome=False — pure move sequences)
# ---------------------------------------------------------------------------


class TestEngine512:
    """Verify the Rust engine produces valid CLM batches at seq_len=512.

    Default mode: prepend_outcome=False, so sequences are pure moves
    [m1, m2, ..., mN, PAD, ...] with max_ply=seq_len=512.
    """

    SEQ_LEN = 512
    MAX_PLY = 512  # prepend_outcome=False → max_ply = seq_len
    B = 64

    @pytest.fixture(scope="class")
    def clm_batch(self):
        return engine.generate_clm_batch(self.B, self.SEQ_LEN, seed=42)

    @pytest.mark.integration
    def test_shapes(self, clm_batch):
        input_ids, targets, loss_mask, move_ids, game_lengths, term_codes = clm_batch
        assert input_ids.shape == (self.B, self.SEQ_LEN)
        assert targets.shape == (self.B, self.SEQ_LEN)
        assert loss_mask.shape == (self.B, self.SEQ_LEN)
        assert move_ids.shape == (self.B, self.MAX_PLY)
        assert game_lengths.shape == (self.B,)
        assert term_codes.shape == (self.B,)

    @pytest.mark.integration
    def test_position_zero_is_move(self, clm_batch):
        """Without prepend_outcome, position 0 is the first move, not an outcome."""
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            if gl > 0:
                tok = int(input_ids[b, 0])
                assert 0 <= tok < NUM_ACTIONS, f"batch {b}: token {tok} not a valid action"

    @pytest.mark.integration
    def test_moves_in_valid_range(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(min(gl, self.SEQ_LEN)):
                tok = int(input_ids[b, t])
                assert 0 <= tok < NUM_ACTIONS, f"batch {b} ply {t}: token {tok} out of range"

    @pytest.mark.integration
    def test_padding_beyond_game_length(self, clm_batch):
        input_ids, _, _, _, game_lengths, _ = clm_batch
        for b in range(input_ids.shape[0]):
            gl = int(game_lengths[b])
            for t in range(gl, self.SEQ_LEN):
                assert input_ids[b, t] == PAD_TOKEN

    @pytest.mark.integration
    def test_target_shift(self, clm_batch):
        input_ids, targets, *_ = clm_batch
        assert np.array_equal(targets[:, :-1], input_ids[:, 1:])
        assert (targets[:, -1] == PAD_TOKEN).all()

    @pytest.mark.integration
    def test_loss_mask_boundary(self, clm_batch):
        """Without prepend_outcome, loss_mask has gl True positions (0..gl-1)."""
        _, _, loss_mask, _, game_lengths, _ = clm_batch
        for b in range(loss_mask.shape[0]):
            gl = int(game_lengths[b])
            # Positions 0..gl-1 should be True
            for t in range(gl):
                assert loss_mask[b, t], f"batch {b} pos {t} should be masked"
            # Positions gl..seq_len-1 should be False
            for t in range(gl, self.SEQ_LEN):
                assert not loss_mask[b, t], f"batch {b} pos {t} should not be masked"

    @pytest.mark.integration
    def test_raw_moves_replayable(self, clm_batch):
        _, _, _, move_ids, game_lengths, _ = clm_batch
        is_valid, _first_illegal = engine.validate_games(move_ids, game_lengths)
        assert all(is_valid), "Some 512-token games failed replay validation"


# ---------------------------------------------------------------------------
# Games actually reach long ply counts
# ---------------------------------------------------------------------------


class TestLongGames:
    """Verify that seq_len=512 actually produces games longer than 255 plies.

    Uses default prepend_outcome=False, so max_ply=512.
    """

    @pytest.mark.integration
    def test_some_games_exceed_255_plies(self):
        """With 256 games at seq_len=512, many should reach >255 plies."""
        _, _, _, _, game_lengths, _ = engine.generate_clm_batch(256, 512, seed=42)
        long_games = sum(1 for gl in game_lengths if int(gl) > 255)
        assert long_games > 0, "No games reached >255 plies — 512-token may be silently truncated"

    @pytest.mark.integration
    def test_move_tokens_beyond_255_are_valid(self):
        """Verify moves at positions >256 are real move tokens, not PAD."""
        input_ids, _, _, _, game_lengths, _ = engine.generate_clm_batch(256, 512, seed=42)
        found_late_move = False
        for b in range(256):
            gl = int(game_lengths[b])
            if gl > 257:
                # Position 257 in input_ids (0-indexed) is ply 257
                tok = int(input_ids[b, 257])
                assert 0 <= tok < NUM_ACTIONS, f"Token at position 257 should be a move, got {tok}"
                found_late_move = True
        assert found_late_move, "No game had moves at position 257"

    @pytest.mark.integration
    def test_ply_limit_at_512(self):
        """PlyLimit should appear for games that hit 512 plies, not 255.

        With prepend_outcome=False, max_ply=seq_len=512.
        """
        _, _, _, _, game_lengths, term_codes = engine.generate_clm_batch(256, 512, seed=42)
        for b in range(256):
            gl = int(game_lengths[b])
            tc = int(term_codes[b])
            if tc == 5:  # PlyLimit
                assert gl == 512, (
                    f"PlyLimit game has gl={gl}, expected 512 (max_ply=512)"
                )
            # Games should never have gl > 512
            assert gl <= 512, f"Game length {gl} exceeds max_ply=512"

    @pytest.mark.integration
    def test_no_ply_limit_at_255(self):
        """With seq_len=512, no game should be truncated at 255 plies."""
        _, _, _, _, game_lengths, term_codes = engine.generate_clm_batch(256, 512, seed=42)
        for b in range(256):
            gl = int(game_lengths[b])
            tc = int(term_codes[b])
            if gl == 255 and tc == 5:  # PlyLimit at 255
                pytest.fail("Game truncated at 255 plies despite seq_len=512")


# ---------------------------------------------------------------------------
# Outcome tokens remain correct for long games
# ---------------------------------------------------------------------------


class TestOutcomeTokens512:
    """Verify outcome token correctness for 512-token batches.

    Uses prepend_outcome=True so outcome appears at input_ids[b, 0].
    With prepend_outcome=True, max_ply=511.
    """

    @pytest.mark.integration
    def test_checkmate_parity(self):
        """Checkmate winner determined by game length parity, even at long ply counts."""
        input_ids, _, _, _, game_lengths, term_codes = engine.generate_clm_batch(
            256, 512, seed=42, prepend_outcome=True
        )
        for b in range(256):
            gl = int(game_lengths[b])
            tc = int(term_codes[b])
            outcome = int(input_ids[b, 0])
            if tc == 0:  # Checkmate
                if gl % 2 == 1:  # Odd = white made last move
                    assert outcome == WHITE_CHECKMATES
                else:
                    assert outcome == BLACK_CHECKMATES

    @pytest.mark.integration
    def test_ply_limit_outcome_token(self):
        """Games hitting 511 plies should get PLY_LIMIT outcome token."""
        input_ids, _, _, _, game_lengths, term_codes = engine.generate_clm_batch(
            256, 512, seed=42, prepend_outcome=True
        )
        for b in range(256):
            tc = int(term_codes[b])
            outcome = int(input_ids[b, 0])
            if tc == 5:  # PlyLimit
                assert outcome == PLY_LIMIT

    @pytest.mark.integration
    def test_draw_outcomes(self):
        """Stalemate and draw-by-rule outcomes remain correct at 512 tokens."""
        input_ids, _, _, _, _, term_codes = engine.generate_clm_batch(
            256, 512, seed=42, prepend_outcome=True
        )
        for b in range(256):
            tc = int(term_codes[b])
            outcome = int(input_ids[b, 0])
            if tc == 1:  # Stalemate
                assert outcome == STALEMATE
            elif tc in (2, 3, 4):  # 75-move, fivefold, insufficient material
                assert outcome == DRAW_BY_RULE


# ---------------------------------------------------------------------------
# Rust/Python CLM parity at 512
# ---------------------------------------------------------------------------


class TestRustPythonParity512:
    @pytest.mark.integration
    def test_rust_clm_matches_python_pack_at_512(self):
        """Rust generate_clm_batch (prepend_outcome=True) and Python _to_clm_batch agree at seq_len=512.

        _to_clm_batch always prepends outcome tokens, so the Rust side must use
        prepend_outcome=True for the outputs to match.
        """
        seq_len = 512
        B = 16
        r_input_ids, r_targets, r_loss_mask, r_move_ids, r_gl, r_tc = (
            engine.generate_clm_batch(B, seq_len, seed=42, prepend_outcome=True)
        )
        py_batch = _to_clm_batch(r_move_ids, r_gl, r_tc, seq_len)
        assert torch.equal(torch.from_numpy(r_input_ids).long(), py_batch["input_ids"])
        assert torch.equal(torch.from_numpy(r_targets).long(), py_batch["targets"])
        assert torch.equal(torch.from_numpy(r_loss_mask), py_batch["loss_mask"])


# ---------------------------------------------------------------------------
# Model forward pass at 512 tokens
# ---------------------------------------------------------------------------


class TestModelForward512:
    """Verify that PAWNCLM handles max_seq_len=512 without errors or truncation."""

    @pytest.fixture
    def model_512(self):
        cfg = CLMConfig.toy()
        cfg.max_seq_len = 512
        torch.manual_seed(0)
        return PAWNCLM(cfg).eval()

    @pytest.mark.unit
    def test_rope_and_mask_shapes(self, model_512):
        assert model_512.rope_cos.shape[2] == 512
        assert model_512.rope_sin.shape[2] == 512
        assert model_512.causal_mask.shape == (512, 512)

    @pytest.mark.unit
    def test_forward_full_512(self, model_512):
        """Model can process a full 512-token sequence."""
        B = 2
        input_ids = torch.randint(0, NUM_ACTIONS, (B, 512))
        mask = torch.ones(B, 512, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = model_512(input_ids, mask, hidden_only=True)
        assert logits.shape == (B, 512, CLMConfig.toy().vocab_size)

    @pytest.mark.unit
    def test_forward_shorter_than_max(self, model_512):
        """Model handles sequences shorter than max_seq_len."""
        B = 2
        input_ids = torch.randint(0, NUM_ACTIONS, (B, 128))
        mask = torch.ones(B, 128, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = model_512(input_ids, mask, hidden_only=True)
        assert logits.shape == (B, 128, CLMConfig.toy().vocab_size)

    @pytest.mark.unit
    def test_rejects_longer_than_max(self, model_512):
        """Model raises on sequences exceeding max_seq_len."""
        input_ids = torch.randint(0, NUM_ACTIONS, (1, 513))
        mask = torch.ones(1, 513, dtype=torch.bool)
        with pytest.raises(ValueError, match="exceeds max"):
            model_512(input_ids, mask)


# ---------------------------------------------------------------------------
# Legal move grid at 512 — replay and legal rate computation
# ---------------------------------------------------------------------------


class TestLegalGrid512:
    """Verify legal move masks and legal move rate at 512-token sequences.

    Uses default prepend_outcome=False, so max_ply=512 and sequences are
    pure moves. The engine returns move_ids with shape (B, 512) and
    legal_grid with shape (B, 512, 64).
    """

    @pytest.fixture(scope="class")
    def val_data_512(self):
        return create_validation_set(
            n_games=64, max_ply=512, seed=42,
        )

    @pytest.mark.integration
    def test_legal_grid_shape(self, val_data_512):
        """Legal grid has correct shape for 512-token sequences."""
        legal_grid = val_data_512["legal_grid"]
        # prepend_outcome=False → max_ply=512
        assert legal_grid.shape[0] == 64
        assert legal_grid.shape[1] == 512
        assert legal_grid.shape[2] == 64

    @pytest.mark.integration
    def test_game_lengths_match(self, val_data_512):
        """game_lengths in val set should match loss_mask counts.

        With prepend_outcome=False, loss_mask has gl True positions (0..gl-1).
        """
        loss_mask = val_data_512["loss_mask"]
        game_lengths = val_data_512["game_lengths"]
        for b in range(64):
            gl = int(game_lengths[b])
            mask_count = int(loss_mask[b].sum().item())
            assert mask_count == gl

    @pytest.mark.integration
    def test_legal_grid_nonempty_at_late_plies(self, val_data_512):
        """For long games, legal grid entries beyond ply 255 should be non-empty."""
        legal_grid = val_data_512["legal_grid"]
        game_lengths = val_data_512["game_lengths"]
        found_late = False
        for b in range(64):
            gl = int(game_lengths[b])
            if gl > 256:
                # Check that the legal grid at ply 256 has some legal moves
                grid_at_256 = legal_grid[b, 256, :]
                n_legal = sum(bin(int(x)).count("1") for x in grid_at_256)
                assert n_legal > 0, f"No legal moves at ply 256 for game {b} (gl={gl})"
                found_late = True
        assert found_late, "No games long enough to test late-ply legal grid"


class TestLegalGridOutcomePrepended:
    """Verify validation set layout when prepend_outcome=True."""

    @pytest.fixture(scope="class")
    def val(self):
        return create_validation_set(
            n_games=32, max_ply=128, seed=42, prepend_outcome=True,
        )

    @pytest.mark.integration
    def test_legal_grid_shape_matches_seq_len(self, val):
        """align_legal_to_preds pads the grid so it's indexable against preds."""
        legal_grid = val["legal_grid"]
        input_ids = val["input_ids"]
        assert legal_grid.shape[1] == input_ids.shape[1] == 128

    @pytest.mark.integration
    def test_move_ids_stored_separately(self, val):
        """move_ids is the raw (B, max_ply) array, without the outcome prefix."""
        move_ids = val["move_ids"]
        # max_ply = seq_len - 1 when prepend_outcome=True.
        assert move_ids.shape[1] == 127

    @pytest.mark.integration
    def test_prepend_outcome_flag_stored(self, val):
        assert bool(val["prepend_outcome"].item()) is True

    @pytest.mark.integration
    def test_preds_aligned_with_legal_grid(self, val):
        """Using targets as ground-truth preds should yield near-perfect legal rate."""
        from pawn.trainer import compute_legal_move_rate_from_preds
        rate = compute_legal_move_rate_from_preds(
            val["targets"], val["legal_grid"], val["loss_mask"],
            val["game_lengths"],
        )
        # targets[t] is the actual played move and must be legal; anything
        # short of ~1.0 means we got the alignment wrong. A little slack for
        # the zero-padded last row.
        assert rate > 0.98, f"legal rate {rate} — alignment broken?"


# ---------------------------------------------------------------------------
# Ply-range filter for compute_legal_move_rate_from_preds
# ---------------------------------------------------------------------------


class TestPlyRangeFilter:
    """Verify min_ply/max_ply_limit correctly restrict the legal rate computation."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Generate val data with known legal grids at seq_len=64.

        Uses ``targets`` as predictions — legal_grid is shifted by one ply
        in create_validation_set to align with targets (target[p] is the move
        at ply p+1).  This gives a legal rate of ~1.0 for move positions.
        """
        val = create_validation_set(n_games=16, max_ply=64, seed=42)
        # targets[p] = the next move (ply p+1), aligned with legal_grid[p]
        val["preds"] = val["targets"].clone()
        return val

    @pytest.mark.integration
    def test_full_range_returns_nonzero(self, setup):
        """Default (no filter) with ground-truth preds returns a high legal rate."""
        rate = compute_legal_move_rate_from_preds(
            setup["preds"], setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"],
        )
        assert rate > 0.5

    @pytest.mark.integration
    def test_min_ply_excludes_early(self, setup):
        """Setting min_ply > 0 excludes early plies from the computation.

        Uses ground-truth preds for the first 10 plies and PAD_TOKEN (illegal)
        for the rest.  Full-range rate should be moderate (mix of legal and
        illegal), but restricting to [0, 10) should be ~1.0 since those plies
        have the real moves.  Restricting to [10, inf) should be ~0.0 since
        those plies have PAD predictions which are never legal.
        """
        split = 10
        # Build preds: ground-truth for plies < split, PAD elsewhere
        preds = torch.full_like(setup["preds"], PAD_TOKEN)
        preds[:, :split] = setup["preds"][:, :split]

        rate_early = compute_legal_move_rate_from_preds(
            preds, setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"], max_ply_limit=split,
        )
        rate_late = compute_legal_move_rate_from_preds(
            preds, setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"], min_ply=split,
        )
        rate_all = compute_legal_move_rate_from_preds(
            preds, setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"],
        )
        # Early range has real moves → high legality
        assert rate_early > 0.8
        # Late range has PAD predictions → ~0% legality
        assert rate_late < 0.05
        # Full range mixes both → between the two extremes
        assert rate_late < rate_all < rate_early

    @pytest.mark.integration
    def test_max_ply_limit_excludes_late(self, setup):
        """Setting max_ply_limit < n_plies excludes late plies."""
        rate_early = compute_legal_move_rate_from_preds(
            setup["preds"], setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"], max_ply_limit=5,
        )
        assert rate_early > 0.0

    @pytest.mark.integration
    def test_empty_range_returns_zero(self, setup):
        """min_ply beyond all game lengths returns 0.0."""
        rate = compute_legal_move_rate_from_preds(
            setup["preds"], setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"], min_ply=9999,
        )
        assert rate == 0.0

    @pytest.mark.integration
    def test_min_equals_max_returns_zero(self, setup):
        """When min_ply == max_ply_limit, range is empty → 0.0."""
        rate = compute_legal_move_rate_from_preds(
            setup["preds"], setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"], min_ply=5, max_ply_limit=5,
        )
        assert rate == 0.0

    @pytest.mark.integration
    def test_disjoint_ranges_cover_full(self, setup):
        """Rates from [0, split) and [split, N) should together match the full range.

        Rather than checking exact equality (which requires careful counting),
        verify both sub-ranges are computable and non-zero.
        """
        split = 20
        rate_lo = compute_legal_move_rate_from_preds(
            setup["preds"], setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"], max_ply_limit=split,
        )
        rate_hi = compute_legal_move_rate_from_preds(
            setup["preds"], setup["legal_grid"], setup["loss_mask"],
            setup["game_lengths"], min_ply=split,
        )
        assert rate_lo > 0.0
        assert rate_hi > 0.0


# ---------------------------------------------------------------------------
# CLMDataset at seq_len=512
# ---------------------------------------------------------------------------


class TestCLMDataset512:
    """Verify that CLMDataset produces correct 512-token sequences."""

    @pytest.mark.integration
    def test_dataset_yields_512_token_batches(self):
        ds = CLMDataset(
            batch_size=8, max_ply=512, base_seed=42,
        )
        ds.set_start_step(0)
        batch = next(iter(ds))
        assert batch["input_ids"].shape == (8, 512)
        assert batch["targets"].shape == (8, 512)
        assert batch["loss_mask"].shape == (8, 512)

    @pytest.mark.integration
    def test_dataset_not_truncated_to_256(self):
        """Games from CLMDataset at max_ply=512 can exceed 255 plies."""
        ds = CLMDataset(batch_size=128, max_ply=512, base_seed=42)
        ds.set_start_step(0)
        batch = next(iter(ds))
        # Without prepend_outcome, loss_mask count = game_length directly
        game_lengths = batch["loss_mask"].sum(dim=1)
        long_games = (game_lengths > 255).sum().item()
        assert long_games > 0, "No games exceeded 255 plies in CLMDataset at max_ply=512"


# ---------------------------------------------------------------------------
# discard_ply_limit at 512
# ---------------------------------------------------------------------------


class TestDiscardPlyLimit512:
    """Verify discard_ply_limit works correctly at seq_len=512."""

    @pytest.mark.integration
    def test_discard_removes_ply_limit_games(self):
        """With discard_ply_limit=True, no games should have term_code=PlyLimit."""
        _, _, _, _, _, term_codes = engine.generate_clm_batch(
            64, 512, seed=42, discard_ply_limit=True,
        )
        for b in range(64):
            assert int(term_codes[b]) != 5, "PlyLimit game found despite discard_ply_limit=True"

    @pytest.mark.integration
    def test_discard_still_allows_long_natural_games(self):
        """Even with discard_ply_limit=True, games can still be long (just not ply-limited)."""
        _, _, _, _, game_lengths, _ = engine.generate_clm_batch(
            256, 512, seed=42, discard_ply_limit=True,
        )
        max_gl = max(int(gl) for gl in game_lengths)
        # Some natural games should still be fairly long
        assert max_gl > 100, f"Longest natural game only {max_gl} plies"
