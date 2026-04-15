"""Tests for pawn/lichess_data.py.

Covers compute_legal_indices, LegalMaskBuilder, LegalMaskCollate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import chess_engine as engine

from pawn.lichess_data import (
    LegalMaskBuilder,
    LegalMaskCollate,
    compute_legal_indices,
)


# ---------------------------------------------------------------------------
# compute_legal_indices
# ---------------------------------------------------------------------------


class TestComputeLegalIndices:
    @pytest.mark.integration
    def test_returns_indices(self):
        """Compute legal indices for small batch; check non-empty int64 output."""
        move_ids, gl, _tc = engine.generate_random_games(4, 63, seed=42)
        indices = compute_legal_indices(move_ids, gl, seq_len=64)
        assert indices.dtype in (np.int64, np.int32)
        assert indices.ndim == 1
        assert len(indices) > 0

    @pytest.mark.integration
    def test_indices_in_flat_range(self):
        """Each index must be in [0, B * seq_len * vocab_size)."""
        B = 4
        seq_len = 64
        vocab_size = 1980
        move_ids, gl, _tc = engine.generate_random_games(B, seq_len - 1, seed=42)
        indices = compute_legal_indices(move_ids, gl, seq_len=seq_len, vocab_size=vocab_size)
        max_idx = B * seq_len * vocab_size
        assert (indices >= 0).all()
        assert (indices < max_idx).all()

    @pytest.mark.integration
    def test_accepts_int_variants(self):
        """Function should accept int16/int64 without error."""
        B = 2
        seq_len = 32
        move_ids, gl, _tc = engine.generate_random_games(B, seq_len - 1, seed=42)
        # Cast to int32 and int64 just to make sure
        idx_16 = compute_legal_indices(
            move_ids.astype(np.int16), gl.astype(np.int16), seq_len=seq_len
        )
        idx_cast = compute_legal_indices(
            move_ids.astype(np.int32), gl.astype(np.int32), seq_len=seq_len
        )
        # Both should produce the same thing (same semantic arguments)
        assert np.array_equal(idx_16, idx_cast)


# ---------------------------------------------------------------------------
# LegalMaskBuilder
# ---------------------------------------------------------------------------


class TestLegalMaskBuilder:
    @pytest.mark.integration
    def test_scatter_output_shape_pure_moves(self):
        B = 4
        max_ply = 32
        vocab = 1980
        builder = LegalMaskBuilder(batch_size=B, seq_len=max_ply, vocab_size=vocab, device="cpu"
        )
        indices = torch.zeros(0, dtype=torch.long)
        mask = builder.scatter(indices, B)
        # Default pure-moves: T == max_ply (no outcome slot)
        assert mask.shape == (B, max_ply, vocab)
        assert mask.dtype == torch.bool
        assert (mask == False).all()

    @pytest.mark.integration
    def test_scatter_output_shape_prepended(self):
        B = 4
        seq_len = 32
        vocab = 1980
        builder = LegalMaskBuilder(
            batch_size=B, seq_len=seq_len, vocab_size=vocab, device="cpu",
            prepend_outcome=True,
        )
        indices = torch.zeros(0, dtype=torch.long)
        mask = builder.scatter(indices, B)
        # Both modes use the same total tensor width. prepend_outcome=True
        # consumes slot 0 for the outcome; pure-moves uses all `seq_len`
        # slots for moves.
        assert mask.shape == (B, seq_len, vocab)

    @pytest.mark.integration
    def test_scatter_sets_bits(self):
        B = 2
        seq_len = 4
        vocab = 16
        builder = LegalMaskBuilder(
            batch_size=B, seq_len=seq_len, vocab_size=vocab, device="cpu",
        )
        # Indices point into the flat (B*T*V) buffer where T == seq_len.
        indices = torch.tensor([5, 10, 100], dtype=torch.long)
        mask = builder.scatter(indices, B)
        flat = mask.view(-1)
        assert flat[5].item()
        assert flat[10].item()
        assert flat[100].item()
        # Sum == number of unique indices
        assert mask.sum().item() == 3

    @pytest.mark.integration
    def test_scatter_clears_prior_state(self):
        """Successive scatter() calls don't leak indices from previous calls."""
        B = 2
        max_ply = 4
        vocab = 16
        builder = LegalMaskBuilder(batch_size=B, seq_len=max_ply, vocab_size=vocab, device="cpu"
        )
        indices_a = torch.tensor([5, 10, 100], dtype=torch.long)
        mask_a = builder.scatter(indices_a, B).clone()
        indices_b = torch.tensor([3, 7], dtype=torch.long)
        mask_b = builder.scatter(indices_b, B)
        # mask_b should not include 5, 10, 100 from the previous call
        flat_b = mask_b.view(-1)
        assert not flat_b[5].item()
        assert not flat_b[10].item()
        assert not flat_b[100].item()
        assert flat_b[3].item()
        assert flat_b[7].item()

    @pytest.mark.integration
    def test_scatter_b_too_large_raises(self):
        builder = LegalMaskBuilder(batch_size=4, seq_len=8, vocab_size=16, device="cpu"
        )
        indices = torch.zeros(0, dtype=torch.long)
        with pytest.raises(ValueError, match="exceeds pre-allocated"):
            builder.scatter(indices, B=99)

    @pytest.mark.integration
    def test_scatter_partial_b(self):
        """scatter with B smaller than pre-allocated returns a view."""
        builder = LegalMaskBuilder(batch_size=8, seq_len=4, vocab_size=16, device="cpu"
        )
        indices = torch.tensor([1], dtype=torch.long)
        mask = builder.scatter(indices, B=3)
        assert mask.shape[0] == 3

    @pytest.mark.integration
    def test_call_produces_mask_from_batch(self):
        """__call__ invokes Rust replay and scatters. Default is pure-moves
        layout — position 0 predicts move 2."""
        B = 4
        seq_len = 63
        move_ids, gl, _tc = engine.generate_random_games(B, seq_len, seed=42)
        builder = LegalMaskBuilder(
            batch_size=B, seq_len=seq_len, vocab_size=1980, device="cpu",
        )
        batch = {"move_ids": move_ids, "game_length": gl}
        mask = builder(batch)
        assert mask.shape == (B, seq_len, 1980)
        per_position_count = mask.sum(dim=-1)
        assert per_position_count[:, 0].min() >= 1

    @pytest.mark.integration
    def test_pure_moves_mask_alignment(self):
        """In pure-moves mode, ``logits[t]`` predicts ``move_ids[t+1]``, so
        the legal mask at position ``t`` must contain ``move_ids[t+1]`` as a
        legal move. Outcome-prefixed mode uses ``move_ids[t]`` instead.

        Regression test for a subtle bug: before the legal-mask shift was
        plumbed through ``compute_legal_indices``, the pure-moves path
        scattered the raw engine mask (indexed by game ply) directly,
        leaving ``logits[t]`` off-by-one so the actual target was treated
        as illegal.
        """
        B = 4
        seq_len = 32
        move_ids, gl, _tc = engine.generate_random_games(B, seq_len, seed=42)

        # Pure-moves: position t should allow move_ids[b, t+1]
        pure_builder = LegalMaskBuilder(
            batch_size=B, seq_len=seq_len, vocab_size=1980, device="cpu",
        )
        pure_mask = pure_builder({"move_ids": move_ids, "game_length": gl})
        for b in range(B):
            length = int(gl[b])
            for t in range(length - 1):
                target = int(move_ids[b, t + 1])
                assert pure_mask[b, t, target].item(), (
                    f"pure-moves mask at (b={b}, t={t}) does not permit the "
                    f"actual target move {target} (move_ids[{b}, {t + 1}])"
                )

        # Outcome-prefixed: position t should allow move_ids[b, t]
        # (logits[0] predicts move_1 from the outcome token).
        prep_builder = LegalMaskBuilder(
            batch_size=B, seq_len=seq_len + 1, vocab_size=1980, device="cpu",
            prepend_outcome=True,
        )
        prep_mask = prep_builder({"move_ids": move_ids, "game_length": gl})
        for b in range(B):
            length = int(gl[b])
            for t in range(length):
                target = int(move_ids[b, t])
                assert prep_mask[b, t, target].item(), (
                    f"outcome-prefixed mask at (b={b}, t={t}) does not permit "
                    f"the actual target move {target} (move_ids[{b}, {t}])"
                )

    @pytest.mark.integration
    def test_call_with_tensor_input(self):
        """__call__ should accept tensors as well as numpy arrays."""
        B = 2
        max_ply = 15
        move_ids, gl, _tc = engine.generate_random_games(B, max_ply, seed=42)
        builder = LegalMaskBuilder(batch_size=B, seq_len=max_ply, vocab_size=1980, device="cpu",
        )
        batch = {
            "move_ids": torch.from_numpy(move_ids),
            "game_length": torch.from_numpy(gl),
        }
        mask = builder(batch)
        assert mask.shape == (B, max_ply, 1980)


# ---------------------------------------------------------------------------
# LegalMaskCollate
# ---------------------------------------------------------------------------


class TestLegalMaskCollate:
    @pytest.mark.integration
    def test_adds_legal_indices(self):
        collate = LegalMaskCollate(seq_len=32, vocab_size=1980)
        # Fake items
        move_ids, gl, _tc = engine.generate_random_games(3, 31, seed=42)
        items = [
            {
                "input_ids": torch.zeros(32, dtype=torch.long),
                "move_ids": torch.from_numpy(move_ids[i]),
                "game_length": int(gl[i]),
            }
            for i in range(3)
        ]
        batch = collate(items)
        assert "legal_indices" in batch
        assert batch["legal_indices"].dtype == torch.long
        # Preserves other keys
        assert "input_ids" in batch
        assert "move_ids" in batch

    @pytest.mark.integration
    def test_indices_match_direct_compute(self):
        """Collated indices should match the direct compute_legal_indices output."""
        B = 2
        seq_len = 32
        move_ids, gl, _tc = engine.generate_random_games(B, seq_len - 1, seed=42)
        collate = LegalMaskCollate(seq_len=seq_len)

        items = [
            {
                "input_ids": torch.zeros(seq_len, dtype=torch.long),
                "move_ids": torch.from_numpy(move_ids[i]),
                "game_length": int(gl[i]),
            }
            for i in range(B)
        ]
        batch = collate(items)
        direct = compute_legal_indices(move_ids, gl, seq_len=seq_len)
        assert np.array_equal(batch["legal_indices"].numpy(), direct)
