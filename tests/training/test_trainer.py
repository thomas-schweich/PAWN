"""Tests for pawn.trainer module: legal move rate, promo grid index, step helpers."""

from __future__ import annotations

import pytest
import torch

from pawn.trainer import (
    _build_promo_grid_index,
    _get_promo_grid_index,
    compute_legal_move_rate,
)


# ---------------------------------------------------------------------------
# _build_promo_grid_index / _get_promo_grid_index
# ---------------------------------------------------------------------------


class TestBuildPromoGridIndex:
    def test_length_is_176(self):
        """44 pairs * 4 promo types = 176 entries."""
        idx = _build_promo_grid_index()
        assert len(idx) == 176

    def test_entries_are_valid_grid_indices(self):
        """Each entry must be in [0, 4095] (grid = 64x64 = 4096 slots)."""
        idx = _build_promo_grid_index()
        for v in idx:
            assert 0 <= v < 4096

    def test_four_consecutive_entries_are_equal(self):
        """Token layout: PROMO_START + pair_idx * 4 + promo_type.
        So each pair_idx corresponds to 4 consecutive entries with the same grid index.
        """
        idx = _build_promo_grid_index()
        for pair_idx in range(44):
            base = idx[pair_idx * 4]
            for promo_type in range(4):
                assert idx[pair_idx * 4 + promo_type] == base


class TestGetPromoGridIndex:
    def test_returns_long_tensor(self, cpu_device):
        t = _get_promo_grid_index(cpu_device)
        assert t.dtype == torch.long
        assert t.shape == (176,)
        assert str(t.device) == "cpu"

    def test_matches_builder(self, cpu_device):
        ref = _build_promo_grid_index()
        t = _get_promo_grid_index(cpu_device)
        assert t.tolist() == ref

    def test_all_values_in_grid_range(self, cpu_device):
        t = _get_promo_grid_index(cpu_device)
        assert (t >= 0).all().item()
        assert (t < 4096).all().item()

    def test_caches_across_calls(self, cpu_device):
        """Module-level cache means second call is O(1) — verified by identity of list."""
        import pawn.trainer as tr

        tr._PROMO_GRID_INDEX = None  # force rebuild
        _get_promo_grid_index(cpu_device)
        cached = tr._PROMO_GRID_INDEX
        assert cached is not None
        _get_promo_grid_index(cpu_device)
        # Still the same cached list (identity check)
        assert tr._PROMO_GRID_INDEX is cached


# ---------------------------------------------------------------------------
# compute_legal_move_rate
# ---------------------------------------------------------------------------


def _make_logits(B: int, T: int, V: int, preds: torch.Tensor) -> torch.Tensor:
    """Build (B,T,V) logits whose argmax equals ``preds`` (shape (B,T))."""
    logits = torch.zeros(B, T, V)
    logits.scatter_(-1, preds.unsqueeze(-1), 10.0)
    return logits


def _pack_grid(dense: torch.Tensor) -> torch.Tensor:
    """Pack (B, max_ply, 64, 64) dense bits to (B, max_ply, 64) int64 bitmask.

    Bit b of packed[..., s] encodes dense[..., s, b].
    """
    B, P, _, _ = dense.shape
    bits = torch.arange(64, dtype=torch.long)
    packed = (dense.long() << bits).sum(dim=-1)
    return packed


class TestComputeLegalMoveRate:
    def test_all_legal_predictions_yields_one(self, cpu_device):
        """If preds hit legal slots, rate == 1.0."""
        B, T, V = 2, 4, 4284
        max_ply = 4
        # Predict token 1 (grid index 0) at every position
        preds = torch.ones(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        # Dense legal mask — set position (0,0) legal everywhere
        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., 0, 0] = 1.0  # (src=0, dst=0) is legal
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(1.0)

    def test_all_illegal_predictions_yields_zero(self, cpu_device):
        """If preds miss, rate == 0.0."""
        B, T, V = 2, 4, 4284
        max_ply = 4
        # Predict token 1 (grid index 0) but only slot (1,1) is legal
        preds = torch.ones(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., 1, 1] = 1.0  # different slot
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_empty_loss_mask_returns_zero(self, cpu_device):
        """No positions to evaluate -> 0.0 by convention."""
        B, T, V = 2, 4, 4284
        max_ply = 4
        preds = torch.zeros(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)
        legal_grid = torch.zeros(B, max_ply, 64, dtype=torch.long)
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == 0.0

    def test_partial_legal_rate(self, cpu_device):
        """Mix of legal and illegal preds yields fractional rate."""
        B, T, V = 2, 2, 4284
        max_ply = 2
        # preds[0,:] = token 1 (grid 0), preds[1,:] = token 1 (grid 0)
        preds = torch.ones(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        # Make slot (0,0) legal only for batch 0, all plies
        dense[0, :, 0, 0] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        # 4 positions total, 2 legal (batch 0, both plies) -> 0.5
        assert rate == pytest.approx(0.5)

    def test_pad_token_predictions_are_not_counted_legal(self, cpu_device):
        """PAD token (0) is outside grid and promo ranges — always illegal."""
        B, T, V = 1, 2, 4284
        max_ply = 2
        preds = torch.zeros(B, T, dtype=torch.long)  # PAD token
        logits = _make_logits(B, T, V, preds)

        # Make everything legal — shouldn't matter, PAD is never counted
        dense = torch.ones(B, max_ply, 64, 64)
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_outcome_token_predictions_are_not_counted_legal(self, cpu_device):
        """Outcome tokens (4273+) are outside grid/promo ranges."""
        B, T, V = 1, 1, 4284
        max_ply = 1
        preds = torch.full((B, T), 4273, dtype=torch.long)  # outcome token
        logits = _make_logits(B, T, V, preds)

        dense = torch.ones(B, max_ply, 64, 64)
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.zeros(B, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_game_length_clips_positions(self, cpu_device):
        """Positions beyond game_length are not counted."""
        B, T, V = 1, 4, 4284
        max_ply = 4
        # All preds legal
        preds = torch.ones(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., 0, 0] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        # game_length 0 -> only position 0 counts
        game_lengths = torch.zeros(B, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        # Only 1 position (idx 0), which is legal -> 1.0
        assert rate == pytest.approx(1.0)

    def test_promo_token_resolves_via_lookup(self, cpu_device):
        """Promotion tokens (4097-4272) use the promo_grid_index lookup."""
        B, T, V = 1, 1, 4284
        max_ply = 1
        idx = _build_promo_grid_index()
        # Pick the first promo token (4097) and look up its grid index
        promo_token = 4097
        grid_idx = idx[0]
        src = grid_idx // 64
        dst = grid_idx % 64

        preds = torch.full((B, T), promo_token, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[0, 0, src, dst] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.zeros(B, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(1.0)

    def test_promo_token_illegal_when_grid_mismatches(self, cpu_device):
        B, T, V = 1, 1, 4284
        max_ply = 1
        promo_token = 4097  # some promo token
        preds = torch.full((B, T), promo_token, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        # Empty legal mask
        legal_grid = torch.zeros(B, max_ply, 64, dtype=torch.long)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.zeros(B, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_returns_python_float(self, cpu_device):
        """compute_legal_move_rate returns plain float, not a tensor."""
        B, T, V = 1, 1, 4284
        max_ply = 1
        preds = torch.zeros(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)
        legal_grid = torch.zeros(B, max_ply, 64, dtype=torch.long)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.zeros(B, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert isinstance(rate, float)

    def test_loss_mask_intersects_game_length(self, cpu_device):
        """Only positions where BOTH loss_mask and <=game_length are counted."""
        B, T, V = 1, 4, 4284
        max_ply = 4
        preds = torch.ones(B, T, dtype=torch.long)  # legal at slot 0
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., 0, 0] = 1.0
        legal_grid = _pack_grid(dense)

        # Only position 2 has loss_mask True
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        loss_mask[0, 2] = True
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        # Only 1 position counted, legal -> 1.0
        assert rate == pytest.approx(1.0)

    def test_n_plies_clips_to_max_ply(self, cpu_device):
        """When T > max_ply, only first max_ply positions are evaluated.

        This exposes whether valid_count reflects only counted positions.
        """
        B, T, V = 1, 4, 4284
        max_ply = 2
        # Predict legal slot 1 (grid 0) at every T=4 positions
        preds = torch.ones(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., 0, 0] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        # Only 2 positions counted (max_ply=2), both legal -> 1.0
        assert rate == pytest.approx(1.0)

    def test_different_preds_per_batch_position(self, cpu_device):
        """Independently predict legal/illegal in different batch/position combos."""
        B, T, V = 2, 2, 4284
        max_ply = 2
        preds = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        # grid indices 0, 1, 2, 3
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        # Batch 0, ply 0: slot (0,0) legal -> pred 1 (grid 0) legal
        dense[0, 0, 0, 0] = 1.0
        # Batch 0, ply 1: no slots legal -> pred 2 (grid 1) illegal
        # Batch 1, ply 0: slot (0,2) legal -> pred 3 (grid 2) legal
        dense[1, 0, 0, 2] = 1.0
        # Batch 1, ply 1: no slots legal -> pred 4 (grid 3) illegal
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T - 1, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        # 2 of 4 legal
        assert rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# train_step / val_step smoke (using toy_model + sample_clm_batch)
# ---------------------------------------------------------------------------


def _batch_to_tensors(batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy arrays from Rust engine to torch tensors for PAWNCLM."""
    ids = torch.from_numpy(batch["input_ids"]).long().to(device)
    tgt = torch.from_numpy(batch["targets"]).long().to(device)
    mask = torch.from_numpy(batch["loss_mask"]).bool().to(device)
    return ids, tgt, mask


class TestTrainStepSmoke:
    """Integration: exercise forward_train on a toy model + real rust batch."""

    def test_forward_train_returns_loss_and_metrics(
        self, toy_model, sample_clm_batch, cpu_device,
    ):
        ids, tgt, mask = _batch_to_tensors(sample_clm_batch, cpu_device)
        loss, metrics = toy_model.forward_train(ids, mask, tgt)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"].ndim == 0
        assert metrics["accuracy"].ndim == 0

    def test_backward_populates_grads(
        self, toy_model, sample_clm_batch, cpu_device,
    ):
        toy_model.train()
        ids, tgt, mask = _batch_to_tensors(sample_clm_batch, cpu_device)
        loss, _ = toy_model.forward_train(ids, mask, tgt)
        loss.backward()
        grads = [p.grad for p in toy_model.parameters() if p.requires_grad]
        assert any(g is not None and (g != 0).any() for g in grads)

    def test_accuracy_in_valid_range(
        self, toy_model, sample_clm_batch, cpu_device,
    ):
        ids, tgt, mask = _batch_to_tensors(sample_clm_batch, cpu_device)
        _, metrics = toy_model.forward_train(ids, mask, tgt)
        acc = metrics["accuracy"].item()
        assert 0.0 <= acc <= 1.0


class TestValStepSmoke:
    """val_step equivalent: forward, no grad, eval mode."""

    def test_no_grads_in_no_grad_forward(
        self, toy_model, sample_clm_batch, cpu_device,
    ):
        toy_model.eval()
        ids, _, mask = _batch_to_tensors(sample_clm_batch, cpu_device)
        with torch.no_grad():
            logits, _ = toy_model(ids, mask)
        # logits should not require grad
        assert not logits.requires_grad

    def test_forward_logits_shape(
        self, toy_model, sample_clm_batch, cpu_device,
    ):
        toy_model.eval()
        ids, _, mask = _batch_to_tensors(sample_clm_batch, cpu_device)
        with torch.no_grad():
            logits, _ = toy_model(ids, mask)
        B, T = ids.shape
        assert logits.shape[0] == B
        assert logits.shape[1] == T
        assert logits.shape[2] == toy_model.cfg.vocab_size
