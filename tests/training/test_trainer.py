"""Tests for pawn.trainer module: legal move rate, action grid index, step helpers."""

from __future__ import annotations

import pytest
import torch

from pawn.config import NUM_ACTIONS, PAD_TOKEN, OUTCOME_TOKEN_BASE
from pawn.trainer import (
    _build_action_grid_index,
    _get_action_grid_index,
    compute_legal_move_rate,
)


# ---------------------------------------------------------------------------
# _build_action_grid_index / _get_action_grid_index
# ---------------------------------------------------------------------------


class TestBuildActionGridIndex:
    def test_length_is_num_actions(self):
        """One grid index per action token."""
        idx = _build_action_grid_index(NUM_ACTIONS)
        assert len(idx) == NUM_ACTIONS

    def test_entries_are_valid_grid_indices(self):
        """Each entry must be in [0, 4095] (grid = 64x64 = 4096 slots)."""
        idx = _build_action_grid_index(NUM_ACTIONS)
        for v in idx:
            assert 0 <= v < 4096

    def test_first_action_is_a1b1(self):
        """Action 0 = a1b1, grid index = 0*64 + 1 = 1."""
        idx = _build_action_grid_index(NUM_ACTIONS)
        assert idx[0] == 0 * 64 + 1  # a1=0, b1=1


class TestGetActionGridIndex:
    def test_returns_long_tensor(self, cpu_device):
        t = _get_action_grid_index(cpu_device)
        assert t.dtype == torch.long
        assert t.shape == (NUM_ACTIONS,)
        assert str(t.device) == "cpu"

    def test_matches_builder(self, cpu_device):
        ref = _build_action_grid_index(NUM_ACTIONS)
        t = _get_action_grid_index(cpu_device)
        assert t.tolist() == ref

    def test_all_values_in_grid_range(self, cpu_device):
        t = _get_action_grid_index(cpu_device)
        assert (t >= 0).all().item()
        assert (t < 4096).all().item()

    def test_caches_across_calls(self, cpu_device):
        """Module-level cache means second call is O(1) -- verified by identity check."""
        import pawn.trainer as tr

        tr._ACTION_GRID_INDEX_CACHE.clear()
        tr._ACTION_GRID_TENSOR_CACHE.clear()
        t1 = _get_action_grid_index(cpu_device)
        assert NUM_ACTIONS in tr._ACTION_GRID_INDEX_CACHE
        t2 = _get_action_grid_index(cpu_device)
        # Same tensor object (cached)
        assert t1 is t2


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

    @pytest.fixture(autouse=True)
    def _grid_index(self):
        """Cache the action-to-grid mapping for test helpers."""
        self._idx = _build_action_grid_index(NUM_ACTIONS)

    def _grid_for_action(self, action: int) -> tuple[int, int]:
        """Return (src, dst) grid coords for an action token."""
        grid = self._idx[action]
        return grid // 64, grid % 64

    def test_all_legal_predictions_yields_one(self, cpu_device):
        """If preds hit legal slots at every position, rate == 1.0 exactly."""
        V = 1980
        B, T = 2, 4
        max_ply = 4
        action = 0  # a1b1
        src, dst = self._grid_for_action(action)
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., src, dst] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(1.0)
        assert B * T > 0

    def test_all_illegal_predictions_yields_zero(self, cpu_device):
        """If preds miss, rate == 0.0."""
        V = 1980
        B, T = 2, 4
        max_ply = 4
        action = 0  # a1b1
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        # Make a different slot legal
        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., 5, 5] = 1.0  # some other slot
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_empty_loss_mask_returns_zero(self, cpu_device):
        """No positions to evaluate -> 0.0 by convention."""
        V = 1980
        B, T = 2, 4
        max_ply = 4
        preds = torch.zeros(B, T, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)
        legal_grid = torch.zeros(B, max_ply, 64, dtype=torch.long)
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == 0.0

    def test_partial_legal_rate(self, cpu_device):
        """Exactly K of N positions have legal argmax -> rate == K/N.

        Every position has ≥1 legal move — the 3 "illegal" ones get a
        different legal move so the row isn't empty (otherwise the
        has_legal_row filter would skip them as end-of-game slots).
        """
        V = 1980
        B, T = 2, 3
        max_ply = 3
        action = 0  # a1b1
        src, dst = self._grid_for_action(action)
        # A different slot that's always non-empty in the "illegal" rows.
        other_src, other_dst = 7, 8
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        # Make the action's slot legal at exactly 3 of 6 positions
        dense[0, 0, src, dst] = 1.0
        dense[0, 1, src, dst] = 1.0
        dense[1, 2, src, dst] = 1.0
        # Populate the other 3 positions with a different legal move so
        # the row isn't empty (wouldn't be skipped) but the argmax is
        # still illegal there.
        dense[0, 2, other_src, other_dst] = 1.0
        dense[1, 0, other_src, other_dst] = 1.0
        dense[1, 1, other_src, other_dst] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(3.0 / 6.0)

    def test_pad_token_predictions_are_not_counted_legal(self, cpu_device):
        """PAD token is outside action range -- always illegal."""
        V = 1980
        B, T = 1, 2
        max_ply = 2
        preds = torch.full((B, T), PAD_TOKEN, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        # Make everything legal -- shouldn't matter, PAD is never counted
        dense = torch.ones(B, max_ply, 64, 64)
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_outcome_token_predictions_are_not_counted_legal(self, cpu_device):
        """Outcome tokens are outside action range."""
        V = 1980
        B, T = 1, 1
        max_ply = 1
        preds = torch.full((B, T), OUTCOME_TOKEN_BASE, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.ones(B, max_ply, 64, 64)
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_game_length_clips_positions(self, cpu_device):
        """Positions beyond game_length are not counted."""
        V = 1980
        B, T = 1, 4
        max_ply = 4
        action = 0
        src, dst = self._grid_for_action(action)
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., src, dst] = 1.0
        legal_grid = _pack_grid(dense)

        # loss_mask with only position 0 True (simulates game_length=1)
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        loss_mask[0, 0] = True
        game_lengths = torch.ones(B, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(1.0)

    def test_action_resolves_via_grid_lookup(self, cpu_device):
        """Action tokens use the action_grid_index lookup to find their grid cell."""
        V = 1980
        B, T = 1, 1
        max_ply = 1
        # Pick an arbitrary action and look up its grid index
        action = 100
        src, dst = self._grid_for_action(action)

        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[0, 0, src, dst] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(1.0)

    def test_action_illegal_when_grid_mismatches(self, cpu_device):
        V = 1980
        B, T = 1, 1
        max_ply = 1
        action = 100
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        # Empty legal mask
        legal_grid = torch.zeros(B, max_ply, 64, dtype=torch.long)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(0.0)

    def test_returns_python_float(self, cpu_device):
        """compute_legal_move_rate returns plain float, not a tensor."""
        V = 1980
        B, T = 1, 1
        max_ply = 1
        preds = torch.full((B, T), PAD_TOKEN, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)
        legal_grid = torch.zeros(B, max_ply, 64, dtype=torch.long)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert isinstance(rate, float)

    def test_loss_mask_restricts_evaluation(self, cpu_device):
        """Only positions where loss_mask is True are counted."""
        V = 1980
        B, T = 1, 4
        max_ply = 4
        action = 0
        src, dst = self._grid_for_action(action)
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., src, dst] = 1.0
        legal_grid = _pack_grid(dense)

        # Only position 2 has loss_mask True
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        loss_mask[0, 2] = True
        game_lengths = torch.full((B,), T, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == pytest.approx(1.0)

    def test_n_plies_clips_to_max_ply(self, cpu_device):
        """When T > max_ply, only first max_ply positions are evaluated."""
        V = 1980
        B, T = 1, 4
        max_ply = 2
        action = 0
        src, dst = self._grid_for_action(action)
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        dense[..., src, dst] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        # Only 2 positions counted (max_ply=2), both legal -> 1.0
        assert rate == pytest.approx(1.0)

    def test_empty_legal_row_skipped(self, cpu_device):
        """A row of all-zero legal moves (engine never filled beyond
        game_length) must be skipped, not counted as illegal. Without
        this filter the terminal PAD-prediction slot in every game
        depressed legal_move_rate by ~1/gl."""
        V = 1980
        B, T = 1, 3
        max_ply = 3
        action = 0
        src, dst = self._grid_for_action(action)
        preds = torch.full((B, T), action, dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        # Positions 0 and 1 have action 0 as a legal move; position 2
        # is all-zeros (simulating a past-end-of-game slot).
        dense[0, 0, src, dst] = 1.0
        dense[0, 1, src, dst] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        # 2 scored positions (both legal), 1 skipped (no legal row) → 1.0
        assert rate == pytest.approx(1.0)

    def test_all_empty_legal_rows_returns_zero(self, cpu_device):
        """If every position has an all-zero legal row, nothing is
        scored — returns 0.0 instead of crashing on /0."""
        V = 1980
        B, T = 2, 2
        max_ply = 2
        preds = torch.zeros((B, T), dtype=torch.long)
        logits = _make_logits(B, T, V, preds)
        dense = torch.zeros(B, max_ply, 64, 64)
        legal_grid = _pack_grid(dense)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)
        rate = compute_legal_move_rate(logits, legal_grid, loss_mask, game_lengths)
        assert rate == 0.0

    def test_different_preds_per_batch_position(self, cpu_device):
        """Independently predict legal/illegal in different batch/position combos."""
        V = 1980
        B, T = 2, 2
        max_ply = 2
        # Pick 4 different actions
        actions = [0, 1, 2, 3]
        grids = [self._grid_for_action(a) for a in actions]
        preds = torch.tensor([[actions[0], actions[1]], [actions[2], actions[3]]], dtype=torch.long)
        logits = _make_logits(B, T, V, preds)

        dense = torch.zeros(B, max_ply, 64, 64)
        # Each ply has ≥1 legal move so has_legal_row passes for all
        # positions; the question is whether the predicted action matches.
        # Use a filler slot that's never the predicted action.
        filler_src, filler_dst = 20, 21
        # Batch 0, ply 0: make action 0's slot legal (pred is action 0 → legal)
        dense[0, 0, grids[0][0], grids[0][1]] = 1.0
        # Batch 0, ply 1: only the filler slot is legal (pred is action 1 → illegal)
        dense[0, 1, filler_src, filler_dst] = 1.0
        # Batch 1, ply 0: make action 2's slot legal (pred is action 2 → legal)
        dense[1, 0, grids[2][0], grids[2][1]] = 1.0
        # Batch 1, ply 1: only the filler slot is legal (pred is action 3 → illegal)
        dense[1, 1, filler_src, filler_dst] = 1.0
        legal_grid = _pack_grid(dense)

        loss_mask = torch.ones(B, T, dtype=torch.bool)
        game_lengths = torch.full((B,), T, dtype=torch.long)

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


@pytest.mark.integration
class TestEvalGameCompletionMetrics:
    """Shared helper for pretrain + cotrain forfeit/game-completion stats."""

    def _toy_model(self, max_seq_len: int, cpu_device: str):
        from pawn.config import CLMConfig
        from pawn.model import PAWNCLM
        cfg = CLMConfig.toy()
        cfg.max_seq_len = max_seq_len
        torch.manual_seed(0)
        model = PAWNCLM(cfg).to(cpu_device).eval()
        return model, cfg

    def test_pure_moves_returns_expected_keys(self, cpu_device):
        from pawn.data import create_validation_set
        from pawn.trainer import eval_game_completion_metrics

        model, cfg = self._toy_model(max_seq_len=64, cpu_device=cpu_device)
        val_data = create_validation_set(n_games=8, max_ply=64, seed=42)

        out = eval_game_completion_metrics(
            model, val_data,
            batch_size=4, vocab_size=cfg.vocab_size,
            device=cpu_device, use_amp=False,
        )

        expected = {
            "val/game_completion_rate",
            "val/avg_pct_completion",
            "val/avg_plies_completed",
            "val/min_forfeit_ply",
            "val/max_forfeit_ply",
            "val/median_forfeit_ply",
        }
        assert expected <= out.keys()
        assert 0.0 <= out["val/game_completion_rate"] <= 1.0

    def test_prepend_outcome_runs_without_shape_errors(self, cpu_device):
        """Regression: outcome token at position 0 breaks the old code path
        that reused input_ids as move_ids; the new helper must use val_data
        move_ids and align_legal_to_preds instead."""
        from pawn.data import create_validation_set
        from pawn.trainer import eval_game_completion_metrics

        model, cfg = self._toy_model(max_seq_len=64, cpu_device=cpu_device)
        val_data = create_validation_set(
            n_games=8, max_ply=64, seed=42, prepend_outcome=True,
        )

        out = eval_game_completion_metrics(
            model, val_data,
            batch_size=4, vocab_size=cfg.vocab_size,
            device=cpu_device, use_amp=False,
        )
        assert "val/game_completion_rate" in out
        assert 0.0 <= out["val/game_completion_rate"] <= 1.0

    def test_missing_move_ids_returns_empty(self, cpu_device):
        from pawn.trainer import eval_game_completion_metrics
        model, cfg = self._toy_model(max_seq_len=32, cpu_device=cpu_device)
        # val_data without move_ids / game_lengths -> helper is a no-op.
        val_data = {
            "input_ids": torch.zeros(2, 32, dtype=torch.long),
            "loss_mask": torch.ones(2, 32, dtype=torch.bool),
        }
        out = eval_game_completion_metrics(
            model, val_data,
            batch_size=2, vocab_size=cfg.vocab_size,
            device=cpu_device, use_amp=False,
        )
        assert out == {}
