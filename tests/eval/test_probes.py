"""Tests for pawn/eval_suite/probes.py.

Covers LinearProbe, BatchedLinearProbe, PROBES dict, extract_probe_data,
get_probe_targets, hidden-state extraction, _compute_loss/_compute_accuracy,
train_single_probe, train_all_probes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pawn.eval_suite.probes import (
    BatchedLinearProbe,
    LinearProbe,
    PROBES,
    _compute_accuracy,
    _compute_batched_accuracy,
    _compute_batched_loss_per_layer,
    _compute_loss,
    _compute_mae,
    _extract_all_hidden_states,
    _extract_hidden_states,
    _extract_targets,
    extract_probe_data,
    get_probe_targets,
    train_all_probes,
    train_single_probe,
)


# ---------------------------------------------------------------------------
# PROBES dictionary
# ---------------------------------------------------------------------------


class TestProbesDict:
    @pytest.mark.unit
    def test_has_9_probes(self):
        assert len(PROBES) == 9

    @pytest.mark.unit
    def test_contains_expected_names(self):
        expected = {
            "piece_type", "side_to_move", "is_check", "castling_rights",
            "ep_square", "material_count", "legal_move_count",
            "halfmove_clock", "game_phase",
        }
        assert set(PROBES.keys()) == expected

    @pytest.mark.unit
    def test_each_entry_is_tuple_of_three(self):
        for name, entry in PROBES.items():
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            n_out, loss_type, desc = entry
            assert isinstance(n_out, int)
            assert isinstance(loss_type, str)
            assert isinstance(desc, str)

    @pytest.mark.unit
    def test_loss_types_valid(self):
        valid = {"ce", "ce_per_square", "bce", "mse"}
        for name, (_, loss_type, _) in PROBES.items():
            assert loss_type in valid, f"{name}: unknown loss_type {loss_type}"

    @pytest.mark.unit
    def test_piece_type_uses_ce_per_square_and_13_classes(self):
        n_out, loss_type, _ = PROBES["piece_type"]
        assert loss_type == "ce_per_square"
        assert n_out == 13 * 64  # 13 classes × 64 squares

    @pytest.mark.unit
    def test_side_to_move_is_bce(self):
        n_out, loss_type, _ = PROBES["side_to_move"]
        assert loss_type == "bce"
        assert n_out == 1

    @pytest.mark.unit
    def test_is_check_is_bce(self):
        n_out, loss_type, _ = PROBES["is_check"]
        assert loss_type == "bce"
        assert n_out == 1

    @pytest.mark.unit
    def test_castling_rights_4_outputs(self):
        n_out, loss_type, _ = PROBES["castling_rights"]
        assert n_out == 4
        assert loss_type == "bce"

    @pytest.mark.unit
    def test_ep_square_65_classes_ce(self):
        n_out, loss_type, _ = PROBES["ep_square"]
        assert n_out == 65  # 64 squares + none
        assert loss_type == "ce"

    @pytest.mark.unit
    def test_material_count_is_mse(self):
        n_out, loss_type, _ = PROBES["material_count"]
        assert n_out == 10
        assert loss_type == "mse"

    @pytest.mark.unit
    def test_game_phase_ce_3_classes(self):
        n_out, loss_type, _ = PROBES["game_phase"]
        assert n_out == 3
        assert loss_type == "ce"


# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------


class TestLinearProbe:
    @pytest.mark.unit
    def test_forward_shape(self):
        probe = LinearProbe(d_model=32, n_outputs=5)
        x = torch.randn(4, 32)
        out = probe(x)
        assert out.shape == (4, 5)

    @pytest.mark.unit
    def test_different_batch_shapes(self):
        probe = LinearProbe(d_model=16, n_outputs=3)
        x = torch.randn(2, 8, 16)
        out = probe(x)
        assert out.shape == (2, 8, 3)

    @pytest.mark.unit
    def test_has_trainable_params(self):
        probe = LinearProbe(d_model=16, n_outputs=4)
        params = list(probe.parameters())
        assert len(params) == 2  # weight + bias
        assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# BatchedLinearProbe
# ---------------------------------------------------------------------------


class TestBatchedLinearProbe:
    @pytest.mark.unit
    def test_forward_shape(self):
        probe = BatchedLinearProbe(n_probes=3, d_model=8, n_outputs=4)
        x = torch.randn(3, 5, 8)  # (L, B, d_model)
        out = probe(x)
        assert out.shape == (3, 5, 4)  # (L, B, n_outputs)

    @pytest.mark.unit
    def test_independent_probes(self):
        """Each layer's probe should produce distinct outputs for same input."""
        probe = BatchedLinearProbe(n_probes=2, d_model=4, n_outputs=3)
        # Same input replicated across layers
        x = torch.ones(2, 1, 4)
        out = probe(x)
        # Weights are randomly initialized, so the two rows should differ
        assert not torch.allclose(out[0, 0], out[1, 0])

    @pytest.mark.unit
    def test_has_trainable_params(self):
        probe = BatchedLinearProbe(n_probes=3, d_model=8, n_outputs=4)
        params = list(probe.parameters())
        assert len(params) == 2
        assert params[0].shape == (3, 8, 4)  # weight
        assert params[1].shape == (3, 1, 4)  # bias

    @pytest.mark.unit
    def test_bias_initialized_zero(self):
        probe = BatchedLinearProbe(n_probes=2, d_model=4, n_outputs=3)
        assert torch.all(probe.bias == 0)


# ---------------------------------------------------------------------------
# _compute_loss / _compute_accuracy
# ---------------------------------------------------------------------------


class TestComputeLoss:
    @pytest.mark.unit
    def test_ce_returns_scalar(self):
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = _compute_loss(logits, targets, "ce", 5)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.unit
    def test_bce_returns_scalar(self):
        logits = torch.randn(4, 1)
        targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        loss = _compute_loss(logits, targets, "bce", 1)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.unit
    def test_mse_returns_scalar(self):
        logits = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        loss = _compute_loss(logits, targets, "mse", 3)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.unit
    def test_ce_per_square_returns_scalar(self):
        # n_outputs=13*64, reshape to (B*64, 13)
        logits = torch.randn(4, 13 * 64)
        targets = torch.randint(0, 13, (4, 64))
        loss = _compute_loss(logits, targets, "ce_per_square", 13 * 64)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.unit
    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            _compute_loss(torch.randn(2, 2), torch.zeros(2, 2), "bogus", 2)


class TestComputeAccuracy:
    @pytest.mark.unit
    def test_ce_accuracy_perfect(self):
        # Logits that strongly prefer the target class
        targets = torch.tensor([0, 1, 2])
        logits = torch.eye(3) * 10
        acc = _compute_accuracy(logits, targets, "ce", 3)
        assert acc == pytest.approx(1.0)

    @pytest.mark.unit
    def test_bce_accuracy(self):
        targets = torch.tensor([[1.0], [0.0]])
        logits = torch.tensor([[5.0], [-5.0]])
        acc = _compute_accuracy(logits, targets, "bce", 1)
        assert acc == pytest.approx(1.0)

    @pytest.mark.unit
    def test_mse_returns_r2(self):
        targets = torch.tensor([[1.0], [2.0], [3.0]])
        acc = _compute_accuracy(targets, targets, "mse", 1)
        # Perfect fit: R² = 1
        assert acc == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.unit
    def test_ce_per_square_accuracy(self):
        # Predict all zeros correctly
        targets = torch.zeros(2, 64, dtype=torch.long)
        # Logits strongly favor class 0 at every square
        logits = torch.zeros(2, 64, 13)
        logits[..., 0] = 10.0
        logits = logits.reshape(2, -1)
        acc = _compute_accuracy(logits, targets, "ce_per_square", 13 * 64)
        assert acc == pytest.approx(1.0)

    @pytest.mark.unit
    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            _compute_accuracy(torch.randn(2, 2), torch.zeros(2, 2), "bogus", 2)


class TestComputeMae:
    @pytest.mark.unit
    def test_zero_mae_for_perfect_fit(self):
        targets = torch.tensor([[1.0], [2.0]])
        mae = _compute_mae(targets, targets)
        assert mae == pytest.approx(0.0)

    @pytest.mark.unit
    def test_mae_positive(self):
        logits = torch.tensor([[1.0], [2.0]])
        targets = torch.tensor([[2.0], [3.0]])
        mae = _compute_mae(logits, targets)
        assert mae == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_batched_loss_per_layer / _compute_batched_accuracy
# ---------------------------------------------------------------------------


class TestBatchedLossAndAcc:
    @pytest.mark.unit
    def test_batched_ce_loss_shape(self):
        L, B, n_out = 3, 4, 5
        logits = torch.randn(L, B, n_out)
        targets = torch.randint(0, n_out, (B,))
        per_layer = _compute_batched_loss_per_layer(logits, targets, "ce", n_out)
        assert per_layer.shape == (L,)

    @pytest.mark.unit
    def test_batched_bce_loss_shape(self):
        L, B, n_out = 3, 4, 2
        logits = torch.randn(L, B, n_out)
        targets = torch.rand(B, n_out).round()  # 0s and 1s
        per_layer = _compute_batched_loss_per_layer(logits, targets, "bce", n_out)
        assert per_layer.shape == (L,)

    @pytest.mark.unit
    def test_batched_mse_loss_shape(self):
        L, B, n_out = 3, 4, 5
        logits = torch.randn(L, B, n_out)
        targets = torch.randn(B, n_out)
        per_layer = _compute_batched_loss_per_layer(logits, targets, "mse", n_out)
        assert per_layer.shape == (L,)

    @pytest.mark.unit
    def test_batched_ce_per_square(self):
        L, B = 2, 3
        logits = torch.randn(L, B, 13 * 64)
        targets = torch.randint(0, 13, (B, 64))
        per_layer = _compute_batched_loss_per_layer(logits, targets, "ce_per_square", 13 * 64)
        assert per_layer.shape == (L,)

    @pytest.mark.unit
    def test_batched_ce_accuracy_perfect(self):
        L, B, n_out = 2, 3, 4
        targets = torch.tensor([0, 1, 2])
        # Build logits that strongly prefer the targets
        logits = torch.zeros(L, B, n_out)
        for b in range(B):
            logits[:, b, targets[b]] = 10.0
        acc = _compute_batched_accuracy(logits, targets, "ce", n_out)
        assert acc.shape == (L,)
        assert torch.allclose(acc, torch.ones(L))

    @pytest.mark.unit
    def test_batched_bce_accuracy_perfect(self):
        L, B, n_out = 2, 3, 2
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # Logits: match targets
        logits = torch.where(targets.bool(), torch.tensor(5.0), torch.tensor(-5.0))
        logits = logits.unsqueeze(0).expand(L, -1, -1).clone()
        acc = _compute_batched_accuracy(logits, targets, "bce", n_out)
        assert acc.shape == (L,)
        assert torch.allclose(acc, torch.ones(L))

    @pytest.mark.unit
    def test_batched_mse_accuracy_raises_nie(self):
        L, B, n_out = 2, 3, 4
        logits = torch.randn(L, B, n_out)
        targets = torch.randn(B, n_out)
        with pytest.raises(NotImplementedError):
            _compute_batched_accuracy(logits, targets, "mse", n_out)


# ---------------------------------------------------------------------------
# extract_probe_data + get_probe_targets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe_data():
    """Tiny probe data: 8 games, max 24 ply."""
    return extract_probe_data(n_games=8, max_ply=24, seed=42)


class TestExtractProbeData:
    @pytest.mark.unit
    def test_returns_expected_keys(self, probe_data: dict):
        for key in ("input_ids", "loss_mask", "boards", "side_to_move",
                    "castling_rights", "ep_square", "is_check",
                    "halfmove_clock", "game_lengths", "legal_move_counts"):
            assert key in probe_data, f"missing key: {key}"

    @pytest.mark.unit
    def test_input_ids_shape(self, probe_data: dict):
        # input_ids has 1 outcome + max_ply moves + PAD
        assert probe_data["input_ids"].ndim == 2
        assert probe_data["input_ids"].shape[0] == 8

    @pytest.mark.unit
    def test_boards_shape(self, probe_data: dict):
        # boards: (n_games, max_ply, 8, 8)
        assert probe_data["boards"].ndim == 4
        assert probe_data["boards"].shape[0] == 8
        assert probe_data["boards"].shape[2] == 8
        assert probe_data["boards"].shape[3] == 8

    @pytest.mark.unit
    def test_boards_valid_piece_ids(self, probe_data: dict):
        # Piece ids in [0, 12]
        b = probe_data["boards"]
        assert int(b.min()) >= 0
        assert int(b.max()) <= 12

    @pytest.mark.unit
    def test_side_to_move_values_0_or_1(self, probe_data: dict):
        s = probe_data["side_to_move"]
        unique = torch.unique(s)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    @pytest.mark.unit
    def test_is_check_values_0_or_1(self, probe_data: dict):
        c = probe_data["is_check"]
        unique = torch.unique(c)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    @pytest.mark.unit
    def test_game_lengths_positive(self, probe_data: dict):
        gl = probe_data["game_lengths"]
        assert (gl > 0).all()

    @pytest.mark.unit
    def test_without_legal_counts(self):
        data = extract_probe_data(n_games=4, max_ply=16, seed=42, include_legal_counts=False)
        assert "legal_move_counts" not in data

    @pytest.mark.unit
    def test_halfmove_clock_nonneg(self, probe_data: dict):
        assert (probe_data["halfmove_clock"] >= 0).all()


class TestGetProbeTargets:
    @pytest.mark.unit
    def test_piece_type_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1, 0])
        game_idx = torch.tensor([0, 1, 2])
        t = get_probe_targets("piece_type", probe_data, ply_idx, game_idx)
        assert t.shape == (3, 64)

    @pytest.mark.unit
    def test_piece_type_values_in_0_12(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1])
        game_idx = torch.tensor([0, 1])
        t = get_probe_targets("piece_type", probe_data, ply_idx, game_idx)
        assert int(t.min()) >= 0
        assert int(t.max()) <= 12

    @pytest.mark.unit
    def test_side_to_move_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1, 2])
        game_idx = torch.tensor([0, 1, 2])
        t = get_probe_targets("side_to_move", probe_data, ply_idx, game_idx)
        assert t.shape == (3, 1)

    @pytest.mark.unit
    def test_is_check_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1])
        game_idx = torch.tensor([0, 1])
        t = get_probe_targets("is_check", probe_data, ply_idx, game_idx)
        assert t.shape == (2, 1)
        # Values 0 or 1
        assert set(t.flatten().tolist()).issubset({0.0, 1.0})

    @pytest.mark.unit
    def test_castling_rights_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1])
        game_idx = torch.tensor([0, 1])
        t = get_probe_targets("castling_rights", probe_data, ply_idx, game_idx)
        assert t.shape == (2, 4)
        # 4 binary flags
        assert set(t.flatten().tolist()).issubset({0.0, 1.0})

    @pytest.mark.unit
    def test_castling_rights_initial_all_set(self, probe_data: dict):
        # At ply 0, all 4 castling flags are set
        ply_idx = torch.tensor([0])
        game_idx = torch.tensor([0])
        t = get_probe_targets("castling_rights", probe_data, ply_idx, game_idx)
        assert torch.allclose(t, torch.ones(1, 4))

    @pytest.mark.unit
    def test_ep_square_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1, 2])
        game_idx = torch.tensor([0, 1, 2])
        t = get_probe_targets("ep_square", probe_data, ply_idx, game_idx)
        assert t.shape == (3,)
        # Values in 0..64
        assert int(t.min()) >= 0
        assert int(t.max()) <= 64

    @pytest.mark.unit
    def test_material_count_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0])
        game_idx = torch.tensor([0])
        t = get_probe_targets("material_count", probe_data, ply_idx, game_idx)
        assert t.shape == (1, 10)

    @pytest.mark.unit
    def test_material_count_initial_position(self, probe_data: dict):
        # At ply 0, each color has P:8, N:2, B:2, R:2, Q:1
        ply_idx = torch.tensor([0])
        game_idx = torch.tensor([0])
        t = get_probe_targets("material_count", probe_data, ply_idx, game_idx)
        expected = torch.tensor([[8.0, 2, 2, 2, 1, 8, 2, 2, 2, 1]])
        assert torch.allclose(t, expected)

    @pytest.mark.unit
    def test_legal_move_count_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1])
        game_idx = torch.tensor([0, 1])
        t = get_probe_targets("legal_move_count", probe_data, ply_idx, game_idx)
        assert t.shape == (2, 1)

    @pytest.mark.unit
    def test_legal_move_count_initial_is_20(self, probe_data: dict):
        ply_idx = torch.tensor([0])
        game_idx = torch.tensor([0])
        t = get_probe_targets("legal_move_count", probe_data, ply_idx, game_idx)
        assert t.item() == 20.0

    @pytest.mark.unit
    def test_halfmove_clock_shape(self, probe_data: dict):
        ply_idx = torch.tensor([0, 1])
        game_idx = torch.tensor([0, 1])
        t = get_probe_targets("halfmove_clock", probe_data, ply_idx, game_idx)
        assert t.shape == (2, 1)

    @pytest.mark.unit
    def test_game_phase_shape_and_values(self, probe_data: dict):
        ply_idx = torch.tensor([0, 5, 10])
        game_idx = torch.tensor([0, 1, 2])
        t = get_probe_targets("game_phase", probe_data, ply_idx, game_idx)
        assert t.shape == (3,)
        # Values in {0, 1, 2}
        assert int(t.min()) >= 0
        assert int(t.max()) <= 2

    @pytest.mark.unit
    def test_game_phase_ply0_is_opening(self, probe_data: dict):
        # Initial position: all pieces present, so opening
        ply_idx = torch.tensor([0])
        game_idx = torch.tensor([0])
        t = get_probe_targets("game_phase", probe_data, ply_idx, game_idx)
        assert t.item() == 0

    @pytest.mark.unit
    def test_unknown_probe_raises(self, probe_data: dict):
        with pytest.raises(ValueError, match="Unknown probe"):
            get_probe_targets("bogus", probe_data,
                              torch.tensor([0]), torch.tensor([0]))

    @pytest.mark.unit
    def test_sequential_game_indices_when_none(self, probe_data: dict):
        # If game_indices is None, sequential is assumed
        ply_idx = torch.tensor([0, 1, 2])
        t_none = get_probe_targets("side_to_move", probe_data, ply_idx, None)
        t_explicit = get_probe_targets(
            "side_to_move", probe_data, ply_idx, torch.arange(3),
        )
        assert torch.equal(t_none, t_explicit)


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------


class TestExtractHiddenStates:
    @pytest.mark.unit
    def test_single_layer_shape(self, probe_data, toy_model, cpu_device):
        h, mask = _extract_hidden_states(
            toy_model, probe_data, cpu_device, layer_idx=0, max_batch=4,
        )
        d_model = toy_model.cfg.d_model
        assert h.shape[1] == d_model
        assert h.shape[0] == int(mask.sum())

    @pytest.mark.unit
    def test_last_layer_extraction(self, probe_data, toy_model, cpu_device):
        h, mask = _extract_hidden_states(
            toy_model, probe_data, cpu_device,
            layer_idx=toy_model.cfg.n_layers, max_batch=4,
        )
        d_model = toy_model.cfg.d_model
        assert h.shape[1] == d_model

    @pytest.mark.unit
    def test_all_layers_shape(self, probe_data, toy_model, cpu_device):
        all_h, mask = _extract_all_hidden_states(
            toy_model, probe_data, cpu_device, max_batch=4,
        )
        # embed + n_layers
        assert len(all_h) == toy_model.cfg.n_layers + 1
        d_model = toy_model.cfg.d_model
        for h in all_h:
            assert h.shape[1] == d_model
            assert h.shape[0] == int(mask.sum())

    @pytest.mark.unit
    def test_valid_mask_matches_game_lengths(self, probe_data, toy_model, cpu_device):
        _, mask = _extract_hidden_states(
            toy_model, probe_data, cpu_device, layer_idx=0, max_batch=4,
        )
        # Sum per game should equal game_lengths (clipped to max_ply)
        game_lengths = probe_data["game_lengths"]
        max_ply = probe_data["boards"].shape[1]
        expected = np.minimum(game_lengths, max_ply)
        per_game = mask.sum(dim=1).numpy()
        assert np.array_equal(per_game, expected)


class TestExtractTargets:
    @pytest.mark.unit
    def test_piece_type_targets(self, probe_data):
        _, mask = _extract_hidden_states(
            None, probe_data, "cpu", layer_idx=0, max_batch=4,
        ) if False else (None, None)
        # Build mask directly
        game_lengths = probe_data["game_lengths"]
        max_ply = probe_data["boards"].shape[1]
        ply_grid = torch.arange(max_ply).unsqueeze(0)
        valid_mask = ply_grid < torch.from_numpy(game_lengths).long().unsqueeze(1)

        t = _extract_targets("piece_type", probe_data, valid_mask)
        assert t.shape[1] == 64
        assert t.shape[0] == int(valid_mask.sum())


# ---------------------------------------------------------------------------
# train_single_probe / train_all_probes (integration, very small)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_val_data():
    """Two small datasets for train/val split."""
    train = extract_probe_data(n_games=16, max_ply=24, seed=1)
    val = extract_probe_data(n_games=8, max_ply=24, seed=2)
    return train, val


class TestTrainSingleProbe:
    @pytest.mark.unit
    def test_returns_expected_keys(self, toy_model, cpu_device, train_val_data):
        train, val = train_val_data
        torch.manual_seed(0)
        result = train_single_probe(
            toy_model, "is_check", train, val, cpu_device,
            n_epochs=2, layer_idx=0, batch_size=32,
        )
        assert "accuracy" in result
        assert "loss" in result
        assert "best_accuracy" in result
        assert "n_train" in result
        assert "n_val" in result

    @pytest.mark.unit
    def test_accuracy_in_range(self, toy_model, cpu_device, train_val_data):
        train, val = train_val_data
        torch.manual_seed(0)
        result = train_single_probe(
            toy_model, "side_to_move", train, val, cpu_device,
            n_epochs=2, layer_idx=0, batch_size=32,
        )
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["loss"] >= 0.0

    @pytest.mark.unit
    def test_mse_probe_includes_mae(self, toy_model, cpu_device, train_val_data):
        train, val = train_val_data
        torch.manual_seed(0)
        result = train_single_probe(
            toy_model, "halfmove_clock", train, val, cpu_device,
            n_epochs=2, layer_idx=0, batch_size=32,
        )
        assert "mae" in result
        assert result["mae"] >= 0.0


class TestTrainAllProbes:
    @pytest.mark.unit
    def test_returns_nested_dict_per_layer(self, toy_model, cpu_device, train_val_data):
        train, val = train_val_data
        torch.manual_seed(0)
        # Only a subset of probes to keep fast
        result = train_all_probes(
            toy_model, train, val, cpu_device,
            per_layer=True, n_epochs=1, verbose=False,
            probe_names=["is_check", "side_to_move"],
        )
        assert "is_check" in result
        assert "side_to_move" in result
        # per_layer=True: embed + n_layers = n_layers + 1 entries
        n_expected = toy_model.cfg.n_layers + 1
        assert len(result["is_check"]) == n_expected
        assert "embed" in result["is_check"]
        assert "layer_0" in result["is_check"]

    @pytest.mark.unit
    def test_top_layer_only(self, toy_model, cpu_device, train_val_data):
        train, val = train_val_data
        torch.manual_seed(0)
        result = train_all_probes(
            toy_model, train, val, cpu_device,
            per_layer=False, n_epochs=1, verbose=False,
            probe_names=["is_check"],
        )
        assert "is_check" in result
        assert len(result["is_check"]) == 1
        last_layer_name = f"layer_{toy_model.cfg.n_layers - 1}"
        assert last_layer_name in result["is_check"]

    @pytest.mark.unit
    def test_result_entry_has_metrics(self, toy_model, cpu_device, train_val_data):
        train, val = train_val_data
        torch.manual_seed(0)
        result = train_all_probes(
            toy_model, train, val, cpu_device,
            per_layer=True, n_epochs=1, verbose=False,
            probe_names=["side_to_move"],
        )
        entry = result["side_to_move"]["embed"]
        assert "accuracy" in entry
        assert "loss" in entry
        assert "best_accuracy" in entry
        assert "n_train" in entry
        assert "n_val" in entry

    @pytest.mark.unit
    def test_unknown_probe_names_ignored(self, toy_model, cpu_device, train_val_data):
        train, val = train_val_data
        torch.manual_seed(0)
        result = train_all_probes(
            toy_model, train, val, cpu_device,
            per_layer=False, n_epochs=1, verbose=False,
            probe_names=["is_check", "bogus_probe_xyz"],
        )
        # Only known probes should appear
        assert "is_check" in result
        assert "bogus_probe_xyz" not in result
