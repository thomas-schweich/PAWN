"""Tests for PAWN model."""

import torch
import pytest

import chess_engine as engine

from pawn.config import CLMConfig, TrainingConfig, WHITE_CHECKMATES, PLY_LIMIT
from pawn.model import PAWNCLM, clm_loss
from pawn.data import CLMDataset, _to_clm_batch, _map_termination_to_outcome


class TestCLMConfig:
    def test_default_config(self):
        cfg = CLMConfig()
        assert cfg.vocab_size == CLMConfig().vocab_size
        assert cfg.max_seq_len == 256
        assert cfg.d_model == 512
        assert cfg.n_layers == 8

    def test_toy_config(self):
        cfg = CLMConfig.toy()
        assert cfg.d_model == 64
        assert cfg.n_layers == 2


class TestPAWNCLM:
    @pytest.fixture
    def toy_model(self):
        return PAWNCLM(CLMConfig.toy())

    @pytest.fixture
    def full_model(self):
        return PAWNCLM(CLMConfig())

    def test_forward_shapes_toy(self, toy_model):
        B, T = 4, 256
        input_ids = torch.randint(0, CLMConfig().vocab_size, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        logits, layer_outputs = toy_model(input_ids, mask)

        assert logits.shape == (B, T, CLMConfig().vocab_size)
        # embed + 2 layers = 3 outputs
        assert len(layer_outputs) == 3
        for lo in layer_outputs:
            assert lo.shape == (B, T, 64)

    def test_forward_shapes_full(self, full_model):
        B, T = 2, 32
        input_ids = torch.randint(0, CLMConfig().vocab_size, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        logits, layer_outputs = full_model(input_ids, mask)

        assert logits.shape == (B, T, CLMConfig().vocab_size)
        assert len(layer_outputs) == 9  # embed + 8 layers

    def test_hidden_only(self, toy_model):
        B, T = 4, 256
        input_ids = torch.randint(0, CLMConfig().vocab_size, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        logits, layer_outputs = toy_model(input_ids, mask, hidden_only=True)

        assert logits.shape == (B, T, CLMConfig().vocab_size)
        assert len(layer_outputs) == 1  # only final hidden

    def test_param_count_toy(self, toy_model):
        n = sum(p.numel() for p in toy_model.parameters())
        assert n < 1_000_000  # toy should be small

    def test_param_count_full(self, full_model):
        n = sum(p.numel() for p in full_model.parameters())
        # ~30M encoder + ~2.2M output head
        assert 25_000_000 < n < 45_000_000

    def test_padding_mask(self, toy_model):
        B, T = 4, 256
        input_ids = torch.randint(1, 4273, (B, T))
        input_ids[:, 100:] = 0  # pad after position 100
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :101] = True

        logits, _ = toy_model(input_ids, mask)

        assert not torch.isnan(logits).any()

    def test_outcome_token_embedding(self, toy_model):
        """Verify outcome tokens get standalone embeddings, not factored."""
        B = 2
        # All outcome tokens
        input_ids = torch.tensor([[WHITE_CHECKMATES] * 256, [PLY_LIMIT] * 256])
        mask = torch.ones(B, 256, dtype=torch.bool)

        logits, _ = toy_model(input_ids, mask)
        assert not torch.isnan(logits).any()
        # Different outcome tokens should produce different embeddings
        emb1 = toy_model.embed(input_ids[:1])
        emb2 = toy_model.embed(input_ids[1:])
        assert not torch.allclose(emb1, emb2)


class TestCLMLoss:
    def test_loss_computation(self):
        B, T, V = 4, 32, CLMConfig().vocab_size
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        loss_mask[:, 20:] = False

        loss, metrics = clm_loss(logits, targets, loss_mask)

        assert loss.shape == ()
        assert loss.item() > 0
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_gradient_flow(self):
        model = PAWNCLM(CLMConfig.toy())
        B, T = 4, 32
        input_ids = torch.randint(0, CLMConfig().vocab_size, (B, T))
        targets = torch.randint(0, CLMConfig().vocab_size, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        logits, _ = model(input_ids, mask)
        loss, _ = clm_loss(logits, targets, mask)
        loss.backward()

        grad_params = [p for p in model.parameters() if p.grad is not None]
        assert len(grad_params) > 0


class TestDataPipeline:
    def test_termination_mapping(self):
        import numpy as np

        # Code 0 = checkmate, odd game_length = white checkmates
        term_codes = np.array([0, 0, 1, 2, 3, 4, 5], dtype=np.uint8)
        game_lengths = np.array([11, 12, 50, 100, 80, 60, 255], dtype=np.int16)

        outcomes = _map_termination_to_outcome(term_codes, game_lengths)

        assert outcomes[0].item() == WHITE_CHECKMATES   # odd length
        assert outcomes[1].item() == 4274                # even length = BLACK_CHECKMATES
        assert outcomes[2].item() == 4275                # STALEMATE
        assert outcomes[3].item() == 4276                # DRAW_BY_RULE
        assert outcomes[4].item() == 4276                # DRAW_BY_RULE
        assert outcomes[5].item() == 4276                # DRAW_BY_RULE
        assert outcomes[6].item() == PLY_LIMIT

    def test_clm_batch_shapes(self):
        seq_len = 256
        engine_max_ply = seq_len - 1  # 255
        move_ids, game_lengths, term_codes = engine.generate_random_games(
            32, engine_max_ply, seed=42
        )
        batch = _to_clm_batch(move_ids, game_lengths, term_codes, seq_len)

        assert batch["input_ids"].shape == (32, 256)
        assert batch["targets"].shape == (32, 256)
        assert batch["loss_mask"].shape == (32, 256)
        assert batch["loss_mask"].dtype == torch.bool

    def test_clm_batch_content(self):
        seq_len = 256
        engine_max_ply = seq_len - 1
        move_ids, game_lengths, term_codes = engine.generate_random_games(
            8, engine_max_ply, seed=42
        )
        batch = _to_clm_batch(move_ids, game_lengths, term_codes, seq_len)

        for b in range(8):
            gl = min(int(game_lengths[b]), seq_len - 1)

            # Position 0 should be an outcome token
            assert batch["input_ids"][b, 0].item() >= 4273

            # Positions 1..gl should be actual moves (1-4272)
            for p in range(1, gl + 1):
                tok = batch["input_ids"][b, p].item()
                assert 1 <= tok <= 4272, f"Expected move token at pos {p}, got {tok}"

            # Positions after gl should be PAD
            if gl + 1 < seq_len:
                assert batch["input_ids"][b, gl + 1].item() == 0

            # Target at position gl should be PAD (end of game)
            assert batch["targets"][b, gl].item() == 0

            # Loss mask should be True for 0..gl, False after
            assert batch["loss_mask"][b, gl].item() is True
            if gl + 1 < seq_len:
                assert batch["loss_mask"][b, gl + 1].item() is False

    def test_dataset_yields_batches(self):
        ds = CLMDataset(batch_size=8, max_ply=256, base_seed=42)
        it = iter(ds)
        batch = next(it)

        assert "input_ids" in batch
        assert "targets" in batch
        assert "loss_mask" in batch
        assert batch["input_ids"].shape[0] == 8

    def test_single_step_overfit(self):
        """Verify model can overfit a single batch (gradients flow correctly)."""
        cfg = CLMConfig.toy()
        model = PAWNCLM(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        move_ids, game_lengths, term_codes = engine.generate_random_games(
            4, 255, seed=42
        )
        batch = _to_clm_batch(move_ids, game_lengths, term_codes, 256)

        initial_loss = None
        for step in range(200):
            logits, _ = model(batch["input_ids"], batch["loss_mask"])
            loss, _ = clm_loss(logits, batch["targets"], batch["loss_mask"])

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.6, (
            f"Loss didn't decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )
