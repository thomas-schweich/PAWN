"""Tests for FiLM (Feature-wise Linear Modulation) adapter."""

from __future__ import annotations

import pytest
import torch

from pawn.adapters.film import FiLMLayer, FiLMCLM


class TestFiLMLayer:
    def test_gamma_init_ones(self):
        layer = FiLMLayer(dim=32)
        assert torch.allclose(layer.gamma, torch.ones(32))

    def test_beta_init_zeros(self):
        layer = FiLMLayer(dim=32)
        assert torch.allclose(layer.beta, torch.zeros(32))

    def test_identity_forward_at_init(self):
        layer = FiLMLayer(dim=16)
        x = torch.randn(2, 8, 16)
        y = layer(x)
        torch.testing.assert_close(y, x)

    def test_forward_applies_gamma_beta(self):
        layer = FiLMLayer(dim=4)
        layer.gamma.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        layer.beta.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
        x = torch.zeros(2, 3, 4)
        y = layer(x)
        expected = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).expand(2, 3, 4)
        torch.testing.assert_close(y, expected)

    def test_broadcasts_over_batch_and_time(self):
        layer = FiLMLayer(dim=4)
        layer.gamma.data = torch.tensor([2.0, 2.0, 2.0, 2.0])
        layer.beta.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
        x = torch.ones(5, 7, 4)
        y = layer(x)
        expected = torch.full((5, 7, 4), 3.0)  # 2*1 + 1 = 3
        torch.testing.assert_close(y, expected)

    def test_param_count(self):
        layer = FiLMLayer(dim=32)
        assert sum(p.numel() for p in layer.parameters()) == 2 * 32

    def test_param_shapes(self):
        layer = FiLMLayer(dim=64)
        assert layer.gamma.shape == (64,)
        assert layer.beta.shape == (64,)


class TestFiLMCLMInvariants:
    def test_backbone_frozen(self, toy_backbone):
        model = FiLMCLM(toy_backbone)
        for name, p in model.backbone.named_parameters():
            assert not p.requires_grad, f"backbone param {name} not frozen"

    def test_film_params_trainable(self, toy_backbone):
        model = FiLMCLM(toy_backbone)
        for film in model.hidden_films:
            assert film.gamma.requires_grad
            assert film.beta.requires_grad
        if model.output_film is not None:
            assert model.output_film.gamma.requires_grad
            assert model.output_film.beta.requires_grad

    def test_zero_init_identity(self, toy_backbone, toy_input_ids, toy_attention_mask):
        with torch.no_grad():
            bare, _ = toy_backbone(toy_input_ids, toy_attention_mask)
        model = FiLMCLM(toy_backbone)
        with torch.no_grad():
            out = model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_zero_init_identity_no_mask(self, toy_backbone, toy_input_ids):
        with torch.no_grad():
            bare, _ = toy_backbone(
                toy_input_ids, torch.ones_like(toy_input_ids, dtype=torch.bool),
            )
        model = FiLMCLM(toy_backbone)
        with torch.no_grad():
            out = model(toy_input_ids)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_forward_hidden_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = FiLMCLM(toy_backbone)
        hidden = model.forward_hidden(toy_input_ids)
        B, T = toy_input_ids.shape
        assert hidden.shape == (B, T, toy_clm_config.d_model)

    def test_project_head_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = FiLMCLM(toy_backbone)
        hidden = model.forward_hidden(toy_input_ids)
        logits = model.project_head(hidden)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_full_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = FiLMCLM(toy_backbone)
        logits = model(toy_input_ids)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_generate(self, toy_backbone, toy_clm_config):
        model = FiLMCLM(toy_backbone)
        ids = torch.randint(0, toy_clm_config.vocab_size, (1, 5))
        logits, kv_cache = model.forward_generate(ids)
        assert logits.shape == (1, 1, toy_clm_config.vocab_size)
        assert len(kv_cache) == toy_clm_config.n_layers
        nxt = torch.randint(0, toy_clm_config.vocab_size, (1, 1))
        logits2, _ = model.forward_generate(nxt, kv_cache)
        assert logits2.shape == (1, 1, toy_clm_config.vocab_size)

    def test_gradient_flow(self, toy_backbone, toy_input_ids, toy_clm_config):
        model = FiLMCLM(toy_backbone)
        targets = torch.randint(0, toy_clm_config.vocab_size, toy_input_ids.shape)
        logits = model(toy_input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
        )
        loss.backward()
        # Backbone params have no grad
        for p in model.backbone.parameters():
            assert p.grad is None
        # FiLM params have grads
        for film in model.hidden_films:
            assert film.gamma.grad is not None
            assert film.beta.grad is not None
        if model.output_film is not None:
            assert model.output_film.gamma.grad is not None
            assert model.output_film.beta.grad is not None

    def test_state_dict_roundtrip(self, toy_backbone, toy_backbone_fresh,
                                   toy_input_ids, toy_attention_mask):
        model = FiLMCLM(toy_backbone)
        # Perturb
        for film in model.hidden_films:
            film.gamma.data = film.gamma.data + torch.randn_like(film.gamma) * 0.01
            film.beta.data = film.beta.data + torch.randn_like(film.beta) * 0.01
        if model.output_film is not None:
            model.output_film.gamma.data += torch.randn_like(model.output_film.gamma) * 0.01
            model.output_film.beta.data += torch.randn_like(model.output_film.beta) * 0.01
        state = model.film_state_dict()
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(state.keys()) == expected

        fresh = FiLMCLM(toy_backbone_fresh)
        fresh.load_film_state_dict(state)
        with torch.no_grad():
            o1 = model(toy_input_ids, toy_attention_mask)
            o2 = fresh(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-6)

    def test_film_parameters_trainable_only(self, toy_backbone):
        model = FiLMCLM(toy_backbone)
        params = model.film_parameters()
        for p in params:
            assert p.requires_grad
        ids = {id(p) for p in params}
        assert len(ids) == len(params)


class TestFiLMCLMConfig:
    def test_use_output_film_true(self, toy_backbone, toy_clm_config):
        model = FiLMCLM(toy_backbone, use_output_film=True)
        assert model.output_film is not None
        # expect 2 layers × 2 × d_model + 2 × vocab_size
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected = (
            toy_clm_config.n_layers * 2 * toy_clm_config.d_model
            + 2 * toy_clm_config.vocab_size
        )
        assert total == expected

    def test_use_output_film_false(self, toy_backbone, toy_clm_config):
        model = FiLMCLM(toy_backbone, use_output_film=False)
        assert model.output_film is None
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected = toy_clm_config.n_layers * 2 * toy_clm_config.d_model
        assert total == expected

    def test_no_output_film_still_identity(self, toy_backbone, toy_input_ids,
                                             toy_attention_mask):
        with torch.no_grad():
            bare, _ = toy_backbone(toy_input_ids, toy_attention_mask)
        model = FiLMCLM(toy_backbone, use_output_film=False)
        with torch.no_grad():
            out = model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_weight_report_init_values(self, toy_backbone):
        model = FiLMCLM(toy_backbone)
        rep = model.film_weight_report()
        # At init: all gamma_dev and beta_norm values should be 0
        for k, v in rep.items():
            assert v == 0.0, f"{k} should be zero at init, got {v}"

    def test_weight_report_after_perturb(self, toy_backbone):
        model = FiLMCLM(toy_backbone)
        for film in model.hidden_films:
            film.gamma.data += 0.5
        rep = model.film_weight_report()
        for k, v in rep.items():
            if "gamma_dev" in k and "hidden" in k:
                assert v > 0.0

    def test_cfg_property(self, toy_backbone, toy_clm_config):
        model = FiLMCLM(toy_backbone)
        assert model.cfg is toy_clm_config

    def test_hidden_films_per_layer(self, toy_backbone, toy_clm_config):
        model = FiLMCLM(toy_backbone)
        assert len(model.hidden_films) == toy_clm_config.n_layers
        for film in model.hidden_films:
            assert isinstance(film, FiLMLayer)
            assert film.gamma.shape == (toy_clm_config.d_model,)
