"""Tests for HybridCLM (LoRA + FiLM combined adapter)."""

from __future__ import annotations

import pytest
import torch

from pawn.adapters.hybrid import HybridCLM
from pawn.adapters.lora import LoRALinear
from pawn.adapters.film import FiLMLayer


class TestHybridCLMInvariants:
    def test_backbone_frozen(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4)
        for name, p in model.backbone.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.requires_grad
                continue
            assert not p.requires_grad, f"backbone param {name} not frozen"

    def test_zero_init_identity(self, toy_backbone, toy_input_ids, toy_attention_mask):
        with torch.no_grad():
            bare, _ = toy_backbone(toy_input_ids, toy_attention_mask)
        model = HybridCLM(toy_backbone, lora_rank=4)
        with torch.no_grad():
            out = model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_zero_init_identity_no_mask(self, toy_backbone, toy_input_ids):
        with torch.no_grad():
            bare, _ = toy_backbone(
                toy_input_ids, torch.ones_like(toy_input_ids, dtype=torch.bool),
            )
        model = HybridCLM(toy_backbone, lora_rank=4)
        with torch.no_grad():
            out = model(toy_input_ids)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_forward_hidden_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = HybridCLM(toy_backbone, lora_rank=4)
        hidden = model.forward_hidden(toy_input_ids)
        B, T = toy_input_ids.shape
        assert hidden.shape == (B, T, toy_clm_config.d_model)

    def test_project_head_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = HybridCLM(toy_backbone, lora_rank=4)
        hidden = model.forward_hidden(toy_input_ids)
        logits = model.project_head(hidden)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_full_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = HybridCLM(toy_backbone, lora_rank=4)
        logits = model(toy_input_ids)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_generate(self, toy_backbone, toy_clm_config):
        model = HybridCLM(toy_backbone, lora_rank=4)
        ids = torch.randint(0, toy_clm_config.vocab_size, (1, 5))
        logits, kv_cache = model.forward_generate(ids)
        assert logits.shape == (1, 1, toy_clm_config.vocab_size)
        assert len(kv_cache) == toy_clm_config.n_layers
        nxt = torch.randint(0, toy_clm_config.vocab_size, (1, 1))
        logits2, _ = model.forward_generate(nxt, kv_cache)
        assert logits2.shape == (1, 1, toy_clm_config.vocab_size)

    def test_gradient_flow(self, toy_backbone, toy_input_ids, toy_clm_config):
        model = HybridCLM(toy_backbone, lora_rank=4)
        # Perturb LoRA B so grads flow through LoRA path
        for p in model.lora_parameters():
            p.data.normal_(0.0, 0.01)
        targets = torch.randint(0, toy_clm_config.vocab_size, toy_input_ids.shape)
        logits = model(toy_input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
        )
        loss.backward()
        # non-adapter backbone params have no grad
        for name, p in model.backbone.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                continue
            assert p.grad is None
        # LoRA + FiLM params have grads
        for p in model.lora_parameters():
            assert p.grad is not None
        for p in model.film_parameters():
            assert p.grad is not None

    def test_state_dict_roundtrip(self, toy_backbone, toy_backbone_fresh,
                                   toy_input_ids, toy_attention_mask):
        model = HybridCLM(toy_backbone, lora_rank=4)
        # Perturb LoRA and FiLM
        for p in model.lora_parameters():
            p.data.normal_(0.0, 0.05)
        for p in model.film_parameters():
            p.data += torch.randn_like(p) * 0.05
        state = model.adapter_state_dict()
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(state.keys()) == expected

        fresh = HybridCLM(toy_backbone_fresh, lora_rank=4)
        fresh.load_adapter_state_dict(state)
        with torch.no_grad():
            o1 = model(toy_input_ids, toy_attention_mask)
            o2 = fresh(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-6)


class TestHybridCLMParams:
    def test_lora_and_film_params_disjoint(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4)
        lora = {id(p) for p in model.lora_parameters()}
        film = {id(p) for p in model.film_parameters()}
        assert lora.isdisjoint(film)

    def test_lora_param_count(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4, attn_targets="qkvo")
        # 2 layers × 4 projs × 2 (A,B) = 16
        assert len(model.lora_parameters()) == 2 * 4 * 2

    def test_film_param_count_no_output(self, toy_backbone, toy_clm_config):
        model = HybridCLM(toy_backbone, lora_rank=4, use_film=True, use_output_film=False)
        # 2 FiLM layers × (gamma, beta) = 4 params
        assert len(model.film_parameters()) == 2 * 2

    def test_film_param_count_with_output(self, toy_backbone, toy_clm_config):
        model = HybridCLM(toy_backbone, lora_rank=4, use_film=True, use_output_film=True)
        # 2 + 1 output FiLM × 2 = 6 params
        assert len(model.film_parameters()) == (2 + 1) * 2

    def test_no_film(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4, use_film=False, use_output_film=False)
        assert model.hidden_films is None
        assert model.output_film is None
        assert len(model.film_parameters()) == 0

    def test_total_param_count_toy(self, toy_backbone, toy_clm_config):
        """Total trainable = LoRA + FiLM; LoRA = 2 layers × 4 × (rank × d + d × rank)."""
        rank = 4
        model = HybridCLM(toy_backbone, lora_rank=rank, use_film=True, use_output_film=False)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        d = toy_clm_config.d_model
        n_layers = toy_clm_config.n_layers
        lora_numel = n_layers * 4 * (rank * d + d * rank)
        film_numel = n_layers * 2 * d
        assert total == lora_numel + film_numel


class TestHybridCLMConfig:
    def test_adapt_ffn_includes_ffn_lora(self, toy_backbone):
        from pawn.adapters.lora import _FFN_TARGETS
        model = HybridCLM(toy_backbone, lora_rank=4, adapt_ffn=True)
        # 2 layers × (4 attn + 3 ffn) × 2 = 28
        assert len(model.lora_parameters()) == 2 * (4 + 3) * 2
        for i in range(len(model.backbone.layers)):
            ffn = model.backbone.get_block(i).ffn
            for name in _FFN_TARGETS:
                assert isinstance(getattr(ffn, name), LoRALinear)

    def test_lora_layers_subset(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4, lora_layers=(0,))
        # Only layer 0 has LoRA; layer 1 attn still has original nn.Linear
        assert isinstance(model.backbone.get_block(0).attn.wq, LoRALinear)
        assert not isinstance(model.backbone.get_block(1).attn.wq, LoRALinear)

    def test_film_layers_subset(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4, film_layers=(1,))
        assert not isinstance(model.hidden_films[0], FiLMLayer)
        assert isinstance(model.hidden_films[1], FiLMLayer)
        # Only 1 FiLM layer → 2 params
        assert len(model.film_parameters()) == 2

    def test_output_film_enabled(self, toy_backbone, toy_clm_config):
        model = HybridCLM(toy_backbone, lora_rank=4, use_output_film=True)
        assert model.output_film is not None
        assert model.output_film.gamma.shape == (toy_clm_config.vocab_size,)

    def test_output_film_disabled(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4, use_output_film=False)
        assert model.output_film is None

    def test_cfg_property(self, toy_backbone, toy_clm_config):
        model = HybridCLM(toy_backbone, lora_rank=4)
        assert model.cfg is toy_clm_config

    def test_weight_report_lora_and_film(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4)
        rep = model.weight_report()
        has_lora = any("lora" in k for k in rep)
        has_film = any("film" in k for k in rep)
        assert has_lora
        assert has_film

    def test_custom_alpha_propagates(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4, lora_alpha=16.0)
        assert model.lora_alpha == 16.0
        # Verify at least one LoRALinear has alpha=16
        mod = model.backbone.get_block(0).attn.wq
        assert isinstance(mod, LoRALinear)
        assert mod.alpha == 16.0

    def test_qv_attn_targets(self, toy_backbone):
        model = HybridCLM(toy_backbone, lora_rank=4, attn_targets="qv")
        # 2 layers × 2 projs × 2 = 8
        assert len(model.lora_parameters()) == 2 * 2 * 2
