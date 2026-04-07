"""Tests for LoRA (Low-Rank Adaptation) adapter."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from pawn.adapters.lora import LoRALinear, LoRACLM, ATTN_PRESETS


class TestLoRALinear:
    def test_lora_B_zero_init(self):
        frozen = nn.Linear(32, 32, bias=False)
        lora = LoRALinear(frozen, rank=4)
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

    def test_lora_A_nonzero_init(self):
        frozen = nn.Linear(32, 32, bias=False)
        lora = LoRALinear(frozen, rank=4)
        # Kaiming init → non-zero
        assert lora.lora_A.abs().sum().item() > 0.0

    def test_identity_at_init_no_bias(self):
        frozen = nn.Linear(32, 48, bias=False)
        lora = LoRALinear(frozen, rank=4)
        x = torch.randn(2, 5, 32)
        expected = frozen(x)
        actual = lora(x)
        torch.testing.assert_close(actual, expected)

    def test_identity_at_init_with_bias(self):
        frozen = nn.Linear(32, 48, bias=True)
        lora = LoRALinear(frozen, rank=4)
        x = torch.randn(2, 5, 32)
        expected = frozen(x)
        actual = lora(x)
        torch.testing.assert_close(actual, expected)

    def test_shapes(self):
        frozen = nn.Linear(16, 24, bias=False)
        lora = LoRALinear(frozen, rank=2)
        assert lora.lora_A.shape == (2, 16)
        assert lora.lora_B.shape == (24, 2)

    def test_scaling_default(self):
        # alpha defaults to rank → scaling = 1.0
        frozen = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(frozen, rank=4)
        assert lora.alpha == 4.0
        assert lora.scaling == 1.0

    def test_scaling_custom_alpha(self):
        frozen = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(frozen, rank=4, alpha=16.0)
        assert lora.alpha == 16.0
        assert lora.scaling == 4.0

    def test_delta_after_nonzero_B(self):
        frozen = nn.Linear(32, 32, bias=False)
        lora = LoRALinear(frozen, rank=4, alpha=16.0)
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 5, 32)
        out = lora(x)
        bare = frozen(x)
        # delta = (x @ A^T) @ B^T * scaling
        expected_delta = (x @ lora.lora_A.T) @ lora.lora_B.T * lora.scaling
        torch.testing.assert_close(out, bare + expected_delta, atol=1e-6, rtol=1e-6)

    def test_frozen_preserved(self):
        frozen = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(frozen, rank=2)
        assert lora.frozen is frozen


class TestLoRACLMInvariants:
    def test_backbone_frozen(self, toy_backbone):
        """Original backbone params (non-LoRA) are all frozen.
        LoRA A/B end up under `.backbone.` because LoRALinear replaces attention
        projections in place — they're trainable by design, so filter them out.
        """
        model = LoRACLM(toy_backbone, rank=4)
        for name, p in model.backbone.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                # these are adapter params injected by LoRALinear
                assert p.requires_grad
                continue
            assert not p.requires_grad, f"backbone param {name} not frozen"

    def test_lora_params_trainable(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0
        # Every LoRALinear in attention should have trainable A, B
        for i in range(len(model.backbone.layers)):
            attn = model.backbone.get_block(i).attn
            for proj_name in model.attn_targets:
                mod = getattr(attn, proj_name)
                assert isinstance(mod, LoRALinear)
                assert mod.lora_A.requires_grad
                assert mod.lora_B.requires_grad
                # frozen wrapped weight should not be trainable
                assert not mod.frozen.weight.requires_grad

    def test_zero_init_identity(self, toy_backbone, toy_input_ids, toy_attention_mask):
        with torch.no_grad():
            bare, _ = toy_backbone(toy_input_ids, toy_attention_mask)
        model = LoRACLM(toy_backbone, rank=4)
        with torch.no_grad():
            out = model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_zero_init_identity_no_mask(self, toy_backbone, toy_input_ids):
        with torch.no_grad():
            bare, _ = toy_backbone(
                toy_input_ids, torch.ones_like(toy_input_ids, dtype=torch.bool),
            )
        model = LoRACLM(toy_backbone, rank=4)
        with torch.no_grad():
            out = model(toy_input_ids)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_forward_hidden_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = LoRACLM(toy_backbone, rank=4)
        hidden = model.forward_hidden(toy_input_ids)
        B, T = toy_input_ids.shape
        assert hidden.shape == (B, T, toy_clm_config.d_model)

    def test_project_head_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = LoRACLM(toy_backbone, rank=4)
        hidden = model.forward_hidden(toy_input_ids)
        logits = model.project_head(hidden)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_full_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = LoRACLM(toy_backbone, rank=4)
        logits = model(toy_input_ids)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_generate(self, toy_backbone, toy_clm_config):
        model = LoRACLM(toy_backbone, rank=4)
        ids = torch.randint(0, toy_clm_config.vocab_size, (1, 5))
        logits, kv_cache = model.forward_generate(ids)
        assert logits.shape == (1, 1, toy_clm_config.vocab_size)
        assert len(kv_cache) == toy_clm_config.n_layers
        nxt = torch.randint(0, toy_clm_config.vocab_size, (1, 1))
        logits2, _ = model.forward_generate(nxt, kv_cache)
        assert logits2.shape == (1, 1, toy_clm_config.vocab_size)

    def test_gradient_flow(self, toy_backbone, toy_input_ids, toy_clm_config):
        model = LoRACLM(toy_backbone, rank=4)
        # Perturb B so delta is non-zero and grads flow non-trivially
        for i in range(len(model.backbone.layers)):
            attn = model.backbone.get_block(i).attn
            for proj_name in model.attn_targets:
                mod = getattr(attn, proj_name)
                mod.lora_B.data.normal_(0.0, 0.01)
        targets = torch.randint(0, toy_clm_config.vocab_size, toy_input_ids.shape)
        logits = model(toy_input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
        )
        loss.backward()
        # Backbone (non-adapter) frozen — no grad
        for name, p in model.backbone.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                continue
            assert p.grad is None
        # LoRA params have grads
        for i in range(len(model.backbone.layers)):
            attn = model.backbone.get_block(i).attn
            for proj_name in model.attn_targets:
                mod = getattr(attn, proj_name)
                assert mod.lora_A.grad is not None
                assert mod.lora_B.grad is not None

    def test_state_dict_roundtrip(self, toy_backbone, toy_backbone_fresh,
                                   toy_input_ids, toy_attention_mask):
        model = LoRACLM(toy_backbone, rank=4)
        # Perturb weights
        for i in range(len(model.backbone.layers)):
            attn = model.backbone.get_block(i).attn
            for proj_name in model.attn_targets:
                mod = getattr(attn, proj_name)
                mod.lora_B.data.normal_(0.0, 0.05)
                mod.lora_A.data.normal_(0.0, 0.05)
        state = model.lora_state_dict()
        expected_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(state.keys()) == expected_keys

        fresh_model = LoRACLM(toy_backbone_fresh, rank=4)
        fresh_model.load_lora_state_dict(state)
        with torch.no_grad():
            o1 = model(toy_input_ids, toy_attention_mask)
            o2 = fresh_model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-6)

    def test_lora_parameters_trainable_only(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4)
        params = model.lora_parameters()
        for p in params:
            assert p.requires_grad
        # Ids must be unique
        ids = {id(p) for p in params}
        assert len(ids) == len(params)


class TestLoRAAttnTargets:
    def test_qkvo_preset(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, attn_targets="qkvo")
        assert model.attn_targets == ATTN_PRESETS["qkvo"]
        # 2 layers × 4 projs × 2 params = 16
        assert len(model.lora_parameters()) == 2 * 4 * 2

    def test_qv_preset(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, attn_targets="qv")
        assert model.attn_targets == ATTN_PRESETS["qv"]
        # 2 layers × 2 projs × 2 params = 8
        assert len(model.lora_parameters()) == 2 * 2 * 2

    def test_qkv_preset(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, attn_targets="qkv")
        assert model.attn_targets == ATTN_PRESETS["qkv"]
        # 2 layers × 3 projs × 2 params = 12
        assert len(model.lora_parameters()) == 2 * 3 * 2

    def test_custom_tuple_targets(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, attn_targets=("wq",))
        # 2 layers × 1 proj × 2 params = 4
        assert len(model.lora_parameters()) == 2 * 1 * 2
        assert model.attn_targets == ("wq",)

    def test_preset_contents(self):
        assert set(ATTN_PRESETS["qkvo"]) == {"wq", "wk", "wv", "wo"}
        assert set(ATTN_PRESETS["qv"]) == {"wq", "wv"}
        assert set(ATTN_PRESETS["qkv"]) == {"wq", "wk", "wv"}


class TestLoRAAdaptFFN:
    def test_adapt_ffn_adds_modules(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, adapt_ffn=True)
        from pawn.adapters.lora import _FFN_TARGETS
        # 2 layers × 3 FFN projs × 2 params extra
        n_total = len(model.lora_parameters())
        # base = 2 layers × 4 attn × 2 = 16
        assert n_total == 2 * 4 * 2 + 2 * 3 * 2
        # check each FFN proj is a LoRALinear
        for i in range(len(model.backbone.layers)):
            ffn = model.backbone.get_block(i).ffn
            for name in _FFN_TARGETS:
                assert isinstance(getattr(ffn, name), LoRALinear)

    def test_no_adapt_ffn_default(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, adapt_ffn=False)
        from pawn.adapters.lora import _FFN_TARGETS
        for i in range(len(model.backbone.layers)):
            ffn = model.backbone.get_block(i).ffn
            for name in _FFN_TARGETS:
                # FFN untouched
                assert not isinstance(getattr(ffn, name), LoRALinear)


class TestLoRALayerSubset:
    def test_subset_layers(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, layers=(0,))
        # Layer 0 should have LoRA, layer 1 should not
        for proj_name in model.attn_targets:
            mod0 = getattr(model.backbone.get_block(0).attn, proj_name)
            mod1 = getattr(model.backbone.get_block(1).attn, proj_name)
            assert isinstance(mod0, LoRALinear)
            assert not isinstance(mod1, LoRALinear)
        assert model.adapted_layers == {0}


class TestLoRACustomAlpha:
    def test_custom_alpha_zero_init_still_identity(self, toy_backbone,
                                                     toy_input_ids, toy_attention_mask):
        with torch.no_grad():
            bare, _ = toy_backbone(toy_input_ids, toy_attention_mask)
        model = LoRACLM(toy_backbone, rank=4, alpha=32.0)
        with torch.no_grad():
            out = model(toy_input_ids, toy_attention_mask)
        # B is still zero — scaling doesn't matter
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_alpha_propagates_to_modules(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, alpha=16.0)
        for i in range(len(model.backbone.layers)):
            attn = model.backbone.get_block(i).attn
            for proj_name in model.attn_targets:
                mod = getattr(attn, proj_name)
                assert mod.alpha == 16.0
                assert mod.scaling == 4.0


class TestLoRAWeightReport:
    def test_report_contains_all_layers(self, toy_backbone):
        model = LoRACLM(toy_backbone, rank=4, attn_targets="qkvo")
        rep = model.lora_weight_report()
        # 2 layers × 4 projs × 2 (A,B) keys = 16
        assert len(rep) == 2 * 4 * 2
        for val in rep.values():
            assert isinstance(val, float)

    def test_report_B_zero_at_init(self, toy_backbone, toy_clm_config):
        model = LoRACLM(toy_backbone, rank=4, attn_targets="qkvo")
        rep = model.lora_weight_report()
        b_keys = [k for k in rep if k.endswith(".B")]
        # 2 layers × 4 projections (Q, K, V, O) = 8 B entries
        expected_count = toy_clm_config.n_layers * 4
        assert len(b_keys) == expected_count, (
            f"expected {expected_count} B keys, got {len(b_keys)}: {b_keys}"
        )
        for k in b_keys:
            assert rep[k] == 0.0, f"{k} should be 0.0 at init, got {rep[k]}"
