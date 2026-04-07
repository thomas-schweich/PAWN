"""Tests for Bottleneck (Houlsby) adapter."""

from __future__ import annotations

import pytest
import torch

from pawn.adapters.bottleneck import BottleneckAdapter, BottleneckCLM


class TestBottleneckAdapter:
    def test_up_weight_zero_init(self):
        adapter = BottleneckAdapter(d_model=64, bottleneck_dim=8)
        assert torch.allclose(adapter.up.weight, torch.zeros_like(adapter.up.weight))

    def test_down_weight_nonzero_init(self):
        adapter = BottleneckAdapter(d_model=64, bottleneck_dim=8)
        # Kaiming init — should NOT be all zero
        assert adapter.down.weight.abs().sum().item() > 0.0

    def test_identity_at_init(self):
        """x + up(gelu(down(x))) == x when up.weight == 0."""
        adapter = BottleneckAdapter(d_model=64, bottleneck_dim=8)
        x = torch.randn(2, 8, 64)
        y = adapter(x)
        torch.testing.assert_close(y, x)

    def test_shapes_preserve(self):
        adapter = BottleneckAdapter(d_model=32, bottleneck_dim=4)
        x = torch.randn(3, 7, 32)
        y = adapter(x)
        assert y.shape == x.shape

    def test_non_identity_after_up_modified(self):
        adapter = BottleneckAdapter(d_model=64, bottleneck_dim=8)
        adapter.up.weight.data.fill_(0.1)
        x = torch.randn(2, 8, 64)
        y = adapter(x)
        assert not torch.allclose(y, x)

    def test_no_bias(self):
        adapter = BottleneckAdapter(d_model=32, bottleneck_dim=4)
        assert adapter.down.bias is None
        assert adapter.up.bias is None

    def test_param_count(self):
        d_model, bn = 64, 8
        adapter = BottleneckAdapter(d_model=d_model, bottleneck_dim=bn)
        total = sum(p.numel() for p in adapter.parameters())
        # Expected: down (bn*d) + up (d*bn) = 2*d*bn
        assert total == 2 * d_model * bn


class TestBottleneckCLMInvariants:
    def test_backbone_frozen(self, toy_backbone):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        # All original backbone params must be frozen
        for name, p in model.backbone.named_parameters():
            assert not p.requires_grad, f"backbone param {name} is not frozen"

    def test_adapter_params_trainable(self, toy_backbone):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0
        # All trainable params live in the adapter modules
        for a in model.attn_adapters:
            if isinstance(a, BottleneckAdapter):
                assert a.down.weight.requires_grad
                assert a.up.weight.requires_grad
        for a in model.ffn_adapters:
            if isinstance(a, BottleneckAdapter):
                assert a.down.weight.requires_grad
                assert a.up.weight.requires_grad

    def test_zero_init_identity(self, toy_backbone, toy_input_ids, toy_attention_mask):
        """Adapter wraps backbone → forward at init matches bare backbone output."""
        # Capture bare-backbone output BEFORE wrapping (wrap freezes but adapters are zero-init)
        with torch.no_grad():
            bare_logits, _ = toy_backbone(toy_input_ids, toy_attention_mask)
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        with torch.no_grad():
            adapted_logits = model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(adapted_logits, bare_logits, atol=1e-5, rtol=1e-5)

    def test_zero_init_identity_no_mask(self, toy_backbone, toy_input_ids):
        """When attention_mask=None, is_causal path is used — still zero-init identity."""
        # Bare backbone with explicit mask of all ones
        with torch.no_grad():
            bare_logits, _ = toy_backbone(
                toy_input_ids, torch.ones_like(toy_input_ids, dtype=torch.bool),
            )
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        with torch.no_grad():
            adapted_logits = model(toy_input_ids)
        # The is_causal path in adapters doesn't use an explicit mask; should produce same logits
        torch.testing.assert_close(adapted_logits, bare_logits, atol=1e-5, rtol=1e-5)

    def test_forward_hidden_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        hidden = model.forward_hidden(toy_input_ids)
        B, T = toy_input_ids.shape
        assert hidden.shape == (B, T, toy_clm_config.d_model)

    def test_project_head_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        hidden = model.forward_hidden(toy_input_ids)
        logits = model.project_head(hidden)
        B, T = toy_input_ids.shape
        assert logits.shape == (B, T, toy_clm_config.vocab_size)

    def test_forward_full_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        logits = model(toy_input_ids)
        B, T = toy_input_ids.shape
        assert logits.shape == (B, T, toy_clm_config.vocab_size)

    def test_forward_generate(self, toy_backbone, toy_clm_config):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        ids = torch.randint(0, toy_clm_config.vocab_size, (1, 5))
        logits, kv_cache = model.forward_generate(ids)
        assert logits.shape == (1, 1, toy_clm_config.vocab_size)
        assert len(kv_cache) == toy_clm_config.n_layers
        # follow-up token
        nxt = torch.randint(0, toy_clm_config.vocab_size, (1, 1))
        logits2, kv_cache2 = model.forward_generate(nxt, kv_cache)
        assert logits2.shape == (1, 1, toy_clm_config.vocab_size)
        assert len(kv_cache2) == toy_clm_config.n_layers

    def test_gradient_flow(self, toy_backbone, toy_input_ids, toy_clm_config):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        # Perturb adapter to break out of zero-init so grads propagate
        for a in model.attn_adapters:
            if isinstance(a, BottleneckAdapter):
                a.up.weight.data.normal_(0.0, 0.01)
        targets = torch.randint(0, toy_clm_config.vocab_size, toy_input_ids.shape)
        logits = model(toy_input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
        )
        loss.backward()
        # Backbone params: no grad (requires_grad=False)
        for p in model.backbone.parameters():
            assert p.grad is None
        # Adapter params: have grad
        at_least_one_adapter_grad = False
        for a in list(model.attn_adapters) + list(model.ffn_adapters):
            if isinstance(a, BottleneckAdapter):
                assert a.down.weight.grad is not None
                assert a.up.weight.grad is not None
                at_least_one_adapter_grad = True
        assert at_least_one_adapter_grad

    def test_state_dict_roundtrip(self, toy_backbone, toy_backbone_fresh,
                                   toy_input_ids, toy_attention_mask):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        # Perturb adapter weights
        for a in list(model.attn_adapters) + list(model.ffn_adapters):
            if isinstance(a, BottleneckAdapter):
                a.down.weight.data.normal_(0.0, 0.5)
                a.up.weight.data.normal_(0.0, 0.01)
        state = model.adapter_state_dict()
        # All keys should correspond to trainable params
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(state.keys()) == names

        # Load into fresh wrapper
        fresh_model = BottleneckCLM(toy_backbone_fresh, bottleneck_dim=4)
        fresh_model.load_adapter_state_dict(state)
        with torch.no_grad():
            out1 = model(toy_input_ids, toy_attention_mask)
            out2 = fresh_model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    def test_adapter_parameters_matches_requires_grad(self, toy_backbone):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        adapter = model.adapter_parameters()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(adapter) == len(trainable)


class TestBottleneckCLMConfig:
    def test_adapt_attn_only(self, toy_backbone):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4,
                              adapt_attn=True, adapt_ffn=False)
        n_attn = sum(1 for a in model.attn_adapters if isinstance(a, BottleneckAdapter))
        n_ffn = sum(1 for a in model.ffn_adapters if isinstance(a, BottleneckAdapter))
        assert n_attn == len(toy_backbone.layers)
        assert n_ffn == 0

    def test_adapt_ffn_only(self, toy_backbone):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4,
                              adapt_attn=False, adapt_ffn=True)
        n_attn = sum(1 for a in model.attn_adapters if isinstance(a, BottleneckAdapter))
        n_ffn = sum(1 for a in model.ffn_adapters if isinstance(a, BottleneckAdapter))
        assert n_attn == 0
        assert n_ffn == len(toy_backbone.layers)

    def test_layer_subset(self, toy_backbone):
        # toy has 2 layers; adapt only layer 0
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4, layers=(0,))
        assert isinstance(model.attn_adapters[0], BottleneckAdapter)
        assert isinstance(model.ffn_adapters[0], BottleneckAdapter)
        # layer 1 should be Identity
        assert not isinstance(model.attn_adapters[1], BottleneckAdapter)
        assert not isinstance(model.ffn_adapters[1], BottleneckAdapter)

    def test_attn_layers_override(self, toy_backbone):
        # attn_layers takes precedence
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4,
                              attn_layers=(1,), ffn_layers=())
        assert not isinstance(model.attn_adapters[0], BottleneckAdapter)
        assert isinstance(model.attn_adapters[1], BottleneckAdapter)
        for a in model.ffn_adapters:
            assert not isinstance(a, BottleneckAdapter)

    def test_param_count_toy(self, toy_backbone, toy_clm_config):
        # toy: d_model=64, n_layers=2; bottleneck_dim=8
        # Both attn + ffn adapted by default:
        # per adapter: 2 * 64 * 8 = 1024; total: 2 layers * 2 pos * 1024 = 4096
        model = BottleneckCLM(toy_backbone, bottleneck_dim=8)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected = 2 * toy_clm_config.n_layers * 2 * toy_clm_config.d_model * 8
        assert total == expected

    def test_attn_layers_empty(self, toy_backbone):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4,
                              attn_layers=(), ffn_layers=())
        # No adapters => no trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) == 0

    def test_adapter_weight_report(self, toy_backbone):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        report = model.adapter_weight_report()
        assert len(report) > 0
        for k in report:
            assert k.startswith("adapter/")
            assert isinstance(report[k], float)

    def test_cfg_property(self, toy_backbone, toy_clm_config):
        model = BottleneckCLM(toy_backbone, bottleneck_dim=4)
        assert model.cfg is toy_clm_config
