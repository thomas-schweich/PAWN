"""Tests for Sparse (random binary mask) adapter."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pawn.adapters.sparse import SparseLinear, SparseCLM


class TestSparseLinear:
    def test_delta_zero_init(self):
        frozen = nn.Linear(16, 16, bias=False)
        mask = torch.ones(16, 16, dtype=torch.bool)
        sparse = SparseLinear(frozen, mask)
        assert torch.allclose(sparse.delta, torch.zeros_like(sparse.delta))

    def test_identity_at_init_all_mask(self):
        frozen = nn.Linear(16, 16, bias=False)
        mask = torch.ones(16, 16, dtype=torch.bool)
        sparse = SparseLinear(frozen, mask)
        x = torch.randn(2, 4, 16)
        torch.testing.assert_close(sparse(x), frozen(x))

    def test_identity_at_init_with_bias(self):
        frozen = nn.Linear(16, 24, bias=True)
        mask = torch.rand(24, 16) > 0.9
        sparse = SparseLinear(frozen, mask)
        x = torch.randn(2, 4, 16)
        torch.testing.assert_close(sparse(x), frozen(x))

    def test_identity_at_init_partial_mask(self):
        frozen = nn.Linear(16, 16, bias=False)
        mask = torch.rand(16, 16) > 0.5
        sparse = SparseLinear(frozen, mask)
        x = torch.randn(2, 4, 16)
        torch.testing.assert_close(sparse(x), frozen(x))

    def test_n_active_matches_mask_sum(self):
        frozen = nn.Linear(16, 16, bias=False)
        mask = torch.rand(16, 16) > 0.3
        sparse = SparseLinear(frozen, mask)
        assert sparse.n_active == int(mask.sum().item())

    def test_n_active_all_ones(self):
        frozen = nn.Linear(8, 8, bias=False)
        mask = torch.ones(8, 8, dtype=torch.bool)
        sparse = SparseLinear(frozen, mask)
        assert sparse.n_active == 64

    def test_n_active_zero(self):
        frozen = nn.Linear(8, 8, bias=False)
        mask = torch.zeros(8, 8, dtype=torch.bool)
        sparse = SparseLinear(frozen, mask)
        assert sparse.n_active == 0

    def test_delta_restricted_to_mask(self):
        """After setting delta uniformly, the effective weight change is only on masked positions."""
        frozen = nn.Linear(8, 8, bias=False)
        frozen.weight.data.zero_()
        mask = torch.rand(8, 8) > 0.5
        sparse = SparseLinear(frozen, mask)
        sparse.delta.data.fill_(0.5)
        # Effective weight = 0 + 0.5 * mask = 0.5 at masked positions
        effective_w = sparse.frozen.weight + sparse.delta * sparse.mask
        assert torch.allclose(effective_w, 0.5 * mask.float())

    def test_mask_is_buffer_not_param(self):
        frozen = nn.Linear(16, 16, bias=False)
        mask = torch.ones(16, 16, dtype=torch.bool)
        sparse = SparseLinear(frozen, mask)
        buffer_names = [n for n, _ in sparse.named_buffers()]
        assert "mask" in buffer_names
        # mask shouldn't appear in parameters
        param_names = [n for n, _ in sparse.named_parameters()]
        assert "mask" not in param_names

    def test_delta_is_trainable(self):
        frozen = nn.Linear(16, 16, bias=False)
        mask = torch.ones(16, 16, dtype=torch.bool)
        sparse = SparseLinear(frozen, mask)
        assert sparse.delta.requires_grad

    def test_forward_after_training_delta(self):
        frozen = nn.Linear(8, 8, bias=False)
        frozen.weight.data.fill_(1.0)
        mask = torch.ones(8, 8, dtype=torch.bool)
        sparse = SparseLinear(frozen, mask)
        sparse.delta.data.fill_(0.5)
        x = torch.ones(1, 1, 8)
        # effective weight = 1 + 0.5 = 1.5 per element; output = 8 * 1.5 = 12.0 per position
        out = sparse(x)
        assert torch.allclose(out, torch.full((1, 1, 8), 12.0))


class TestSparseCLMInvariants:
    def test_backbone_frozen(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1)
        # Non-delta params should be frozen; delta params are trainable
        for name, p in model.backbone.named_parameters():
            if "delta" in name:
                assert p.requires_grad
            else:
                assert not p.requires_grad, f"backbone param {name} not frozen"

    def test_sparse_params_trainable(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0
        # 2 layers × 4 projs by default
        assert len(trainable) == 2 * 4

    def test_zero_init_identity(self, toy_backbone, toy_input_ids, toy_attention_mask):
        with torch.no_grad():
            bare, _ = toy_backbone(toy_input_ids, toy_attention_mask)
        model = SparseCLM(toy_backbone, density=0.1)
        with torch.no_grad():
            out = model(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_zero_init_identity_no_mask(self, toy_backbone, toy_input_ids):
        with torch.no_grad():
            bare, _ = toy_backbone(
                toy_input_ids, torch.ones_like(toy_input_ids, dtype=torch.bool),
            )
        model = SparseCLM(toy_backbone, density=0.1)
        with torch.no_grad():
            out = model(toy_input_ids)
        torch.testing.assert_close(out, bare, atol=1e-5, rtol=1e-5)

    def test_forward_hidden_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = SparseCLM(toy_backbone, density=0.1)
        hidden = model.forward_hidden(toy_input_ids)
        B, T = toy_input_ids.shape
        assert hidden.shape == (B, T, toy_clm_config.d_model)

    def test_project_head_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = SparseCLM(toy_backbone, density=0.1)
        hidden = model.forward_hidden(toy_input_ids)
        logits = model.project_head(hidden)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_full_shape(self, toy_backbone, toy_clm_config, toy_input_ids):
        model = SparseCLM(toy_backbone, density=0.1)
        logits = model(toy_input_ids)
        assert logits.shape == (*toy_input_ids.shape, toy_clm_config.vocab_size)

    def test_forward_generate(self, toy_backbone, toy_clm_config):
        model = SparseCLM(toy_backbone, density=0.1)
        ids = torch.randint(0, toy_clm_config.vocab_size, (1, 5))
        logits, kv_cache = model.forward_generate(ids)
        assert logits.shape == (1, 1, toy_clm_config.vocab_size)
        assert len(kv_cache) == toy_clm_config.n_layers
        nxt = torch.randint(0, toy_clm_config.vocab_size, (1, 1))
        logits2, _ = model.forward_generate(nxt, kv_cache)
        assert logits2.shape == (1, 1, toy_clm_config.vocab_size)

    def test_gradient_flow(self, toy_backbone, toy_input_ids, toy_clm_config):
        model = SparseCLM(toy_backbone, density=0.1)
        # Perturb deltas so grads propagate non-trivially
        for p in model.sparse_parameters():
            p.data.normal_(0.0, 0.01)
        targets = torch.randint(0, toy_clm_config.vocab_size, toy_input_ids.shape)
        logits = model(toy_input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
        )
        loss.backward()
        # backbone weights have no grad
        for name, p in model.backbone.named_parameters():
            if "delta" in name:
                continue
            assert p.grad is None
        # delta params have grads
        for p in model.sparse_parameters():
            assert p.grad is not None

    def test_state_dict_roundtrip(self, toy_backbone, toy_backbone_fresh,
                                   toy_input_ids, toy_attention_mask):
        model = SparseCLM(toy_backbone, density=0.1, seed=7)
        # Perturb deltas
        for p in model.sparse_parameters():
            p.data.normal_(0.0, 0.1)
        state = model.sparse_state_dict()
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(state.keys()) == expected

        # Fresh model with same seed => identical masks
        fresh = SparseCLM(toy_backbone_fresh, density=0.1, seed=7)
        fresh.load_sparse_state_dict(state)
        with torch.no_grad():
            o1 = model(toy_input_ids, toy_attention_mask)
            o2 = fresh(toy_input_ids, toy_attention_mask)
        torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-6)

    def test_sparse_parameters_trainable_only(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1)
        params = model.sparse_parameters()
        for p in params:
            assert p.requires_grad


class TestSparseCLMDensity:
    @pytest.mark.parametrize("density", [0.1, 0.5])
    def test_density_approx(self, toy_clm_config, density):
        from pawn.model import PAWNCLM
        torch.manual_seed(0)
        backbone = PAWNCLM(toy_clm_config)
        model = SparseCLM(backbone, density=density, seed=42)
        # Count active entries
        n_active = model.n_active_params()
        # 2 layers × 4 projs × d_model × d_model elements each
        total_possible = (
            toy_clm_config.n_layers * 4 * toy_clm_config.d_model * toy_clm_config.d_model
        )
        actual_density = n_active / total_possible
        # 15% relative tolerance — tight enough to catch bugs, loose enough
        # for the stochastic mask (each mask has 64*64=4096 Bernoulli draws).
        assert abs(actual_density - density) / density < 0.15, (
            f"expected density ~{density}, got {actual_density} "
            f"(relative error {abs(actual_density - density) / density:.2%})"
        )

    def test_density_deterministic_with_seed(self, toy_clm_config):
        from pawn.model import PAWNCLM
        torch.manual_seed(0)
        bb1 = PAWNCLM(toy_clm_config)
        torch.manual_seed(0)
        bb2 = PAWNCLM(toy_clm_config)
        m1 = SparseCLM(bb1, density=0.05, seed=123)
        m2 = SparseCLM(bb2, density=0.05, seed=123)
        assert m1.n_active_params() == m2.n_active_params()

    def test_density_high_vs_low(self, toy_clm_config):
        from pawn.model import PAWNCLM
        torch.manual_seed(0)
        bb_low = PAWNCLM(toy_clm_config)
        torch.manual_seed(0)
        bb_high = PAWNCLM(toy_clm_config)
        low = SparseCLM(bb_low, density=0.01, seed=1).n_active_params()
        high = SparseCLM(bb_high, density=0.5, seed=1).n_active_params()
        assert low < high


class TestSparseCLMConfig:
    def test_default_attn_targets(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1)
        assert model.attn_targets == ("wq", "wk", "wv", "wo")

    def test_custom_attn_targets(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1, attn_targets=("wq", "wv"))
        assert model.attn_targets == ("wq", "wv")
        # 2 layers × 2 projs = 4 SparseLinear modules => 4 delta params
        assert len(model.sparse_parameters()) == 2 * 2

    def test_adapt_ffn_true(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1, adapt_ffn=True)
        # 2 layers × (4 attn + 3 ffn) = 14 deltas
        assert len(model.sparse_parameters()) == 2 * (4 + 3)

    def test_adapt_ffn_false(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1, adapt_ffn=False)
        assert len(model.sparse_parameters()) == 2 * 4

    def test_layer_subset(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1, layers=(0,))
        # Only layer 0 should have SparseLinear
        attn0 = model.backbone.get_block(0).attn
        attn1 = model.backbone.get_block(1).attn
        assert isinstance(attn0.wq, SparseLinear)
        assert not isinstance(attn1.wq, SparseLinear)
        assert model.adapted_layers == {0}

    def test_sparse_weight_report(self, toy_backbone):
        model = SparseCLM(toy_backbone, density=0.1)
        rep = model.sparse_weight_report()
        assert len(rep) > 0
        # At init, all deltas are zero
        for k, v in rep.items():
            assert v == 0.0, f"{k} should be 0 at init"

    def test_cfg_property(self, toy_backbone, toy_clm_config):
        model = SparseCLM(toy_backbone, density=0.1)
        assert model.cfg is toy_clm_config
