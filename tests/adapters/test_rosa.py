"""Tests for RoSA (Robust Sparse Adaptation) adapter."""

import torch
import pytest

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.adapters.rosa import (
    RoSALinear,
    RoSACLM,
    RetroBottleneckCLM,
    generate_gradient_masks,
)
from pawn.adapters.sparse import SparseCLM, SparseLinear


class SimpleMaskBuilder:
    """Dummy mask builder that allows all moves."""
    def __call__(self, batch):
        B, T = batch["input_ids"].shape[:2]
        return torch.ones(B, T, CLMConfig().vocab_size, dtype=torch.bool)


@pytest.fixture
def toy_backbone():
    return PAWNCLM(CLMConfig.toy())


class TestRoSALinear:
    def test_identity_init(self):
        """Output should equal frozen output at init (B=0, delta=0, sparse disabled)."""
        frozen = torch.nn.Linear(64, 64, bias=False)
        rosa = RoSALinear(frozen, rank=4)

        x = torch.randn(2, 8, 64)
        expected = frozen(x)
        actual = rosa(x)
        torch.testing.assert_close(actual, expected)

    def test_identity_init_with_bias(self):
        frozen = torch.nn.Linear(64, 64, bias=True)
        rosa = RoSALinear(frozen, rank=4)

        x = torch.randn(2, 8, 64)
        expected = frozen(x)
        actual = rosa(x)
        torch.testing.assert_close(actual, expected)

    def test_lora_contributes_after_training(self):
        """After modifying B, output should differ from frozen."""
        frozen = torch.nn.Linear(64, 64, bias=False)
        rosa = RoSALinear(frozen, rank=4)
        rosa.lora_B.data.fill_(1.0)

        x = torch.randn(2, 8, 64)
        expected = frozen(x)
        actual = rosa(x)
        assert not torch.allclose(actual, expected)

    def test_sparse_disabled_at_init(self):
        frozen = torch.nn.Linear(64, 64, bias=False)
        rosa = RoSALinear(frozen, rank=4)
        assert not rosa.sparse_enabled
        assert not rosa.delta.requires_grad

    def test_set_mask_enables_sparse(self):
        frozen = torch.nn.Linear(64, 64, bias=False)
        rosa = RoSALinear(frozen, rank=4)

        mask = torch.rand(64, 64) > 0.99
        rosa.set_mask(mask)
        assert rosa.sparse_enabled
        assert rosa.delta.requires_grad
        assert rosa.n_active == mask.sum().item()

    def test_sparse_contributes_after_mask(self):
        frozen = torch.nn.Linear(64, 64, bias=False)
        rosa = RoSALinear(frozen, rank=4)

        mask = torch.ones(64, 64, dtype=torch.bool)
        rosa.set_mask(mask)
        rosa.delta.data.fill_(0.1)

        x = torch.randn(2, 8, 64)
        # With B=0 (LoRA is zero), output = frozen(x) + F.linear(x, 0.1 * ones)
        out = rosa(x)
        frozen_out = frozen(x)
        assert not torch.allclose(out, frozen_out)

    def test_disable_lora(self):
        frozen = torch.nn.Linear(64, 64, bias=False)
        rosa = RoSALinear(frozen, rank=4)
        rosa.lora_B.data.fill_(1.0)
        rosa.lora_enabled = False

        x = torch.randn(2, 8, 64)
        # With LoRA disabled, should be same as frozen (sparse also disabled)
        expected = frozen(x)
        actual = rosa(x)
        torch.testing.assert_close(actual, expected)


class TestRoSACLM:
    def test_wrapping_identity(self):
        """With zero-init, RoSACLM output should match bare backbone."""
        torch.manual_seed(42)
        backbone = PAWNCLM(CLMConfig.toy())
        input_ids = torch.randint(0, CLMConfig().vocab_size, (2, 32))

        # Get bare backbone output BEFORE wrapping (wrapping modifies in-place)
        with torch.no_grad():
            bare_logits = backbone(
                input_ids, torch.ones(2, 32, dtype=torch.bool),
            )[0]

        # Now wrap
        model = RoSACLM(backbone, rank=4)
        with torch.no_grad():
            rosa_logits = model.forward(input_ids)

        torch.testing.assert_close(rosa_logits, bare_logits)

    def test_forward_shapes(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4)
        input_ids = torch.randint(0, CLMConfig().vocab_size, (2, 32))

        hidden = model.forward_hidden(input_ids)
        assert hidden.shape == (2, 32, CLMConfig.toy().d_model)

        logits = model.project_head(hidden)
        assert logits.shape == (2, 32, CLMConfig().vocab_size)

        full_logits = model.forward(input_ids)
        assert full_logits.shape == (2, 32, CLMConfig().vocab_size)

    def test_forward_generate(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4)
        input_ids = torch.randint(0, CLMConfig().vocab_size, (1, 5))

        logits, kv_cache = model.forward_generate(input_ids)
        assert logits.shape == (1, 1, CLMConfig().vocab_size)
        assert len(kv_cache) == CLMConfig.toy().n_layers

        # Next token
        next_id = torch.randint(0, CLMConfig().vocab_size, (1, 1))
        logits2, kv_cache2 = model.forward_generate(next_id, kv_cache)
        assert logits2.shape == (1, 1, CLMConfig().vocab_size)

    def test_lora_parameters(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4, attn_targets="qkvo")
        params = model.lora_parameters()
        # 2 layers * 4 projections * 2 (A, B) = 16 parameters
        assert len(params) == 2 * 4 * 2

    def test_sparse_parameters(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4, attn_targets="qkvo")
        params = model.sparse_parameters()
        # 2 layers * 4 projections = 8 delta parameters
        assert len(params) == 2 * 4

    def test_disable_lora(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4)
        model.disable_lora()
        for _, module in model._rosa_modules():
            assert not module.lora_enabled
            assert not module.lora_A.requires_grad
            assert not module.lora_B.requires_grad

    def test_reinit_lora(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4)
        # Modify B to non-zero
        for _, module in model._rosa_modules():
            module.lora_B.data.fill_(1.0)
        model.reinit_lora()
        for _, module in model._rosa_modules():
            assert torch.allclose(module.lora_B, torch.zeros_like(module.lora_B))

    def test_set_masks(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4, attn_targets="qkvo")
        d_model = CLMConfig.toy().d_model

        masks = {}
        for key, module in model._rosa_modules():
            masks[key] = torch.rand(d_model, d_model) > 0.99

        model.set_masks(masks)
        for key, module in model._rosa_modules():
            assert module.sparse_enabled
            assert module.n_active == masks[key].sum().item()

    def test_n_active_sparse_params(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4, attn_targets="qkvo")
        d_model = CLMConfig.toy().d_model
        density = 0.01

        masks = {}
        for key, _ in model._rosa_modules():
            masks[key] = torch.rand(d_model, d_model) < density

        model.set_masks(masks)
        n_active = model.n_active_sparse_params()
        expected = sum(m.sum().item() for m in masks.values())
        assert n_active == expected

    def test_adapter_state_dict_roundtrip(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4)
        d_model = CLMConfig.toy().d_model

        # Set some masks
        masks = {}
        for key, _ in model._rosa_modules():
            masks[key] = torch.rand(d_model, d_model) > 0.99
        model.set_masks(masks)

        # Modify LoRA B weights
        for _, module in model._rosa_modules():
            module.lora_B.data.fill_(0.5)

        state = model.adapter_state_dict()

        # Verify masks are in state dict
        mask_keys = [k for k in state if k.startswith("mask/")]
        assert len(mask_keys) > 0

        # Create fresh model and load
        fresh_backbone = PAWNCLM(CLMConfig.toy())
        fresh_model = RoSACLM(fresh_backbone, rank=4)
        fresh_model.load_adapter_state_dict(state)

        for _, module in fresh_model._rosa_modules():
            assert module.sparse_enabled
            assert torch.allclose(module.lora_B, torch.tensor(0.5))

    def test_adapter_weight_report(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4)
        report = model.adapter_weight_report()
        # Should have entries for LoRA B norms
        assert any("lora_B" in k for k in report)

    def test_adapt_ffn(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4, adapt_ffn=True)
        rosa_keys = [k for k, _ in model._rosa_modules()]
        ffn_keys = [k for k in rosa_keys if "w_gate" in k or "w_up" in k or "w_down" in k]
        # 2 layers * 3 FFN projections = 6
        assert len(ffn_keys) == 2 * 3

    def test_gradient_flow(self, toy_backbone):
        model = RoSACLM(toy_backbone, rank=4)
        d_model = CLMConfig.toy().d_model

        # Enable sparse
        masks = {}
        for key, _ in model._rosa_modules():
            masks[key] = torch.rand(d_model, d_model) > 0.99
        model.set_masks(masks)

        input_ids = torch.randint(1, 4273, (2, 8))
        targets = torch.randint(1, 4273, (2, 8))
        hidden = model.forward_hidden(input_ids)
        logits = model.project_head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
        )
        loss.backward()

        # Both LoRA and sparse params should have gradients
        for _, module in model._rosa_modules():
            assert module.lora_A.grad is not None
            assert module.lora_B.grad is not None
            assert module.delta.grad is not None


class TestRetroBottleneckCLM:
    def test_forward_shapes(self, toy_backbone):
        # First inject sparse adapters via SparseCLM
        sparse_model = SparseCLM(toy_backbone, density=0.01)
        model = RetroBottleneckCLM(sparse_model.backbone, bottleneck_dim=4)

        input_ids = torch.randint(0, CLMConfig().vocab_size, (2, 16))
        hidden = model.forward_hidden(input_ids)
        assert hidden.shape == (2, 16, CLMConfig.toy().d_model)

        logits = model.forward(input_ids)
        assert logits.shape == (2, 16, CLMConfig().vocab_size)

    def test_forward_generate(self, toy_backbone):
        sparse_model = SparseCLM(toy_backbone, density=0.01)
        model = RetroBottleneckCLM(sparse_model.backbone, bottleneck_dim=4)

        input_ids = torch.randint(0, CLMConfig().vocab_size, (1, 5))
        logits, kv_cache = model.forward_generate(input_ids)
        assert logits.shape == (1, 1, CLMConfig().vocab_size)

    def test_adapter_parameters(self, toy_backbone):
        sparse_model = SparseCLM(toy_backbone, density=0.01)
        model = RetroBottleneckCLM(sparse_model.backbone, bottleneck_dim=4)
        params = model.adapter_parameters()
        # Should have both sparse delta params and bottleneck params
        assert len(params) > 0

    def test_separate_param_groups(self, toy_backbone):
        sparse_model = SparseCLM(toy_backbone, density=0.01)
        model = RetroBottleneckCLM(sparse_model.backbone, bottleneck_dim=4)
        sparse_params = model.sparse_parameters()
        bottleneck_params = model.bottleneck_parameters()
        # Both should be non-empty
        assert len(sparse_params) > 0
        assert len(bottleneck_params) > 0
        # No overlap
        sparse_ids = {id(p) for p in sparse_params}
        bottleneck_ids = {id(p) for p in bottleneck_params}
        assert sparse_ids.isdisjoint(bottleneck_ids)

    def test_identity_init(self, toy_backbone):
        """Bottleneck adapters should be identity-init (up=zeros)."""
        sparse_model = SparseCLM(toy_backbone, density=0.01)
        model = RetroBottleneckCLM(sparse_model.backbone, bottleneck_dim=4)
        from pawn.adapters.bottleneck import BottleneckAdapter
        for adapter in model.attn_adapters:
            if isinstance(adapter, BottleneckAdapter):
                assert torch.allclose(adapter.up.weight, torch.zeros_like(adapter.up.weight))

    def test_gradient_flow(self, toy_backbone):
        sparse_model = SparseCLM(toy_backbone, density=0.01)
        model = RetroBottleneckCLM(sparse_model.backbone, bottleneck_dim=4)

        input_ids = torch.randint(1, 4273, (2, 8))
        targets = torch.randint(1, 4273, (2, 8))
        hidden = model.forward_hidden(input_ids)
        logits = model.project_head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
        )
        loss.backward()

        grads = [p for p in model.adapter_parameters() if p.grad is not None]
        assert len(grads) > 0


class TestGenerateGradientMasks:
    def test_smoke(self, toy_backbone):
        """Mask generation should produce masks with correct shapes and density."""
        model = RoSACLM(toy_backbone, rank=4, attn_targets="qkvo")
        d_model = CLMConfig.toy().d_model
        density = 0.05

        # Create a simple synthetic dataloader
        batches = []
        for _ in range(4):
            B, T = 4, 32
            ids = torch.randint(1, 4273, (B, T))
            tgt = torch.randint(1, 4273, (B, T))
            msk = torch.ones(B, T, dtype=torch.bool)
            # Simple legal mask -- all True (no illegal moves)
            batches.append({
                "input_ids": ids,
                "targets": tgt,
                "loss_mask": msk,
            })

        masks = generate_gradient_masks(
            model, batches, SimpleMaskBuilder(),
            density=density, alpha=2,
            device="cpu", use_amp=False, max_batches=4,
        )

        # Check we got masks for all projections
        rosa_keys = [k for k, _ in model._rosa_modules()]
        assert set(masks.keys()) == set(rosa_keys)

        # Check shapes and approximate density
        for key, mask in masks.items():
            assert mask.shape == (d_model, d_model)
            assert mask.dtype == torch.bool
            actual_density = mask.float().mean().item()
            assert abs(actual_density - density) < 0.02, (
                f"{key}: expected density ~{density}, got {actual_density}"
            )

    def test_masks_are_informative(self, toy_backbone):
        """Masks from gradient accumulation should not be random-looking."""
        model = RoSACLM(toy_backbone, rank=4, attn_targets="qv")

        # Train LoRA for a few steps to build up signal
        optimizer = torch.optim.Adam(model.lora_parameters(), lr=1e-3)
        for _ in range(10):
            ids = torch.randint(1, 4273, (4, 16))
            logits = model.forward(ids)
            loss = logits.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batches = []
        for _ in range(4):
            batches.append({
                "input_ids": torch.randint(1, 4273, (4, 16)),
                "targets": torch.randint(1, 4273, (4, 16)),
                "loss_mask": torch.ones(4, 16, dtype=torch.bool),
            })

        masks = generate_gradient_masks(
            model, batches, SimpleMaskBuilder(),
            density=0.05, alpha=2,
            device="cpu", use_amp=False, max_batches=4,
        )

        # After warm-up, masks should exist and be non-trivial
        for key, mask in masks.items():
            assert mask.any(), f"Mask {key} is all-False"

    def test_delta_zeroed_after_generation(self, toy_backbone):
        """Delta parameters should be zero after mask generation."""
        model = RoSACLM(toy_backbone, rank=4, attn_targets="qv")

        batches = [{
            "input_ids": torch.randint(1, 4273, (2, 8)),
            "targets": torch.randint(1, 4273, (2, 8)),
            "loss_mask": torch.ones(2, 8, dtype=torch.bool),
        }]

        generate_gradient_masks(
            model, batches, SimpleMaskBuilder(),
            density=0.05, device="cpu", use_amp=False, max_batches=1,
        )

        for _, module in model._rosa_modules():
            assert torch.allclose(module.delta, torch.zeros_like(module.delta))
            # sparse_enabled and requires_grad should be restored to original state
            assert not module.sparse_enabled
            assert not module.delta.requires_grad
