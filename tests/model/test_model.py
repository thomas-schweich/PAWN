"""Tests for PAWN model (pawn/model.py).

Covers PAWNCLM, RMSNorm, Attention, SwiGLUFFN, TransformerBlock, RoPE,
CLMEmbedding, clm_loss, forward_train, forward_generate, KV-cache.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from pawn.config import (
    BLACK_CHECKMATES,
    CLMConfig,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    WHITE_CHECKMATES,
)
from pawn.model import (
    Attention,
    CLMEmbedding,
    PAWNCLM,
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
    _apply_rope,
    _precompute_rope_freqs,
    clm_loss,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    @pytest.mark.unit
    def test_output_shape(self):
        norm = RMSNorm(32)
        x = torch.randn(4, 16, 32)
        out = norm(x)
        assert out.shape == x.shape

    @pytest.mark.unit
    def test_matches_reference(self):
        """RMSNorm should match a direct numpy/torch reference implementation."""
        dim = 32
        eps = 1e-6
        norm = RMSNorm(dim, eps=eps)
        torch.manual_seed(0)
        norm.weight.data = torch.randn(dim) * 0.1 + 1.0
        x = torch.randn(2, 8, dim)

        # Reference: x * rsqrt(mean(x**2) + eps) * weight
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        expected = (x / rms) * norm.weight

        out = norm(x)
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.unit
    def test_scale_invariance(self):
        """RMSNorm(kx) ≈ RMSNorm(x) (up to scale bias from eps)."""
        norm = RMSNorm(32)
        x = torch.randn(4, 8, 32)
        out_x = norm(x)
        # Scaling by 100 produces the same output (RMSNorm is scale-invariant)
        out_100x = norm(x * 100.0)
        assert torch.allclose(out_x, out_100x, atol=1e-3)

    @pytest.mark.unit
    def test_weight_init_to_ones(self):
        norm = RMSNorm(16)
        assert torch.equal(norm.weight, torch.ones(16))

    @pytest.mark.unit
    def test_gradient_flow(self):
        norm = RMSNorm(16)
        x = torch.randn(2, 4, 16, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.unit
    def test_zero_input_finite(self):
        """Zero input should produce finite output due to eps."""
        norm = RMSNorm(16)
        x = torch.zeros(2, 4, 16)
        out = norm(x)
        assert torch.isfinite(out).all()
        # Zero input → zero output (since x * rsqrt(eps) * weight = 0)
        assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    @pytest.mark.unit
    def test_freq_formula(self):
        """Precomputed freqs match 1/base^(2i/dim) × t."""
        dim = 16
        max_len = 32
        base = 10000.0
        freqs = _precompute_rope_freqs(dim, max_len, base)
        assert freqs.shape == (max_len, dim // 2)

        # freqs[t, i] = t * (1 / base^(2i/dim))
        for t in (0, 1, 5, 20):
            for i in range(dim // 2):
                expected = t * (1.0 / (base ** (2 * i / dim)))
                assert math.isclose(freqs[t, i].item(), expected, rel_tol=1e-5)

    @pytest.mark.unit
    def test_freq_zero_at_t_zero(self):
        """At t=0 all freqs are 0 (identity rotation)."""
        freqs = _precompute_rope_freqs(16, 32, 10000.0)
        assert torch.equal(freqs[0], torch.zeros(8))

    @pytest.mark.unit
    def test_apply_rope_preserves_shape(self):
        B, H, T, D = 2, 4, 8, 16
        x = torch.randn(B, H, T, D)
        freqs = _precompute_rope_freqs(D, T, 10000.0)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        out = _apply_rope(x, cos, sin)
        assert out.shape == x.shape

    @pytest.mark.unit
    def test_apply_rope_preserves_norm(self):
        """Rotation preserves vector norm (per 2-d pair)."""
        B, H, T, D = 1, 1, 4, 8
        x = torch.randn(B, H, T, D)
        freqs = _precompute_rope_freqs(D, T, 10000.0)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        out = _apply_rope(x, cos, sin)
        # Per-pair norms preserved
        x_pairs = x.reshape(B, H, T, D // 2, 2)
        out_pairs = out.reshape(B, H, T, D // 2, 2)
        x_norms = x_pairs.pow(2).sum(-1)
        out_norms = out_pairs.pow(2).sum(-1)
        assert torch.allclose(x_norms, out_norms, atol=1e-5)

    @pytest.mark.unit
    def test_apply_rope_identity_at_t_zero(self):
        """Applying RoPE at position 0 is the identity."""
        B, H, T, D = 1, 1, 1, 8
        x = torch.randn(B, H, T, D)
        freqs = _precompute_rope_freqs(D, T, 10000.0)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        out = _apply_rope(x, cos, sin)
        # cos=1, sin=0 at t=0 → out=x
        assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# SwiGLUFFN
# ---------------------------------------------------------------------------


class TestSwiGLUFFN:
    @pytest.mark.unit
    def test_output_shape(self, toy_clm_config):
        ffn = SwiGLUFFN(toy_clm_config)
        x = torch.randn(2, 8, toy_clm_config.d_model)
        out = ffn(x)
        assert out.shape == x.shape

    @pytest.mark.unit
    def test_activation_formula(self, toy_clm_config):
        """Output = w_down(silu(w_gate(x)) * w_up(x))."""
        ffn = SwiGLUFFN(toy_clm_config)
        x = torch.randn(2, 4, toy_clm_config.d_model)

        gate = ffn.w_gate(x)
        up = ffn.w_up(x)
        expected = ffn.w_down(F.silu(gate) * up)

        out = ffn(x)
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.unit
    def test_no_bias_parameters(self, toy_clm_config):
        """All three linear layers are bias=False."""
        ffn = SwiGLUFFN(toy_clm_config)
        assert ffn.w_gate.bias is None
        assert ffn.w_up.bias is None
        assert ffn.w_down.bias is None

    @pytest.mark.unit
    def test_gradient_flow(self, toy_clm_config):
        ffn = SwiGLUFFN(toy_clm_config)
        x = torch.randn(2, 4, toy_clm_config.d_model, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestAttention:
    @pytest.mark.unit
    def test_output_shape(self, toy_clm_config):
        attn = Attention(toy_clm_config)
        B, T, D = 2, 8, toy_clm_config.d_model
        x = torch.randn(B, T, D)
        freqs = _precompute_rope_freqs(
            D // toy_clm_config.n_heads, T, toy_clm_config.rope_base
        )
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        out = attn(x, cos, sin, mask=None)
        assert out.shape == (B, T, D)

    @pytest.mark.unit
    def test_causal_mask_via_is_causal(self, toy_clm_config):
        """Passing mask=None uses is_causal=True."""
        attn = Attention(toy_clm_config)
        B, T, D = 1, 8, toy_clm_config.d_model
        torch.manual_seed(0)
        x = torch.randn(B, T, D)
        freqs = _precompute_rope_freqs(
            D // toy_clm_config.n_heads, T, toy_clm_config.rope_base
        )
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        out1 = attn(x, cos, sin, mask=None)
        # Causal: changing a future token at position T-1 should NOT change output at 0
        x2 = x.clone()
        x2[0, T - 1] = x2[0, T - 1] + 100.0
        out2 = attn(x2, cos, sin, mask=None)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)

    @pytest.mark.unit
    def test_forward_kv_no_cache_matches_forward(self, toy_clm_config):
        """forward_kv with no cache should equal forward (both causal)."""
        attn = Attention(toy_clm_config)
        B, T, D = 1, 4, toy_clm_config.d_model
        torch.manual_seed(0)
        x = torch.randn(B, T, D)
        freqs = _precompute_rope_freqs(
            D // toy_clm_config.n_heads, T, toy_clm_config.rope_base
        )
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        out1 = attn(x, cos, sin, mask=None)
        out2, cache = attn.forward_kv(x, cos, sin, None)
        assert torch.allclose(out1, out2, atol=1e-5)
        # Cache shape
        k, v = cache
        head_dim = D // toy_clm_config.n_heads
        assert k.shape == (B, toy_clm_config.n_heads, T, head_dim)
        assert v.shape == (B, toy_clm_config.n_heads, T, head_dim)

    @pytest.mark.unit
    def test_no_bias(self, toy_clm_config):
        attn = Attention(toy_clm_config)
        assert attn.wq.bias is None
        assert attn.wk.bias is None
        assert attn.wv.bias is None
        assert attn.wo.bias is None


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    @pytest.mark.unit
    def test_output_shape(self, toy_clm_config):
        block = TransformerBlock(toy_clm_config)
        B, T, D = 2, 8, toy_clm_config.d_model
        x = torch.randn(B, T, D)
        freqs = _precompute_rope_freqs(
            D // toy_clm_config.n_heads, T, toy_clm_config.rope_base
        )
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        out = block(x, cos, sin)
        assert out.shape == (B, T, D)

    @pytest.mark.unit
    def test_residual_structure(self, toy_clm_config):
        """Output = x + attn(norm(x)) + ffn(norm(x + attn(norm(x))))."""
        block = TransformerBlock(toy_clm_config)
        B, T, D = 1, 4, toy_clm_config.d_model
        torch.manual_seed(0)
        x = torch.randn(B, T, D)
        freqs = _precompute_rope_freqs(
            D // toy_clm_config.n_heads, T, toy_clm_config.rope_base
        )
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        mid = x + block.attn(block.attn_norm(x), cos, sin, None)
        expected = mid + block.ffn(block.ffn_norm(mid))
        out = block(x, cos, sin)
        assert torch.allclose(out, expected, atol=1e-5)

    @pytest.mark.unit
    def test_forward_kv_returns_new_cache(self, toy_clm_config):
        block = TransformerBlock(toy_clm_config)
        B, T, D = 1, 4, toy_clm_config.d_model
        torch.manual_seed(0)
        x = torch.randn(B, T, D)
        head_dim = D // toy_clm_config.n_heads
        freqs = _precompute_rope_freqs(head_dim, T, toy_clm_config.rope_base)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        out, cache = block.forward_kv(x, cos, sin, None)
        assert out.shape == (B, T, D)
        assert cache[0].shape == (B, toy_clm_config.n_heads, T, head_dim)


# ---------------------------------------------------------------------------
# CLMEmbedding
# ---------------------------------------------------------------------------


class TestCLMEmbedding:
    @pytest.mark.unit
    def test_output_shape(self, toy_clm_config):
        emb = CLMEmbedding(toy_clm_config)
        input_ids = torch.randint(0, 1968, (2, 8))
        out = emb(input_ids)
        assert out.shape == (2, 8, toy_clm_config.d_model)

    @pytest.mark.unit
    def test_pad_gets_pad_embed(self, toy_clm_config):
        """PAD token (1968) → pad_embed parameter (zero by default)."""
        emb = CLMEmbedding(toy_clm_config)
        input_ids = torch.tensor([[PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]])
        out = emb(input_ids)
        # pad_embed is initialized to zeros
        expected = emb.pad_embed.expand(1, 3, -1)
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.unit
    def test_pad_embed_zero_init(self, toy_clm_config):
        emb = CLMEmbedding(toy_clm_config)
        assert torch.equal(emb.pad_embed, torch.zeros(toy_clm_config.d_model))

    @pytest.mark.unit
    def test_outcome_token_routes_to_outcome_embed(self, toy_clm_config):
        """Outcome tokens (4273-4283) use outcome_embed, not src/dst/promo."""
        emb = CLMEmbedding(toy_clm_config)
        # Zero out all other embeddings so only outcome_embed matters
        with torch.no_grad():
            emb.src_embed.weight.zero_()
            emb.dst_embed.weight.zero_()
            emb.promo_embed.weight.zero_()
            emb.pad_embed.zero_()

        input_ids = torch.tensor([[WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE]])
        out = emb(input_ids)
        # Result should be exactly outcome_embed at those indices
        expected = torch.stack(
            [
                emb.outcome_embed.weight[0],
                emb.outcome_embed.weight[1],
                emb.outcome_embed.weight[2],
            ]
        ).unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.unit
    def test_move_token_is_factored(self, toy_clm_config):
        """Move token → src_embed[s] + dst_embed[d] + promo_embed[p]."""
        emb = CLMEmbedding(toy_clm_config)

        # Pick a non-promo move: token 1 (first move in vocab)
        token_id = 1
        decomp = emb.decomp_table[token_id]
        s, d, p = decomp[0].item(), decomp[1].item(), decomp[2].item()

        input_ids = torch.tensor([[token_id]])
        out = emb(input_ids)[0, 0]
        expected = (
            emb.src_embed.weight[s]
            + emb.dst_embed.weight[d]
            + emb.promo_embed.weight[p]
        )
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.unit
    def test_outcome_tokens_use_outcome_embed_not_decomp(self, toy_clm_config):
        """Outcome tokens are above the decomp table and overridden in forward().

        The decomp table covers only action tokens [0, NUM_ACTIONS).
        Outcome tokens (>= OUTCOME_TOKEN_BASE) and PAD (PAD_TOKEN) are
        clamped to the last table entry during lookup, but then overridden
        by outcome_embed and pad_embed respectively.  Verify the override
        produces the correct embedding.
        """
        from pawn.config import NUM_ACTIONS

        emb = CLMEmbedding(toy_clm_config)
        # Decomp table covers only action tokens
        assert emb.decomp_table.shape[0] == NUM_ACTIONS

        # Outcome tokens are above the table range
        assert OUTCOME_TOKEN_BASE >= NUM_ACTIONS

        # Verify outcome tokens produce outcome_embed, not factored embeddings
        with torch.no_grad():
            emb.src_embed.weight.zero_()
            emb.dst_embed.weight.zero_()
            emb.promo_embed.weight.zero_()
            emb.pad_embed.zero_()

        input_ids = torch.tensor([[WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE]])
        out = emb(input_ids)
        expected = torch.stack(
            [
                emb.outcome_embed.weight[0],
                emb.outcome_embed.weight[1],
                emb.outcome_embed.weight[2],
            ]
        ).unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.unit
    def test_decomp_table_shape(self, toy_clm_config):
        from pawn.config import NUM_ACTIONS
        emb = CLMEmbedding(toy_clm_config)
        assert emb.decomp_table.shape == (NUM_ACTIONS, 3)


# ---------------------------------------------------------------------------
# PAWNCLM forward
# ---------------------------------------------------------------------------


class TestPAWNCLMForward:
    @pytest.mark.unit
    def test_forward_shape(self, toy_model, toy_clm_config, synth_input_ids, full_mask):
        B, T = 2, 16
        ids = synth_input_ids(B, T, toy_clm_config.vocab_size)
        mask = full_mask(B, T)
        logits, layer_outputs = toy_model(ids, mask)
        assert logits.shape == (B, T, toy_clm_config.vocab_size)

    @pytest.mark.unit
    def test_layer_outputs_count(
        self, toy_model, toy_clm_config, synth_input_ids, full_mask
    ):
        """hidden_only=False returns embed + per-layer outputs."""
        B, T = 2, 8
        ids = synth_input_ids(B, T, toy_clm_config.vocab_size)
        mask = full_mask(B, T)
        _, layer_outputs = toy_model(ids, mask, hidden_only=False)
        assert len(layer_outputs) == toy_clm_config.n_layers + 1  # embed + each layer

    @pytest.mark.unit
    def test_layer_outputs_hidden_only(
        self, toy_model, toy_clm_config, synth_input_ids, full_mask
    ):
        B, T = 2, 8
        ids = synth_input_ids(B, T, toy_clm_config.vocab_size)
        mask = full_mask(B, T)
        _, layer_outputs = toy_model(ids, mask, hidden_only=True)
        assert len(layer_outputs) == 1
        assert layer_outputs[0].shape == (B, T, toy_clm_config.d_model)

    @pytest.mark.unit
    def test_layer_output_shapes(
        self, toy_model, toy_clm_config, synth_input_ids, full_mask
    ):
        B, T = 2, 8
        ids = synth_input_ids(B, T, toy_clm_config.vocab_size)
        mask = full_mask(B, T)
        _, layer_outputs = toy_model(ids, mask, hidden_only=False)
        for lo in layer_outputs:
            assert lo.shape == (B, T, toy_clm_config.d_model)

    @pytest.mark.unit
    def test_layer_output_first_is_embed(
        self, toy_model, toy_clm_config, synth_input_ids, full_mask
    ):
        """When hidden_only=False, layer_outputs[0] is the raw embedding."""
        B, T = 2, 4
        ids = synth_input_ids(B, T, toy_clm_config.vocab_size)
        mask = full_mask(B, T)
        _, layer_outputs = toy_model(ids, mask, hidden_only=False)
        expected_emb = toy_model.embed(ids)
        assert torch.allclose(layer_outputs[0], expected_emb, atol=1e-6)

    @pytest.mark.unit
    def test_seq_len_exceeds_max_raises(
        self, toy_model, toy_clm_config, synth_input_ids, full_mask
    ):
        T = toy_clm_config.max_seq_len + 1
        ids = synth_input_ids(1, T, toy_clm_config.vocab_size)
        mask = full_mask(1, T)
        with pytest.raises(ValueError, match="exceeds max"):
            toy_model(ids, mask)

    @pytest.mark.unit
    def test_no_nan_with_padding(self, toy_model, toy_clm_config):
        B, T = 2, 16
        ids = torch.randint(0, 1968, (B, T))
        ids[:, 10:] = PAD_TOKEN
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :11] = True
        logits, _ = toy_model(ids, mask)
        assert torch.isfinite(logits).all()

    @pytest.mark.unit
    def test_causal_in_forward(self, toy_model, toy_clm_config, full_mask):
        """Future tokens should not affect past logits (causal)."""
        B, T = 1, 8
        torch.manual_seed(0)
        ids = torch.randint(0, 1968, (B, T))
        mask = full_mask(B, T)
        logits1, _ = toy_model(ids, mask)

        ids2 = ids.clone()
        ids2[0, T - 1] = (ids2[0, T - 1] + 100) % 1968
        logits2, _ = toy_model(ids2, mask)
        # Logits at position 0 should be identical
        assert torch.allclose(logits1[0, 0], logits2[0, 0], atol=1e-5)

    @pytest.mark.unit
    def test_deterministic(self, toy_model, toy_clm_config, full_mask):
        B, T = 2, 8
        torch.manual_seed(0)
        ids = torch.randint(0, toy_clm_config.vocab_size, (B, T))
        mask = full_mask(B, T)
        logits1, _ = toy_model(ids, mask)
        logits2, _ = toy_model(ids, mask)
        assert torch.equal(logits1, logits2)

    @pytest.mark.unit
    def test_param_init_not_zero(self, toy_clm_config):
        """Verify initialization scheme: all 2D params nonzero, RMSNorm=1, pad=0."""
        model = PAWNCLM(toy_clm_config)

        # ALL 2D parameters should have non-zero std (properly initialized)
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                assert p.std().item() > 0, (
                    f"{name}: 2D parameter has zero std (bad init)"
                )

        # RMSNorm weights should be initialized to 1.0
        for name, p in model.named_parameters():
            if "norm" in name.lower() and "weight" in name and p.dim() == 1:
                assert torch.allclose(p, torch.ones_like(p)), (
                    f"{name}: RMSNorm weight not initialized to 1.0"
                )

        # Pad embedding should be initialized to zero
        assert torch.equal(
            model.embed.pad_embed, torch.zeros(toy_clm_config.d_model)
        ), "pad_embed should be initialized to zero"


# ---------------------------------------------------------------------------
# PAWNCLM forward_train
# ---------------------------------------------------------------------------


class TestForwardTrain:
    @pytest.mark.unit
    def test_returns_loss_and_metrics(self, toy_model, small_clm_sequences):
        loss, metrics = toy_model.forward_train(
            small_clm_sequences["input_ids"],
            small_clm_sequences["loss_mask"],
            small_clm_sequences["targets"],
        )
        assert loss.dim() == 0
        assert "loss" in metrics
        assert "accuracy" in metrics

    @pytest.mark.unit
    def test_metrics_are_tensors(self, toy_model, small_clm_sequences):
        """Metrics should be GPU/CPU tensors, not .item() floats (perf)."""
        _, metrics = toy_model.forward_train(
            small_clm_sequences["input_ids"],
            small_clm_sequences["loss_mask"],
            small_clm_sequences["targets"],
        )
        assert isinstance(metrics["loss"], torch.Tensor)
        assert isinstance(metrics["accuracy"], torch.Tensor)

    @pytest.mark.unit
    def test_loss_matches_manual(self, toy_model, small_clm_sequences):
        """forward_train loss should match a manual full-projection loss."""
        ids = small_clm_sequences["input_ids"]
        mask = small_clm_sequences["loss_mask"]
        tgt = small_clm_sequences["targets"]

        loss_train, _ = toy_model.forward_train(ids, mask, tgt)

        # Manual path: forward + cross-entropy on masked positions
        logits, _ = toy_model(ids, mask)
        flat_logits = logits[mask]
        flat_tgt = tgt[mask]
        manual_loss = F.cross_entropy(flat_logits, flat_tgt)

        assert torch.allclose(loss_train, manual_loss, atol=1e-4)

    @pytest.mark.unit
    def test_gradient_flow(self, toy_clm_config, small_clm_sequences):
        model = PAWNCLM(toy_clm_config)
        loss, _ = model.forward_train(
            small_clm_sequences["input_ids"],
            small_clm_sequences["loss_mask"],
            small_clm_sequences["targets"],
        )
        loss.backward()
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert n_with_grad > 0

    @pytest.mark.unit
    def test_accuracy_in_range(self, toy_model, small_clm_sequences):
        """Accuracy must be in [0, 1] and an untrained model should not be perfect."""
        _, metrics = toy_model.forward_train(
            small_clm_sequences["input_ids"],
            small_clm_sequences["loss_mask"],
            small_clm_sequences["targets"],
        )
        acc = metrics["accuracy"].item()
        assert 0.0 <= acc <= 1.0

        # An untrained toy model on random-ish data should not achieve perfect
        # accuracy (vocab=1980, so chance ≈ 0.05%).  Strictly less than 1.0.
        assert acc < 1.0, "untrained model should not predict perfectly"

        # Cross-check: recompute accuracy from the logits and targets manually
        ids = small_clm_sequences["input_ids"]
        mask = small_clm_sequences["loss_mask"]
        tgt = small_clm_sequences["targets"]
        logits, _ = toy_model(ids, mask)
        preds = logits[mask].argmax(dim=-1)
        expected_acc = (preds == tgt[mask]).float().mean().item()
        assert abs(acc - expected_acc) < 1e-5, (
            f"forward_train accuracy {acc} != manual accuracy {expected_acc}"
        )

    @pytest.mark.unit
    def test_seq_len_exceeds_max_raises(self, toy_model, toy_clm_config):
        T = toy_clm_config.max_seq_len + 1
        ids = torch.randint(0, toy_clm_config.vocab_size, (1, T))
        mask = torch.ones(1, T, dtype=torch.bool)
        tgt = torch.randint(0, toy_clm_config.vocab_size, (1, T))
        with pytest.raises(ValueError, match="exceeds max"):
            toy_model.forward_train(ids, mask, tgt)


# ---------------------------------------------------------------------------
# KV-cache (forward_generate)
# ---------------------------------------------------------------------------


class TestKVCache:
    @pytest.mark.unit
    def test_prefill_output_shape(self, toy_model, toy_clm_config, synth_input_ids):
        B, T = 2, 6
        ids = synth_input_ids(B, T, toy_clm_config.vocab_size)
        logits, cache = toy_model.forward_generate(ids, kv_cache=None)
        assert logits.shape == (B, 1, toy_clm_config.vocab_size)
        assert len(cache) == toy_clm_config.n_layers

    @pytest.mark.unit
    def test_cache_extends_on_decode(
        self, toy_model, toy_clm_config, synth_input_ids
    ):
        B = 1
        prefix = synth_input_ids(B, 4, toy_clm_config.vocab_size)
        _, cache = toy_model.forward_generate(prefix, kv_cache=None)
        head_dim = toy_clm_config.d_model // toy_clm_config.n_heads
        assert cache[0][0].shape == (B, toy_clm_config.n_heads, 4, head_dim)

        # Now feed a single new token
        nxt = synth_input_ids(B, 1, toy_clm_config.vocab_size, seed=1)
        _, cache2 = toy_model.forward_generate(nxt, kv_cache=cache)
        assert cache2[0][0].shape == (B, toy_clm_config.n_heads, 5, head_dim)

    @pytest.mark.unit
    def test_kv_equivalent_to_full_forward(
        self, toy_model, toy_clm_config, synth_input_ids, full_mask
    ):
        """Stepwise KV-cache decode should match full forward at last position."""
        B = 1
        prefix = synth_input_ids(B, 5, toy_clm_config.vocab_size)
        next_tok = synth_input_ids(B, 1, toy_clm_config.vocab_size, seed=1)

        # Full forward on the combined sequence
        full = torch.cat([prefix, next_tok], dim=1)
        full_mask_t = full_mask(B, 6)
        full_logits, _ = toy_model(full, full_mask_t)
        expected_last_logits = full_logits[:, -1:, :]

        # KV-cache path: prefill prefix, then decode single token
        _, cache = toy_model.forward_generate(prefix, kv_cache=None)
        decode_logits, _ = toy_model.forward_generate(next_tok, kv_cache=cache)

        assert torch.allclose(decode_logits, expected_last_logits, atol=1e-4)

    @pytest.mark.unit
    def test_kv_multiple_decode_steps(
        self, toy_model, toy_clm_config, synth_input_ids, full_mask
    ):
        """Multiple sequential decode steps should match full forward at each step."""
        B = 1
        n_decode = 3
        prefix = synth_input_ids(B, 4, toy_clm_config.vocab_size)
        decodes = [
            synth_input_ids(B, 1, toy_clm_config.vocab_size, seed=i + 1)
            for i in range(n_decode)
        ]

        _, cache = toy_model.forward_generate(prefix, kv_cache=None)
        kv_outputs = []
        for tok in decodes:
            logits, cache = toy_model.forward_generate(tok, kv_cache=cache)
            kv_outputs.append(logits)

        # Full forward on entire sequence
        full = torch.cat([prefix, *decodes], dim=1)
        full_mask_t = full_mask(B, full.shape[1])
        full_logits, _ = toy_model(full, full_mask_t)

        # Compare each decoded step with the corresponding position
        for i, kv in enumerate(kv_outputs):
            pos = 4 + i
            expected = full_logits[:, pos : pos + 1, :]
            assert torch.allclose(kv, expected, atol=1e-4)

    @pytest.mark.unit
    def test_kv_seq_len_exceeds_max_raises(self, toy_model, toy_clm_config):
        T = toy_clm_config.max_seq_len + 1
        ids = torch.zeros(1, T, dtype=torch.long)
        with pytest.raises(ValueError, match="exceeds max"):
            toy_model.forward_generate(ids, kv_cache=None)


# ---------------------------------------------------------------------------
# clm_loss
# ---------------------------------------------------------------------------


class TestCLMLoss:
    @pytest.mark.unit
    def test_returns_scalar(self):
        B, T, V = 2, 8, 32
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)
        loss, metrics = clm_loss(logits, targets, mask)
        assert loss.dim() == 0
        assert loss.item() > 0

    @pytest.mark.unit
    def test_ignores_masked_positions(self):
        """Loss should only depend on masked-in positions."""
        B, T, V = 1, 4, 8
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask = torch.tensor([[True, True, False, False]])

        loss1, _ = clm_loss(logits, targets, mask)

        # Changing masked-out targets should not change the loss
        targets2 = targets.clone()
        targets2[0, 2] = (targets2[0, 2] + 1) % V
        targets2[0, 3] = (targets2[0, 3] + 3) % V
        loss2, _ = clm_loss(logits, targets2, mask)
        assert torch.allclose(loss1, loss2, atol=1e-6)

    @pytest.mark.unit
    def test_matches_manual_cross_entropy(self):
        B, T, V = 1, 4, 8
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask = torch.tensor([[True, True, True, False]])

        loss, _ = clm_loss(logits, targets, mask)

        # Manual: cross entropy over first 3 positions only
        manual = F.cross_entropy(logits[mask], targets[mask])
        assert torch.allclose(loss, manual, atol=1e-6)

    @pytest.mark.unit
    def test_accuracy_perfect_predictions(self):
        """Perfect predictions → accuracy = 1.0."""
        B, T, V = 2, 4, 8
        targets = torch.randint(0, V, (B, T))
        logits = F.one_hot(targets, V).float() * 100.0
        mask = torch.ones(B, T, dtype=torch.bool)
        _, metrics = clm_loss(logits, targets, mask)
        assert metrics["accuracy"] == 1.0

    @pytest.mark.unit
    def test_accuracy_all_wrong(self):
        B, T, V = 2, 4, 8
        targets = torch.zeros(B, T, dtype=torch.long)
        # Force predictions to token V-1 (not 0)
        logits = torch.zeros(B, T, V)
        logits[..., V - 1] = 100.0
        mask = torch.ones(B, T, dtype=torch.bool)
        _, metrics = clm_loss(logits, targets, mask)
        assert metrics["accuracy"] == 0.0

    @pytest.mark.unit
    def test_all_masked_out(self):
        """Edge: entirely masked out. Loss should be NaN from CE (ignore_index all)."""
        B, T, V = 1, 4, 8
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask = torch.zeros(B, T, dtype=torch.bool)
        loss, _ = clm_loss(logits, targets, mask)
        # cross_entropy with all-ignore returns nan
        assert torch.isnan(loss) or loss.item() == 0.0


# ---------------------------------------------------------------------------
# PAWNCLM weight initialization
# ---------------------------------------------------------------------------


class TestInit:
    @pytest.mark.unit
    def test_init_weights_std(self, toy_clm_config):
        """2D params should be initialized with roughly std=0.02."""
        model = PAWNCLM(toy_clm_config)
        # Collect std of linear weights
        for name, p in model.named_parameters():
            if p.dim() > 1 and "weight" in name and p.numel() > 100:
                std = p.std().item()
                # Tolerate range 0.01-0.04 (small params bounce)
                assert 0.005 < std < 0.05, f"{name}: std={std}"

    @pytest.mark.unit
    def test_param_count_toy(self, toy_model):
        n = sum(p.numel() for p in toy_model.parameters())
        assert n < 1_000_000

    @pytest.mark.unit
    def test_buffers_registered(self, toy_model):
        """rope_cos, rope_sin, causal_mask should be buffers, not params."""
        buffer_names = {name for name, _ in toy_model.named_buffers()}
        assert "rope_cos" in buffer_names
        assert "rope_sin" in buffer_names
        assert "causal_mask" in buffer_names


# ---------------------------------------------------------------------------
# get_block accessor
# ---------------------------------------------------------------------------


class TestGetBlock:
    @pytest.mark.unit
    def test_returns_transformer_block(self, toy_model, toy_clm_config):
        for i in range(toy_clm_config.n_layers):
            blk = toy_model.get_block(i)
            assert isinstance(blk, TransformerBlock)
