"""Specialized (from-scratch) chess language model.

A lightweight transformer trained directly on Lichess games without a
pretrained backbone. Used as a baseline comparison against adapter
strategies — how well can N parameters do when trained from scratch
versus adapting a frozen PAWN backbone?

Architecture mirrors PAWN (RMSNorm, RoPE, SwiGLU-style FFN, weight-tied
embedding) but at much smaller scales (32-192 d_model, 2-4 layers).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawn.config import PAD_TOKEN


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


def _apply_rope(
    x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
) -> torch.Tensor:
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_r.unbind(-1)
    out0 = x0 * rope_cos - x1 * rope_sin
    out1 = x0 * rope_sin + x1 * rope_cos
    return torch.stack([out0, out1], dim=-1).reshape(x.shape).to(x.dtype)


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = _RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = _RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.wo(attn_out)
        h = self.ffn_norm(x)
        x = x + self.w2(F.gelu(self.w1(h)))
        return x


class SpecializedCLM(nn.Module):
    """Lightweight standalone chess language model (no pretrained backbone).

    Weight-tied embedding/output. Same interface as adapter CLMs:
    forward_hidden() + project_head().
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 256,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        pad_idx = PAD_TOKEN if vocab_size > PAD_TOKEN else None
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            _Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = _RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        head_dim = d_model // n_heads
        freqs = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer("rope_cos", angles.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("rope_sin", angles.sin().unsqueeze(0).unsqueeze(0))

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        T = x.shape[1]
        cos: torch.Tensor = self.rope_cos[:, :, :T, :]  # type: ignore[index]
        sin: torch.Tensor = self.rope_sin[:, :, :T, :]  # type: ignore[index]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def project_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)
