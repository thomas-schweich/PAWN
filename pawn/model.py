"""PAWN: Causal Language Model for chess move prediction.

Decoder-only transformer (`Vaswani et al., 2017
<https://arxiv.org/abs/1706.03762>`_) with next-token prediction over
the move vocabulary.  Key architectural choices drawn from subsequent
work:

* **RMSNorm** -- `Zhang & Sennrich, 2019 <https://arxiv.org/abs/1910.07467>`_
* **SwiGLU** FFN -- `Shazeer, 2020 <https://arxiv.org/abs/2002.05202>`_
* **Rotary Position Embeddings (RoPE)** -- `Su et al., 2021
  <https://arxiv.org/abs/2104.09864>`_
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from pawn.config import CLMConfig, OUTCOME_TOKEN_BASE
from chess_engine import export_move_vocabulary

# ROCm flash attention backward has stride mismatches with torch.compile.
# Set this to use MATH backend instead (enables compile + AMP on ROCm).
SDPA_BACKEND: SDPBackend | None = None


def _build_decomposition_table() -> torch.Tensor:
    """Build static token -> (src, dst, promo_type) lookup table.

    Returns int16[4278, 3]. PAD (0) and outcome tokens (4273-4277)
    map to (0, 0, 0) — handled by standalone embeddings.
    """
    vocab = export_move_vocabulary()
    table = torch.zeros(4278, 3, dtype=torch.int16)

    for token_idx, uci_str in vocab["token_to_move"].items():
        if token_idx >= OUTCOME_TOKEN_BASE:
            continue  # Outcome tokens use standalone embeddings
        src_name = uci_str[:2]
        dst_name = uci_str[2:4]
        promo_suffix = uci_str[4:] if len(uci_str) > 4 else ""

        sq_names = vocab["square_names"]
        src_sq = sq_names.index(src_name)
        dst_sq = sq_names.index(dst_name)

        promo_type = 0
        if promo_suffix:
            promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
            promo_type = promo_map[promo_suffix]

        table[token_idx] = torch.tensor([src_sq, dst_sq, promo_type], dtype=torch.int16)

    return table


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (`Zhang & Sennrich, 2019
    <https://arxiv.org/abs/1910.07467>`_)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float() if x.dtype != torch.float32 else x
        norm = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * norm).to(x.dtype) * self.weight


def _apply_rope(
    x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings (`Su et al., 2021
    <https://arxiv.org/abs/2104.09864>`_).

    x: (B, n_heads, T, head_dim)
    rope_cos, rope_sin: (1, 1, T, head_dim // 2)
    """
    x_f = x.float() if x.dtype != torch.float32 else x
    x_r = x_f.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_r.unbind(-1)

    out0 = x0 * rope_cos - x1 * rope_sin
    out1 = x0 * rope_sin + x1 * rope_cos

    out = torch.stack([out0, out1], dim=-1).reshape(x.shape)
    return out.to(x.dtype) if x.dtype != torch.float32 else out


def _precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor. Returns (max_len, dim // 2)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs


class Attention(nn.Module):
    def __init__(self, cfg: CLMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        if SDPA_BACKEND is not None:
            with sdpa_kernel(SDPA_BACKEND):
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=(mask is None)
                )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=(mask is None)
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn_out)

    def forward_kv(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward with KV-cache for autoregressive generation.

        Args:
            x: (B, T_new, d_model) — full sequence for prefill, single token for decode.
            rope_cos/sin: (1, 1, T_new, head_dim//2) — RoPE for the new positions only.
            kv_cache: optional (K, V) each (B, n_heads, T_cached, head_dim).

        Returns:
            out: (B, T_new, d_model)
            new_cache: (K, V) each (B, n_heads, T_total, head_dim)
        """
        B, T_new, _ = x.shape

        q = self.wq(x).view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # Prefill (no cache): causal mask. Decode (with cache): single query
        # attends to all cached keys — no mask needed.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(kv_cache is None)
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_new, -1)
        return self.wo(attn_out), (k, v)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (`Shazeer, 2020
    <https://arxiv.org/abs/2002.05202>`_)."""

    def __init__(self, cfg: CLMConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    attn_norm: RMSNorm
    attn: Attention
    ffn_norm: RMSNorm
    ffn: SwiGLUFFN

    def __init__(self, cfg: CLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward_kv(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward with KV-cache."""
        attn_out, new_cache = self.attn.forward_kv(
            self.attn_norm(x), rope_cos, rope_sin, kv_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class CLMEmbedding(nn.Module):
    """Factored input embeddings for CLM.

    Move tokens use factored embedding: src_embed[s] + dst_embed[d] + promo_embed[p].
    PAD and outcome tokens use standalone embeddings.
    """

    decomp_table: torch.Tensor

    def __init__(self, cfg: CLMConfig):
        super().__init__()
        self.d_model = cfg.d_model

        # Factored move components
        self.src_embed = nn.Embedding(64, cfg.d_model)
        self.dst_embed = nn.Embedding(64, cfg.d_model)
        self.promo_embed = nn.Embedding(5, cfg.d_model)  # 0=none, 1=q, 2=r, 3=b, 4=n

        # Standalone embeddings
        self.pad_embed = nn.Parameter(torch.zeros(cfg.d_model))
        self.outcome_embed = nn.Embedding(cfg.n_outcomes, cfg.d_model)

        # Static decomposition table: token_idx -> (src, dst, promo_type)
        self.register_buffer("decomp_table", _build_decomposition_table(), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T) int tensor of token indices [0..4277]
        Returns: (B, T, d_model)
        """
        # Decompose all tokens (PAD and outcomes get (0,0,0) from the table —
        # their factored embeddings are garbage but will be overridden below)
        flat = input_ids.long().clamp(0, 4277)
        decomp = self.decomp_table[flat]  # (B, T, 3)
        src_idx = decomp[..., 0].long()
        dst_idx = decomp[..., 1].long()
        promo_idx = decomp[..., 2].long()

        emb = self.src_embed(src_idx) + self.dst_embed(dst_idx) + self.promo_embed(promo_idx)

        # Override PAD positions (branchless for torch.compile)
        pad_mask = (input_ids == 0).unsqueeze(-1)  # (B, T, 1)
        emb = torch.where(pad_mask, self.pad_embed, emb)

        # Override outcome token positions (branchless)
        # Compute outcome embeddings for ALL positions (clamp makes non-outcome
        # indices safe); torch.where selects only at actual outcome positions.
        outcome_idx = (input_ids - OUTCOME_TOKEN_BASE).clamp(0, self.outcome_embed.num_embeddings - 1)
        outcome_embs = self.outcome_embed(outcome_idx)
        outcome_mask = (input_ids >= OUTCOME_TOKEN_BASE).unsqueeze(-1)  # (B, T, 1)
        emb = torch.where(outcome_mask, outcome_embs, emb)

        return emb


class PAWNCLM(nn.Module):
    """PAWN: Causal Language Model for chess.

    Predicts the next token (move or padding) via softmax over the
    full vocabulary. No factored output head, no grid, no BCE.
    """

    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    causal_mask: torch.Tensor
    embed: CLMEmbedding
    layers: nn.ModuleList
    final_norm: RMSNorm
    lm_head: nn.Linear

    def __init__(self, cfg: CLMConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = CLMEmbedding(cfg)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Static buffers
        rope_freqs = _precompute_rope_freqs(
            cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_base
        )
        self.register_buffer(
            "rope_cos", rope_freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "rope_sin", rope_freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False
        )
        self.register_buffer(
            "causal_mask",
            torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool).tril(),
            persistent=False,
        )

        self._init_weights()

    def get_block(self, i: int) -> TransformerBlock:
        """Typed accessor for transformer layers (avoids ModuleList type erasure)."""
        return self.layers[i]  # type: ignore[return-value]

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_only: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        input_ids: (B, T) token indices
        attention_mask: (B, T) bool — True for real tokens (outcome + moves)
        hidden_only: if True, skip intermediate layer collection and return
            only the final hidden state in layer_outputs.

        Returns:
            logits: (B, T, vocab_size)
            layer_outputs: list of (B, T, d_model) from each layer
        """
        x = self.embed(input_ids)

        T = input_ids.shape[1]
        if T > self.rope_cos.shape[2]:
            raise ValueError(
                f"Sequence length {T} exceeds max {self.rope_cos.shape[2]}"
            )
        causal = self.causal_mask[:T, :T]
        padding = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        mask = causal.unsqueeze(0) & padding  # (B, 1, T, T)

        rope_cos = self.rope_cos[:, :, :T, :]
        rope_sin = self.rope_sin[:, :, :T, :]

        if hidden_only:
            for layer in self.layers:
                x = layer(x, rope_cos, rope_sin, mask)
            layer_outputs = [x]
        else:
            layer_outputs = [x]  # embedding output
            for layer in self.layers:
                x = layer(x, rope_cos, rope_sin, mask)
                layer_outputs.append(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, layer_outputs

    def forward_train(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Training-optimized forward: computes lm_head only at non-padding
        positions to avoid materializing the full (B, T, vocab_size) logits
        tensor. Returns loss and metrics directly.

        Metrics are returned as raw GPU tensors to avoid CUDA synchronization.
        Call .item() on them only when you need to log (e.g. every N steps).

        Args:
            input_ids: (B, T) token indices
            loss_mask: (B, T) bool — True for positions included in loss
                       (outcome + moves, not padding). Also used as the
                       attention padding mask for SDPA.
            targets: (B, T) target token indices (padding positions ignored)

        Returns:
            loss: scalar tensor (for backward)
            metrics: dict with loss and accuracy as GPU tensors (no .item())
        """
        x = self.embed(input_ids)

        T = input_ids.shape[1]
        if T > self.rope_cos.shape[2]:
            raise ValueError(
                f"Sequence length {T} exceeds max {self.rope_cos.shape[2]}"
            )
        causal = self.causal_mask[:T, :T]
        padding = loss_mask.unsqueeze(1).unsqueeze(2)
        mask = causal.unsqueeze(0) & padding

        rope_cos = self.rope_cos[:, :, :T, :]
        rope_sin = self.rope_sin[:, :, :T, :]

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, mask)

        x = self.final_norm(x)

        # Project only valid positions through lm_head to save ~25% memory
        valid_x = x[loss_mask]                       # (N_valid, d_model)
        valid_logits = self.lm_head(valid_x)         # (N_valid, vocab_size)
        valid_targets = targets[loss_mask]            # (N_valid,)

        loss = F.cross_entropy(valid_logits, valid_targets)

        with torch.no_grad():
            preds = valid_logits.argmax(dim=-1)
            accuracy = (preds == valid_targets).float().mean()

        return loss, {"loss": loss.detach(), "accuracy": accuracy}

    def forward_eval(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return final hidden states (B, T, d_model) without lm_head projection.

        Use for memory-efficient evaluation: project only needed positions
        through lm_head on the caller side instead of materializing the full
        (B, T, vocab_size) logits tensor.
        """
        x = self.embed(input_ids)

        T = input_ids.shape[1]
        if T > self.rope_cos.shape[2]:
            raise ValueError(
                f"Sequence length {T} exceeds max {self.rope_cos.shape[2]}"
            )
        causal = self.causal_mask[:T, :T]
        padding = attention_mask.unsqueeze(1).unsqueeze(2)
        mask = causal.unsqueeze(0) & padding

        rope_cos = self.rope_cos[:, :, :T, :]
        rope_sin = self.rope_sin[:, :, :T, :]

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, mask)

        return self.final_norm(x)

    def forward_generate(
        self,
        input_ids: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with KV-cache for autoregressive generation.

        Prefill (kv_cache=None): processes full input, builds cache.
        Decode (kv_cache provided): processes single new token, extends cache.

        Always returns logits for the last position only to save memory.

        Args:
            input_ids: (B, T) for prefill, (B, 1) for decode.
            kv_cache: None for prefill, list of (K, V) per layer for decode.

        Returns:
            logits: (B, 1, vocab_size)
            new_kv_cache: list of (K, V) per layer.
        """
        x = self.embed(input_ids)

        T_new = input_ids.shape[1]
        T_total = T_new
        if kv_cache is not None:
            T_cached = kv_cache[0][0].shape[2]
            T_total = T_cached + T_new
            rope_cos = self.rope_cos[:, :, T_cached:T_total, :]
            rope_sin = self.rope_sin[:, :, T_cached:T_total, :]
        else:
            rope_cos = self.rope_cos[:, :, :T_new, :]
            rope_sin = self.rope_sin[:, :, :T_new, :]
        if T_total > self.rope_cos.shape[2]:
            raise ValueError(
                f"Sequence length {T_total} exceeds max {self.rope_cos.shape[2]}"
            )

        new_kv_cache = []
        for i in range(len(self.layers)):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = self.get_block(i).forward_kv(x, rope_cos, rope_sin, layer_cache)
            new_kv_cache.append(new_cache)

        x = self.final_norm(x[:, -1:, :])
        logits = self.lm_head(x)

        return logits, new_kv_cache


_IGNORE_INDEX = -100


def clm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute CLM cross-entropy loss on non-padding positions.

    Uses ignore_index on a flat view to avoid materializing a copy of
    all valid-position logits (which would be ~50K × 4278 floats).

    Args:
        logits: (B, T, vocab_size)
        targets: (B, T) target token indices
        loss_mask: (B, T) bool — True for positions included in loss

    Returns:
        loss: scalar
        metrics: dict with loss value and accuracy
    """
    B, T, V = logits.shape

    # Flat views — no copy
    logits_flat = logits.view(-1, V)
    # Set padding targets to ignore_index so cross_entropy skips them
    targets_flat = torch.where(loss_mask.view(-1), targets.view(-1), _IGNORE_INDEX)

    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=_IGNORE_INDEX)

    # Top-1 accuracy (only at valid positions)
    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        valid = targets_flat != _IGNORE_INDEX
        accuracy = (preds[valid] == targets_flat[valid]).float().mean().item()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
    }

    return loss, metrics
