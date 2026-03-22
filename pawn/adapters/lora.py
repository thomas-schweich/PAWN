"""LoRA (Low-Rank Adaptation) for PAWN.

Injects rank-r adapters into Q, K, V, O attention projections (and optionally
FFN projections) in all transformer layers:

    output = frozen_linear(x) + (x @ A^T) @ B^T * (alpha / rank)

B is zero-initialized so the model starts identical to the frozen backbone.
Total trainable params (rank=4, attention-only): 131,072.
"""

import math

import torch
import torch.nn as nn

from pawn.config import CLMConfig
from pawn.model import PAWNCLM, Attention


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adapter.

    output = frozen_linear(x) + (x @ A^T) @ B^T * (alpha / rank)
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.frozen = frozen_linear
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        in_features = frozen_linear.in_features
        out_features = frozen_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.frozen(x)
        lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base + lora


ATTN_PRESETS = {
    "qkvo": ("wq", "wk", "wv", "wo"),
    "qv": ("wq", "wv"),
    "qkv": ("wq", "wk", "wv"),
}
_FFN_TARGETS = ("w_gate", "w_up", "w_down")


class LoRACLM(nn.Module):
    """Frozen PAWN backbone with LoRA adapters.

    LoRA is injected into attention projections (and optionally FFN) in every
    transformer layer. Only LoRA parameters are trainable.
    """

    def __init__(
        self,
        backbone: PAWNCLM,
        rank: int = 4,
        alpha: float | None = None,
        attn_targets: str | tuple[str, ...] = "qkvo",
        adapt_ffn: bool = False,
        layers: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.adapt_ffn = adapt_ffn

        # Resolve attention targets
        if isinstance(attn_targets, str):
            self.attn_targets = ATTN_PRESETS[attn_targets]
        else:
            self.attn_targets = tuple(attn_targets)

        # Resolve layer indices (default: all)
        n_layers = len(backbone.layers)
        self.adapted_layers = set(layers if layers is not None else range(n_layers))

        # Freeze the entire backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Inject LoRA into selected layers
        for layer_idx, block in enumerate(backbone.layers):
            if layer_idx not in self.adapted_layers:
                continue

            attn: Attention = block.attn
            for proj_name in self.attn_targets:
                original = getattr(attn, proj_name)
                setattr(attn, proj_name, LoRALinear(original, rank, self.alpha))

            if adapt_ffn:
                ffn = block.ffn
                for proj_name in _FFN_TARGETS:
                    original = getattr(ffn, proj_name)
                    setattr(ffn, proj_name, LoRALinear(original, rank, self.alpha))

    @property
    def cfg(self) -> CLMConfig:
        return self.backbone.cfg

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run backbone layers (with LoRA), return normed hidden states.

        Returns (B, T, d_model) -- before lm_head projection. Uses
        is_causal=True in SDPA (Flash Attention).
        """
        bb = self.backbone
        x = bb.embed(input_ids)

        T = input_ids.shape[1]
        rope_cos = bb.rope_cos[:, :, :T, :]
        rope_sin = bb.rope_sin[:, :, :T, :]

        for layer in bb.layers:
            x = layer(x, rope_cos, rope_sin, None)

        return bb.final_norm(x)

    def project_head(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states through lm_head.

        x: (*, d_model) -- works for both (B, T, d) and (N_valid, d).
        """
        return self.backbone.lm_head(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass. Returns logits (B, T, V).

        When attention_mask is None, uses is_causal=True in SDPA.
        """
        bb = self.backbone
        x = bb.embed(input_ids)

        T = input_ids.shape[1]
        if attention_mask is not None:
            causal = bb.causal_mask[:T, :T]
            padding = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = causal.unsqueeze(0) & padding
        else:
            mask = None

        rope_cos = bb.rope_cos[:, :, :T, :]
        rope_sin = bb.rope_sin[:, :, :T, :]

        for layer in bb.layers:
            x = layer(x, rope_cos, rope_sin, mask)

        x = bb.final_norm(x)
        return self.project_head(x)

    def forward_generate(
        self,
        input_ids: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward with KV-cache for autoregressive generation."""
        bb = self.backbone
        x = bb.embed(input_ids)

        T_new = input_ids.shape[1]
        if kv_cache is not None:
            T_cached = kv_cache[0][0].shape[2]
            rope_cos = bb.rope_cos[:, :, T_cached:T_cached + T_new, :]
            rope_sin = bb.rope_sin[:, :, T_cached:T_cached + T_new, :]
        else:
            rope_cos = bb.rope_cos[:, :, :T_new, :]
            rope_sin = bb.rope_sin[:, :, :T_new, :]

        new_kv_cache = []
        for i, layer in enumerate(bb.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = layer.forward_kv(x, rope_cos, rope_sin, layer_cache)
            new_kv_cache.append(new_cache)

        x = bb.final_norm(x[:, -1:, :])
        logits = bb.lm_head(x)

        return logits, new_kv_cache

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only trainable LoRA parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract LoRA weights for saving."""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]):
        """Load LoRA weights."""
        own = self.state_dict()
        for k, v in state.items():
            if k in own:
                own[k] = v
        self.load_state_dict(own, strict=False)

    def lora_weight_report(self) -> dict[str, float]:
        """Per-layer LoRA weight norms for monitoring."""
        report = {}
        for layer_idx, block in enumerate(self.backbone.layers):
            attn = block.attn
            for proj_name in self.attn_targets:
                module = getattr(attn, proj_name)
                if isinstance(module, LoRALinear):
                    report[f"layer{layer_idx}.{proj_name}.A"] = module.lora_A.data.norm().item()
                    report[f"layer{layer_idx}.{proj_name}.B"] = module.lora_B.data.norm().item()

            if self.adapt_ffn:
                ffn = block.ffn
                for proj_name in _FFN_TARGETS:
                    module = getattr(ffn, proj_name)
                    if isinstance(module, LoRALinear):
                        report[f"layer{layer_idx}.{proj_name}.A"] = module.lora_A.data.norm().item()
                        report[f"layer{layer_idx}.{proj_name}.B"] = module.lora_B.data.norm().item()

        return report
