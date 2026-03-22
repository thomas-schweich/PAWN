"""Bottleneck adapters for PAWN.

Inserts small residual MLP bottlenecks after the attention sublayer and/or
the FFN sublayer within each transformer block (Houlsby et al. 2019):

    x = x + up(gelu(down(x)))

The up-projection is zero-initialized so the model starts identical to
the frozen backbone. bottleneck_dim controls the parameter budget.

Total trainable params (bottleneck_dim=8, both positions, 8 layers):
    2 × 8 × 2 × 512 × 8 = 131,072
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawn.config import CLMConfig
from pawn.model import PAWNCLM


class BottleneckAdapter(nn.Module):
    """Residual bottleneck: x + up(gelu(down(x)))."""

    def __init__(self, d_model: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


class BottleneckCLM(nn.Module):
    """Frozen PAWN backbone with bottleneck adapters.

    Adapters are inserted after the attention sublayer and/or FFN sublayer
    within each transformer block. Only adapter parameters are trainable.
    """

    def __init__(
        self,
        backbone: PAWNCLM,
        bottleneck_dim: int = 8,
        adapt_attn: bool = True,
        adapt_ffn: bool = True,
        layers: tuple[int, ...] | None = None,
        attn_layers: tuple[int, ...] | None = None,
        ffn_layers: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.bottleneck_dim = bottleneck_dim
        self.adapt_attn = adapt_attn
        self.adapt_ffn = adapt_ffn
        cfg = backbone.cfg
        n_layers = len(backbone.layers)

        self.adapted_layers = set(layers if layers is not None else range(n_layers))

        # Per-layer overrides: if attn_layers/ffn_layers are specified,
        # they take precedence over the global adapt_attn/adapt_ffn flags.
        if attn_layers is not None:
            self._attn_set = set(attn_layers)
        elif adapt_attn:
            self._attn_set = set(self.adapted_layers)
        else:
            self._attn_set = set()

        if ffn_layers is not None:
            self._ffn_set = set(ffn_layers)
        elif adapt_ffn:
            self._ffn_set = set(self.adapted_layers)
        else:
            self._ffn_set = set()

        # Freeze the entire backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Create adapter modules (Identity for non-adapted layers)
        self.attn_adapters = nn.ModuleList()
        self.ffn_adapters = nn.ModuleList()
        for i in range(n_layers):
            if i in self._attn_set:
                self.attn_adapters.append(BottleneckAdapter(cfg.d_model, bottleneck_dim))
            else:
                self.attn_adapters.append(nn.Identity())
            if i in self._ffn_set:
                self.ffn_adapters.append(BottleneckAdapter(cfg.d_model, bottleneck_dim))
            else:
                self.ffn_adapters.append(nn.Identity())

    @property
    def cfg(self) -> CLMConfig:
        return self.backbone.cfg

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run backbone sublayers with adapters, return normed hidden states."""
        bb = self.backbone
        x = bb.embed(input_ids)

        T = input_ids.shape[1]
        rope_cos = bb.rope_cos[:, :, :T, :]
        rope_sin = bb.rope_sin[:, :, :T, :]

        for i in range(len(bb.layers)):
            block = bb.get_block(i)
            # Attention sublayer + adapter
            x = x + block.attn(block.attn_norm(x), rope_cos, rope_sin, None)
            x = self.attn_adapters[i](x)

            # FFN sublayer + adapter
            x = x + block.ffn(block.ffn_norm(x))
            x = self.ffn_adapters[i](x)

        return bb.final_norm(x)

    def project_head(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states through lm_head."""
        return self.backbone.lm_head(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass. Returns logits (B, T, V)."""
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

        for i in range(len(bb.layers)):
            block = bb.get_block(i)
            x = x + block.attn(block.attn_norm(x), rope_cos, rope_sin, mask)
            x = self.attn_adapters[i](x)

            x = x + block.ffn(block.ffn_norm(x))
            x = self.ffn_adapters[i](x)

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
        for i in range(len(bb.layers)):
            block = bb.get_block(i)
            # KV-cache forward for attention
            layer_cache = kv_cache[i] if kv_cache is not None else None
            attn_out, new_cache = block.attn.forward_kv(
                block.attn_norm(x), rope_cos, rope_sin, layer_cache,
            )
            x = x + attn_out
            x = self.attn_adapters[i](x)
            new_kv_cache.append(new_cache)

            x = x + block.ffn(block.ffn_norm(x))
            x = self.ffn_adapters[i](x)

        x = bb.final_norm(x[:, -1:, :])
        logits = bb.lm_head(x)
        return logits, new_kv_cache

    # --- Parameter management ---

    def adapter_parameters(self) -> list[nn.Parameter]:
        """Return only trainable adapter parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract adapter weights for saving."""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]):
        """Load adapter weights."""
        params = dict(self.named_parameters())
        for k, v in state.items():
            if k in params:
                params[k].data.copy_(v)

    def adapter_weight_report(self) -> dict[str, float]:
        """Per-layer adapter weight norms for monitoring."""
        report = {}
        for i in range(len(self.backbone.layers)):
            a = self.attn_adapters[i]
            if isinstance(a, BottleneckAdapter):
                report[f"adapter/layer{i}.attn.down"] = a.down.weight.data.norm().item()
                report[f"adapter/layer{i}.attn.up"] = a.up.weight.data.norm().item()
            a = self.ffn_adapters[i]
            if isinstance(a, BottleneckAdapter):
                report[f"adapter/layer{i}.ffn.down"] = a.down.weight.data.norm().item()
                report[f"adapter/layer{i}.ffn.up"] = a.up.weight.data.norm().item()
        return report
