"""Sparse adaptation for PAWN.

Perturbs a random subset of frozen weight elements. Each selected weight
gets an additive trainable delta (zero-initialized, so the model starts
identical to the frozen backbone).  Related to sparse fine-tuning ideas
from the lottery ticket literature (`Frankle & Carbin, 2018
<https://arxiv.org/abs/1803.03635>`_, ICLR 2019).

    output = linear(x)  where  W = W_frozen + delta * mask

The binary mask is fixed at init. Effective trainable parameters equal
the number of True entries in the mask, controlled by the density parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawn.config import CLMConfig
from pawn.model import PAWNCLM, Attention


class SparseLinear(nn.Module):
    """Frozen linear with a sparse additive delta.

    output = F.linear(x, W_frozen + delta * mask, bias)
    """

    mask: torch.Tensor

    def __init__(self, frozen_linear: nn.Linear, mask: torch.Tensor):
        super().__init__()
        self.frozen = frozen_linear
        self.delta = nn.Parameter(torch.zeros_like(frozen_linear.weight))
        self.register_buffer("mask", mask)

    @property
    def n_active(self) -> int:
        return int(self.mask.sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.frozen.weight + self.delta * self.mask
        return F.linear(x, w, self.frozen.bias)


def _random_mask(shape: tuple[int, ...], density: float,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Create a random binary mask with the given density (fraction of True)."""
    return torch.rand(shape, generator=generator) < density


_ATTN_TARGETS = ("wq", "wk", "wv", "wo")
_FFN_TARGETS = ("w_gate", "w_up", "w_down")


class SparseCLM(nn.Module):
    """Frozen PAWN backbone with sparse weight perturbation.

    A random subset of weight elements in attention (and optionally FFN)
    projections are made trainable. Only the masked delta values are
    effectively learned.
    """

    def __init__(
        self,
        backbone: PAWNCLM,
        density: float = 0.01,
        attn_targets: tuple[str, ...] = _ATTN_TARGETS,
        adapt_ffn: bool = False,
        layers: tuple[int, ...] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.backbone = backbone
        self.density = density
        self.adapt_ffn = adapt_ffn
        self.attn_targets = tuple(attn_targets)

        n_layers = len(backbone.layers)
        self.adapted_layers = set(layers if layers is not None else range(n_layers))

        # Freeze the entire backbone
        for p in backbone.parameters():
            p.requires_grad = False

        gen = torch.Generator().manual_seed(seed)

        # Inject sparse adapters
        for layer_idx in range(len(backbone.layers)):
            if layer_idx not in self.adapted_layers:
                continue
            block = backbone.get_block(layer_idx)

            attn: Attention = block.attn
            for proj_name in self.attn_targets:
                original = getattr(attn, proj_name)
                mask = _random_mask(original.weight.shape, density, gen)
                setattr(attn, proj_name, SparseLinear(original, mask))

            if adapt_ffn:
                ffn = block.ffn
                for proj_name in _FFN_TARGETS:
                    original = getattr(ffn, proj_name)
                    mask = _random_mask(original.weight.shape, density, gen)
                    setattr(ffn, proj_name, SparseLinear(original, mask))

    @property
    def cfg(self) -> CLMConfig:
        return self.backbone.cfg

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run backbone layers (with sparse deltas), return normed hidden states."""
        bb = self.backbone
        x = bb.embed(input_ids)

        T = input_ids.shape[1]
        rope_cos = bb.rope_cos[:, :, :T, :]
        rope_sin = bb.rope_sin[:, :, :T, :]

        for layer in bb.layers:
            x = layer(x, rope_cos, rope_sin, None)

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
        for i in range(len(bb.layers)):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = bb.get_block(i).forward_kv(x, rope_cos, rope_sin, layer_cache)
            new_kv_cache.append(new_cache)

        x = bb.final_norm(x[:, -1:, :])
        logits = bb.lm_head(x)
        return logits, new_kv_cache

    # --- Parameter management ---

    def sparse_parameters(self) -> list[nn.Parameter]:
        """Return only trainable sparse delta parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def n_active_params(self) -> int:
        """Count of actually active (masked-in) parameters."""
        total = 0
        for layer_idx in range(len(self.backbone.layers)):
            block = self.backbone.get_block(layer_idx)
            for proj_name in self.attn_targets:
                module = getattr(block.attn, proj_name)
                if isinstance(module, SparseLinear):
                    total += module.n_active
            if self.adapt_ffn:
                for proj_name in _FFN_TARGETS:
                    module = getattr(block.ffn, proj_name)
                    if isinstance(module, SparseLinear):
                        total += module.n_active
        return total

    def sparse_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract sparse delta weights for saving."""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_sparse_state_dict(self, state: dict[str, torch.Tensor]):
        """Load sparse delta weights."""
        own = self.state_dict()
        for k, v in state.items():
            if k in own:
                own[k] = v
        self.load_state_dict(own, strict=False)

    def sparse_weight_report(self) -> dict[str, float]:
        """Per-layer sparse delta norms for monitoring."""
        report = {}
        for layer_idx in range(len(self.backbone.layers)):
            block = self.backbone.get_block(layer_idx)
            for proj_name in self.attn_targets:
                module = getattr(block.attn, proj_name)
                if isinstance(module, SparseLinear):
                    masked_delta = module.delta.data * module.mask
                    report[f"sparse/layer{layer_idx}.{proj_name}.delta"] = masked_delta.norm().item()
            if self.adapt_ffn:
                for proj_name in _FFN_TARGETS:
                    module = getattr(block.ffn, proj_name)
                    if isinstance(module, SparseLinear):
                        masked_delta = module.delta.data * module.mask
                        report[f"sparse/layer{layer_idx}.{proj_name}.delta"] = masked_delta.norm().item()
        return report
