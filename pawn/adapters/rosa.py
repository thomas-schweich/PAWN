"""RoSA (Robust Sparse Adaptation) for PAWN.

Implements the method from `Nikdan et al., 2024
<https://arxiv.org/abs/2401.04679>`_ ("RoSA: Accurate Parameter-Efficient
Fine-Tuning via Robust Adaptation").

Combines a low-rank adapter (LoRA) with a gradient-informed sparse adapter
on each frozen projection matrix:

    output = frozen(x) + (x @ A^T) @ B^T * scaling + F.linear(x, delta * mask)

Training follows three phases:
  1. LoRA warm-up (sparse disabled)
  2. Gradient-based mask generation (Algorithm 1)
  3. Joint LoRA + sparse training (or sparse-only in retrospective modes)

Also provides RetroBottleneckCLM for the retrospective sparse + bottleneck
ablation, combining gradient-informed sparse masks with nonlinear bottleneck
adapters.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawn.config import CLMConfig
from pawn.model import PAWNCLM, Attention
from pawn.adapters.lora import ATTN_PRESETS, _FFN_TARGETS
from pawn.adapters.sparse import SparseLinear, SparseCLM
from pawn.adapters.bottleneck import BottleneckAdapter


# ---------------------------------------------------------------------------
# RoSALinear -- compound LoRA + sparse linear
# ---------------------------------------------------------------------------

class RoSALinear(nn.Module):
    """Frozen linear with both a low-rank and a sparse additive adapter.

    output = frozen(x) + lora(x) + F.linear(x, delta * mask)

    LoRA starts with B=0 (identity init). Sparse starts with delta=0 and
    mask=all-False (disabled). Call ``set_mask`` to activate the sparse branch.
    """

    mask: torch.Tensor

    def __init__(
        self,
        frozen_linear: nn.Linear,
        rank: int,
        alpha: float | None = None,
        lora_enabled: bool = True,
        sparse_enabled: bool = False,
    ):
        super().__init__()
        self.frozen = frozen_linear
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        self.lora_enabled = lora_enabled
        self.sparse_enabled = sparse_enabled

        in_features = frozen_linear.in_features
        out_features = frozen_linear.out_features

        # LoRA branch
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Sparse branch
        self.delta = nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer("mask", torch.zeros(out_features, in_features, dtype=torch.bool))

        # Sparse delta starts frozen (no grad until mask is set)
        if not sparse_enabled:
            self.delta.requires_grad = False

    def set_mask(self, new_mask: torch.Tensor):
        """Set the sparse mask and enable the sparse branch."""
        self.mask.copy_(new_mask)
        self.sparse_enabled = True
        self.delta.requires_grad = True

    @property
    def n_active(self) -> int:
        return int(self.mask.sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.frozen(x)
        if self.lora_enabled:
            out = out + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        if self.sparse_enabled:
            out = out + F.linear(x, self.delta * self.mask)
        return out


# ---------------------------------------------------------------------------
# RoSACLM -- wrapper around frozen PAWNCLM
# ---------------------------------------------------------------------------

class RoSACLM(nn.Module):
    """Frozen PAWN backbone with RoSA adapters (LoRA + gradient-informed sparse).

    Injects ``RoSALinear`` into attention (and optionally FFN) projections.
    During warm-up, only the LoRA branch is active. After mask generation,
    both branches train jointly.
    """

    def __init__(
        self,
        backbone: PAWNCLM,
        rank: int = 4,
        alpha: float | None = None,
        attn_targets: str | tuple[str, ...] = "qkvo",
        adapt_ffn: bool = False,
        layers: tuple[int, ...] | None = None,
        lora_enabled: bool = True,
        sparse_enabled: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.rank = rank
        self._alpha = alpha if alpha is not None else float(rank)
        self.adapt_ffn = adapt_ffn

        if isinstance(attn_targets, str):
            self.attn_targets = ATTN_PRESETS[attn_targets]
        else:
            self.attn_targets = tuple(attn_targets)

        n_layers = len(backbone.layers)
        self.adapted_layers = set(layers if layers is not None else range(n_layers))

        # Freeze the entire backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Inject RoSA adapters
        for layer_idx in range(n_layers):
            if layer_idx not in self.adapted_layers:
                continue
            block = backbone.get_block(layer_idx)

            attn: Attention = block.attn
            for proj_name in self.attn_targets:
                original = getattr(attn, proj_name)
                setattr(attn, proj_name, RoSALinear(
                    original, rank, self._alpha,
                    lora_enabled=lora_enabled,
                    sparse_enabled=sparse_enabled,
                ))

            if adapt_ffn:
                ffn = block.ffn
                for proj_name in _FFN_TARGETS:
                    original = getattr(ffn, proj_name)
                    setattr(ffn, proj_name, RoSALinear(
                        original, rank, self._alpha,
                        lora_enabled=lora_enabled,
                        sparse_enabled=sparse_enabled,
                    ))

    @property
    def cfg(self) -> CLMConfig:
        return self.backbone.cfg

    # --- Module iteration helpers ---

    def _rosa_modules(self) -> Iterator[tuple[str, RoSALinear]]:
        """Iterate (key, module) for all RoSALinear modules."""
        for layer_idx in sorted(self.adapted_layers):
            block = self.backbone.get_block(layer_idx)
            for proj_name in self.attn_targets:
                module = getattr(block.attn, proj_name)
                if isinstance(module, RoSALinear):
                    yield f"layer{layer_idx}.{proj_name}", module
            if self.adapt_ffn:
                for proj_name in _FFN_TARGETS:
                    module = getattr(block.ffn, proj_name)
                    if isinstance(module, RoSALinear):
                        yield f"layer{layer_idx}.{proj_name}", module

    # --- Forward methods ---

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run backbone layers (with RoSA), return normed hidden states."""
        bb = self.backbone
        x = bb.embed(input_ids)

        T = input_ids.shape[1]
        rope_cos = bb.rope_cos[:, :, :T, :]
        rope_sin = bb.rope_sin[:, :, :T, :]

        for layer in bb.layers:
            x = layer(x, rope_cos, rope_sin, None)

        return bb.final_norm(x)

    def project_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.lm_head(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

    # --- Mask management ---

    def set_masks(self, masks: dict[str, torch.Tensor]):
        """Apply gradient-derived masks to all RoSALinear modules."""
        for key, module in self._rosa_modules():
            if key in masks:
                module.set_mask(masks[key])

    def enable_sparse(self):
        """Enable sparse branch on all RoSALinear modules."""
        for _, module in self._rosa_modules():
            module.sparse_enabled = True
            module.delta.requires_grad = True

    def disable_lora(self):
        """Disable and freeze the LoRA branch on all RoSALinear modules."""
        for _, module in self._rosa_modules():
            module.lora_enabled = False
            module.lora_A.requires_grad = False
            module.lora_B.requires_grad = False

    def reinit_lora(self):
        """Re-initialize LoRA weights (A to kaiming, B to zero)."""
        for _, module in self._rosa_modules():
            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            module.lora_B.data.zero_()

    # --- Parameter management ---

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only LoRA A/B parameters."""
        params = []
        for _, module in self._rosa_modules():
            params.append(module.lora_A)
            params.append(module.lora_B)
        return params

    def sparse_parameters(self) -> list[nn.Parameter]:
        """Return only sparse delta parameters."""
        return [module.delta for _, module in self._rosa_modules()]

    def adapter_parameters(self) -> list[nn.Parameter]:
        """Return all trainable adapter parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def n_active_sparse_params(self) -> int:
        """Count of mask-selected (active) sparse parameters."""
        return sum(module.n_active for _, module in self._rosa_modules())

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract all trainable weights plus masks for saving."""
        state = {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        # Also save masks for reproducibility
        for key, module in self._rosa_modules():
            state[f"mask/{key}"] = module.mask.clone()
        return state

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]):
        """Load adapter weights and masks."""
        params = dict(self.named_parameters())
        for k, v in state.items():
            if k.startswith("mask/"):
                # Restore mask
                mask_key = k[5:]  # strip "mask/"
                for key, module in self._rosa_modules():
                    if key == mask_key:
                        module.set_mask(v)
                        break
            elif k in params:
                params[k].data.copy_(v)

    def adapter_weight_report(self) -> dict[str, float]:
        """Per-layer weight norms for monitoring."""
        report = {}
        for key, module in self._rosa_modules():
            report[f"rosa/{key}.lora_B"] = module.lora_B.data.norm().item()
            if module.sparse_enabled:
                masked = module.delta.data * module.mask
                report[f"rosa/{key}.delta"] = masked.norm().item()
        return report


# ---------------------------------------------------------------------------
# RetroBottleneckCLM -- sparse (gradient masks) + bottleneck adapters
# ---------------------------------------------------------------------------

class RetroBottleneckCLM(nn.Module):
    """Frozen PAWN backbone with gradient-informed sparse projections and
    bottleneck adapters after each sublayer.

    The sparse projections use masks derived from a LoRA warm-up phase
    (applied via SparseCLM with overwritten mask buffers). Bottleneck
    adapters add nonlinearity that sparse-only adaptation cannot express.
    """

    def __init__(
        self,
        backbone: PAWNCLM,
        bottleneck_dim: int = 8,
        adapt_attn: bool = True,
        adapt_ffn: bool = True,
        layers: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.bottleneck_dim = bottleneck_dim
        cfg = backbone.cfg
        n_layers = len(backbone.layers)
        adapted = set(layers if layers is not None else range(n_layers))

        # Freeze backbone (may already be frozen by SparseCLM, but be explicit)
        for p in backbone.parameters():
            p.requires_grad = False

        # Bottleneck adapters (same pattern as BottleneckCLM)
        self.attn_adapters = nn.ModuleList()
        self.ffn_adapters = nn.ModuleList()
        for i in range(n_layers):
            if i in adapted and adapt_attn:
                self.attn_adapters.append(BottleneckAdapter(cfg.d_model, bottleneck_dim))
            else:
                self.attn_adapters.append(nn.Identity())
            if i in adapted and adapt_ffn:
                self.ffn_adapters.append(BottleneckAdapter(cfg.d_model, bottleneck_dim))
            else:
                self.ffn_adapters.append(nn.Identity())

    @property
    def cfg(self) -> CLMConfig:
        return self.backbone.cfg

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        x = bb.embed(input_ids)

        T = input_ids.shape[1]
        rope_cos = bb.rope_cos[:, :, :T, :]
        rope_sin = bb.rope_sin[:, :, :T, :]

        for i in range(len(bb.layers)):
            block = bb.get_block(i)
            # Attention sublayer (SparseLinear already injected by SparseCLM)
            x = x + block.attn(block.attn_norm(x), rope_cos, rope_sin, None)
            x = self.attn_adapters[i](x)
            # FFN sublayer
            x = x + block.ffn(block.ffn_norm(x))
            x = self.ffn_adapters[i](x)

        return bb.final_norm(x)

    def project_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.lm_head(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        """All trainable parameters (sparse deltas + bottleneck weights)."""
        return [p for p in self.parameters() if p.requires_grad]

    def sparse_parameters(self) -> list[nn.Parameter]:
        """Only sparse delta parameters (from SparseLinear modules)."""
        params = []
        for layer_idx in range(len(self.backbone.layers)):
            block = self.backbone.get_block(layer_idx)
            for proj_name in ("wq", "wk", "wv", "wo"):
                module = getattr(block.attn, proj_name, None)
                if isinstance(module, SparseLinear):
                    params.append(module.delta)
            for proj_name in _FFN_TARGETS:
                module = getattr(block.ffn, proj_name, None)
                if isinstance(module, SparseLinear):
                    params.append(module.delta)
        return params

    def bottleneck_parameters(self) -> list[nn.Parameter]:
        """Only bottleneck adapter parameters."""
        params = []
        for adapter in self.attn_adapters:
            if isinstance(adapter, BottleneckAdapter):
                params.extend(adapter.parameters())
        for adapter in self.ffn_adapters:
            if isinstance(adapter, BottleneckAdapter):
                params.extend(adapter.parameters())
        return params

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]):
        params = dict(self.named_parameters())
        for k, v in state.items():
            if k in params:
                params[k].data.copy_(v)

    def adapter_weight_report(self) -> dict[str, float]:
        report = {}
        # Sparse delta norms
        for layer_idx in range(len(self.backbone.layers)):
            block = self.backbone.get_block(layer_idx)
            for proj_name in ("wq", "wk", "wv", "wo"):
                module = getattr(block.attn, proj_name, None)
                if isinstance(module, SparseLinear):
                    masked = module.delta.data * module.mask
                    report[f"sparse/layer{layer_idx}.{proj_name}.delta"] = masked.norm().item()
        # Bottleneck norms
        for i, adapter in enumerate(self.attn_adapters):
            if isinstance(adapter, BottleneckAdapter):
                report[f"bottleneck/layer{i}.attn.up"] = adapter.up.weight.data.norm().item()
        for i, adapter in enumerate(self.ffn_adapters):
            if isinstance(adapter, BottleneckAdapter):
                report[f"bottleneck/layer{i}.ffn.up"] = adapter.up.weight.data.norm().item()
        return report


# ---------------------------------------------------------------------------
# Gradient-based mask generation (Algorithm 1 from the paper)
# ---------------------------------------------------------------------------

def generate_gradient_masks(
    model: RoSACLM,
    dataloader,
    mask_builder,
    density: float,
    alpha: int = 2,
    device: str = "cuda",
    use_amp: bool = False,
    max_batches: int = 32,
) -> dict[str, torch.Tensor]:
    """Generate sparse masks via gradient accumulation on a warmed-up model.

    Temporarily enables sparse on all RoSALinear modules with all-True masks
    to capture full gradients, accumulates ``|grad|^alpha``, and selects the
    top-k positions per weight matrix at the target density.

    Args:
        model: RoSACLM after LoRA warm-up.
        dataloader: Training data (only ``max_batches`` batches are used).
        mask_builder: LegalMaskBuilder for computing legal move masks.
        density: Target fraction of weight elements to select (0.0-1.0).
        alpha: Gradient accumulation exponent (1=mean magnitude, 2=Fisher).
        device: Device string.
        use_amp: Whether to use AMP for forward pass.
        max_batches: Number of batches for gradient accumulation.

    Returns:
        Dictionary mapping ``"layer{i}.{proj}"`` to boolean mask tensors.
    """
    model.train()

    # Save original state and temporarily set all-True masks
    original_states: list[tuple[RoSALinear, bool, bool]] = []
    accumulators: dict[str, torch.Tensor] = {}

    for key, module in model._rosa_modules():
        original_states.append((module, module.sparse_enabled, module.delta.requires_grad))
        # Enable sparse with all-True mask so delta.grad captures full weight gradient
        module.mask.fill_(True)
        module.sparse_enabled = True
        module.delta.requires_grad = True
        # Initialize accumulator
        accumulators[key] = torch.zeros_like(module.delta)

    # Clear any stale gradients before accumulation
    model.zero_grad(set_to_none=True)

    # Accumulate gradients
    n_batches = 0
    for batch in dataloader:
        if n_batches >= max_batches:
            break

        ids = batch["input_ids"].to(device)
        tgt = batch["targets"].to(device)
        msk = batch["loss_mask"].to(device)

        # Build legal mask
        if "legal_indices" in batch:
            legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])
        else:
            legal_mask = mask_builder(batch)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            hidden = model.forward_hidden(ids)
            valid_hidden = hidden[msk]
            valid_logits = model.project_head(valid_hidden)

        valid_legal = legal_mask[msk]
        valid_logits = valid_logits.float()
        valid_logits.masked_fill_(~valid_legal, float("-inf"))
        valid_targets = tgt[msk]

        if valid_targets.shape[0] == 0:
            continue

        loss = F.cross_entropy(valid_logits, valid_targets)
        loss.backward()

        # Accumulate |grad|^alpha
        with torch.no_grad():
            for key, module in model._rosa_modules():
                if module.delta.grad is not None:
                    accumulators[key] += module.delta.grad.abs().pow(alpha)
                    module.delta.grad = None

        # Also zero LoRA grads
        for p in model.lora_parameters():
            if p.grad is not None:
                p.grad = None

        n_batches += 1

    # Restore original state and zero out deltas
    for module, orig_sparse, orig_requires_grad in original_states:
        module.mask.zero_()
        module.sparse_enabled = orig_sparse
        module.delta.requires_grad = orig_requires_grad
        module.delta.data.zero_()

    # Generate masks: top-k by accumulated gradient magnitude
    masks: dict[str, torch.Tensor] = {}
    for key, acc in accumulators.items():
        flat = acc.flatten()
        k = max(1, int(density * flat.numel()))
        _, top_indices = flat.topk(k)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[top_indices] = True
        masks[key] = mask.reshape(acc.shape)

    return masks
