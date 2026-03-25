"""LoRA + FiLM hybrid adapter for PAWN.

Combines both adaptation methods on a frozen backbone:

- `LoRA <https://arxiv.org/abs/2106.09685>`_ (Hu et al., 2021) modifies
  attention (and optionally FFN) projections within layers
- `FiLM <https://arxiv.org/abs/1709.07871>`_ (Perez et al., 2017) applies
  per-channel affine transforms between layers

Both are identity-initialized so the model starts identical to the frozen
backbone. The two methods are complementary: LoRA changes how attention
computes (cross-channel mixing), FiLM rescales the residual stream
(diagonal modulation).
"""

import torch
import torch.nn as nn

from pawn.config import CLMConfig
from pawn.model import PAWNCLM, Attention
from pawn.adapters.lora import LoRALinear, ATTN_PRESETS, _FFN_TARGETS
from pawn.adapters.film import FiLMLayer


class HybridCLM(nn.Module):
    """Frozen PAWN backbone with LoRA + FiLM adapters.

    LoRA is injected into attention projections within selected layers.
    FiLM layers sit between transformer blocks (and optionally on output).
    """

    def __init__(
        self,
        backbone: PAWNCLM,
        # LoRA config
        lora_rank: int = 4,
        lora_alpha: float | None = None,
        attn_targets: str | tuple[str, ...] = "qkvo",
        adapt_ffn: bool = False,
        lora_layers: tuple[int, ...] | None = None,
        # FiLM config
        use_film: bool = True,
        use_output_film: bool = False,
        film_layers: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        cfg = backbone.cfg
        n_layers = len(backbone.layers)

        # LoRA settings
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else float(lora_rank)
        self.adapt_ffn = adapt_ffn
        if isinstance(attn_targets, str):
            self.attn_targets = ATTN_PRESETS[attn_targets]
        else:
            self.attn_targets = tuple(attn_targets)
        self.lora_layer_set = set(lora_layers if lora_layers is not None else range(n_layers))

        # FiLM settings
        self.use_film = use_film
        self.use_output_film = use_output_film
        self.film_layer_set = set(film_layers if film_layers is not None else range(n_layers))

        # Freeze the entire backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Inject LoRA
        for layer_idx in range(n_layers):
            if layer_idx not in self.lora_layer_set:
                continue
            block = backbone.get_block(layer_idx)

            attn: Attention = block.attn
            for proj_name in self.attn_targets:
                original = getattr(attn, proj_name)
                setattr(attn, proj_name, LoRALinear(original, lora_rank, self.lora_alpha))
            if adapt_ffn:
                ffn = block.ffn
                for proj_name in _FFN_TARGETS:
                    original = getattr(ffn, proj_name)
                    setattr(ffn, proj_name, LoRALinear(original, lora_rank, self.lora_alpha))

        # Create FiLM layers (Identity for non-adapted layers)
        if use_film:
            self.hidden_films = nn.ModuleList()
            for i in range(n_layers):
                if i in self.film_layer_set:
                    self.hidden_films.append(FiLMLayer(cfg.d_model))
                else:
                    self.hidden_films.append(nn.Identity())
        else:
            self.hidden_films = None

        if use_output_film:
            self.output_film = FiLMLayer(cfg.vocab_size)
        else:
            self.output_film = None

    @property
    def cfg(self) -> CLMConfig:
        return self.backbone.cfg

    def forward_hidden(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run backbone layers (LoRA inside + FiLM after), return normed hidden."""
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

        for i, layer in enumerate(bb.layers):
            x = layer(x, rope_cos, rope_sin, mask)
            if self.hidden_films is not None:
                x = self.hidden_films[i](x)

        return bb.final_norm(x)

    def project_head(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states through lm_head + optional output FiLM."""
        logits = self.backbone.lm_head(x)
        if self.output_film is not None:
            logits = self.output_film(logits)
        return logits

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

        for i, layer in enumerate(bb.layers):
            x = layer(x, rope_cos, rope_sin, mask)
            if self.hidden_films is not None:
                x = self.hidden_films[i](x)

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
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = block.forward_kv(x, rope_cos, rope_sin, layer_cache)
            if self.hidden_films is not None:
                x = self.hidden_films[i](x)
            new_kv_cache.append(new_cache)

        x = bb.final_norm(x[:, -1:, :])
        logits = bb.lm_head(x)
        if self.output_film is not None:
            logits = self.output_film(logits)

        return logits, new_kv_cache

    # --- Parameter management ---

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only LoRA A/B parameters."""
        params = []
        for layer_idx in range(len(self.backbone.layers)):
            block = self.backbone.get_block(layer_idx)
            for proj_name in self.attn_targets:
                module = getattr(block.attn, proj_name)
                if isinstance(module, LoRALinear):
                    params.append(module.lora_A)
                    params.append(module.lora_B)
            if self.adapt_ffn:
                for proj_name in _FFN_TARGETS:
                    module = getattr(block.ffn, proj_name)
                    if isinstance(module, LoRALinear):
                        params.append(module.lora_A)
                        params.append(module.lora_B)
        return params

    def film_parameters(self) -> list[nn.Parameter]:
        """Return only FiLM parameters."""
        params = []
        if self.hidden_films is not None:
            for film in self.hidden_films:
                if isinstance(film, FiLMLayer):
                    params.extend(film.parameters())
        if self.output_film is not None:
            params.extend(self.output_film.parameters())
        return params

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract all trainable (LoRA + FiLM) weights for saving."""
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]):
        """Load adapter weights."""
        own = self.state_dict()
        for k, v in state.items():
            if k in own:
                own[k] = v
        self.load_state_dict(own, strict=False)

    def weight_report(self) -> dict[str, float]:
        """Combined LoRA + FiLM weight report for monitoring."""
        report = {}

        # LoRA norms
        for layer_idx in range(len(self.backbone.layers)):
            block = self.backbone.get_block(layer_idx)
            for proj_name in self.attn_targets:
                module = getattr(block.attn, proj_name)
                if isinstance(module, LoRALinear):
                    report[f"lora/layer{layer_idx}.{proj_name}.B"] = module.lora_B.data.norm().item()
            if self.adapt_ffn:
                for proj_name in _FFN_TARGETS:
                    module = getattr(block.ffn, proj_name)
                    if isinstance(module, LoRALinear):
                        report[f"lora/layer{layer_idx}.{proj_name}.B"] = module.lora_B.data.norm().item()

        # FiLM norms
        if self.hidden_films is not None:
            for i, film in enumerate(self.hidden_films):
                if isinstance(film, FiLMLayer):
                    report[f"film/hidden_{i}/gamma_dev"] = (film.gamma - 1.0).norm().item()
                    report[f"film/hidden_{i}/beta_norm"] = film.beta.norm().item()
        if self.output_film is not None:
            report["film/output/gamma_dev"] = (self.output_film.gamma - 1.0).norm().item()
            report["film/output/beta_norm"] = self.output_film.beta.norm().item()

        return report
