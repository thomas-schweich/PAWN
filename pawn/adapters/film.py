"""FiLM conditioning for PAWN.

Implements Feature-wise Linear Modulation following `Perez et al., 2017
<https://arxiv.org/abs/1709.07871>`_
("FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018).

Applies learned per-channel affine transforms after each transformer block
and on the output logits:

    h_adapted = γ_l ⊙ h_l + β_l        (hidden layers)
    logits_adapted = γ_out ⊙ logits + β_out  (output)

Identity-initialized (γ=1, β=0) so the wrapped model starts identical to
the frozen backbone. Total trainable params: 8×2×512 + 2×4278 = 16,748.
"""

import torch
import torch.nn as nn
from pawn.config import CLMConfig
from pawn.model import PAWNCLM


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: y = γ * x + β."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta


class FiLMCLM(nn.Module):
    """Frozen PAWN backbone with FiLM adapters.

    FiLM layers are inserted after every transformer block and on the
    output logits. Only FiLM parameters are trainable.
    """

    def __init__(self, backbone: PAWNCLM, use_output_film: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_output_film = use_output_film
        cfg = backbone.cfg

        # Freeze the entire backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Hidden-layer FiLM: one per transformer block
        self.hidden_films = nn.ModuleList([
            FiLMLayer(cfg.d_model) for _ in range(cfg.n_layers)
        ])

        # Output FiLM: applied to logits (optional)
        if use_output_film:
            self.output_film = FiLMLayer(cfg.vocab_size)
        else:
            self.output_film = None

    @property
    def cfg(self) -> CLMConfig:
        return self.backbone.cfg

    def forward_hidden(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run backbone layers + FiLM adapters, return normed hidden states.

        Returns (B, T, d_model) — before lm_head projection.
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

        for layer, film in zip(bb.layers, self.hidden_films):
            x = layer(x, rope_cos, rope_sin, mask)
            x = film(x)

        return bb.final_norm(x)

    def project_head(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states through lm_head + optional output FiLM.

        x: (*, d_model) — works for both (B, T, d) and (N_valid, d).
        """
        logits = self.backbone.lm_head(x)
        if self.output_film is not None:
            logits = self.output_film(logits)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the backbone with FiLM adapters. Returns logits (B, T, V).

        When attention_mask is None, uses is_causal=True in SDPA (enables
        Flash Attention / efficient kernels).  Safe for causal LM training
        because loss_mask already excludes padding positions.
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

        for layer, film in zip(bb.layers, self.hidden_films):
            x = layer(x, rope_cos, rope_sin, mask)
            x = film(x)

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
            x = self.hidden_films[i](x)
            new_kv_cache.append(new_cache)

        x = bb.final_norm(x[:, -1:, :])
        logits = bb.lm_head(x)
        if self.output_film is not None:
            logits = self.output_film(logits)

        return logits, new_kv_cache

    def film_parameters(self) -> list[nn.Parameter]:
        """Return only trainable FiLM parameters."""
        params = []
        for film in self.hidden_films:
            params.extend(film.parameters())
        if self.output_film is not None:
            params.extend(self.output_film.parameters())
        return params

    def film_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract FiLM weights for saving."""
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.data.clone()
        return state

    def load_film_state_dict(self, state: dict[str, torch.Tensor]):
        """Load FiLM weights."""
        own = self.state_dict()
        for k, v in state.items():
            if k in own:
                own[k] = v
        self.load_state_dict(own, strict=False)

    def film_weight_report(self) -> dict[str, float]:
        """Per-layer FiLM deviation from identity, for monitoring."""
        report = {}
        for i, film in enumerate(self.hidden_films):
            if isinstance(film, FiLMLayer):
                report[f"hidden_{i}/gamma_dev"] = (film.gamma - 1.0).norm().item()
                report[f"hidden_{i}/beta_norm"] = film.beta.norm().item()
        if self.output_film is not None:
            report["output/gamma_dev"] = (self.output_film.gamma - 1.0).norm().item()
            report["output/beta_norm"] = self.output_film.beta.norm().item()
        return report
