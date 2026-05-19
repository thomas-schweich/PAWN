"""Convert legacy PyTorch PAWN checkpoints to JAX checkpoints.

The three published checkpoints (``pawn-{small,base,large}``) and any other
PyTorch-format checkpoint produced by the pre-migration codebase can be
brought across one-time via ``convert_legacy_checkpoint``. The output is a
standalone JAX checkpoint readable by ``pawn.jax.checkpoint.load_model``; it
retains the legacy model's hyperparameters — notably head count (the legacy
``pawn-large`` used 8 heads / head_dim 80, which is **not** a supernet
variant).

PyTorch → JAX mapping:

* PyTorch ``nn.Linear`` stores weights as ``(out, in)``; JAX uses
  ``(in, out)``. All linear projections (``wq``, ``wk``, ``wv``, ``wo``,
  ``w_gate``, ``w_up``, ``w_down``, ``lm_head``) are transposed on
  conversion.
* PyTorch ``nn.ModuleList`` of layers stores per-layer parameters under
  ``layers.{i}.…``; JAX stacks them on a leading axis. The converter
  stacks per-layer tensors in index order so the JAX ``lax.scan`` over
  the stacked axis applies layers in the original order.
* PyTorch buffers (``rope_cos`` / ``rope_sin`` / ``causal_mask`` /
  ``embed.decomp_table``) are non-persistent and not serialised in the
  state_dict; they have no JAX-side counterpart in the parameter PyTree.
* PyTorch ``TrainingConfig`` fields (``dropout`` etc.) are dropped — the
  JAX ``ModelConfig`` does not carry them.

The converter is framework-neutral on the read side (``safetensors.numpy``;
no torch import) so converting a legacy checkpoint does not require torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from pawn.jax.checkpoint import save_model
from pawn.jax.config import (
    MAX_SEQ_LEN,
    N_OUTCOMES,
    VOCAB_SIZE,
    ModelConfig,
)
from pawn.jax.model import PAWNModel

# State_dict keys produced by ``pawn.model.PAWNCLM`` (legacy PyTorch model).
# Listed here so a missing key fails loudly with a clear message rather than
# at the point of dict indexing inside the layer-stacking helpers.
_EMBED_KEYS = (
    "embed.src_embed.weight",
    "embed.dst_embed.weight",
    "embed.promo_embed.weight",
    "embed.pad_embed",
    "embed.outcome_embed.weight",
)
_HEAD_KEYS = ("final_norm.weight", "lm_head.weight")
_PER_LAYER_KEYS = (
    "attn_norm.weight",
    "attn.wq.weight",
    "attn.wk.weight",
    "attn.wv.weight",
    "attn.wo.weight",
    "ffn_norm.weight",
    "ffn.w_gate.weight",
    "ffn.w_up.weight",
    "ffn.w_down.weight",
)


def legacy_to_model_config(legacy_model_config: dict) -> ModelConfig:
    """Build a JAX ``ModelConfig`` from a legacy PyTorch ``model_config`` dict.

    Drops legacy-only fields (``dropout``); fills sensible defaults for fields
    that older configs may have omitted (``rope_base``).
    """
    return ModelConfig(
        d_model=legacy_model_config["d_model"],
        n_layers=legacy_model_config["n_layers"],
        n_heads=legacy_model_config["n_heads"],
        d_ff=legacy_model_config["d_ff"],
        vocab_size=legacy_model_config.get("vocab_size", VOCAB_SIZE),
        max_seq_len=legacy_model_config.get("max_seq_len", MAX_SEQ_LEN),
        n_outcomes=legacy_model_config.get("n_outcomes", N_OUTCOMES),
        rope_base=legacy_model_config.get("rope_base", 10000.0),
    )


def _check_keys(state: dict[str, np.ndarray], cfg: ModelConfig) -> None:
    """Raise ``KeyError`` with a clear message if any required key is absent."""
    expected: set[str] = set(_EMBED_KEYS) | set(_HEAD_KEYS)
    for i in range(cfg.n_layers):
        for suffix in _PER_LAYER_KEYS:
            expected.add(f"layers.{i}.{suffix}")
    missing = sorted(expected - set(state))
    if missing:
        raise KeyError(
            f"legacy state_dict is missing required keys: {missing}"
        )


def convert_state_dict(
    state: dict[str, np.ndarray], cfg: ModelConfig
) -> PAWNModel:
    """Translate a PyTorch state_dict to a JAX ``PAWNModel`` under ``cfg``.

    Stacks per-layer tensors on a leading axis and transposes
    ``(out, in)`` linear weights to ``(in, out)``. Constructs ``PAWNModel``
    directly so no random keys are consumed.
    """
    _check_keys(state, cfg)
    n_layers = cfg.n_layers

    def stack(suffix: str) -> jax.Array:
        return jnp.asarray(
            np.stack([state[f"layers.{i}.{suffix}"] for i in range(n_layers)])
        )

    def stack_t(suffix: str) -> jax.Array:
        return jnp.asarray(
            np.stack(
                [state[f"layers.{i}.{suffix}"].T for i in range(n_layers)]
            )
        )

    return PAWNModel(
        src_embed=jnp.asarray(state["embed.src_embed.weight"]),
        dst_embed=jnp.asarray(state["embed.dst_embed.weight"]),
        promo_embed=jnp.asarray(state["embed.promo_embed.weight"]),
        pad_embed=jnp.asarray(state["embed.pad_embed"]),
        outcome_embed=jnp.asarray(state["embed.outcome_embed.weight"]),
        attn_norm=stack("attn_norm.weight"),
        wq=stack_t("attn.wq.weight"),
        wk=stack_t("attn.wk.weight"),
        wv=stack_t("attn.wv.weight"),
        wo=stack_t("attn.wo.weight"),
        ffn_norm=stack("ffn_norm.weight"),
        w_gate=stack_t("ffn.w_gate.weight"),
        w_up=stack_t("ffn.w_up.weight"),
        w_down=stack_t("ffn.w_down.weight"),
        final_norm=jnp.asarray(state["final_norm.weight"]),
        lm_head=jnp.asarray(state["lm_head.weight"].T),
        cfg=cfg,
    )


def convert_legacy_checkpoint(src: str | Path, dst: str | Path) -> None:
    """Convert a legacy PyTorch checkpoint directory to a JAX checkpoint.

    ``src`` must contain ``config.json`` (with a ``model_config`` dict) and
    ``model.safetensors`` (the PyTorch state_dict). ``dst`` is overwritten if
    it exists. Reads via ``safetensors.numpy`` so no torch import is needed.
    """
    src_path = Path(src)
    config = json.loads(
        (src_path / "config.json").read_text(encoding="utf-8")
    )
    cfg = legacy_to_model_config(config["model_config"])
    state = load_file(src_path / "model.safetensors")
    model = convert_state_dict(state, cfg)
    save_model(dst, model)
