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
* Legacy ``CLMConfig`` fields not present on the JAX ``ModelConfig``
  (``dropout`` etc.) are dropped.

Pre-vocab-transition checkpoints (the ~60k coordinate-vocabulary trained
before the searchless_chess 1,968-action transition) are **rejected**: the
JAX model uses the current ``decomp_table`` so converting old token layouts
would silently embed every move incorrectly. Use the
``pre-vocab-transition`` git tag in PyTorch to work with those checkpoints.

The converter is framework-neutral on the read side (``safetensors.numpy``;
no torch import) so converting a legacy checkpoint does not require torch.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from pawn.jax.checkpoint import (
    IncompleteCheckpointError,
    save_model,
    verify_checkpoint,
)
from pawn.jax.config import (
    MAX_SEQ_LEN,
    N_OUTCOMES,
    VOCAB_SIZE,
    ModelConfig,
)
from pawn.jax.model import PAWNModel


class IncompatibleCheckpointError(Exception):
    """Raised when a legacy checkpoint cannot be safely converted to JAX."""


# State_dict keys produced by ``pawn.model.PAWNCLM`` (legacy PyTorch model).
# Listed explicitly so a missing key fails loudly with a clear message rather
# than at the point of dict indexing inside the stacking helper.
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

_REQUIRED_CFG_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(ModelConfig)
    if f.default is dataclasses.MISSING
    and f.default_factory is dataclasses.MISSING
)


def legacy_to_model_config(
    legacy_model_config: dict[str, Any],
) -> ModelConfig:
    """Build a JAX ``ModelConfig`` from a legacy PyTorch ``model_config`` dict.

    Drops legacy-only fields (``dropout``); fills sensible defaults for
    fields that older configs may have omitted (``rope_base``,
    ``max_seq_len``). Rejects pre-vocab-transition checkpoints whose
    vocabulary layout differs from the current one — the JAX model uses the
    current decomp table and would embed old tokens incorrectly.
    """
    missing = [f for f in _REQUIRED_CFG_FIELDS if f not in legacy_model_config]
    if missing:
        raise KeyError(
            f"legacy model_config is missing required fields {missing}; "
            f"keys present: {sorted(legacy_model_config.keys())}"
        )
    vocab_size = legacy_model_config.get("vocab_size", VOCAB_SIZE)
    n_outcomes = legacy_model_config.get("n_outcomes", N_OUTCOMES)
    if vocab_size != VOCAB_SIZE or n_outcomes != N_OUTCOMES:
        raise IncompatibleCheckpointError(
            f"legacy checkpoint uses vocab_size={vocab_size}, "
            f"n_outcomes={n_outcomes}; this build only converts the current "
            f"searchless_chess vocabulary (vocab_size={VOCAB_SIZE}, "
            f"n_outcomes={N_OUTCOMES}). Pre-vocab-transition checkpoints "
            "are accessible only via the `pre-vocab-transition` git tag."
        )
    return ModelConfig(
        d_model=legacy_model_config["d_model"],
        n_layers=legacy_model_config["n_layers"],
        n_heads=legacy_model_config["n_heads"],
        d_ff=legacy_model_config["d_ff"],
        vocab_size=vocab_size,
        max_seq_len=legacy_model_config.get("max_seq_len", MAX_SEQ_LEN),
        n_outcomes=n_outcomes,
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

    def stack(suffix: str, *, transpose: bool = False) -> jax.Array:
        arrs = [state[f"layers.{i}.{suffix}"] for i in range(n_layers)]
        if transpose:
            arrs = [a.T for a in arrs]
        return jnp.asarray(np.stack(arrs))

    return PAWNModel(
        src_embed=jnp.asarray(state["embed.src_embed.weight"]),
        dst_embed=jnp.asarray(state["embed.dst_embed.weight"]),
        promo_embed=jnp.asarray(state["embed.promo_embed.weight"]),
        pad_embed=jnp.asarray(state["embed.pad_embed"]),
        outcome_embed=jnp.asarray(state["embed.outcome_embed.weight"]),
        attn_norm=stack("attn_norm.weight"),
        wq=stack("attn.wq.weight", transpose=True),
        wk=stack("attn.wk.weight", transpose=True),
        wv=stack("attn.wv.weight", transpose=True),
        wo=stack("attn.wo.weight", transpose=True),
        ffn_norm=stack("ffn_norm.weight"),
        w_gate=stack("ffn.w_gate.weight", transpose=True),
        w_up=stack("ffn.w_up.weight", transpose=True),
        w_down=stack("ffn.w_down.weight", transpose=True),
        final_norm=jnp.asarray(state["final_norm.weight"]),
        lm_head=jnp.asarray(state["lm_head.weight"].T),
        cfg=cfg,
    )


def convert_legacy_checkpoint(src: str | Path, dst: str | Path) -> None:
    """Convert a legacy PyTorch checkpoint directory to a JAX checkpoint.

    ``src`` must contain ``config.json`` (with a ``model_config`` dict) and
    ``model.safetensors`` (the PyTorch state_dict). If ``src/.complete``
    exists (full pretraining checkpoints carry it), the sentinel is verified
    before any weights are read — a corrupt source raises rather than
    silently producing a "valid"-looking JAX checkpoint of corrupt bytes.
    Sentinel-absent directories are accepted without integrity check when
    they look like an HF-snapshot layout (``model.safetensors`` +
    ``config.json``, optionally alongside ``README.md`` / ``LICENSE`` /
    ``.gitattributes``); a directory containing full-checkpoint payload
    files (``optimizer.safetensors`` / ``training_state.json``) without a
    sentinel is rejected as a corrupted / interrupted save. ``dst`` is
    overwritten if it exists.
    """
    src_path = Path(src)
    # Sentinel handling: full pretraining checkpoints carry ``.complete`` and
    # MUST verify; sentinel-absent directories may be either a bare HF
    # snapshot (model.safetensors + config.json plus the usual README /
    # LICENSE / .gitattributes from ``huggingface_hub.snapshot_download``)
    # or a corrupted full checkpoint (the sentinel was lost in an interrupted
    # save while optimizer / training-state files were already on disk). We
    # discriminate on the presence of *payload* files specific to a full
    # training checkpoint — README and similar metadata files are fine.
    sentinel = src_path / ".complete"
    payload_files = {"optimizer.safetensors", "training_state.json"}
    present = {entry.name for entry in src_path.iterdir() if entry.is_file()}
    present_payload = present & payload_files
    if sentinel.exists():
        verify_checkpoint(src_path)
    elif present_payload:
        raise IncompleteCheckpointError(
            f"{src_path} has full-checkpoint payload files "
            f"({sorted(present_payload)}) but no .complete sentinel — "
            "looks like a corrupted / interrupted save. Restore the "
            "sentinel or remove the payload files before converting."
        )
    config_path = src_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "model_config" not in config:
        raise KeyError(
            f"{config_path} has no 'model_config' key; top-level keys: "
            f"{sorted(config.keys())}"
        )
    cfg = legacy_to_model_config(config["model_config"])
    state = load_file(src_path / "model.safetensors")
    model = convert_state_dict(state, cfg)
    save_model(dst, model)
