"""v1 PyTorch → v2 JAX checkpoint converter.

The **only** v1↔v2 bridge per plan §5: a single function
:func:`convert_legacy_checkpoint` that reads a v1 torch
``.safetensors`` checkpoint (from a HuggingFace repo or a local path),
transposes linear weights from ``(out, in)`` to JAX's ``(in, out)``
convention, builds a :class:`pawn.config.ModelConfig` at the v1
dimensions, and writes a JAX checkpoint under
``$HF_HOME/pawn-jax-converted/<sha>/``.

Pre-vocab-transition checkpoints (vocab_size != 1980) are rejected
loudly. The conversion is cached by source *identifier* (HF repo ID
or local path string), not by file content — a second call with the
same source string returns the cached output directly. If the
upstream HF repo has been re-published with new weights, pass
``force=True`` to re-convert; the same is true for local paths whose
contents have changed under the same path.

For each per-layer v1 field (``layers.<i>.attn.wq.weight`` etc.), the
converter stacks the N independent layer tensors into a single
``(n_layers, in, out)`` array matching v2's ``TransformerLayer`` field
shape.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file as st_load
from safetensors.numpy import save_file as st_save

from pawn.checkpoint import save_model
from pawn.config import (
    HEAD_DIM,
    MAX_SEQ_LEN,
    NUM_ACTIONS,
    ROPE_BASE,
    VOCAB_SIZE,
    ModelConfig,
)
from pawn.model import PAWNModel, TransformerLayer, _build_decomp_table

__all__ = [
    "convert_legacy_checkpoint",
]


_CONVERTED_CACHE_VERSION = 1


def _converted_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "pawn-jax-converted"
    return Path.home() / ".cache" / "huggingface" / "pawn-jax-converted"


def _source_id_hash(source: str) -> str:
    """SHA-256 of the source *identifier* string (HF repo ID or local
    path), versioned by ``_CONVERTED_CACHE_VERSION``.

    Not a hash of the file content — re-publishing the upstream
    checkpoint with new weights under the same identifier does NOT
    invalidate this key. The matching `convert_legacy_checkpoint(...,
    force=True)` flag is the documented escape hatch.
    """
    return hashlib.sha256(
        f"{_CONVERTED_CACHE_VERSION}|{source}".encode("utf-8")
    ).hexdigest()[:32]


def _download_legacy_checkpoint(source: str) -> Path:
    """Resolve ``source`` to a local directory holding the v1 checkpoint.

    If ``source`` is a local path that exists, return it as-is. Otherwise
    treat it as a HF repo ID and download via :mod:`huggingface_hub`.
    """
    p = Path(source)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            f"source {source!r} is not a local directory and "
            f"huggingface_hub isn't installed; pip install huggingface_hub"
        ) from e
    local = snapshot_download(repo_id=source, repo_type="model")
    return Path(local)


def _read_legacy_config(checkpoint_dir: Path) -> dict:
    """Read the v1 config.json (or training_config.json) from the
    checkpoint dir. v1 may have placed the config under either name
    depending on the era.

    Returns a flat dict — the published v1 format wraps the model
    fields under a ``model_config`` sub-dict
    (``{"format_version": 1, "model_config": {"vocab_size": 1980, ...}}``)
    while the parity fixture writes them flat. Pull from ``model_config``
    when present so callers can read ``vocab_size`` / ``d_model`` /
    ``n_layers`` etc. at the top level regardless of era.
    """
    for name in ("config.json", "training_config.json"):
        candidate = checkpoint_dir / name
        if candidate.is_file():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "model_config" in raw and isinstance(
                raw["model_config"], dict
            ):
                # Surface the nested model fields at the top level.
                merged = dict(raw["model_config"])
                # Preserve other top-level keys for context (training_config,
                # format_version, etc.) without letting them shadow the
                # model-config values.
                for k, v in raw.items():
                    if k == "model_config":
                        continue
                    merged.setdefault(k, v)
                return merged
            return raw
    raise FileNotFoundError(
        f"v1 checkpoint at {checkpoint_dir} has no config.json or "
        f"training_config.json"
    )


def _load_v1_state(checkpoint_dir: Path) -> dict[str, np.ndarray]:
    """Load v1 model.safetensors as a flat name→numpy dict."""
    path = checkpoint_dir / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(
            f"v1 checkpoint at {checkpoint_dir} has no model.safetensors"
        )
    return st_load(str(path))


def _stack_per_layer(
    state: dict[str, np.ndarray], n_layers: int, suffix: str
) -> np.ndarray:
    """Stack v1's per-layer fields into a single (n_layers, ...) tensor.

    v1 stored each layer as ``layers.<i>.<suffix>`` (e.g.
    ``layers.0.attn.wq.weight``). We stack across the leading layer
    axis so the v2 ``TransformerLayer`` field gets one
    (n_layers, ...) tensor.
    """
    pieces = []
    for i in range(n_layers):
        key = f"layers.{i}.{suffix}"
        if key not in state:
            raise KeyError(f"v1 checkpoint missing field {key!r}")
        pieces.append(state[key])
    return np.stack(pieces, axis=0)


def _transpose_linear(weight: np.ndarray) -> np.ndarray:
    """v1 stores nn.Linear weights as (out, in); JAX einsum expects
    (in, out). Transpose the last two axes."""
    if weight.ndim != 2:
        raise ValueError(
            f"linear weight expected 2-D (out, in), got shape {weight.shape}"
        )
    return weight.T


def _build_v2_model(
    state: dict[str, np.ndarray], cfg: ModelConfig
) -> PAWNModel:
    """Assemble a v2 PAWNModel from the v1 state dict.

    Per-layer fields are stacked + (where needed) transposed. The
    decomp_table is rebuilt from the engine vocab; RoPE phase tables
    are recomputed inside the forward pass (no stored state for them
    on v2).
    """
    n_layers = cfg.n_layers

    # Per-layer linears need stack + transpose.
    def linear_field(suffix: str) -> jnp.ndarray:
        stacked = _stack_per_layer(state, n_layers, suffix)  # (L, out, in)
        # Transpose the last two axes → (L, in, out)
        return jnp.asarray(np.transpose(stacked, (0, 2, 1)))

    # Per-layer norms are 1-D, no transpose needed.
    def norm_field(suffix: str) -> jnp.ndarray:
        return jnp.asarray(_stack_per_layer(state, n_layers, suffix))

    layers = TransformerLayer(
        attn_norm_w=norm_field("attn_norm.weight"),
        wq=linear_field("attn.wq.weight"),
        wk=linear_field("attn.wk.weight"),
        wv=linear_field("attn.wv.weight"),
        wo=linear_field("attn.wo.weight"),
        ffn_norm_w=norm_field("ffn_norm.weight"),
        w_gate=linear_field("ffn.w_gate.weight"),
        w_up=linear_field("ffn.w_up.weight"),
        w_down=linear_field("ffn.w_down.weight"),
    )

    # Embeddings + final norm + lm_head.
    return PAWNModel(
        embed_src=jnp.asarray(state["embed.src_embed.weight"]),
        embed_dst=jnp.asarray(state["embed.dst_embed.weight"]),
        embed_promo=jnp.asarray(state["embed.promo_embed.weight"]),
        embed_pad=jnp.asarray(state["embed.pad_embed"]),
        embed_outcome=jnp.asarray(state["embed.outcome_embed.weight"]),
        layers=layers,
        final_norm_w=jnp.asarray(state["final_norm.weight"]),
        lm_head=jnp.asarray(_transpose_linear(state["lm_head.weight"])),
        decomp_table=_build_decomp_table(),
        cfg=cfg,
    )


def convert_legacy_checkpoint(
    source: str,
    *,
    output_dir: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Convert a v1 PyTorch checkpoint to v2 JAX format.

    ``source`` can be a HuggingFace repo ID (e.g.
    ``"thomas-schweich/pawn-base"``) or a local directory holding the
    v1 checkpoint's ``model.safetensors`` + ``config.json``.

    Output cached under ``$HF_HOME/pawn-jax-converted/<sha>/`` (or
    explicit ``output_dir``). Subsequent calls with the same source
    return the cached path. Set ``force=True`` to re-convert.

    Rejects pre-vocab-transition checkpoints loudly: any v1 checkpoint
    whose ``vocab_size`` isn't :data:`pawn.config.VOCAB_SIZE` (1980)
    used the older ~60k-token vocab and is incompatible with v2's
    factored embeddings + 1968-action output head.

    Returns the path to the v2 checkpoint directory.
    """
    if output_dir is not None:
        final = Path(output_dir)
    else:
        final = _converted_cache_root() / _source_id_hash(source)

    if final.is_dir() and not force:
        # Cache hit — return the existing converted checkpoint.
        return final

    local_v1 = _download_legacy_checkpoint(source)
    v1_config = _read_legacy_config(local_v1)
    v1_vocab = int(v1_config.get("vocab_size", -1))
    if v1_vocab != VOCAB_SIZE:
        raise ValueError(
            f"v1 checkpoint at {source!r} has vocab_size={v1_vocab}, expected "
            f"{VOCAB_SIZE}. This looks like a pre-vocab-transition checkpoint "
            f"(older ~60k-token vocab); the v2 stack can only convert "
            f"post-transition checkpoints. Use the `pre-vocab-transition` "
            f"git tag if you need to read the older format."
        )

    v1_state = _load_v1_state(local_v1)
    d_model = int(v1_config["d_model"])
    n_heads = int(v1_config["n_heads"])
    # head_dim is derived from v1's d_model/n_heads — NOT the v2 fixed
    # HEAD_DIM. v1 large used head_dim=80; v2 SUPERNET uses 64. The
    # converter preserves v1's architecture so forward parity holds.
    head_dim = d_model // n_heads
    cfg = ModelConfig(
        d_model=d_model,
        n_layers=int(v1_config["n_layers"]),
        n_heads=n_heads,
        d_ff=int(v1_config["d_ff"]),
        head_dim=head_dim,
        vocab_size=VOCAB_SIZE,
        max_seq_len=int(v1_config.get("max_seq_len", MAX_SEQ_LEN)),
        rope_base=float(v1_config.get("rope_base", ROPE_BASE)),
        n_outcomes=int(v1_config.get("n_outcomes", VOCAB_SIZE - NUM_ACTIONS - 1)),
    )

    model = _build_v2_model(v1_state, cfg)
    # Use the v2 save path with `force` semantics: if the cache dir
    # exists (e.g. interrupted prior conversion), delete and re-create.
    if final.exists() and force:
        import shutil

        shutil.rmtree(final)
    save_model(
        model, final, run_config={"source": source, "legacy_vocab_size": v1_vocab}
    )
    return final
