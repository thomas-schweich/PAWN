"""Thin PyTorch loader for JAX-format PAWN checkpoints.

Lets external PyTorch users load a PAWN checkpoint without installing JAX or
Equinox. Reads the canonical JAX safetensors schema (see
``pawn.jax.checkpoint``), reverses the JAX ``(in, out)`` Linear convention to
PyTorch's ``(out, in)``, unstacks the per-layer leading axis into individual
modules, and returns an inference-only PyTorch ``nn.Module`` mirroring the
architecture (RMSNorm, RoPE, SwiGLU, plain attention, factored embeddings,
untied LM head).

This module is self-contained on purpose so non-JAX users can drop it in
without pulling ``pawn.jax.checkpoint`` (which imports JAX). The atomic
``.complete`` sentinel verification and the three exception classes mirror
``pawn.jax.checkpoint``; the duplication is intentional for the migration
window and will collapse when ``pawn.jax`` becomes the single checkpoint
module.

Inference-only: training paths (loss, optimizer state, KV cache) are not
implemented. ``model.eval()`` and ``torch.no_grad()`` are the caller's
responsibility.

Dependencies: ``torch``, ``safetensors``, ``chess_engine`` (the move
decomposition table is read once at model construction).
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

PAD_TOKEN = 1968
OUTCOME_TOKEN_BASE = 1969
NUM_ACTIONS = 1968
RMSNORM_EPS = 1e-6

# The factored-embedding override order in ``_Embedding.forward`` (PAD then
# outcome) relies on the two ranges being disjoint with PAD strictly below
# the outcome base. Enforced at import so a future vocab shift doesn't
# silently break the overrides.
assert PAD_TOKEN < OUTCOME_TOKEN_BASE, (
    f"PAD_TOKEN ({PAD_TOKEN}) must be < OUTCOME_TOKEN_BASE "
    f"({OUTCOME_TOKEN_BASE}) for the embedding override order to be correct"
)

# Canonical safetensors schema (16 keys, declaration order) — must match
# pawn.jax.checkpoint._PARAM_FIELDS. Hardcoded here so the loader stays
# JAX-dependency-free.
_SCHEMA_KEYS: tuple[str, ...] = (
    "src_embed", "dst_embed", "promo_embed", "pad_embed", "outcome_embed",
    "attn_norm", "wq", "wk", "wv", "wo",
    "ffn_norm", "w_gate", "w_up", "w_down",
    "final_norm", "lm_head",
)

# Checkpoint format version recognised by this loader. Mirrors
# ``pawn.jax.checkpoint.CHECKPOINT_FORMAT_VERSION``; bumped together when the
# schema changes.
_CHECKPOINT_FORMAT_VERSION = 1
_SENTINEL_FILE = ".complete"


# ---------------------------------------------------------------------------
# Exceptions (mirror pawn.jax.checkpoint; duplicated to keep the loader
# self-contained for non-JAX consumers).
# ---------------------------------------------------------------------------


class IncompleteCheckpointError(Exception):
    """Raised when a checkpoint directory is missing its .complete sentinel."""


class CheckpointIntegrityError(Exception):
    """Raised when a checkpoint's files do not match its .complete sentinel
    or the safetensors key set does not match the canonical schema."""


class UnsupportedCheckpointVersionError(Exception):
    """Raised when a checkpoint's format_version is not understood."""


# ---------------------------------------------------------------------------
# Stateless building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TorchModelConfig:
    """Mirror of ``pawn.jax.config.ModelConfig`` — architecture only."""

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = 1980
    max_seq_len: int = 512
    n_outcomes: int = 11
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@functools.cache
def _build_decomp_table() -> torch.Tensor:
    """token -> (src_sq, dst_sq, promo_type) lookup from the Rust engine.

    Cached: identical for every model, so the Rust call runs once per process.
    """
    from chess_engine import export_move_vocabulary

    vocab = export_move_vocabulary()
    sq: list[str] = vocab["square_names"]
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    table = torch.zeros((NUM_ACTIONS, 3), dtype=torch.int64)
    for token_idx, uci in vocab["token_to_move"].items():
        table[int(token_idx)] = torch.tensor(
            [sq.index(uci[:2]), sq.index(uci[2:4]), promo_map.get(uci[4:], 0)]
        )
    return table


def _to_f32(x: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32 if not already — used by RMSNorm/RoPE to keep the
    reduction and rotation in fp32 regardless of the residual stream dtype."""
    return x if x.dtype == torch.float32 else x.float()


def _rope_tables(
    head_dim: int, seq_len: int, base: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv)
    return freqs.cos(), freqs.sin()


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x32 = _to_f32(x)
    pairs = x32.reshape(*x.shape[:-1], -1, 2)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    out = torch.stack([out0, out1], dim=-1).reshape(x.shape)
    return out.to(x.dtype)


class _RMSNorm(nn.Module):
    """RMSNorm matching ``pawn.jax.model._rmsnorm``.

    The reduction and the weight multiply both run in fp32, the result is
    downcast once. This intentionally diverges from the legacy
    ``pawn.model.RMSNorm`` (which downcasts before the weight multiply) so a
    bf16 residual stream stays bf16 instead of being promoted by the fp32
    weight — load-bearing for JAX-checkpoint parity at non-fp32 dtypes.
    """

    weight: nn.Parameter

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = _to_f32(x)
        norm = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + RMSNORM_EPS)
        return (x32 * norm * self.weight).to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, cfg: TorchModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        def split(proj: nn.Linear) -> torch.Tensor:
            return proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(split(self.wq), cos, sin)
        k = _apply_rope(split(self.wk), cos, sin)
        v = split(self.wv)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (self.head_dim**-0.5)
        # Cast the mask into the scores' dtype so a caller that ``.to(bf16)``s
        # the module doesn't get a silent fp32 upcast through `scores + mask`.
        weights = torch.softmax(scores + mask.to(scores.dtype), dim=-1)
        # Zero fully-masked rows (degenerate all-padding sequences whose
        # softmax would otherwise be NaN), matching the JAX implementation.
        weights = weights.nan_to_num(nan=0.0)
        ctx = torch.einsum("bhqk,bhkd->bhqd", weights, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(ctx)


class _SwiGLU(nn.Module):
    def __init__(self, cfg: TorchModelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class _Block(nn.Module):
    def __init__(self, cfg: TorchModelConfig) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(cfg.d_model)
        self.attn = _Attention(cfg)
        self.ffn_norm = _RMSNorm(cfg.d_model)
        self.ffn = _SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class _Embedding(nn.Module):
    decomp_table: torch.Tensor

    def __init__(self, cfg: TorchModelConfig) -> None:
        super().__init__()
        self.src_embed = nn.Embedding(64, cfg.d_model)
        self.dst_embed = nn.Embedding(64, cfg.d_model)
        self.promo_embed = nn.Embedding(5, cfg.d_model)
        self.pad_embed = nn.Parameter(torch.zeros(cfg.d_model))
        self.outcome_embed = nn.Embedding(cfg.n_outcomes, cfg.d_model)
        self.register_buffer(
            "decomp_table", _build_decomp_table(), persistent=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        safe = tokens.clamp(0, NUM_ACTIONS - 1).long()
        decomp = self.decomp_table[safe]
        emb = (
            self.src_embed(decomp[..., 0])
            + self.dst_embed(decomp[..., 1])
            + self.promo_embed(decomp[..., 2])
        )
        is_pad = (tokens == PAD_TOKEN).unsqueeze(-1)
        emb = torch.where(is_pad, self.pad_embed, emb)
        n_outcomes = self.outcome_embed.num_embeddings
        oc_idx = (tokens - OUTCOME_TOKEN_BASE).clamp(0, n_outcomes - 1).long()
        is_outcome = (tokens >= OUTCOME_TOKEN_BASE).unsqueeze(-1)
        return torch.where(is_outcome, self.outcome_embed(oc_idx), emb)


class PAWNTorch(nn.Module):
    """Inference-only PyTorch PAWN model, loadable from JAX-format checkpoints.

    Architectural mirror of ``pawn.jax.model.PAWNModel``: RMSNorm, RoPE,
    SwiGLU FFN, plain attention, factored embeddings, untied LM head,
    pre-norm residual blocks.
    """

    def __init__(self, cfg: TorchModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = _Embedding(cfg)
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = _RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def block(self, i: int) -> _Block:
        """Typed accessor for the i-th transformer block."""
        return cast(_Block, self.layers[i])

    def forward(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``(B, T, vocab_size)`` logits.

        Args:
            tokens: ``(B, T)`` int token ids.
            attn_mask: ``(B, T)`` bool — ``True`` at real (non-padding)
                positions, ``False`` at padding. Same convention as the JAX
                model.
        """
        b, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        x = self.embed(tokens)
        cos, sin = _rope_tables(
            self.cfg.head_dim, t, self.cfg.rope_base, tokens.device
        )
        causal = torch.tril(
            torch.ones((t, t), dtype=torch.bool, device=tokens.device)
        )
        keep = causal[None, None, :, :] & attn_mask[:, None, None, :]
        mask = torch.where(
            keep, 0.0, float("-inf")
        ).to(torch.float32)
        for i in range(self.cfg.n_layers):
            x = self.block(i)(x, cos, sin, mask)
        x = self.final_norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Sentinel + load
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sentinel(directory: Path) -> None:
    """Verify ``.complete`` exists and matches the directory file-set + hashes.

    Mirrors ``pawn.jax.checkpoint._verify_sentinel`` exactly; duplicated to
    keep the loader independent of the JAX-dependent checkpoint module.
    """
    sentinel = directory / _SENTINEL_FILE
    if not sentinel.exists():
        raise IncompleteCheckpointError(
            f"{directory} is missing its {_SENTINEL_FILE} sentinel — "
            "likely a partial write from an interrupted save."
        )
    data = json.loads(sentinel.read_text(encoding="utf-8"))
    if "files" not in data:
        raise CheckpointIntegrityError(
            f"{sentinel} is malformed — missing the 'files' key"
        )
    listed: dict[str, str] = data["files"]
    present = {
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and entry.name != _SENTINEL_FILE
    }
    if present != set(listed):
        raise CheckpointIntegrityError(
            f"{directory} contains files {sorted(present)} but its "
            f"{_SENTINEL_FILE} sentinel lists {sorted(listed)}"
        )
    for name, expected in listed.items():
        actual = _sha256_file(directory / name)
        if actual != expected:
            raise CheckpointIntegrityError(
                f"SHA-256 mismatch for {name} in {directory}: "
                f"expected {expected[:16]}…, got {actual[:16]}…"
            )


def load_pawn(checkpoint_path: str | Path) -> PAWNTorch:
    """Load a JAX-format PAWN checkpoint into an inference-only PyTorch model.

    Reads ``config.json`` and ``model.safetensors`` from ``checkpoint_path``;
    if a ``.complete`` sentinel is present, verifies the file-set and SHA-256
    hashes before loading. Validates the checkpoint's ``format_version``,
    ensures the safetensors key set matches the canonical schema, then
    reverses the JAX ``(in, out)`` Linear convention to PyTorch's
    ``(out, in)`` and unstacks the per-layer leading axis into individual
    transformer blocks.

    Raises ``IncompleteCheckpointError`` / ``CheckpointIntegrityError`` on
    integrity failures and ``UnsupportedCheckpointVersionError`` on a
    format-version mismatch.
    """
    path = Path(checkpoint_path)
    if (path / _SENTINEL_FILE).exists():
        _verify_sentinel(path)
    config_path = path / "config.json"
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    version = config.get("format_version")
    if version != _CHECKPOINT_FORMAT_VERSION:
        raise UnsupportedCheckpointVersionError(
            f"checkpoint {path} has format_version {version!r}; this loader "
            f"reads version {_CHECKPOINT_FORMAT_VERSION}"
        )
    if "model_config" not in config:
        raise KeyError(
            f"{config_path} has no 'model_config' key; top-level keys: "
            f"{sorted(config.keys())}"
        )
    mc: dict[str, Any] = config["model_config"]
    cfg = TorchModelConfig(
        d_model=int(mc["d_model"]),
        n_layers=int(mc["n_layers"]),
        n_heads=int(mc["n_heads"]),
        d_ff=int(mc["d_ff"]),
        vocab_size=int(mc.get("vocab_size", 1980)),
        max_seq_len=int(mc.get("max_seq_len", 512)),
        n_outcomes=int(mc.get("n_outcomes", 11)),
        rope_base=float(mc.get("rope_base", 10000.0)),
    )
    state = load_file(path / "model.safetensors")
    if set(state) != set(_SCHEMA_KEYS):
        missing = sorted(set(_SCHEMA_KEYS) - set(state))
        extra = sorted(set(state) - set(_SCHEMA_KEYS))
        raise CheckpointIntegrityError(
            f"{path}/model.safetensors key set does not match the schema "
            f"— missing {missing}, unexpected {extra}"
        )
    model = PAWNTorch(cfg)
    with torch.no_grad():
        # Embeddings: JAX (vocab, d) matches PyTorch nn.Embedding.weight shape.
        model.embed.src_embed.weight.copy_(state["src_embed"])
        model.embed.dst_embed.weight.copy_(state["dst_embed"])
        model.embed.promo_embed.weight.copy_(state["promo_embed"])
        model.embed.pad_embed.copy_(state["pad_embed"])
        model.embed.outcome_embed.weight.copy_(state["outcome_embed"])
        # Per-layer: unstack the leading axis and transpose linear weights
        # from JAX (in, out) to PyTorch nn.Linear (out, in).
        for i in range(cfg.n_layers):
            block = model.block(i)
            block.attn_norm.weight.copy_(state["attn_norm"][i])
            block.attn.wq.weight.copy_(state["wq"][i].T.contiguous())
            block.attn.wk.weight.copy_(state["wk"][i].T.contiguous())
            block.attn.wv.weight.copy_(state["wv"][i].T.contiguous())
            block.attn.wo.weight.copy_(state["wo"][i].T.contiguous())
            block.ffn_norm.weight.copy_(state["ffn_norm"][i])
            block.ffn.w_gate.weight.copy_(state["w_gate"][i].T.contiguous())
            block.ffn.w_up.weight.copy_(state["w_up"][i].T.contiguous())
            block.ffn.w_down.weight.copy_(state["w_down"][i].T.contiguous())
        # Head.
        model.final_norm.weight.copy_(state["final_norm"])
        model.lm_head.weight.copy_(state["lm_head"].T.contiguous())
    return model
