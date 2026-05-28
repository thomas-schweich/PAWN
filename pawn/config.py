"""PAWN model configuration: vocab constants + supernet/variant dimensions.

This module is the single source of truth for the model's discrete
dimensions and the vocabulary contract. It is intentionally lightweight
— a few frozen dataclasses + module-level constants — so it can be
imported by non-training code (the dashboard, the lab MCP server)
without dragging in JAX. The Equinox ``PAWNModel`` in :mod:`pawn.model`
is built from these configs but does not own them.

Layout:

- **Vocab constants** (``PAD_TOKEN``, ``OUTCOME_TOKEN_BASE``,
  ``NUM_ACTIONS``, ``VOCAB_SIZE``, the named outcome IDs) — must stay
  in lockstep with ``engine/src/vocab.rs``. Pre-vocab-transition
  checkpoints used a ~60k-token vocabulary and are not loadable in v2;
  check out the ``pre-vocab-transition`` git tag to access them.
- **Sequence + RoPE** (``MAX_SEQ_LEN``, ``ROPE_BASE``).
- :data:`HEAD_DIM` = 64, fixed across all nested variants so width
  slices align to whole heads and RoPE phase tables are
  variant-invariant.
- :class:`ModelConfig` — frozen dataclass for one architecture's
  dimensions.
- :data:`SUPERNET` + :data:`VARIANTS` — the production supernet and
  its three nested slices.
- :data:`TINY_SUPERNET` + :data:`TINY_VARIANTS` — same shape, smaller;
  for verification runs.
- :func:`validate_nested` — assert a ``ModelConfig`` describes a valid
  nested slice of another. Called once at import for the constants
  below, and re-called by ``pawn.model.sliced`` at slicing time.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

__all__ = [
    # Vocab
    "NUM_ACTIONS",
    "PAD_TOKEN",
    "OUTCOME_TOKEN_BASE",
    "N_PRETRAINING_OUTCOMES",
    "N_TOTAL_OUTCOMES",
    "N_INPUT_TOKENS",
    "VOCAB_SIZE",
    "WHITE_CHECKMATES",
    "BLACK_CHECKMATES",
    "STALEMATE",
    "DRAW_BY_RULE",
    "PLY_LIMIT",
    "WHITE_RESIGNS",
    "BLACK_RESIGNS",
    "DRAW_BY_AGREEMENT",
    "WHITE_WINS_ON_TIME",
    "BLACK_WINS_ON_TIME",
    "DRAW_BY_TIME",
    # Sequence + RoPE
    "MAX_SEQ_LEN",
    "ROPE_BASE",
    "HEAD_DIM",
    # Bucketing
    "PRETRAIN_BUCKETS",
    # Configs
    "ModelConfig",
    "NestingError",
    "SUPERNET",
    "VARIANTS",
    "TINY_SUPERNET",
    "TINY_VARIANTS",
    "validate_nested",
]


# ---------------------------------------------------------------------------
# Vocabulary (must match engine/src/vocab.rs exactly)
# ---------------------------------------------------------------------------

NUM_ACTIONS: Final[int] = 1968
PAD_TOKEN: Final[int] = 1968
OUTCOME_TOKEN_BASE: Final[int] = 1969
N_PRETRAINING_OUTCOMES: Final[int] = 5  # Tokens 1969–1973 (natural game terminations)
N_TOTAL_OUTCOMES: Final[int] = 11       # Tokens 1969–1979 (incl. Lichess-specific)
# Total token IDs that may appear in inputs: 1968 actions + 1 PAD + 11 outcomes = 1980.
# A.2 housekeeping: lm_head output covers 1969 columns — the action tokens plus PAD.
# Outcome tokens (1969..1979) are *inputs* only (placed at position 0 under
# prepend_outcome=True) and never appear as targets — dropping their lm_head columns
# saves ~11/1980 = 0.6% of lm_head FLOPs with zero correctness impact. PAD stays in
# the output vocab because the generation diagnostic path samples it as a termination
# signal; the trade-off of dropping PAD too is examined separately in A.2-aggressive.
N_INPUT_TOKENS: Final[int] = NUM_ACTIONS + 1 + N_TOTAL_OUTCOMES  # 1980
VOCAB_SIZE: Final[int] = NUM_ACTIONS + 1  # 1969 — lm_head output width

# Named outcome token IDs (kept verbatim from v1 / engine vocab.rs)
WHITE_CHECKMATES: Final[int] = 1969
BLACK_CHECKMATES: Final[int] = 1970
STALEMATE: Final[int] = 1971
DRAW_BY_RULE: Final[int] = 1972         # 75-move, fivefold repetition, insufficient material
PLY_LIMIT: Final[int] = 1973            # Hit max plies (also used for truncated Lichess games)
WHITE_RESIGNS: Final[int] = 1974
BLACK_RESIGNS: Final[int] = 1975
DRAW_BY_AGREEMENT: Final[int] = 1976
WHITE_WINS_ON_TIME: Final[int] = 1977
BLACK_WINS_ON_TIME: Final[int] = 1978
DRAW_BY_TIME: Final[int] = 1979


# ---------------------------------------------------------------------------
# Sequence + RoPE
# ---------------------------------------------------------------------------

MAX_SEQ_LEN: Final[int] = 512
ROPE_BASE: Final[float] = 10000.0


# ---------------------------------------------------------------------------
# A.1 — Length bucketing edges
# ---------------------------------------------------------------------------
#
# Single source of truth for the pretraining bucket-edge schedule. The
# trainer compiles one program per (bucket, variant) combination and
# the persistent compilation cache carries the result across runs.
#
# Edges are ascending bucket widths in tokens. The top edge MUST equal
# the trainer's max seq_len (typically MAX_SEQ_LEN). The FLOP-model
# search in scripts/bench/bucket_search.py over 100k random games at
# SUPERNET shape gives:
#     K=2 {384, 512} → 15.6% compute savings
#     K=3 {256, 384, 512} → 20.4% compute savings  (recommended)
#     K=4 {128, 256, 384, 512} → 21.8% (diminishing returns)
#
# Set this empty tuple to disable bucketing entirely (single T=seq_len
# bucket). Trainer code handles len==0 by falling through to the
# unbucketed path.
PRETRAIN_BUCKETS: Final[tuple[int, ...]] = (256, 384, MAX_SEQ_LEN)


# ---------------------------------------------------------------------------
# Architecture: head_dim fixed across variants
# ---------------------------------------------------------------------------

HEAD_DIM: Final[int] = 64


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Dimensions of one transformer architecture.

    For a supernet, the d_* fields describe the full-width version; for
    a variant, they describe the slice taken from a supernet.
    :func:`validate_nested` checks the parent/child relationship.

    All fields are validated in ``__post_init__`` so it's impossible to
    construct a self-inconsistent config (e.g. d_model not divisible by
    n_heads).
    """

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    head_dim: int = HEAD_DIM
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    rope_base: float = ROPE_BASE
    n_outcomes: int = N_TOTAL_OUTCOMES

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"d_model ({self.d_model}) must equal head_dim ({self.head_dim}) "
                f"* n_heads ({self.n_heads})"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.d_ff < self.d_model:
            raise ValueError(
                f"d_ff ({self.d_ff}) must be ≥ d_model ({self.d_model})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.n_outcomes < 0:
            raise ValueError(f"n_outcomes must be non-negative, got {self.n_outcomes}")
        if self.rope_base <= 0:
            raise ValueError(f"rope_base must be positive, got {self.rope_base}")


class NestingError(ValueError):
    """``variant`` can't be cleanly extracted as a nested slice of ``supernet``.

    Distinct subclass of :class:`ValueError` so callers (e.g.
    ``pawn.model.sliced``) can catch nesting violations specifically
    without losing the existing ``except ValueError`` semantics.
    """


def validate_nested(variant: ModelConfig, supernet: ModelConfig) -> None:
    """Assert ``variant`` is a valid nested slice of ``supernet``.

    Every dimension of the variant must be a non-strict subset of the
    supernet's, and head_dim / vocab / seq / RoPE base must match
    exactly so the supernet's per-token, per-position, and per-head
    structure is reusable.

    Raises :class:`NestingError` on any mismatch.
    """
    if variant.head_dim != supernet.head_dim:
        raise NestingError(
            f"head_dim mismatch: variant={variant.head_dim} supernet={supernet.head_dim}"
        )
    if variant.n_layers != supernet.n_layers:
        raise NestingError(
            f"n_layers mismatch: variant={variant.n_layers} supernet={supernet.n_layers} "
            f"(variants share the supernet's depth)"
        )
    if variant.d_model > supernet.d_model:
        raise NestingError(
            f"variant d_model {variant.d_model} exceeds supernet {supernet.d_model}"
        )
    # n_heads <= supernet.n_heads is *implied* by d_model + head_dim being a
    # ModelConfig invariant (head_dim × n_heads == d_model, head_dim equal).
    # No standalone n_heads check needed.
    if variant.d_ff > supernet.d_ff:
        raise NestingError(
            f"variant d_ff {variant.d_ff} exceeds supernet {supernet.d_ff}"
        )
    if variant.vocab_size != supernet.vocab_size:
        raise NestingError(
            f"vocab_size mismatch: variant={variant.vocab_size} supernet={supernet.vocab_size}"
        )
    if variant.max_seq_len != supernet.max_seq_len:
        raise NestingError(
            f"max_seq_len mismatch: variant={variant.max_seq_len} supernet={supernet.max_seq_len}"
        )
    # Float equality is safe here because rope_base is a default literal
    # (10000.0) in every supernet/variant constant; user-constructed configs
    # that derive rope_base arithmetically should reuse the constant rather
    # than recompute it.
    if variant.rope_base != supernet.rope_base:
        raise NestingError(
            f"rope_base mismatch: variant={variant.rope_base} supernet={supernet.rope_base}"
        )
    if variant.n_outcomes != supernet.n_outcomes:
        raise NestingError(
            f"n_outcomes mismatch: variant={variant.n_outcomes} supernet={supernet.n_outcomes}"
        )


# ---------------------------------------------------------------------------
# Production supernet (large's dimensions) and its nested variants.
# small and base are *inner [:d_V, :d_V] slices* of every weight matrix;
# large IS the supernet. d_ff is set close to 8/3 × d_model (the Llama-1/2/3
# SwiGLU ratio) and rounded up to multiples of 128 for tensor-core tile
# alignment on Blackwell. The earlier 4× ratio was sized for ReLU/GeLU MLPs;
# SwiGLU's gate+up double the projection count, so a 4× width is over-
# parameterised. Switching to 8/3 cuts ~30% of FFN compute — and FFN GEMMs
# are 75% of step time per the round-3 profiler trace at LARGE — for a
# measured ~22% step-time reduction with no quality loss in the published
# Llama / PaLM ablations.
# ---------------------------------------------------------------------------

SUPERNET: Final[ModelConfig] = ModelConfig(
    d_model=640,
    n_layers=10,
    n_heads=10,
    d_ff=1792,  # 8/3 × 640 = 1706.67, next multiple of 128 is 1792 (ratio 2.8)
)

# `MappingProxyType` makes the dict structurally immutable — `VARIANTS["small"] = ...`
# raises `TypeError` rather than silently swapping a config out from under
# callers. `Final` alone would only block rebinding the name.
VARIANTS: Final[Mapping[str, ModelConfig]] = MappingProxyType(
    {
        # 8/3 × 256 = 682.67, next 128-multiple = 768 (ratio 3.0). Variant
        # ratios drift slightly above 8/3 because the inner-slice
        # constraint + 128-multiple constraint can't both hit 8/3 exactly
        # at small d_model.
        "small": ModelConfig(d_model=256, n_layers=10, n_heads=4, d_ff=768),
        # 8/3 × 512 = 1365.33, next 128-multiple = 1408 (ratio 2.75).
        "base": ModelConfig(d_model=512, n_layers=10, n_heads=8, d_ff=1408),
        "large": SUPERNET,
    }
)


# ---------------------------------------------------------------------------
# Tiny supernet (for verification runs that don't need production scale).
# Same nesting structure as the production set; same head_dim.
# ---------------------------------------------------------------------------

TINY_SUPERNET: Final[ModelConfig] = ModelConfig(
    d_model=192,
    n_layers=4,
    n_heads=3,
    d_ff=768,
)

TINY_VARIANTS: Final[Mapping[str, ModelConfig]] = MappingProxyType(
    {
        "small": ModelConfig(d_model=64, n_layers=4, n_heads=1, d_ff=256),
        "base": ModelConfig(d_model=128, n_layers=4, n_heads=2, d_ff=512),
        "large": TINY_SUPERNET,
    }
)


def _check_constants_nest_at_import() -> None:
    """Sanity-check the module-level supernet/variant pairs once at import.

    Catches any drift between :data:`SUPERNET` and :data:`VARIANTS` (and
    likewise the TINY pair) so a typo in the dimensions surfaces as an
    import-time error instead of mid-training.
    """
    for cfg in VARIANTS.values():
        validate_nested(cfg, SUPERNET)
    for cfg in TINY_VARIANTS.values():
        validate_nested(cfg, TINY_SUPERNET)


_check_constants_nest_at_import()
