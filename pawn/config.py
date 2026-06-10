"""PAWN model configuration: vocab constants + supernet/variant dimensions.

This module is the single source of truth for the model's discrete
dimensions and the vocabulary contract. It is intentionally lightweight
— a few frozen dataclasses + module-level constants — so it can be
imported by non-training code (the dashboard, the lab MCP server)
without dragging in JAX. The Equinox ``PAWNModel`` in :mod:`pawn.model`
is built from these configs but does not own them.

Layout:

- **Vocab constants** (``PAD_TOKEN``, ``OUTCOME_TOKEN_BASE``,
  ``NUM_ACTIONS``, ``BOS_TOKEN``, ``NULL_TOKEN``, ``N_CONTROL_RESERVED``,
  ``VOCAB_SIZE``, the named outcome IDs). The action / PAD / outcome IDs
  stay in lockstep with ``engine/src/vocab.rs``; ``BOS``/``NULL``/reserved
  are Python-side control tokens above the engine's emission space.
  ``VOCAB_SIZE`` (``V`` = 2000) is the model's uniform input/output vocab.
  Pre-vocab-transition
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
    "BOS_TOKEN",
    "NULL_TOKEN",
    "N_CONTROL_RESERVED",
    "VOCAB_SIZE",
    "CONDITIONING_KINDS",
    "MASK_VERSION",
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
    "V1_VOCAB_SIZE",
    "FACTORED_V1_LARGE",
    "TINY_FACTORED",
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

# Python-side control tokens. The Rust engine emits only moves + PAD + outcomes
# (IDs 0..1979); BOS / NULL / reserved are assembled by Python (the conditioning
# prefix) and never come off the engine. ``engine/src/vocab.rs`` keeps its own
# ``VOCAB_SIZE = 1980`` for the engine's emission space — distinct from the
# model's full vocab below.
BOS_TOKEN: Final[int] = 1980            # Sequence-start, slot 0 of every prefix
NULL_TOKEN: Final[int] = 1981           # Fills a conditioning slot a game lacks
N_CONTROL_RESERVED: Final[int] = 18     # Tokens 1982–1999 reserved for future control kinds

# Uniform input/output vocab. Input vocab == output vocab == V; there is no
# N_INPUT_TOKENS / VOCAB_SIZE split. Reserved / NULL / control columns (≥1981)
# exist in the embedding + logit tables but are masked to -inf before softmax-CE
# so they can't be sampled or accrue gradient.
VOCAB_SIZE: Final[int] = 2000           # V = 1968 actions + PAD + 11 outcomes + BOS + NULL + 18 reserved

# ---------------------------------------------------------------------------
# Conditioning prefix (Phase-A Chunk 4)
# ---------------------------------------------------------------------------
#
# Every sequence is assembled as ``[BOS][cond…][ply…][PAD…]`` where the
# ``[cond…]`` slots carry one control token per entry in the run's
# ``conditioning`` list. ``CONDITIONING_KINDS`` is the registry of valid
# kinds; ``BaseRunConfig.conditioning`` is validated against it and the
# shared assembler in :mod:`pawn.corpus` resolves each kind to its
# per-game token (or :data:`NULL_TOKEN` when a game lacks that value).
#
# The prefix width is ``C = 1 + len(conditioning)`` — BOS is always
# present, so ``C >= 1`` even with no conditioning. Move position ``i``
# (0-indexed) lives at sequence slot ``C + i``; the loss is supervised
# on slots ``[C-1 .. C-1 + game_length - 1]`` (the first-move
# ``prefix→ply_1`` prediction IS supervised, the predict-PAD slot is not).
#
# The registry is a frozenset so it is hashable + order-independent; the
# *order* a run conditions in is the order of its ``conditioning`` list,
# not this set.
CONDITIONING_KINDS: Final[frozenset[str]] = frozenset({"outcome"})

# Layout/mask contract version. Bumped whenever the prefix-assembly /
# loss-mask / move-position convention changes. Baked into every
# ``config.json`` (via :mod:`pawn.checkpoint`) AND the lichess on-disk
# cache key (:mod:`pawn.lichess_data`); a load-time assert refuses a
# checkpoint / cache entry whose ``mask_version`` doesn't match the
# builder's. Version 1 is the ``[BOS][cond…][ply…]`` layout with the
# first-move-supervised / predict-PAD-excluded loss mask (Chunk 4); the
# pre-Chunk-4 un-prefixed v1 contract was the implicit version 0.
MASK_VERSION: Final[int] = 1

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
    # Untied by default. A tied output head (one ``embed_tokens[V,d]`` used as
    # both the input lookup and, transposed, the output projection) with NO
    # logit regularisation collapses the output distribution to uniform in a
    # single gradient step during pretraining: the cross-entropy gradient
    # ``(softmax − onehot)·x`` has a column-common component that drives all V
    # rows of ``embed_tokens`` toward collinearity → ``embed_tokens.T`` goes
    # near-rank-1 → constant logits → ``loss = ln(V)`` dead fixed point. Global
    # grad-norm clipping bounds the magnitude but not the *direction* of that
    # step, so it does not prevent it. v1 was stable because input embeddings
    # and the output projection (``lm_head``) were separate tensors with
    # independent gradients. Untied is therefore the canonical v2 default; set
    # ``tie_embeddings=True`` only with a logit scale + z-loss (GPT-2/PaLM
    # recipe), which this code does not yet add.
    tie_embeddings: bool = False
    # v1-architecture factored embeddings (``src+dst+promo`` summed per move
    # token + standalone PAD/outcome rows) instead of the uniform
    # ``embed_tokens[V, d]`` table. Selects
    # :class:`pawn.factored_model.FactoredPAWNModel` in the checkpoint layer
    # and the trainer. Requires ``tie_embeddings=False`` — the factored input
    # path has no ``[V, d]`` table to tie the output head against. Default
    # False keeps every existing config / checkpoint byte-compatible
    # (``config.json`` files without the key parse to False).
    factored_embeddings: bool = False

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
        if self.factored_embeddings and self.tie_embeddings:
            raise ValueError(
                "factored_embeddings is incompatible with tie_embeddings: the "
                "factored input path has no [V, d] table to tie the output "
                "head against"
            )


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

    B.5: variants may use **fewer layers** than the supernet — a
    variant with ``n_layers < supernet.n_layers`` is sliced as the
    supernet's first ``variant.n_layers`` layers (a depth-and-width
    slice). The variant's per-layer weights still nest by width
    under the supernet's per-layer weights.

    Raises :class:`NestingError` on any mismatch.
    """
    if variant.head_dim != supernet.head_dim:
        raise NestingError(
            f"head_dim mismatch: variant={variant.head_dim} supernet={supernet.head_dim}"
        )
    if variant.n_layers > supernet.n_layers:
        raise NestingError(
            f"variant n_layers {variant.n_layers} exceeds supernet {supernet.n_layers}"
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
    # Tied vs untied embeddings change the field set (a tied model has no
    # standalone lm_head), so a variant must match the supernet's choice to
    # share weight tensors under the nested slice.
    if variant.tie_embeddings != supernet.tie_embeddings:
        raise NestingError(
            f"tie_embeddings mismatch: variant={variant.tie_embeddings} "
            f"supernet={supernet.tie_embeddings}"
        )
    # Factored vs uniform embeddings are different field sets entirely —
    # no nested-slice relationship exists across the two architectures.
    if variant.factored_embeddings != supernet.factored_embeddings:
        raise NestingError(
            f"factored_embeddings mismatch: variant={variant.factored_embeddings} "
            f"supernet={supernet.factored_embeddings}"
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


# ---------------------------------------------------------------------------
# Factored (v1-architecture) configs — the LR-schedule confound experiment.
# FACTORED_V1_LARGE reproduces v1-large's published architecture exactly
# (thomas-schweich/pawn-large config.json: d=640, 10 layers, 8 heads ×
# head_dim 80, d_ff=2560, vocab 1980, max_seq_len 512, rope_base 10000,
# 11 outcomes; lm_head untied). 66.91M params. Trained by
# `scripts/train_jax.py --arch factored-v1` under the v2 recipe so the
# architecture is the only variable vs. the v2-large teacher run.
# ---------------------------------------------------------------------------

# v1's emission-space vocabulary: 1,968 actions + PAD + 11 outcomes. No
# BOS / NULL / reserved rows — v1 trained on bare move sequences.
V1_VOCAB_SIZE: Final[int] = NUM_ACTIONS + 1 + N_TOTAL_OUTCOMES  # = 1980

FACTORED_V1_LARGE: Final[ModelConfig] = ModelConfig(
    d_model=640,
    n_layers=10,
    n_heads=8,
    d_ff=2560,
    head_dim=80,  # v1's native head_dim (640 / 8), not the v2 nesting 64
    vocab_size=V1_VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    rope_base=ROPE_BASE,
    n_outcomes=N_TOTAL_OUTCOMES,
    tie_embeddings=False,
    factored_embeddings=True,
)

# Tiny factored config for unit tests / smoke runs (same head_dim:d_model
# ratio family as the tiny supernet; dims chosen so a CPU/GPU test forward
# is fast).
TINY_FACTORED: Final[ModelConfig] = ModelConfig(
    d_model=96,
    n_layers=2,
    n_heads=2,
    d_ff=192,
    head_dim=48,
    vocab_size=V1_VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    rope_base=ROPE_BASE,
    n_outcomes=N_TOTAL_OUTCOMES,
    tie_embeddings=False,
    factored_embeddings=True,
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
