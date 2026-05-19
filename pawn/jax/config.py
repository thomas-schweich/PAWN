"""Architecture configuration for the JAX/Equinox PAWN model.

PAWN trains a single shared-weight *supernet*; the small/base/large variants
are nested slices of it (see ``docs/jax-migration.md`` §5). Every variant fixes
``head_dim = 64`` so the width slices are exact and RoPE is variant-invariant.

``ModelConfig`` is the generic per-model architecture description. It also
describes standalone (non-supernet) models — e.g. a legacy PyTorch checkpoint
converted to JAX keeps its original head count and need not nest.
"""

from __future__ import annotations

from dataclasses import dataclass

# Token vocabulary — must match engine/src/vocab.rs and pawn.config.
NUM_ACTIONS = 1968
PAD_TOKEN = 1968
OUTCOME_TOKEN_BASE = 1969
N_OUTCOMES = 11
VOCAB_SIZE = 1980
MAX_SEQ_LEN = 512


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for one PAWN model.

    Used for the supernet, for each sliced variant, and for standalone models
    such as converted legacy checkpoints.
    """

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    n_outcomes: int = N_OUTCOMES
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# The supernet — large's dimensions. head_dim = 640 / 10 = 64.
SUPERNET = ModelConfig(d_model=640, n_layers=10, n_heads=10, d_ff=2560)

# Variants — nested slices of SUPERNET. All three have head_dim = 64.
# NOTE: "large" *is* the supernet (10 heads, head_dim 64). The legacy PyTorch
# pawn-large checkpoint used 8 heads (head_dim 80); a converted legacy
# checkpoint is therefore a standalone ModelConfig with n_heads=8 — never
# VARIANTS["large"], and validate_nested() would correctly reject it.
VARIANTS: dict[str, ModelConfig] = {
    "small": ModelConfig(d_model=256, n_layers=8, n_heads=4, d_ff=1024),
    "base": ModelConfig(d_model=512, n_layers=8, n_heads=8, d_ff=2048),
    "large": SUPERNET,
}

# Tiny config for tests only — does not nest into SUPERNET.
TOY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256)

_NESTED_EQUAL_FIELDS = ("vocab_size", "max_seq_len", "n_outcomes", "rope_base")
_NESTED_LEQ_FIELDS = ("d_model", "n_layers", "d_ff")


def validate_nested(variant: ModelConfig, supernet: ModelConfig) -> None:
    """Raise ``ValueError`` unless ``variant`` is a valid slice of ``supernet``.

    A variant slices cleanly only if it shares the supernet's head dimension
    (so a width slice drops whole heads), keeps the vocab / context / outcome
    layout identical, and is no larger than the supernet on any sliced axis.
    """
    if variant.head_dim != supernet.head_dim:
        raise ValueError(
            f"variant head_dim={variant.head_dim} != supernet "
            f"head_dim={supernet.head_dim}; width slices would not align to heads"
        )
    for field in _NESTED_EQUAL_FIELDS:
        if getattr(variant, field) != getattr(supernet, field):
            raise ValueError(
                f"variant {field}={getattr(variant, field)} must equal supernet "
                f"{field}={getattr(supernet, field)}"
            )
    for field in _NESTED_LEQ_FIELDS:
        if getattr(variant, field) > getattr(supernet, field):
            raise ValueError(
                f"variant {field}={getattr(variant, field)} exceeds supernet "
                f"{field}={getattr(supernet, field)}"
            )
