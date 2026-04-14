"""PAWN model and training configuration."""

from dataclasses import dataclass


# ---- New vocabulary (searchless_chess) ----
# Token layout: actions [0, 1967], PAD [1968], outcomes [1969, 1979]
# Must match engine/src/vocab.rs

PAD_TOKEN = 1968
OUTCOME_TOKEN_BASE = 1969
NUM_ACTIONS = 1968

# Pretraining outcomes (random games — natural terminations)
WHITE_CHECKMATES = 1969
BLACK_CHECKMATES = 1970
STALEMATE = 1971
DRAW_BY_RULE = 1972       # 75-move, fivefold repetition, insufficient material
PLY_LIMIT = 1973          # Hit max plies (also used for truncated Lichess games)

# Lichess-specific outcomes (finetuning data)
WHITE_RESIGNS = 1974      # Normal termination, white wins, no checkmate
BLACK_RESIGNS = 1975      # Normal termination, black wins, no checkmate
DRAW_BY_AGREEMENT = 1976  # Normal termination, draw, not stalemate
WHITE_WINS_ON_TIME = 1977 # Time forfeit, white wins
BLACK_WINS_ON_TIME = 1978 # Time forfeit, black wins
DRAW_BY_TIME = 1979       # Time forfeit, draw (insufficient mating material)

N_PRETRAINING_OUTCOMES = 5  # Tokens 1969-1973
N_TOTAL_OUTCOMES = 11       # Tokens 1969-1979


@dataclass
class CLMConfig:
    """Model architecture hyperparameters."""

    # Vocabulary: 1968 actions + 1 PAD + 11 outcomes = 1980
    vocab_size: int = 1980
    max_seq_len: int = 512
    n_outcomes: int = N_TOTAL_OUTCOMES

    # Transformer
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048  # 4x expansion
    dropout: float = 0.0

    # RoPE
    rope_base: float = 10000.0

    @classmethod
    def small(cls) -> "CLMConfig":
        """~9.5M parameters."""
        return cls(d_model=256, n_layers=8, n_heads=4, d_ff=1024)

    @classmethod
    def base(cls) -> "CLMConfig":
        """~35.8M parameters (default)."""
        return cls()

    @classmethod
    def large(cls) -> "CLMConfig":
        """~68.4M parameters."""
        return cls(d_model=640, n_layers=10, n_heads=8, d_ff=2560)

    @classmethod
    def toy(cls) -> "CLMConfig":
        return cls(
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    total_steps: int = 100_000

    # Batch
    batch_size: int = 256
    max_ply: int = 512  # Matches max_seq_len (no outcome prefix by default)
    discard_ply_limit: bool = False  # Only train on games that ended naturally
    num_workers: int = 4

    # Precision
    use_amp: bool = True

    # Gradient accumulation
    accumulation_steps: int = 1

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 5000

    # Pause (for LR exploration — stop early without killing)
    pause_after_steps: int | None = None

    # Ablations
    prepend_outcome: bool = False  # Prepend outcome token at position 0 for outcome-conditioned training
    mate_boost: float = 0.0  # Probability of taking mate-in-1 (0.0=random, 1.0=always)

    # Seeds
    base_seed: int = 42
    val_seed: int = (2**63) - 1
    val_games: int = 512

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "pawn"

    # Device
    device: str = "cuda"

    @classmethod
    def toy(cls) -> "TrainingConfig":
        return cls(
            lr=1e-3,
            batch_size=32,
            total_steps=5000,
            warmup_steps=100,
            eval_interval=100,
            checkpoint_interval=1000,
            num_workers=2,
            use_amp=False,
            val_games=64,
        )
