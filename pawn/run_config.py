"""Pydantic models for all PAWN training configurations.

``RunConfig`` is the single source of truth for every parameter that
training scripts, sweep tooling, and the lab MCP server care about.
JSON Schema is derived automatically — call
``PretrainConfig.model_json_schema()`` or
``AdapterConfig.model_json_schema()`` for introspection.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class BaseRunConfig(BaseModel):
    """Fields shared by pretraining and adapter runs."""

    # Data ----------------------------------------------------------------
    elo_min: int | None = None
    elo_max: int | None = None
    max_games: int | None = None
    val_games: int = 50_000
    min_ply: int = 10

    # Training ------------------------------------------------------------
    total_steps: int | None = None
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_frac: float = 0.05
    warmup_steps: int | None = None  # explicit override of warmup_frac
    max_grad_norm: float = 1.0
    patience: int | None = None
    eval_interval: int | None = None
    log_interval: int = 100
    pause_after_steps: int | None = None

    # Ablations -----------------------------------------------------------
    mate_boost: float = 0.0
    no_outcome_token: bool = False
    discard_ply_limit: bool = False

    # GPU / precision -----------------------------------------------------
    amp_dtype: Literal["float16", "bfloat16", "none"] = "bfloat16"
    no_compile: bool = False
    sdpa_math: bool = False
    num_workers: int = 4
    device: str = "cuda"

    # IO ------------------------------------------------------------------
    log_dir: str | None = None
    hf_repo: str | None = None
    local_checkpoints: bool = False
    resume: str | None = None
    wandb: bool = False
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _check_checkpoint_mode(self) -> "BaseRunConfig":
        if self.hf_repo and self.local_checkpoints:
            raise ValueError(
                "--hf-repo and --local-checkpoints are mutually exclusive"
            )
        if not self.hf_repo and not self.local_checkpoints:
            raise ValueError(
                "One of --hf-repo or --local-checkpoints is required"
            )
        return self


class PretrainConfig(BaseRunConfig):
    """Pretraining from scratch on random games."""

    run_type: Literal["pretrain"] = "pretrain"
    variant: Literal["toy", "small", "base", "large"] = "base"

    # Architecture overrides (optional — override variant defaults)
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None

    # Pretrain-specific
    accumulation_steps: int = 1
    checkpoint_interval: int = 5000
    max_seq_len: int = 256
    legality_late_ply: int | None = None  # defaults to max_seq_len // 2 at runtime
    val_games: int = 512  # override BaseRunConfig's 50K — pretrain uses on-the-fly data


class AdapterConfig(BaseRunConfig):
    """Adapter finetuning on Lichess data."""

    run_type: Literal["adapter"] = "adapter"
    strategy: Literal[
        "bottleneck", "lora", "film", "sparse",
        "rosa", "hybrid", "specialized_clm", "unfreeze",
    ]
    checkpoint: str = "thomas-schweich/pawn-base"
    pgn: str = "thomas-schweich/pawn-lichess-full"

    # Placement -----------------------------------------------------------
    adapter_layers: str | None = None

    # Bottleneck component ------------------------------------------------
    bottleneck_dim: int | None = None
    no_adapt_attn: bool = False
    no_adapt_ffn: bool = False

    # LoRA component ------------------------------------------------------
    lora_rank: int | None = None
    lora_targets: Literal["qkvo", "qv", "qkv"] | None = None
    lora_ffn: bool = False

    # Sparse component ----------------------------------------------------
    density: float | None = None
    sparse_targets: Literal["qkvo", "qv", "qkv"] | None = None
    sparse_ffn: bool = False

    # FiLM component ------------------------------------------------------
    use_output_film: bool = False

    # RoSA ----------------------------------------------------------------
    rosa_mode: Literal["rosa", "retro-sparse", "retro-bottleneck"] | None = (
        None
    )
    rosa_warmup_steps: int = 128
    mask_samples: int = 32
    grad_alpha: Literal[1, 2] = 2

    # Unfreeze / specialized_clm ------------------------------------------
    unfreeze_layers: str | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None

    # Adapter training specific -------------------------------------------
    epochs: int = 50
    val_every: int = 1


RunConfig = Annotated[
    Union[PretrainConfig, AdapterConfig], Field(discriminator="run_type")
]
"""Discriminated union of all run config types."""
