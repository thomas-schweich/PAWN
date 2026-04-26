"""Pydantic models for all PAWN training configurations.

``RunConfig`` is the single source of truth for every parameter that
training scripts, sweep tooling, and the lab MCP server care about.
JSON Schema is derived automatically — call
``PretrainConfig.model_json_schema()`` or
``AdapterConfig.model_json_schema()`` for introspection.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseRunConfig(BaseModel):
    """Fields shared by pretraining and adapter runs."""

    # Fail fast when callers pass stale CLI/JSON fields (e.g. the old
    # ``legacy_vocab`` flag, or misspelled/removed hybrid args). Pydantic
    # defaults to silently ignoring unknown fields, which makes it easy
    # to burn a whole training run on a config that doesn't mean what
    # the user thinks it means.
    model_config = ConfigDict(extra="forbid")

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
    # LR schedule shape:
    #   ``"cosine"``    — classic warmup → cosine decay to 0.
    #   ``"wsd"``       — Warmup-Stable-Decay: hold peak LR through the
    #                     bulk of training and only decay over the
    #                     final ``decay_frac`` of steps. Useful when
    #                     the cosine tail is leaving accuracy on the
    #                     table.
    #   ``"constant"``  — linear warmup → hold peak LR indefinitely.
    #                     Pair with ``patience`` to stop on val plateau.
    #   ``"one_cycle"`` — Smith (2018) one-cycle: ramp up from
    #                     peak/25 → peak over ``warmup_frac`` of steps,
    #                     then cosine-decay to peak/10000 over the rest.
    #   ``"infinite"``  — warmup → cosine cooldown to
    #                     ``stable_lr_ratio * peak`` over ``cooldown_frac``
    #                     of steps → flat stable plateau → final decay
    #                     to 0 over the last ``decay_frac`` of steps.
    #                     The stable-phase LR is independent of
    #                     ``total_steps``, so any checkpoint in that
    #                     phase is a valid resumption point: extend
    #                     ``total_steps`` on resume and the plateau
    #                     simply lasts longer. See Hägele et al. (2024)
    #                     arXiv:2405.18392.
    lr_schedule: Literal[
        "cosine", "wsd", "constant", "one_cycle", "infinite"
    ] = "cosine"
    # Fraction of ``total_steps`` used for the WSD decay phase (and
    # reused for the final decay of the ``"infinite"`` schedule).
    # Ignored for schedules without a final decay. For WSD the stable
    # phase gets ``1 - warmup_frac - decay_frac``; for "infinite" the
    # stable plateau gets ``1 - warmup_frac - cooldown_frac - decay_frac``.
    decay_frac: float = 0.1
    # Decay-phase curve: ``"linear"`` (default) or ``"cosine"`` for a
    # half-cosine fall. Used for WSD's single decay and for the final
    # decay of ``"infinite"``. Ignored otherwise. (Historical name
    # retained for backwards compatibility; applies beyond WSD now.)
    wsd_decay_shape: Literal["linear", "cosine"] = "linear"
    # Fraction of ``total_steps`` used for the ``"infinite"`` schedule's
    # peak → stable cosine cooldown. Ignored unless
    # ``lr_schedule == "infinite"``.
    cooldown_frac: float = 0.2
    # Stable plateau LR for the ``"infinite"`` schedule, expressed as a
    # fraction of the peak LR. Ignored unless
    # ``lr_schedule == "infinite"``.
    stable_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int | None = None
    eval_interval: int | None = None
    log_interval: int = 100
    pause_after_steps: int | None = None

    # Ablations -----------------------------------------------------------
    mate_boost: float = 0.0
    prepend_outcome: bool = False  # Prepend outcome token at position 0 for outcome-conditioned training
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
    wandb_project: str = "pawn"
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
    variant: Literal["toy", "small", "base", "large", "custom"] = "base"

    # Architecture overrides. With one of the named presets, these override
    # preset defaults on a per-field basis. With variant="custom" all four
    # must be set explicitly — there is no underlying preset to fall back on.
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None

    # Pretrain-specific
    accumulation_steps: int = 1
    checkpoint_interval: int = 5000
    max_seq_len: int = 512
    legality_late_ply: int | None = None  # defaults to max_seq_len // 2 at runtime
    val_games: int = 512  # override BaseRunConfig's 50K — pretrain uses on-the-fly data

    @model_validator(mode="after")
    def _check_custom_arch(self) -> "PretrainConfig":
        if self.variant == "custom":
            missing = [
                n for n, v in (
                    ("d_model", self.d_model),
                    ("n_layers", self.n_layers),
                    ("n_heads", self.n_heads),
                    ("d_ff", self.d_ff),
                )
                if v is None
            ]
            if missing:
                raise ValueError(
                    f"variant='custom' requires {', '.join(missing)} to be set explicitly"
                )
            # Narrowed by the missing-fields check above, but pyright
            # can't follow that across the `if`.
            assert self.d_model is not None
            assert self.n_heads is not None
            assert self.n_layers is not None
            assert self.d_ff is not None
            if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0 or self.d_ff <= 0:
                raise ValueError(
                    f"variant='custom' requires positive d_model/n_heads/n_layers/d_ff, "
                    f"got d_model={self.d_model}, n_heads={self.n_heads}, "
                    f"n_layers={self.n_layers}, d_ff={self.d_ff}"
                )
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"variant='custom' requires d_model ({self.d_model}) to be "
                    f"divisible by n_heads ({self.n_heads}); otherwise attention "
                    f"head_dim is non-integer and the model crashes at first forward"
                )
        return self


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
    checkpoint_interval: int = 5000

    # Data sizing ---------------------------------------------------------
    # ``steps_per_epoch`` is the canonical way to size an adapter run.
    # ``"all"`` resolves at runtime to ``n_train // (batch_size *
    # accumulation_steps)`` once the cache materializes; the resolved
    # integer is what gets written to ``run_config.json`` so the saved
    # config never holds the sentinel string.
    steps_per_epoch: int | Literal["all"] | None = None
    # Per-epoch shuffle seed. ``None`` lets the trainer pick at run start
    # and persist into the checkpoint. Set explicitly for full
    # reproducibility across machines.
    data_seed: int | None = None

    # Legality handling ---------------------------------------------------
    # By default, adapter training masks illegal moves to -inf before
    # cross-entropy. Set ``disable_legal_mask=True`` to compute the loss
    # over the full vocabulary — matching pretraining, where the model
    # has to learn legality from scratch. This tests whether the
    # backbone's legality understanding survives adapter finetuning.
    disable_legal_mask: bool = False
    # Additional penalty (lambda) on probability mass assigned to
    # illegal moves. Added to the loss as ``illegal_penalty *
    # mean_illegal_prob_mass``. Only meaningful when
    # ``disable_legal_mask=True`` (otherwise illegal probability mass
    # is zero by construction). Higher values push the model harder to
    # keep its own legality signal intact.
    illegal_penalty: float = 0.0

    @model_validator(mode="after")
    def _check_legality_flags(self) -> "AdapterConfig":
        if self.illegal_penalty < 0:
            raise ValueError(
                f"illegal_penalty must be >= 0, got {self.illegal_penalty}"
            )
        if self.illegal_penalty > 0 and not self.disable_legal_mask:
            raise ValueError(
                "illegal_penalty > 0 has no effect while legal masking is "
                "active (all illegal logits are -inf, so illegal prob mass "
                "is zero by construction). Pass disable_legal_mask=True to "
                "let the model assign probability to illegal moves and be "
                "penalized for it."
            )
        return self

    @model_validator(mode="after")
    def _check_data_sizing(self) -> "AdapterConfig":
        # ``"all"`` is the only legal string; positive ints; ``None``.
        spe = self.steps_per_epoch
        if isinstance(spe, str) and spe != "all":
            raise ValueError(
                f"steps_per_epoch must be a positive int or 'all', got {spe!r}"
            )
        if isinstance(spe, int) and spe <= 0:
            raise ValueError(
                f"steps_per_epoch must be > 0 when given as an int, got {spe}"
            )
        if spe is not None and self.max_games is not None:
            raise ValueError(
                "steps_per_epoch and max_games are mutually exclusive. "
                "max_games is deprecated; pass steps_per_epoch (int or "
                "'all') directly."
            )
        # Emit the deprecation warning only when the user actually set
        # ``max_games`` (not when it inherits its ``None`` default), so
        # that re-loading a saved config that wrote ``steps_per_epoch``
        # is silent.
        if (
            self.max_games is not None
            and "max_games" in self.model_fields_set
        ):
            warnings.warn(
                "max_games is deprecated for adapter runs; pass "
                "steps_per_epoch (int or 'all') instead. "
                "max_games is interpreted as "
                "`steps_per_epoch = max_games // batch_size`.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self


class CotrainVariant(BaseModel):
    """Per-variant spec within a co-training run.

    Architecture fields override the preset selected by ``variant``. With
    ``variant="custom"`` there is no preset — ``d_model``, ``n_layers``,
    ``n_heads``, and ``d_ff`` must all be set explicitly.
    ``resume`` is set automatically by ``lab_resume`` — not user-facing.
    """

    name: str
    variant: Literal["toy", "small", "base", "large", "custom"] = "base"
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None
    max_seq_len: int = 512
    resume: str | None = None

    @model_validator(mode="after")
    def _check_custom_arch(self) -> "CotrainVariant":
        if self.variant == "custom":
            missing = [
                n for n, v in (
                    ("d_model", self.d_model),
                    ("n_layers", self.n_layers),
                    ("n_heads", self.n_heads),
                    ("d_ff", self.d_ff),
                )
                if v is None
            ]
            if missing:
                raise ValueError(
                    f"variant='custom' requires {', '.join(missing)} to be "
                    f"set explicitly on variant '{self.name}'"
                )
            assert self.d_model is not None
            assert self.n_heads is not None
            assert self.n_layers is not None
            assert self.d_ff is not None
            if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0 or self.d_ff <= 0:
                raise ValueError(
                    f"variant='custom' on '{self.name}' requires positive "
                    f"d_model/n_heads/n_layers/d_ff, got d_model={self.d_model}, "
                    f"n_heads={self.n_heads}, n_layers={self.n_layers}, "
                    f"d_ff={self.d_ff}"
                )
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"variant='custom' on '{self.name}' requires d_model "
                    f"({self.d_model}) to be divisible by n_heads "
                    f"({self.n_heads}); otherwise attention head_dim is "
                    f"non-integer and the model crashes at first forward"
                )
        return self


class CotrainConfig(BaseRunConfig):
    """Co-training multiple model variants on shared data batches."""

    run_type: Literal["cotrain"] = "cotrain"
    variants: list[CotrainVariant]
    checkpoint_interval: int = 5000
    shm_checkpoints: bool = False
    val_games: int = 512  # override BaseRunConfig's 50K — pretrain uses on-the-fly data

    # Post-training evaluation — runs per-slot probes, diagnostics, and
    # (optional) Lichess eval on each variant's best checkpoint after
    # training completes. Writes eval_results.json to each variant's
    # run directory.
    run_evals: bool = False
    lichess_pgn: str | None = None
    publish_results: bool = False

    @model_validator(mode="after")
    def _check_cotrain(self) -> "CotrainConfig":
        if not self.variants:
            raise ValueError("variants must contain at least one entry")
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)):
            raise ValueError(f"variant names must be unique, got {names}")
        if self.shm_checkpoints and not self.hf_repo:
            raise ValueError(
                "--shm-checkpoints requires --hf-repo "
                "(HF is the only durable store)"
            )
        if self.publish_results and not self.hf_repo:
            raise ValueError(
                "publish_results requires hf_repo — there's no branch to "
                "push eval_results.json to without one"
            )
        if self.publish_results and not self.run_evals:
            raise ValueError(
                "publish_results requires run_evals=True — "
                "run_post_training_evals is the only thing that writes "
                "the eval_results.json file, so there's nothing to "
                "publish without it"
            )
        if self.lichess_pgn and not self.run_evals:
            raise ValueError(
                "lichess_pgn is only meaningful when run_evals=True"
            )
        if self.resume is not None:
            raise ValueError(
                "CotrainConfig does not use the top-level 'resume' field. "
                "Set 'resume' on each variant in the 'variants' list instead."
            )
        return self


RunConfig = Annotated[
    Union[PretrainConfig, AdapterConfig, CotrainConfig],
    Field(discriminator="run_type"),
]
"""Discriminated union of all run config types."""
