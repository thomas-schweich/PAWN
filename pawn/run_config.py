"""Pydantic models for every PAWN v2 training configuration.

:class:`RunConfig` is the discriminated-union root that every JAX-side
training driver, sweep tool, and the lab MCP server reads through. The
field set matches v1 ``pawn/run_config.py`` verbatim (per plan §7's
backward-compatibility contract — the names users type and the schemas
they grep are durable) plus a handful of v2-specific additions
(``supernet``, ``variant``, ``k``, ``max_corpus_gb``, ``seq_len``).
Torch-only fields (``amp_dtype``, ``device``, ``num_workers``,
``no_compile``, ``sdpa_math``) are dropped — JAX manages its own
backend.

The v1 ``CotrainConfig`` is GONE BY DESIGN: the supernet's joint loss
replaces multi-variant co-training, so the pretrainer in S6 trains the
supernet directly and slices the three variants at publish time.

Discriminated union (``run_type``):
- ``"pretrain"`` → :class:`PretrainConfig` — pretrain the supernet.
- ``"adapter"`` → :class:`AdapterConfig` — adapter finetuning on a
  Lichess Elo band with one of 8 strategies.
- ``"specialized_clm"`` → :class:`SpecializedCLMConfig` — train a
  standalone from-scratch CLM (no backbone), distinct from
  ``--strategy specialized_clm`` which dispatches through the
  adapter trainer.

Every model sets ``extra="forbid"`` so a stale field (renamed flag,
misspelled JSON key) raises at config-load time rather than silently
training the wrong thing.

JSON Schema is derived automatically — ``PretrainConfig.model_json_schema()``
/ ``AdapterConfig.model_json_schema()`` / ``SpecializedCLMConfig.model_json_schema()``
are the public surfaces the lab MCP server's ``lab_schema`` returns.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BaseRunConfig",
    "PretrainConfig",
    "AdapterConfig",
    "SpecializedCLMConfig",
    "RunConfig",
]


# Helpful aliases so per-config Literal[...] declarations stay readable.
SupernetName = Literal["tiny", "production"]
VariantName = Literal["small", "base", "large"]
LRScheduleName = Literal["cosine", "wsd", "constant", "one_cycle", "infinite"]
LRDecayShape = Literal["linear", "cosine"]


class BaseRunConfig(BaseModel):
    """Fields shared by every JAX-side run config.

    Subclasses add the fields particular to their training mode
    (pretrain / adapter / specialized_clm) plus their ``run_type``
    discriminator.
    """

    # `extra="forbid"` is the keystone: a misspelled or removed field
    # raises at config-load time. Pydantic's default of silently dropping
    # unknown fields would let a stale `--legacy-vocab` flag survive a
    # rename and burn a whole training run.
    model_config = ConfigDict(extra="forbid")

    # --- Data ----------------------------------------------------------
    elo_min: int | None = None
    elo_max: int | None = None
    max_games: int | None = None
    val_games: int = 50_000
    min_ply: int = 10

    # --- Training core -------------------------------------------------
    total_steps: int | None = None
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0

    # LR schedule controls (see lr_schedule docstring for shape details).
    warmup_frac: float = 0.05
    # Explicit override for warmup_frac (in steps). When set, warmup
    # uses this exact step count regardless of total_steps.
    warmup_steps: int | None = None
    lr_schedule: LRScheduleName = "cosine"
    # Fraction of total_steps in the final decay phase (WSD / infinite).
    decay_frac: float = 0.1
    # Decay-phase curve (WSD's single decay; infinite's final decay).
    wsd_decay_shape: LRDecayShape = "linear"
    # Cooldown-phase fraction for the `infinite` schedule.
    cooldown_frac: float = 0.2
    # Stable-plateau LR as a fraction of peak LR (`infinite` only).
    stable_lr_ratio: float = 0.1

    max_grad_norm: float = 1.0
    patience: int | None = None
    eval_interval: int | None = None
    log_interval: int = 100
    pause_after_steps: int | None = None

    # --- Ablations -----------------------------------------------------
    mate_boost: float = 0.0
    # Prepend the game-outcome token at position 0 for outcome-conditioned
    # training. All 5 generation diagnostics in S8 gate on this flag.
    prepend_outcome: bool = False
    discard_ply_limit: bool = False

    # --- JAX-specific (new in v2) --------------------------------------
    # Which supernet config the run trains / slices from. ``"production"``
    # = `SUPERNET` (d=640, 10 layers, 10 heads); ``"tiny"`` =
    # `TINY_SUPERNET` (d=192, 4 layers, 3 heads) for verification runs.
    supernet: SupernetName = "production"
    # Sequence length at training time. Defaults to MAX_SEQ_LEN; smaller
    # values let tiny-supernet smoke runs fit on small GPUs.
    seq_len: int = 512
    # Inner-scan length: K steps per `lax.scan` body invocation. Higher
    # K amortises host overhead more aggressively at the cost of larger
    # compile time + memory. Validation cadence bounds K for adapter
    # runs (`K ≤ val_every` would otherwise drop val checkpoints).
    k: int = 50
    # Soft cap on the corpus's resident-memory size, in gigabytes. The
    # corpus + Lichess data path uses this to decide how many games to
    # tile into the GPU-resident batch buffer before falling back to
    # the streaming path.
    max_corpus_gb: float = 8.0

    # --- IO ------------------------------------------------------------
    log_dir: str | None = None
    hf_repo: str | None = None
    hf_bucket: str | None = None
    local_checkpoints: bool = False
    resume: str | None = None
    wandb: bool = False
    wandb_project: str = "pawn"
    cache_dir: str | None = None

    # -------------------------------------------------------------------
    # Cross-field validators
    # -------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_checkpoint_mode(self) -> "BaseRunConfig":
        """At least one of hf_repo / hf_bucket / local_checkpoints; hf_repo
        and local_checkpoints are mutually exclusive.

        Per v1's documented contract (`hf_bucket` is "Mutually compatible
        with hf_repo: the trainer pushes to both"), `hf_repo + hf_bucket`
        is an allowed combination, not an error. The hard mutex is only
        hf_repo vs local_checkpoints — local is fundamentally an
        alternative to pushing anywhere.
        """
        if self.hf_repo and self.local_checkpoints:
            raise ValueError(
                "hf_repo and local_checkpoints are mutually exclusive"
            )
        if (
            not self.hf_repo
            and not self.local_checkpoints
            and not self.hf_bucket
        ):
            raise ValueError(
                "one of hf_repo, hf_bucket, or local_checkpoints is required"
            )
        return self

    @model_validator(mode="after")
    def _check_lr_schedule_fractions(self) -> "BaseRunConfig":
        """The LR schedule's phase fractions must sum to ≤ 1.0.

        For ``infinite`` the total budget is
        ``warmup_frac + cooldown_frac + decay_frac + stable``, with
        stable ≥ 0 the remaining fraction; the three explicit fractions
        therefore can't exceed 1.0. For ``wsd`` only
        ``warmup_frac + decay_frac`` must fit. For ``cosine`` /
        ``constant`` / ``one_cycle`` only ``warmup_frac`` matters (must
        be in [0, 1]).
        """
        for name, val in (
            ("warmup_frac", self.warmup_frac),
            ("decay_frac", self.decay_frac),
            ("cooldown_frac", self.cooldown_frac),
        ):
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1], got {val}"
                )
        if self.lr_schedule == "wsd":
            total = self.warmup_frac + self.decay_frac
            if total > 1.0:
                raise ValueError(
                    f"wsd schedule requires warmup_frac + decay_frac ≤ 1, "
                    f"got {self.warmup_frac} + {self.decay_frac} = {total}"
                )
        elif self.lr_schedule == "infinite":
            total = self.warmup_frac + self.cooldown_frac + self.decay_frac
            if total > 1.0:
                raise ValueError(
                    f"infinite schedule requires warmup_frac + cooldown_frac "
                    f"+ decay_frac ≤ 1, got {self.warmup_frac} + "
                    f"{self.cooldown_frac} + {self.decay_frac} = {total}"
                )
            if not 0.0 < self.stable_lr_ratio <= 1.0:
                raise ValueError(
                    f"stable_lr_ratio must be in (0, 1], got {self.stable_lr_ratio}"
                )
        return self

    @model_validator(mode="after")
    def _check_positive_training_dims(self) -> "BaseRunConfig":
        """Reject zero / negative values on fields whose semantics make
        non-positive nonsensical (seq_len, k, batch_size, max_corpus_gb,
        total_steps, val_games, log_interval, weight_decay≥0)."""
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.max_corpus_gb <= 0:
            raise ValueError(
                f"max_corpus_gb must be positive, got {self.max_corpus_gb}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError(
                f"total_steps must be positive when set, got {self.total_steps}"
            )
        if self.val_games <= 0:
            raise ValueError(f"val_games must be positive, got {self.val_games}")
        if self.log_interval <= 0:
            raise ValueError(
                f"log_interval must be positive, got {self.log_interval}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        return self


class PretrainConfig(BaseRunConfig):
    """Pretrain the supernet on Rust-engine random games.

    The supernet's joint loss (the v1 cotrain replacement) is what
    pretraining produces. Variant selection (`small`/`base`/`large`)
    selects which slice of the supernet the smoke verification slices,
    not the training shape — the supernet is always trained at the
    full `supernet`-sized model.
    """

    run_type: Literal["pretrain"] = "pretrain"
    variant: VariantName = "base"

    # Pretrain-specific
    accumulation_steps: int = 1
    checkpoint_interval: int = 5000
    # If null at runtime, defaults to seq_len // 2. Late-ply positions
    # past this threshold get a stricter legality check.
    legality_late_ply: int | None = None
    # Override BaseRunConfig's 50K — pretrain uses on-the-fly random games,
    # so eval cost dominates and a smaller val set keeps cadence sane.
    val_games: int = 512

    @model_validator(mode="after")
    def _check_pretrain(self) -> "PretrainConfig":
        if self.accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps must be positive, got {self.accumulation_steps}"
            )
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {self.checkpoint_interval}"
            )
        return self


class AdapterConfig(BaseRunConfig):
    """Adapter finetuning on Elo-stratified Lichess data.

    One of 8 strategies dispatches via `--strategy`; each strategy has
    its own subset of the fields below. RoSA modes 1–3 are all
    in-scope (`rosa_mode` Literal trio); the v1 `mask_samples` /
    `grad_alpha` hyperparameters are preserved verbatim per plan §6.

    `unfreeze_layers` accepts the v1 explicit-pick comma-separated
    string form (e.g. ``"5,6,7"``); per plan §6 this is the v1 contract
    and must keep working, not a top-N count.
    """

    run_type: Literal["adapter"] = "adapter"
    # All 10 keys in `pawn.adapter_trainer.STRATEGIES`. The three RoSA
    # modes (`rosa`, `rosa-retro-sparse`, `rosa-retro-bottleneck`) are
    # distinct CLI strategies per CLAUDE.md's adapter table — they share
    # init/apply but the `rosa_mode` field selects the sub-mode.
    strategy: Literal[
        "bottleneck", "lora", "film", "sparse",
        "rosa", "rosa-retro-sparse", "rosa-retro-bottleneck",
        "hybrid", "specialized_clm", "unfreeze",
    ]
    # Which v1 checkpoint to adapt. Loaded through
    # `pawn.legacy.convert_legacy_checkpoint` when this points at the
    # published torch repos; loaded directly when this is already a JAX
    # checkpoint dir.
    checkpoint: str = "thomas-schweich/pawn-base"
    # Which supernet/variant slice the adapter targets when the
    # checkpoint is supernet-derived. For v1-published checkpoints this
    # is inferred from the checkpoint's own config.
    variant: VariantName = "base"
    # Lichess parquet source — HF repo or local directory.
    pgn: str = "thomas-schweich/pawn-lichess-full"
    # Optional: carve val from train (for sources without a held-out
    # split); default uses the dataset's `validation` split.
    pgn_val_split: str | None = "validation"

    # --- Placement -----------------------------------------------------
    adapter_layers: str | None = None

    # --- Bottleneck component ------------------------------------------
    bottleneck_dim: int | None = None
    # Extra hidden Linear+GELU stages inside each Houlsby adapter MLP.
    # 0 = standard two-layer block. Adds `n_hidden · bottleneck_dim²`
    # params per adapter.
    bottleneck_n_hidden: int = 0
    no_adapt_attn: bool = False
    no_adapt_ffn: bool = False

    # --- LoRA component ------------------------------------------------
    lora_rank: int | None = None
    lora_targets: Literal["qkvo", "qv", "qkv"] | None = None
    lora_ffn: bool = False

    # --- Sparse component ----------------------------------------------
    density: float | None = None
    sparse_targets: Literal["qkvo", "qv", "qkv"] | None = None
    sparse_ffn: bool = False

    # --- FiLM component ------------------------------------------------
    # Plan §10 S3 explicitly: default True for v2 (the v1 default was
    # False but the field name is preserved per §7's CLI compat).
    use_output_film: bool = True

    # --- RoSA ----------------------------------------------------------
    # Three modes per plan §6 "specifically and non-negotiably":
    # rosa (standard) + retro-sparse + retro-bottleneck.
    rosa_mode: Literal["rosa", "retro-sparse", "retro-bottleneck"] | None = None
    rosa_warmup_steps: int = 128
    mask_samples: int = 32
    grad_alpha: Literal[1, 2] = 2

    # --- Unfreeze / specialized_clm (within-adapter dispatch) ----------
    # v1 explicit-pick comma-separated form, e.g. "5,6,7". Resolved to
    # a list[int] at trainer entry by `pawn.adapter_trainer`.
    unfreeze_layers: str | None = None
    # specialized_clm-strategy architecture overrides (when
    # `strategy == "specialized_clm"`). The standalone v2 run lives in
    # `SpecializedCLMConfig` below; this set is here for parity with the
    # v1 in-AdapterConfig dispatch path.
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None

    # --- Adapter training cadence --------------------------------------
    epochs: int = 50
    val_every: int = 1
    checkpoint_interval: int = 5000

    # --- Data sizing ---------------------------------------------------
    # Canonical sizing knob (replaces v1 max_games at the adapter
    # boundary). `"all"` resolves to `n_train_games // batch_size` once
    # the cache materialises; the resolved integer is what gets written
    # to the saved run config.
    steps_per_epoch: int | Literal["all"] | None = None
    data_seed: int | None = None

    # --- Legality handling ---------------------------------------------
    disable_legal_mask: bool = False
    illegal_penalty: float = 0.0

    # -------------------------------------------------------------------
    # Cross-field validators
    # -------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_strategy_inputs(self) -> "AdapterConfig":
        """Strategy → required-field consistency.

        Each strategy has a small set of fields it must have set; this
        catches the common bug of `--strategy lora` without a
        `--lora-rank`.
        """
        # `hybrid` = LoRA + FiLM, so it also requires a LoRA rank.
        if self.strategy in ("lora", "hybrid"):
            if self.lora_rank is None:
                raise ValueError(
                    f"strategy={self.strategy} requires lora_rank"
                )
            if self.lora_rank <= 0:
                raise ValueError(
                    f"lora_rank must be positive, got {self.lora_rank}"
                )
        if self.strategy == "sparse":
            if self.density is None:
                raise ValueError("strategy=sparse requires density")
            if not 0.0 < self.density <= 1.0:
                raise ValueError(
                    f"density must be in (0, 1], got {self.density}"
                )
        if self.strategy == "bottleneck":
            if self.bottleneck_dim is None:
                raise ValueError("strategy=bottleneck requires bottleneck_dim")
            if self.bottleneck_dim <= 0:
                raise ValueError(
                    f"bottleneck_dim must be positive, got {self.bottleneck_dim}"
                )
            if self.bottleneck_n_hidden < 0:
                raise ValueError(
                    f"bottleneck_n_hidden must be >= 0, got {self.bottleneck_n_hidden}"
                )
        if self.strategy == "rosa" and self.rosa_mode is None:
            raise ValueError(
                "strategy=rosa requires rosa_mode (one of "
                "'rosa'|'retro-sparse'|'retro-bottleneck')"
            )
        if self.strategy == "unfreeze" and self.unfreeze_layers is None:
            raise ValueError(
                "strategy=unfreeze requires unfreeze_layers "
                "(comma-separated layer indices, e.g. '5,6,7')"
            )
        if self.strategy == "specialized_clm":
            if (
                self.d_model is None
                or self.n_layers is None
                or self.n_heads is None
                or self.d_ff is None
            ):
                raise ValueError(
                    "strategy=specialized_clm requires d_model, n_layers, "
                    "n_heads, d_ff"
                )
            # Mirror SpecializedCLMConfig._check_arch — head_dim must be an
            # integer, otherwise attention crashes at first forward.
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"strategy=specialized_clm requires d_model "
                    f"({self.d_model}) divisible by n_heads "
                    f"({self.n_heads}); otherwise head_dim is non-integer"
                )
        return self

    @model_validator(mode="after")
    def _check_adapter_cadence(self) -> "AdapterConfig":
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.val_every <= 0:
            raise ValueError(
                f"val_every must be positive, got {self.val_every}"
            )
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got "
                f"{self.checkpoint_interval}"
            )
        return self

    @model_validator(mode="after")
    def _check_unfreeze_layers_form(self) -> "AdapterConfig":
        """`unfreeze_layers` must be a comma-separated list of
        non-negative integers (v1 contract: e.g. `"5,6,7"`).

        Whitespace around commas is tolerated by the validator but
        normalised away in the stored value, so downstream
        `s.split(",")` doesn't trip on `"5, 6, 7"` → `[" 6", " 7"]`.
        """
        if self.unfreeze_layers is None:
            return self
        s = self.unfreeze_layers.strip()
        if not s:
            raise ValueError("unfreeze_layers must not be empty")
        parts = [p.strip() for p in s.split(",")]
        for p in parts:
            if not p or not p.isdigit():
                raise ValueError(
                    f"unfreeze_layers must be comma-separated non-negative "
                    f"ints (e.g. '5,6,7'), got {self.unfreeze_layers!r}"
                )
        # Normalise to no-whitespace form so downstream splitters work.
        normalised = ",".join(parts)
        if normalised != self.unfreeze_layers:
            object.__setattr__(self, "unfreeze_layers", normalised)
        return self

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
                "max_games is deprecated for adapter runs; pass "
                "steps_per_epoch (int or 'all') directly."
            )
        return self


class SpecializedCLMConfig(BaseRunConfig):
    """Train a from-scratch standalone CLM with no pretrained backbone.

    A small transformer with bare ``d_model`` / ``n_layers`` / ``n_heads``
    / ``d_ff`` (no ``specialized_`` prefix per plan §10 S3). Distinct
    from the adapter-dispatch ``--strategy specialized_clm`` path —
    this is the top-level config when you want to train a small CLM
    without any reference to a supernet or adapter machinery.
    """

    run_type: Literal["specialized_clm"] = "specialized_clm"

    # Required architecture fields (no preset to fall back on).
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int

    # Lichess data source — same shape as AdapterConfig.
    pgn: str = "thomas-schweich/pawn-lichess-full"
    pgn_val_split: str | None = "validation"

    # Cadence
    epochs: int = 50
    val_every: int = 1
    checkpoint_interval: int = 5000
    steps_per_epoch: int | Literal["all"] | None = None
    data_seed: int | None = None

    @model_validator(mode="after")
    def _check_arch(self) -> "SpecializedCLMConfig":
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}); otherwise attention head_dim is "
                f"non-integer and the model crashes at first forward"
            )
        return self

    @model_validator(mode="after")
    def _check_data_sizing(self) -> "SpecializedCLMConfig":
        """Same `steps_per_epoch` / `max_games` semantics as AdapterConfig
        — both drive the Lichess-cache sizing logic in the trainer."""
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
                "steps_per_epoch and max_games are mutually exclusive"
            )
        return self

    @model_validator(mode="after")
    def _check_cadence(self) -> "SpecializedCLMConfig":
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.val_every <= 0:
            raise ValueError(
                f"val_every must be positive, got {self.val_every}"
            )
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got "
                f"{self.checkpoint_interval}"
            )
        return self


# Discriminated union: pydantic dispatches by run_type at parse time.
# `pawn.lab` returns this via `lab_schema`; CLI drivers in S13 build
# the appropriate subclass and the trainer reads the right discriminator.
RunConfig = Annotated[
    Union[PretrainConfig, AdapterConfig, SpecializedCLMConfig],
    Field(discriminator="run_type"),
]
"""Discriminated union of all v2 JAX run-config types."""
