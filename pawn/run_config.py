"""Pydantic models for all PAWN training configurations.

``BaseRunConfig`` / ``PretrainConfig`` / ``AdapterConfig`` /
``SpecializedCLMConfig`` are the single source of truth for every
parameter that training scripts, sweep tooling, and the lab MCP server
care about. JSON Schema is derived automatically — call
``PretrainConfig.model_json_schema()`` /
``AdapterConfig.model_json_schema()`` for introspection (used by
``pawn.lab.lab_schema``).

This module is the load-bearing keystone that the rest of the JAX
migration sits on. Compared to v1 (``docs/jax-migration.md`` §8):

- ``CotrainConfig`` / ``CotrainVariant`` removed — the supernet (S3)
  is the trained model; "cotrain" is gone by design (§2).
- GPU / precision fields (``amp_dtype``, ``no_compile``, ``sdpa_math``,
  ``device``, ``num_workers``) removed — JAX manages these. The
  ``PAWN_ALLOW_CPU`` escape hatch is an env var, not a config field.
- v1 field names preserved (``lora_rank``, ``density``,
  ``use_output_film``, ``no_adapt_attn``, ``no_adapt_ffn``) per §8.3.
- ``lora_targets`` / ``sparse_targets`` widened from
  ``Literal[...]`` to ``list[str]`` (subsets of ``{q,k,v,o}``)
  per §8.4.
- ``rosa_mode`` / ``rosa_warmup_steps`` / ``mask_samples`` /
  ``grad_alpha`` removed; ``rosa_warmup_frac`` /
  ``rosa_top_k_frac`` added per §8.4.
- ``bucket_size`` / ``lora_ffn`` / ``sparse_ffn`` removed per §8.4.
- ``unfreeze_layers: str`` (explicit picks like "5,6,7") →
  ``n_unfreeze: int`` (top-N count) per §8.4. The regression is
  documented; restoring v1 flexibility is a follow-up.
- ``SpecializedCLMConfig`` is now a nested config (lives inside
  ``AdapterConfig`` when ``strategy == "specialized_clm"``); its
  ``d_model`` / ``n_layers`` / ``n_heads`` / ``d_ff`` keep canonical
  names (no ``specialized_`` prefix) per §8.3.
- Added v2 fields: ``supernet`` (which supernet shape — ``tiny`` vs.
  ``supernet``), JAX scan chunk size (``k``), corpus safety guard
  (``max_corpus_gb``), and separate ``corpus_seed`` / ``model_seed``
  RNG seeds.
- IO field ``log_dir`` renamed to ``logs_dir`` to match the deploy
  wrapper convention (the ``deploy/pod.sh`` / ``deploy/vast.sh``
  wrappers append ``--logs-dir`` automatically).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


_VALID_TARGETS = ("q", "k", "v", "o")


class BaseRunConfig(BaseModel):
    """Fields shared by pretraining and adapter runs."""

    # Fail fast when callers pass stale CLI/JSON fields (e.g. the old
    # ``legacy_vocab`` flag, the discarded v2 ``rank`` rename, or
    # misspelled hybrid args). Pydantic defaults to silently ignoring
    # unknown fields, which made it possible in v1 to burn a whole
    # training run on a config that did not mean what the user thought
    # it meant — the same lesson the JAX migration's first attempt
    # rediscovered when it replaced this module with scattered
    # ``raise SystemExit`` guards (see ``docs/jax-migration.md`` §8.1).
    model_config = ConfigDict(extra="forbid")

    # Training ------------------------------------------------------------
    total_steps: int | None = None
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_frac: float = 0.05
    warmup_steps: int | None = None  # explicit override of warmup_frac
    # LR schedule shape:
    #   ``"cosine"``    — warmup → cosine decay to 0.
    #   ``"wsd"``       — Warmup-Stable-Decay: hold peak LR through the
    #                     bulk of training and only decay over the
    #                     final ``decay_frac`` of steps.
    #   ``"constant"``  — linear warmup → hold peak LR indefinitely.
    #                     Pair with ``patience`` to stop on val plateau.
    #   ``"one_cycle"`` — Smith (2018) one-cycle: ramp up to peak over
    #                     ``warmup_frac`` of steps, then cosine-decay
    #                     to peak/10000 over the rest.
    #   ``"infinite"``  — warmup → cosine cooldown to
    #                     ``stable_lr_ratio * peak`` over
    #                     ``cooldown_frac`` of steps → flat stable
    #                     plateau → final decay to 0 over the last
    #                     ``decay_frac`` of steps. Any checkpoint in
    #                     the plateau is a valid resumption point.
    lr_schedule: Literal[
        "cosine", "wsd", "constant", "one_cycle", "infinite"
    ] = "cosine"
    decay_frac: float = 0.1
    wsd_decay_shape: Literal["linear", "cosine"] = "linear"
    cooldown_frac: float = 0.2
    stable_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int | None = None
    eval_interval: int | None = None
    log_interval: int = 100
    pause_after_steps: int | None = None

    # JAX-specific --------------------------------------------------------
    # Inner steps per ``lax.scan`` call — amortises JIT dispatch. ``K * B``
    # games are consumed per scan call; size ``k`` so per-step throughput
    # is host-bound rather than launch-bound. For adapter training, ``k``
    # additionally bounds the validation cadence (``k <= val_every``).
    k: int = 50

    # Seeds ---------------------------------------------------------------
    seed: int = 42
    # Falls back to ``seed`` if None — split so the Rust-engine corpus
    # generation and the JAX model parameter init can be re-seeded
    # independently for reproducibility experiments.
    corpus_seed: int | None = None
    model_seed: int | None = None

    # Ablations -----------------------------------------------------------
    mate_boost: float = 0.0
    # Prepend the outcome token at position 0 for outcome-conditioned
    # training. Controls the model's ``prepend_outcome`` runtime flag
    # AND the eval-time ``outcome_prefix_trained`` gate that
    # ``impossible_task_test`` / ``improbable_task_test`` /
    # ``outcome_signal_test`` / ``prefix_continuation_test`` /
    # ``poisoned_prefix_test`` all condition on (§5.1).
    prepend_outcome: bool = False
    discard_ply_limit: bool = False

    # IO ------------------------------------------------------------------
    # Renamed from v1 ``log_dir`` to match the deploy wrappers
    # (``deploy/pod.sh launch`` and ``deploy/vast.sh launch`` append
    # ``--logs-dir`` to every launch command, so the trainers must
    # accept ``--logs-dir`` not ``--log-dir``).
    logs_dir: str | None = None
    hf_repo: str | None = None
    # Optional bucket-path autosave alongside (or instead of) the
    # model-repo branch push. Accepts ``<namespace>/<bucket-name>`` or
    # a full ``hf://buckets/<ns>/<name>[/subpath]`` URL. Files land at
    # ``<bucket>/logs/<run_slug>/{checkpoints/...,metrics.jsonl}``.
    # Mutually compatible with ``hf_repo``: the trainer pushes to both.
    hf_bucket: str | None = None
    local_checkpoints: bool = False
    resume: str | None = None
    wandb: bool = False
    wandb_project: str = "pawn"
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _check_checkpoint_mode(self) -> "BaseRunConfig":
        if self.hf_repo and self.local_checkpoints:
            raise ValueError(
                "--hf-repo and --local-checkpoints are mutually exclusive"
            )
        if (
            not self.hf_repo
            and not self.local_checkpoints
            and not self.hf_bucket
        ):
            raise ValueError(
                "One of --hf-repo, --hf-bucket, or --local-checkpoints "
                "is required"
            )
        return self

    @model_validator(mode="after")
    def _check_k_positive(self) -> "BaseRunConfig":
        if self.k <= 0:
            raise ValueError(
                f"k (lax.scan inner chunk size) must be > 0, got {self.k}"
            )
        return self

    @model_validator(mode="after")
    def _check_k_total_steps(self) -> "BaseRunConfig":
        # The JAX scan-chunk loop assumes ``total_steps`` divides
        # cleanly into ``k`` so the last chunk isn't padded. Padded
        # final chunks introduce the well-known AdamW-weight-decay
        # drift the trainer has a ``lax.cond`` guard against — but
        # cleaner to require divisibility up-front.
        if self.total_steps is not None and self.total_steps % self.k != 0:
            raise ValueError(
                f"total_steps ({self.total_steps}) must be a "
                f"multiple of k ({self.k}); the lax.scan "
                f"chunk loop assumes no padded final chunk"
            )
        return self

    @model_validator(mode="after")
    def _check_lr_schedule_fractions(self) -> "BaseRunConfig":
        # warmup_frac / decay_frac / cooldown_frac / stable_lr_ratio
        # are all fractions in [0, 1] — out-of-range values produce
        # silently-broken schedules (e.g. warmup_frac=2.0 makes the
        # ramp never finish; decay_frac=1.5 makes the WSD plateau go
        # negative). Validate up-front instead of letting the bug
        # surface mid-run.
        for name, value in (
            ("warmup_frac", self.warmup_frac),
            ("decay_frac", self.decay_frac),
            ("cooldown_frac", self.cooldown_frac),
            ("stable_lr_ratio", self.stable_lr_ratio),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"{name} must be in [0, 1], got {value}"
                )
        # WSD: warmup + decay must fit inside [0, 1] so the stable
        # plateau gets non-negative length.
        if self.lr_schedule == "wsd":
            if self.warmup_frac + self.decay_frac > 1.0:
                raise ValueError(
                    f"WSD schedule needs warmup_frac + decay_frac "
                    f"<= 1, got {self.warmup_frac} + "
                    f"{self.decay_frac} = {self.warmup_frac + self.decay_frac}"
                )
        # Infinite: warmup + cooldown + decay must fit so the
        # stable plateau gets non-negative length.
        if self.lr_schedule == "infinite":
            total = self.warmup_frac + self.cooldown_frac + self.decay_frac
            if total > 1.0:
                raise ValueError(
                    f"infinite schedule needs warmup_frac + "
                    f"cooldown_frac + decay_frac <= 1, got "
                    f"{self.warmup_frac} + {self.cooldown_frac} "
                    f"+ {self.decay_frac} = {total}"
                )
        return self


class SpecializedCLMConfig(BaseModel):
    """Architecture for a from-scratch CLM, used by
    ``AdapterConfig`` when ``strategy == "specialized_clm"``.

    Field names follow plan §8.3: NO ``specialized_`` prefix. The
    prefix isn't needed inside a per-strategy config — ``d_model`` is
    already unambiguous in the strategy's scope.
    """

    model_config = ConfigDict(extra="forbid")

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int

    @model_validator(mode="after")
    def _check_divisibility(self) -> "SpecializedCLMConfig":
        # Positivity FIRST — otherwise ``d_model % n_heads`` with
        # ``n_heads=0`` raises ZeroDivisionError instead of the
        # intended pydantic ValidationError.
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError(
                f"d_model and n_heads must be positive, got "
                f"d_model={self.d_model}, n_heads={self.n_heads}"
            )
        if self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError(
                f"n_layers and d_ff must be positive, got "
                f"n_layers={self.n_layers}, d_ff={self.d_ff}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads}); otherwise attention "
                f"head_dim is non-integer"
            )
        return self


class PretrainConfig(BaseRunConfig):
    """Pretraining the JAX supernet on random games."""

    run_type: Literal["pretrain"] = "pretrain"

    # Which supernet shape — ``"tiny"`` for verification (
    # ``TINY_SUPERNET`` / ``TINY_VARIANTS`` in ``pawn.config``) or
    # ``"supernet"`` for the production ``SUPERNET`` / ``VARIANTS``.
    # The supernet trains all three nested variants jointly; there is
    # no per-variant choice at pretrain time (variants are sliced out
    # at publish time — see ``docs/jax-migration.md`` §5.5).
    supernet: Literal["tiny", "supernet"] = "supernet"

    # Per-variant loss weighting in the supernet joint loss (§5.3).
    # ``None`` → equal weight (1.0 for each variant in the supernet).
    # Example: ``{"small": 1.0, "base": 1.0, "large": 2.0}`` weights
    # large 2x in the summed loss.
    variant_loss_weights: dict[str, float] | None = None

    # Pretrain-specific ---------------------------------------------------
    checkpoint_interval: int = 5000
    max_seq_len: int = 512
    # Explicit override of ``max_seq_len`` for the actual trained
    # sequence length (the model is built at ``max_seq_len`` but the
    # train-time RoPE positions only span ``seq_len``). ``None`` →
    # use ``max_seq_len``.
    seq_len: int | None = None
    # Abort upfront if the estimated Rust corpus footprint exceeds this
    # bound — the production ``SUPERNET`` × 100K steps × B=256 × T=512
    # is ~122 GiB on disk, so the 64 GiB default catches misconfigured
    # runs before they fill the volume.
    max_corpus_gb: float = 64.0
    val_games: int = 512

    @model_validator(mode="after")
    def _check_variant_loss_weights(self) -> "PretrainConfig":
        if self.variant_loss_weights is None:
            return self
        for name, weight in self.variant_loss_weights.items():
            if name not in ("small", "base", "large"):
                raise ValueError(
                    f"variant_loss_weights key {name!r} must be one of "
                    f"'small' / 'base' / 'large'"
                )
            if weight < 0:
                raise ValueError(
                    f"variant_loss_weights[{name!r}] must be >= 0, "
                    f"got {weight}"
                )
        return self


class AdapterConfig(BaseRunConfig):
    """Adapter finetuning on Lichess data, over a sliced supernet
    variant or a converted legacy checkpoint."""

    run_type: Literal["adapter"] = "adapter"
    strategy: Literal[
        "lora",
        "film",
        "unfreeze",
        "bottleneck",
        "hybrid",
        "sparse",
        "rosa",
        "specialized_clm",
    ]

    # Data ----------------------------------------------------------------
    elo_min: int | None = None
    elo_max: int | None = None
    max_games: int | None = None
    min_ply: int = 10

    # Backbone selection (ignored for ``specialized_clm`` — that one
    # trains from scratch). Pass EITHER ``--checkpoint <path>`` for a
    # converted legacy checkpoint or ``--supernet <tiny|supernet>`` +
    # ``--variant <small|base|large>`` for a slice of a supernet
    # produced by ``scripts/train_jax.py``.
    supernet: Literal["tiny", "supernet"] | None = None
    variant: Literal["small", "base", "large", "custom"] | None = None
    checkpoint: str | None = None
    pgn: str = "thomas-schweich/pawn-lichess-full"

    # Placement -----------------------------------------------------------
    # Comma-separated layer indices ("0,2,4") restricting which
    # transformer layers carry the adapter. ``None`` → every layer.
    adapter_layers: str | None = None

    # Bottleneck (Houlsby adapters) ---------------------------------------
    bottleneck_dim: int | None = None
    # Extra hidden Linear+GELU stages between ``down`` and ``up``
    # inside each adapter MLP. ``0`` reproduces the standard two-layer
    # block. Adds ``n_hidden * bottleneck_dim^2`` params per adapter.
    bottleneck_n_hidden: int = 0
    no_adapt_attn: bool = False
    no_adapt_ffn: bool = False

    # LoRA / Hybrid / RoSA shared shape -----------------------------------
    # ``lora_rank`` (not ``rank``) per §8.3 — kept v1's canonical name
    # because ``rank`` would be ambiguous now that LoRA / Hybrid / RoSA
    # all share it.
    lora_rank: int | None = None
    lora_alpha: float | None = None
    # ``list[str]`` per §8.4 — v1's ``Literal["qkvo","qv","qkv"]`` is
    # widened to any subset of ``{q,k,v,o}``.
    lora_targets: list[str] | None = None

    # Sparse component ----------------------------------------------------
    # ``density`` (not ``sparse_density``) per §8.3 — the v2 rename was
    # gratuitous; ``density`` was already sparse-only in v1.
    density: float | None = None
    sparse_targets: list[str] | None = None
    sparse_hard: bool = False

    # FiLM component ------------------------------------------------------
    # Default-true and not polarity-flipped per §8.3 — v2's
    # ``film_output`` with ``--no-film-output`` was a gratuitous
    # polarity flip from v1's ``use_output_film=True``.
    use_output_film: bool = True

    # RoSA ----------------------------------------------------------------
    rosa_targets: list[str] | None = None
    # Fraction of ``total_steps`` for Phase 1 (LoRA warmup). v1's
    # ``rosa_warmup_steps: int`` was rescaled to a fraction per §8.4 so
    # it scales with ``total_steps``.
    rosa_warmup_frac: float = 0.4
    # Fraction of weights to keep in the Phase-2 gradient-magnitude
    # mask. v1's ``mask_samples`` / ``grad_alpha`` are gone per §8.4 —
    # the v2 RoSA mask-gen algorithm is single forward+backward with
    # an all-True mask.
    rosa_top_k_frac: float = 0.01

    # Unfreeze ------------------------------------------------------------
    # ``n_unfreeze: int`` (top-N count) replaces v1's
    # ``unfreeze_layers: str`` (explicit picks like "5,6,7"). The
    # regression to top-N-only is documented in §8.4; restoring v1
    # flexibility is a future-work item.
    n_unfreeze: int | None = None
    include_lm_head: bool = False
    include_embeddings: bool = False

    # SpecializedCLM (nested config) --------------------------------------
    specialized_clm: SpecializedCLMConfig | None = None

    # Adapter training specific -------------------------------------------
    # Validate every N scan-chunks. ``k * val_every`` is the absolute
    # step interval between validation runs.
    val_every: int = 100
    val_frac: float = 0.05
    checkpoint_interval: int = 5000

    # Legality handling ---------------------------------------------------
    # By default, adapter training masks illegal moves to -inf before
    # cross-entropy. Set ``disable_legal_mask=True`` to compute the
    # loss over the full vocabulary — matching pretraining, where the
    # model has to learn legality from scratch.
    disable_legal_mask: bool = False
    # Additional penalty (lambda) on probability mass assigned to
    # illegal moves. Added to the loss as ``illegal_penalty *
    # mean_illegal_prob_mass``. Only meaningful when
    # ``disable_legal_mask=True``.
    illegal_penalty: float = 0.0

    @model_validator(mode="after")
    def _check_strategy_args(self) -> "AdapterConfig":
        # Per-strategy required-arg checks
        if self.strategy in ("lora", "hybrid", "rosa"):
            if self.lora_rank is None:
                raise ValueError(
                    f"{self.strategy!r} requires --lora-rank"
                )
            if self.lora_rank <= 0:
                raise ValueError(
                    f"lora_rank must be > 0, got {self.lora_rank}"
                )
        if self.strategy in ("bottleneck", "hybrid"):
            if self.bottleneck_dim is None:
                raise ValueError(
                    f"{self.strategy!r} requires --bottleneck-dim"
                )
            if self.bottleneck_dim <= 0:
                raise ValueError(
                    f"bottleneck_dim must be > 0, got "
                    f"{self.bottleneck_dim}"
                )
        if self.strategy == "sparse":
            if self.density is None:
                raise ValueError("sparse requires --density")
            if not (0 < self.density <= 1):
                raise ValueError(
                    f"density must be in (0, 1], got {self.density}"
                )
        if self.strategy == "rosa":
            if not (0 < self.rosa_top_k_frac < 1):
                raise ValueError(
                    f"rosa_top_k_frac must be in (0, 1), got "
                    f"{self.rosa_top_k_frac}"
                )
            if not (0 < self.rosa_warmup_frac < 1):
                raise ValueError(
                    f"rosa_warmup_frac must be in (0, 1), got "
                    f"{self.rosa_warmup_frac}"
                )
        if self.strategy == "unfreeze":
            if self.n_unfreeze is None or self.n_unfreeze <= 0:
                raise ValueError(
                    "unfreeze requires --n-unfreeze > 0"
                )
        if self.strategy == "specialized_clm":
            if self.specialized_clm is None:
                raise ValueError(
                    "specialized_clm requires the nested "
                    "specialized_clm config (d_model / n_layers / "
                    "n_heads / d_ff)"
                )

        # Bottleneck no-op guard from v1 (§8.3 kept this invariant)
        if self.strategy in ("bottleneck", "hybrid"):
            if self.no_adapt_attn and self.no_adapt_ffn:
                raise ValueError(
                    f"--no-adapt-attn AND --no-adapt-ffn makes "
                    f"{self.strategy!r} a no-op (nothing left to "
                    f"adapt). Set at most one."
                )

        # LoRA / Sparse / RoSA targets must be subsets of {q,k,v,o}
        for fname, fval in (
            ("lora_targets", self.lora_targets),
            ("sparse_targets", self.sparse_targets),
            ("rosa_targets", self.rosa_targets),
        ):
            if fval is None:
                continue
            if not fval:
                raise ValueError(
                    f"{fname} must contain at least one of "
                    f"{{q,k,v,o}}"
                )
            bad = [t for t in fval if t not in _VALID_TARGETS]
            if bad:
                raise ValueError(
                    f"{fname} must be a subset of "
                    f"{{q,k,v,o}}, got invalid entries {bad}"
                )
            if len(set(fval)) != len(fval):
                raise ValueError(
                    f"{fname} contains duplicates: {fval}"
                )
        return self

    @model_validator(mode="after")
    def _check_backbone(self) -> "AdapterConfig":
        if self.strategy == "specialized_clm":
            if (
                self.checkpoint is not None
                or self.supernet is not None
                or self.variant is not None
            ):
                raise ValueError(
                    "specialized_clm trains from scratch — "
                    "checkpoint / supernet / variant should not be "
                    "set"
                )
        else:
            has_explicit_ckpt = self.checkpoint is not None
            has_supernet_variant = (
                self.supernet is not None and self.variant is not None
            )
            if not has_explicit_ckpt and not has_supernet_variant:
                raise ValueError(
                    f"{self.strategy!r} needs a backbone: pass "
                    f"--checkpoint <path>, OR --supernet "
                    f"+ --variant"
                )
            if has_explicit_ckpt and has_supernet_variant:
                raise ValueError(
                    "Pass EITHER --checkpoint <path> OR --supernet "
                    "+ --variant (not both — they'd disagree)"
                )
        return self

    @model_validator(mode="after")
    def _check_legality_flags(self) -> "AdapterConfig":
        if self.illegal_penalty < 0:
            raise ValueError(
                f"illegal_penalty must be >= 0, got "
                f"{self.illegal_penalty}"
            )
        if self.illegal_penalty > 0 and not self.disable_legal_mask:
            raise ValueError(
                "illegal_penalty > 0 has no effect while legal "
                "masking is active (all illegal logits are -inf, so "
                "illegal prob mass is zero by construction). Pass "
                "disable_legal_mask=True to let the model assign "
                "probability to illegal moves and be penalised for it."
            )
        return self

    @model_validator(mode="after")
    def _check_val_every(self) -> "AdapterConfig":
        if self.k > self.val_every:
            raise ValueError(
                f"k ({self.k}) must be <= val_every "
                f"({self.val_every}); otherwise the lax.scan "
                f"chunk overshoots a validation point"
            )
        return self


RunConfig = Annotated[
    Union[PretrainConfig, AdapterConfig],
    Field(discriminator="run_type"),
]
"""Discriminated union of all run config types — drop into
``TypeAdapter(RunConfig).validate_python(json_dict)`` to dispatch on
``run_type`` automatically."""


__all__ = [
    "BaseRunConfig",
    "PretrainConfig",
    "AdapterConfig",
    "SpecializedCLMConfig",
    "RunConfig",
]
