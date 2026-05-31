"""Pydantic models for every PAWN v2 training configuration.

:class:`RunConfig` is the discriminated-union root that every JAX-side
training driver, sweep tool, and the lab MCP server reads through. The
field set matches v1 ``pawn/run_config.py`` verbatim (per plan §7's
backward-compatibility contract — the names users type and the schemas
they grep are durable) plus a handful of v2-specific additions
(``supernet``, ``variant``, ``k``, ``max_corpus_gb``, ``seq_len``).
Torch-only fields (``device``, ``num_workers``, ``no_compile``,
``sdpa_math``) are dropped — JAX manages its own backend. The
``amp_dtype`` field is *retained* — v2 honours bf16 mixed-precision
forward compute per plan §5 ("Forward casts parameters to bf16 with
fp32 accumulation; the master copy and Adam moments stay fp32").

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
- ``"distill"`` → :class:`DistillConfig` — distil a frozen teacher
  checkpoint into a from-scratch student (plan §7/§7.1, logit-only KL/CE).

Every model sets ``extra="forbid"`` so a stale field (renamed flag,
misspelled JSON key) raises at config-load time rather than silently
training the wrong thing.

JSON Schema is derived automatically — ``PretrainConfig.model_json_schema()``
/ ``AdapterConfig.model_json_schema()`` / ``SpecializedCLMConfig.model_json_schema()``
are the public surfaces the lab MCP server's ``lab_schema`` returns.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pawn.config import CONDITIONING_KINDS, MAX_SEQ_LEN

__all__ = [
    "BaseRunConfig",
    "PretrainConfig",
    "AdapterConfig",
    "SpecializedCLMConfig",
    "DistillConfig",
    "RunConfig",
]


# Helpful aliases so per-config Literal[...] declarations stay readable.
SupernetName = Literal["tiny", "production"]
VariantName = Literal["small", "base", "large"]
LRScheduleName = Literal["cosine", "wsd", "constant", "one_cycle", "infinite"]
LRDecayShape = Literal["linear", "cosine"]
# C.1: optimizer choice. "adamw" is the default. Lion (sign-based, no
# second moment) halves the optimizer state and ~3-5% step time, but
# needs LR ~3× lower than AdamW. Adafactor is not wired in this build:
# its tree-map signature collides with the eqx pytree's int-typed
# decomp_table leaf under JAX 0.10's strictened None-vs-leaf check;
# adding it requires an `eqx.filter`-aware wrapper.
OptimizerName = Literal["adamw", "lion"]
# B1: distillation objective. ``kl`` matches the teacher's soft targets only,
# ``ce`` is ground-truth cross-entropy only, ``mix`` interpolates
# ``alpha·ce + (1-alpha)·kl``.
DistillObjective = Literal["kl", "ce", "mix"]


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
    # C.1: optimizer choice — "adamw" (default), "lion", or "adafactor".
    # See `OptimizerName` for the tradeoff summary. Switching requires
    # an LR retune (Lion typically wants LR/3 of AdamW's value).
    optimizer: OptimizerName = "adamw"

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
    # Ordered control-token kinds prepended to every sequence as the
    # conditioning prefix. The sequence is assembled as
    # ``[BOS][cond…][ply…][PAD…]``; each entry resolves to one control
    # token per game (see :data:`pawn.config.CONDITIONING_KINDS` for the
    # valid kinds and :mod:`pawn.corpus` for the resolution). The prefix
    # width is ``C = 1 + len(conditioning)`` (BOS is always present, so
    # ``C >= 1`` even with no conditioning). The 5 generation diagnostics
    # in S8 condition on whatever kinds appear here (``["outcome"]`` is
    # the outcome-conditioned setup they were written for).
    #
    # Replaces the v1 ``prepend_outcome: bool`` flag —
    # ``prepend_outcome=True`` ⇒ ``conditioning=["outcome"]``. The legacy
    # JSON key is migrated with a deprecation warning by
    # ``_migrate_prepend_outcome`` (mode="before") so old configs still
    # load under ``extra="forbid"``.
    conditioning: list[str] = Field(default_factory=list)
    discard_ply_limit: bool = False

    # --- Mixed precision (v1-parity field name) -------------------------
    # bf16/fp16 forward compute with fp32 master weights + Adam moments
    # (plan §5). The model forward casts activations to this dtype at
    # `_run_layers` entry; `_rmsnorm` and `softmax` upcast to fp32
    # internally and downcast back, matching standard "weights in fp32,
    # compute in bf16" recipes. Defaults to bf16 because v1 defaulted
    # to bf16 (and the perf gap surfaced in the post-r4 review was due
    # to this contract not being wired up).
    amp_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # --- SDPA fast-path (parity #43) -----------------------------------
    # Opt into `jax.nn.dot_product_attention` (XLA implementation) for
    # the attention block. The plan §5 marked SDPA out of scope citing
    # fused-kernel maturity on JAX-on-ROCm; in practice the XLA impl
    # works fine for inference shapes but OOMs at training shapes on
    # RDNA 3 hardware (64 KB shared memory < the 128 KB the fused
    # kernel requests at B≥2 T=512). Off by default — superseded by
    # ``use_flash`` (Pallas) on GPU, which is faster than the XLA SDPA
    # on every shape we've measured. See `tests/test_jax_model.py
    # ::test_use_sdpa_matches_plain_attention_within_fp32_noise` for
    # the correctness guard.
    use_sdpa: bool = False
    # --- Pallas flash attention ---------------------------------------
    # Route the attention block through
    # `jax.experimental.pallas.ops.gpu.attention.mha` — a Triton-flavoured
    # fused attention kernel that ships with JAX. Measured ~6× faster
    # than the plain materialised path at BASE T=512 on RDNA3 (gfx1100)
    # and beats the XLA SDPA path. **On by default**: training runs
    # pick it up automatically. CPU runs (`PAWN_ALLOW_CPU=1` smoke
    # tests) auto-fall-back to the plain path at script startup so the
    # config default doesn't need to be flipped. `use_flash` wins over
    # `use_sdpa` when both are set.
    use_flash: bool = True

    # --- Supernet variant sampling -------------------------------------
    # Sandwich-sample one non-supernet variant per step instead of the
    # exhaustive sum-over-all loop. The supernet (``is_supernet=True``)
    # variant always runs; one of the remaining variants is drawn
    # uniformly and its CE is scaled by ``N`` to keep the per-step loss
    # an unbiased estimator of the full sum. Empirical 5090 measurement
    # at SUPERNET (d=640, d_ff=1792, n_layers=10), B=64, T=512, bf16,
    # 3-variant supernet loss surface: 165.49 ms (deterministic) →
    # 149.06 ms (stochastic), i.e. ~10% throughput improvement. MatFormer
    # / matryoshka-supernet literature shows quality convergence to the
    # full sum at the cost of a small variance increase in the
    # small-variant gradient signal. **On by default** (H.1 housekeeping):
    # supernet pretraining is the production loss surface and stochastic
    # sampling is its design intent. Tests that require the deterministic
    # exhaustive sum opt out explicitly via ``stochastic_variants=False``.
    stochastic_variants: bool = True

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
    # v1 carried an `hf_bucket` autosave target ("the trainer pushes to
    # both" — files at `<bucket>/logs/<run_slug>/...`). v2's JAX trainer
    # never wired the bucket-push primitive (there is no `submit_bucket`
    # equivalent in `pawn/lifecycle.py`), so a bucket-only run would
    # validate and then save/push nothing — silent total checkpoint loss
    # on a long run. Until the bucket-push path is built (tracked as a
    # blocker in docs/V2_PARITY_AUDIT.md), `_check_checkpoint_mode`
    # rejects any config that names `hf_bucket` rather than letting it
    # masquerade as a working destination.
    hf_bucket: str | None = None
    local_checkpoints: bool = False
    resume: str | None = None
    wandb: bool = False
    wandb_project: str = "pawn"
    cache_dir: str | None = None

    @property
    def C(self) -> int:
        """Conditioning prefix width ``C = 1 + len(conditioning)``.

        Move position ``i`` (0-indexed) lives at sequence slot ``C + i``;
        the loss is supervised on slots ``[C-1 .. C-1 + game_length - 1]``.
        BOS is always present, so ``C >= 1``.
        """
        return 1 + len(self.conditioning)

    # -------------------------------------------------------------------
    # Cross-field validators
    # -------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_prepend_outcome(cls, data: Any) -> Any:
        """Migrate the v1 ``prepend_outcome: bool`` key to ``conditioning``.

        ``prepend_outcome=True`` ⇒ ``conditioning=["outcome"]``;
        ``prepend_outcome=False`` ⇒ ``conditioning=[]``. Emit a
        ``DeprecationWarning`` so users update their JSON configs. Without
        this, ``extra="forbid"`` would reject the legacy key with an
        opaque "extra field" error instead of a pointer to the new one.
        """
        if not isinstance(data, dict) or "prepend_outcome" not in data:
            return data
        legacy = data.pop("prepend_outcome")
        if "conditioning" in data:
            raise ValueError(
                "pass either the legacy `prepend_outcome` flag or the new "
                "`conditioning` list, not both"
            )
        warnings.warn(
            "`prepend_outcome` is deprecated; use "
            "`conditioning=[\"outcome\"]` (True) or `conditioning=[]` "
            "(False) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        data["conditioning"] = ["outcome"] if legacy else []
        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_max_seq_len(cls, data: Any) -> Any:
        """Migrate the v1 ``max_seq_len`` JSON key to v2's ``seq_len``.

        v2 renamed the training-time context-window field from
        ``max_seq_len`` to ``seq_len`` (``max_seq_len`` is now reserved
        for the architectural ceiling :data:`pawn.config.MAX_SEQ_LEN`).
        A verbatim v1 run-config JSON carrying ``max_seq_len`` would
        otherwise be rejected by ``extra="forbid"`` with an opaque "extra
        field" error instead of a pointer to the rename. Emit a
        ``DeprecationWarning`` and fold the value into ``seq_len``.
        """
        if not isinstance(data, dict) or "max_seq_len" not in data:
            return data
        legacy = data.pop("max_seq_len")
        if "seq_len" in data:
            raise ValueError(
                "pass either the legacy `max_seq_len` key or the new "
                "`seq_len`, not both"
            )
        warnings.warn(
            "`max_seq_len` is deprecated as a run-config field; use "
            "`seq_len` instead (max_seq_len now names the architectural "
            "ceiling pawn.config.MAX_SEQ_LEN).",
            DeprecationWarning,
            stacklevel=2,
        )
        data["seq_len"] = legacy
        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_amp_dtype_none(cls, data: Any) -> Any:
        """Migrate the v1 ``amp_dtype: "none"`` value to ``"float32"``.

        v1's ``amp_dtype`` Literal admitted ``"none"`` to mean "no mixed
        precision — run the forward in fp32". v2 spells that
        ``"float32"`` (and drops the ``use_amp`` bool that v1 derived from
        ``amp_dtype != "none"``). A verbatim v1 JSON config with
        ``"amp_dtype": "none"`` would otherwise fail the v2 Literal with an
        opaque enum error. Rewrite it to ``"float32"`` with a
        ``DeprecationWarning`` so old configs still load.
        """
        if not isinstance(data, dict) or data.get("amp_dtype") != "none":
            return data
        warnings.warn(
            "`amp_dtype=\"none\"` is deprecated; use `amp_dtype=\"float32\"` "
            "(v2 has no separate `use_amp` flag — float32 is the no-mixed-"
            "precision setting).",
            DeprecationWarning,
            stacklevel=2,
        )
        data = dict(data)
        data["amp_dtype"] = "float32"
        return data

    @model_validator(mode="after")
    def _check_conditioning(self) -> "BaseRunConfig":
        """Every conditioning kind must be registered, and no kind may
        repeat (a duplicate slot would silently double-condition)."""
        seen: set[str] = set()
        for kind in self.conditioning:
            if kind not in CONDITIONING_KINDS:
                raise ValueError(
                    f"unknown conditioning kind {kind!r}; valid kinds are "
                    f"{sorted(CONDITIONING_KINDS)}"
                )
            if kind in seen:
                raise ValueError(
                    f"duplicate conditioning kind {kind!r}; each kind may "
                    f"appear at most once"
                )
            seen.add(kind)
        return self

    @model_validator(mode="after")
    def _check_seq_len_budget(self) -> "BaseRunConfig":
        """``seq_len`` must fit the conditioning prefix AND at least one
        move slot inside ``max_seq_len``.

        Net-new in Chunk 4: previously only a runtime ``T > max_seq_len``
        check existed in ``pawn.model``. The prefix consumes ``C`` slots
        (even when every conditioning value resolves to NULL), so the
        budget is ``C < seq_len <= MAX_SEQ_LEN``.
        """
        if self.seq_len > MAX_SEQ_LEN:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be <= MAX_SEQ_LEN "
                f"({MAX_SEQ_LEN})"
            )
        if self.seq_len <= self.C:
            raise ValueError(
                f"seq_len ({self.seq_len}) must exceed the conditioning "
                f"prefix width C={self.C} (= 1 BOS + {self.C - 1} "
                f"conditioning slot(s)) so at least one move slot remains"
            )
        return self

    @model_validator(mode="after")
    def _check_checkpoint_mode(self) -> "BaseRunConfig":
        """Exactly one of hf_repo / local_checkpoints; hf_bucket is not
        yet wired in v2 and is rejected up front.

        v1 documented `hf_bucket` as a functional autosave target
        ("Mutually compatible with hf_repo: the trainer pushes to both",
        files at `<bucket>/logs/<run_slug>/...`). v2's JAX trainer has no
        bucket-push primitive (see the field comment above and
        docs/V2_PARITY_AUDIT.md), so accepting `hf_bucket` would let a
        bucket-only run validate and then save/push nothing — silent
        total checkpoint loss. Reject it with an actionable error instead
        of advertising a destination that drops every checkpoint.
        """
        if self.hf_repo and self.local_checkpoints:
            raise ValueError(
                "hf_repo and local_checkpoints are mutually exclusive"
            )
        if self.hf_bucket is not None:
            raise ValueError(
                "hf_bucket autosave is not implemented in v2 (the JAX "
                "trainer has no bucket-push path). Use --hf-repo for "
                "durable pushes or --local-checkpoints for local-only "
                "saves. Tracking: docs/V2_PARITY_AUDIT.md."
            )
        if not self.hf_repo and not self.local_checkpoints:
            raise ValueError(
                "one of hf_repo or local_checkpoints is required"
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
        # H10: the global-norm clip threshold is threaded into
        # `make_optimizer`'s branchless clip as `max_norm / max(g_norm,
        # max_norm)`. A non-positive threshold makes that scale 0 (or
        # negative), silently zeroing or sign-flipping every gradient, so
        # reject it at parse time rather than producing a no-learning run.
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )
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
    # Held-out validation cadence: run the validation pass + emit
    # ``type=val`` records every ``val_every`` steps. ``None`` (default)
    # disables the held-out eval (no val records, no patience signal) —
    # the cheap-loss-curve-only mode. The flag name matches the v1 adapter
    # cadence knob; v1 pretrain spelled it ``eval_interval`` (still
    # accepted via ``--config`` JSON and mapped to ``val_every`` when
    # ``val_every`` is unset).
    val_every: int | None = None
    # If null at runtime, defaults to seq_len // 2. Late-ply positions
    # past this threshold get a stricter legality check.
    legality_late_ply: int | None = None
    # Override BaseRunConfig's 50K — pretrain uses on-the-fly random games,
    # so eval cost dominates and a smaller val set keeps cadence sane.
    val_games: int = 512

    # Which variants to train jointly. None (default) = all three
    # (small/base/large) — the supernet joint loss. A subset trains only
    # those variants; ("large",) is a standalone-large teacher pretrain for
    # the distillation-canonical ladder (plan §7). Distinct from `variant`
    # above, which only selects the smoke-verification slice.
    variants: tuple[VariantName, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_eval_interval_to_val_every(cls, data: Any) -> Any:
        """v1 pretrain spelled the validation cadence ``eval_interval``;
        v2 PretrainConfig spells it ``val_every`` (parity with the adapter
        cadence knob). Map a verbatim v1 ``eval_interval`` onto
        ``val_every`` when the latter is unset so old configs still drive
        the held-out eval. ``eval_interval`` remains a valid BaseRunConfig
        field, so we copy (not pop) it — both stay readable.
        """
        if not isinstance(data, dict):
            return data
        if data.get("val_every") is None and data.get("eval_interval") is not None:
            data = dict(data)
            data["val_every"] = data["eval_interval"]
        return data

    @model_validator(mode="after")
    def _check_pretrain(self) -> "PretrainConfig":
        # `BaseRunConfig.total_steps` is `int | None = None` so the
        # AdapterConfig path (which has its own override) can default
        # it. The pretrain script genuinely requires it — surface that
        # as a pydantic ValueError rather than a `print + return 2`
        # in the script (PR #115 review #5).
        if self.total_steps is None:
            raise ValueError(
                "PretrainConfig requires total_steps; pass --total-steps N "
                "or set it in the JSON config"
            )
        if self.accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps must be positive, got {self.accumulation_steps}"
            )
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {self.checkpoint_interval}"
            )
        if self.val_every is not None and self.val_every <= 0:
            raise ValueError(
                f"val_every must be positive when set, got {self.val_every}"
            )
        if self.patience is not None and self.patience <= 0:
            raise ValueError(
                f"patience must be positive when set, got {self.patience}"
            )
        if self.patience is not None and self.val_every is None:
            raise ValueError(
                "patience requires val_every (early stopping keys on the "
                "held-out validation loss, which is only computed when "
                "val_every is set)"
            )
        if self.pause_after_steps is not None and self.pause_after_steps <= 0:
            raise ValueError(
                "pause_after_steps must be positive when set, got "
                f"{self.pause_after_steps}"
            )
        if self.variants is not None:
            if len(self.variants) == 0:
                raise ValueError(
                    "variants, if set, must be a non-empty subset of "
                    "small/base/large"
                )
            if len(set(self.variants)) != len(self.variants):
                raise ValueError(
                    f"variants must not contain duplicates, got {self.variants}"
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
    # Which v2 checkpoint to adapt. Loaded through
    # ``pawn.checkpoint.resolve_checkpoint_source`` — accepts a local
    # directory or a HF repo ID (snapshot-downloaded then read as a v2
    # safetensors checkpoint). v1 PyTorch artifacts are not loadable in
    # v2; the legacy converter was removed in the H.2 housekeeping commit.
    # Defaults to the v2 supernet base slice published from production
    # pretraining runs.
    checkpoint: str = "thomas-schweich/pawn-base-v2"
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
        # Mirror PretrainConfig — every training run needs a concrete
        # step budget, surfaced as a pydantic ValueError at parse time
        # so the lab manager + discriminated-union dispatch fail loudly
        # rather than at script entry (round-1 review-type-correctness).
        if self.total_steps is None:
            raise ValueError(
                "AdapterConfig requires total_steps; pass --total-steps N "
                "or set it in the JSON config"
            )
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
        # v1 parity: emit the deprecation warning only when the user
        # *explicitly* set `max_games` (not when it inherits its `None`
        # default), so re-loading a saved config that already wrote
        # `steps_per_epoch` stays silent. v1 interpreted `max_games` as
        # `steps_per_epoch = max_games // batch_size` at the adapter
        # boundary; the same conversion is the documented migration path.
        if self.max_games is not None and "max_games" in self.model_fields_set:
            warnings.warn(
                "max_games is deprecated for adapter runs; pass "
                "steps_per_epoch (int or 'all') instead. max_games is "
                "interpreted as `steps_per_epoch = max_games // batch_size`.",
                DeprecationWarning,
                stacklevel=2,
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
        # Mirror PretrainConfig / AdapterConfig — surface the missing
        # `total_steps` at pydantic parse time so the discriminated-union
        # dispatch fails loudly rather than deferring to the script's
        # entry point (round-1 review-type-correctness).
        if self.total_steps is None:
            raise ValueError(
                "SpecializedCLMConfig requires total_steps; pass "
                "--total-steps N or set it in the JSON config"
            )
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
        if self.max_games is not None and "max_games" in self.model_fields_set:
            warnings.warn(
                "max_games is deprecated; pass steps_per_epoch (int or "
                "'all') instead. max_games is interpreted as "
                "`steps_per_epoch = max_games // batch_size`.",
                DeprecationWarning,
                stacklevel=2,
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


class DistillConfig(BaseRunConfig):
    """Distil a frozen teacher checkpoint into a from-scratch student.

    The canonical-ladder mechanism (plan §7): a frozen teacher supplies
    soft targets that a from-scratch student (specialized_clm shapes)
    matches via a temperature-scaled KL (optionally mixed with ground-truth
    cross-entropy). Logit-only — the student is **not** a width slice of the
    teacher, so there is no hidden-state matching.

    The student's architecture comes from either ``student_supernet`` (a
    ``tiny``/``production`` :data:`pawn.config.SUPERNET` preset) or the four
    explicit ``d_model`` / ``n_layers`` / ``n_heads`` / ``d_ff`` dims. The
    teacher's conditioning / ``C`` is inherited at load time from its own
    checkpoint (reuse of the Phase-A load-time C-assert), so this config
    only carries the student + objective knobs.
    """

    run_type: Literal["distill"] = "distill"

    # --- Teacher -------------------------------------------------------
    # The frozen teacher checkpoint to distil from — a local v2 checkpoint
    # directory or a HF repo ID (resolved via
    # ``pawn.checkpoint.resolve_checkpoint_source``). Required; the CLI maps
    # ``--distill-from`` onto it.
    distill_from: str

    # --- Objective -----------------------------------------------------
    objective: DistillObjective = "mix"
    # Softmax temperature for the KL term (Hinton et al. 2015). >1 softens
    # the teacher distribution; the KL is scaled by ``temperature²`` to keep
    # its gradient comparable to the CE term.
    temperature: float = 2.0
    # mix weight: ``alpha·ce + (1-alpha)·kl``. alpha=1 ⇒ pure CE, alpha=0 ⇒
    # pure KL.
    alpha: float = 0.5

    # --- Student architecture ------------------------------------------
    # Preset student shape. ``None`` ⇒ the four explicit dims are required.
    student_supernet: SupernetName | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None

    # --- Data source (same shape as AdapterConfig) ---------------------
    pgn: str = "thomas-schweich/pawn-lichess-full"
    pgn_val_split: str | None = "validation"

    # --- Cadence -------------------------------------------------------
    checkpoint_interval: int = 5000
    # (B2) micro-batches accumulated per optimizer step. >1 emits
    # ``(K, N, B, T)`` chunks so the distill trainer's accumulation scan
    # sums N micro-grads before each update — effective batch N×B at B's
    # per-step memory cost. Mirrors PretrainConfig.accumulation_steps so the
    # distill trainer reaches the same effective-batch knob the other
    # trainers expose. Default 1 (no accumulation).
    accumulation_steps: int = 1

    @model_validator(mode="after")
    def _check_distill(self) -> "DistillConfig":
        if self.total_steps is None:
            raise ValueError(
                "DistillConfig requires total_steps; pass --total-steps N "
                "or set it in the JSON config"
            )
        if self.accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps must be positive, got "
                f"{self.accumulation_steps}"
            )
        if self.objective in ("kl", "mix") and self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}"
            )
        if self.objective == "mix" and not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be in [0, 1] for objective=mix, got {self.alpha}"
            )
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got "
                f"{self.checkpoint_interval}"
            )
        return self

    @model_validator(mode="after")
    def _check_student_arch(self) -> "DistillConfig":
        """Exactly one of ``student_supernet`` or the explicit-dim quad
        must specify the student shape — never both, never neither.

        A preset + partial dims would silently ignore the dims; a missing
        preset + partial dims would crash at student init. Surface both as
        a parse-time ValueError so the discriminated-union dispatch fails
        loudly.
        """
        explicit = (self.d_model, self.n_layers, self.n_heads, self.d_ff)
        any_explicit = any(v is not None for v in explicit)
        all_explicit = all(v is not None for v in explicit)
        if self.student_supernet is not None:
            if any_explicit:
                raise ValueError(
                    "pass either student_supernet OR explicit student dims "
                    "(d_model/n_layers/n_heads/d_ff), not both"
                )
            return self
        if not all_explicit:
            raise ValueError(
                "DistillConfig requires either student_supernet or all four "
                "explicit student dims (d_model, n_layers, n_heads, d_ff)"
            )
        # All four explicit dims present — validate them (mirrors
        # SpecializedCLMConfig._check_arch).
        assert (
            self.d_model is not None and self.n_layers is not None
            and self.n_heads is not None and self.d_ff is not None
        )
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


# Discriminated union: pydantic dispatches by run_type at parse time.
# `pawn.lab` returns this via `lab_schema`; CLI drivers in S13 build
# the appropriate subclass and the trainer reads the right discriminator.
RunConfig = Annotated[
    Union[PretrainConfig, AdapterConfig, SpecializedCLMConfig, DistillConfig],
    Field(discriminator="run_type"),
]
"""Discriminated union of all v2 JAX run-config types."""
