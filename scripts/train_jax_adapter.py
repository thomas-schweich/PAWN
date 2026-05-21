"""JAX adapter training driver.

Trains one of the eight adapter strategies
(``lora`` / ``film`` / ``unfreeze`` / ``bottleneck`` / ``hybrid`` /
``sparse`` / ``rosa`` / ``specialized_clm``) on a finite corpus
(Rust-engine random games serve as the verification proxy for the
production Lichess Elo-slice cache).

Execution order:

  1. Parse args; resolve the strategy. Validates ``--total-steps > 0``,
     ``--total-steps % --k == 0``, ``--batch-size > 0``, ``--seq-len
     > 0``, ``--seq-len <= supernet.max_seq_len``, plus per-strategy
     argument constraints (e.g. ``--rank > 0`` for LoRA / RoSA;
     ``--bottleneck-dim > 0`` for Bottleneck).
  2. Build the LR schedule (catches misconfigs before any filesystem
     write).
  3. Estimate corpus memory; abort if it exceeds ``--max-corpus-gb``.
  4. Generate the Rust-engine corpus (treat as a finite dataset:
     train + val split).
  5. Create the run directory and write ``config.json``.
  6. Init the supernet, slice to ``--variant`` (or build a from-
     scratch model for ``specialized_clm``), wrap with the chosen
     adapter, init two-tier optimizer state, build the K-step scan +
     eval callable.
  7. Train for ``--total-steps`` chunks; eval every ``--val-every``
     chunks; log per-chunk train + val metrics to ``metrics.jsonl``.

The eval split is the last ``--val-frac`` of the generated corpus.

RoSA dispatch ships the parameterisation + a single-phase joint
training schedule for verification-scale runs; the legacy three-
phase schedule (LoRA warmup → gradient-mask gen → joint training)
lands in a follow-up chunk — see ``docs/jax-migration.md`` §12.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.adapter_trainer import (
    AdapterModelProto,
    init_adapter_state,
    make_adapter_scan_step,
    make_adapter_train_step,
    make_eval_step,
)
from pawn.adapters import (
    BottleneckConfig,
    FiLMConfig,
    HybridConfig,
    LoRAConfig,
    RoSAConfig,
    RoSAModel,
    SparseConfig,
    SpecializedCLMConfig,
    UnfreezeConfig,
    adapter_filter,
    bottleneck_adapter_filter,
    film_adapter_filter,
    hybrid_adapter_filter,
    init_bottleneck_model,
    init_film_model,
    init_hybrid_model,
    init_lora_model,
    init_rosa_model,
    init_specialized_clm,
    init_sparse_model,
    init_unfreeze_model,
    rosa_adapter_filter,
    rosa_compute_phase2_mask,
    rosa_lora_only_adapter_filter,
    rosa_set_mask,
    specialized_clm_adapter_filter,
    sparse_adapter_filter,
    unfreeze_adapter_filter,
    unfreeze_gradient_mask,
)
from pawn.config import (
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
    ModelConfig,
)
from pawn.corpus import generate_corpus
from pawn.model import PAWNModel, init_model, sliced
from pawn.trainer import Batch, make_lr_schedule, make_optimizer

STRATEGIES = (
    "lora", "film", "unfreeze", "bottleneck",
    "hybrid", "sparse", "rosa", "specialized_clm",
)


def _slug() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{ts}_{os.getpid()}"


def _resolve_supernet(name: str) -> tuple[ModelConfig, dict[str, ModelConfig]]:
    if name == "tiny":
        return TINY_SUPERNET, TINY_VARIANTS
    if name == "supernet":
        return SUPERNET, VARIANTS
    raise ValueError(
        f"unknown --supernet {name!r}; expected one of tiny|supernet"
    )


def _stage(arr: np.ndarray) -> jax.Array:
    """One-shot host→device transfer."""
    return jnp.asarray(arr)


# ---------------------------------------------------------------------------
# Per-strategy build dispatch
# ---------------------------------------------------------------------------


# Each build function returns a 3-tuple:
#   (model, adapter_filter_fn, gradient_mask | None)
# ``gradient_mask`` is only non-None for strategies that need per-
# element gradient gating (``unfreeze``: per-layer slicing of layer-
# stacked weights).
BuildResult = tuple[
    eqx.Module,
    # See ``pawn.adapter_trainer.AdapterFilterFn`` for why this uses
    # variadic-arg typing instead of ``Callable[[eqx.Module],
    # eqx.Module]``.
    Callable[..., eqx.Module],
    eqx.Module | None,
]


def _build_lora(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    cfg = LoRAConfig(
        rank=args.rank, targets=tuple(args.lora_targets),
        alpha=args.lora_alpha,
    )
    return init_lora_model(backbone, cfg, key), adapter_filter, None


def _build_film(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    cfg = FiLMConfig(use_output_film=args.film_output)
    return init_film_model(backbone, cfg, key), film_adapter_filter, None


def _build_unfreeze(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    cfg = UnfreezeConfig(
        n_unfreeze=args.n_unfreeze,
        include_lm_head=args.include_lm_head,
        include_embeddings=args.include_embeddings,
    )
    model = init_unfreeze_model(backbone, cfg, key)
    return model, unfreeze_adapter_filter, unfreeze_gradient_mask(model)


def _build_bottleneck(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    cfg = BottleneckConfig(
        bottleneck_dim=args.bottleneck_dim,
        n_hidden=args.bottleneck_n_hidden,
        adapt_attn=not args.bottleneck_no_attn,
        adapt_ffn=not args.bottleneck_no_ffn,
    )
    return init_bottleneck_model(backbone, cfg, key), bottleneck_adapter_filter, None


def _build_hybrid(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    cfg = HybridConfig(
        lora=LoRAConfig(
            rank=args.rank, targets=tuple(args.lora_targets),
            alpha=args.lora_alpha,
        ),
        film=FiLMConfig(use_output_film=args.film_output),
    )
    return init_hybrid_model(backbone, cfg, key), hybrid_adapter_filter, None


def _build_sparse(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    cfg = SparseConfig(
        targets=tuple(args.sparse_targets),
        density=args.sparse_density,
        hard=args.sparse_hard,
    )
    return init_sparse_model(backbone, cfg, key), sparse_adapter_filter, None


def _build_rosa(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    cfg = RoSAConfig(
        rank=args.rank, targets=tuple(args.rosa_targets),
        alpha=args.lora_alpha,
    )
    model = init_rosa_model(backbone, cfg, key)
    # Default ``init_rosa_model`` sets every mask to all-False so the
    # identity-at-init invariant holds (B = 0, Δ = 0, mask = False →
    # W_eff = W, sparse leg vanishes). For the three-phase schedule
    # (``--rosa-warmup-frac > 0``) that's the right initial condition
    # — Phase 2 computes the real mask before Phase 3 starts using
    # Δ. For ``--rosa-warmup-frac == 0`` (single-phase joint training)
    # the mask never gets computed by the trainer, and an all-False
    # mask leaves ``delta * mask = 0`` so ``dL/dΔ`` is zero
    # everywhere and Δ never trains — silently defeating the
    # documented joint-training semantics. Prime the mask to
    # all-True so Δ trains dense alongside A, B (the user explicitly
    # opted out of sparsity by setting warmup-frac to 0; the dense
    # joint mode is the legitimate fallback).
    if args.rosa_warmup_frac == 0.0:
        n_layers = backbone.cfg.n_layers
        d = backbone.cfg.d_model
        all_true_masks = {
            t: jnp.ones((n_layers, d, d), dtype=jnp.bool_)
            for t in cfg.targets
        }
        model = rosa_set_mask(model, all_true_masks)
    return model, rosa_adapter_filter, None


def _build_specialized_clm(
    backbone: PAWNModel, args: argparse.Namespace, key: jax.Array,
) -> BuildResult:
    # specialized_clm doesn't wrap a backbone — it's a from-scratch
    # PAWNModel at the small-baseline scale. We discard the
    # sliced-backbone argument and use the strategy's config to
    # build a fresh model.
    del backbone
    cfg = SpecializedCLMConfig(
        d_model=args.specialized_d_model,
        n_layers=args.specialized_n_layers,
        n_heads=args.specialized_n_heads,
        d_ff=args.specialized_d_ff,
    )
    return init_specialized_clm(cfg, key), specialized_clm_adapter_filter, None


_BUILDERS: dict[str, Callable[[PAWNModel, argparse.Namespace, jax.Array], BuildResult]] = {
    "lora": _build_lora,
    "film": _build_film,
    "unfreeze": _build_unfreeze,
    "bottleneck": _build_bottleneck,
    "hybrid": _build_hybrid,
    "sparse": _build_sparse,
    "rosa": _build_rosa,
    "specialized_clm": _build_specialized_clm,
}


def _strategy_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-serialisable dict of the strategy's per-run config
    for inclusion in ``config.json``."""
    s = args.strategy
    if s == "lora" or s == "hybrid":
        d: dict[str, Any] = {
            "rank": args.rank,
            "targets": list(args.lora_targets),
            "alpha": args.lora_alpha,
        }
        if s == "hybrid":
            d["film_output"] = args.film_output
        return d
    if s == "film":
        return {"use_output_film": args.film_output}
    if s == "unfreeze":
        return {
            "n_unfreeze": args.n_unfreeze,
            "include_lm_head": args.include_lm_head,
            "include_embeddings": args.include_embeddings,
        }
    if s == "bottleneck":
        return {
            "bottleneck_dim": args.bottleneck_dim,
            "n_hidden": args.bottleneck_n_hidden,
            "adapt_attn": not args.bottleneck_no_attn,
            "adapt_ffn": not args.bottleneck_no_ffn,
        }
    if s == "sparse":
        return {
            "targets": list(args.sparse_targets),
            "density": args.sparse_density,
            "hard": args.sparse_hard,
        }
    if s == "rosa":
        return {
            "rank": args.rank,
            "targets": list(args.rosa_targets),
            "alpha": args.lora_alpha,
            # RoSA-schedule hyperparameters — without these the
            # saved config can't be used to re-run the same training
            # behavior (single-phase dense vs three-phase sparse,
            # top-k density).
            "warmup_frac": args.rosa_warmup_frac,
            "top_k_frac": args.rosa_top_k_frac,
        }
    if s == "specialized_clm":
        return {
            "d_model": args.specialized_d_model,
            "n_layers": args.specialized_n_layers,
            "n_heads": args.specialized_n_heads,
            "d_ff": args.specialized_d_ff,
        }
    raise ValueError(f"unknown strategy {s!r}")  # pragma: no cover — argparse guards


def _validate_strategy_args(
    args: argparse.Namespace, variant_cfg: ModelConfig | None,
) -> None:
    """Per-strategy argument sanity checks. ``variant_cfg`` is the
    sliced backbone config for non-specialized_clm strategies, or
    ``None`` for specialized_clm — needed for variant-aware checks
    like ``--n-unfreeze <= n_layers``."""
    if args.strategy in ("lora", "hybrid", "rosa"):
        if args.rank <= 0:
            raise SystemExit(
                f"--rank={args.rank} must be positive for {args.strategy}"
            )
        # LoRA / Hybrid / RoSA share --lora-alpha. Pre-fix (Codex
        # round-4 P2), invalid alpha was caught inside
        # `LoRAConfig.__post_init__` AFTER corpus generation +
        # run_dir creation, leaving orphan dirs on validation
        # failure.
        if args.lora_alpha is not None and args.lora_alpha <= 0:
            raise SystemExit(
                f"--lora-alpha={args.lora_alpha} must be positive (or None)"
            )
    if args.strategy == "bottleneck":
        if args.bottleneck_dim <= 0:
            raise SystemExit(
                f"--bottleneck-dim={args.bottleneck_dim} must be positive"
            )
        if args.bottleneck_n_hidden < 0:
            raise SystemExit(
                f"--bottleneck-n-hidden={args.bottleneck_n_hidden} must be >= 0"
            )
        # At least one of attn / FFN must be enabled or the adapter
        # contributes nothing. Codex round-4 P2.
        if args.bottleneck_no_attn and args.bottleneck_no_ffn:
            raise SystemExit(
                "--bottleneck-no-attn and --bottleneck-no-ffn cannot both be "
                "set — that disables the adapter entirely (no trainable "
                "parameters)"
            )
    if args.strategy == "sparse" and not 0.0 < args.sparse_density <= 1.0:
        raise SystemExit(
            f"--sparse-density={args.sparse_density} must be in (0, 1]"
        )
    if args.strategy == "specialized_clm":
        # --specialized-* shape constraints. Pre-fix (Codex round-4
        # P2), invalid shapes were caught inside
        # `ModelConfig.__post_init__` AFTER corpus generation +
        # run_dir creation.
        if args.specialized_d_model <= 0 or args.specialized_n_heads <= 0:
            raise SystemExit(
                f"--specialized-d-model={args.specialized_d_model} and "
                f"--specialized-n-heads={args.specialized_n_heads} must be "
                f"positive"
            )
        if args.specialized_d_model % args.specialized_n_heads != 0:
            raise SystemExit(
                f"--specialized-d-model={args.specialized_d_model} must be "
                f"divisible by --specialized-n-heads={args.specialized_n_heads}"
            )
        head_dim = args.specialized_d_model // args.specialized_n_heads
        if head_dim % 2 != 0:
            raise SystemExit(
                f"specialized head_dim = d_model / n_heads = {head_dim} "
                f"must be even (RoPE rotates pairs)"
            )
        if args.specialized_d_ff <= 0:
            raise SystemExit(
                f"--specialized-d-ff={args.specialized_d_ff} must be positive"
            )
        if args.specialized_n_layers <= 0:
            raise SystemExit(
                f"--specialized-n-layers={args.specialized_n_layers} must "
                f"be positive"
            )
    if args.strategy == "unfreeze":
        if args.n_unfreeze < 0:
            raise SystemExit(
                f"--n-unfreeze={args.n_unfreeze} must be >= 0"
            )
        # Variant-aware check: n_unfreeze can't exceed the chosen
        # backbone's depth. Fired upfront so the user doesn't burn
        # corpus generation only to crash inside ``init_unfreeze_model``.
        if variant_cfg is not None and args.n_unfreeze > variant_cfg.n_layers:
            raise SystemExit(
                f"--n-unfreeze={args.n_unfreeze} exceeds the chosen "
                f"variant's n_layers={variant_cfg.n_layers}"
            )
        # At least one of (layer-stacked / lm_head / embeddings) must
        # be trainable, else the adapter is a no-op (the run would
        # spend full compute updating zero parameters). Codex round-5
        # P2.
        if (
            args.n_unfreeze == 0
            and not args.include_lm_head
            and not args.include_embeddings
        ):
            raise SystemExit(
                "Unfreeze configuration would train zero parameters: "
                "set at least one of --n-unfreeze > 0, --include-lm-head, "
                "or --include-embeddings."
            )
    if args.strategy == "rosa":
        if not 0.0 <= args.rosa_warmup_frac < 1.0:
            raise SystemExit(
                f"--rosa-warmup-frac={args.rosa_warmup_frac} must be in [0, 1)"
            )
        if not 0.0 < args.rosa_top_k_frac <= 1.0:
            raise SystemExit(
                f"--rosa-top-k-frac={args.rosa_top_k_frac} must be in (0, 1]"
            )


def _compute_rosa_phase2_dl_ddelta(
    rosa_model: RoSAModel, batch: Batch,
) -> dict[str, jax.Array]:
    """Compute ``dL/dΔ_q``, ``dL/dΔ_k``, ``dL/dΔ_v``, ``dL/dΔ_o`` on
    a batch with ``mask = all-True``.

    Δ at this point is still zero (from init); the all-True mask
    means Δ contributes additively to ``W_eff``, so
    ``dL/dΔ_{ij} = dL/dW_eff_{ij}`` and is non-zero even though Δ
    itself is zero. The gradient signal is what Phase-2 ranks by
    magnitude to pick the top-k entries for the fixed sparse mask.

    Returns ``{target: dL/dΔ_array}`` for each active target in
    ``rosa_model.rosa.cfg.targets``.
    """
    from pawn.adapters.rosa import sparse_only_adapter_filter
    from pawn.trainer import cross_entropy_loss

    cfg = rosa_model.rosa.cfg
    n_layers = rosa_model.backbone.cfg.n_layers
    d = rosa_model.backbone.cfg.d_model
    all_true = {
        t: jnp.ones((n_layers, d, d), dtype=jnp.bool_) for t in cfg.targets
    }
    primed = rosa_set_mask(rosa_model, all_true)
    trainable, frozen = eqx.partition(
        primed, sparse_only_adapter_filter(primed),
    )
    tokens, attn_mask, targets, loss_mask = batch

    def loss_fn(trn: eqx.Module) -> jax.Array:
        # eqx.combine preserves the concrete RoSAModel type at runtime;
        # cast through the same structural protocol the adapter
        # trainer uses so pyright accepts ``m(tokens, attn_mask)``
        # without a suppression.
        m = cast(AdapterModelProto, eqx.combine(trn, frozen))
        logits = m(tokens, attn_mask)
        return cross_entropy_loss(logits, targets, loss_mask)

    grads = eqx.filter_grad(loss_fn)(trainable)
    return {
        t: getattr(grads.rosa, f"delta_{t}") for t in cfg.targets
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", choices=STRATEGIES, default="lora",
        help="adapter strategy to dispatch.",
    )
    parser.add_argument(
        "--supernet", choices=["tiny", "supernet"], default="tiny",
    )
    parser.add_argument(
        "--variant", default="base",
        help="which sliced variant to train the adapter on (one of "
        "the keys of the supernet's VARIANTS dict). Ignored when "
        "--strategy=specialized_clm (which trains a from-scratch model).",
    )
    # LoRA / Hybrid / RoSA shared args.
    parser.add_argument(
        "--rank", type=int, default=4,
        help="LoRA / Hybrid / RoSA rank.",
    )
    parser.add_argument(
        "--lora-targets", nargs="+", default=["q", "v"],
        choices=["q", "k", "v", "o"],
        help="LoRA / Hybrid target attention projections.",
    )
    parser.add_argument(
        "--lora-alpha", type=float, default=None,
        help="LoRA / Hybrid / RoSA scaling. ``None`` → alpha = rank.",
    )
    # FiLM / Hybrid shared args. ``FiLMConfig.use_output_film``
    # defaults to True in the library; expose the opt-OUT flag here
    # so default-flagless invocations preserve the legacy / library
    # default (Codex round-5 P2).
    parser.add_argument(
        "--no-film-output", dest="film_output", action="store_false",
        help="FiLM / Hybrid: disable lm_head output modulation "
             "(default: enabled, matching FiLMConfig's library default).",
    )
    parser.set_defaults(film_output=True)
    # Unfreeze args.
    parser.add_argument(
        "--n-unfreeze", type=int, default=2,
        help="Unfreeze: number of top transformer layers to unfreeze.",
    )
    parser.add_argument(
        "--include-lm-head", action="store_true", default=True,
        help="Unfreeze: also train final_norm + lm_head.",
    )
    parser.add_argument(
        "--no-include-lm-head", dest="include_lm_head",
        action="store_false",
    )
    parser.add_argument(
        "--include-embeddings", action="store_true",
        help="Unfreeze: also train input embedding tables.",
    )
    # Bottleneck args.
    parser.add_argument(
        "--bottleneck-dim", type=int, default=32,
        help="Bottleneck: hidden dimension of the Houlsby MLP.",
    )
    parser.add_argument(
        "--bottleneck-n-hidden", type=int, default=0,
        help="Bottleneck: extra hidden layers (0 = single-projection).",
    )
    parser.add_argument(
        "--bottleneck-no-attn", action="store_true",
        help="Bottleneck: skip the attention-side adapter.",
    )
    parser.add_argument(
        "--bottleneck-no-ffn", action="store_true",
        help="Bottleneck: skip the FFN-side adapter.",
    )
    # Sparse args.
    parser.add_argument(
        "--sparse-targets", nargs="+", default=["q", "v"],
        choices=["q", "k", "v", "o"],
        help="Sparse: attention projections to wrap.",
    )
    parser.add_argument(
        "--sparse-density", type=float, default=0.01,
        help="Sparse: initial density (fraction of unmasked entries).",
    )
    parser.add_argument(
        "--sparse-hard", action="store_true",
        help="Sparse: use the straight-through-estimator hard mask "
             "(default: soft sigmoid).",
    )
    # RoSA args.
    parser.add_argument(
        "--rosa-targets", nargs="+", default=["q", "v"],
        choices=["q", "k", "v", "o"],
        help="RoSA: attention projections to wrap.",
    )
    parser.add_argument(
        "--rosa-warmup-frac", type=float, default=0.5,
        help="RoSA: fraction of total chunks spent in Phase 1 (LoRA "
             "warmup). The remaining chunks are Phase 3 (joint training); "
             "Phase 2 (gradient-magnitude mask generation) takes one "
             "extra batch between them and doesn't consume a training "
             "chunk. Set to 0 to skip Phases 1 + 2 entirely and run "
             "single-phase joint training from step 0.",
    )
    parser.add_argument(
        "--rosa-top-k-frac", type=float, default=0.01,
        help="RoSA: fraction of sparse-leg entries (per layer per "
             "target) kept active by the Phase-2 mask. Default 0.01 "
             "= top 1%% by |dL/dΔ| magnitude.",
    )
    # SpecializedCLM args.
    parser.add_argument(
        "--specialized-d-model", type=int, default=84,
        help="SpecializedCLM: model width.",
    )
    parser.add_argument(
        "--specialized-n-layers", type=int, default=2,
        help="SpecializedCLM: number of transformer layers.",
    )
    parser.add_argument(
        "--specialized-n-heads", type=int, default=2,
        help="SpecializedCLM: number of attention heads.",
    )
    parser.add_argument(
        "--specialized-d-ff", type=int, default=192,
        help="SpecializedCLM: FFN hidden dim.",
    )
    # Training args.
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="fraction of the generated corpus held out for validation.",
    )
    parser.add_argument(
        "--val-every", type=int, default=5,
        help="run validation every N chunks (and at the final chunk).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--max-corpus-gb", type=float, default=8.0)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # Upfront guards — same shape as the pretrain driver.
    if args.k <= 0:
        raise SystemExit(f"--k={args.k} must be a positive integer")
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size={args.batch_size} must be positive")
    if args.seq_len <= 0:
        raise SystemExit(f"--seq-len={args.seq_len} must be positive")
    if args.total_steps <= 0:
        raise SystemExit(f"--total-steps={args.total_steps} must be positive")
    if args.total_steps % args.k != 0:
        raise SystemExit(
            f"--total-steps={args.total_steps} must be a multiple of --k={args.k}"
        )
    if not 0.0 < args.val_frac < 1.0:
        raise SystemExit(
            f"--val-frac={args.val_frac} must be in (0, 1)"
        )

    # Variant resolution upfront so strategy validation (e.g.
    # --n-unfreeze ≤ variant.n_layers) can see the variant config
    # before any filesystem side-effect (Codex round-3 P2).
    supernet_cfg, variants = _resolve_supernet(args.supernet)
    if args.strategy != "specialized_clm":
        if args.variant not in variants:
            raise SystemExit(
                f"--variant={args.variant!r} not in {list(variants.keys())}"
            )
        variant_cfg: ModelConfig | None = variants[args.variant]
    else:
        variant_cfg = None
    if args.seq_len > supernet_cfg.max_seq_len:
        raise SystemExit(
            f"--seq-len={args.seq_len} exceeds supernet.max_seq_len="
            f"{supernet_cfg.max_seq_len}"
        )

    _validate_strategy_args(args, variant_cfg)

    # RoSA: pre-compute the warmup-chunk split here (before any
    # corpus generation or filesystem write) so a degenerate
    # rosa-warmup-frac that leaves no Phase-3 chunks fails fast
    # without leaking an orphan ``jax_adapter_run_*`` directory.
    # (Codex round-2 P3 + the test_validation_failures_do_not_create_run_dir
    # invariant from the original driver port.)
    n_chunks_for_validation = args.total_steps // args.k
    rosa_three_phase = (
        args.strategy == "rosa" and args.rosa_warmup_frac > 0.0
    )
    if rosa_three_phase:
        rosa_warmup_chunks = max(
            1, int(round(args.rosa_warmup_frac * n_chunks_for_validation)),
        )
        if rosa_warmup_chunks >= n_chunks_for_validation:
            raise SystemExit(
                f"--rosa-warmup-frac={args.rosa_warmup_frac} leaves no "
                f"chunks for Phase 3 joint training (rosa_warmup_chunks="
                f"{rosa_warmup_chunks}, n_chunks={n_chunks_for_validation}). "
                f"Reduce --rosa-warmup-frac or increase --total-steps."
            )

    corpus_seed = args.seed if args.corpus_seed is None else args.corpus_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed

    try:
        sched = make_lr_schedule(
            peak_lr=args.lr,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
        )
    except ValueError as exc:
        raise SystemExit(f"LR-schedule configuration error: {exc}") from exc

    # Train + val games. Round n_train_games so we get exactly
    # total_steps × batch_size, and add the val fraction on top.
    n_train_games = args.total_steps * args.batch_size
    n_val_games = max(1, int(n_train_games * args.val_frac))
    # Round n_val_games to a multiple of batch_size so the eval loop
    # processes whole batches.
    n_val_games = (n_val_games // args.batch_size) * args.batch_size
    if n_val_games == 0:
        raise SystemExit(
            f"--val-frac={args.val_frac} produces <1 batch of val games. "
            "Increase --val-frac or --total-steps."
        )
    n_total = n_train_games + n_val_games
    bytes_per_game = args.seq_len * 10 + 1
    estimated_gb = n_total * bytes_per_game / (1024 ** 3)
    if estimated_gb > args.max_corpus_gb:
        raise SystemExit(
            f"corpus would need ~{estimated_gb:.2f} GiB > "
            f"--max-corpus-gb={args.max_corpus_gb}"
        )

    variant_label = "from-scratch" if args.strategy == "specialized_clm" else args.variant
    print(
        f"[setup] strategy={args.strategy} supernet={args.supernet} "
        f"variant={variant_label} total_steps={args.total_steps} "
        f"K={args.k} B={args.batch_size} T={args.seq_len} "
        f"n_train_games={n_train_games} n_val_games={n_val_games}"
    )

    t0 = time.perf_counter()
    print(
        f"[corpus] generating {n_total} games (seed={corpus_seed})"
    )
    corpus = generate_corpus(
        n_games=n_total,
        max_ply=args.seq_len,
        seq_len=args.seq_len,
        seed=corpus_seed,
    )
    print(f"[corpus] done in {time.perf_counter() - t0:.1f}s")

    # Train / val split: first n_train, then n_val.
    train_tokens = corpus.tokens[:n_train_games]
    train_attn = corpus.attn_mask[:n_train_games]
    train_targets = corpus.targets[:n_train_games]
    train_loss = corpus.loss_mask[:n_train_games]
    val_tokens = corpus.tokens[n_train_games:]
    val_attn = corpus.attn_mask[n_train_games:]
    val_targets = corpus.targets[n_train_games:]
    val_loss = corpus.loss_mask[n_train_games:]

    run_dir = Path(args.logs_dir) / f"jax_adapter_run_{_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "strategy": args.strategy,
                "strategy_config": _strategy_config_dict(args),
                "supernet": args.supernet,
                "variant": variant_label,
                "supernet_cfg": dataclasses.asdict(supernet_cfg),
                "variant_cfg": (
                    dataclasses.asdict(variant_cfg) if variant_cfg is not None else None
                ),
                "total_steps": args.total_steps,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "k": args.k,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "val_frac": args.val_frac,
                "val_every": args.val_every,
                "n_train_games": n_train_games,
                "n_val_games": n_val_games,
                "seed": args.seed,
                "corpus_seed": corpus_seed,
                "model_seed": model_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Init: slice → wrap with the chosen adapter → partition for
    # adapter training. specialized_clm doesn't need a backbone at
    # all (it builds a from-scratch model from its `--specialized-*`
    # args), so we dispatch it without ever calling ``init_model`` on
    # the supernet — at SUPERNET scale this saves ~200 MB of needless
    # allocation that init_model would otherwise burn just to be
    # discarded by ``_build_specialized_clm``. Codex round-3 P2.
    key = jax.random.PRNGKey(model_seed)
    builder_key = jax.random.PRNGKey(model_seed + 1)
    if args.strategy == "specialized_clm":
        cfg = SpecializedCLMConfig(
            d_model=args.specialized_d_model,
            n_layers=args.specialized_n_layers,
            n_heads=args.specialized_n_heads,
            d_ff=args.specialized_d_ff,
        )
        model: eqx.Module = init_specialized_clm(cfg, builder_key)
        filter_fn: Callable[..., eqx.Module] = specialized_clm_adapter_filter
        gradient_mask: eqx.Module | None = None
    else:
        assert variant_cfg is not None  # guarded above; specialized_clm is the only None case
        supernet = init_model(supernet_cfg, key)
        backbone = sliced(supernet, variant_cfg)
        build_fn = _BUILDERS[args.strategy]
        model, filter_fn, gradient_mask = build_fn(
            backbone, args, builder_key,
        )
    # RoSA Phase-1 override: when the three-phase schedule is active
    # (``--rosa-warmup-frac > 0``), the warmup runs with the LoRA-only
    # filter; only A, B accumulate gradients. The joint filter the
    # builder returned applies after the Phase-2 mask gen
    # (which happens inside the training loop). When
    # ``--rosa-warmup-frac == 0`` the original joint filter is used
    # from step 0 and the three-phase scheduler is a no-op.
    # ``rosa_three_phase`` was pre-computed above as part of the
    # validation pass.
    if rosa_three_phase:
        active_filter_fn: Callable[..., eqx.Module] = rosa_lora_only_adapter_filter
    else:
        active_filter_fn = filter_fn
    optimizer = make_optimizer(sched)
    state, frozen = init_adapter_state(
        model, optimizer, adapter_filter_fn=active_filter_fn,
    )
    train_step = make_adapter_train_step(
        optimizer, frozen, gradient_mask=gradient_mask,
    )
    scan = make_adapter_scan_step(train_step, args.k)
    eval_step = make_eval_step(frozen)

    n_chunks = args.total_steps // args.k
    # ``rosa_warmup_chunks`` was pre-computed during the upfront
    # validation pass when ``rosa_three_phase`` was True; for non-RoSA
    # or zero-warmup runs it's 0 (no phase transition).
    if not rosa_three_phase:
        rosa_warmup_chunks = 0
    if rosa_three_phase:
        print(
            f"[train] n_chunks={n_chunks} (K={args.k} steps each); "
            f"RoSA Phase 1 = {rosa_warmup_chunks} chunks, "
            f"Phase 3 = {n_chunks - rosa_warmup_chunks} chunks; "
            f"top-k frac = {args.rosa_top_k_frac}"
        )
    else:
        print(f"[train] n_chunks={n_chunks} (K={args.k} steps each)")

    # Pre-shape train corpus into [N_chunks, K, B, T] views.
    chunk_shape = (n_chunks, args.k, args.batch_size, args.seq_len)
    train_tokens_chunks = train_tokens.reshape(chunk_shape)
    train_attn_chunks = train_attn.reshape(chunk_shape)
    train_targets_chunks = train_targets.reshape(chunk_shape)
    train_loss_chunks = train_loss.reshape(chunk_shape)

    def run_validation(trainable: eqx.Module) -> tuple[float, int]:
        """Iterate the val set in batches, return (mean_loss, total_n).
        We iterate on the host (eval is a one-shot diagnostic, not on
        the critical path) and accumulate weighted by n_supervised."""
        n = val_tokens.shape[0]
        total_loss = 0.0
        total_n = 0
        for s in range(0, n, args.batch_size):
            e = s + args.batch_size
            batch = (
                _stage(val_tokens[s:e]),
                _stage(val_attn[s:e]),
                _stage(val_targets[s:e]),
                _stage(val_loss[s:e]),
            )
            m = eval_step(trainable, batch)
            ns = int(m["n_supervised"])
            total_loss += float(m["loss"]) * ns
            total_n += ns
        return (total_loss / max(total_n, 1), total_n)

    best_val = float("inf")
    with metrics_path.open("w", encoding="utf-8") as mf:
        wall0 = time.perf_counter()
        step_start = 0
        for chunk_i in range(n_chunks):
            # RoSA Phase 2 → Phase 3 transition. Runs once, between
            # the warmup chunk and the first joint-training chunk.
            # Computes |dL/dΔ| on the current chunk's data (with
            # mask=all-True so Δ contributes additively to the
            # forward), picks the top ``rosa_top_k_frac`` entries per
            # layer / target, calls ``rosa_set_mask``, then re-inits
            # the adapter state under the joint filter so A, B, Δ
            # train together against the fixed mask.
            if rosa_three_phase and chunk_i == rosa_warmup_chunks:
                warmed_model = cast(
                    RoSAModel, eqx.combine(state.trainable, frozen),
                )
                # Gradient-estimation source: the LAST sub-batch of
                # the LAST Phase-1 chunk. Using chunk 0's first
                # sub-batch (an earlier draft of this code) silently
                # discarded the informational value of warmup — the
                # gradient signal was sampled against unseen data.
                # The last-Phase-1-sub-batch is the freshest
                # post-warmup data the model has encountered; the
                # legacy implementation averaged over a handful of
                # batches but the per-layer rank-order saturates
                # quickly so one batch is sufficient.
                mask_batch: Batch = (
                    _stage(train_tokens_chunks[chunk_i - 1, -1]),
                    _stage(train_attn_chunks[chunk_i - 1, -1]),
                    _stage(train_targets_chunks[chunk_i - 1, -1]),
                    _stage(train_loss_chunks[chunk_i - 1, -1]),
                )
                dl_ddelta = _compute_rosa_phase2_dl_ddelta(
                    warmed_model, mask_batch,
                )
                phase2_masks = rosa_compute_phase2_mask(
                    warmed_model, dl_ddelta, args.rosa_top_k_frac,
                )
                final_model = rosa_set_mask(warmed_model, phase2_masks)
                # Re-init the adapter state under the joint filter.
                # The optimizer instance is shared with Phase 1, but
                # ``optimizer.init`` produces a fresh ``opt_state``
                # whose internal step counter starts at 0 — so the
                # LR schedule replays its warmup ramp at the start
                # of Phase 3. This is a known design choice (Phase 3
                # gets its own warmup, which can help stabilise the
                # newly-active Δ gradients) rather than a strict
                # match for any production schedule. Callers who want
                # a continuous LR across phases should set
                # ``--warmup-steps`` small relative to
                # ``--total-steps`` * (1 - ``--rosa-warmup-frac``) so
                # the replayed warmup is a small fraction of Phase 3.
                #
                # We do preserve the FRAMEWORK-LEVEL step counter
                # (``state.step``) across the transition by manually
                # setting it after re-init — that's what the
                # metrics-log step column reads, so dropping it
                # would produce non-monotonic step counts and an
                # underreported total step count in the final row
                # (Codex round-2 P2). The Optax-internal step
                # counter (inside opt_state) still resets, by
                # design.
                phase1_step = state.step
                state, frozen = init_adapter_state(
                    final_model, optimizer, adapter_filter_fn=filter_fn,
                )
                state = state._replace(step=phase1_step)
                train_step = make_adapter_train_step(
                    optimizer, frozen, gradient_mask=gradient_mask,
                )
                scan = make_adapter_scan_step(train_step, args.k)
                eval_step = make_eval_step(frozen)
                if not args.quiet:
                    n_active = sum(
                        int(m.sum()) for m in phase2_masks.values()
                    )
                    # Local name avoids shadowing the outer
                    # ``n_total`` (= n_train_games + n_val_games)
                    # used during corpus sizing.
                    n_mask_total = sum(
                        m.size for m in phase2_masks.values()
                    )
                    pct = 100.0 * n_active / max(n_mask_total, 1)
                    print(
                        f"[rosa] Phase 2 → 3 transition at chunk "
                        f"{chunk_i + 1}/{n_chunks}: "
                        f"{n_active}/{n_mask_total} ({pct:.2f}%) sparse-"
                        f"leg entries active across targets="
                        f"{list(phase2_masks.keys())}"
                    )

            # Stage chunk to device.
            chunk: Batch = (
                _stage(train_tokens_chunks[chunk_i]),
                _stage(train_attn_chunks[chunk_i]),
                _stage(train_targets_chunks[chunk_i]),
                _stage(train_loss_chunks[chunk_i]),
            )
            t_chunk = time.perf_counter()
            # K-step ``lax.scan`` — one XLA dispatch per chunk instead
            # of K. Per-step metrics come back stacked on a leading
            # ``[K]`` axis.
            state, chunk_metrics = scan(state, chunk)
            jax.block_until_ready((state, chunk_metrics))
            dt = time.perf_counter() - t_chunk
            step_end = int(state.step)

            host_metrics = {
                mk: np.asarray(v) for mk, v in chunk_metrics.items()
            }
            row: dict[str, float | int | None] = {
                "chunk": chunk_i,
                "step_start": step_start,
                "step_end": step_end,
                "wall_s": time.perf_counter() - wall0,
                "chunk_wall_s": dt,
                "train_loss_mean": float(host_metrics["loss"].mean()),
                "train_loss_last": float(host_metrics["loss"][-1]),
                "grad_norm_mean": float(host_metrics["grad_norm"].mean()),
                "grad_norm_max": float(host_metrics["grad_norm"].max()),
                "val_loss": None,
                "val_n": None,
            }
            do_val = (
                (chunk_i + 1) % args.val_every == 0
                or chunk_i + 1 == n_chunks
            )
            if do_val:
                vloss, vn = run_validation(state.trainable)
                row["val_loss"] = vloss
                row["val_n"] = vn
                if vloss < best_val:
                    best_val = vloss
            mf.write(json.dumps(row) + "\n")
            if (chunk_i + 1) % 10 == 0 or chunk_i + 1 == n_chunks:
                mf.flush()
            if not args.quiet:
                vs = f" val={row['val_loss']:.4f}" if row["val_loss"] is not None else ""
                print(
                    f"[chunk {chunk_i + 1}/{n_chunks}] step={step_end} "
                    f"train_loss={row['train_loss_mean']:.4f}{vs} "
                    f"grad_norm={row['grad_norm_mean']:.3f} dt={dt:.2f}s"
                )
            step_start = step_end
    print(
        f"[done] total wall = {time.perf_counter() - wall0:.1f}s; "
        f"best_val={best_val:.4f}; run_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
