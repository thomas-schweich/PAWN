#!/usr/bin/env python3
"""Adapter fine-tuning entry point — one of 8 strategies on a Lichess Elo band.

Dispatches through `pawn.adapter_trainer.STRATEGIES`. Validates the
run config via pydantic; reads Lichess data via
`pawn.lichess_data.load_lichess_corpus`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pawn.adapter_trainer import (
    STRATEGIES,
    AdapterTrainState,
    dispatch_filter,
    dispatch_init,
    generate_rosa_masks,
    make_adapter_scan_step,
    make_adapter_train_step,
    make_adapter_val_metrics,
    rosa_phase1_to_phase3,
)
from pawn.adapters import (
    BottleneckConfig,
    FiLMConfig,
    HybridConfig,
    LoRAConfig,
    RoSAConfig,
    SparseConfig,
    SpecializedCLMConfig as AdapterSpecializedCLMConfig,
    UnfreezeConfig,
)
from pawn.checkpoint import resolve_checkpoint_source, save_model
from pawn.config import SUPERNET, TINY_SUPERNET, VARIANTS, TINY_VARIANTS
from pawn.corpus import Corpus, generate_corpus, legal_mask_for_games
from pawn.jax_setup import require_accelerator, resolve_device, setup_jax_caching
from pawn.lichess_data import load_lichess_corpus
from pawn.lifecycle import (
    HFPushTracker,
    build_training_state,
    drain_push_queue,
    install_sigterm_handler,
    push_checkpoint_async,
    read_resume_rng_blocks,
    write_schedule_health,
)
from pawn.logging import MetricsLogger, get_git_info
from pawn.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
    require_wandb_available,
)
from pawn.model import PAWNModel, init_model, sliced
from pawn.run_config import AdapterConfig
from pawn.trainer import (
    Batch, flatten_opt_state, make_lr_schedule,
    make_optimizer, slice_batch,
)


class _Cadence(NamedTuple):
    """Resolved adapter loop cadence/sampling knobs (B2 / plan §8.3).

    ``effective_total_steps`` is the step budget the loop runs;
    ``eval_interval`` is the validation cadence in steps; ``data_seed`` is
    the train-stream RNG seed (val uses ``data_seed + 1``); ``epoch_steps``
    is one epoch's step count (carried for diagnostics)."""

    effective_total_steps: int
    eval_interval: int
    data_seed: int
    epoch_steps: int


def _resolve_cadence(cfg: AdapterConfig, n_train_games: int) -> _Cadence:
    """Resolve the previously-inert ``epochs`` / ``steps_per_epoch`` /
    ``data_seed`` / ``val_every`` knobs into concrete loop parameters.

    Ports v1's ``total_steps = epochs × steps_per_epoch`` budget and
    ``epoch % val_every`` validation cadence onto v2's step-based loop:

    - ``data_seed`` (``None`` → 0) seeds the batch-sampling RNG.
    - ``steps_per_epoch`` is canonical for adapters (CLAUDE.md): an int is
      used verbatim, ``"all"`` resolves to ``n_train_games // batch_size``,
      and ``None`` falls back to ``cfg.total_steps`` as a single whole-run
      epoch (so ``epochs`` is then a no-op multiplier on the required
      ``total_steps`` budget and the step-based ``eval_interval`` drives
      validation).
    - When ``steps_per_epoch`` is set the budget is ``epochs × epoch_steps``
      and validation runs every ``val_every × epoch_steps`` steps.

    Pure + JAX-free so the resolution can be unit-tested directly.
    """
    assert cfg.total_steps is not None  # AdapterConfig requires it
    if cfg.steps_per_epoch is None:
        epoch_steps = cfg.total_steps
        effective_total_steps = cfg.total_steps
        eval_interval = cfg.eval_interval or cfg.log_interval
    else:
        if cfg.steps_per_epoch == "all":
            epoch_steps = max(1, n_train_games // cfg.batch_size)
        else:
            epoch_steps = cfg.steps_per_epoch
        effective_total_steps = cfg.epochs * epoch_steps
        eval_interval = cfg.val_every * epoch_steps
    data_seed = cfg.data_seed if cfg.data_seed is not None else 0
    return _Cadence(
        effective_total_steps=effective_total_steps,
        eval_interval=eval_interval,
        data_seed=data_seed,
        epoch_steps=epoch_steps,
    )


def _step_time(elapsed: float, step: int, run_start: int) -> float:
    """Mean wall-time per step for the *current* run (§8.3 step_time fix).

    ``elapsed`` is the seconds since the loop's ``t0``; ``step`` is the
    absolute step just reached; ``run_start`` is the absolute step the run
    began from (``resume_step`` on a resume, else 0). Dividing by the *number
    of steps this run actually executed* (``step - run_start``) — not the
    absolute ``step`` — keeps the reported per-step time honest after a
    resume, where the old absolute-``step`` divisor understated it by folding
    in steps from the prior run. ``max(1, …)`` guards the degenerate
    ``step == run_start`` first-row case.

    Pure + JAX-free so the divisor is unit-testable directly.
    """
    return elapsed / max(1, step - run_start)


def _chunk_bound(
    absolute: int,
    remaining: int,
    k: int,
    eval_interval: int,
    checkpoint_interval: int,
) -> int:
    """Size of the next host-loop chunk (the core C3 boundary-shrink).

    ``absolute`` is the number of steps already completed in this phase;
    ``remaining`` is how many are still owed. A chunk is the ``lax.scan``
    K-step unit: the scan only surfaces the *final* carry, so the chunk must
    never straddle an ``eval_interval`` / ``checkpoint_interval`` boundary —
    otherwise the host can't read the exact post-step state to eval or
    checkpoint on it. The chunk is therefore capped to ``k`` and ``remaining``
    and then shortened to land on whichever boundary is closest.

    ``eval_interval - (absolute % eval_interval)`` is the distance to the next
    eval boundary: when ``absolute`` is itself a multiple of the interval this
    is a full ``eval_interval`` (the boundary at ``absolute`` is already past),
    and otherwise it lands exactly on the next multiple. The same holds for the
    checkpoint distance. Dropping the ``- (absolute % …)`` term, or shifting
    the boundary test elsewhere, would let chunks cross boundaries and silently
    skip evals/checkpoints — so the arithmetic is pure + JAX-free here to be
    unit-testable directly.
    """
    to_eval = eval_interval - (absolute % eval_interval)
    to_ckpt = checkpoint_interval - (absolute % checkpoint_interval)
    return min(k, remaining, to_eval, to_ckpt)


class _AdapterResume(NamedTuple):
    """Result of the ``--resume`` restore branch (H3).

    ``backbone`` / ``adapter`` are the spliced PyTrees the loop trains;
    ``opt_state`` is the optimizer state to resume from. ``adapter_restored``
    records whether the trained adapter was faithfully recovered from a
    sidecar — when it is ``False`` the adapter came back cold and
    ``opt_state`` is *cold-started* (never warm-loaded from
    ``OPTIMIZER_FILE``) so warm Adam moments can't be spliced onto cold
    params, the H3 corruption C2 fixes.
    """

    backbone: PAWNModel
    adapter: Any  # one of the *Adapter types — eqx.Module
    opt_state: optax.OptState
    adapter_restored: bool


def write_adapter_checkpoint(
    *,
    effective: Any,
    backbone: PAWNModel,
    adapter: Any,
    out: "Path",
    optimizer_state: dict[str, np.ndarray],
    step: int,
    run_config: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Write an adapter checkpoint, selecting the strategy-appropriate sidecar.

    This is the single save-side branch selector that mirrors
    :func:`restore_adapter_resume_state` on the resume side. Factored out of
    ``main`` so the save path the trainer actually ships is unit-tested
    against the same code rather than re-implemented in tests (H3).

    ``effective = apply_fn(backbone, adapter)``. Three layouts:

    * ``BottleneckEffective`` — save the frozen ``backbone`` + the typed
      ``adapter.safetensors`` sidecar (the Houlsby MLP can't fold).
    * ``FiLMEffective`` — save the effective backbone (carries any folded
      LoRA corrections for hybrid) + the FiLM typed sidecar. For ``hybrid``
      (a ``HybridAdapter`` whose LoRA half folds *irreversibly* into the
      saved backbone) ALSO write the raw ``(backbone, adapter)`` resume
      sidecar so ``--resume`` restores both halves exactly — a FiLM-sidecar-
      only resume would cold-start the LoRA params under a warm optimiser
      (H3). Pure FiLM has no folded half and skips this.
    * ``PAWNModel`` — the weight-folded path (lora / sparse / unfreeze /
      specialized_clm). The folded ``model.safetensors`` is one-way (sparse
      masks aren't recoverable), so persist the raw ``(backbone, adapter)``
      resume sidecar; the warm Adam moments index exactly that split.
    """
    from pawn.adapters.bottleneck import (
        BottleneckEffective,
        save_bottleneck_adapter,
    )
    from pawn.adapters.film import FiLMEffective, save_film_adapter
    from pawn.adapters.hybrid import HybridAdapter
    from pawn.checkpoint import save_adapter_resume_state

    # H7: the caller may pass a richer training_state (scheduler + RNG
    # blocks) so a resume can restore the data-stream RNG. Default to the
    # bare step counter when omitted (back-compat with prior callers/tests).
    if training_state is None:
        training_state = {"step": int(step)}
    if isinstance(effective, BottleneckEffective):
        save_model(
            effective.backbone, out,
            run_config=run_config,
            optimizer_state=optimizer_state,
            training_state=training_state,
        )
        save_bottleneck_adapter(effective.adapter, out)
    elif isinstance(effective, FiLMEffective):
        save_model(
            effective.backbone, out,
            run_config=run_config,
            optimizer_state=optimizer_state,
            training_state=training_state,
        )
        save_film_adapter(effective.adapter, out)
        if isinstance(adapter, HybridAdapter):
            save_adapter_resume_state(backbone, adapter, out)
    else:
        # PAWNModel — the standard weight-folded save path. The narrowing
        # assertion satisfies pyright: the only apply_fn return types in v2
        # are PAWNModel / BottleneckEffective / FiLMEffective; the branches
        # above peeled off the wrappers, leaving the PAWNModel case here.
        assert isinstance(effective, PAWNModel), (
            f"unexpected effective type {type(effective).__name__}; "
            "extend the save dispatch in write_adapter_checkpoint"
        )
        save_model(
            effective, out,
            run_config=run_config,
            optimizer_state=optimizer_state,
            training_state=training_state,
        )
        save_adapter_resume_state(backbone, adapter, out)


def restore_adapter_resume_state(
    *,
    strategy: str,
    strategy_cfg: object,
    ckpt_dir: "Path",
    loaded_backbone: PAWNModel,
    cold_backbone: PAWNModel,
    cold_adapter: Any,
    optimizer: optax.GradientTransformation,
) -> _AdapterResume:
    """Select the resume restore branch and decide warm-vs-cold opt_state.

    Factored out of ``main`` so the branch selection + the H3 cold-start
    fallback are unit-testable in isolation. Three sidecar families restore
    the trained adapter faithfully; absent the expected sidecar the adapter
    can only come back cold, and then ``opt_state`` is cold-started so warm
    Adam moments never index cold params.

    Inputs:
      * ``loaded_backbone`` — the backbone read from ``model.safetensors``
        (folded effective for weight-folding adapters; raw frozen backbone
        for the wrapper adapters).
      * ``cold_backbone`` / ``cold_adapter`` — freshly cold-built templates
        carrying the exact PyTree structure for the resume-sidecar
        deserialise.
    """
    from pawn.adapters.bottleneck import (
        ADAPTER_SAFETENSORS,
        BottleneckConfig,
        load_bottleneck_adapter,
    )
    from pawn.adapters.film import FiLMConfig, load_film_adapter
    from pawn.checkpoint import (
        ADAPTER_RESUME_FILE,
        OPTIMIZER_FILE,
        load_adapter_resume_state,
    )
    from pawn.trainer import unflatten_opt_state

    typed_sidecar = ckpt_dir / ADAPTER_SAFETENSORS
    resume_sidecar = ckpt_dir / ADAPTER_RESUME_FILE
    adapter_fully_restored = False
    if typed_sidecar.is_file() and isinstance(strategy_cfg, BottleneckConfig):
        # Bottleneck: the whole adapter lives in the typed sidecar; the
        # saved `model.safetensors` is the untouched frozen backbone
        # (wrappers don't fold). Full restore.
        backbone = loaded_backbone
        adapter: Any = load_bottleneck_adapter(ckpt_dir, strategy_cfg)
        adapter_fully_restored = True
    elif typed_sidecar.is_file() and isinstance(strategy_cfg, FiLMConfig):
        # FiLM: the entire trained adapter (gamma/beta + optional output
        # FiLM) lives in the typed sidecar; the saved `model.safetensors`
        # is the raw frozen backbone. Full restore.
        backbone = loaded_backbone
        adapter = load_film_adapter(ckpt_dir, strategy_cfg)
        adapter_fully_restored = True
    elif resume_sidecar.is_file():
        # Weight-folding adapters: the folded `model.safetensors` is
        # one-way (sparse masks aren't recoverable from it), so restore the
        # raw (backbone, adapter) split from the resume sidecar. The
        # freshly-built `backbone` / `adapter` templates carry the matching
        # PyTree structure; the deserialise overwrites every leaf
        # (including sparse masks and the specialized_clm standalone model)
        # with the saved value. Full restore.
        backbone, adapter = load_adapter_resume_state(
            cold_backbone, cold_adapter, ckpt_dir
        )
        adapter_fully_restored = True
    else:
        # No usable sidecar — fall back to the folded backbone so the
        # forward at least runs; the cold adapter + cold opt_state below
        # keep H3 safe.
        backbone = loaded_backbone
        adapter = cold_adapter
    # Build the optimizer template *after* the adapter rebuild so the
    # opt-state shape matches what the restored adapter trains.
    flt = dispatch_filter(strategy)(adapter)
    opt_state: optax.OptState = optimizer.init(eqx.filter(adapter, flt))
    opt_path = ckpt_dir / OPTIMIZER_FILE
    if not adapter_fully_restored:
        # No usable sidecar for this strategy — the adapter came back
        # cold. Never splice warm Adam moments onto cold params; that
        # corrupts the first post-resume step (H3). Cold-start opt_state.
        print(
            f"[train_jax_adapter] adapter for strategy {strategy!r} "
            "could not be fully restored from the checkpoint (no resume "
            "sidecar); cold-starting opt_state to avoid a warm/cold "
            "mismatch.",
            file=sys.stderr,
        )
    elif opt_path.is_file():
        from safetensors.numpy import load_file as st_load
        flat = st_load(str(opt_path))
        opt_state = unflatten_opt_state(opt_state, flat)
    else:
        print(
            f"[train_jax_adapter] WARNING: no {OPTIMIZER_FILE} in "
            f"{ckpt_dir}; resuming with fresh opt_state.",
            file=sys.stderr,
        )
    return _AdapterResume(
        backbone=backbone,
        adapter=adapter,
        opt_state=opt_state,
        adapter_restored=adapter_fully_restored,
    )


def _strategy_config_from_run(cfg: AdapterConfig) -> object:
    """Build the adapter's strategy Config from the AdapterConfig fields.

    ``--adapter-layers`` (``cfg.adapter_layers``) is parsed once into the
    validated layer-index tuple and threaded into every placement-aware
    strategy (lora / bottleneck / sparse / hybrid / rosa). ``None`` (the
    default) means "every layer". The bound check is deferred to the
    adapter init (which knows ``n_layers``); here we only parse the syntax.
    """
    from pawn.adapters.placement import parse_adapter_layers

    s = cfg.strategy
    # Bound-checking against the real n_layers happens in the adapter init:
    # every placement-aware init routes ``cfg.layers`` through
    # ``layer_placement_mask(layers, backbone.cfg.n_layers)``, which raises
    # ``ValueError`` on any index >= n_layers (a silent out-of-bounds
    # scatter would otherwise drop the write and adapt nothing). At
    # config-build time the backbone isn't loaded, so we only validate the
    # *syntax* here against a permissive sentinel.
    layers = (
        parse_adapter_layers(cfg.adapter_layers, n_layers=1 << 30)
        if cfg.adapter_layers is not None
        else None
    )
    if s == "lora":
        return LoRAConfig(
            rank=cfg.lora_rank or 4,
            targets=cfg.lora_targets or "qkvo",
            ffn=cfg.lora_ffn,
            layers=layers,
        )
    if s == "film":
        return FiLMConfig(use_output_film=cfg.use_output_film)
    if s == "bottleneck":
        return BottleneckConfig(
            dim=cfg.bottleneck_dim or 8,
            n_hidden=cfg.bottleneck_n_hidden,
            no_adapt_attn=cfg.no_adapt_attn,
            no_adapt_ffn=cfg.no_adapt_ffn,
            layers=layers,
        )
    if s == "hybrid":
        return HybridConfig(
            lora=LoRAConfig(
                rank=cfg.lora_rank or 4,
                targets=cfg.lora_targets or "qkvo",
                ffn=cfg.lora_ffn,
                layers=layers,
            ),
            film=FiLMConfig(use_output_film=cfg.use_output_film),
        )
    if s == "sparse":
        return SparseConfig(
            density=cfg.density or 0.01,
            targets=cfg.sparse_targets or "qkvo",
            ffn=cfg.sparse_ffn,
            layers=layers,
        )
    if s in ("rosa", "rosa-retro-sparse", "rosa-retro-bottleneck"):
        # The three RoSA-family strategies share init/apply; the mode is
        # what differs. Default `mode` to the strategy name suffix when
        # the user uses --strategy directly; honour an explicit
        # --rosa-mode override.
        from typing import cast

        from pawn.adapters.rosa import RoSAMode
        suffix_to_mode: dict[str, RoSAMode] = {
            "rosa": "rosa",
            "rosa-retro-sparse": "retro-sparse",
            "rosa-retro-bottleneck": "retro-bottleneck",
        }
        mode: RoSAMode = (
            cast(RoSAMode, cfg.rosa_mode) if cfg.rosa_mode is not None
            else suffix_to_mode[s]
        )
        rosa_kwargs: dict[str, Any] = dict(
            mode=mode,
            lora_rank=cfg.lora_rank or 4,
            density=cfg.density or 0.01,
            rosa_warmup_steps=cfg.rosa_warmup_steps,
            mask_samples=cfg.mask_samples,
            grad_alpha=cfg.grad_alpha,
            lora_targets=cfg.lora_targets or "qkvo",
            lora_ffn=cfg.lora_ffn,
            sparse_targets=cfg.sparse_targets or "qkvo",
            sparse_ffn=cfg.sparse_ffn,
            layers=layers,
        )
        # `bottleneck_dim` is the Houlsby width used by the `retro-bottleneck`
        # RoSA sub-mode (unused for `rosa` / `retro-sparse`). Thread it through
        # only when set so RoSAConfig's own default (8) stands otherwise. The
        # `rosa-ratio` sweep (H9) drives this knob — it maps onto the `rosa`
        # strategy in `retro-bottleneck` mode and translates its swept ratio
        # into this field, so it must be consumed here rather than dropped.
        if cfg.bottleneck_dim is not None:
            rosa_kwargs["bottleneck_dim"] = cfg.bottleneck_dim
        return RoSAConfig(**rosa_kwargs)
    if s == "unfreeze":
        return UnfreezeConfig(layers=cfg.unfreeze_layers or "5,6,7")
    if s == "specialized_clm":
        return AdapterSpecializedCLMConfig(
            d_model=cfg.d_model or 64,
            n_layers=cfg.n_layers or 2,
            n_heads=cfg.n_heads or 2,
            d_ff=cfg.d_ff or 256,
        )
    raise ValueError(f"unknown strategy {s!r}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="train_jax_adapter")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--strategy", required=False)
    ap.add_argument("--supernet", choices=("tiny", "production"), default=None)
    ap.add_argument("--variant", choices=("small", "base", "large"), default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pgn", default=None)
    ap.add_argument("--pgn-val-split", default=None)
    ap.add_argument("--elo-min", type=int, default=None)
    ap.add_argument("--elo-max", type=int, default=None)
    ap.add_argument("--min-ply", type=int, default=None)
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-targets", default=None)
    # v1 CLI-compat: route LoRA into the FFN sublayers too (not just
    # attention projections). Was reachable only via --config JSON in v2.
    ap.add_argument("--lora-ffn", action="store_true",
                    help="apply LoRA to the FFN sublayers in addition to "
                         "the attention projections (lora_targets).")
    ap.add_argument("--density", type=float, default=None)
    # v1 CLI-compat: which attention projections the sparse mask targets.
    ap.add_argument("--sparse-targets", choices=("qkvo", "qv", "qkv"),
                    default=None,
                    help="attention projections the sparse adapter masks "
                         "(default qkvo). Mirrors --lora-targets.")
    # v1 CLI-compat: extend the sparse mask to the FFN sublayers too.
    ap.add_argument("--sparse-ffn", action="store_true",
                    help="apply the sparse mask to the FFN sublayers in "
                         "addition to the attention projections.")
    ap.add_argument("--bottleneck-dim", type=int, default=None)
    # v1 CLI-compat: extra hidden Linear+GELU stages inside each Houlsby
    # adapter MLP (0 = standard two-layer block).
    ap.add_argument("--bottleneck-n-hidden", type=int, default=None,
                    help="extra hidden Linear+GELU stages inside each "
                         "Houlsby bottleneck adapter MLP (0 = standard "
                         "two-layer block).")
    ap.add_argument("--use-output-film", action="store_true")
    ap.add_argument("--no-adapt-attn", action="store_true")
    ap.add_argument("--no-adapt-ffn", action="store_true")
    ap.add_argument("--rosa-mode", default=None)
    ap.add_argument("--adapter-layers", default=None,
                    help="restrict the adapter (lora/bottleneck/sparse/"
                         "hybrid/rosa) to an explicit comma-separated subset "
                         "of transformer layers, e.g. '5,6,7'. Default: all "
                         "layers. Mirrors v1 per-layer placement.")
    ap.add_argument("--unfreeze-layers", default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--no-pgn", action="store_true",
                    help="use random-game corpus instead of Lichess parquet")
    ap.add_argument("--local-checkpoints", action="store_true")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from an adapter_step_<N> checkpoint dir. "
                         "Bottleneck and FiLM adapters automatically detect "
                         "the adapter.safetensors sidecar.")
    ap.add_argument("--use-sdpa", action="store_true",
                    help="opt the attention block into "
                         "jax.nn.dot_product_attention (parity #43). "
                         "Off by default — superseded by Pallas flash on GPU.")
    ap.add_argument("--no-flash", action="store_true",
                    help="disable the Pallas flash-attention kernel "
                         "(force the plain materialised QK^T path). "
                         "Flash is the default on GPU; CPU runs "
                         "auto-fall-back regardless of this flag.")
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--wandb", action="store_true",
                    help="enable the Weights & Biases metric mirror "
                         "(requires --extra wandb). Hard-errors if the "
                         "extra isn't installed.")
    ap.add_argument("--log-interval", type=int, default=None,
                    help="Steps between metrics rows; defaults to the "
                         "BaseRunConfig log_interval (100)")
    return ap.parse_args(argv)


def _build_config(args: argparse.Namespace) -> AdapterConfig:
    base: dict = {"run_type": "adapter"}
    if args.config is not None:
        base.update(json.loads(args.config.read_text()))
        base.setdefault("run_type", "adapter")
    # `store_true` flags whose default-False would clobber a JSON-set
    # True (round-1 codex P2: `--use-sdpa` defaulted to False, was
    # `not None`, and silently overrode a JSON `"use_sdpa": true`).
    # These are AdapterConfig fields, so they round-trip through the
    # pydantic config — but we only merge from CLI when actually set.
    # `--wandb` is now registered on the adapter parser too (H7: wandb
    # wired into *both* entry points), so it joins the opt-in store_true
    # set below rather than being silently dropped.
    _CLI_STORE_TRUE_FLAGS = ("use_sdpa", "use_output_film", "no_adapt_attn",
                             "no_adapt_ffn", "wandb", "lora_ffn", "sparse_ffn")
    for flag, val in vars(args).items():
        if flag in (
            "config", "no_pgn", "local_checkpoints", "logs_dir", "resume",
            "no_flash",
            *_CLI_STORE_TRUE_FLAGS,
        ):
            # `resume` is operated on directly from `args` (Path, not a
            # pydantic-validated string). `no_pgn` and
            # `local_checkpoints` are consumed in main(), not in the
            # config. The store_true flags need explicit opt-in
            # handling so an absent CLI flag doesn't override a JSON
            # True.
            continue
        if val is None:
            continue
        base[flag] = val
    # Opt-in handling for store_true flags: only merge into the config
    # when the user actually passed them (val is True).
    for flag in _CLI_STORE_TRUE_FLAGS:
        if getattr(args, flag, False):
            base[flag] = True
    if args.no_flash:
        # `--no-flash` is the explicit opt-out; default `use_flash`
        # is True in the run config so an absent CLI flag must not
        # set the value either way.
        base["use_flash"] = False
    if args.local_checkpoints:
        base["local_checkpoints"] = True
    return AdapterConfig(**base)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = _build_config(args)
    if cfg.total_steps is None:
        print("error: --total-steps is required", file=sys.stderr)
        return 2
    if cfg.strategy not in STRATEGIES:
        print(f"error: unknown strategy {cfg.strategy!r}", file=sys.stderr)
        return 2

    # Early --resume validation runs *before* any JAX device init /
    # HF checkpoint download / local model load so guard failures
    # don't waste bandwidth or compile time. Round-3 bug-detector
    # MINOR: the prior order made the RoSA-resume test depend on HF
    # network state in CI. This block is pure argument validation
    # (read a JSON sidecar, check a step counter) and touches no JAX
    # device — it deliberately runs *ahead* of `require_accelerator()`
    # and `setup_jax_caching()` so that on accelerator backends prone
    # to a flaky device-init segfault (JAX-on-ROCm/WSL2), the guard
    # message is still emitted deterministically rather than being lost
    # when the runtime dies during device init. We only peek at the
    # resume-specific sidecar files; full restore happens further down
    # once the backbone is in hand.
    is_rosa_strategy = cfg.strategy in (
        "rosa", "rosa-retro-sparse", "rosa-retro-bottleneck",
    )
    resume_step_early = 0
    if args.resume is not None:
        _resume_dir = Path(args.resume)
        _ts_path = _resume_dir / "training_state.json"
        if not _ts_path.is_file():
            raise SystemExit(
                f"[train_jax_adapter] --resume requires "
                f"{_ts_path.name} in the checkpoint dir; got "
                f"{_resume_dir} with no such file. The training-state "
                "sidecar carries the saved step counter and is "
                "load-bearing for the resume contract."
            )
        _ts_data = json.loads(_ts_path.read_text(encoding="utf-8"))
        resume_step_early = int(_ts_data.get("step", 0))
        if is_rosa_strategy and resume_step_early > 0:
            raise SystemExit(
                "[train_jax_adapter] --resume is not supported for RoSA "
                "strategies: Phase 2 mask-generation state is not "
                "persisted in the checkpoint, so re-running from a "
                "Phase 1/2/3 boundary would either re-execute completed "
                "work (exceeding total_steps) or apply stale masks. "
                "Restart the run from step 0 instead."
            )

    require_accelerator()
    cache_path = setup_jax_caching()
    if cache_path is not None:
        print(f"JAX compilation cache: {cache_path}")

    # Build / load backbone.
    if cfg.strategy == "specialized_clm":
        # Standalone — no backbone needed.
        backbone = init_model(TINY_SUPERNET, key=0)  # placeholder
    else:
        # v2-only path. `resolve_checkpoint_source` handles local dirs +
        # HF repo IDs uniformly; v1 PyTorch repos are no longer loadable
        # (the legacy converter was removed in the H.2 housekeeping commit).
        # Users wanting v1 artifacts check out the `v1.0.0` git tag.
        from pawn.checkpoint import load_model
        from pawn.corpus import (
            assert_conditioning_C,
            conditioning_from_run_block,
            conditioning_to_C,
        )
        ckpt_dir = resolve_checkpoint_source(cfg.checkpoint)
        backbone, backbone_run_block = load_model(ckpt_dir)
        # Load-time C-mismatch guard (Phase-A spec Chunk 4): the corpus is
        # built below with `cfg.conditioning`, so a backbone trained under a
        # different conditioning width would place every move at a different
        # absolute RoPE offset (silent drift). Cross-check against the
        # backbone's persisted conditioning and fail loudly on a mismatch.
        backbone_C = conditioning_to_C(
            conditioning_from_run_block(backbone_run_block)
        )
        assert_conditioning_C(cfg.conditioning, backbone_C)
        # When the loaded model has the supernet's depth we slice into a
        # variant; otherwise treat it as standalone (e.g., a previously
        # published from-scratch CLM run).
        target_supernet = TINY_SUPERNET if cfg.supernet == "tiny" else SUPERNET
        looks_like_supernet = backbone.cfg.n_layers == target_supernet.n_layers
        if cfg.variant != "large" and looks_like_supernet:
            variant_cfg = (
                TINY_VARIANTS[cfg.variant]
                if cfg.supernet == "tiny"
                else VARIANTS[cfg.variant]
            )
            backbone = sliced(backbone, variant_cfg)

    # Build adapter.
    strategy_cfg = _strategy_config_from_run(cfg)
    init = dispatch_init(cfg.strategy)
    adapter = init(backbone, strategy_cfg, key=jax.random.key(0))

    # Optimizer over the adapter only.
    schedule = make_lr_schedule(cfg, cfg.total_steps)
    optimizer = make_optimizer(cfg, schedule)

    # Resume: splice (backbone, adapter, opt_state, step) from a checkpoint.
    #
    # The warm optimizer state (Adam moments + clip counter) only stays
    # consistent if the adapter PyTree it indexes is restored to the *exact*
    # params it was trained against — applying warm moments to cold-started
    # params corrupts the first post-resume step (H3). Two sidecar families
    # carry the trained adapter:
    #
    #   * Wrapper adapters (bottleneck / FiLM) write the *raw* frozen backbone
    #     as `model.safetensors` plus a typed `adapter.safetensors` holding
    #     the whole trained adapter. Load the backbone + the typed sidecar →
    #     full restore.
    #   * Weight-folding adapters (lora / sparse / unfreeze / specialized_clm
    #     / hybrid) publish the *folded* effective model as `model.safetensors`
    #     (one-way; sparse masks aren't even recoverable from it), and write
    #     the raw `(backbone, adapter)` PyTree to
    #     `adapter_resume_state.eqx`. Load that sidecar → exact (backbone,
    #     adapter) restore, so the warm moments index the same params.
    #
    # When the expected sidecar is absent (an older checkpoint, or a manual
    # `model.safetensors`-only drop), the adapter can't be faithfully restored;
    # we cold-start opt_state so warm moments never index cold params.
    #
    # `resume_step` is already extracted upstream (right after
    # `require_accelerator`) so the RoSA / training-state checks fire before
    # any HF download.
    resume_step = resume_step_early
    if args.resume is not None:
        from pawn.checkpoint import load_model
        from pawn.corpus import (
            assert_conditioning_C,
            conditioning_from_run_block,
            conditioning_to_C,
        )

        ckpt_dir = Path(args.resume)
        # `backbone` / `adapter` entering here are the freshly-built templates
        # (cold init against `cfg`). They carry the exact PyTree structure the
        # save-time `state.backbone` / `state.adapter` had, so they double as
        # deserialise templates for the resume sidecar. Read the folded
        # `model.safetensors` separately for the run block + the
        # wrapper-adapter (bottleneck/FiLM) frozen-backbone restore.
        loaded_backbone, backbone_run_block = load_model(ckpt_dir)
        # Same load-time C-mismatch guard as the initial-load path: the
        # corpus below is built with `cfg.conditioning`, so the resumed
        # backbone's persisted conditioning must agree or moves drift.
        resume_C = conditioning_to_C(
            conditioning_from_run_block(backbone_run_block)
        )
        assert_conditioning_C(cfg.conditioning, resume_C)

        # Branch selection + H3 cold-start fallback are factored into
        # `restore_adapter_resume_state` so they can be unit-tested.
        resumed = restore_adapter_resume_state(
            strategy=cfg.strategy,
            strategy_cfg=strategy_cfg,
            ckpt_dir=ckpt_dir,
            loaded_backbone=loaded_backbone,
            cold_backbone=backbone,
            cold_adapter=adapter,
            optimizer=optimizer,
        )
        backbone = resumed.backbone
        adapter = resumed.adapter
        opt_state = resumed.opt_state
    else:
        flt = dispatch_filter(cfg.strategy)(adapter)
        opt_state = optimizer.init(eqx.filter(adapter, flt))
    state = AdapterTrainState(
        backbone=backbone, adapter=adapter, opt_state=opt_state,
        step=jnp.int32(resume_step), key=jax.random.key(0),
    )
    _DTYPE_MAP = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": None,
    }
    compute_dtype = _DTYPE_MAP[cfg.amp_dtype]
    # Pallas flash requires the GPU backend; CPU smoke runs
    # (`PAWN_ALLOW_CPU=1`) auto-fall-back to plain attention.
    use_flash = cfg.use_flash and jax.default_backend() == "gpu"
    # Adapter-loss legality (v1 ``apply_legal_mask`` default-ON):
    # ``disable_legal_mask`` flips the hard mask off; ``illegal_penalty``
    # adds the soft penalty term (only meaningful with the hard mask off,
    # enforced by AdapterConfig._check_legality_flags). Both need a
    # per-position legal mask attached to each batch.
    apply_legal = not cfg.disable_legal_mask
    illegal_penalty = cfg.illegal_penalty
    need_legal = apply_legal or illegal_penalty > 0.0
    train_step = make_adapter_train_step(
        cfg.strategy, optimizer,
        compute_dtype=compute_dtype, use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        apply_legal=apply_legal, illegal_penalty=illegal_penalty,
    )
    # H11 (§8.3): drive the train loop through the K-step ``lax.scan`` so the
    # whole inner chunk is one compiled program (no per-step dispatch). The
    # single-step ``train_step`` is kept only as the scan body; eval uses the
    # forward-only ``val_metrics_fn`` below.
    scan_step = make_adapter_scan_step(train_step)

    # Richer val metrics (v1 parity: loss / top1 / top5 / illegal_pred_rate).
    val_metrics_fn = make_adapter_val_metrics(
        cfg.strategy, apply_legal=apply_legal, illegal_penalty=illegal_penalty,
    )

    apply_fn = STRATEGIES[cfg.strategy].apply

    # Data: Lichess corpus or random games. The val_corpus is the held-out
    # `validation` split for PGN (cfg.pgn_val_split, default "validation")
    # or a fresh-seed random corpus when --no-pgn is set.
    if args.no_pgn:
        corpus = generate_corpus(
            n_games=max(cfg.batch_size * 10, 1000),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=0,
            conditioning=cfg.conditioning,
        )
        val_corpus = generate_corpus(
            n_games=max(cfg.batch_size * 4, 100),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=1,
            conditioning=cfg.conditioning,
        )
    else:
        corpus = load_lichess_corpus(
            cfg.pgn,
            split="train",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max,
            min_ply=cfg.min_ply,
            seq_len=cfg.seq_len,
            max_games=getattr(cfg, "max_games", None),
            conditioning=cfg.conditioning,
        )
        val_corpus = load_lichess_corpus(
            cfg.pgn,
            split=cfg.pgn_val_split or "validation",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max,
            min_ply=cfg.min_ply,
            seq_len=cfg.seq_len,
            max_games=getattr(cfg, "val_games", None),
            conditioning=cfg.conditioning,
        )

    logger = MetricsLogger(
        log_dir=args.logs_dir, run_prefix=f"adapter-{cfg.strategy}",
        device=resolve_device(), suffix=cfg.variant,
    )
    logger.log_config(run_type="adapter", config=cfg.model_dump())

    # W&B mirror — gated on `--wandb` (cfg.wandb) *and* the `wandb` extra.
    # `--wandb` without the extra is a hard error (H7: wandb wired into
    # both entry points; no silent metric drop).
    wandb_run = None
    if cfg.wandb:
        require_wandb_available()
        wandb_run = init_wandb(
            project=cfg.wandb_project, slug=logger.slug,
            run_config=cfg.model_dump(),
            git_hash=get_git_info().get("git_hash"),
            job_type="adapter",
            run_dir_name=logger.run_dir.name,
        )

    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None
    should_shutdown = install_sigterm_handler()

    # Steps already persisted this run. A best-val eval (`_on_val`) and the
    # run-end final-save can both target the same `adapter_step_<final>` path
    # when the final step improves val_loss and `total_steps <
    # checkpoint_interval` (the common short-run / smoke case). `save_model`
    # refuses to overwrite an existing checkpoint, so `_save` is idempotent
    # per step: a second request for an already-written step is a no-op
    # rather than a `FileExistsError`.
    saved_steps: set[int] = set()

    def _save(step_int: int) -> None:
        """Save the *effective* model — `apply_fn(backbone, adapter)` —
        not the frozen backbone. The whole point of adapter training is
        that `state.adapter` holds the trained delta; saving the
        backbone discards it. The effective model loads cleanly via
        `pawn.checkpoint.load_model` and downstream eval scripts treat
        it as an ordinary published checkpoint.

        Also persists `optimizer.safetensors` (Adam moments + clip
        counter) via `flatten_opt_state` so an adapter `--resume` doesn't
        cold-start the optimiser — mirrors the pretrain save path added
        in commit 92d618b for PR review #1, and addresses the round-1
        review-bug-detector finding that adapter resumes were emitting
        the "no optimizer.safetensors" warning every time.

        Bottleneck-style adapters (and RoSA's ``retro-bottleneck`` mode)
        return a wrapper module rather than a folded :class:`PAWNModel`,
        because their Houlsby residual MLP has a GELU nonlinearity that
        can't collapse into the backbone's weight tensors. For these
        the save path writes the frozen backbone as the main checkpoint
        and dumps the adapter weights as ``adapter.safetensors`` in the
        same directory — :func:`pawn.adapters.bottleneck.load_bottleneck_adapter`
        re-composes the wrapper at resume time.

        Checkpoints land under the MetricsLogger's per-run directory so
        two concurrent runs can't collide on the same path.
        """
        if step_int in saved_steps:
            # Already written this run (e.g. a best-val save at the final
            # step that the run-end final-save would otherwise re-target).
            return
        effective = apply_fn(state.backbone, state.adapter)
        out = logger.run_dir / f"adapter_step_{step_int:08d}"
        # H7: persist the scheduler identity + the JAX key + both numpy
        # data-stream RNGs (train + val) so a `--resume` continues the exact
        # same batch-index sequence rather than replaying from the seed.
        training_state = build_training_state(
            step=int(state.step),
            schedule=cfg.lr_schedule,
            lr_peak=cfg.lr,
            rng_key=state.key,
            numpy_rngs={"train": rng, "val": val_rng},
        )
        # Single save-side branch selector, shared with the resume tests so
        # the shipped sidecar selection (incl. the hybrid resume-sidecar
        # guard) is unit-tested rather than re-implemented (H3).
        write_adapter_checkpoint(
            effective=effective,
            backbone=state.backbone,
            adapter=state.adapter,
            out=out,
            optimizer_state=flatten_opt_state(state.opt_state),
            step=int(state.step),
            run_config=cfg.model_dump(),
            training_state=training_state,
        )
        saved_steps.add(step_int)
        if push_tracker:
            push_checkpoint_async(out, push_tracker)

    # B2 / plan §8.3: consume the previously-inert cadence/sampling knobs
    # (``epochs`` / ``steps_per_epoch`` / ``data_seed`` / ``val_every``) via
    # the pure :func:`_resolve_cadence` helper now that the corpus has
    # materialised (``steps_per_epoch="all"`` needs ``corpus.n_games``).
    # ``data_seed`` seeds the train sampler; the held-out val stream uses
    # ``data_seed + 1`` to stay decorrelated. The LR schedule keeps
    # ``cfg.total_steps`` as its decay timeline; a resolved budget that
    # differs is exactly the ``actual ≠ planned`` case
    # ``schedule_health.json`` records.
    cadence = _resolve_cadence(cfg, corpus.n_games)
    effective_total_steps = cadence.effective_total_steps
    eval_interval = cadence.eval_interval
    rng = np.random.default_rng(cadence.data_seed)
    val_rng = np.random.default_rng(cadence.data_seed + 1)
    # H7: on resume, restore the persisted data-stream RNG state so the
    # resumed run continues the *same* batch-index sequence rather than
    # replaying from the seed (which would re-train on already-seen batches).
    # RoSA rejects --resume upstream, so this only affects the non-RoSA path.
    if args.resume is not None:
        _resume_key, _resume_np = read_resume_rng_blocks(Path(args.resume))
        if "train" in _resume_np:
            rng = _resume_np["train"]
        if "val" in _resume_np:
            val_rng = _resume_np["val"]
        if _resume_key is not None:
            state = AdapterTrainState(
                backbone=state.backbone, adapter=state.adapter,
                opt_state=state.opt_state, step=state.step, key=_resume_key,
            )
    t0 = time.time()
    final_step = 0
    is_rosa = is_rosa_strategy
    # H11 (§8.3): ``run_start`` is the absolute step the run began from —
    # ``resume_step`` for a resumed non-RoSA run, 0 otherwise (RoSA rejects
    # ``--resume`` upstream). The ``step_time`` divisor must be
    # ``(step - run_start)`` so a resume reports the per-step wall time of
    # the *current* run rather than dividing this run's elapsed time by the
    # absolute step counter (which would understate it after a resume).
    run_start = resume_step

    # Patience / best-val checkpoint selection (v1 parity). ``best`` holds
    # the lowest val_loss seen and the step it occurred at; ``counter`` is
    # the consecutive-no-improvement eval count; ``stop`` is the
    # early-stopping flag the chunk loop polls. ``patience`` (None ⇒
    # disabled) is the v1 patience budget. The best-val step is also written
    # to ``best_step.json`` at run end so downstream consumers can recover
    # it without re-scanning ``metrics.jsonl``.
    patience_state: dict[str, Any] = {
        "best_loss": float("inf"),
        "best_step": None,
        "counter": 0,
        "stop": False,
    }

    def _on_val(val_loss: float, step: int) -> None:
        """Update the patience / best-val tracker after an eval.

        A strictly-lower val_loss resets the patience counter, records the
        new best step, and persists a best-val checkpoint immediately so it
        survives even if it doesn't land on a ``checkpoint_interval``
        boundary (v1 best-checkpoint persistence). No improvement increments
        the counter; reaching ``cfg.patience`` consecutive misses sets the
        early-stop flag the chunk loop polls.
        """
        if np.isfinite(val_loss) and val_loss < patience_state["best_loss"]:
            patience_state["best_loss"] = val_loss
            patience_state["best_step"] = step
            patience_state["counter"] = 0
            # Persist the best so it's never lost to a non-boundary eval.
            if step % cfg.checkpoint_interval != 0:
                _save(step)
        else:
            patience_state["counter"] += 1
            if (
                cfg.patience is not None
                and patience_state["counter"] >= cfg.patience
            ):
                patience_state["stop"] = True
                print(
                    f"[train_jax_adapter] early stopping at step {step} "
                    f"(patience={cfg.patience}, best val_loss="
                    f"{patience_state['best_loss']:.4f} @ step "
                    f"{patience_state['best_step']})",
                    file=sys.stderr,
                )

    def _legal_for(src: Corpus, idx: np.ndarray) -> jax.Array | None:
        """Engine-replayed ``(..., V)`` legal mask for ``idx`` games in
        ``src``, or ``None`` when legality is disabled and the penalty is
        off (the random-game / no-legality-knob path). ``idx`` may be
        ``(B,)`` or ``(n, B)`` — the leading axes are preserved."""
        if not need_legal:
            return None
        flat = np.asarray(idx).reshape(-1)
        lm = legal_mask_for_games(src, flat)  # (flat, T, V)
        lm = lm.reshape(idx.shape + lm.shape[1:])
        return jnp.asarray(lm)

    def _gather_chunk(n: int) -> Batch:
        """Pre-gather ``n`` train batches into one ``(n, B, T)`` Batch.

        Each scan element is a ``(B, T)`` batch, so the leading axis is the
        K-step axis the :func:`make_adapter_scan_step` ``lax.scan`` iterates.
        Sampling ``(n, B)`` game indices in one draw keeps the train stream's
        RNG sequence identical to the single-step loop (``rng`` is consumed
        in the same order). When legality is active the engine replays each
        game to attach the per-position ``legal_mask`` (v1 ``apply_legal_mask``
        parity)."""
        idx = rng.integers(0, corpus.n_games, size=(n, cfg.batch_size))
        return Batch(
            tokens=jnp.asarray(corpus.tokens[idx]),
            targets=jnp.asarray(corpus.targets[idx]),
            attn_mask=jnp.asarray(corpus.attn_mask[idx]),
            loss_mask=jnp.asarray(corpus.loss_mask[idx]),
            legal_mask=_legal_for(corpus, idx),
        )

    def _run_steps(
        scan_fn: object,  # eqx-jitted K-step lax.scan closure
        n_steps: int,
        start_step: int,
    ) -> tuple[AdapterTrainState, int]:
        """Drive ``n_steps`` adapter steps through the K-step ``lax.scan``
        (H11). Used by both the non-RoSA path and per-phase RoSA
        orchestration. Returns ``(new_state, last_step_done)`` where
        ``last_step_done`` is the absolute step counter the loop reached so
        the outer scope can drive checkpoints and resumes.

        Each chunk is capped to ``cfg.k`` and shortened so it never crosses
        an ``eval_interval`` / ``checkpoint_interval`` boundary — the scan
        only surfaces the *final* carry state, so eval (forward-only,
        single-step) and checkpointing run on the host at the exact boundary
        step. Per-step training losses come back as a ``(chunk,)`` array and
        are replayed at each ``log_interval`` boundary without a per-step
        host sync.

        ``state`` is the enclosing-scope carry (declared ``nonlocal``), not a
        parameter: the chunk loop advances it in place each iteration so the
        in-loop checkpoint path and ``_save`` — which read ``state.adapter`` /
        ``state.opt_state`` / ``state.step`` as free variables off that same
        cell — observe the *post-step* state at every boundary. Before this
        was wired through the enclosing binding, an intermediate checkpoint
        (``total_steps > checkpoint_interval``) saved the cold-init adapter /
        previous-phase carry with a stale step counter (OBS-R3-1)."""
        nonlocal final_step, state
        done = 0
        while done < n_steps:
            absolute = start_step + done  # steps already completed this phase
            remaining = n_steps - done
            # Shrink the chunk so its trailing edge lands on the next
            # eval / checkpoint boundary (whichever is closest) — that's
            # where the host needs the exact post-step state.
            chunk = _chunk_bound(
                absolute,
                remaining,
                cfg.k,
                eval_interval,
                cfg.checkpoint_interval,
            )
            batches = _gather_chunk(chunk)
            state, losses = scan_fn(state, batches)  # type: ignore[operator]
            losses_np = np.asarray(losses)
            chunk_start = absolute  # absolute step *before* this chunk
            for i in range(chunk):
                step = chunk_start + i + 1
                if step % cfg.log_interval == 0:
                    train_metrics = dict(
                        loss=float(losses_np[i]),
                        lr=np.asarray(schedule(step)).item(),
                        step_time=_step_time(
                            time.time() - t0, step, run_start
                        ),
                    )
                    logger.log_train(step=step, **train_metrics)
                    log_metrics(wandb_run, train_metrics, step=step)
            done += chunk
            final_step = start_step + done
            if final_step % eval_interval == 0:
                val_idx = val_rng.integers(
                    0, val_corpus.n_games, size=cfg.batch_size
                )
                val_batch = Batch(
                    tokens=jnp.asarray(val_corpus.tokens[val_idx]),
                    targets=jnp.asarray(val_corpus.targets[val_idx]),
                    attn_mask=jnp.asarray(val_corpus.attn_mask[val_idx]),
                    loss_mask=jnp.asarray(val_corpus.loss_mask[val_idx]),
                    legal_mask=_legal_for(val_corpus, val_idx),
                )
                metrics = val_metrics_fn(
                    state.backbone, state.adapter, val_batch
                )
                val_loss = float(metrics.loss)
                # v1-parity richer val metrics (top1 / top5 /
                # illegal_pred_rate) alongside the bare val_loss.
                val_record = dict(
                    val_loss=val_loss,
                    val_top1=float(metrics.top1),
                    val_top5=float(metrics.top5),
                    val_illegal_pred_rate=float(metrics.illegal_pred_rate),
                    val_source=cfg.pgn_val_split if not args.no_pgn else "random",
                )
                logger.log_val(step=final_step, **val_record)
                log_metrics(wandb_run, val_record, step=final_step)
                # Patience / best-val checkpoint selection (v1 parity).
                # ``patience_state`` is the enclosing-scope tracker; a new
                # best val_loss resets the counter and tags the checkpoint
                # so ``find_best_adapter_step`` can recover it.
                _on_val(val_loss, final_step)
            if final_step % cfg.checkpoint_interval == 0:
                _save(final_step)
            if should_shutdown():
                return state, final_step
            if patience_state["stop"]:
                return state, final_step
        return state, final_step

    # H7: write schedule_health.json at *every* exit path (normal,
    # SIGTERM, exception). `reason_for_stop` defaults to `completed`
    # (full-budget run); SIGTERM and exceptions overwrite it. The
    # planned budget is the resolved `effective_total_steps`; `final_step`
    # is what actually ran (their disagreement on a `completed` stop is
    # the structural-bug signal the lab runner flags).
    reason_for_stop = "completed"
    try:
        if is_rosa:
            # Phase 1: LoRA warmup (state.adapter init'd with
            # lora_active=True, sparse_active=False; bottleneck branch is
            # silenced via apply_rosa's sparse_active gate).
            rosa_cfg = state.adapter.cfg
            warmup_n = min(rosa_cfg.rosa_warmup_steps, effective_total_steps)
            # `--resume + RoSA` is rejected upstream (where args.resume is
            # parsed) so resume_step is guaranteed to be 0 here.
            state, last = _run_steps(scan_step, warmup_n, start_step=0)
            if not should_shutdown() and warmup_n < effective_total_steps:
                # Phase 2: gather `mask_samples` batches and accumulate
                # |grad|^grad_alpha on each sparse delta to derive the
                # density-thresholded boolean masks. The deltas are zeroed
                # before Phase 3 so the sparse contribution starts at zero.
                mask_batches: list[Batch] = []
                for _ in range(rosa_cfg.mask_samples):
                    idx = rng.integers(0, corpus.n_games, size=cfg.batch_size)
                    mb = slice_batch(corpus, idx)
                    if need_legal:
                        mb = eqx.tree_at(
                            lambda b: b.legal_mask, mb,
                            _legal_for(corpus, idx),
                            is_leaf=lambda x: x is None,
                        )
                    mask_batches.append(mb)
                new_sparse = generate_rosa_masks(
                    state.backbone, state.adapter, mask_batches,
                    compute_dtype=compute_dtype, apply_legal=apply_legal,
                )
                # Phase 3: re-init LoRA (kaiming A, zero B), install masks,
                # flip toggles per mode (rosa keeps LoRA on; retro modes
                # drop LoRA). The optimizer is re-initialised because the
                # branch toggles change which arrays receive gradients.
                new_adapter = rosa_phase1_to_phase3(
                    state.adapter, new_sparse, key=jax.random.key(1),
                )
                flt = dispatch_filter(cfg.strategy)(new_adapter)
                opt_state = optimizer.init(eqx.filter(new_adapter, flt))
                state = AdapterTrainState(
                    backbone=state.backbone,
                    adapter=new_adapter,
                    opt_state=opt_state,
                    step=state.step,
                    key=state.key,
                )
                # Phase 3 train step has the same `apply_rosa` closure but
                # the adapter's toggles are now Phase-3-shaped, so the
                # composed forward includes sparse (and bottleneck where
                # appropriate). Re-build the jitted step to pick up the new
                # opt_state shape.
                train_step = make_adapter_train_step(
                    cfg.strategy, optimizer,
                    compute_dtype=compute_dtype,
                    use_sdpa=cfg.use_sdpa, use_flash=use_flash,
                    apply_legal=apply_legal, illegal_penalty=illegal_penalty,
                )
                scan_step = make_adapter_scan_step(train_step)
                phase3_remaining = effective_total_steps - warmup_n
                state, _ = _run_steps(
                    scan_step, phase3_remaining, start_step=warmup_n,
                )
        else:
            # `resume_step` is the absolute step the saved checkpoint
            # reached; remaining work is `effective_total_steps - resume_step`
            # so the run honours the resolved (epochs × steps_per_epoch, or
            # plain total_steps) budget.
            remaining = max(0, effective_total_steps - resume_step)
            state, _ = _run_steps(
                scan_step, remaining, start_step=resume_step,
            )
        # Always emit a final checkpoint at run-end, even when
        # `total_steps < checkpoint_interval` (short LoRA smokes, sweeps,
        # the §3 criterion 7 acceptance command). Without this the trained
        # adapter weights are discarded silently when the loop exits before
        # crossing a checkpoint boundary.
        if final_step > 0 and final_step % cfg.checkpoint_interval != 0:
            _save(final_step)
        if should_shutdown():
            reason_for_stop = "sigterm"
        elif patience_state["stop"]:
            # Early-stopped on the held-out val loss (v1 parity). This is a
            # clean exit; ``actual != planned`` is expected and is *not* the
            # structural-bug signal a ``completed`` mismatch would be.
            reason_for_stop = "patience"
        elif final_step == 0 and resume_step > 0:
            # Resumed at/past the resolved budget — no steps ran. The
            # actual step count is the saved checkpoint's, not 0.
            reason_for_stop = "resume_no_op"
    except BaseException:
        reason_for_stop = "exception"
        raise
    finally:
        # `actual` is what ran this session, or the resumed step on a no-op.
        actual_total = final_step if final_step > 0 else resume_step
        actual_final_lr = float(
            np.asarray(schedule(max(0, actual_total - 1))).item()
        )
        write_schedule_health(
            logger.run_dir,
            schedule=cfg.lr_schedule,
            planned_total_steps=effective_total_steps,
            actual_total_steps=actual_total,
            lr_peak=cfg.lr,
            actual_final_lr=actual_final_lr,
            reason_for_stop=reason_for_stop,
        )
        # Best-val checkpoint selection (v1 parity): record the step that
        # minimised val_loss so downstream publish / eval can pick it
        # without re-scanning metrics.jsonl. Fall back to the live
        # patience tracker when no metrics file is readable.
        from pawn.checkpoint import find_best_adapter_step
        best_step = find_best_adapter_step(logger.run_dir / "metrics.jsonl")
        if best_step is None:
            best_step = patience_state["best_step"]
        if best_step is not None:
            (logger.run_dir / "best_step.json").write_text(
                json.dumps(
                    {
                        "best_step": int(best_step),
                        "best_val_loss": (
                            None
                            if not np.isfinite(patience_state["best_loss"])
                            else float(patience_state["best_loss"])
                        ),
                        "checkpoint": f"adapter_step_{int(best_step):08d}",
                    }
                ),
                encoding="utf-8",
            )
        # Only an in-loop exception is a failed run; completed / sigterm /
        # resume_no_op are clean exits.
        finish_wandb(wandb_run, exit_code=1 if reason_for_stop == "exception" else 0)
    if push_tracker:
        # Mirror `scripts/train_jax.py` — only `timeouts > 0` implies a
        # worker is stuck; `errors > 0` is an exit-cleanly upload
        # failure that doesn't need the abandon-thread path.
        timeouts, _errors = drain_push_queue(push_tracker, timeout=300.0)
        push_tracker.shutdown(drain_succeeded=(timeouts == 0))
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
