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
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.adapter_trainer import (
    STRATEGIES,
    AdapterTrainState,
    dispatch_filter,
    dispatch_init,
    generate_rosa_masks,
    make_adapter_train_step,
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
from pawn.corpus import generate_corpus
from pawn.jax_setup import setup_jax_caching
from pawn.lichess_data import load_lichess_corpus
from pawn.lifecycle import (
    HFPushTracker,
    drain_push_queue,
    install_sigterm_handler,
    push_checkpoint_async,
)
from pawn.logging import MetricsLogger
from pawn.model import init_model, sliced
from pawn.run_config import AdapterConfig
from pawn.trainer import (
    cross_entropy_loss, flatten_opt_state, make_lr_schedule,
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


def _resolve_device() -> str:
    """Return the device label for MetricsLogger / GPU-stats source.

    Reads ``jax.default_backend()`` so the logger picks the right
    ``smi`` shell-out (``rocm-smi`` on ROCm, ``nvidia-smi`` on CUDA)
    instead of hardcoding ``cuda``. Falls back to ``cpu`` when JAX
    reports no accelerator — the script then exits unless the operator
    has set ``PAWN_ALLOW_CPU=1`` (parity with the v1 escape hatch
    pinned in plan §6).
    """
    import jax

    backend = jax.default_backend()
    if backend == "gpu":
        # `jax.devices()[0]` reports `rocm:0` on ROCm and `cuda:0` on
        # NVIDIA; the prefix is what the MetricsLogger keys on.
        dev_str = str(jax.devices()[0]).lower()
        if "rocm" in dev_str:
            return "rocm"
        return "cuda"
    if backend == "tpu":
        return "tpu"
    return "cpu"


def _require_accelerator() -> None:
    """Refuse to run training on CPU unless ``PAWN_ALLOW_CPU=1`` is set.

    JAX silently falls back to CPU if no GPU plugin is installed; without
    this guard, an operator can start a multi-hour run and only notice
    much later (per CLAUDE.md / plan §6 the v1 escape hatch is
    ``PAWN_ALLOW_CPU=1``; preserve it).
    """
    import os

    import jax

    if jax.default_backend() == "cpu" and os.environ.get("PAWN_ALLOW_CPU") != "1":
        raise SystemExit(
            "JAX resolved to the CPU backend; refusing to run training. "
            "Install a GPU jaxlib plugin (--extra rocm or --extra cu128), "
            "or set PAWN_ALLOW_CPU=1 to override."
        )


def _strategy_config_from_run(cfg: AdapterConfig) -> object:
    """Build the adapter's strategy Config from the AdapterConfig fields."""
    s = cfg.strategy
    if s == "lora":
        return LoRAConfig(
            rank=cfg.lora_rank or 4,
            targets=cfg.lora_targets or "qkvo",
            ffn=cfg.lora_ffn,
        )
    if s == "film":
        return FiLMConfig(use_output_film=cfg.use_output_film)
    if s == "bottleneck":
        return BottleneckConfig(
            dim=cfg.bottleneck_dim or 8,
            n_hidden=cfg.bottleneck_n_hidden,
            no_adapt_attn=cfg.no_adapt_attn,
            no_adapt_ffn=cfg.no_adapt_ffn,
        )
    if s == "hybrid":
        return HybridConfig(
            lora=LoRAConfig(
                rank=cfg.lora_rank or 4,
                targets=cfg.lora_targets or "qkvo",
                ffn=cfg.lora_ffn,
            ),
            film=FiLMConfig(use_output_film=cfg.use_output_film),
        )
    if s == "sparse":
        return SparseConfig(
            density=cfg.density or 0.01,
            targets=cfg.sparse_targets or "qkvo",
            ffn=cfg.sparse_ffn,
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
        return RoSAConfig(
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
        )
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
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--bottleneck-dim", type=int, default=None)
    ap.add_argument("--use-output-film", action="store_true")
    ap.add_argument("--no-adapt-attn", action="store_true")
    ap.add_argument("--no-adapt-ffn", action="store_true")
    ap.add_argument("--rosa-mode", default=None)
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
                         "Bottleneck-style adapters automatically detect "
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
    # NOTE: `--wandb` is intentionally absent — the adapter parser
    # doesn't register it (the pretrain script does). Adding it here
    # would be a no-op because `getattr(args, "wandb", False)` always
    # returns False on the adapter side. (Round-2 bug-detector MINOR.)
    _CLI_STORE_TRUE_FLAGS = ("use_sdpa", "use_output_film", "no_adapt_attn",
                             "no_adapt_ffn")
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
    # device — it deliberately runs *ahead* of `_require_accelerator()`
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

    _require_accelerator()
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

    # Resume: splice adapter + step + opt-state from a checkpoint dir.
    # The backbone is taken from the checkpoint (the saved `model.safetensors`
    # *is* the backbone for bottleneck-style adapters; for weight-folded
    # adapters it's the folded effective model — both load cleanly via
    # `pawn.checkpoint.load_model` since the v2 schema treats them
    # identically). Bottleneck adapters auto-detect the sidecar via
    # `load_bottleneck_adapter`; other strategies cold-start the adapter
    # PyTree but warm-start the optimizer state, which still preserves
    # Adam moments + clip counter across the resume boundary.
    # `resume_step` is already extracted upstream (right after
    # `_require_accelerator`) so the RoSA / training-state checks
    # fire before any HF download. Below we restore the backbone +
    # adapter + optimizer state from the checkpoint.
    resume_step = resume_step_early
    if args.resume is not None:
        from pawn.adapters.bottleneck import (
            ADAPTER_SAFETENSORS,
            BottleneckConfig,
            load_bottleneck_adapter,
        )
        from pawn.checkpoint import OPTIMIZER_FILE, load_model
        from pawn.trainer import unflatten_opt_state

        from pawn.corpus import (
            assert_conditioning_C,
            conditioning_from_run_block,
            conditioning_to_C,
        )

        ckpt_dir = Path(args.resume)
        backbone, backbone_run_block = load_model(ckpt_dir)
        # Same load-time C-mismatch guard as the initial-load path: the
        # corpus below is built with `cfg.conditioning`, so the resumed
        # backbone's persisted conditioning must agree or moves drift.
        resume_C = conditioning_to_C(
            conditioning_from_run_block(backbone_run_block)
        )
        assert_conditioning_C(cfg.conditioning, resume_C)
        # Bottleneck sidecar: re-compose the wrapper. Otherwise fall
        # back to the freshly-initialised adapter (weight-folded
        # adapters bake into the backbone at save time, so they don't
        # need a separate restore step).
        sidecar = ckpt_dir / ADAPTER_SAFETENSORS
        if sidecar.is_file() and isinstance(strategy_cfg, BottleneckConfig):
            adapter = load_bottleneck_adapter(ckpt_dir, strategy_cfg)
        # Build the optimizer template *after* adapter rebuild so the
        # opt-state shape matches what the loaded adapter trains.
        flt = dispatch_filter(cfg.strategy)(adapter)
        opt_state = optimizer.init(eqx.filter(adapter, flt))
        opt_path = ckpt_dir / OPTIMIZER_FILE
        if opt_path.is_file():
            from safetensors.numpy import load_file as st_load
            flat = st_load(str(opt_path))
            opt_state = unflatten_opt_state(opt_state, flat)
        else:
            print(
                f"[train_jax_adapter] WARNING: no {OPTIMIZER_FILE} in "
                f"{args.resume}; resuming with fresh opt_state.",
                file=sys.stderr,
            )
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
    train_step = make_adapter_train_step(
        cfg.strategy, optimizer,
        compute_dtype=compute_dtype, use_sdpa=cfg.use_sdpa, use_flash=use_flash,
    )

    apply_fn = STRATEGIES[cfg.strategy].apply

    @eqx.filter_jit
    def val_step(backbone, adapter, batch):
        effective = apply_fn(backbone, adapter)
        return cross_entropy_loss(effective, batch)

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
        device=_resolve_device(), suffix=cfg.variant,
    )
    logger.log_config(run_type="adapter", config=cfg.model_dump())

    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None
    should_shutdown = install_sigterm_handler()

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
        from pawn.adapters.bottleneck import (
            BottleneckEffective,
            save_bottleneck_adapter,
        )
        effective = apply_fn(state.backbone, state.adapter)
        out = logger.run_dir / f"adapter_step_{step_int:08d}"
        if isinstance(effective, BottleneckEffective):
            save_model(
                effective.backbone, out,
                run_config=cfg.model_dump(),
                optimizer_state=flatten_opt_state(state.opt_state),
                training_state={"step": int(state.step)},
            )
            save_bottleneck_adapter(effective.adapter, out)
        else:
            # PAWNModel — the standard weight-folded save path. The
            # narrowing assertion satisfies pyright: the only two
            # apply_fn return types in v2 are PAWNModel and
            # BottleneckEffective; the isinstance branch above peeled
            # off the wrapper, leaving the PAWNModel case here.
            from pawn.model import PAWNModel as _PAWNModel
            assert isinstance(effective, _PAWNModel), (
                f"unexpected effective type {type(effective).__name__}; "
                "extend the save dispatch in train_jax_adapter._save"
            )
            save_model(
                effective, out,
                run_config=cfg.model_dump(),
                optimizer_state=flatten_opt_state(state.opt_state),
                training_state={"step": int(state.step)},
            )
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
    t0 = time.time()
    final_step = 0
    is_rosa = is_rosa_strategy

    def _run_steps(
        state: AdapterTrainState,
        step_fn: object,  # eqx-jitted closure
        n_steps: int,
        start_step: int,
    ) -> tuple[AdapterTrainState, int]:
        """Inner loop chunk used by both non-RoSA paths and per-phase
        RoSA orchestration. Returns ``(new_state, last_step_done)``
        where ``last_step_done`` is the absolute step counter the loop
        reached so the outer scope can drive checkpoints and resumes."""
        nonlocal final_step
        for offset in range(n_steps):
            absolute = start_step + offset
            idx = rng.integers(0, corpus.n_games, size=cfg.batch_size)
            batch = slice_batch(corpus, idx)
            state, loss = step_fn(state, batch)  # type: ignore[operator]
            final_step = absolute + 1
            if final_step % cfg.log_interval == 0:
                logger.log_train(
                    step=final_step, loss=float(loss),
                    lr=np.asarray(schedule(int(state.step))).item(),
                    step_time=(time.time() - t0) / final_step,
                )
            if final_step % eval_interval == 0:
                val_idx = val_rng.integers(
                    0, val_corpus.n_games, size=cfg.batch_size
                )
                val_batch = slice_batch(val_corpus, val_idx)
                val_loss = float(val_step(state.backbone, state.adapter, val_batch))
                logger.log_val(
                    step=final_step, val_loss=val_loss,
                    val_source=cfg.pgn_val_split if not args.no_pgn else "random",
                )
            if final_step % cfg.checkpoint_interval == 0:
                _save(final_step)
            if should_shutdown():
                return state, final_step
        return state, final_step

    if is_rosa:
        # Phase 1: LoRA warmup (state.adapter init'd with
        # lora_active=True, sparse_active=False; bottleneck branch is
        # silenced via apply_rosa's sparse_active gate).
        rosa_cfg = state.adapter.cfg
        warmup_n = min(rosa_cfg.rosa_warmup_steps, effective_total_steps)
        # `--resume + RoSA` is rejected upstream (where args.resume is
        # parsed) so resume_step is guaranteed to be 0 here.
        state, last = _run_steps(state, train_step, warmup_n, start_step=0)
        if not should_shutdown() and warmup_n < effective_total_steps:
            # Phase 2: gather `mask_samples` batches and accumulate
            # |grad|^grad_alpha on each sparse delta to derive the
            # density-thresholded boolean masks. The deltas are zeroed
            # before Phase 3 so the sparse contribution starts at zero.
            mask_batches: list = []
            for _ in range(rosa_cfg.mask_samples):
                idx = rng.integers(0, corpus.n_games, size=cfg.batch_size)
                mask_batches.append(slice_batch(corpus, idx))
            new_sparse = generate_rosa_masks(
                state.backbone, state.adapter, mask_batches,
                compute_dtype=compute_dtype,
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
            )
            phase3_remaining = effective_total_steps - warmup_n
            state, _ = _run_steps(
                state, train_step, phase3_remaining, start_step=warmup_n,
            )
    else:
        # `resume_step` is the absolute step the saved checkpoint
        # reached; remaining work is `effective_total_steps - resume_step`
        # so the run honours the resolved (epochs × steps_per_epoch, or
        # plain total_steps) budget.
        remaining = max(0, effective_total_steps - resume_step)
        state, _ = _run_steps(
            state, train_step, remaining, start_step=resume_step,
        )
    # Always emit a final checkpoint at run-end, even when
    # `total_steps < checkpoint_interval` (short LoRA smokes, sweeps,
    # the §3 criterion 7 acceptance command). Without this the trained
    # adapter weights are discarded silently when the loop exits before
    # crossing a checkpoint boundary.
    if final_step > 0 and final_step % cfg.checkpoint_interval != 0:
        _save(final_step)
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
