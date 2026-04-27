"""Adapter training building blocks.

This module contains all the functions needed to build, configure, and train
adapter models on Lichess data. It is imported by ``scripts/train.py`` (the
unified entry point) and can also be used programmatically.

Strategies:
  bottleneck       -- Houlsby residual MLP adapters
  lora             -- Low-rank attention/FFN adaptation
  film             -- Feature-wise linear modulation
  sparse           -- Random binary mask weight perturbations
  rosa             -- Gradient-informed sparse + LoRA (3-phase)
  hybrid           -- LoRA + FiLM combined
  specialized_clm  -- From-scratch standalone transformer (no backbone)
  unfreeze         -- Fine-tune top N backbone layers directly
"""

from __future__ import annotations

import gc
import json
import math
import signal
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pawn.config import CLMConfig
from pawn.lichess_data import (
    LegalMaskBuilder,
    compute_legal_indices,
)
from wandb import Run as WandbRun

from pawn.logging import MetricsLogger
from pawn.model import PAWNCLM
from pawn.wandb_utils import log_metrics

STRATEGIES = [
    "bottleneck",
    "lora",
    "film",
    "sparse",
    "rosa",
    "hybrid",
    "specialized_clm",
    "unfreeze",
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def load_backbone(checkpoint_path: str, device: str) -> PAWNCLM:
    from pawn.checkpoint import load_backbone_weights

    state_dict, model_config = load_backbone_weights(checkpoint_path, device)
    cfg = CLMConfig(**model_config) if model_config else CLMConfig()
    model = PAWNCLM(cfg).to(device)
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    model.eval()
    return model


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def wsd_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    decay_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
    decay_shape: str = "linear",
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup-Stable-Decay scheduler.

    Three phases:
        [0, warmup_steps)                         – linear 0 → 1
        [warmup_steps, total_steps - decay_steps) – flat 1 (stable)
        [total_steps - decay_steps, total_steps]  – 1 → min_lr_ratio

    ``decay_shape`` selects the decay curve:
      * ``"linear"`` (default) — straight line 1 → min_lr_ratio
      * ``"cosine"`` — half-cosine fall; matches the tail of a standard
        cosine schedule over the decay window. Empirically ~0.1-0.2pp
        better than linear in several recent WSD papers at the same
        budget (e.g. MiniCPM).

    When the stable phase is exhausted (``warmup + decay > total``) the
    stable phase is clipped to zero and we fall back to a warmup →
    decay schedule of the chosen shape.
    """
    if decay_shape not in ("linear", "cosine"):
        raise ValueError(
            f"Unknown wsd decay_shape: {decay_shape!r} "
            "(expected 'linear' or 'cosine')"
        )
    stable_end = max(warmup_steps, total_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < stable_end:
            return 1.0
        decay_window = max(total_steps - stable_end, 1)
        progress = min((step - stable_end) / decay_window, 1.0)
        if decay_shape == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        # Cosine: 0.5 * (1 + cos(π * progress)) maps 0→1 and 1→0.
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def constant_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup → hold peak LR indefinitely.

    The cleanest control against WSD: "does any decay help at all?"
    Pair with patience-based early stopping — without a schedule-driven
    end, the run keeps training at peak LR until val stops improving.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def infinite_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    cooldown_steps: int,
    decay_steps: int,
    total_steps: int,
    stable_lr_ratio: float = 0.1,
    # ``min_lr_ratio`` has no CLI / config path and is pinned to 0.0 in
    # production. Kept on the signature for tests and direct callers
    # that want a non-zero final-decay floor.
    min_lr_ratio: float = 0.0,
    # Default aligns with ``BaseRunConfig.wsd_decay_shape`` ("linear"),
    # which is what ``build_scheduler`` threads through the config
    # path. Direct instantiation otherwise silently diverges from the
    # ``--lr-schedule infinite`` behaviour.
    final_decay_shape: str = "linear",
) -> torch.optim.lr_scheduler.LambdaLR:
    """Infinite / restart-friendly LR schedule.

    Four phases:
        [0, warmup_steps)                                     – 0 → 1
        [warmup_steps, warmup+cooldown)                       – cosine 1 → stable
        [warmup+cooldown, total_steps - decay_steps)          – flat stable
        [total_steps - decay_steps, total_steps]              – stable → min

    The stable-phase LR depends only on ``stable_lr_ratio`` (not on
    ``total_steps``), so any checkpoint taken during that phase can be
    resumed with an extended ``total_steps`` without any LR
    discontinuity — the plateau simply lasts longer before the final
    decay kicks in. See Hägele et al. (2024) arXiv:2405.18392.

    ``final_decay_shape`` selects the final-decay curve (``"linear"`` or
    ``"cosine"``). The peak→stable cooldown is always cosine.
    """
    if final_decay_shape not in ("linear", "cosine"):
        raise ValueError(
            f"Unknown infinite final_decay_shape: {final_decay_shape!r} "
            "(expected 'linear' or 'cosine')"
        )
    if not 0.0 <= stable_lr_ratio <= 1.0:
        raise ValueError(
            f"stable_lr_ratio must be in [0, 1], got {stable_lr_ratio}"
        )
    if not 0.0 <= min_lr_ratio <= stable_lr_ratio:
        raise ValueError(
            f"min_lr_ratio ({min_lr_ratio}) must be in "
            f"[0, stable_lr_ratio={stable_lr_ratio}]"
        )
    if cooldown_steps < 0 or decay_steps < 0:
        raise ValueError(
            "cooldown_steps and decay_steps must be non-negative, "
            f"got {cooldown_steps}, {decay_steps}"
        )
    cooldown_end = warmup_steps + cooldown_steps
    final_decay_start = max(cooldown_end, total_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < cooldown_end:
            progress = (step - warmup_steps) / max(cooldown_steps, 1)
            return stable_lr_ratio + (1.0 - stable_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        if step < final_decay_start:
            return stable_lr_ratio
        progress = min(
            (step - final_decay_start) / max(total_steps - final_decay_start, 1),
            1.0,
        )
        span = stable_lr_ratio - min_lr_ratio
        if final_decay_shape == "linear":
            return min_lr_ratio + span * (1.0 - progress)
        return min_lr_ratio + span * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def one_cycle_schedule(
    optimizer: torch.optim.Optimizer,
    peak_step: int,
    total_steps: int,
    initial_div: float = 25.0,
    final_div: float = 1e4,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Smith one-cycle schedule with cosine annealing between phases.

    Ramps from ``peak_lr / initial_div`` up to ``peak_lr`` over the first
    ``peak_step`` steps, then decays cosine-wise to
    ``peak_lr / final_div`` over the remaining steps. No separate
    warmup — the ramp-up is the warmup.

    References:
        Smith (2018) "A disciplined approach to neural network
        hyper-parameters" arXiv:1803.09820.
    """
    if peak_step <= 0:
        raise ValueError("one_cycle_schedule requires peak_step > 0")
    if peak_step >= total_steps:
        raise ValueError("one_cycle_schedule requires peak_step < total_steps")
    initial_frac = 1.0 / initial_div
    final_frac = 1.0 / final_div

    def lr_lambda(step: int) -> float:
        if step <= peak_step:
            # Cosine ramp from initial_frac to 1.0.
            progress = step / max(peak_step, 1)
            return initial_frac + (1.0 - initial_frac) * 0.5 * (
                1.0 - math.cos(math.pi * progress)
            )
        # Cosine decay from 1.0 to final_frac.
        progress = min((step - peak_step) / max(total_steps - peak_step, 1), 1.0)
        return final_frac + (1.0 - final_frac) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
    decay_steps: int | None = None,
    wsd_decay_shape: str = "linear",
    cooldown_steps: int | None = None,
    stable_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Dispatch one of the supported LR schedules by name.

    Supported schedules: ``"cosine"``, ``"wsd"``, ``"constant"``,
    ``"one_cycle"``, ``"infinite"``. See the individual scheduler
    docstrings for parameter meaning. ``warmup_steps`` is reused as the
    ramp-up for ``one_cycle`` — set it to ~0.3 * total_steps for the
    canonical Smith shape. ``infinite`` additionally requires
    ``cooldown_steps`` (peak→stable cosine window) and ``decay_steps``
    (final stable→0 window); ``wsd_decay_shape`` selects the final
    decay curve.
    """
    if schedule == "cosine":
        return cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    if schedule == "wsd":
        if decay_steps is None:
            raise ValueError("WSD schedule requires decay_steps")
        return wsd_schedule(
            optimizer, warmup_steps, decay_steps, total_steps,
            decay_shape=wsd_decay_shape,
        )
    if schedule == "constant":
        return constant_warmup_schedule(optimizer, warmup_steps)
    if schedule == "one_cycle":
        return one_cycle_schedule(optimizer, warmup_steps, total_steps)
    if schedule == "infinite":
        if decay_steps is None or cooldown_steps is None:
            raise ValueError(
                "infinite schedule requires both decay_steps and cooldown_steps"
            )
        return infinite_schedule(
            optimizer, warmup_steps, cooldown_steps, decay_steps, total_steps,
            stable_lr_ratio=stable_lr_ratio,
            final_decay_shape=wsd_decay_shape,
        )
    raise ValueError(f"Unknown lr_schedule: {schedule!r}")


def build_compiled_step(
    model: nn.Module,
    *,
    apply_legal_mask: bool,
    illegal_penalty: float,
    amp_dtype: torch.dtype | None,
    use_compile: bool,
) -> Any:
    """Build a (possibly compiled) per-step function for the hot loop.

    Returns a callable ``step(ids, msk, legal_mask, targets) -> (loss,
    top1_sum, n_pos)``. Combines the backbone forward, sparse logit
    projection, optional legal-mask, cross-entropy, and top-1 count
    into a single graph that ``torch.compile(mode="reduce-overhead")``
    can fuse and CUDA-graph-capture per (B, T) shape.

    The data-dependent ``hidden[msk]`` filter forces a graph break at
    that point — cudagraph trees fall back to eager for the masked
    section and resume capture for the loss tail. The bulk of kernel-
    launch overhead lives in the backbone forward (8 layers ×
    {LayerNorm, attn, FFN}), which is fully captured.
    """
    use_amp = amp_dtype is not None

    def _step(
        ids: torch.Tensor,
        msk: torch.Tensor,
        legal_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            hidden = model.forward_hidden(ids)  # type: ignore[attr-defined]
            valid_hidden = hidden[msk]
            valid_logits = model.project_head(valid_hidden)  # type: ignore[attr-defined]
        valid_legal = legal_mask[msk]
        valid_logits = valid_logits.float()
        if apply_legal_mask:
            valid_logits = valid_logits.masked_fill(~valid_legal, float("-inf"))
        valid_targets = targets[msk]
        loss = F.cross_entropy(valid_logits, valid_targets)
        if illegal_penalty > 0:
            probs = torch.softmax(valid_logits, dim=-1)
            illegal_mass = (
                probs.masked_fill(valid_legal, 0.0).sum(dim=-1).mean()
            )
            loss = loss + illegal_penalty * illegal_mass
        # Top-1 count stays inside the compiled graph so we don't
        # spawn a separate kernel-launch chain for diagnostics.
        preds = valid_logits.argmax(dim=-1)
        top1_sum = (preds == valid_targets).sum()
        return loss, top1_sum

    if use_compile:
        # ``reduce-overhead`` enables CUDA graph capture for the static-
        # shape regions; the bucketed collate keeps the number of
        # distinct (B, T) shapes small. ``dynamic=False`` lets inductor
        # specialize one graph per shape rather than emitting a single
        # symbol-shape graph that defeats the cudagraph win.
        import torch._dynamo as _dynamo
        # 8 buckets × the few apply_legal_mask / scaler combinations a
        # run actually exercises is well under 64; bumping the cache
        # limit avoids a "recompile limit hit" silent fallback.
        _dynamo.config.cache_size_limit = max(
            _dynamo.config.cache_size_limit, 64
        )
        compiled = torch.compile(_step, mode="reduce-overhead", dynamic=False)

        # ``hidden[msk]`` produces a data-dependent shape, so cudagraph
        # trees splits the step into two captured subgraphs. Without an
        # explicit step boundary the second-subgraph output buffer
        # (``valid_logits``) gets overwritten by the next step's
        # forward before ``backward`` can consume it. Marking the step
        # boundary tells the cudagraph manager to start a new memory
        # epoch on each call. The wrapper keeps this concern out of
        # the train loop.
        def _wrapped(ids, msk, legal_mask, targets):
            torch.compiler.cudagraph_mark_step_begin()
            return compiled(ids, msk, legal_mask, targets)

        return _wrapped
    return _step


def sparse_forward(
    model: nn.Module,
    ids: torch.Tensor,
    msk: torch.Tensor,
    legal_mask: torch.Tensor,
    amp_dtype: torch.dtype | None,
    device: str,
    apply_legal_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward + sparse logit projection.

    Returns ``(valid_logits, valid_legal)`` — both indexed by the
    loss mask. When ``apply_legal_mask=True`` (the default), illegal
    logits are filled with ``-inf`` so the softmax never assigns mass
    to them. When False, the raw logits are returned and the caller
    is expected to use ``valid_legal`` (a bool tensor over positions ×
    vocab) for any legality-aware loss term or metric.
    """
    with torch.amp.autocast(
        "cuda", dtype=amp_dtype, enabled=amp_dtype is not None
    ):
        hidden = model.forward_hidden(ids)  # type: ignore[attr-defined]
        valid_hidden = hidden[msk]
        valid_logits = model.project_head(valid_hidden)  # type: ignore[attr-defined]
    valid_legal = legal_mask[msk]
    valid_logits = valid_logits.float()
    if apply_legal_mask:
        valid_logits.masked_fill_(~valid_legal, float("-inf"))
    return valid_logits, valid_legal


def illegal_probability_mass(
    valid_logits: torch.Tensor,
    valid_legal: torch.Tensor,
) -> torch.Tensor:
    """Mean (over positions) of the softmax mass landing on illegal moves.

    Returns a scalar tensor on the same device as the inputs.

    .. warning::
       If ``valid_logits`` has been masked with ``-inf`` at every
       position in a row (all-illegal), ``torch.softmax`` produces NaN.
       Callers that apply the hard legal mask upstream should
       short-circuit instead of calling this — the answer is
       analytically zero and the softmax is undefined.
    """
    probs = torch.softmax(valid_logits, dim=-1)
    return probs.masked_fill(valid_legal, 0.0).sum(dim=-1).mean()


def compute_adapter_loss(
    valid_logits: torch.Tensor,
    valid_targets: torch.Tensor,
    valid_legal: torch.Tensor,
    *,
    illegal_penalty: float,
) -> torch.Tensor:
    """Cross-entropy over moves, plus optional illegal-mass penalty.

    When ``illegal_penalty > 0`` we add ``lambda * E[P_illegal]`` —
    the expected softmax mass landing on illegal moves, averaged over
    positions — to the loss. If legality was masked inside
    ``sparse_forward``, the illegal probability mass is exactly zero,
    so the penalty term contributes nothing regardless of ``lambda``.
    """
    loss = F.cross_entropy(valid_logits, valid_targets)
    if illegal_penalty > 0:
        loss = loss + illegal_penalty * illegal_probability_mass(
            valid_logits, valid_legal
        )
    return loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    mask_builder: LegalMaskBuilder,
    device: str,
    amp_dtype: torch.dtype | None = None,
    precomputed_indices: list[torch.Tensor] | None = None,
    apply_legal_mask: bool = True,
    illegal_penalty: float = 0.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_positions = 0

    # Legality diagnostics. When the hard mask is applied upstream,
    # illegal logits are ``-inf``: the argmax is always legal and the
    # softmax mass on illegal positions is analytically zero. We
    # short-circuit that path both to avoid wasted work and because
    # ``softmax`` over an all-``-inf`` row (possible if a position has
    # zero legal moves) would NaN and poison the metric. When the mask
    # is off we track the metrics on-GPU and sync once at the end —
    # the training loop already does this for speed on ROCm.
    track_illegal = not apply_legal_mask
    illegal_pred_count = torch.zeros((), device=device)
    illegal_mass_sum = torch.zeros((), device=device)

    for i, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device, non_blocking=True)
        tgt = batch["targets"].to(device, non_blocking=True)
        msk = batch["loss_mask"].to(device, non_blocking=True)
        # Per-batch ``T`` from BucketedLegalMaskCollate; falls back to
        # the input width when the legacy fixed-T collate is in use.
        T_batch = int(batch.get("T_actual", ids.shape[1]))
        if precomputed_indices is not None:
            # Precomputed indices may be either a flat tensor (legacy
            # fixed-T path) or a (T, indices) tuple from
            # :func:`precompute_val_masks` under bucketing.
            entry = precomputed_indices[i]
            if isinstance(entry, tuple):
                T_b, idx_b = entry
                legal_mask = mask_builder.scatter(idx_b, ids.shape[0], T=T_b)
            else:
                legal_mask = mask_builder.scatter(entry, ids.shape[0])
        elif "legal_indices" in batch:
            legal_mask = mask_builder.scatter(
                batch["legal_indices"], ids.shape[0], T=T_batch,
            )
        else:
            legal_mask = mask_builder(batch)

        valid_logits, valid_legal = sparse_forward(
            model, ids, msk, legal_mask, amp_dtype, device,
            apply_legal_mask=apply_legal_mask,
        )
        valid_targets = tgt[msk]
        n_pos = valid_targets.shape[0]
        if n_pos == 0:
            continue

        loss = compute_adapter_loss(
            valid_logits, valid_targets, valid_legal,
            illegal_penalty=illegal_penalty,
        )
        preds = valid_logits.argmax(dim=-1)
        top1 = (preds == valid_targets).float().mean().item()
        top5 = valid_logits.topk(5, dim=-1).indices
        top5_acc = (
            (top5 == valid_targets.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )

        if track_illegal:
            pred_legal = valid_legal.gather(1, preds.unsqueeze(-1)).squeeze(-1)
            illegal_pred_count += (~pred_legal).sum()
            # ``illegal_probability_mass`` returns the per-batch mean;
            # re-weight by ``n_pos`` so the final division by
            # ``total_positions`` gives a correct overall mean even
            # when batches have different sizes.
            illegal_mass_sum += illegal_probability_mass(
                valid_logits, valid_legal
            ) * n_pos

        total_loss += loss.item() * n_pos
        total_top1 += top1 * n_pos
        total_top5 += top5_acc * n_pos
        total_positions += n_pos

    if total_positions == 0:
        return {
            "loss": 0.0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "illegal_pred_rate": 0.0,
            "illegal_prob_mass": 0.0,
        }

    if track_illegal:
        illegal_pred_rate = (illegal_pred_count / total_positions).item()
        illegal_prob_mass = (illegal_mass_sum / total_positions).item()
    else:
        # Analytically zero under hard masking; see note above.
        illegal_pred_rate = 0.0
        illegal_prob_mass = 0.0

    return {
        "loss": total_loss / total_positions,
        "top1_accuracy": total_top1 / total_positions,
        "top5_accuracy": total_top5 / total_positions,
        "illegal_pred_rate": illegal_pred_rate,
        "illegal_prob_mass": illegal_prob_mass,
    }


def parse_layers(s: str | None) -> tuple[int, ...] | None:
    if s is None:
        return None
    return tuple(int(x) for x in s.split(","))


_SCHEDULES_THAT_REACH_ZERO = ("cosine", "wsd", "infinite", "one_cycle")


def resume_state(global_step: int, steps_per_epoch: int) -> tuple[int, int]:
    """Derive ``(start_epoch, skip_batches)`` from a saved ``global_step``.

    With cache-first the canonical resume state is ``global_step``: the
    trainer enters epoch ``global_step // steps_per_epoch`` and asks the
    sampler to skip the first ``global_step % steps_per_epoch`` batches.
    No FF iteration; the seeded permutation slices to the right tail.

    Edge cases this codifies:
      - ``global_step == 0``: fresh run, ``start_epoch=0, skip=0``.
      - exact epoch boundary (e.g. just finished epoch 2 with
        ``steps_per_epoch=1000``, ``global_step=2000``):
        ``start_epoch=2, skip=0`` so the next loop iteration is epoch 2
        (which the for-range will skip past if epochs<=2 — that's the
        ``resume_no_op`` case handled at the schedule_health write).
      - mid-epoch (e.g. ``global_step=2500, steps_per_epoch=1000``):
        ``start_epoch=2, skip=500`` to enter epoch 2 partway through.
    """
    if steps_per_epoch <= 0:
        raise ValueError(
            f"steps_per_epoch must be > 0, got {steps_per_epoch}"
        )
    if global_step < 0:
        raise ValueError(f"global_step must be >= 0, got {global_step}")
    return global_step // steps_per_epoch, global_step % steps_per_epoch


def write_schedule_health(
    run_dir: Any,
    *,
    schedule: str,
    planned_total_steps: int,
    actual_total_steps: int,
    lr_peak: float,
    actual_final_lr: float,
    reason_for_stop: str,
) -> dict[str, Any]:
    """Write ``schedule_health.json`` and return its contents.

    Records whether training reached the planned end of the LR schedule.
    The "loud" path — ``actual_total_steps != planned_total_steps`` on a
    schedule that should reach zero — also prints a red banner to
    stderr/stdout so a post-hoc reader doesn't have to find the file
    themselves. ``reason_for_stop`` of ``"completed"`` plus a step
    mismatch is the structural-bug signal: with cache-first that
    combination should not occur, and a future regression that brings
    it back will surface here.
    """
    from pathlib import Path

    run_dir = Path(run_dir)
    should_reach_zero = schedule in _SCHEDULES_THAT_REACH_ZERO
    completion_ratio = (
        actual_total_steps / planned_total_steps
        if planned_total_steps > 0
        else 0.0
    )
    health: dict[str, Any] = {
        "format_version": 1,
        "schedule": schedule,
        "should_reach_zero": should_reach_zero,
        "planned_total_steps": int(planned_total_steps),
        "actual_total_steps": int(actual_total_steps),
        "completion_ratio": completion_ratio,
        "lr_peak": float(lr_peak),
        "actual_final_lr": float(actual_final_lr),
        "reason_for_stop": reason_for_stop,
    }
    (run_dir / "schedule_health.json").write_text(
        json.dumps(health, indent=2)
    )

    if (
        actual_total_steps != planned_total_steps
        and should_reach_zero
        and reason_for_stop in ("completed", "step_limit")
    ):
        print(
            "\033[31m"
            f"WARNING: schedule did not run to completion: "
            f"actual_total_steps={actual_total_steps} != "
            f"planned_total_steps={planned_total_steps}. "
            f"Final LR={actual_final_lr:.3e} "
            f"(peak={lr_peak:.3e}). reason={reason_for_stop}.\033[0m",
            flush=True,
        )

    return health


def precompute_val_masks(
    val_loader: DataLoader[Any],
    mask_builder: LegalMaskBuilder,
    vocab_size: int,
) -> list[torch.Tensor | tuple[int, torch.Tensor]]:
    """Precompute legal-mask indices for the val set.

    With :class:`BucketedLegalMaskCollate` the val loader emits a
    per-batch ``T_actual`` and ``legal_indices`` is already on the
    batch — pass them through verbatim so :func:`evaluate` can scatter
    at the right shape. Without bucketing we fall back to the legacy
    path (fixed ``mask_builder.T`` for every batch) and store flat
    tensors.
    """
    indices: list[torch.Tensor | tuple[int, torch.Tensor]] = []
    for batch in val_loader:
        if "legal_indices" in batch and "T_actual" in batch:
            idx_t = batch["legal_indices"]
            if not isinstance(idx_t, torch.Tensor):
                idx_t = torch.from_numpy(idx_t)
            T_b = int(batch["T_actual"])
            indices.append((T_b, idx_t.pin_memory()))
            continue
        move_ids = batch["move_ids"]
        if isinstance(move_ids, torch.Tensor):
            move_ids = move_ids.numpy()
        game_lengths = np.asarray(batch["game_length"], dtype=np.int16)
        idx = compute_legal_indices(
            move_ids,
            game_lengths,
            mask_builder.T,
            vocab_size,
            prepend_outcome=mask_builder.prepend_outcome,
        )
        indices.append(torch.from_numpy(idx).pin_memory())
    return indices


# ---------------------------------------------------------------------------
# Model construction per strategy
# ---------------------------------------------------------------------------


def build_model(
    args: Any, device: str
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    """Build the model for the given strategy.

    Returns (model, trainable_params, param_count, state_dict_fn, weight_report_fn).
    """
    strategy = args.strategy
    layers = parse_layers(args.adapter_layers)

    if strategy == "specialized_clm":
        return _build_specialized_clm(args, device)

    # All other strategies need a backbone
    print(f"Loading backbone: {args.checkpoint}")
    backbone = load_backbone(args.checkpoint, device)

    if strategy == "bottleneck":
        return _build_bottleneck(backbone, args, layers, device)
    elif strategy == "lora":
        return _build_lora(backbone, args, layers, device)
    elif strategy == "film":
        return _build_film(backbone, args, device)
    elif strategy == "sparse":
        return _build_sparse(backbone, args, layers, device)
    elif strategy == "hybrid":
        return _build_hybrid(backbone, args, layers, device)
    elif strategy == "unfreeze":
        return _build_unfreeze(backbone, args, device)
    elif strategy == "rosa":
        # RoSA is special -- returns the warmup model for Phase 1.
        # Phase 3 model is built after mask generation in main().
        return _build_rosa_warmup(backbone, args, layers, device)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _build_bottleneck(
    backbone: PAWNCLM,
    args: Any,
    layers: tuple[int, ...] | None,
    device: str,
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    from pawn.adapters.bottleneck import BottleneckCLM

    dim = args.bottleneck_dim or 8
    model = BottleneckCLM(
        backbone,
        bottleneck_dim=dim,
        adapt_attn=not args.no_adapt_attn,
        adapt_ffn=not args.no_adapt_ffn,
        layers=layers,
        n_hidden=getattr(args, "bottleneck_n_hidden", 0),
    ).to(device)
    params = model.adapter_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.adapter_state_dict, model.adapter_weight_report


def _build_lora(
    backbone: PAWNCLM,
    args: Any,
    layers: tuple[int, ...] | None,
    device: str,
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    from pawn.adapters.lora import LoRACLM

    rank = args.lora_rank or 4
    model = LoRACLM(
        backbone,
        rank=rank,
        attn_targets=args.lora_targets,
        adapt_ffn=args.lora_ffn,
        layers=layers,
    ).to(device)
    params = model.lora_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.lora_state_dict, model.lora_weight_report


def _build_film(
    backbone: PAWNCLM, args: Any, device: str
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    from pawn.adapters.film import FiLMCLM

    model = FiLMCLM(
        backbone,
        use_output_film=args.use_output_film,
    ).to(device)
    params = model.film_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.film_state_dict, model.film_weight_report


def _build_sparse(
    backbone: PAWNCLM,
    args: Any,
    layers: tuple[int, ...] | None,
    device: str,
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    from pawn.adapters.lora import ATTN_PRESETS
    from pawn.adapters.sparse import SparseCLM

    density = args.density or 0.01
    attn_targets = ATTN_PRESETS[args.sparse_targets or "qkvo"]
    model = SparseCLM(
        backbone,
        density=density,
        attn_targets=attn_targets,
        adapt_ffn=args.sparse_ffn,
        layers=layers,
    ).to(device)
    params = model.sparse_parameters()
    n = model.n_active_params()
    return model, params, n, model.sparse_state_dict, model.sparse_weight_report


def _build_hybrid(
    backbone: PAWNCLM,
    args: Any,
    layers: tuple[int, ...] | None,
    device: str,
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    from pawn.adapters.hybrid import HybridCLM

    rank = args.lora_rank or 4
    model = HybridCLM(
        backbone,
        lora_rank=rank,
        attn_targets=args.lora_targets,
        adapt_ffn=args.lora_ffn,
        lora_layers=layers,
        use_film=True,
        use_output_film=args.use_output_film,
    ).to(device)
    lora_params = model.lora_parameters()
    film_params = model.film_parameters()
    params = lora_params + film_params
    n = sum(p.numel() for p in params)
    return model, params, n, model.adapter_state_dict, model.weight_report


def _build_unfreeze(
    backbone: PAWNCLM, args: Any, device: str
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    unfreeze_layers = parse_layers(args.unfreeze_layers)
    if unfreeze_layers is None:
        raise ValueError(
            "--unfreeze-layers is required for unfreeze strategy"
        )

    # Freeze everything first
    for p in backbone.parameters():
        p.requires_grad = False

    # Unfreeze specified layers
    for layer_idx in unfreeze_layers:
        block = backbone.get_block(layer_idx)
        for p in block.parameters():
            p.requires_grad = True

    backbone.to(device)
    params = [p for p in backbone.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)

    def state_dict_fn() -> dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in backbone.state_dict().items()
            if any(f"layers.{i}." in k for i in unfreeze_layers)
        }

    def weight_report_fn() -> dict[str, Any]:
        return {"unfrozen_layers": list(unfreeze_layers), "n_trainable": n}

    # Wrap backbone so it has forward_hidden and project_head
    class UnfreezeWrapper(nn.Module):
        def __init__(self, bb: nn.Module):
            super().__init__()
            self.bb = bb
            self.cfg = bb.cfg  # type: ignore[attr-defined]

        def forward_hidden(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.bb.forward_hidden(input_ids, attention_mask)  # type: ignore[attr-defined]

        def project_head(self, x: torch.Tensor) -> torch.Tensor:
            return self.bb.lm_head(x)  # type: ignore[attr-defined]

        def load_adapter_state_dict(
            self, state: dict[str, torch.Tensor]
        ) -> None:
            self.bb.load_state_dict(state, strict=False)

    model = UnfreezeWrapper(backbone).to(device)
    return model, params, n, state_dict_fn, weight_report_fn


def _build_specialized_clm(
    args: Any, device: str
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    from pawn.specialized_clm import SpecializedCLM

    d_model = args.d_model or 84
    n_layers = args.n_layers or 2
    n_heads = args.n_heads or 4
    d_ff = d_model * 4
    vocab_size = CLMConfig().vocab_size

    model = SpecializedCLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
    ).to(device)

    params = list(model.parameters())
    n = sum(p.numel() for p in params)

    def state_dict_fn() -> dict[str, torch.Tensor]:
        # ``SpecializedCLM`` ties ``lm_head.weight`` to ``embed.weight``;
        # safetensors rejects shared storage, so drop the tied duplicate.
        # The tying is restored automatically at load time because the
        # constructor re-establishes it before ``load_state_dict``.
        sd = dict(model.state_dict())
        sd.pop("lm_head.weight", None)
        return sd

    def weight_report_fn() -> dict[str, Any]:
        return {}

    return model, params, n, state_dict_fn, weight_report_fn


def _build_rosa_warmup(
    backbone: PAWNCLM,
    args: Any,
    layers: tuple[int, ...] | None,
    device: str,
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    """Build the Phase 1 (LoRA warmup) model for RoSA."""
    from pawn.adapters.rosa import RoSACLM

    rank = args.lora_rank or 4
    model = RoSACLM(
        backbone,
        rank=rank,
        attn_targets=args.lora_targets,
        adapt_ffn=args.lora_ffn,
        layers=layers,
        lora_enabled=True,
        sparse_enabled=False,
    ).to(device)
    params = model.lora_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.adapter_state_dict, model.adapter_weight_report


# ---------------------------------------------------------------------------
# RoSA phase helpers
# ---------------------------------------------------------------------------


def rosa_warmup(
    model: nn.Module,
    train_loader: DataLoader[Any],
    mask_builder: LegalMaskBuilder,
    args: Any,
    device: str,
    amp_dtype: torch.dtype | None,
    logger: MetricsLogger,
    wandb_run: WandbRun | None = None,
) -> int:
    """Phase 1: Train LoRA-only for warmup_steps."""
    lr = args.lr
    lora_params = model.lora_parameters()  # type: ignore[attr-defined]
    optimizer = torch.optim.AdamW(
        lora_params, lr=lr, weight_decay=args.weight_decay
    )
    # GradScaler is only needed for fp16 (underflow risk). bf16 has
    # fp32-range exponents and the scaler just adds per-step overhead.
    scaler = torch.amp.GradScaler() if amp_dtype == torch.float16 else None

    apply_legal_mask = not bool(getattr(args, "disable_legal_mask", False))
    illegal_penalty = float(getattr(args, "illegal_penalty", 0.0))

    model.train()
    step = 0
    total_loss = 0.0
    n_positions = 0

    print(
        f"\n=== RoSA Phase 1: LoRA warm-up ({args.rosa_warmup_steps} steps) ===",
        flush=True,
    )

    while step < args.rosa_warmup_steps:
        for batch in train_loader:
            if step >= args.rosa_warmup_steps:
                break
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            T_batch = int(batch.get("T_actual", ids.shape[1]))
            if "legal_indices" in batch:
                legal_mask = mask_builder.scatter(
                    batch["legal_indices"], ids.shape[0], T=T_batch,
                )
            else:
                legal_mask = mask_builder(batch)

            valid_logits, valid_legal = sparse_forward(
                model, ids, msk, legal_mask, amp_dtype, device,
                apply_legal_mask=apply_legal_mask,
            )
            valid_targets = tgt[msk]
            if valid_targets.shape[0] == 0:
                continue

            loss = compute_adapter_loss(
                valid_logits, valid_targets, valid_legal,
                illegal_penalty=illegal_penalty,
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    lora_params, args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    lora_params, args.max_grad_norm
                )
                optimizer.step()

            n_pos = valid_targets.shape[0]
            total_loss += loss.item() * n_pos
            n_positions += n_pos
            step += 1

            if step % 32 == 0 or step == args.rosa_warmup_steps:
                avg = total_loss / max(n_positions, 1)
                print(
                    f"  Warmup step {step}/{args.rosa_warmup_steps} | loss={avg:.4f}",
                    flush=True,
                )
                log_metrics(
                    wandb_run,
                    {"rosa/warmup_loss": avg, "rosa/phase": 1},
                    step=step,
                )

    print(
        f"  Warm-up complete (avg loss={total_loss / max(n_positions, 1):.4f})"
    )
    return step


def rosa_mask_generation(
    model: nn.Module,
    train_loader: DataLoader[Any],
    mask_builder: LegalMaskBuilder,
    args: Any,
    device: str,
    amp_dtype: torch.dtype | None,
    logger: MetricsLogger,
    wandb_run: WandbRun | None = None,
) -> dict[str, torch.Tensor]:
    """Phase 2: Generate gradient-informed sparse masks."""
    from pawn.adapters.rosa import generate_gradient_masks

    print(
        f"\n=== RoSA Phase 2: Mask generation (density={args.density}, "
        f"alpha={args.grad_alpha}) ===",
        flush=True,
    )

    use_amp = amp_dtype is not None
    masks = generate_gradient_masks(
        model,  # type: ignore[arg-type]
        train_loader,
        mask_builder,
        density=args.density or 0.01,
        alpha=args.grad_alpha,
        device=device,
        use_amp=use_amp,
        max_batches=args.mask_samples,
        apply_legal_mask=not bool(
            getattr(args, "disable_legal_mask", False)
        ),
        illegal_penalty=float(getattr(args, "illegal_penalty", 0.0)),
    )

    total_active = sum(m.sum().item() for m in masks.values())
    total_elements = sum(m.numel() for m in masks.values())
    print(
        f"  Total: {total_active:,.0f} / {total_elements:,} "
        f"({100 * total_active / total_elements:.2f}%)",
        flush=True,
    )
    # Log at the warmup-step boundary so the rosa/* timeline aligns with
    # the Phase 1 warmup loss curve rather than auto-incrementing to an
    # arbitrary position.
    log_metrics(
        wandb_run,
        {
            "rosa/phase": 2,
            "rosa/mask_density": total_active / max(total_elements, 1),
            "rosa/total_active_params": total_active,
            "rosa/total_params": total_elements,
        },
        step=getattr(args, "rosa_warmup_steps", 0),
    )
    return masks


def rosa_build_phase3(
    warmup_model: nn.Module,
    masks: dict[str, torch.Tensor],
    args: Any,
    device: str,
) -> tuple[nn.Module, list[torch.nn.Parameter], int, Any, Any]:
    """Build the Phase 3 model based on rosa_mode."""
    from pawn.adapters.lora import ATTN_PRESETS, _FFN_TARGETS
    from pawn.adapters.rosa import RetroBottleneckCLM
    from pawn.adapters.sparse import SparseCLM, SparseLinear

    mode = args.rosa_mode

    if mode == "rosa":
        warmup_model.set_masks(masks)  # type: ignore[attr-defined]
        warmup_model.reinit_lora()  # type: ignore[attr-defined]
        params = warmup_model.adapter_parameters()  # type: ignore[attr-defined]
        n = sum(p.numel() for p in params)
        return (
            warmup_model,
            params,
            n,
            warmup_model.adapter_state_dict,  # type: ignore[attr-defined]
            warmup_model.adapter_weight_report,  # type: ignore[attr-defined]
        )

    # Retrospective modes: reload backbone
    del warmup_model
    gc.collect()
    torch.cuda.empty_cache()

    backbone = load_backbone(args.checkpoint, device)
    attn_targets = ATTN_PRESETS[args.lora_targets or "qkvo"]

    sparse_model = SparseCLM(
        backbone,
        density=args.density or 0.01,
        attn_targets=attn_targets,
        adapt_ffn=args.sparse_ffn,
    )

    # Overwrite random masks with gradient-derived masks
    for layer_idx in range(len(backbone.layers)):
        block = backbone.get_block(layer_idx)
        for proj_name in attn_targets:
            module = getattr(block.attn, proj_name, None)
            if isinstance(module, SparseLinear):
                key = f"layer{layer_idx}.{proj_name}"
                if key in masks:
                    module.mask.copy_(masks[key])
        if args.sparse_ffn:
            for proj_name in _FFN_TARGETS:
                module = getattr(block.ffn, proj_name, None)
                if isinstance(module, SparseLinear):
                    key = f"layer{layer_idx}.{proj_name}"
                    if key in masks:
                        module.mask.copy_(masks[key])

    if mode == "retro-sparse":
        sparse_model = sparse_model.to(device)
        params = sparse_model.sparse_parameters()
        n = sparse_model.n_active_params()
        return (
            sparse_model,
            params,
            n,
            sparse_model.sparse_state_dict,
            sparse_model.sparse_weight_report,
        )

    # retro-bottleneck
    dim = args.bottleneck_dim or 8
    model = RetroBottleneckCLM(
        sparse_model.backbone,
        bottleneck_dim=dim,
        n_hidden=getattr(args, "bottleneck_n_hidden", 0),
    ).to(device)
    params = model.adapter_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.adapter_state_dict, model.adapter_weight_report


# ---------------------------------------------------------------------------
# Config JSON
# ---------------------------------------------------------------------------


def build_config_json(args: Any, param_count: int) -> dict[str, Any]:
    """Build the normalized config dict with nulls for irrelevant params."""
    layers = parse_layers(args.adapter_layers)

    cfg: dict[str, Any] = {
        "strategy": args.strategy,
        "param_count": param_count,
        # Backbone
        "checkpoint": args.checkpoint
        if args.strategy != "specialized_clm"
        else None,
        "pgn": args.pgn,
        "elo_min": args.elo_min,
        "elo_max": args.elo_max,
        # Data sizing — ``steps_per_epoch`` is canonical (the ``"all"``
        # sentinel is resolved before this is written). Legacy
        # ``max_games`` is recorded only when present without a resolved
        # ``steps_per_epoch`` so old saved configs don't lose that info,
        # but new resolved configs omit it.
        "steps_per_epoch": getattr(args, "steps_per_epoch", None),
        "data_seed": getattr(args, "data_seed", None),
        "max_games": (
            args.max_games
            if getattr(args, "steps_per_epoch", None) is None
            else None
        ),
        "val_games": args.val_games,
        # Training
        "total_steps": args.total_steps,
        "eval_interval": args.eval_interval,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_frac": args.warmup_frac,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "amp_dtype": args.amp_dtype,
        "patience": args.patience,
        # Placement
        "adapter_layers": list(layers) if layers else None,
        # Bottleneck
        "bottleneck_dim": args.bottleneck_dim
        if args.strategy in ("bottleneck", "rosa")
        else None,
        # Only ``retro-bottleneck`` (and the plain ``bottleneck`` strategy)
        # actually instantiates Houlsby MLPs; ``retro-sparse`` and joint
        # ``rosa`` don't, so persisting the field there would be a misleading
        # artifact in saved configs.
        "bottleneck_n_hidden": getattr(args, "bottleneck_n_hidden", 0)
        if args.strategy == "bottleneck"
        or (
            args.strategy == "rosa"
            and getattr(args, "rosa_mode", None) == "retro-bottleneck"
        )
        else None,
        "adapt_attn": (not args.no_adapt_attn)
        if args.strategy == "bottleneck"
        else None,
        "adapt_ffn": (not args.no_adapt_ffn)
        if args.strategy == "bottleneck"
        else None,
        # Low-rank
        "lora_rank": args.lora_rank
        if args.strategy in ("lora", "rosa", "hybrid")
        else None,
        "lora_targets": args.lora_targets
        if args.strategy in ("lora", "rosa", "hybrid")
        else None,
        "lora_ffn": args.lora_ffn
        if args.strategy in ("lora", "rosa", "hybrid")
        else None,
        # Sparse
        "density": args.density
        if args.strategy in ("sparse", "rosa")
        else None,
        "sparse_targets": args.sparse_targets
        if args.strategy in ("sparse", "rosa")
        else None,
        "sparse_ffn": args.sparse_ffn
        if args.strategy in ("sparse", "rosa")
        else None,
        # Mask generation
        "rosa_mode": args.rosa_mode if args.strategy == "rosa" else None,
        "rosa_warmup_steps": args.rosa_warmup_steps
        if args.strategy == "rosa"
        else None,
        "mask_samples": args.mask_samples
        if args.strategy == "rosa"
        else None,
        "grad_alpha": args.grad_alpha if args.strategy == "rosa" else None,
        # FiLM
        "use_output_film": args.use_output_film
        if args.strategy in ("film", "hybrid")
        else None,
        # From-scratch
        "d_model": args.d_model
        if args.strategy == "specialized_clm"
        else None,
        "n_layers": args.n_layers
        if args.strategy == "specialized_clm"
        else None,
        "n_heads": args.n_heads
        if args.strategy == "specialized_clm"
        else None,
        # Unfreeze
        "unfreeze_layers": list(
            parse_layers(args.unfreeze_layers) or ()
        )
        if args.strategy == "unfreeze"
        else None,
        # Legality handling
        "disable_legal_mask": bool(
            getattr(args, "disable_legal_mask", False)
        ),
        "illegal_penalty": float(getattr(args, "illegal_penalty", 0.0)),
    }
    return cfg


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    trainable_params: list[torch.nn.Parameter],
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    mask_builder: LegalMaskBuilder,
    val_legal_indices: list[torch.Tensor],
    logger: MetricsLogger,
    args: Any,
    device: str,
    amp_dtype: torch.dtype | None,
    gpu_cfg: dict[str, Any],
    state_dict_fn: Any,
    weight_report_fn: Any,
    param_count: int,
    wandb_run: WandbRun | None = None,
) -> tuple[float, dict[str, Any]]:
    """Unified training loop for all strategies."""
    from pawn import model as model_module
    from pawn.checkpoint import save_adapter_checkpoint

    # Set SDPA backend (escape hatch for ROCm flash attn debugging). The
    # full-step ``torch.compile`` wrap below replaces the per-call
    # ``forward_hidden`` compile that ``apply_gpu_config`` used to do —
    # we still need the side effect of pinning ``SDPA_BACKEND`` before
    # any compile traces, since compiled code captures the backend at
    # trace time.
    if gpu_cfg.get("sdpa_backend") is not None:
        model_module.SDPA_BACKEND = gpu_cfg["sdpa_backend"]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ``steps_per_epoch`` is canonical and exact: ``run_adapter`` resolves
    # ``args.steps_per_epoch`` to an integer (or the legacy
    # ``args.max_games // batch_size``) before constructing the loader.
    # The DataLoader's length is therefore deterministic and the LR
    # scheduler's ``total_steps`` matches the number of steps the loop
    # will actually take.
    steps_per_epoch = int(args.steps_per_epoch)
    if args.total_steps:
        total_steps = args.total_steps
        expected = args.epochs * steps_per_epoch
        if total_steps != expected:
            raise ValueError(
                f"total_steps={total_steps} != epochs × steps_per_epoch "
                f"= {args.epochs} × {steps_per_epoch} = {expected}. "
                "Set total_steps to match, or omit it and let it be "
                "derived from epochs and steps_per_epoch."
            )
    else:
        total_steps = args.epochs * steps_per_epoch

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(args.warmup_frac * total_steps)
    schedule = getattr(args, "lr_schedule", "cosine")
    decay_frac = getattr(args, "decay_frac", 0.1)
    decay_steps = (
        int(decay_frac * total_steps)
        if schedule in ("wsd", "infinite")
        else None
    )
    cooldown_steps = (
        int(getattr(args, "cooldown_frac", 0.2) * total_steps)
        if schedule == "infinite"
        else None
    )
    scheduler = build_scheduler(
        optimizer, warmup_steps, total_steps,
        schedule=schedule, decay_steps=decay_steps,
        wsd_decay_shape=getattr(args, "wsd_decay_shape", "linear"),
        cooldown_steps=cooldown_steps,
        stable_lr_ratio=float(getattr(args, "stable_lr_ratio", 0.1)),
    )
    # GradScaler is only needed for fp16 (underflow risk). bf16 has
    # fp32-range exponents and the scaler just adds per-step overhead.
    scaler = torch.amp.GradScaler() if amp_dtype == torch.float16 else None

    # Legality options — default to current behavior (mask on, no penalty)
    # when the args namespace predates these fields (e.g. callers still
    # constructing a bare ``argparse.Namespace`` by hand).
    apply_legal_mask = not bool(getattr(args, "disable_legal_mask", False))
    illegal_penalty = float(getattr(args, "illegal_penalty", 0.0))

    # Build the (possibly compiled) per-step function. Closes over
    # ``apply_legal_mask`` / ``illegal_penalty`` / ``amp_dtype`` so
    # dynamo specializes one graph per legality regime — which is fine,
    # both flags are set at run start and don't change mid-run.
    step_fn = build_compiled_step(
        model,
        apply_legal_mask=apply_legal_mask,
        illegal_penalty=illegal_penalty,
        amp_dtype=amp_dtype,
        use_compile=bool(gpu_cfg.get("use_compile", False)),
    )

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    if args.resume:
        print(f"Resuming from: {args.resume}")
        from pawn.checkpoint import load_adapter_checkpoint

        ckpt = load_adapter_checkpoint(args.resume, device=device)
        # Try to load adapter state
        if "adapter_state_dict" in ckpt:
            state = ckpt["adapter_state_dict"]
            if hasattr(model, "load_adapter_state_dict"):
                model.load_adapter_state_dict(state)  # type: ignore[operator]
            elif hasattr(model, "load_lora_state_dict"):
                model.load_lora_state_dict(state)  # type: ignore[operator]
            elif hasattr(model, "load_film_state_dict"):
                model.load_film_state_dict(state)  # type: ignore[operator]
            elif hasattr(model, "load_sparse_state_dict"):
                model.load_sparse_state_dict(state)  # type: ignore[operator]
            else:
                model.load_state_dict(state, strict=False)
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        # ``steps_per_epoch`` is exact under cache-first; refuse to resume
        # silently across a different sizing. (Catches max_games-vs-cache
        # mismatches that would otherwise put the resumed run on a
        # different LR trajectory than the original.)
        saved_spe = ckpt.get("steps_per_epoch")
        if saved_spe is not None and int(saved_spe) != steps_per_epoch:
            raise ValueError(
                f"Resume mismatch: checkpoint was saved with "
                f"steps_per_epoch={saved_spe}, but the current run has "
                f"steps_per_epoch={steps_per_epoch}. Use the same data "
                "sizing on resume."
            )
        # ``epoch`` and ``skip_batches`` are derived from ``global_step``
        # and ``steps_per_epoch`` (no FF iteration needed). The saved
        # ``epoch`` field is informational; the canonical state is
        # ``global_step``. See :func:`resume_state` for the contract.
        start_epoch, resume_skip_batches = resume_state(
            global_step, steps_per_epoch
        )
        # Honor a saved ``data_seed`` if present so the per-epoch
        # permutation matches the original run.
        saved_data_seed = ckpt.get("data_seed")
        if saved_data_seed is not None:
            args.data_seed = int(saved_data_seed)
        del ckpt

    # Record the step we enter the loop at so the final save can tell
    # whether any training happened this session. On resume-past-end
    # (epoch budget exhausted on a prior run) the loop body never
    # executes and ``val_metrics`` would otherwise hold the baseline
    # eval — writing that as the final state would be misleading.
    initial_step = global_step

    # Baseline eval
    if not args.resume:
        print("\nBaseline (before training):")
        baseline = evaluate(
            model,
            val_loader,
            mask_builder,
            device,
            amp_dtype=amp_dtype,
            precomputed_indices=val_legal_indices,
            apply_legal_mask=apply_legal_mask,
            illegal_penalty=illegal_penalty,
        )
        print(
            f"  loss={baseline['loss']:.4f}, top1={baseline['top1_accuracy']:.4%}, "
            f"top5={baseline['top5_accuracy']:.4%}"
        )
        baseline_record: dict[str, Any] = {
            "epoch": -1,
            "train_loss": baseline["loss"],
            "train_top1": baseline["top1_accuracy"],
            "val_loss": baseline["loss"],
            "val_top1": baseline["top1_accuracy"],
            "val_top5": baseline["top5_accuracy"],
            "val_illegal_pred_rate": baseline["illegal_pred_rate"],
            "val_illegal_prob_mass": baseline["illegal_prob_mass"],
        }
        logger.log_train(step=0, **baseline_record)
        log_metrics(wandb_run, baseline_record, step=0)
        val_metrics = baseline
    else:
        val_metrics = evaluate(
            model,
            val_loader,
            mask_builder,
            device,
            amp_dtype=amp_dtype,
            precomputed_indices=val_legal_indices,
            apply_legal_mask=apply_legal_mask,
            illegal_penalty=illegal_penalty,
        )

    # Graceful shutdown
    _shutdown = False

    def _handle_signal(signum: int, frame: Any) -> None:
        nonlocal _shutdown
        _shutdown = True

    old_term = signal.signal(signal.SIGTERM, _handle_signal)
    old_int = signal.signal(signal.SIGINT, _handle_signal)

    ckpt_dir = logger.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    hf_branch = f"run/{logger.run_dir.name}" if args.hf_repo else None

    # Background pusher for HF uploads. Construct eagerly even without an
    # ``hf_repo`` so the shutdown path can ``wait()`` unconditionally; the
    # pool never sees work in that case.
    from pawn.checkpoint import BackgroundCheckpointPusher
    hf_pusher = BackgroundCheckpointPusher(thread_name_prefix="hf-adapter")
    # Bucket pushes go on a separate single-worker pool so a slow
    # bucket sync doesn't block the model-repo branch push (and vice
    # versa). Both are no-ops when the corresponding config field is
    # unset; constructing eagerly lets the shutdown path
    # ``.wait()`` unconditionally.
    bucket_pusher = BackgroundCheckpointPusher(thread_name_prefix="hf-bucket")

    eval_interval = args.eval_interval
    step_limit = args.total_steps
    pause_step = args.pause_after_steps
    # Tracks why the training loop exited; ``schedule_health.json``
    # records this so post-hoc audits can distinguish "ran to
    # completion" from SIGTERM, patience, or pause boundaries.
    stop_reason = "completed"

    print(f"\nTraining for up to {args.epochs} epochs ({total_steps} steps)")
    print(
        f"  Warmup: {warmup_steps} steps, LR: {args.lr}, AMP: {args.amp_dtype}"
    )
    if not apply_legal_mask:
        print(
            f"  Legal-move masking DISABLED; illegal_penalty={illegal_penalty}"
        )

    def _do_eval() -> dict[str, float]:
        return evaluate(
            model,
            val_loader,
            mask_builder,
            device,
            amp_dtype=amp_dtype,
            precomputed_indices=val_legal_indices,
            apply_legal_mask=apply_legal_mask,
            illegal_penalty=illegal_penalty,
        )

    def _save_step_checkpoint(vm: dict[str, float], ep: int) -> None:
        """Save a step-tagged adapter checkpoint.

        Matches ``pawn.trainer.Trainer.save_checkpoint``: one
        ``step_{global_step:08d}/`` directory per save, never overwritten.
        Downstream tools discover the best-by-val-loss step via
        ``metrics.jsonl``; see ``pawn.checkpoint.find_best_adapter_step``
        (adapter records use ``type="train"`` with a ``val_loss`` kwarg —
        the pretraining helper ``scripts/export_hf_repo.find_best_step``
        filters on ``type="val"`` and does not apply).

        Idempotent: if ``step_path`` already exists (e.g. the same step
        was already saved by the interval hook and the training loop is
        now at termination), this is a no-op. Callers can invoke freely
        without a preflight ``exists()`` check.
        """
        step_path = ckpt_dir / f"step_{global_step:08d}"
        if step_path.exists():
            return
        save_adapter_checkpoint(
            step_path,
            state_dict_fn(),
            config=build_config_json(args, param_count),
            epoch=ep,
            step=global_step,
            val_metrics=vm,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            extra={
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "steps_per_epoch": steps_per_epoch,
                "data_seed": getattr(args, "data_seed", None),
            },
        )
        if args.hf_repo and hf_branch:
            hf_pusher.submit(
                step_path,
                args.hf_repo,
                hf_branch,
                step=global_step,
                metrics_path=str(logger.metrics_path),
            )
        # Bucket push runs in parallel with the model-repo branch push
        # (when both are set). The bucket path is a self-contained
        # snapshot — checkpoint dir + a copy of metrics.jsonl — under
        # ``logs/<run_slug>/`` inside the bucket.
        if getattr(args, "hf_bucket", None):
            bucket_pusher.submit_bucket(
                step_path,
                args.hf_bucket,
                run_slug=logger.run_dir.name,
                step=global_step,
                metrics_path=str(logger.metrics_path),
            )

    # Resume: ``global_step`` is canonical. ``start_epoch`` and
    # ``resume_skip_batches`` were derived from it above (or set to 0 / 0
    # for fresh runs below). The :class:`SeededEpochSampler` consumes
    # those via ``set_epoch(epoch, skip_batches=...)`` and produces the
    # exact tail of the original epoch's permutation — no FF iteration.
    if not args.resume:
        resume_skip_batches = 0  # type: ignore[assignment]

    # Bind the train sampler — :class:`SeededEpochSampler` is the
    # contract here. The DataLoader exposes it via ``.sampler``.
    from pawn.lichess_cache import SeededEpochSampler

    sampler = train_loader.sampler
    if not isinstance(sampler, SeededEpochSampler):
        raise TypeError(
            f"adapter trainer requires SeededEpochSampler, got "
            f"{type(sampler).__name__}. Build the loader via "
            "scripts/train.py:run_adapter so the sampler is wired in."
        )

    epoch = start_epoch  # default if loop doesn't execute (resume past end)
    for epoch in range(start_epoch, args.epochs):
        skip = resume_skip_batches if epoch == start_epoch else 0
        sampler.set_epoch(epoch, skip_batches=skip)
        if skip:
            print(
                f"  Resume: epoch {epoch}, skipping the first {skip:,} "
                f"batches via deterministic permutation slice (no FF).",
                flush=True,
            )
        model.train()
        # Accumulate loss/top1/positions as GPU tensors so the training
        # loop never issues a ``.item()`` per step. Each ``.item()`` is
        # a blocking host-device sync that prevents the GPU from running
        # batch N+1 while the CPU reads the scalar for batch N — on
        # ROCm this shows up as ~30% GPU utilization even when the
        # data pipeline is idle.
        epoch_loss_t = torch.zeros((), device=device)
        # top-1 / position accumulators hold counts, so they're int64
        # to match ``(preds == targets).sum()`` and ``msk.sum()``.
        # Keeping the dtype explicit avoids a silent float32 cast on
        # ``+=``.
        epoch_top1_t = torch.zeros((), device=device, dtype=torch.int64)
        epoch_positions_t = torch.zeros((), device=device, dtype=torch.int64)
        log_loss_t = torch.zeros((), device=device)
        log_top1_t = torch.zeros((), device=device, dtype=torch.int64)
        log_positions_t = torch.zeros((), device=device, dtype=torch.int64)
        t0 = time.time()

        # The sampler's permutation is set; iterating yields exactly the
        # remaining-after-skip indices. Persistent workers (when enabled)
        # are kept alive across epochs and pick up the new permutation
        # on each ``iter(train_loader)``.
        loader_iter = iter(train_loader)

        for batch in loader_iter:
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            T_batch = int(batch.get("T_actual", ids.shape[1]))
            legal_mask = mask_builder.scatter(
                batch["legal_indices"], ids.shape[0], T=T_batch,
            )

            loss, top1_sum = step_fn(ids, msk, legal_mask, tgt)

            # Detach + clone before any post-step work. Under
            # ``mode="reduce-overhead"`` cudagraph trees, the tensors
            # returned from a compiled step alias the graph's static
            # output buffers; the next ``step_fn`` call overwrites
            # them. ``loss.backward()`` consumes ``loss`` before the
            # next call (safe), but the diagnostic accumulators below
            # would otherwise read stale values once the next step
            # runs. Cloning into private storage decouples them.
            loss_for_log = loss.detach().clone()
            top1_sum = top1_sum.clone()

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                )
                optimizer.step()
            scheduler.step()

            # Position count from the loss mask itself — keeps things on
            # GPU and avoids any per-step sync.
            n_pos_t = msk.sum()
            loss_sum = loss_for_log * n_pos_t.to(loss_for_log.dtype)
            epoch_loss_t += loss_sum
            epoch_top1_t += top1_sum
            epoch_positions_t += n_pos_t
            log_loss_t += loss_sum
            log_top1_t += top1_sum
            log_positions_t += n_pos_t
            global_step += 1

            if global_step % args.log_interval == 0:
                # Sync only at the log boundary. ``log_positions`` may
                # be 0 if the bucket happened to contain entirely empty
                # rows — guard against the zero-division.
                log_pos = int(log_positions_t.item())
                if log_pos > 0:
                    avg_loss = (log_loss_t / log_pos).item()
                    avg_top1 = (log_top1_t.float() / log_pos).item()
                    log_loss_t.zero_()
                    log_top1_t.zero_()
                    log_positions_t.zero_()
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  step {global_step:6d} | loss={avg_loss:.4f} "
                        f"top1={avg_top1:.4%} lr={lr:.2e}",
                        flush=True,
                    )
                    logger.log_train(
                        step=global_step,
                        epoch=epoch,
                        lr=lr,
                        train_loss=avg_loss,
                        train_top1=avg_top1,
                    )
                    log_metrics(
                        wandb_run,
                        {
                            "train/loss": avg_loss,
                            "train/top1": avg_top1,
                            "train/lr": lr,
                        },
                        step=global_step,
                    )

            if eval_interval and global_step % eval_interval == 0:
                val_metrics = _do_eval()
                print(
                    f"  [eval @ step {global_step}] "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_top1={val_metrics['top1_accuracy']:.4%} "
                    f"val_top5={val_metrics['top5_accuracy']:.4%} "
                    f"illegal_pred={val_metrics['illegal_pred_rate']:.2%}"
                )
                try:
                    report = weight_report_fn()
                except Exception as e:
                    print(f"WARNING: weight_report_fn failed: {e}", flush=True)
                    report = {}
                eval_record: dict[str, Any] = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "val_loss": val_metrics["loss"],
                    "val_top1": val_metrics["top1_accuracy"],
                    "val_top5": val_metrics["top5_accuracy"],
                    "val_illegal_pred_rate": val_metrics["illegal_pred_rate"],
                    "val_illegal_prob_mass": val_metrics["illegal_prob_mass"],
                    **report,
                }
                logger.log_train(step=global_step, **eval_record)
                log_metrics(wandb_run, eval_record, step=global_step)
                # Saves triggered by the eval that just ran get fresh
                # ``val_metrics`` in ``training_state.json``. Non-eval-aligned
                # checkpoint_interval boundaries are handled in a separate
                # block below so users can request sub-eval-frequency saves
                # for LR-probing workflows (e.g. ``checkpoint_interval=500``
                # with ``eval_interval=2000``).
                new_best = val_metrics["loss"] < best_val_loss
                at_interval = global_step % args.checkpoint_interval == 0
                if new_best:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                if new_best or at_interval:
                    _save_step_checkpoint(val_metrics, epoch)
                if (
                    not new_best
                    and args.patience is not None
                    and patience_counter >= args.patience
                ):
                    print(
                        f"\n  Early stopping at step {global_step} "
                        f"(patience={args.patience})"
                    )
                    stop_reason = "patience"
                    break
                model.train()

            # Out-of-eval-block interval saves for the
            # ``checkpoint_interval < eval_interval`` case. The embedded
            # ``val_metrics`` is from the most recent eval (possibly from
            # earlier in training) so metadata may be stale — the model
            # weights, optimizer state, and scheduler step at
            # ``step_{global_step:08d}/`` are always current. Callers that
            # need authoritative per-step val stats should cross-reference
            # ``metrics.jsonl``. Idempotent: no-op when the step already
            # has a checkpoint from the eval-aligned path above.
            elif (
                args.checkpoint_interval
                and global_step % args.checkpoint_interval == 0
            ):
                _save_step_checkpoint(val_metrics, epoch)

            if step_limit and global_step >= step_limit:
                stop_reason = "step_limit"
                break
            if pause_step and global_step >= pause_step:
                stop_reason = "paused"
                break
            if _shutdown:
                stop_reason = "sigterm"
                break

        dt = time.time() - t0
        # One sync at end of epoch to pull the accumulated totals.
        denom = max(int(epoch_positions_t.item()), 1)
        train_loss = (epoch_loss_t / denom).item()
        train_top1 = (epoch_top1_t.float() / denom).item()

        do_val = not eval_interval and (
            (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        )
        if do_val:
            val_metrics = _do_eval()

        try:
            report = weight_report_fn()
        except Exception as e:
            print(f"WARNING: weight_report_fn failed: {e}", flush=True)
            report = {}
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "val_illegal_pred_rate": val_metrics.get("illegal_pred_rate", 0.0),
            "val_illegal_prob_mass": val_metrics.get("illegal_prob_mass", 0.0),
            "epoch_time_s": dt,
            **report,
        }
        logger.log_train(step=global_step, **epoch_record)
        log_metrics(wandb_run, epoch_record, step=global_step)

        print(
            f"  Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} train_top1={train_top1:.4%} | "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_accuracy']:.4%} "
            f"val_top5={val_metrics['top5_accuracy']:.4%} | {dt:.1f}s"
        )

        if do_val:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if args.patience is not None and patience_counter >= args.patience:
                    print(
                        f"\n  Early stopping at epoch {epoch} "
                        f"(patience={args.patience})"
                    )
                    stop_reason = "patience"
                    break
            _save_step_checkpoint(val_metrics, epoch)

        if eval_interval and args.patience is not None and patience_counter >= args.patience:
            # ``stop_reason`` was already set to ``"patience"`` inside
            # the batch loop's eval block.
            break
        if step_limit and global_step >= step_limit:
            print(f"\n  Reached step limit ({step_limit})")
            stop_reason = "step_limit"
            break
        if pause_step and global_step >= pause_step:
            print(f"\n  Paused at step {global_step} (pause_after_steps={pause_step})")
            stop_reason = "paused"
            break
        if _shutdown:
            print("Shutdown requested, saving checkpoint...")
            stop_reason = "sigterm"
            break

    # Final checkpoint — only write if this session actually trained. On
    # resume-past-end the loop doesn't execute, so ``val_metrics`` still
    # holds the baseline eval; saving it as the final state would be
    # misleading. ``_save_step_checkpoint`` itself is idempotent, so
    # there's no separate existence check needed here.
    if global_step > initial_step:
        _save_step_checkpoint(val_metrics, epoch)

    # Restore the caller's signal handlers before draining the HF pusher.
    # ``hf_pusher.wait()`` can block for seconds-to-minutes on a slow
    # upload; handing SIGINT/SIGTERM back first lets a second Ctrl-C/TERM
    # actually interrupt instead of getting swallowed by our latch.
    signal.signal(signal.SIGTERM, old_term)
    signal.signal(signal.SIGINT, old_int)

    # Drain the background HF pusher before returning so pending uploads
    # aren't lost when the process exits. This is a no-op when hf_repo
    # was unset.
    hf_pusher.wait()
    bucket_pusher.wait()

    # Write ``schedule_health.json`` next to ``metrics.jsonl`` so
    # post-hoc audits can distinguish "ran to completion" from
    # "stopped early with LR mid-decay". On cache-first the
    # ``actual_total_steps != planned_total_steps`` case only happens
    # for SIGTERM / patience / pause / explicit step_limit; the silent
    # truncation that motivated this file (streaming `steps_per_epoch`
    # estimate drift) is no longer reachable.
    #
    # Special case: resume-past-end (loop never executed because
    # ``start_epoch >= args.epochs``). The default ``stop_reason``
    # initializer is ``"completed"``, which combined with the unchanged
    # ``global_step`` would trip the structural-bug banner — but
    # there's no bug here, the run is simply already done.
    if global_step == initial_step and stop_reason == "completed":
        stop_reason = "resume_no_op"
    write_schedule_health(
        logger.run_dir,
        schedule=getattr(args, "lr_schedule", "cosine"),
        planned_total_steps=int(total_steps),
        actual_total_steps=int(global_step),
        lr_peak=float(args.lr),
        actual_final_lr=float(optimizer.param_groups[0]["lr"]),
        reason_for_stop=stop_reason,
    )

    return best_val_loss, val_metrics
