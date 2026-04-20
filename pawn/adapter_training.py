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
from pawn.gpu import apply_gpu_config
from pawn.lichess_data import (
    LegalMaskBuilder,
    compute_legal_indices,
)
from pawn.logging import MetricsLogger
from pawn.model import PAWNCLM

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
) -> torch.optim.lr_scheduler.LambdaLR:
    """Dispatch one of the supported LR schedules by name.

    Supported schedules: ``"cosine"``, ``"wsd"``, ``"constant"``,
    ``"one_cycle"``. See the individual scheduler docstrings for
    parameter meaning. ``warmup_steps`` is reused as the ramp-up for
    ``one_cycle`` — set it to ~0.3 * total_steps for the canonical
    Smith shape.
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
    raise ValueError(f"Unknown lr_schedule: {schedule!r}")


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
        if precomputed_indices is not None:
            legal_mask = mask_builder.scatter(
                precomputed_indices[i], ids.shape[0]
            )
        elif "legal_indices" in batch:
            legal_mask = mask_builder.scatter(
                batch["legal_indices"], ids.shape[0]
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


def precompute_val_masks(
    val_loader: DataLoader[Any],
    mask_builder: LegalMaskBuilder,
    vocab_size: int,
) -> list[torch.Tensor]:
    indices: list[torch.Tensor] = []
    for batch in val_loader:
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
        return model.state_dict()

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
            if "legal_indices" in batch:
                legal_mask = mask_builder.scatter(
                    batch["legal_indices"], ids.shape[0]
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
        "max_games": args.max_games,
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
) -> tuple[float, dict[str, Any]]:
    """Unified training loop for all strategies."""
    from pawn import model as model_module
    from pawn.checkpoint import save_adapter_checkpoint

    # Compile
    model.forward_hidden = apply_gpu_config(  # type: ignore[attr-defined]
        gpu_cfg, model_module, model.forward_hidden  # type: ignore[attr-defined]
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Compute total steps
    streaming = hasattr(train_loader.dataset, "shard_files")
    if args.total_steps:
        total_steps = args.total_steps
    elif streaming:
        est_games = min(
            args.max_games or 1_000_000,
            len(train_loader.dataset.shard_files) * 60_000,  # type: ignore[union-attr]
        )
        total_steps = args.epochs * (est_games // args.batch_size)
    else:
        total_steps = args.epochs * len(train_loader)

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(args.warmup_frac * total_steps)
    schedule = getattr(args, "lr_schedule", "cosine")
    decay_frac = getattr(args, "decay_frac", 0.1)
    decay_steps = int(decay_frac * total_steps) if schedule == "wsd" else None
    scheduler = build_scheduler(
        optimizer, warmup_steps, total_steps,
        schedule=schedule, decay_steps=decay_steps,
        wsd_decay_shape=getattr(args, "wsd_decay_shape", "linear"),
    )
    # GradScaler is only needed for fp16 (underflow risk). bf16 has
    # fp32-range exponents and the scaler just adds per-step overhead.
    scaler = torch.amp.GradScaler() if amp_dtype == torch.float16 else None

    # Legality options — default to current behavior (mask on, no penalty)
    # when the args namespace predates these fields (e.g. callers still
    # constructing a bare ``argparse.Namespace`` by hand).
    apply_legal_mask = not bool(getattr(args, "disable_legal_mask", False))
    illegal_penalty = float(getattr(args, "illegal_penalty", 0.0))

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
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        del ckpt

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
        logger.log_train(
            step=0,
            epoch=-1,
            train_loss=baseline["loss"],
            train_top1=baseline["top1_accuracy"],
            val_loss=baseline["loss"],
            val_top1=baseline["top1_accuracy"],
            val_top5=baseline["top5_accuracy"],
            val_illegal_pred_rate=baseline["illegal_pred_rate"],
            val_illegal_prob_mass=baseline["illegal_prob_mass"],
        )
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

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ckpt_dir = logger.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    hf_branch = f"run/{logger.run_dir.name}" if args.hf_repo else None

    eval_interval = args.eval_interval
    step_limit = args.total_steps
    pause_step = args.pause_after_steps

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
        ``metrics.jsonl``. See ``scripts/export_hf_repo.find_best_step``.
        """
        step_path = ckpt_dir / f"step_{global_step:08d}"
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
            },
        )
        if args.hf_repo and hf_branch:
            from pawn.checkpoint import push_checkpoint_to_hf

            try:
                push_checkpoint_to_hf(
                    step_path,
                    args.hf_repo,
                    hf_branch,
                    step=global_step,
                )
            except Exception as e:
                print(f"WARNING: HF push failed: {e}")

    epoch = start_epoch  # default if loop doesn't execute (resume past end)
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)  # type: ignore[union-attr]
        model.train()
        # Accumulate loss/top1 as GPU tensors so the training loop never
        # issues a ``.item()`` per step. Each ``.item()`` is a blocking
        # host-device sync that prevents the GPU from running batch N+1
        # while the CPU reads the scalar for batch N — on ROCm this shows
        # up as ~30% GPU utilization even when the data pipeline is idle.
        epoch_loss_t = torch.zeros((), device=device)
        # top-1 accumulators hold a count of correct predictions, so
        # they're int64 to match ``(preds == targets).sum()``. Keeping
        # the dtype explicit avoids a silent float32 cast on ``+=``.
        epoch_top1_t = torch.zeros((), device=device, dtype=torch.int64)
        epoch_positions = 0
        log_loss_t = torch.zeros((), device=device)
        log_top1_t = torch.zeros((), device=device, dtype=torch.int64)
        log_positions = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            legal_mask = mask_builder.scatter(
                batch["legal_indices"], ids.shape[0]
            )

            valid_logits, valid_legal = sparse_forward(
                model, ids, msk, legal_mask, amp_dtype, device,
                apply_legal_mask=apply_legal_mask,
            )
            valid_targets = tgt[msk]
            loss = compute_adapter_loss(
                valid_logits, valid_targets, valid_legal,
                illegal_penalty=illegal_penalty,
            )

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

            # Stay on GPU: no ``.item()`` inside the hot loop.
            with torch.no_grad():
                preds = valid_logits.argmax(dim=-1)
                top1_sum = (preds == valid_targets).sum()

            n_pos = valid_targets.shape[0]
            loss_sum = loss.detach() * n_pos
            epoch_loss_t += loss_sum
            epoch_top1_t += top1_sum
            epoch_positions += n_pos
            log_loss_t += loss_sum
            log_top1_t += top1_sum
            log_positions += n_pos
            global_step += 1

            if global_step % args.log_interval == 0 and log_positions > 0:
                avg_loss = (log_loss_t / log_positions).item()
                avg_top1 = (log_top1_t / log_positions).item()
                log_loss_t.zero_()
                log_top1_t.zero_()
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
                log_positions = 0

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
                except Exception:
                    report = {}
                logger.log_train(
                    step=global_step,
                    epoch=epoch,
                    lr=optimizer.param_groups[0]["lr"],
                    val_loss=val_metrics["loss"],
                    val_top1=val_metrics["top1_accuracy"],
                    val_top5=val_metrics["top5_accuracy"],
                    val_illegal_pred_rate=val_metrics["illegal_pred_rate"],
                    val_illegal_prob_mass=val_metrics["illegal_prob_mass"],
                    **report,
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if args.patience is not None and patience_counter >= args.patience:
                        print(
                            f"\n  Early stopping at step {global_step} "
                            f"(patience={args.patience})"
                        )
                        break
                model.train()

            if (
                args.checkpoint_interval
                and global_step % args.checkpoint_interval == 0
            ):
                _save_step_checkpoint(val_metrics, epoch)

            if step_limit and global_step >= step_limit:
                break
            if pause_step and global_step >= pause_step:
                break
            if _shutdown:
                break

        dt = time.time() - t0
        # One sync at end of epoch to pull the accumulated totals.
        denom = max(epoch_positions, 1)
        train_loss = (epoch_loss_t / denom).item()
        train_top1 = (epoch_top1_t / denom).item()

        do_val = not eval_interval and (
            (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        )
        if do_val:
            val_metrics = _do_eval()

        try:
            report = weight_report_fn()
        except Exception:
            report = {}
        logger.log_train(
            step=global_step,
            epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            train_top1=train_top1,
            val_loss=val_metrics["loss"],
            val_top1=val_metrics["top1_accuracy"],
            val_top5=val_metrics["top5_accuracy"],
            val_illegal_pred_rate=val_metrics.get("illegal_pred_rate", 0.0),
            val_illegal_prob_mass=val_metrics.get("illegal_prob_mass", 0.0),
            epoch_time_s=dt,
            **report,
        )

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
                    break
            _save_step_checkpoint(val_metrics, epoch)

        if eval_interval and args.patience is not None and patience_counter >= args.patience:
            break  # step-based early stopping triggered inside batch loop
        if step_limit and global_step >= step_limit:
            print(f"\n  Reached step limit ({step_limit})")
            break
        if pause_step and global_step >= pause_step:
            print(f"\n  Paused at step {global_step} (pause_after_steps={pause_step})")
            break
        if _shutdown:
            print("Shutdown requested, saving checkpoint...")
            break

    # Final checkpoint — only write if this step hasn't already been saved
    # by the interval/epoch hooks above.
    final_path = ckpt_dir / f"step_{global_step:08d}"
    if not final_path.exists():
        _save_step_checkpoint(val_metrics, epoch)

    return best_val_loss, val_metrics
