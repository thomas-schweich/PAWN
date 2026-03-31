#!/usr/bin/env python3
"""Unified adapter training script for behavioral cloning on Lichess games.

Trains any adapter strategy on a frozen PAWN backbone (or from scratch)
to predict human moves in a given Elo band. Outputs a normalized JSON
config alongside metrics.jsonl for Optuna-driven sweeps.

Strategies:
  bottleneck  -- Houlsby residual MLP adapters
  lora        -- Low-rank attention/FFN adaptation
  film        -- Feature-wise linear modulation
  sparse      -- Random binary mask weight perturbations
  rosa        -- Gradient-informed sparse + LoRA (3-phase)
  hybrid      -- LoRA + FiLM combined
  specialized_clm -- From-scratch standalone transformer (no backbone)
  unfreeze    -- Fine-tune top N backbone layers directly

Usage:
    python scripts/train_adapter.py --strategy bottleneck \
        --checkpoint thomas-schweich/pawn-base \
        --pgn thomas-schweich/pawn-lichess-full \
        --bottleneck-dim 610 --adapter-layers 4,5,6,7 \
        --lr 5e-4 --batch-size 256 --total-steps 100000 \
        --elo-min 1800 --elo-max 1900 --local-checkpoints
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pawn.config import CLMConfig, PAD_TOKEN
from pawn.model import PAWNCLM
from pawn.logging import MetricsLogger
from pawn.gpu import configure_gpu, apply_gpu_config

from pawn.lichess_data import (
    compute_legal_indices,
    prepare_lichess_dataset,
    LegalMaskBuilder,
    LegalMaskCollate,
    LichessDataset,
)

STRATEGIES = [
    "bottleneck", "lora", "film", "sparse",
    "rosa", "hybrid", "specialized_clm", "unfreeze",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified adapter training for PAWN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Strategy
    p.add_argument("--strategy", type=str, required=True, choices=STRATEGIES)

    # Backbone
    p.add_argument("--checkpoint", type=str, default="thomas-schweich/pawn-base",
                    help="PAWN checkpoint (HF repo ID or local path). Ignored for tiny.")
    p.add_argument("--pgn", type=str, required=True,
                    help="Lichess data (HF repo ID or local parquet directory)")
    p.add_argument("--log-dir", type=str, default=None,
                    help="Parent log directory (default: <project>/logs)")

    # Placement (shared across adapter strategies)
    p.add_argument("--adapter-layers", type=str, default=None,
                    help="Comma-separated layer indices for adapters (default: all)")

    # Bottleneck component
    p.add_argument("--bottleneck-dim", type=int, default=None,
                    help="Bottleneck hidden dimension")
    p.add_argument("--no-adapt-attn", action="store_true",
                    help="Skip bottleneck after attention sublayer")
    p.add_argument("--no-adapt-ffn", action="store_true",
                    help="Skip bottleneck after FFN sublayer")

    # Low-rank component
    p.add_argument("--lora-rank", type=int, default=None,
                    help="LoRA rank")
    p.add_argument("--lora-targets", type=str, default="qkvo",
                    choices=["qkvo", "qv", "qkv"],
                    help="Which attention projections to adapt")
    p.add_argument("--lora-ffn", action="store_true",
                    help="Also apply LoRA to FFN projections")

    # Sparse component
    p.add_argument("--density", type=float, default=None,
                    help="Sparse mask density")
    p.add_argument("--sparse-targets", type=str, default="qkvo",
                    choices=["qkvo", "qv", "qkv"],
                    help="Which attention projections for sparse masks")
    p.add_argument("--sparse-ffn", action="store_true",
                    help="Also apply sparse to FFN projections")

    # Mask generation (rosa only)
    p.add_argument("--rosa-mode", type=str, default="rosa",
                    choices=["rosa", "retro-sparse", "retro-bottleneck"],
                    help="RoSA training mode")
    p.add_argument("--rosa-warmup-steps", type=int, default=128,
                    help="LoRA warm-up steps before mask generation")
    p.add_argument("--mask-samples", type=int, default=32,
                    help="Batches for gradient accumulation during mask generation")
    p.add_argument("--grad-alpha", type=int, default=2, choices=[1, 2],
                    help="Gradient exponent: 1=mean, 2=Fisher")

    # FiLM component
    p.add_argument("--use-output-film", action="store_true",
                    help="Apply FiLM to output logits")

    # From-scratch (tiny only)
    p.add_argument("--d-model", type=int, default=None,
                    help="Model dimension for from-scratch training")
    p.add_argument("--n-layers", type=int, default=None,
                    help="Number of layers for from-scratch training")
    p.add_argument("--n-heads", type=int, default=None,
                    help="Number of attention heads for from-scratch training")

    # Unfreeze
    p.add_argument("--unfreeze-layers", type=str, default=None,
                    help="Comma-separated layer indices to unfreeze (e.g. '6,7')")

    # Data
    p.add_argument("--max-games", type=int, default=1_000_000)
    p.add_argument("--val-games", type=int, default=50_000)
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--elo-min", type=int, default=None)
    p.add_argument("--elo-max", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default=None,
                    help="Prefetch filtered shards to this dir (e.g. /dev/shm)")

    # Training
    p.add_argument("--total-steps", type=int, default=None,
                    help="Stop after this many steps (overrides epoch count)")
    p.add_argument("--eval-interval", type=int, default=None,
                    help="Evaluate every N steps (default: every epoch)")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--amp-dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "none"],
                    help="AMP dtype (float16, bfloat16, or none to disable)")

    # Device / precision
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--sdpa-math", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None)
    ckpt_group.add_argument("--local-checkpoints", action="store_true")

    return p.parse_args()


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


def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sparse_forward(model, ids, msk, legal_mask, amp_dtype, device):
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
        hidden = model.forward_hidden(ids)
        valid_hidden = hidden[msk]
        valid_logits = model.project_head(valid_hidden)
    valid_legal = legal_mask[msk]
    valid_logits = valid_logits.float()
    valid_logits.masked_fill_(~valid_legal, float("-inf"))
    return valid_logits


@torch.no_grad()
def evaluate(model, dataloader, mask_builder, device, amp_dtype=None,
             precomputed_indices=None):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_positions = 0

    for i, batch in enumerate(dataloader):
        ids = batch["input_ids"].to(device, non_blocking=True)
        tgt = batch["targets"].to(device, non_blocking=True)
        msk = batch["loss_mask"].to(device, non_blocking=True)
        if precomputed_indices is not None:
            legal_mask = mask_builder.scatter(precomputed_indices[i], ids.shape[0])
        elif "legal_indices" in batch:
            legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])
        else:
            legal_mask = mask_builder(batch)

        valid_logits = sparse_forward(model, ids, msk, legal_mask, amp_dtype, device)
        valid_targets = tgt[msk]
        n_pos = valid_targets.shape[0]
        if n_pos == 0:
            continue

        loss = F.cross_entropy(valid_logits, valid_targets)
        preds = valid_logits.argmax(dim=-1)
        top1 = (preds == valid_targets).float().mean().item()
        top5 = valid_logits.topk(5, dim=-1).indices
        top5_acc = (top5 == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

        total_loss += loss.item() * n_pos
        total_top1 += top1 * n_pos
        total_top5 += top5_acc * n_pos
        total_positions += n_pos

    if total_positions == 0:
        return {"loss": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0}
    return {
        "loss": total_loss / total_positions,
        "top1_accuracy": total_top1 / total_positions,
        "top5_accuracy": total_top5 / total_positions,
    }


def parse_layers(s: str | None) -> tuple[int, ...] | None:
    if s is None:
        return None
    return tuple(int(x) for x in s.split(","))


def precompute_val_masks(val_loader, mask_builder, vocab_size):
    indices = []
    for batch in val_loader:
        move_ids = batch["move_ids"]
        if isinstance(move_ids, torch.Tensor):
            move_ids = move_ids.numpy()
        game_lengths = np.asarray(batch["game_length"], dtype=np.int16)
        idx = compute_legal_indices(
            move_ids, game_lengths, mask_builder.T, vocab_size,
        )
        indices.append(torch.from_numpy(idx).pin_memory())
    return indices


# ---------------------------------------------------------------------------
# Model construction per strategy
# ---------------------------------------------------------------------------

def build_model(args, device):
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
        # RoSA is special — returns the warmup model for Phase 1.
        # Phase 3 model is built after mask generation in main().
        return _build_rosa_warmup(backbone, args, layers, device)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _build_bottleneck(backbone, args, layers, device):
    from pawn.adapters.bottleneck import BottleneckCLM
    dim = args.bottleneck_dim or 8
    model = BottleneckCLM(
        backbone, bottleneck_dim=dim,
        adapt_attn=not args.no_adapt_attn,
        adapt_ffn=not args.no_adapt_ffn,
        layers=layers,
    ).to(device)
    params = model.adapter_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.adapter_state_dict, model.adapter_weight_report


def _build_lora(backbone, args, layers, device):
    from pawn.adapters.lora import LoRACLM
    rank = args.lora_rank or 4
    model = LoRACLM(
        backbone, rank=rank,
        attn_targets=args.lora_targets,
        adapt_ffn=args.lora_ffn,
        layers=layers,
    ).to(device)
    params = model.lora_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.lora_state_dict, model.lora_weight_report


def _build_film(backbone, args, device):
    from pawn.adapters.film import FiLMCLM
    model = FiLMCLM(
        backbone, use_output_film=args.use_output_film,
    ).to(device)
    params = model.film_parameters()
    n = sum(p.numel() for p in params)
    return model, params, n, model.film_state_dict, model.film_weight_report


def _build_sparse(backbone, args, layers, device):
    from pawn.adapters.sparse import SparseCLM
    from pawn.adapters.lora import ATTN_PRESETS
    density = args.density or 0.01
    attn_targets = ATTN_PRESETS[args.sparse_targets]
    model = SparseCLM(
        backbone, density=density,
        attn_targets=attn_targets,
        adapt_ffn=args.sparse_ffn,
        layers=layers,
    ).to(device)
    params = model.sparse_parameters()
    n = model.n_active_params()
    return model, params, n, model.sparse_state_dict, model.sparse_weight_report


def _build_hybrid(backbone, args, layers, device):
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


def _build_unfreeze(backbone, args, device):
    unfreeze_layers = parse_layers(args.unfreeze_layers)
    if unfreeze_layers is None:
        raise ValueError("--unfreeze-layers is required for unfreeze strategy")

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

    def state_dict_fn():
        return {k: v for k, v in backbone.state_dict().items()
                if any(f"layers.{i}." in k for i in unfreeze_layers)}

    def weight_report_fn():
        return {"unfrozen_layers": list(unfreeze_layers), "n_trainable": n}

    # Wrap backbone so it has forward_hidden and project_head
    class UnfreezeWrapper(nn.Module):
        def __init__(self, bb: nn.Module):
            super().__init__()
            self.bb = bb
            self.cfg = bb.cfg  # type: ignore[attr-defined]

        def forward_hidden(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            return self.bb.forward_hidden(input_ids, attention_mask)  # type: ignore[attr-defined]

        def project_head(self, x: torch.Tensor) -> torch.Tensor:
            return self.bb.lm_head(x)  # type: ignore[attr-defined]

        def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            self.bb.load_state_dict(state, strict=False)

    model = UnfreezeWrapper(backbone).to(device)
    return model, params, n, state_dict_fn, weight_report_fn


def _build_specialized_clm(args, device):
    from pawn.specialized_clm import SpecializedCLM

    d_model = args.d_model or 84
    n_layers = args.n_layers or 2
    n_heads = args.n_heads or 4
    d_ff = d_model * 4
    vocab_size = CLMConfig().vocab_size

    model = SpecializedCLM(
        vocab_size=vocab_size, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
    ).to(device)

    params = list(model.parameters())
    n = sum(p.numel() for p in params)

    def state_dict_fn():
        return model.state_dict()

    def weight_report_fn():
        return {}

    return model, params, n, state_dict_fn, weight_report_fn


def _build_rosa_warmup(backbone, args, layers, device):
    """Build the Phase 1 (LoRA warmup) model for RoSA."""
    from pawn.adapters.rosa import RoSACLM
    rank = args.lora_rank or 4
    model = RoSACLM(
        backbone, rank=rank,
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

def rosa_warmup(model, train_loader, mask_builder, args, device, amp_dtype, logger):
    """Phase 1: Train LoRA-only for warmup_steps."""
    lr = args.lr
    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler() if amp_dtype is not None else None

    model.train()
    step = 0
    total_loss = 0.0
    n_positions = 0

    print(f"\n=== RoSA Phase 1: LoRA warm-up ({args.rosa_warmup_steps} steps) ===",
          flush=True)

    while step < args.rosa_warmup_steps:
        for batch in train_loader:
            if step >= args.rosa_warmup_steps:
                break
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            if "legal_indices" in batch:
                legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])
            else:
                legal_mask = mask_builder(batch)

            valid_logits = sparse_forward(model, ids, msk, legal_mask, amp_dtype, device)
            valid_targets = tgt[msk]
            if valid_targets.shape[0] == 0:
                continue

            loss = F.cross_entropy(valid_logits, valid_targets)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()

            n_pos = valid_targets.shape[0]
            total_loss += loss.item() * n_pos
            n_positions += n_pos
            step += 1

            if step % 32 == 0 or step == args.rosa_warmup_steps:
                avg = total_loss / max(n_positions, 1)
                print(f"  Warmup step {step}/{args.rosa_warmup_steps} | loss={avg:.4f}",
                      flush=True)

    print(f"  Warm-up complete (avg loss={total_loss / max(n_positions, 1):.4f})")
    return step


def rosa_mask_generation(model, train_loader, mask_builder, args, device, amp_dtype, logger):
    """Phase 2: Generate gradient-informed sparse masks."""
    from pawn.adapters.rosa import generate_gradient_masks

    print(f"\n=== RoSA Phase 2: Mask generation (density={args.density}, "
          f"alpha={args.grad_alpha}) ===", flush=True)

    use_amp = amp_dtype is not None
    masks = generate_gradient_masks(
        model, train_loader, mask_builder,
        density=args.density or 0.01,
        alpha=args.grad_alpha,
        device=device, use_amp=use_amp,
        max_batches=args.mask_samples,
    )

    total_active = sum(m.sum().item() for m in masks.values())
    total_elements = sum(m.numel() for m in masks.values())
    print(f"  Total: {total_active:,.0f} / {total_elements:,} "
          f"({100 * total_active / total_elements:.2f}%)")
    return masks


def rosa_build_phase3(warmup_model, masks, args, device):
    """Build the Phase 3 model based on rosa_mode."""
    from pawn.adapters.rosa import RetroBottleneckCLM
    from pawn.adapters.sparse import SparseCLM, SparseLinear
    from pawn.adapters.lora import ATTN_PRESETS, _FFN_TARGETS

    mode = args.rosa_mode

    if mode == "rosa":
        warmup_model.set_masks(masks)
        warmup_model.reinit_lora()
        params = warmup_model.adapter_parameters()
        n = sum(p.numel() for p in params)
        return warmup_model, params, n, warmup_model.adapter_state_dict, warmup_model.adapter_weight_report

    # Retrospective modes: reload backbone
    del warmup_model
    gc.collect()
    torch.cuda.empty_cache()

    backbone = load_backbone(args.checkpoint, device)
    attn_targets = ATTN_PRESETS[args.lora_targets]

    sparse_model = SparseCLM(
        backbone, density=args.density or 0.01,
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
        return sparse_model, params, n, sparse_model.sparse_state_dict, sparse_model.sparse_weight_report

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

def build_config_json(args, param_count: int) -> dict:
    """Build the normalized config dict with nulls for irrelevant params."""
    layers = parse_layers(args.adapter_layers)

    cfg = {
        "strategy": args.strategy,
        "param_count": param_count,

        # Backbone
        "checkpoint": args.checkpoint if args.strategy != "specialized_clm" else None,
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
        "bottleneck_dim": args.bottleneck_dim if args.strategy in ("bottleneck", "rosa") else None,
        "adapt_attn": (not args.no_adapt_attn) if args.strategy == "bottleneck" else None,
        "adapt_ffn": (not args.no_adapt_ffn) if args.strategy == "bottleneck" else None,

        # Low-rank
        "lora_rank": args.lora_rank if args.strategy in ("lora", "rosa", "hybrid") else None,
        "lora_targets": args.lora_targets if args.strategy in ("lora", "rosa", "hybrid") else None,
        "lora_ffn": args.lora_ffn if args.strategy in ("lora", "rosa", "hybrid") else None,

        # Sparse
        "density": args.density if args.strategy in ("sparse", "rosa") else None,
        "sparse_targets": args.sparse_targets if args.strategy in ("sparse", "rosa") else None,
        "sparse_ffn": args.sparse_ffn if args.strategy in ("sparse", "rosa") else None,

        # Mask generation
        "rosa_mode": args.rosa_mode if args.strategy == "rosa" else None,
        "rosa_warmup_steps": args.rosa_warmup_steps if args.strategy == "rosa" else None,
        "mask_samples": args.mask_samples if args.strategy == "rosa" else None,
        "grad_alpha": args.grad_alpha if args.strategy == "rosa" else None,

        # FiLM
        "use_output_film": args.use_output_film if args.strategy in ("film", "hybrid") else None,

        # From-scratch
        "d_model": args.d_model if args.strategy == "specialized_clm" else None,
        "n_layers": args.n_layers if args.strategy == "specialized_clm" else None,
        "n_heads": args.n_heads if args.strategy == "specialized_clm" else None,

        # Unfreeze
        "unfreeze_layers": list(parse_layers(args.unfreeze_layers) or ()) if args.strategy == "unfreeze" else None,
    }
    return cfg


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model, trainable_params, train_loader, val_loader, mask_builder,
          val_legal_indices, logger, args, device, amp_dtype, gpu_cfg,
          state_dict_fn, weight_report_fn, param_count):
    """Unified training loop for all strategies."""
    from pawn import model as model_module
    from pawn.checkpoint import save_adapter_checkpoint

    # Compile
    model.forward_hidden = apply_gpu_config(gpu_cfg, model_module, model.forward_hidden)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    # Compute total steps
    streaming = hasattr(train_loader.dataset, 'shard_files')
    if args.total_steps:
        total_steps = args.total_steps
    elif streaming:
        est_games = min(args.max_games or 1_000_000,
                        len(train_loader.dataset.shard_files) * 60_000)
        total_steps = args.epochs * (est_games // args.batch_size)
    else:
        total_steps = args.epochs * len(train_loader)

    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler() if amp_dtype is not None else None

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
            if hasattr(model, 'load_adapter_state_dict'):
                model.load_adapter_state_dict(state)
            elif hasattr(model, 'load_lora_state_dict'):
                model.load_lora_state_dict(state)
            elif hasattr(model, 'load_film_state_dict'):
                model.load_film_state_dict(state)
            elif hasattr(model, 'load_sparse_state_dict'):
                model.load_sparse_state_dict(state)
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
        baseline = evaluate(model, val_loader, mask_builder, device,
                            amp_dtype=amp_dtype, precomputed_indices=val_legal_indices)
        print(f"  loss={baseline['loss']:.4f}, top1={baseline['top1_accuracy']:.4%}, "
              f"top5={baseline['top5_accuracy']:.4%}")
        logger.log_train(step=0, epoch=-1,
            train_loss=baseline["loss"], train_top1=baseline["top1_accuracy"],
            val_loss=baseline["loss"], val_top1=baseline["top1_accuracy"],
            val_top5=baseline["top5_accuracy"],
        )
        val_metrics = baseline
    else:
        val_metrics = evaluate(model, val_loader, mask_builder, device,
                               amp_dtype=amp_dtype, precomputed_indices=val_legal_indices)

    # Graceful shutdown
    _shutdown = False
    def _handle_signal(signum, frame):
        nonlocal _shutdown
        _shutdown = True
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ckpt_dir = logger.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    hf_branch = f"run/{logger.run_dir.name}" if args.hf_repo else None

    eval_interval = args.eval_interval
    step_limit = args.total_steps

    print(f"\nTraining for up to {args.epochs} epochs ({total_steps} steps)")
    print(f"  Warmup: {warmup_steps} steps, LR: {args.lr}, AMP: {args.amp_dtype}")

    def _do_eval():
        return evaluate(model, val_loader, mask_builder, device,
                        amp_dtype=amp_dtype, precomputed_indices=val_legal_indices)

    def _save_best(vm, ep):
        save_adapter_checkpoint(
            ckpt_dir / "best", state_dict_fn(),
            config=build_config_json(args, param_count),
            epoch=ep, step=global_step, val_metrics=vm,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            extra={"best_val_loss": best_val_loss, "patience_counter": patience_counter},
        )
        if args.hf_repo and hf_branch:
            from pawn.checkpoint import push_checkpoint_to_hf
            try:
                push_checkpoint_to_hf(ckpt_dir / "best", args.hf_repo, hf_branch,
                                      step=global_step)
            except Exception as e:
                print(f"WARNING: HF push failed: {e}")

    epoch = max(start_epoch, 0)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_top1 = 0.0
        epoch_positions = 0
        log_loss = 0.0
        log_top1 = 0.0
        log_positions = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device, non_blocking=True)
            msk = batch["loss_mask"].to(device, non_blocking=True)
            legal_mask = mask_builder.scatter(batch["legal_indices"], ids.shape[0])

            valid_logits = sparse_forward(model, ids, msk, legal_mask, amp_dtype, device)
            valid_targets = tgt[msk]
            loss = F.cross_entropy(valid_logits, valid_targets)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = valid_logits.argmax(dim=-1)
                top1 = (preds == valid_targets).float().mean().item()

            n_pos = valid_targets.shape[0]
            epoch_loss += loss.item() * n_pos
            epoch_top1 += top1 * n_pos
            epoch_positions += n_pos
            log_loss += loss.item() * n_pos
            log_top1 += top1 * n_pos
            log_positions += n_pos
            global_step += 1

            if global_step % args.log_interval == 0 and log_positions > 0:
                avg_loss = log_loss / log_positions
                avg_top1 = log_top1 / log_positions
                lr = optimizer.param_groups[0]["lr"]
                print(f"  step {global_step:6d} | loss={avg_loss:.4f} "
                      f"top1={avg_top1:.4%} lr={lr:.2e}")
                log_loss = log_top1 = log_positions = 0

            if eval_interval and global_step % eval_interval == 0:
                val_metrics = _do_eval()
                print(f"  [eval @ step {global_step}] "
                      f"val_loss={val_metrics['loss']:.4f} "
                      f"val_top1={val_metrics['top1_accuracy']:.4%} "
                      f"val_top5={val_metrics['top5_accuracy']:.4%}")
                logger.log_train(step=global_step, epoch=epoch,
                    lr=optimizer.param_groups[0]["lr"],
                    val_loss=val_metrics["loss"],
                    val_top1=val_metrics["top1_accuracy"],
                    val_top5=val_metrics["top5_accuracy"],
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    _save_best(val_metrics, epoch)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"\n  Early stopping at step {global_step} (patience={args.patience})")
                        break
                model.train()

            if step_limit and global_step >= step_limit:
                break
            if _shutdown:
                break

        dt = time.time() - t0
        train_loss = epoch_loss / max(epoch_positions, 1)
        train_top1 = epoch_top1 / max(epoch_positions, 1)

        do_val = not eval_interval and (
            (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        )
        if do_val:
            val_metrics = _do_eval()

        try:
            report = weight_report_fn()
        except Exception:
            report = {}
        logger.log_train(step=global_step, epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss=train_loss, train_top1=train_top1,
            val_loss=val_metrics["loss"],
            val_top1=val_metrics["top1_accuracy"],
            val_top5=val_metrics["top5_accuracy"],
            epoch_time_s=dt, **report,
        )

        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_top1={train_top1:.4%} | "
              f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_accuracy']:.4%} "
              f"val_top5={val_metrics['top5_accuracy']:.4%} | {dt:.1f}s")

        if do_val:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                _save_best(val_metrics, epoch)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                    break

        if eval_interval and patience_counter >= args.patience:
            break  # step-based early stopping triggered inside batch loop
        if step_limit and global_step >= step_limit:
            print(f"\n  Reached step limit ({step_limit})")
            break
        if _shutdown:
            print("Shutdown requested, saving checkpoint...")
            break

    # Final checkpoint
    save_adapter_checkpoint(
        ckpt_dir / "final", state_dict_fn(),
        config=build_config_json(args, param_count),
        epoch=epoch, step=global_step, val_metrics=val_metrics,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        extra={"best_val_loss": best_val_loss, "patience_counter": patience_counter},
    )

    return best_val_loss, val_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device
    amp_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "none": None}
    amp_dtype = amp_dtype_map[args.amp_dtype]

    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parent.parent / "logs"
    logger = MetricsLogger(str(log_dir), run_prefix=args.strategy, device=device)
    out_dir = logger.run_dir

    print(f"Strategy: {args.strategy}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # GPU config
    from pawn import model as model_module
    gpu_cfg = configure_gpu(
        device, no_compile=args.no_compile, no_amp=(amp_dtype is None),
        sdpa_math=args.sdpa_math,
    )
    gpu_cfg["use_amp"] = amp_dtype is not None
    gpu_cfg["amp_dtype"] = amp_dtype

    # Build model
    vocab_size: int = CLMConfig().vocab_size
    if args.strategy == "rosa":
        # RoSA needs special 3-phase handling
        model, _, _, _, _ = build_model(args, device)
        vocab_size = getattr(getattr(model, 'cfg', None), 'vocab_size', vocab_size)
    else:
        model, trainable_params, param_count, state_dict_fn, weight_report_fn = build_model(args, device)
        cfg_obj = getattr(model, 'cfg', None) or getattr(getattr(model, 'bb', None), 'cfg', None)
        if cfg_obj is not None:
            vocab_size = cfg_obj.vocab_size
        print(f"Trainable params: {param_count:,}")

    # Prepare data
    max_ply = 255
    streaming = args.elo_min is not None or args.elo_max is not None

    from pawn.shard_loader import ShardedLichessDataset, load_val_shards
    if streaming:
        print(f"\nShard-parallel loading: {args.pgn} [{args.elo_min}, {args.elo_max})")
        val_data = load_val_shards(
            args.pgn, elo_min=args.elo_min, elo_max=args.elo_max,
            min_ply=args.min_ply, max_games=args.val_games,
            cache_dir=args.cache_dir,
        )
        val_ds = LichessDataset(val_data, start=0, end=val_data["n_games"])
        train_ds = ShardedLichessDataset(
            args.pgn, elo_min=args.elo_min, elo_max=args.elo_max,
            min_ply=args.min_ply, max_games=args.max_games,
            cache_dir=args.cache_dir,
        )
        print(f"  Val: {len(val_ds):,} games, Train: {len(train_ds.shard_files)} shards")
    else:
        print(f"\nPreparing data: {args.pgn}")
        data = prepare_lichess_dataset(
            args.pgn, max_ply=255, max_games=args.max_games, min_ply=args.min_ply,
        )
        n_total = data["n_games"]
        n_val = min(args.val_games, n_total // 5)
        n_train = n_total - n_val
        print(f"  Train: {n_train} games, Val: {n_val} games")
        train_ds = LichessDataset(data, start=0, end=n_train).share_memory()
        val_ds = LichessDataset(data, start=n_train, end=n_total)

    collate = LegalMaskCollate(seq_len=max_ply + 1, vocab_size=vocab_size)
    n_workers = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=not streaming,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=n_workers > 0, collate_fn=collate,
        multiprocessing_context='spawn' if n_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    mask_builder = LegalMaskBuilder(args.batch_size, max_ply=255,
                                    vocab_size=vocab_size, device=device)

    # Precompute val legal indices
    val_legal_indices = precompute_val_masks(val_loader, mask_builder, vocab_size)
    print(f"  Precomputed legal masks for {len(val_legal_indices)} val batches")

    # Log config
    if args.strategy == "rosa":
        # For RoSA, we don't know param_count until Phase 3
        logger.log_config(run_type=args.strategy, **{
            k: v for k, v in build_config_json(args, 0).items()
            if k not in ("strategy", "param_count")
        })

        # Phase 1: LoRA warmup (no compile for short warmup)
        rosa_warmup(model, train_loader, mask_builder, args, device, amp_dtype, logger)

        # Phase 2: Mask generation
        masks = rosa_mask_generation(model, train_loader, mask_builder, args, device,
                                     amp_dtype, logger)

        # Phase 3: Build final model and train
        print(f"\n=== RoSA Phase 3: {args.rosa_mode} training ===")
        model, trainable_params, param_count, state_dict_fn, weight_report_fn = \
            rosa_build_phase3(model, masks, args, device)
        print(f"Trainable params: {param_count:,}")
    else:
        logger.log_config(run_type=args.strategy, **{
            k: v for k, v in build_config_json(args, param_count).items()
            if k not in ("strategy",)
        })

    # Write config JSON
    config = build_config_json(args, param_count)
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")

    # Train
    best_val_loss, final_metrics = train(
        model, trainable_params, train_loader, val_loader, mask_builder,
        val_legal_indices, logger, args, device, amp_dtype, gpu_cfg,
        state_dict_fn, weight_report_fn, param_count,
    )

    logger.close()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
