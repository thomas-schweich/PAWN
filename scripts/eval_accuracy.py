#!/usr/bin/env python3
"""Evaluate adapter move-prediction accuracy (MAIA-compatible).

Supports LoRA, FiLM, and hybrid adapters. By default skips the first
10 ply to match the evaluation methodology from `MAIA
<https://arxiv.org/abs/2006.01855>`_ (McIlroy-Young et al., "Aligning
Superhuman AI with Human Behavior: Chess as a Model System", KDD 2020).
Opening moves are too book-ish to be informative.

Reports overall accuracy, per-phase accuracy (opening / middle / late),
and optionally per-ply accuracy.

Usage:
    uv run python scripts/eval_accuracy.py \
        --checkpoint /path/to/checkpoint.pt \
        --adapter-checkpoint lora_runs/best.pt \
        --pgn /path/to/lichess_1200_1400.pgn \
        --min-eval-ply 10
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.lichess_data import (
    prepare_lichess_dataset,
    LegalMaskBuilder,
    LichessDataset,
)


def parse_args():
    p = argparse.ArgumentParser(description="MAIA-compatible accuracy evaluation")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to PAWN backbone checkpoint")
    p.add_argument("--adapter-checkpoint", type=str, required=True,
                    help="Path to trained adapter checkpoint (LoRA/FiLM/hybrid)")
    p.add_argument("--pgn", type=str, required=True,
                    help="Path to Lichess PGN file")

    # Eval settings
    p.add_argument("--min-eval-ply", type=int, default=10,
                    help="Skip first N ply (default: 10, matching MAIA)")
    p.add_argument("--max-games", type=int, default=50_000)
    p.add_argument("--val-start", type=int, default=10_000,
                    help="Game index to start evaluation from")
    p.add_argument("--val-games", type=int, default=2_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--per-ply", action="store_true",
                    help="Report per-ply accuracy")

    # Device
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-amp", action="store_true")

    return p.parse_args()


def _detect_adapter_type(config: dict) -> str:
    """Auto-detect adapter type from config dict.

    The config dict comes from load_adapter_checkpoint()["config"], which
    contains training args for both legacy .pt files and new-format directories.
    """
    if "checkpoint_type" in config:
        return config["checkpoint_type"]
    if "bottleneck_dim" in config:
        return "bottleneck"
    if "lora_rank" in config and config.get("use_film") is not None:
        return "hybrid"
    if "lora_rank" in config:
        return "lora"
    if "density" in config:
        return "sparse"
    if "no_output_film" in config:
        return "film"

    raise ValueError("Cannot detect adapter type from config keys: "
                     + ", ".join(config.keys()))


def load_model(checkpoint_path: str, adapter_path: str, device: str):
    """Load backbone + adapter, auto-detecting adapter type."""
    from pawn.checkpoint import load_backbone_weights, load_adapter_checkpoint

    # Backbone
    state_dict, model_config = load_backbone_weights(checkpoint_path, device)
    cfg = CLMConfig(**model_config) if model_config else CLMConfig()
    backbone = PAWNCLM(cfg).to(device)
    backbone.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    backbone.eval()

    # Adapter
    adapter_data = load_adapter_checkpoint(adapter_path, device)
    adapter_weights = adapter_data["adapter_state_dict"]
    adapter_config = adapter_data.get("config", {})
    adapter_type = _detect_adapter_type(adapter_config)

    if adapter_type == "lora":
        from pawn.adapters.lora import LoRACLM
        model = LoRACLM(
            backbone,
            rank=adapter_config.get("lora_rank", 4),
            alpha=adapter_config.get("lora_alpha", None),
            attn_targets=adapter_config.get("lora_targets", "qkvo"),
            adapt_ffn=adapter_config.get("lora_ffn", False),
            layers=tuple(int(x) for x in adapter_config["lora_layers"].split(","))
                   if adapter_config.get("lora_layers") else None,
        ).to(device)
        model.load_lora_state_dict(adapter_weights)

    elif adapter_type == "film":
        from pawn.adapters.film import FiLMCLM
        has_output = not adapter_config.get("no_output_film", False)
        if any(k.startswith("output_film.") for k in adapter_weights):
            has_output = True
        model = FiLMCLM(backbone, use_output_film=has_output).to(device)
        model.load_film_state_dict(adapter_weights)

    elif adapter_type == "hybrid":
        from pawn.adapters.hybrid import HybridCLM
        lora_layers = adapter_config.get("lora_layers")
        film_layers = adapter_config.get("film_layers")
        model = HybridCLM(
            backbone,
            lora_rank=adapter_config.get("lora_rank", 4),
            lora_alpha=adapter_config.get("lora_alpha", None),
            attn_targets=adapter_config.get("lora_targets", "qkvo"),
            adapt_ffn=adapter_config.get("lora_ffn", False),
            lora_layers=tuple(int(x) for x in lora_layers.split(",")) if lora_layers else None,
            use_film=adapter_config.get("use_film", True),
            use_output_film=adapter_config.get("output_film", False),
            film_layers=tuple(int(x) for x in film_layers.split(",")) if film_layers else None,
        ).to(device)
        model.load_adapter_state_dict(adapter_weights)

    elif adapter_type == "sparse":
        from pawn.adapters.sparse import SparseCLM
        _attn_presets = {"qkvo": ("wq", "wk", "wv", "wo"), "qv": ("wq", "wv"), "qkv": ("wq", "wk", "wv")}
        sparse_layers = adapter_config.get("sparse_layers")
        model = SparseCLM(
            backbone,
            density=adapter_config.get("density", 0.01),
            attn_targets=_attn_presets.get(adapter_config.get("sparse_targets", "qkvo"),
                                           ("wq", "wk", "wv", "wo")),
            adapt_ffn=adapter_config.get("sparse_ffn", False),
            layers=tuple(int(x) for x in sparse_layers.split(",")) if sparse_layers else None,
            seed=adapter_config.get("sparse_seed", 42),
        ).to(device)
        model.load_sparse_state_dict(adapter_weights)

    elif adapter_type == "bottleneck":
        from pawn.adapters.bottleneck import BottleneckCLM
        adapter_layers_str = adapter_config.get("adapter_layers")
        model = BottleneckCLM(
            backbone,
            bottleneck_dim=adapter_config.get("bottleneck_dim", 8),
            adapt_attn=adapter_config.get("adapt_attn", True),
            adapt_ffn=adapter_config.get("adapt_ffn", True),
            layers=tuple(int(x) for x in adapter_layers_str.split(",")) if adapter_layers_str else None,
        ).to(device)
        model.load_adapter_state_dict(adapter_weights)

    model.eval()
    return model, adapter_type


@torch.no_grad()
def evaluate_maia(
    model,
    dataloader,
    mask_builder,
    device: str,
    min_eval_ply: int = 10,
    use_amp: bool = False,
    per_ply: bool = False,
):
    """Evaluate accuracy, skipping the first min_eval_ply positions.

    In our sequence format, position k predicts move k (0-indexed).
    Skipping the first 10 ply means only evaluating at positions >= 10.
    """
    model.eval()

    # Phase buckets: opening (ply 0-19), middle (20-59), late (60+)
    phase_bins = {"opening": (0, 20), "middle": (20, 60), "late": (60, 999)}
    phase_stats = {name: {"loss": 0.0, "top1": 0.0, "top5": 0.0, "n": 0}
                   for name in phase_bins}

    # Overall stats (respecting min_eval_ply)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_positions = 0

    # Per-ply stats (optional)
    ply_top1 = {} if per_ply else None
    ply_count = {} if per_ply else None

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        tgt = batch["targets"].to(device)
        msk = batch["loss_mask"].to(device)
        legal_mask = mask_builder(batch)

        B, T = ids.shape

        # Full forward to get all logits
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(ids)  # (B, T, V)

        logits = logits.float()
        legal = legal_mask[:B]
        logits.masked_fill_(~legal, float("-inf"))

        # Iterate positions to bucket by ply
        for pos in range(T):
            # pos in sequence = ply index (position 0 predicts move 0)
            pos_mask = msk[:, pos]  # (B,) — which games are still active
            if not pos_mask.any():
                continue

            pos_logits = logits[pos_mask, pos, :]  # (N, V)
            pos_targets = tgt[pos_mask, pos]        # (N,)
            n = pos_targets.shape[0]

            pos_loss = F.cross_entropy(pos_logits, pos_targets).item()
            pos_preds = pos_logits.argmax(dim=-1)
            pos_top1 = (pos_preds == pos_targets).float().mean().item()
            pos_top5_idx = pos_logits.topk(5, dim=-1).indices
            pos_top5 = (pos_top5_idx == pos_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

            # Overall (respecting cutoff)
            if pos >= min_eval_ply:
                total_loss += pos_loss * n
                total_top1 += pos_top1 * n
                total_top5 += pos_top5 * n
                total_positions += n

            # Phase buckets (always from ply 0 for full picture)
            for name, (lo, hi) in phase_bins.items():
                if lo <= pos < hi:
                    s = phase_stats[name]
                    s["loss"] += pos_loss * n
                    s["top1"] += pos_top1 * n
                    s["top5"] += pos_top5 * n
                    s["n"] += n

            # Per-ply
            if per_ply:
                ply_top1[pos] = ply_top1.get(pos, 0.0) + pos_top1 * n
                ply_count[pos] = ply_count.get(pos, 0) + n

    # Aggregate
    results = {}

    if total_positions > 0:
        results["overall"] = {
            "min_eval_ply": min_eval_ply,
            "loss": total_loss / total_positions,
            "top1_accuracy": total_top1 / total_positions,
            "top5_accuracy": total_top5 / total_positions,
            "n_positions": total_positions,
        }

    results["phases"] = {}
    for name, s in phase_stats.items():
        if s["n"] > 0:
            results["phases"][name] = {
                "loss": s["loss"] / s["n"],
                "top1_accuracy": s["top1"] / s["n"],
                "top5_accuracy": s["top5"] / s["n"],
                "n_positions": s["n"],
            }

    if per_ply:
        results["per_ply"] = {
            ply: {"top1_accuracy": ply_top1[ply] / ply_count[ply], "n": ply_count[ply]}
            for ply in sorted(ply_top1.keys())
        }

    return results


def main():
    args = parse_args()
    device = args.device
    use_amp = not args.no_amp and device.startswith("cuda")

    print(f"Loading model + adapter...")
    model, adapter_type = load_model(args.checkpoint, args.adapter_checkpoint, device)
    print(f"  Adapter type: {adapter_type}")

    # Prepare data
    print(f"Preparing evaluation data: {args.pgn}")
    data = prepare_lichess_dataset(
        args.pgn, max_ply=255, max_games=args.max_games, min_ply=10,
    )
    n_total = data["n_games"]
    val_start = min(args.val_start, n_total)
    val_end = min(val_start + args.val_games, n_total)
    print(f"  Using games [{val_start}:{val_end}] ({val_end - val_start} games)")

    val_ds = LichessDataset(data, start=val_start, end=val_end)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    vocab_size = model.cfg.vocab_size
    mask_builder = LegalMaskBuilder(args.batch_size, max_ply=255,
                                    vocab_size=vocab_size, device=device)

    # Evaluate
    print(f"\nEvaluating (min_eval_ply={args.min_eval_ply})...")
    results = evaluate_maia(
        model, val_loader, mask_builder, device,
        min_eval_ply=args.min_eval_ply,
        use_amp=use_amp,
        per_ply=args.per_ply,
    )

    # Report
    if "overall" in results:
        o = results["overall"]
        print(f"\n=== Overall (ply >= {args.min_eval_ply}) ===")
        print(f"  Loss:      {o['loss']:.4f}")
        print(f"  Top-1:     {o['top1_accuracy']:.4%}")
        print(f"  Top-5:     {o['top5_accuracy']:.4%}")
        print(f"  Positions: {o['n_positions']:,}")

    if results.get("phases"):
        print(f"\n=== By Phase ===")
        print(f"  {'Phase':<10} {'Top-1':>10} {'Top-5':>10} {'Loss':>10} {'N':>10}")
        print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for name in ("opening", "middle", "late"):
            if name in results["phases"]:
                p = results["phases"][name]
                print(f"  {name:<10} {p['top1_accuracy']:10.4%} {p['top5_accuracy']:10.4%} "
                      f"{p['loss']:10.4f} {p['n_positions']:10,}")

    if args.per_ply and "per_ply" in results:
        print(f"\n=== Per-Ply Top-1 ===")
        print(f"  {'Ply':>5} {'Top-1':>10} {'N':>10}")
        print(f"  {'─'*5} {'─'*10} {'─'*10}")
        for ply, stats in sorted(results["per_ply"].items()):
            marker = " " if ply >= args.min_eval_ply else "*"
            print(f"  {ply:5d} {stats['top1_accuracy']:10.4%} {stats['n']:10,}{marker}")
        print(f"  (* = excluded from overall by --min-eval-ply)")

    # Save
    out_dir = Path(args.adapter_checkpoint).parent
    out_path = out_dir / "eval_maia.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
