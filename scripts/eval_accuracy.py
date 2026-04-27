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
        --checkpoint thomas-schweich/pawn-base \
        --adapter-checkpoint logs/run_*/checkpoints/step_00104000 \
        --pgn thomas-schweich/pawn-lichess-full \
        --min-eval-ply 10

Adapter checkpoints are saved at ``step_{global_step:08d}/`` — matching
the pretraining layout. Pick the best step by scanning ``metrics.jsonl``
for the lowest ``val_loss`` record, or simply use the final step.
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
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to PAWN backbone checkpoint. Optional for "
                         "specialized_clm adapters (which carry their own "
                         "weights and don't wrap a backbone).")
    p.add_argument("--adapter-checkpoint", type=str, required=True,
                    help="Path to trained adapter checkpoint")
    p.add_argument("--pgn", type=str, required=True,
                    help="Lichess dataset: a .parquet file or a HuggingFace "
                         "dataset repo ID (e.g. thomas-schweich/pawn-lichess-full). "
                         "Raw PGN input is not supported; convert first with "
                         "scripts/extract_lichess_parquet.py.")

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
    # Within-distribution eval: filter the held-out slice to the same
    # Elo band the adapter was trained on. Without this the eval runs
    # on whatever Elo distribution happens to land in the val window,
    # which silently mis-compares against the training slice.
    p.add_argument("--elo-min", type=int, default=None,
                    help="Filter to games with both players' Elo >= this. "
                         "Set to match the adapter's training Elo band.")
    p.add_argument("--elo-max", type=int, default=None,
                    help="Filter to games with both players' Elo < this.")

    # Device / precision. Default is fp32 (``none``) — fp16 AMP overflows
    # on ceiling-scale adapters whose adapter activations exceed fp16's
    # representable range, producing NaN-corrupted accuracy that looks
    # like ~9% (random). bf16 has fp32-range exponents and is the safe
    # speed-up choice on supported GPUs.
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--amp-dtype",
        type=str,
        choices=("none", "bfloat16", "float16"),
        default="none",
        help="AMP dtype for the eval forward pass (default: none = fp32). "
             "Use bfloat16 for speed; fp16 can overflow on large adapters.",
    )

    p.add_argument("--prepend-outcome", action="store_true",
                    help="Match an outcome-prefixed backbone. Default is "
                         "pure-moves, matching the canonical training layout.")

    return p.parse_args()


def _detect_adapter_type(config: dict) -> str:
    """Auto-detect adapter type from config dict.

    The order matters: retro-bottleneck adapter configs carry both
    ``bottleneck_dim`` and ``density``, so the rosa branch has to come
    first or the dispatch silently lands on the bottleneck-only loader
    (which produces ~9% accuracy because it ignores the sparse deltas).
    Same for specialized_clm — it carries arch fields, no backbone.
    """
    # Explicit strategy field is the most reliable signal (present in
    # configs written by the current `build_config_json`).
    strategy = config.get("strategy")
    if strategy == "specialized_clm":
        return "specialized_clm"
    if strategy == "rosa":
        mode = config.get("rosa_mode")
        if mode in ("retro-bottleneck",):
            return "retro_bottleneck"
        if mode in ("retro-sparse",):
            return "retro_sparse"
        return "rosa"  # joint LoRA + sparse
    if strategy in {"bottleneck", "lora", "film", "sparse", "hybrid", "unfreeze"}:
        return strategy

    # Legacy / explicit override for older configs.
    if "checkpoint_type" in config:
        return config["checkpoint_type"]

    # Fallback heuristic for older configs without ``strategy``.
    rosa_mode = config.get("rosa_mode")
    if rosa_mode == "retro-bottleneck":
        return "retro_bottleneck"
    if rosa_mode == "retro-sparse":
        return "retro_sparse"
    if rosa_mode is not None:
        return "rosa"
    if "density" in config and config.get("density") is not None and "bottleneck_dim" in config:
        # Both fields populated and no explicit ``rosa_mode`` — treat as
        # the retro-bottleneck combo since pure bottleneck doesn't carry
        # ``density``.
        return "retro_bottleneck"
    if "bottleneck_dim" in config and config.get("bottleneck_dim") is not None:
        return "bottleneck"
    if "lora_rank" in config and config.get("use_film") is not None:
        return "hybrid"
    if "lora_rank" in config and config.get("lora_rank") is not None:
        return "lora"
    if "density" in config and config.get("density") is not None:
        return "sparse"
    if config.get("use_output_film") is not None:
        return "film"

    raise ValueError(
        "Cannot detect adapter type from config keys: "
        + ", ".join(config.keys())
    )


_ATTN_PRESETS = {
    "qkvo": ("wq", "wk", "wv", "wo"),
    "qv": ("wq", "wv"),
    "qkv": ("wq", "wk", "wv"),
}


def _layer_tuple(value: object) -> tuple[int, ...] | None:
    """Coerce an ``adapter_layers`` value to a tuple, accepting both the
    list form written by the current ``build_config_json`` and the legacy
    comma-separated string form."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    if isinstance(value, str):
        return tuple(int(x) for x in value.split(","))
    raise TypeError(f"unrecognized layer spec: {value!r}")


def load_model(
    checkpoint_path: str | None, adapter_path: str, device: str,
):
    """Load adapter (and backbone, when applicable), auto-detecting type.

    For ``specialized_clm`` adapters there is no backbone: the saved
    weights *are* the entire model. Pass ``checkpoint_path=None`` (or
    omit ``--checkpoint`` on the CLI) for that case.
    """
    from pawn.checkpoint import load_backbone_weights, load_adapter_checkpoint

    # Adapter first — its config decides whether we need a backbone.
    adapter_data = load_adapter_checkpoint(adapter_path, device)
    adapter_weights = adapter_data["adapter_state_dict"]
    adapter_config = adapter_data.get("config", {})
    adapter_type = _detect_adapter_type(adapter_config)

    needs_backbone = adapter_type != "specialized_clm"
    backbone: PAWNCLM | None = None
    if needs_backbone:
        if checkpoint_path is None:
            raise ValueError(
                f"Adapter type {adapter_type!r} requires --checkpoint "
                "(a PAWN backbone path or HF repo id)."
            )
        state_dict, model_config = load_backbone_weights(
            checkpoint_path, device
        )
        cfg = CLMConfig(**model_config) if model_config else CLMConfig()
        backbone = PAWNCLM(cfg).to(device)
        backbone.load_state_dict(state_dict)
        del state_dict
        gc.collect()
        backbone.eval()

    if adapter_type == "specialized_clm":
        from pawn.specialized_clm import SpecializedCLM

        d_model = int(adapter_config["d_model"])
        n_layers = int(adapter_config["n_layers"])
        n_heads = int(adapter_config["n_heads"])
        d_ff = adapter_config.get("d_ff") or d_model * 4
        vocab_size = CLMConfig().vocab_size
        model = SpecializedCLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
        ).to(device)
        # The saved adapter_state_dict for spec_clm is the full model
        # state; load_state_dict accepts it directly.
        model.load_state_dict(adapter_weights)

    elif adapter_type == "lora":
        from pawn.adapters.lora import LoRACLM
        assert backbone is not None
        model = LoRACLM(
            backbone,
            rank=adapter_config.get("lora_rank", 4),
            alpha=adapter_config.get("lora_alpha", None),
            attn_targets=adapter_config.get("lora_targets", "qkvo"),
            adapt_ffn=adapter_config.get("lora_ffn", False),
            layers=_layer_tuple(
                adapter_config.get("lora_layers")
                or adapter_config.get("adapter_layers")
            ),
        ).to(device)
        model.load_lora_state_dict(adapter_weights)

    elif adapter_type == "film":
        from pawn.adapters.film import FiLMCLM
        assert backbone is not None
        has_output = adapter_config.get("use_output_film", False)
        if any(k.startswith("output_film.") for k in adapter_weights):
            has_output = True
        model = FiLMCLM(backbone, use_output_film=has_output).to(device)
        model.load_film_state_dict(adapter_weights)

    elif adapter_type == "hybrid":
        from pawn.adapters.hybrid import HybridCLM
        assert backbone is not None
        model = HybridCLM(
            backbone,
            lora_rank=adapter_config.get("lora_rank", 4),
            lora_alpha=adapter_config.get("lora_alpha", None),
            attn_targets=adapter_config.get("lora_targets", "qkvo"),
            adapt_ffn=adapter_config.get("lora_ffn", False),
            lora_layers=_layer_tuple(adapter_config.get("lora_layers")),
            use_film=adapter_config.get("use_film", True),
            use_output_film=adapter_config.get("use_output_film", False),
            film_layers=_layer_tuple(adapter_config.get("film_layers")),
        ).to(device)
        model.load_adapter_state_dict(adapter_weights)

    elif adapter_type == "sparse":
        from pawn.adapters.sparse import SparseCLM
        assert backbone is not None
        model = SparseCLM(
            backbone,
            density=adapter_config.get("density", 0.01),
            attn_targets=_ATTN_PRESETS.get(
                adapter_config.get("sparse_targets", "qkvo"),
                ("wq", "wk", "wv", "wo"),
            ),
            adapt_ffn=adapter_config.get("sparse_ffn", False),
            layers=_layer_tuple(adapter_config.get("sparse_layers")),
            seed=adapter_config.get("sparse_seed", 42),
        ).to(device)
        model.load_sparse_state_dict(adapter_weights)

    elif adapter_type == "bottleneck":
        from pawn.adapters.bottleneck import BottleneckCLM
        assert backbone is not None
        model = BottleneckCLM(
            backbone,
            bottleneck_dim=adapter_config.get("bottleneck_dim", 8),
            adapt_attn=adapter_config.get("adapt_attn", True),
            adapt_ffn=adapter_config.get("adapt_ffn", True),
            layers=_layer_tuple(adapter_config.get("adapter_layers")),
        ).to(device)
        model.load_adapter_state_dict(adapter_weights)

    elif adapter_type in ("retro_bottleneck", "retro_sparse", "rosa"):
        # Retro-* and joint RoSA all need the backbone decorated with
        # ``SparseLinear`` modules so the saved sparse deltas have a
        # home to load into. ``SparseCLM`` mutates ``backbone`` in place.
        from pawn.adapters.sparse import SparseCLM
        from pawn.adapters.rosa import RetroBottleneckCLM, RoSACLM
        assert backbone is not None

        sparse_targets = _ATTN_PRESETS.get(
            adapter_config.get("sparse_targets", "qkvo"),
            ("wq", "wk", "wv", "wo"),
        )
        # The sparse ``layers`` arg in retro modes was always None at
        # train time (sparse on all attn layers), so we reproduce that
        # here unless the saved config explicitly set ``sparse_layers``.
        sparse_layers = _layer_tuple(adapter_config.get("sparse_layers"))
        SparseCLM(
            backbone,
            density=adapter_config.get("density") or 0.01,
            attn_targets=sparse_targets,
            adapt_ffn=adapter_config.get("sparse_ffn", False),
            layers=sparse_layers,
        )

        if adapter_type == "retro_bottleneck":
            model = RetroBottleneckCLM(
                backbone,
                bottleneck_dim=adapter_config.get("bottleneck_dim", 8),
                adapt_attn=adapter_config.get("adapt_attn", True),
                adapt_ffn=adapter_config.get("adapt_ffn", True),
                layers=_layer_tuple(adapter_config.get("adapter_layers")),
            ).to(device)
            model.load_adapter_state_dict(adapter_weights)
        elif adapter_type == "retro_sparse":
            from pawn.adapters.sparse import SparseCLM as _SparseCLM
            # The sparse decoration is already in place; wrap with a
            # SparseCLM facade so the existing forward path works.
            model = _SparseCLM(
                backbone,
                density=adapter_config.get("density") or 0.01,
                attn_targets=sparse_targets,
                adapt_ffn=adapter_config.get("sparse_ffn", False),
                layers=sparse_layers,
            ).to(device)
            model.load_sparse_state_dict(adapter_weights)
        else:  # joint rosa
            model = RoSACLM(
                backbone,
                rank=adapter_config.get("lora_rank", 4),
                attn_targets=adapter_config.get("lora_targets", "qkvo"),
                adapt_ffn=adapter_config.get("lora_ffn", False),
                layers=_layer_tuple(adapter_config.get("adapter_layers")),
                lora_enabled=True,
                sparse_enabled=True,
            ).to(device)
            model.load_adapter_state_dict(adapter_weights)

    else:
        raise ValueError(f"Unhandled adapter type: {adapter_type!r}")

    model.eval()
    return model, adapter_type


@torch.no_grad()
def evaluate_maia(
    model,
    dataloader,
    mask_builder,
    device: str,
    min_eval_ply: int = 10,
    amp_dtype: torch.dtype | None = None,
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

    # Per-ply stats (only populated when per_ply=True)
    ply_top1: dict[int, float] = {}
    ply_count: dict[int, int] = {}

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        tgt = batch["targets"].to(device)
        msk = batch["loss_mask"].to(device)
        legal_mask = mask_builder(batch)

        B, T = ids.shape

        # Full forward to get all logits. ``amp_dtype=None`` runs fp32
        # (the safe default for ceiling-scale adapters whose adapter
        # activations exceed fp16 range and silently NaN otherwise).
        if amp_dtype is None:
            logits = model(ids)
        else:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
                logits = model(ids)

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
    amp_dtype_map = {
        "none": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    amp_dtype: torch.dtype | None = amp_dtype_map[args.amp_dtype]
    if amp_dtype is not None and not device.startswith("cuda"):
        # AMP autocast is a CUDA-only path here; on CPU just run fp32.
        amp_dtype = None

    print(f"Loading model + adapter...")
    model, adapter_type = load_model(args.checkpoint, args.adapter_checkpoint, device)
    print(f"  Adapter type: {adapter_type}")

    # Auto-detect the backbone's training-time sequence format so the
    # data layout matches what the model saw. `--prepend-outcome` forces
    # the flag when the checkpoint metadata is missing or ambiguous.
    # specialized_clm runs don't carry a backbone, so there's no
    # metadata to read — just default to pure-moves unless the user
    # forced the flag explicitly.
    from pawn.checkpoint import get_prepend_outcome, read_checkpoint_metadata
    if args.prepend_outcome:
        prepend_outcome = True
        print("  prepend_outcome: True (forced by --prepend-outcome)")
    elif args.checkpoint is None:
        prepend_outcome = False
        print(
            "  prepend_outcome: False "
            "(no backbone checkpoint; standalone model)"
        )
    else:
        try:
            saved = read_checkpoint_metadata(args.checkpoint)
            prepend_outcome = get_prepend_outcome(saved.get("training_config"))
            print(f"  prepend_outcome: {prepend_outcome} (from checkpoint metadata)")
        except (FileNotFoundError, OSError, ValueError):
            prepend_outcome = False
            print("  prepend_outcome: False (default; checkpoint metadata unavailable)")

    # Prepare data
    seq_len = model.cfg.max_seq_len
    print(f"Preparing evaluation data: {args.pgn}")
    if args.elo_min is not None or args.elo_max is not None:
        print(
            f"  Elo filter: [{args.elo_min}, {args.elo_max}) "
            "(both players in range)"
        )
    data = prepare_lichess_dataset(
        args.pgn, max_ply=seq_len, max_games=args.max_games, min_ply=10,
        elo_min=args.elo_min, elo_max=args.elo_max,
        prepend_outcome=prepend_outcome,
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
    mask_builder = LegalMaskBuilder(
        args.batch_size, seq_len=seq_len, vocab_size=vocab_size, device=device,
        prepend_outcome=prepend_outcome,
    )

    # Evaluate
    print(f"\nEvaluating (min_eval_ply={args.min_eval_ply})...")
    results = evaluate_maia(
        model, val_loader, mask_builder, device,
        min_eval_ply=args.min_eval_ply,
        amp_dtype=amp_dtype,
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
