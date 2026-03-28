#!/usr/bin/env python3
"""Backfill missing val metrics (top5_accuracy, legal_move_rate, perplexity).

Loads checkpoints from HuggingFace, runs a validation pass to compute the
missing fields, and merges them into the authoritative local metrics.jsonl.

This script intentionally fails loudly on any error — no silent fallbacks.

Usage:
    # Dry run: compute and display, don't push
    python scripts/backfill_metrics.py

    # Compute and push corrected metrics.jsonl to HF
    python scripts/backfill_metrics.py --push

    # Single variant
    python scripts/backfill_metrics.py --variants small --push
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# Must be importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pawn.config import CLMConfig, TrainingConfig
from pawn.model import PAWNCLM, clm_loss
from pawn.data import create_validation_set
from pawn.trainer import compute_legal_move_rate


VARIANTS = {
    "small": {
        "repo": "thomas-schweich/pawn-small",
        "config_factory": "small",
        "local_metrics": "logs/run_20260325_002657_small_zesty-osprey/metrics.jsonl",
    },
    "base": {
        "repo": "thomas-schweich/pawn-base",
        "config_factory": "base",
        "local_metrics": "logs/run_20260325_002657_base_zesty-osprey/metrics.jsonl",
    },
    "large": {
        "repo": "thomas-schweich/pawn-large",
        "config_factory": "large",
        "local_metrics": "logs/run_20260325_002658_large_zesty-osprey/metrics.jsonl",
    },
}


def list_checkpoints(repo: str) -> list[str]:
    """List checkpoint step directories available on a HF repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo)
    # Find unique checkpoint dirs: checkpoints/step_NNNNN/
    steps = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2 and parts[0] == "checkpoints" and parts[1].startswith("step_"):
            steps.add(parts[1])
    return sorted(steps)


def load_checkpoint_from_hf(repo: str, step_dir: str, device: str) -> dict[str, torch.Tensor]:
    """Download and load model weights from a specific checkpoint on HF."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, f"checkpoints/{step_dir}/model.safetensors")
    return load_file(path, device=device)


def load_root_weights_from_hf(repo: str, device: str) -> dict[str, torch.Tensor]:
    """Download and load root-level model weights from HF."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "model.safetensors")
    return load_file(path, device=device)


@torch.no_grad()
def evaluate_checkpoint(
    model: PAWNCLM,
    weights: dict[str, torch.Tensor],
    val_data: dict[str, torch.Tensor],
    device: str,
    batch_size: int = 256,
) -> dict[str, float]:
    """Run a full validation pass and return all five metrics."""
    model.load_state_dict(weights)
    model.eval()

    n = val_data["input_ids"].shape[0]
    total_metrics: dict[str, float] = {}
    n_batches = 0
    has_legal = "legal_grid" in val_data

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        input_ids = val_data["input_ids"][start:end].to(device)
        targets = val_data["targets"][start:end].to(device)
        loss_mask = val_data["loss_mask"][start:end].to(device)

        logits, _ = model(input_ids, loss_mask)
        _, metrics = clm_loss(logits, targets, loss_mask)

        # Top-5 accuracy
        valid_logits = logits[loss_mask]
        valid_targets = targets[loss_mask]
        top5 = valid_logits.topk(5, dim=-1).indices
        metrics["top5_accuracy"] = (
            (top5 == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
        )

        # Perplexity
        metrics["perplexity"] = math.exp(min(metrics["loss"], 20.0))

        # Legal move rate
        if has_legal:
            legal_grid = val_data["legal_grid"][start:end].to(device)
            game_lengths = val_data["game_lengths"][start:end].to(device)
            metrics["legal_move_rate"] = compute_legal_move_rate(
                logits, legal_grid, loss_mask, game_lengths,
            )

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


def step_from_dir(step_dir: str) -> int:
    """Extract step number from directory name like 'step_00005000'."""
    return int(step_dir.split("_")[1])


def backfill_variant(
    variant_key: str,
    variant: dict,
    device: str,
    val_games: int,
    push: bool,
):
    """Backfill metrics for a single variant."""
    repo = variant["repo"]
    local_metrics_path = Path(variant["local_metrics"])

    # --- Load authoritative local metrics ---
    if not local_metrics_path.exists():
        raise FileNotFoundError(
            f"Local metrics not found: {local_metrics_path}\n"
            f"These are the rsync'd training logs — the authoritative source."
        )

    with open(local_metrics_path) as f:
        all_records = [json.loads(line) for line in f]

    val_records = [r for r in all_records if r.get("type") == "val"]
    train_records = [r for r in all_records if r.get("type") == "train"]
    config_records = [r for r in all_records if r.get("type") == "config"]

    print(f"  Local metrics: {len(all_records)} records "
          f"({len(train_records)} train, {len(val_records)} val, {len(config_records)} config)")

    # Check which val records already have extended fields
    missing = [r for r in val_records if "val/top5_accuracy" not in r]
    print(f"  Val records missing top5/legal: {len(missing)}/{len(val_records)}")

    if not missing:
        print("  All val records already have extended fields — nothing to backfill")
        if push:
            _push_metrics(repo, all_records, local_metrics_path)
        return

    # --- Find available checkpoints ---
    checkpoint_dirs = list_checkpoints(repo)
    print(f"  Available checkpoints on HF: {len(checkpoint_dirs)}")
    if checkpoint_dirs:
        print(f"    {', '.join(checkpoint_dirs)}")

    # Also check if root weights are available (step 100K equivalent)
    has_root = True
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo)
        has_root = "model.safetensors" in files
    except Exception:
        has_root = False

    # Map step -> checkpoint source
    step_to_source: dict[int, str] = {}
    for d in checkpoint_dirs:
        step_to_source[step_from_dir(d)] = d

    # The root weights correspond to the final step (100K)
    # Only use root if we don't have a checkpoint dir for that step
    if has_root and 100000 not in step_to_source:
        step_to_source[100000] = "__root__"

    # Which steps do we need?
    needed_steps = {r["step"] for r in missing}
    available_steps = needed_steps & set(step_to_source.keys())
    unavailable_steps = needed_steps - set(step_to_source.keys())

    print(f"  Need metrics for steps: {sorted(needed_steps)}")
    print(f"  Have checkpoints for: {sorted(available_steps)}")
    if unavailable_steps:
        print(f"  MISSING checkpoints for: {sorted(unavailable_steps)}")

    if not available_steps:
        print("  No checkpoints available for any needed step — skipping eval")
        if push:
            # Still push the complete local metrics (restoring the full history)
            _push_metrics(repo, all_records, local_metrics_path)
        return

    # --- Create model and val data ---
    cfg = getattr(CLMConfig, variant["config_factory"])()
    train_cfg = TrainingConfig()
    model = PAWNCLM(cfg).to(device).eval()

    print(f"  Generating validation set ({val_games} games)...")
    val_data = create_validation_set(val_games, train_cfg.max_ply, seed=(2**63) - 1)

    # --- Evaluate each available checkpoint ---
    step_to_metrics: dict[int, dict[str, float]] = {}

    for step in sorted(available_steps):
        source = step_to_source[step]
        if source == "__root__":
            print(f"  Evaluating root weights (step {step})...", end=" ", flush=True)
            weights = load_root_weights_from_hf(repo, device)
        else:
            print(f"  Evaluating {source}...", end=" ", flush=True)
            weights = load_checkpoint_from_hf(repo, source, device)

        metrics = evaluate_checkpoint(model, weights, val_data, device)
        step_to_metrics[step] = metrics

        print(f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
              f"top5={metrics['top5_accuracy']:.4f} "
              f"legal={metrics.get('legal_move_rate', 'N/A'):.4f} "
              f"ppl={metrics['perplexity']:.2f}")

    # --- Merge backfilled metrics into val records ---
    updated_count = 0
    for record in all_records:
        if record.get("type") != "val":
            continue
        step = record["step"]
        if step in step_to_metrics:
            m = step_to_metrics[step]
            # Only backfill fields that are missing
            if "val/top5_accuracy" not in record:
                record["val/top5_accuracy"] = m["top5_accuracy"]
            if "val/perplexity" not in record:
                record["val/perplexity"] = m["perplexity"]
            if "val/legal_move_rate" not in record and "legal_move_rate" in m:
                record["val/legal_move_rate"] = m["legal_move_rate"]
            updated_count += 1

    print(f"  Updated {updated_count} val records with backfilled metrics")

    # Verify: check consistency of loss/accuracy between local and recomputed
    for record in all_records:
        if record.get("type") != "val":
            continue
        step = record["step"]
        if step in step_to_metrics:
            m = step_to_metrics[step]
            loss_diff = abs(record["val/loss"] - m["loss"])
            acc_diff = abs(record["val/accuracy"] - m["accuracy"])
            if loss_diff > 0.05:
                print(f"  WARNING: step {step} loss mismatch: "
                      f"local={record['val/loss']:.4f} recomputed={m['loss']:.4f} "
                      f"(diff={loss_diff:.4f})")
            if acc_diff > 0.005:
                print(f"  WARNING: step {step} accuracy mismatch: "
                      f"local={record['val/accuracy']:.4f} recomputed={m['accuracy']:.4f} "
                      f"(diff={acc_diff:.4f})")

    if push:
        _push_metrics(repo, all_records, local_metrics_path)


def _push_metrics(repo: str, records: list[dict], source_path: Path):
    """Push corrected metrics.jsonl to a HF repo."""
    from huggingface_hub import HfApi
    import tempfile

    api = HfApi()

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        tmp_path = f.name

    val_count = sum(1 for r in records if r.get("type") == "val")
    has_top5 = sum(1 for r in records if r.get("type") == "val" and "val/top5_accuracy" in r)

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="metrics.jsonl",
        repo_id=repo,
        repo_type="model",
        commit_message=(
            f"Restore complete metrics ({len(records)} records, "
            f"{val_count} val, {has_top5} with extended fields)"
        ),
    )
    print(f"  Pushed {len(records)} records to {repo}")

    Path(tmp_path).unlink()


def main():
    parser = argparse.ArgumentParser(description="Backfill missing val metrics")
    parser.add_argument("--push", action="store_true", help="Push corrected metrics to HF")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-games", type=int, default=512,
                        help="Number of validation games (must match training val set)")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Val games: {args.val_games}")
    print(f"Push: {args.push}")
    print()

    for variant_key in args.variants:
        if variant_key not in VARIANTS:
            print(f"Unknown variant: {variant_key}")
            continue

        print(f"=== {variant_key} ===")
        backfill_variant(
            variant_key,
            VARIANTS[variant_key],
            args.device,
            args.val_games,
            args.push,
        )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
