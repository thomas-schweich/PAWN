#!/usr/bin/env python3
"""Train multiple PAWN model variants simultaneously on shared data.

All models see the exact same batches in the same order, eliminating
data generation overhead and ensuring comparable training conditions.

Usage:
    uv run python scripts/train_all.py --local-checkpoints
    uv run python scripts/train_all.py --hf-repo thomas-schweich/pawn-{variant}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

from pawn.cotrain import ModelSlot, run_cotrain
from pawn.model import PAWNCLM
from pawn.run_config import CotrainConfig, CotrainVariant


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train small/base/large PAWN models simultaneously")
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    p.add_argument("--total-steps", type=int, default=100_000, help="Total training steps")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size (shared across models)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--checkpoint-interval", type=int, default=5000)
    p.add_argument("--discard-ply-limit", action="store_true")
    p.add_argument("--no-outcome-token", action="store_true",
                    help="Deprecated no-op (sequences are pure moves by default)")
    p.add_argument("--prepend-outcome", action="store_true",
                    help="Prepend outcome token at position 0 for outcome-conditioned training")
    p.add_argument("--patience", type=int, default=10,
                    help="Stop if no val loss improvement for N eval intervals (0=disabled)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--legacy-vocab", action="store_true",
                    help="Use old 4284-token PAWN vocabulary (for reproducing old experiments)")

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None,
                            help="HF repo prefix (appends -{variant}). E.g. thomas-schweich/pawn")
    ckpt_group.add_argument("--local-checkpoints", action="store_true")

    p.add_argument("--shm-checkpoints", action="store_true",
                    help="Write checkpoints to /dev/shm (RAM-backed, instant writes). "
                         "Requires --hf-repo since /dev/shm is volatile.")

    p.add_argument("--run-evals", action="store_true",
                    help="Run probes, diagnostics, and Lichess eval after training completes")
    p.add_argument("--lichess-pgn", type=str, default=None,
                    help="Path to Lichess PGN file for eval (required with --run-evals)")
    p.add_argument("--publish-results", action="store_true",
                    help="Push eval results to HuggingFace (requires --hf-repo and --run-evals)")
    return p.parse_args()


def _args_to_cotrain_config(args) -> CotrainConfig:
    """Build a CotrainConfig from argparse namespace."""
    variants = [
        CotrainVariant(name="small", variant="small", legacy_vocab=args.legacy_vocab),
        CotrainVariant(name="base", variant="base", legacy_vocab=args.legacy_vocab),
        CotrainVariant(name="large", variant="large", legacy_vocab=args.legacy_vocab),
    ]

    return CotrainConfig(
        variants=variants,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        discard_ply_limit=args.discard_ply_limit,
        no_outcome_token=args.no_outcome_token,
        prepend_outcome=args.prepend_outcome,
        patience=args.patience,
        wandb=args.wandb,
        hf_repo=args.hf_repo,
        local_checkpoints=args.local_checkpoints,
        shm_checkpoints=args.shm_checkpoints,
    )


def _run_post_training_evals(slots: list[ModelSlot], args):
    """Run probes, diagnostics, and Lichess eval on best checkpoint per variant."""
    from pawn.eval_suite.probes import extract_probe_data, train_all_probes
    from pawn.eval_suite.diagnostics import (
        generate_diagnostic_corpus,
        extract_diagnostic_positions, evaluate_diagnostic_positions,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for slot in slots:
        print(f"\n--- Evaluating {slot.name} ---")

        # Use tracked best val step (kept on /dev/shm if shm_checkpoints)
        best_step = slot.best_val_step
        best_loss = slot.best_val_loss

        ckpt_path = os.path.join(slot.checkpoint_dir, f"step_{best_step:08d}")
        if not os.path.isdir(ckpt_path):
            # Fall back to latest
            ckpts = sorted(Path(slot.checkpoint_dir).glob("step_*"))
            ckpt_path = str(ckpts[-1]) if ckpts else None

        if not ckpt_path:
            print(f"  No checkpoint found, skipping")
            continue

        print(f"  Best checkpoint: {ckpt_path} (val_loss={best_loss:.4f})")

        # Load model (unwrapped)
        from pawn.checkpoint import load_backbone_weights
        state_dict, _ = load_backbone_weights(ckpt_path)
        model = PAWNCLM(slot.model_cfg).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        results = {}

        # 1. Probes — match the checkpoint's training-time sequence
        # format, not the extract_probe_data / train_all_probes defaults.
        # Without this, probe data for a --prepend-outcome run would be
        # generated as pure moves while hidden states were extracted at
        # ply_offset=0, measuring off-by-one shifted representations.
        prepend_outcome = slot.train_cfg.prepend_outcome
        no_outcome_token = not prepend_outcome
        print(f"  Running probes (prepend_outcome={prepend_outcome})...")
        train_data = extract_probe_data(
            2048, 256, seed=12345, prepend_outcome=prepend_outcome,
        )
        val_data = extract_probe_data(
            512, 256, seed=54321, prepend_outcome=prepend_outcome,
        )
        probe_results = train_all_probes(
            model, train_data, val_data, device=device,
            per_layer=True, n_epochs=20, verbose=True,
            no_outcome_token=no_outcome_token,
        )
        results["probes"] = probe_results
        del train_data, val_data

        # 2. Diagnostics
        print("  Running diagnostics...")
        corpus = generate_diagnostic_corpus(n_per_category=10_000)
        positions = extract_diagnostic_positions(corpus, max_per_category=10_000)
        diag_results = evaluate_diagnostic_positions(model, positions, corpus, device=device)
        results["diagnostics"] = diag_results
        del corpus, positions

        # 3. Lichess eval (if PGN provided)
        if args.lichess_pgn:
            print("  Running Lichess eval...")
            from pawn.eval_suite.lichess import prepare_lichess_corpus, evaluate_on_lichess
            lichess_data = prepare_lichess_corpus(args.lichess_pgn, max_games_per_band=1000)
            lichess_results = evaluate_on_lichess(model, lichess_data, device=device)
            results["lichess"] = lichess_results

        # Save results
        results_path = os.path.join(slot.run_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved: {results_path}")

        # Publish to HF
        if args.publish_results and slot.hf_repo and slot.hf_branch:
            from huggingface_hub import HfApi
            api = HfApi()
            try:
                api.upload_file(
                    path_or_fileobj=results_path,
                    path_in_repo="eval_results.json",
                    repo_id=slot.hf_repo,
                    repo_type="model",
                    revision=slot.hf_branch,
                    commit_message=f"Eval results (best step {best_step})",
                )
                print(f"  Published to {slot.hf_repo}@{slot.hf_branch}")
            except Exception as e:
                print(f"  WARNING: HF publish failed: {e}")

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    config = _args_to_cotrain_config(args)
    slots = run_cotrain(config)

    # Post-training evals (CLI-only feature, not available through the lab)
    if args.run_evals:
        print("\n" + "=" * 60)
        print("POST-TRAINING EVALUATION")
        print("=" * 60)
        _run_post_training_evals(slots, args)


if __name__ == "__main__":
    try:
        mp.set_start_method("forkserver", force=True)
    except ValueError:
        mp.set_start_method("spawn", force=True)
    main()
