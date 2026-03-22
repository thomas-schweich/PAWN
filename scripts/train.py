#!/usr/bin/env python3
"""PAWN training entry point."""

import argparse
import sys

import torch
import torch.multiprocessing as mp

from pawn.config import CLMConfig, TrainingConfig
from pawn.trainer import CLMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PAWN model")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["small", "base", "large", "toy"],
                        help="Model variant: small (~10M), base (~36M), large (~68M), toy (testing)")
    parser.add_argument("--toy", action="store_true", help="Alias for --variant toy")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num data workers")
    parser.add_argument("--accumulation-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--resume-logs", action="append", default=[], metavar="RUN_DIR",
                        help="Run directory whose metrics.jsonl should be spliced into the new log")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint dir")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory")
    return parser.parse_args()


def main():
    args = parse_args()

    variant = "toy" if args.toy else args.variant
    variant_factory = {"small": CLMConfig.small, "base": CLMConfig.base,
                       "large": CLMConfig.large, "toy": CLMConfig.toy}
    model_cfg = variant_factory[variant]()
    train_cfg = TrainingConfig.toy() if variant == "toy" else TrainingConfig()

    if args.device:
        train_cfg.device = args.device
    elif not torch.cuda.is_available():
        if variant == "toy":
            train_cfg.device = "cpu"
            print("CUDA not available, falling back to CPU (toy mode)")
        else:
            print("ERROR: CUDA is required for full model training.")
            print("Use --toy for CPU-based testing, or --device cpu to force CPU.")
            sys.exit(1)

    if args.total_steps is not None:
        train_cfg.total_steps = args.total_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.accumulation_steps is not None:
        train_cfg.accumulation_steps = args.accumulation_steps
    if args.wandb:
        train_cfg.use_wandb = True
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        train_cfg.log_dir = args.log_dir

    print(f"Model config: {model_cfg}")
    print(f"Training config: {train_cfg}")

    trainer = CLMTrainer(train_cfg, model_cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.resume_logs:
        trainer.seed_logs(args.resume_logs, trainer.global_step)

    trainer.train()


if __name__ == "__main__":
    try:
        mp.set_start_method("forkserver", force=True)
    except ValueError:
        mp.set_start_method("spawn", force=True)
    main()
