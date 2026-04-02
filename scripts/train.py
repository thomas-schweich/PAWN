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
    parser.add_argument("--discard-ply-limit", action="store_true",
                        help="Only train on games that ended naturally (no ply limit truncation)")
    parser.add_argument("--no-outcome-token", action="store_true",
                        help="Strip outcome token from sequences (ablation experiment)")
    parser.add_argument("--mate-boost", type=float, default=0.0,
                        help="Probability of taking mate-in-1 when available (0.0=random, 1.0=always)")

    # Architecture overrides (for sweeps — override the named variant)
    parser.add_argument("--d-model", type=int, default=None, help="Override d_model")
    parser.add_argument("--n-layers", type=int, default=None, help="Override n_layers")
    parser.add_argument("--n-heads", type=int, default=None, help="Override n_heads")
    parser.add_argument("--d-ff", type=int, default=None, help="Override d_ff")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps")

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--hf-repo", type=str, default=None,
                            help="Push checkpoints to this HuggingFace repo (requires HF_TOKEN)")
    ckpt_group.add_argument("--local-checkpoints", action="store_true",
                            help="Save checkpoints locally only (no HuggingFace push)")
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
    if args.discard_ply_limit:
        train_cfg.discard_ply_limit = True
    if args.no_outcome_token:
        train_cfg.no_outcome_token = True
    if args.mate_boost > 0.0:
        train_cfg.mate_boost = args.mate_boost

    # Architecture overrides
    if args.d_model is not None:
        model_cfg.d_model = args.d_model
    if args.n_layers is not None:
        model_cfg.n_layers = args.n_layers
    if args.n_heads is not None:
        model_cfg.n_heads = args.n_heads
    if args.d_ff is not None:
        model_cfg.d_ff = args.d_ff
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.warmup_steps is not None:
        train_cfg.warmup_steps = args.warmup_steps

    print(f"Model config: {model_cfg}")
    print(f"Training config: {train_cfg}")

    trainer = CLMTrainer(train_cfg, model_cfg, hf_repo=args.hf_repo)

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
