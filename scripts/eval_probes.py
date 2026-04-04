#!/usr/bin/env python3
"""Run linear probes on all trained checkpoints and write results to JSON."""

import argparse
import json
import sys
from pathlib import Path

import torch

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.eval_suite.probes import extract_probe_data, train_all_probes


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> PAWNCLM:
    from pawn.checkpoint import load_backbone_weights
    state_dict, model_config = load_backbone_weights(checkpoint_path, device)
    if model_config:
        cfg = CLMConfig(**model_config)
    else:
        # Fallback: infer from state dict shapes
        d_model = state_dict["embed.src_embed.weight"].shape[1]
        n_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")) + 1
        if d_model == 256 and n_layers == 8:
            cfg = CLMConfig.small()
        elif d_model == 512 and n_layers == 8:
            cfg = CLMConfig.base()
        elif d_model == 640 and n_layers == 10:
            cfg = CLMConfig.large()
        else:
            cfg = CLMConfig(d_model=d_model, n_layers=n_layers)
    model = PAWNCLM(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run linear probes on checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory containing run dirs")
    parser.add_argument("--n-games", type=int, default=4096, help="Games for probe train set")
    parser.add_argument("--n-val-games", type=int, default=1024, help="Games for probe val set")
    parser.add_argument("--n-epochs", type=int, default=20, help="Probe training epochs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run", type=str, default=None, help="Only evaluate this run dir name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate a single checkpoint path directly (skips log-dir scan)")
    parser.add_argument("--no-outcome-token", action="store_true",
                        help="Strip outcome token from sequences (auto-detected from checkpoint config)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        from pawn.gpu import configure_gpu
        gpu_cfg = configure_gpu()
        import pawn.model as model_module
        model_module.SDPA_BACKEND = gpu_cfg.get("sdpa_backend")

    log_dir = Path(args.log_dir)

    # Find all runs with checkpoints
    runs = []
    if args.checkpoint:
        # Direct checkpoint path — build a minimal run entry
        ckpt_path = Path(args.checkpoint)
        # Try to load config from checkpoint dir
        cfg = {}
        cfg_file = ckpt_path / "config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg = json.load(f)
        run_dir = log_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        runs.append((run_dir, ckpt_path, cfg))
    else:
        for config_path in sorted(log_dir.glob("run_*/config.json")):
            run_dir = config_path.parent
            if args.run and run_dir.name != args.run:
                continue
            # Find checkpoints: directory-based (safetensors) or legacy .pt
            checkpoints = sorted(
                [d for d in run_dir.glob("checkpoints/step_*") if d.is_dir()]
                or list(run_dir.glob("checkpoints/step_*.pt"))
            )
            if not checkpoints:
                continue
            latest = checkpoints[-1]
            with open(config_path) as f:
                cfg = json.load(f)
            runs.append((run_dir, latest, cfg))

    if not runs:
        print("No runs with checkpoints found.")
        sys.exit(1)

    print(f"Found {len(runs)} runs to evaluate")

    # Generate probe data once (shared across all models with same max_ply)
    max_ply = 256
    print(f"\nGenerating probe data: {args.n_games} train + {args.n_val_games} val games...")
    train_data = extract_probe_data(args.n_games, max_ply, seed=12345)
    val_data = extract_probe_data(args.n_val_games, max_ply, seed=54321)
    print("Done.")

    for run_dir, ckpt_path, run_cfg in runs:
        model_cfg = run_cfg.get("model", {})
        train_cfg = run_cfg.get("training", {})
        variant = f"{model_cfg.get('d_model', '?')}d/{model_cfg.get('n_layers', '?')}L"
        discard = train_cfg.get("discard_ply_limit", False)
        step = ckpt_path.stem.replace("step_", "")

        print(f"\n{'='*60}")
        print(f"Run: {run_dir.name}  ({variant}, discard_ply={discard}, step={step})")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model = load_model_from_checkpoint(str(ckpt_path), device)

        # Auto-detect no_outcome_token from checkpoint config
        no_outcome = args.no_outcome_token or train_cfg.get("no_outcome_token", False)
        if no_outcome:
            print(f"  [no_outcome_token=True] Stripping outcome token from probe inputs")

        results = train_all_probes(
            model, train_data, val_data, device,
            per_layer=True, n_epochs=args.n_epochs, verbose=True,
            no_outcome_token=no_outcome, use_amp=(device == "cuda"),
        )

        # Save results
        output = {
            "run": run_dir.name,
            "checkpoint": str(ckpt_path),
            "step": int(step),
            "variant": variant,
            "discard_ply_limit": discard,
            "no_outcome_token": no_outcome,
            "model_config": model_cfg,
            "probes": {
                pname: {
                    lname: {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
                    for lname, metrics in layer_results.items()
                }
                for pname, layer_results in results.items()
            },
        }

        out_path = run_dir / "probe_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {out_path}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
