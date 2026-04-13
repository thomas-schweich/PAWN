#!/usr/bin/env python3
"""Run linear probes on all trained checkpoints and write results to JSON."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.eval_suite.probes import extract_probe_data, train_all_probes


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> PAWNCLM:
    from pawn.checkpoint import load_backbone_weights
    # Load weights on CPU first, load them into a CPU-instantiated model,
    # then move the final model to the target device. This avoids the
    # transient 2x-on-device peak that happens when both the state_dict
    # and the model parameters are briefly resident on the GPU during load.
    state_dict, model_config = load_backbone_weights(checkpoint_path, "cpu")
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
    model = PAWNCLM(cfg)
    model.load_state_dict(state_dict)
    del state_dict
    model.to(device)
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
    parser.add_argument("--top-layer-only", action="store_true",
                        help="Only probe the top layer (skip per-layer sweep)")
    parser.add_argument("--max-ply", type=int, default=None,
                        help="Override probe sequence length (defaults to model's max_seq_len)")
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
        # Prefer the run's top-level config.json (keys: model/training); fall
        # back to the per-checkpoint config (keys: model_config/training_config).
        cfg: dict = {}
        run_cfg_file = ckpt_path.parent.parent / "config.json"
        ckpt_cfg_file = ckpt_path / "config.json"
        if run_cfg_file.exists():
            with open(run_cfg_file) as f:
                cfg = json.load(f)
        elif ckpt_cfg_file.exists():
            with open(ckpt_cfg_file) as f:
                raw = json.load(f)
            # Normalize to the run-level schema
            cfg = {
                "model": raw.get("model_config") or raw.get("model") or {},
                "training": raw.get("training_config") or raw.get("training") or {},
            }
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

    # Size-1 LRU probe-data cache keyed by (max_ply, prepend_outcome). If the
    # script sweeps runs with differing seq lengths or outcome-prefix settings
    # this keeps exactly one dataset resident — regenerating is much cheaper
    # than carrying multiple multi-GiB caches.
    _probe_cache: dict = {"key": None, "data": None}

    def get_probe_data(max_ply: int, prepend_outcome: bool) -> tuple[dict, dict]:
        key = (max_ply, prepend_outcome)
        if _probe_cache["key"] != key:
            # Drop the previous entry before generating the new one.
            _probe_cache["data"] = None
            _probe_cache["key"] = None
            gc.collect()
            print(
                f"\nGenerating probe data: {args.n_games} train + {args.n_val_games} val games "
                f"(max_ply={max_ply}, prepend_outcome={prepend_outcome})..."
            )
            train_data = extract_probe_data(
                args.n_games, max_ply, seed=12345, prepend_outcome=prepend_outcome,
            )
            val_data = extract_probe_data(
                args.n_val_games, max_ply, seed=54321, prepend_outcome=prepend_outcome,
            )
            print("Done.")
            _probe_cache["key"] = key
            _probe_cache["data"] = (train_data, val_data)
        return _probe_cache["data"]

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

        # Auto-detect prepend_outcome from the checkpoint's saved config.
        # Uses infer_prepend_outcome so legacy-vocab / 256-ctx checkpoints
        # (which were always outcome-prefixed pre-2026-04-08) are handled
        # correctly; modern checkpoints use the exact persisted flag.
        from pawn.checkpoint import infer_prepend_outcome
        prepend_outcome, reason = infer_prepend_outcome(train_cfg, model_cfg)
        if args.no_outcome_token:
            prepend_outcome = False
            reason = "forced by --no-outcome-token CLI override"
        no_outcome = not prepend_outcome
        max_ply = args.max_ply or model_cfg.get("max_seq_len") or 256
        print(
            f"  max_ply={max_ply}, prepend_outcome={prepend_outcome} ({reason})"
        )

        train_data, val_data = get_probe_data(max_ply, prepend_outcome)

        results = train_all_probes(
            model, train_data, val_data, device,
            per_layer=not args.top_layer_only, n_epochs=args.n_epochs, verbose=True,
            no_outcome_token=no_outcome, use_amp=(device != "cpu"),
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
