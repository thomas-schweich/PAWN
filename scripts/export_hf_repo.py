#!/usr/bin/env python3
"""Export a training run to HuggingFace repo format.

Converts .pt checkpoints to safetensors and structures files for HF upload:
  - Root: best checkpoint (model.safetensors, config.json, metrics.jsonl, README.md)
  - checkpoints/step_NNNN/: other checkpoints with truncated metrics

Usage:
    python scripts/export_hf_repo.py \
        --run-dir logs/run_20260322_182707 \
        --output-dir export/pawn-base \
        --repo-name pawn-base \
        --github-url https://github.com/thomas-schweich/PAWN
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def find_best_step(metrics_path: Path) -> int | None:
    """Find the step with lowest val loss from metrics.jsonl."""
    best_loss = float("inf")
    best_step = None
    with open(metrics_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") != "val":
                continue
            loss = record.get("val/loss", float("inf"))
            step = record.get("step")
            if loss < best_loss and step is not None:
                best_loss = loss
                best_step = step
    return best_step


def truncate_metrics(metrics_path: Path, up_to_step: int) -> list[str]:
    """Return metrics lines up to and including the given step."""
    lines = []
    with open(metrics_path) as f:
        for line in f:
            lines.append(line)
            record = json.loads(line)
            if record.get("type") in ("train", "val") and record.get("step", 0) > up_to_step:
                break
    return lines


def convert_pt_to_safetensors(pt_path: Path, output_dir: Path):
    """Convert a .pt checkpoint to safetensors + JSON directory format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    # Model weights -> safetensors
    state_dict = ckpt["model_state_dict"]
    tensors = {k: v.cpu().contiguous() for k, v in state_dict.items()}
    save_file(tensors, output_dir / "model.safetensors")

    # Config
    config = {
        "format_version": 1,
        "checkpoint_type": "pretrain",
        "model_config": ckpt.get("model_config", {}),
        "training_config": ckpt.get("training_config", {}),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Optimizer -> safetensors (if present)
    if "optimizer_state_dict" in ckpt:
        from pawn.checkpoint import _flatten_optimizer_state, _rng_to_json, _json_default
        opt_tensors, opt_meta = _flatten_optimizer_state(ckpt["optimizer_state_dict"])
        if opt_tensors:
            save_file(opt_tensors, output_dir / "optimizer.safetensors")

        # Training state
        training_state = {
            "format_version": 1,
            "global_step": ckpt.get("global_step", 0),
            "scheduler_state_dict": ckpt.get("scheduler_state_dict"),
            "scaler_state_dict": ckpt.get("scaler_state_dict"),
            "optimizer_meta": opt_meta,
        }
        rng_state = {}
        if ckpt.get("torch_rng_state") is not None:
            rng_state.update(_rng_to_json(ckpt["torch_rng_state"], ckpt.get("cuda_rng_state")))
        training_state.update(rng_state)

        with open(output_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2, default=_json_default)


def generate_readme(
    repo_name: str, model_config: dict, training_config: dict,
    best_step: int, val_loss: float, val_acc: float,
    github_url: str, extra_desc: str = "",
) -> str:
    """Generate a HuggingFace model card README."""
    d_model = model_config.get("d_model", "?")
    n_layers = model_config.get("n_layers", "?")
    n_heads = model_config.get("n_heads", "?")
    discard = training_config.get("discard_ply_limit", False)

    # Infer variant name
    variant = "base"
    if d_model == 256:
        variant = "small"
    elif d_model == 640:
        variant = "large"

    params = {"small": "9.5M", "base": "35.8M", "large": "68.4M"}.get(variant, "?")

    return f"""---
license: apache-2.0
library_name: pytorch
tags:
  - chess
  - transformer
  - causal-lm
  - world-model
datasets:
  - random-self-play
model-index:
  - name: {repo_name}
    results:
      - task:
          type: next-move-prediction
        metrics:
          - name: Val Loss
            type: loss
            value: {val_loss}
          - name: Val Accuracy
            type: accuracy
            value: {val_acc}
---

# {repo_name.upper()}

A causal transformer trained on random chess games, designed as a testbed for finetuning and augmentation methods at small scales.
{extra_desc}

## Model Details

| | |
|---|---|
| **Parameters** | {params} |
| **Architecture** | Decoder-only transformer (RMSNorm, SwiGLU, RoPE) |
| **d_model** | {d_model} |
| **Layers** | {n_layers} |
| **Heads** | {n_heads} |
| **Vocabulary** | 4,278 tokens (4,096 grid + 176 promotions + 5 outcomes + 1 PAD) |
| **Sequence length** | 256 |
| **Best val loss** | {val_loss:.4f} (step {best_step:,}) |
| **Best val accuracy** | {val_acc:.1%} |

## Usage

```python
import torch
from safetensors.torch import load_file
from pawn.config import CLMConfig
from pawn.model import PAWNCLM

cfg = CLMConfig.{variant}()
model = PAWNCLM(cfg)
model.load_state_dict(load_file("model.safetensors"))
model.eval()
```

## Training

Trained from scratch on random self-play games generated by a Rust chess engine (shakmaty).
See the [PAWN repository]({github_url}) for training code, data pipeline, and evaluation suite.

## License

Apache 2.0
"""


def main():
    parser = argparse.ArgumentParser(description="Export training run to HF repo format")
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF repo")
    parser.add_argument("--repo-name", required=True, help="Repository name for README")
    parser.add_argument("--github-url", default="https://github.com/thomas-schweich/PAWN")
    parser.add_argument("--best-only", action="store_true", help="Only export best checkpoint")
    parser.add_argument("--extra-desc", default="", help="Extra description for README")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    metrics_path = run_dir / "metrics.jsonl"

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found")
        return

    # Find best step
    best_step = find_best_step(metrics_path)
    if best_step is None:
        print("ERROR: No val records found in metrics.jsonl")
        return
    print(f"Best val step: {best_step}")

    # Find best val metrics
    best_val_loss, best_val_acc = float("inf"), 0.0
    with open(metrics_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "val" and r.get("step") == best_step:
                best_val_loss = r.get("val/loss", float("inf"))
                best_val_acc = r.get("val/accuracy", 0.0)
                break

    # Find all .pt checkpoints
    ckpt_dir = run_dir / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("step_*.pt")) if ckpt_dir.exists() else []
    if not checkpoints:
        print("ERROR: No checkpoints found")
        return

    # Find nearest checkpoint to best step
    best_ckpt = min(checkpoints, key=lambda p: abs(
        int(p.stem.replace("step_", "")) - best_step
    ))
    print(f"Best checkpoint: {best_ckpt}")

    # Read config from checkpoint
    ckpt_data = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
    model_config = ckpt_data.get("model_config", {})
    training_config = ckpt_data.get("training_config", {})
    del ckpt_data

    # Export best checkpoint to root
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting best checkpoint to {output_dir}/")
    convert_pt_to_safetensors(best_ckpt, output_dir)

    # Copy full metrics.jsonl
    shutil.copy2(metrics_path, output_dir / "metrics.jsonl")

    # Generate README
    readme = generate_readme(
        args.repo_name, model_config, training_config,
        best_step, best_val_loss, best_val_acc,
        args.github_url, args.extra_desc,
    )
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    # Export other checkpoints
    if not args.best_only:
        for ckpt in checkpoints:
            if ckpt == best_ckpt:
                continue
            step_name = ckpt.stem  # e.g. "step_00005000"
            step_num = int(step_name.replace("step_", ""))
            step_dir = output_dir / "checkpoints" / step_name
            print(f"  Exporting {step_name}...")
            convert_pt_to_safetensors(ckpt, step_dir)

            # Truncated metrics
            truncated = truncate_metrics(metrics_path, step_num)
            with open(step_dir / "metrics.jsonl", "w") as f:
                f.writelines(truncated)

    print(f"\nExport complete: {output_dir}")
    print(f"  Best: model.safetensors, config.json, metrics.jsonl, README.md")
    if not args.best_only:
        print(f"  Checkpoints: {len(checkpoints) - 1} in checkpoints/")


if __name__ == "__main__":
    main()
