#!/usr/bin/env python3
"""Export a training run to HuggingFace repo format.

Finds the best checkpoint by val loss and packages files for HF upload:
  - Root: best checkpoint (model.safetensors, config.json, .complete,
    metrics.jsonl, README.md)
  - checkpoints/step_NNNN/: other checkpoints with truncated metrics

The v2 trainer writes one ``metrics.jsonl`` row per scan chunk;
val rows are detected by ``row["val_loss"] is not None`` (no separate
``type`` field). The exported README's usage snippet loads via
``pawn.checkpoint.load_model`` (the v2 JAX path).

Usage:
    python scripts/export_hf_repo.py \
        --run-dir logs/jax_adapter_run_20260520_140000_123456_789 \
        --output-dir export/pawn-base \
        --repo-name pawn-base \
        --github-url https://github.com/thomas-schweich/PAWN
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pawn._sentinel import write_sentinel


def find_best_step(metrics_path: Path) -> int | None:
    """Find the step (from ``step_end``) with the lowest val loss.

    v2 ``metrics.jsonl`` has one row per scan chunk; val chunks carry
    ``val_loss`` (a float), non-val chunks carry ``null``. Pretrain
    runs never write ``val_loss`` and will fall through to
    ``None`` — only adapter runs have val data.
    """
    best_loss = float("inf")
    best_step: int | None = None
    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line)
            vloss = row.get("val_loss")
            step = row.get("step_end")
            if vloss is None or step is None:
                continue
            if vloss < best_loss:
                best_loss = vloss
                best_step = step
    return best_step


def truncate_metrics(metrics_path: Path, up_to_step: int) -> list[str]:
    """Return metrics lines whose ``step_end`` is ≤ ``up_to_step``."""
    lines: list[str] = []
    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line)
            step = row.get("step_end")
            if step is not None and step > up_to_step:
                break
            lines.append(line)
    return lines


def copy_checkpoint(src: Path, dst: Path) -> None:
    """Copy a directory-format JAX checkpoint into the HF export layout.

    The source's ``.complete`` sentinel is dropped during copy (its
    SHA-256 hashes are keyed to the source's file set) and a fresh
    sentinel is written for ``dst`` so the exported checkpoint loads
    cleanly via ``pawn.checkpoint.load_model``.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == ".complete":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    write_sentinel(dst)


def generate_readme(
    repo_name: str, model_config: dict, training_config: dict,
    best_step: int, val_loss: float,
    github_url: str, extra_desc: str = "",
) -> str:
    """Generate a HuggingFace model card README.

    All architecture fields come from the checkpoint's saved
    ``model_config``, so the card stays correct regardless of whether
    the checkpoint was trained as a supernet or a sliced variant.
    """
    d_model = model_config.get("d_model", "?")
    n_layers = model_config.get("n_layers", "?")
    n_heads = model_config.get("n_heads", "?")
    d_ff = model_config.get("d_ff", "?")
    vocab_size = model_config.get("vocab_size", "?")
    max_seq_len = model_config.get("max_seq_len", "?")

    return f"""---
license: apache-2.0
library_name: jax
tags:
  - chess
  - transformer
  - causal-lm
  - world-model
  - jax
  - equinox
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
---

# {repo_name.upper()}

A causal transformer trained on random chess games, designed as a testbed for finetuning and augmentation methods at small scales.
{extra_desc}

## Model Details

| | |
|---|---|
| **Architecture** | Decoder-only transformer (RMSNorm, SwiGLU, RoPE), JAX/Equinox |
| **d_model** | {d_model} |
| **Layers** | {n_layers} |
| **Heads** | {n_heads} |
| **d_ff** | {d_ff} |
| **Vocabulary size** | {vocab_size} |
| **Sequence length** | {max_seq_len} |
| **Best val loss** | {val_loss:.4f} (step {best_step:,}) |

## Usage

```python
from pawn.checkpoint import load_model

# load_model verifies the .complete SHA-256 sentinel and rebuilds
# the PAWNModel from the saved ModelConfig.
model = load_model("path/to/exported_checkpoint")
# model(tokens, attn_mask) -> [B, T, vocab_size] logits
```

A thin PyTorch loader for non-JAX consumers ships at
``pawn.torch_loader.load_pawn`` (requires the ``torch-loader`` extra).

## Training

Trained from scratch on random self-play games generated by a Rust chess engine (shakmaty).
See the [PAWN repository]({github_url}) for training code, data pipeline, and evaluation suite.

## License

Apache 2.0
"""


def main() -> None:
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

    best_step = find_best_step(metrics_path)
    if best_step is None:
        print(
            "ERROR: No val records found in metrics.jsonl — only "
            "adapter runs (train_jax_adapter.py) produce val_loss "
            "rows. Pretrain runs have no val pass; use the final "
            "checkpoint manually."
        )
        return
    print(f"Best val step: {best_step}")

    best_val_loss = float("inf")
    with open(metrics_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("step_end") == best_step and r.get("val_loss") is not None:
                best_val_loss = r["val_loss"]
                break

    ckpt_dir = run_dir / "checkpoints"
    checkpoints = sorted(d for d in ckpt_dir.glob("step_*") if d.is_dir()) if ckpt_dir.exists() else []
    if not checkpoints:
        print("ERROR: No checkpoints found")
        return

    best_ckpt = min(
        checkpoints,
        key=lambda p: abs(int(p.name.replace("step_", "")) - best_step),
    )
    print(f"Best checkpoint: {best_ckpt}")

    with open(best_ckpt / "config.json") as f:
        config = json.load(f)
    model_config = config.get("model_config", {})
    training_config = config.get("training_config", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting best checkpoint to {output_dir}/")
    copy_checkpoint(best_ckpt, output_dir)

    shutil.copy2(metrics_path, output_dir / "metrics.jsonl")

    readme = generate_readme(
        args.repo_name, model_config, training_config,
        best_step, best_val_loss,
        args.github_url, args.extra_desc,
    )
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    if not args.best_only:
        for ckpt in checkpoints:
            if ckpt == best_ckpt:
                continue
            step_name = ckpt.name  # e.g. "step_00005000"
            step_num = int(step_name.replace("step_", ""))
            step_dir = output_dir / "checkpoints" / step_name
            print(f"  Exporting {step_name}...")
            copy_checkpoint(ckpt, step_dir)

            truncated = truncate_metrics(metrics_path, step_num)
            with open(step_dir / "metrics.jsonl", "w") as f:
                f.writelines(truncated)

    print(f"\nExport complete: {output_dir}")
    print("  Best: model.safetensors, config.json, .complete, metrics.jsonl, README.md")
    if not args.best_only:
        print(f"  Checkpoints: {len(checkpoints) - 1} in checkpoints/")


if __name__ == "__main__":
    main()
