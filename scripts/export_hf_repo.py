#!/usr/bin/env python3
"""Package a finished local training run into HuggingFace-repo layout.

The v2 trainer pushes to HF *during* training via ``--hf-repo``; this tool
is the post-hoc packager for runs done with ``--local-checkpoints`` (no HF
push wired up) that you later decide to publish. It mirrors the v1
``export_hf_repo.py`` workflow, ported onto the v2 safetensors checkpoint
format:

  - Root: the checkpoint *nearest* the best-val step (``model.safetensors`` /
    ``config.json`` / optional ``optimizer.safetensors`` /
    ``training_state.json``), a truncated ``metrics.jsonl``, and a
    rendered ``README.md`` model card. (Validation runs on a different
    cadence than checkpointing, so the best-val step rarely has an exact
    checkpoint — nearest-step selection mirrors v1.)
  - ``checkpoints/<prefix>step_NNNNNNNN/``: every *other* checkpoint, each
    with a metrics log truncated to its own step. ``<prefix>`` is "",
    ``adapter_`` or ``distill_`` depending on which trainer produced the run.

Every copied checkpoint gets a freshly-recomputed ``.complete`` SHA-256
sentinel (the source sentinel's hashes are keyed to the source dir's file
order, so it's dropped and rewritten) — so the exported tree is loadable
via every ``pawn.checkpoint`` load path, all of which verify the sentinel.

Usage::

    python scripts/export_hf_repo.py \\
        --run-dir logs/pretrain_20260322_182707_abc123 \\
        --output-dir export/pawn-base-v2 \\
        --repo-name pawn-base-v2 \\
        --github-url https://github.com/thomas-schweich/PAWN

For new runs, prefer pushing to HF during training:

    python scripts/train_jax.py        --hf-repo USER/repo ...
    python scripts/train_jax_adapter.py --hf-repo USER/repo ...
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# A v2 checkpoint directory is ``<prefix>step_NNNNNNNN`` where ``<prefix>`` is
# one of "" (pretrain, train_jax.py), "adapter_" (train_jax_adapter.py), or
# "distill_" (train_jax_distill.py). We discover the actual name on disk rather
# than hardcoding the pretrain spelling so adapter/distill local runs package.
_CHECKPOINT_RE = re.compile(r"^(?:[A-Za-z]+_)?step_(\d+)$")

from pawn._sentinel import SENTINEL_NAME, write_sentinel
from pawn.checkpoint import (
    CONFIG_FILE,
    MODEL_FILE,
    OPTIMIZER_FILE,
    TRAINING_STATE_FILE,
)
from pawn.lifecycle import truncate_metrics_jsonl

# Files inside a v2 checkpoint dir that, when present, are part of the
# integrity-checked payload (passed to ``write_sentinel``). The optional
# files are only listed when they actually exist in the source dir.
_REQUIRED_PAYLOAD: tuple[str, ...] = (MODEL_FILE, CONFIG_FILE)
_OPTIONAL_PAYLOAD: tuple[str, ...] = (OPTIMIZER_FILE, TRAINING_STATE_FILE)


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(step, checkpoint_dir)`` pairs found directly under ``run_dir``.

    Matches any of the three v2 checkpoint spellings —
    ``step_NNNNNNNN`` (pretrain), ``adapter_step_NNNNNNNN`` (adapter), and
    ``distill_step_NNNNNNNN`` (distill) — and parses the trailing numeric step
    out of the directory name. Sorted by step so the caller can rely on order.
    """
    found: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if match is None:
            continue
        found.append((int(match.group(1)), child))
    return sorted(found, key=lambda pair: pair[0])


def find_best_step(metrics_path: Path) -> int | None:
    """Find the step with the lowest ``val/loss`` from ``metrics.jsonl``.

    Reads the v2 metric schema (``type == "val"`` records carry the
    namespaced ``val/loss`` key; see
    :meth:`pawn.logging.MetricsLogger.log_val`). Returns ``None`` when no
    val record carries both a finite loss and a step.
    """
    best_loss = float("inf")
    best_step: int | None = None
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                # A partially-flushed write from a crashed run leaves a
                # truncated last line; skip it rather than aborting the
                # whole export (parity with truncate_metrics_jsonl).
                continue
            if record.get("type") != "val":
                continue
            loss = record.get("val/loss")
            step = record.get("step")
            if loss is None or step is None:
                continue
            # ``record`` is ``dict[str, Any]`` (json.loads); cast through the
            # numeric types the annotation promises so no ``Any`` leaks into
            # ``best_step`` / the nearest-checkpoint ``abs(...)`` arithmetic.
            loss = float(loss)
            if loss < best_loss:
                best_loss = loss
                best_step = int(step)
    return best_step


def best_val_metrics(metrics_path: Path, step: int) -> tuple[float, float]:
    """Return ``(val_loss, val_accuracy)`` for the val record at ``step``.

    ``val/accuracy`` falls back to ``val/top1`` (the same scalar under the
    pre-promotion spelling) and finally to ``0.0`` when neither is present.
    """
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            rec_step = record.get("step")
            if record.get("type") != "val" or rec_step is None:
                continue
            if int(rec_step) != step:
                continue
            loss = float(record.get("val/loss", float("inf")))
            acc = record.get("val/accuracy", record.get("val/top1", 0.0))
            return loss, float(acc if acc is not None else 0.0)
    return float("inf"), 0.0


def truncate_metrics(metrics_path: Path, up_to_step: int) -> str:
    """Return ``metrics.jsonl`` content up to and including ``up_to_step``.

    Thin alias for :func:`pawn.lifecycle.truncate_metrics_jsonl` (the
    battle-tested implementation the trainer uses to keep the co-uploaded
    log from running ahead of the checkpoint it sits beside). Delegating
    keeps a single truncation implementation, so this packager inherits its
    JSONDecodeError guard — a partially-flushed last line from a crashed
    run passes through rather than aborting the export.
    """
    return truncate_metrics_jsonl(metrics_path, up_to_step)


def copy_checkpoint(src: Path, dst: Path) -> None:
    """Copy a v2 directory-format checkpoint into the export layout.

    The source ``.complete`` sentinel is dropped (its hashes are keyed to
    the source dir's file order) and a fresh sentinel is written for
    ``dst`` over exactly the payload files present, so the exported
    checkpoint passes ``pawn.checkpoint`` sentinel verification on load.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == SENTINEL_NAME:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    payload = list(_REQUIRED_PAYLOAD) + [
        name for name in _OPTIONAL_PAYLOAD if (dst / name).is_file()
    ]
    write_sentinel(dst, payload)


def generate_readme(
    repo_name: str,
    model_block: dict[str, Any],
    run_block: dict[str, Any],
    best_step: int,
    val_loss: float,
    val_acc: float,
    github_url: str,
    extra_desc: str = "",
) -> str:
    """Render a HuggingFace model card from the checkpoint's saved config.

    Architecture fields come from ``config.json``'s ``model`` block (the
    v2 layout written by :func:`pawn.checkpoint.save_model`), so the card
    stays correct regardless of whether the checkpoint was a named variant
    slice or a custom arch.
    """
    d_model = model_block.get("d_model", "?")
    n_layers = model_block.get("n_layers", "?")
    n_heads = model_block.get("n_heads", "?")
    d_ff = model_block.get("d_ff", "?")
    vocab_size = model_block.get("vocab_size", "?")
    max_seq_len = model_block.get("max_seq_len", "?")
    tie_embeddings = model_block.get("tie_embeddings", "?")
    conditioning = run_block.get("conditioning", [])

    return f"""---
license: apache-2.0
library_name: safetensors
tags:
  - chess
  - transformer
  - causal-lm
  - world-model
  - jax
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
| **Architecture** | Decoder-only transformer (RMSNorm, SwiGLU, RoPE) |
| **d_model** | {d_model} |
| **Layers** | {n_layers} |
| **Heads** | {n_heads} |
| **d_ff** | {d_ff} |
| **Vocabulary size** | {vocab_size} |
| **Sequence length** | {max_seq_len} |
| **Tied embeddings** | {tie_embeddings} |
| **Conditioning** | {conditioning or "none"} |
| **Best val loss** | {val_loss:.4f} (step {best_step:,}) |
| **Best val accuracy** | {val_acc:.1%} |

## Usage

```python
from pawn.checkpoint import load_model, resolve_checkpoint_source

# Reconstructs the exact architecture this checkpoint was trained with
# from config.json's `model` block and verifies the .complete sentinel.
# `model.cfg` is the ModelConfig; `run_block` is the saved run config.
model, run_block = load_model(resolve_checkpoint_source("{repo_name}"))
```

## Training

Trained from scratch on random self-play games generated by a Rust chess engine (shakmaty),
using the v2 JAX / Equinox / Optax stack.
See the [PAWN repository]({github_url}) for training code, data pipeline, and evaluation suite.

## License

Apache 2.0
"""


def _read_config_blocks(
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the ``(model, run)`` blocks from a checkpoint's ``config.json``.

    The ``run`` block is optional in the v2 layout (a bare ``save_model``
    without a run config omits it); default it to ``{}`` so the README
    renderer never KeyErrors.
    """
    payload = json.loads(
        (checkpoint_dir / CONFIG_FILE).read_text(encoding="utf-8")
    )
    model_block: dict[str, Any] = payload.get("model", {})
    run_block: dict[str, Any] = payload.get("run", {})
    return model_block, run_block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package a finished v2 training run into HF-repo layout"
    )
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for the HF repo"
    )
    parser.add_argument(
        "--repo-name", required=True, help="Repository name for the README"
    )
    parser.add_argument(
        "--github-url", default="https://github.com/thomas-schweich/PAWN"
    )
    parser.add_argument(
        "--best-only", action="store_true", help="Only export the best checkpoint"
    )
    parser.add_argument(
        "--extra-desc", default="", help="Extra description for the README"
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    metrics_path = run_dir / "metrics.jsonl"

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found", file=sys.stderr)
        return 1

    best_val_step = find_best_step(metrics_path)
    if best_val_step is None:
        print(
            "ERROR: no val records with a val/loss + step found in "
            "metrics.jsonl",
            file=sys.stderr,
        )
        return 1
    print(f"Best val step: {best_val_step}")

    # The trainer saves checkpoints on the `checkpoint_interval` cadence, but
    # validation runs on the independent `val_every` cadence — so the best-val
    # step usually has no exactly-matching checkpoint dir. Pick the checkpoint
    # whose step is *nearest* to the best-val step (v1 parity:
    # `min(..., key=lambda p: abs(step - best_step))`), across whichever
    # checkpoint prefix the run used (pretrain / adapter / distill).
    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        print(
            f"ERROR: no checkpoint dirs (step_*/adapter_step_*/distill_step_*) "
            f"found under {run_dir}",
            file=sys.stderr,
        )
        return 1
    best_step, best_dir = min(
        checkpoints, key=lambda pair: abs(pair[0] - best_val_step)
    )
    if best_step != best_val_step:
        print(
            f"Best val step {best_val_step} has no exact checkpoint; using "
            f"nearest checkpoint at step {best_step} ({best_dir.name})"
        )

    val_loss, val_acc = best_val_metrics(metrics_path, best_val_step)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Root = the best checkpoint, plus a truncated metrics log + README. The
    # log is truncated to whichever is later of the exported checkpoint step
    # and the best-val step, so the val record the README cites is never cut
    # when the nearest checkpoint precedes the best-val step.
    copy_checkpoint(best_dir, output_dir)
    (output_dir / "metrics.jsonl").write_text(
        truncate_metrics(metrics_path, max(best_step, best_val_step)),
        encoding="utf-8",
    )
    model_block, run_block = _read_config_blocks(best_dir)
    (output_dir / "README.md").write_text(
        generate_readme(
            args.repo_name,
            model_block,
            run_block,
            best_val_step,
            val_loss,
            val_acc,
            args.github_url,
            args.extra_desc,
        ),
        encoding="utf-8",
    )
    print(f"Exported best checkpoint (step {best_step}) -> {output_dir}")

    if not args.best_only:
        other_steps = [
            (step, ckpt)
            for step, ckpt in discover_checkpoints(run_dir)
            if step != best_step
        ]
        for step, ckpt in other_steps:
            dst = output_dir / "checkpoints" / ckpt.name
            copy_checkpoint(ckpt, dst)
            (dst / "metrics.jsonl").write_text(
                truncate_metrics(metrics_path, step), encoding="utf-8"
            )
            print(f"Exported checkpoint (step {step}) -> {dst}")

    print(f"\nDone. HF-repo layout written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
