"""JAX linear-probe evaluator.

Trains the 9 linear probes from ``pawn.probes.PROBES`` on a JAX checkpoint's
hidden states across the embedding output plus every transformer layer, and
writes per-layer accuracy / loss / (R² + MAE for regression probes) to JSON.

Usage:

    uv run python scripts/eval_probes_jax.py \\
        --checkpoint ~/.cache/huggingface/pawn-jax-converted/pawn-small \\
        --n-games 4096 --n-val-games 1024 --n-epochs 20

    # Fresh-init verification on the TINY supernet
    uv run python scripts/eval_probes_jax.py \\
        --supernet tiny --variant small --n-games 256 --n-val-games 128 \\
        --n-epochs 2

Output: ``<logs_dir>/probe_results.json`` (or
``<checkpoint_parent>/probe_results.json`` when ``--checkpoint`` is supplied
without ``--logs-dir``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import jax

from pawn.checkpoint import load_model
from pawn.config import (
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
    ModelConfig,
    validate_nested,
)
from pawn.model import PAWNModel, init_model, sliced
from pawn.probes import PROBES, extract_probe_data, train_probes


def _resolve_model_from_supernet(
    supernet_name: str, variant: str, key: jax.Array
) -> PAWNModel:
    if supernet_name == "tiny":
        supernet_cfg = TINY_SUPERNET
        variants = TINY_VARIANTS
    else:
        supernet_cfg = SUPERNET
        variants = VARIANTS
    if variant != "supernet" and variant not in variants:
        raise SystemExit(
            f"unknown variant '{variant}'; choose from "
            f"{sorted(variants.keys())} or 'supernet'."
        )
    target_cfg: ModelConfig = (
        supernet_cfg if variant == "supernet" else variants[variant]
    )
    # ``validate_nested(variant, supernet)`` — the supernet sits on the
    # right; ``target_cfg`` may equal ``supernet_cfg`` when variant ==
    # 'supernet', in which case the check trivially passes.
    validate_nested(target_cfg, supernet_cfg)
    supernet = init_model(supernet_cfg, key)
    return supernet if variant == "supernet" else sliced(supernet, target_cfg)


def _make_run_dir(
    base: Path, checkpoint: Path | None, logs_dir: Path | None
) -> Path:
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    if checkpoint is not None:
        out = checkpoint.parent
        out.mkdir(parents=True, exist_ok=True)
        return out
    out = base / f"jax_probes_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="path to a JAX checkpoint directory; omit to use a freshly-"
             "initialised model from --supernet / --variant.",
    )
    parser.add_argument(
        "--supernet", choices=("tiny", "supernet"), default="tiny",
        help="only used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--variant", default="small",
        help="variant slug or 'supernet'. Only used without --checkpoint.",
    )
    parser.add_argument(
        "--n-games", type=int, default=4096,
        help="probe-train corpus size (games).",
    )
    parser.add_argument(
        "--n-val-games", type=int, default=1024,
        help="probe-val corpus size (games).",
    )
    parser.add_argument(
        "--max-ply", type=int, default=None,
        help="CLM sequence length for the probe corpus (defaults to the "
             "model's max_seq_len, capped at 256 for runtime).",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=20,
        help="passes over the train corpus.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="probe AdamW peak LR.",
    )
    parser.add_argument(
        "--game-batch-size", type=int, default=64,
        help="games per model forward pass.",
    )
    parser.add_argument(
        "--inner-batch-size", type=int, default=256,
        help="per-position SGD mini-batch size.",
    )
    parser.add_argument(
        "--prepend-outcome", action="store_true",
        help="prepend the outcome token at CLM position 0 (matches "
             "checkpoints trained with the outcome-prefix layout).",
    )
    parser.add_argument(
        "--probe", action="append", default=None,
        help=f"restrict to a specific probe (repeat); choices: {list(PROBES.keys())}",
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="seed used for both probe init and corpus generation. Val "
             "corpus is generated with seed+1.",
    )
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is not None and not args.checkpoint.is_dir():
        raise SystemExit(f"--checkpoint {args.checkpoint} is not a directory")
    if args.probe:
        unknown = [p for p in args.probe if p not in PROBES]
        if unknown:
            raise SystemExit(f"unknown probe(s): {unknown}")

    init_key = jax.random.key(args.seed)
    if args.checkpoint is not None:
        if not args.quiet:
            print(f"[probes] loading checkpoint {args.checkpoint}")
        model = load_model(args.checkpoint)
    else:
        model = _resolve_model_from_supernet(
            args.supernet, args.variant, init_key,
        )

    if args.max_ply is not None:
        max_ply = int(args.max_ply)
    else:
        # Cap by the model's max_seq_len, but also by 256 so the default
        # n_games × max_ply doesn't blow up on TINY for small probe sweeps.
        max_ply = min(model.cfg.max_seq_len, 256)
    if max_ply <= 0:
        raise SystemExit(f"--max-ply must be > 0, got {max_ply}")
    if max_ply > model.cfg.max_seq_len:
        raise SystemExit(
            f"--max-ply={max_ply} exceeds model.cfg.max_seq_len="
            f"{model.cfg.max_seq_len}"
        )

    if not args.quiet:
        print(
            f"[probes] cfg d={model.cfg.d_model} L={model.cfg.n_layers} "
            f"H={model.cfg.n_heads}; max_ply={max_ply}; "
            f"prepend_outcome={args.prepend_outcome}"
        )
        print(
            f"[probes] generating {args.n_games} train + "
            f"{args.n_val_games} val games"
        )
    # Legal-move counting is the bulk of the engine work in
    # ``extract_probe_data``. Skip it unless the active probe set actually
    # consumes ``legal_move_count``. (Codex P2)
    active_probes = args.probe if args.probe else list(PROBES.keys())
    include_legal = "legal_move_count" in active_probes
    train = extract_probe_data(
        n_games=args.n_games, max_ply=max_ply, seed=args.seed,
        prepend_outcome=args.prepend_outcome,
        include_legal_counts=include_legal,
    )
    val = extract_probe_data(
        n_games=args.n_val_games, max_ply=max_ply, seed=args.seed + 1,
        prepend_outcome=args.prepend_outcome,
        include_legal_counts=include_legal,
    )

    results = train_probes(
        model, train, val,
        n_epochs=args.n_epochs, lr=args.lr,
        game_batch_size=args.game_batch_size,
        inner_batch_size=args.inner_batch_size,
        probe_names=args.probe,
        key=jax.random.key(args.seed + 7),
        verbose=not args.quiet,
    )

    out_dir = _make_run_dir(Path("logs"), args.checkpoint, args.logs_dir)
    out_path = out_dir / "probe_results.json"
    output = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "supernet": args.supernet if args.checkpoint is None else None,
        "variant": args.variant if args.checkpoint is None else None,
        "model_config": {
            "d_model": model.cfg.d_model,
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "d_ff": model.cfg.d_ff,
            "max_seq_len": model.cfg.max_seq_len,
        },
        "n_games": args.n_games,
        "n_val_games": args.n_val_games,
        "max_ply": max_ply,
        "n_epochs": args.n_epochs,
        "prepend_outcome": args.prepend_outcome,
        "probes": {
            name: {
                lname: {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in metrics.items()}
                for lname, metrics in layer_results.items()
            }
            for name, layer_results in results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    if not args.quiet:
        print(f"[probes] wrote {out_path}")
        for name in results:
            best = max(
                results[name][lname]["best_accuracy"]
                for lname in results[name]
            )
            best_layer = max(
                results[name],
                key=lambda lname: results[name][lname]["best_accuracy"],
            )
            print(f"  {name:>20s}: best={best:.4f} @ {best_layer}")


if __name__ == "__main__":
    main()
