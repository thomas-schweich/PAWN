#!/usr/bin/env python3
"""Linear probes on FROZEN per-layer hidden states (H6).

Forwards the frozen backbone on engine games, extracts each layer's
residual-stream hidden state, and fits a held-out linear probe per layer
against an engine-derived board feature
(:func:`chess_engine.extract_board_states`). Emits per-layer held-out
probe accuracy — real signal, not synthetic noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess_engine as engine

from pawn.checkpoint import load_model, resolve_checkpoint_source
from pawn.corpus import conditioning_from_run_block
from pawn.probes import (
    occupancy_labeler,
    piece_type_labeler,
    run_layer_probes,
    side_to_move_labeler,
)

# Probe-feature registry: name -> (labeler_factory, n_classes). A factory
# takes the optional `--probe-square` and returns the BoardLabeler.
_FEATURES = {
    "side_to_move": (lambda sq: side_to_move_labeler, 2),
    "occupancy": (lambda sq: occupancy_labeler(sq), 2),
    "piece_type": (lambda sq: piece_type_labeler(sq), 7),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_probes_jax")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-games", type=int, default=256)
    ap.add_argument("--max-ply", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--feature", choices=sorted(_FEATURES), default="side_to_move",
        help="board feature to probe for (default: side_to_move)",
    )
    ap.add_argument(
        "--probe-square", type=int, default=28,
        help="board square 0..63 (rank-major) for occupancy/piece_type",
    )
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    model, run_block = load_model(ckpt_path)
    # Rebuild the checkpoint's own conditioning layout so probe states use
    # the same absolute-RoPE prefix the model trained under (plan §8.1).
    conditioning = tuple(conditioning_from_run_block(run_block))

    labeler_factory, n_classes = _FEATURES[args.feature]
    labeler = labeler_factory(args.probe_square)

    # Engine self-play games — the frozen forward + engine labels run on
    # these. The probe never trains the backbone.
    move_ids, game_lengths, _term = engine.generate_random_games(
        args.n_games, args.max_ply, args.seed
    )

    per_layer = run_layer_probes(
        model, move_ids, game_lengths,
        n_classes=n_classes, labeler=labeler,
        conditioning=conditioning,
        n_epochs=args.n_epochs, val_frac=args.val_frac,
        batch_size=args.batch_size, key=args.seed,
    )

    layers_payload = {
        str(layer): {
            "val_accuracy": r.accuracy,
            "train_accuracy": r.train_accuracy,
            "n_train": r.n_train,
            "n_val": r.n_val,
        }
        for layer, r in sorted(per_layer.items())
    }
    best_layer, best = max(per_layer.items(), key=lambda kv: kv[1].accuracy)
    payload = {
        "checkpoint": ckpt,
        "feature": args.feature,
        "probe_square": args.probe_square if args.feature != "side_to_move" else None,
        "n_classes": n_classes,
        "n_games": args.n_games,
        "layers": layers_payload,
        "best_layer": best_layer,
        "best_val_accuracy": best.accuracy,
        # Back-compat headline field consumed by run_evals_backbone /
        # dashboards: the best held-out per-layer probe accuracy.
        "probe_accuracy": best.accuracy,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
