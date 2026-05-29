#!/usr/bin/env python3
"""Move-accuracy + per-phase evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pawn.checkpoint import load_model, resolve_checkpoint_source
from pawn.corpus import generate_corpus
from pawn.eval import PhaseBoundaries, compute_per_phase_accuracy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_jax")
    ap.add_argument("--checkpoint", required=True,
                    help="v2 HF repo ID or local checkpoint dir. "
                         "(v1 PyTorch repos require `git checkout v1.0.0`.)")
    ap.add_argument("--n-games", type=int, default=512)
    ap.add_argument("--max-ply", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    model, _ = load_model(ckpt_path)
    corpus = generate_corpus(
        n_games=args.n_games, max_ply=args.max_ply, seq_len=args.seq_len, seed=0
    )
    result = compute_per_phase_accuracy(
        model, corpus, batch_size=args.batch_size,
    )
    payload = {
        "checkpoint": ckpt,
        "n_games": args.n_games,
        "overall_accuracy": result.overall,
        "opening_accuracy": result.opening,
        "midgame_accuracy": result.midgame,
        "endgame_accuracy": result.endgame,
        "n_supervised": result.n_total,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
