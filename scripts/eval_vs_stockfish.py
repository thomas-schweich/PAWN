#!/usr/bin/env python3
"""Elo-stratified Maia-style accuracy on held-out Lichess games.

Replaces the v1 stockfish-playoff harness (which is gone — the v2 plan
§3 criterion 13 just needs Maia-style per-Elo-bin accuracy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pawn.checkpoint import load_model, resolve_checkpoint_source
from pawn.corpus import Corpus
from pawn.lichess_data import load_lichess_corpus
from pawn.lichess_eval import (
    EloBin,
    compute_elo_stratified_accuracy,
    default_elo_bins,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_vs_stockfish")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pgn", default="thomas-schweich/pawn-lichess-full")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-games-per-bin", type=int, default=100)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    model = load_model(ckpt_path)

    bins = default_elo_bins()
    bins_corpora: dict[EloBin, Corpus] = {}
    for b in bins:
        try:
            c = load_lichess_corpus(
                args.pgn, split=args.split,
                elo_min=b.lo, elo_max=b.hi,
                seq_len=args.seq_len, max_games=args.max_games_per_bin,
            )
        except (ValueError, FileNotFoundError):
            continue
        bins_corpora[b] = c
    results = compute_elo_stratified_accuracy(model, bins_corpora)
    payload = {
        "checkpoint": ckpt,
        "results": [
            {"elo_bin": r.bin.label, "accuracy": r.accuracy, "n_games": r.n_games}
            for r in results
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
