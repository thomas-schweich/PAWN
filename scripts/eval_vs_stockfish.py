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

from pawn.checkpoint import load_eval_model, resolve_checkpoint_source
from pawn.corpus import Corpus, conditioning_from_run_block
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
    ap.add_argument("--min-eval-ply", type=int, default=10,
                    help="MAIA opening-skip for the per-bin headline "
                         "metrics (default 10).")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    # load_eval_model re-applies adapter sidecars so an adapter checkpoint
    # is scored as the ADAPTED model across every Elo bin.
    model, run_block = load_eval_model(ckpt_path)
    # Build every Elo-bin corpus with the checkpoint's own conditioning so
    # the move positions match the layout the model was trained under
    # (plan §8.1) — a hardcoded default would shift the absolute RoPE
    # offset and corrupt the per-move accuracy.
    conditioning = conditioning_from_run_block(run_block)

    bins = default_elo_bins()
    bins_corpora: dict[EloBin, Corpus] = {}
    for b in bins:
        try:
            c = load_lichess_corpus(
                args.pgn, split=args.split,
                elo_min=b.lo, elo_max=b.hi,
                seq_len=args.seq_len, max_games=args.max_games_per_bin,
                conditioning=conditioning,
            )
        except (ValueError, FileNotFoundError):
            continue
        bins_corpora[b] = c
    results = compute_elo_stratified_accuracy(
        model, bins_corpora, min_eval_ply=args.min_eval_ply,
    )
    payload = {
        "checkpoint": ckpt,
        "results": [
            {
                "elo_bin": r.bin.label,
                "n_games": r.n_games,
                "loss": r.loss,
                "perplexity": r.perplexity,
                "top1_accuracy": r.accuracy,
                "top5_accuracy": r.top5,
                "legal_move_rate": r.legal_move_rate,
            }
            for r in results
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
