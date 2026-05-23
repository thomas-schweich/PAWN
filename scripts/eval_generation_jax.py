#!/usr/bin/env python3
"""5 generation diagnostics — all gated on outcome_prefix_trained."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pawn.checkpoint import load_model
from pawn.generation import run_all_diagnostics
from pawn.legacy import convert_legacy_checkpoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_generation_jax")
    ap.add_argument("--checkpoint", required=True)
    gate = ap.add_mutually_exclusive_group(required=True)
    gate.add_argument("--outcome-prefix-trained", dest="trained", action="store_true")
    gate.add_argument(
        "--no-outcome-prefix-trained", dest="trained", action="store_false"
    )
    ap.add_argument("--edge-cases", action="store_true",
                    help="also run edge-case diagnostics via engine.edge_case_bits")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = args.checkpoint
    if "/" in ckpt and not Path(ckpt).exists():
        ckpt_path = convert_legacy_checkpoint(ckpt)
    else:
        ckpt_path = Path(ckpt)
    model = load_model(ckpt_path)

    results = run_all_diagnostics(model, outcome_prefix_trained=args.trained)

    if args.edge_cases:
        import chess_engine as engine
        from pawn.eval_suite.diagnostics import compute_edge_case_accuracy
        moves, lens, _ = engine.generate_random_games(64, 64, 42)
        edge = compute_edge_case_accuracy(model, moves, lens)
        results["edge_cases"] = {
            r.label: {"accuracy": r.accuracy, "n_positions": r.n_positions}
            for r in edge
        }

    print(json.dumps(results, indent=2, default=str))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
