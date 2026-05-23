#!/usr/bin/env python3
"""Convert v1 published HF checkpoints to v2 JAX format.

Walks `pawn-{small, base, large}` (HF repos) and converts each via
`pawn.legacy.convert_legacy_checkpoint`. Reports the cached output
path. The full parity check (mean Δlogit ≤ 1e-3, max ≤ 1e-4 vs v1)
runs in S16's final-smoke verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pawn.checkpoint import load_model
from pawn.legacy import convert_legacy_checkpoint

DEFAULT_REPOS = (
    "thomas-schweich/pawn-small",
    "thomas-schweich/pawn-base",
    "thomas-schweich/pawn-large",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="convert_published_checkpoints")
    ap.add_argument(
        "--repos", nargs="*", default=list(DEFAULT_REPOS),
        help="HF repo IDs to convert",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    results = []
    for repo in args.repos:
        try:
            out = convert_legacy_checkpoint(repo, force=args.force)
            model = load_model(out)
            results.append({
                "repo": repo,
                "output": str(out),
                "d_model": model.cfg.d_model,
                "n_layers": model.cfg.n_layers,
                "n_heads": model.cfg.n_heads,
                "head_dim": model.cfg.head_dim,
                "status": "ok",
            })
        except Exception as e:
            results.append({"repo": repo, "status": "error", "error": str(e)})
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
