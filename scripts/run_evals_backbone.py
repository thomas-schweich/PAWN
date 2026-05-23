#!/usr/bin/env python3
"""Batch-eval orchestrator — runs all 4 eval scripts per checkpoint.

For each input checkpoint, invokes `eval_jax.py`,
`eval_probes_jax.py`, `eval_generation_jax.py`, and
`eval_vs_stockfish.py`. Writes `<output_dir>/<name>/eval_results.json`
per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_evals_backbone")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--outcome-prefix-trained", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for ckpt in args.checkpoints:
        name = ckpt.replace("/", "_")
        cdir = args.output_dir / name
        cdir.mkdir(parents=True, exist_ok=True)
        results: dict[str, object] = {"checkpoint": ckpt}
        acc_path = cdir / "accuracy.json"
        subprocess.run(
            [sys.executable, "scripts/eval_jax.py",
             "--checkpoint", ckpt, "--output", str(acc_path)],
            check=False,
        )
        results["accuracy"] = (
            json.loads(acc_path.read_text()) if acc_path.exists() else None
        )
        prob_path = cdir / "probes.json"
        subprocess.run(
            [sys.executable, "scripts/eval_probes_jax.py",
             "--checkpoint", ckpt, "--output", str(prob_path)],
            check=False,
        )
        results["probes"] = (
            json.loads(prob_path.read_text()) if prob_path.exists() else None
        )
        gen_path = cdir / "generation.json"
        gen_args = [sys.executable, "scripts/eval_generation_jax.py",
                    "--checkpoint", ckpt, "--output", str(gen_path)]
        gen_args.append(
            "--outcome-prefix-trained"
            if args.outcome_prefix_trained
            else "--no-outcome-prefix-trained"
        )
        subprocess.run(gen_args, check=False)
        results["generation"] = (
            json.loads(gen_path.read_text()) if gen_path.exists() else None
        )
        lich_path = cdir / "lichess.json"
        subprocess.run(
            [sys.executable, "scripts/eval_vs_stockfish.py",
             "--checkpoint", ckpt, "--output", str(lich_path)],
            check=False,
        )
        results["lichess"] = (
            json.loads(lich_path.read_text()) if lich_path.exists() else None
        )
        (cdir / "eval_results.json").write_text(json.dumps(results, indent=2))
        summary.append({"checkpoint": ckpt, "output": str(cdir)})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
