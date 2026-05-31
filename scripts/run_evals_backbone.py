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


def _is_real_probe_payload(payload: object) -> bool:
    """True when a probes.json carries genuine per-layer held-out probe
    output (H6) — at least one feature's ``layers`` map has an entry that
    reports a held-out split (``n_val > 0``). Guards against emitting a
    placeholder or partial probe result into the aggregated eval bundle.

    The v2 probe payload nests per-layer metrics under
    ``probes[feature]["layers"]`` (a suite of features), so we scan every
    feature's layer map.
    """
    if not isinstance(payload, dict):
        return False
    probes = payload.get("probes")
    if not isinstance(probes, dict) or not probes:
        return False
    for feat in probes.values():
        if not isinstance(feat, dict):
            continue
        layers = feat.get("layers")
        if not isinstance(layers, dict):
            continue
        if any(
            isinstance(v, dict)
            and isinstance(v.get("n_val"), int)
            and v["n_val"] > 0
            for v in layers.values()
        ):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_evals_backbone")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--outcome-prefix-trained", action="store_true")
    # Edge-case diagnostics (engine quota-controlled coverage of all 10
    # labels). Off by default so the cheap generation suite stays fast;
    # the model card's `diagnostics` table is only populated when this is
    # passed. `--edge-per-label` controls per-(colour, label) coverage.
    ap.add_argument("--edge-cases", action="store_true")
    ap.add_argument("--edge-per-label", type=int, default=10)
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
        # Probes (H6): the real per-layer, held-out probe path. We only
        # surface `results["probes"]` when the probe script produced an
        # output carrying genuine per-layer held-out accuracies — i.e. it
        # ran the frozen-forward + engine-label pipeline. A missing/partial
        # file (subprocess failure) leaves the key absent rather than
        # emitting placeholder noise.
        prob_path = cdir / "probes.json"
        # Probe the full feature suite (side_to_move, occupancy, piece_type,
        # is_check, castling_rights, ep_square, game_phase + the MSE
        # regression probes), not just side_to_move — v1 parity.
        subprocess.run(
            [sys.executable, "scripts/eval_probes_jax.py",
             "--checkpoint", ckpt, "--all-features", "--output", str(prob_path)],
            check=False,
        )
        if prob_path.exists():
            probe_payload = json.loads(prob_path.read_text())
            if _is_real_probe_payload(probe_payload):
                results["probes"] = probe_payload
        gen_path = cdir / "generation.json"
        gen_args = [sys.executable, "scripts/eval_generation_jax.py",
                    "--checkpoint", ckpt, "--output", str(gen_path)]
        gen_args.append(
            "--outcome-prefix-trained"
            if args.outcome_prefix_trained
            else "--no-outcome-prefix-trained"
        )
        if args.edge_cases:
            gen_args += ["--edge-cases",
                         "--edge-per-label", str(args.edge_per_label)]
        subprocess.run(gen_args, check=False)
        generation = (
            json.loads(gen_path.read_text()) if gen_path.exists() else None
        )
        results["generation"] = generation
        # Surface the edge-case diagnostics under the top-level
        # `diagnostics` key the model-card consumer
        # (`generate_model_cards.format_diagnostic`) reads. Edge data lives
        # in `generation["edge_cases"]` (per-label sampled metrics +
        # accuracy); only present when --edge-cases ran and the script
        # emitted it.
        if isinstance(generation, dict):
            edge = generation.get("edge_cases")
            if isinstance(edge, dict):
                results["diagnostics"] = edge
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
