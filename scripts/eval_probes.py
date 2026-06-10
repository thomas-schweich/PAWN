#!/usr/bin/env python3
"""Backwards-compatibility wrapper for the v1 linear-probes eval.

`scripts/eval_probes_jax.py` is the v2 replacement and the canonical
single-checkpoint entry point. This wrapper preserves the two v1
invocation shapes that the v2 script does not itself implement:

* **Single checkpoint** (``--checkpoint``): forwards every argv straight
  to ``eval_probes_jax.py`` (a one-line DeprecationWarning is printed).
* **Log-dir scan** (``--log-dir`` / ``--run``): walks ``<log-dir>/*`` run
  directories, finds the latest ``step_*`` checkpoint in each, runs the v2
  probe script per checkpoint, and writes a v1-schema ``probe_results.json``
  into each run directory.

The scan mode keeps the v1 batch-over-runs workflow working post-swap; it
shells out to the v2 script per run so there is a single probe
implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _forward_single(argv: list[str]) -> int:
    """Exec the v2 script in-place, forwarding argv unchanged."""
    target = Path(__file__).parent / "eval_probes_jax.py"
    print(
        "[scripts/eval_probes.py] DeprecationWarning: forwarding to "
        "scripts/eval_probes_jax.py. Update invocations to use the v2 "
        "entry point directly.",
        file=sys.stderr,
        flush=True,
    )
    os.execvp(sys.executable, [sys.executable, str(target), *argv])
    return 0  # unreachable (execvp replaces the process)


def _find_runs(log_dir: Path, run: str | None) -> list[tuple[Path, Path]]:
    """Return ``(run_dir, latest_checkpoint)`` pairs under ``log_dir``.

    A run directory is any immediate child of ``log_dir`` that contains at
    least one ``step_*`` checkpoint directory. The latest (lexicographically
    last, i.e. highest zero-padded step) checkpoint is selected.
    """
    runs: list[tuple[Path, Path]] = []
    for run_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        if run is not None and run_dir.name != run:
            continue
        checkpoints = sorted(
            d for d in run_dir.glob("step_*") if d.is_dir()
        )
        if not checkpoints:
            continue
        runs.append((run_dir, checkpoints[-1]))
    return runs


def main(argv: list[str] | None = None) -> int:
    raw = argv if argv is not None else sys.argv[1:]

    ap = argparse.ArgumentParser(prog="eval_probes", add_help=True)
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="scan this directory for run subdirs with checkpoints")
    ap.add_argument("--run", type=str, default=None,
                    help="restrict the scan to this run dir name")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="single checkpoint — forwarded to eval_probes_jax.py")
    # Probe knobs forwarded verbatim to the v2 script in scan mode.
    ap.add_argument("--n-games", type=int, default=256)
    ap.add_argument("--n-val-games", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=None)
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-ply", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature", type=str, default="side_to_move")
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--probe-square", type=int, default=28)
    ap.add_argument("--top-layer-only", action="store_true")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--prepend-outcome", dest="prepend_outcome",
                     action="store_true", default=None)
    grp.add_argument("--pure-moves", dest="prepend_outcome",
                     action="store_false")
    args = ap.parse_args(raw)

    # Single-checkpoint mode: pure forward, no scan. (Reject the ambiguous
    # combination of --checkpoint AND --log-dir — they are different modes.)
    if args.checkpoint is not None:
        if args.log_dir is not None:
            ap.error("--checkpoint and --log-dir are mutually exclusive")
        return _forward_single(raw)

    if args.log_dir is None:
        ap.error("one of --checkpoint or --log-dir is required")

    log_dir: Path = args.log_dir
    if not log_dir.is_dir():
        ap.error(f"--log-dir {log_dir} is not a directory")

    runs = _find_runs(log_dir, args.run)
    if not runs:
        print(f"No runs with checkpoints found under {log_dir}", file=sys.stderr)
        return 1

    target = Path(__file__).parent / "eval_probes_jax.py"

    def _forwarded_flags() -> list[str]:
        flags = [
            "--n-games", str(args.n_games),
            "--n-val-games", str(args.n_val_games),
            "--n-epochs", str(args.n_epochs),
            "--val-frac", str(args.val_frac),
            "--batch-size", str(args.batch_size),
            "--max-ply", str(args.max_ply),
            "--seed", str(args.seed),
            "--feature", args.feature,
            "--probe-square", str(args.probe_square),
        ]
        if args.val_seed is not None:
            flags += ["--val-seed", str(args.val_seed)]
        if args.all_features:
            flags.append("--all-features")
        if args.top_layer_only:
            flags.append("--top-layer-only")
        if args.prepend_outcome is True:
            flags.append("--prepend-outcome")
        elif args.prepend_outcome is False:
            flags.append("--pure-moves")
        return flags

    print(f"Found {len(runs)} run(s) with checkpoints under {log_dir}")
    summary: list[dict[str, object]] = []
    for run_dir, ckpt in runs:
        out_path = run_dir / "probe_results.json"
        cmd = [
            sys.executable, str(target),
            "--checkpoint", str(ckpt),
            "--output", str(out_path),
            *_forwarded_flags(),
        ]
        print(f"\n{'=' * 60}\nRun: {run_dir.name}\nCheckpoint: {ckpt}\n{'=' * 60}")
        rc = subprocess.run(cmd, check=False).returncode
        entry: dict[str, object] = {
            "run": run_dir.name,
            "checkpoint": str(ckpt),
            "returncode": rc,
            "output": str(out_path) if out_path.exists() else None,
        }
        if out_path.exists():
            # Stamp the v1-schema run/step fields the scan owns (the per-run
            # dir name + the checkpoint step) onto the v2 probe payload.
            payload = json.loads(out_path.read_text())
            payload["run"] = run_dir.name
            step_raw = ckpt.name.replace("step_", "")
            payload["step"] = int(step_raw) if step_raw.isdigit() else step_raw
            out_path.write_text(json.dumps(payload, indent=2))
        summary.append(entry)
    print("\n" + json.dumps(summary, indent=2))
    return 0 if all(e["returncode"] == 0 for e in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
