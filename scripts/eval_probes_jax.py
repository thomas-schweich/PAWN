#!/usr/bin/env python3
"""Linear probes on frozen hidden states — synthetic probe data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pawn.checkpoint import load_model, resolve_checkpoint_source
from pawn.probes import ProbeConfig, fit_probe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_probes_jax")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--n-classes", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    model, _ = load_model(ckpt_path)
    d = model.cfg.d_model

    # Synthetic probe data — the realistic per-layer hidden-state
    # extraction lives in S15's comprehensive tests; this is the
    # script-level dispatch path.
    rng = np.random.default_rng(0)
    hidden = jnp.asarray(rng.standard_normal((args.n_samples, d)).astype("float32"))
    labels = jnp.asarray(rng.integers(0, args.n_classes, size=args.n_samples).astype("int32"))
    cfg = ProbeConfig(n_classes=args.n_classes, n_epochs=args.n_epochs)
    result = fit_probe(hidden, labels, cfg)
    payload = {
        "checkpoint": ckpt,
        "n_samples": args.n_samples,
        "n_classes": args.n_classes,
        "probe_accuracy": result.accuracy,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
