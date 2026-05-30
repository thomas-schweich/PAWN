#!/usr/bin/env python3
"""5 generation diagnostics — all gated on outcome_prefix_trained.

The diagnostics run real autoregressive generation (see
:mod:`pawn.generation`) and aggregate the v1-parity metrics dict
(outcome match rate, forfeit rate, mean game length, post-terminal
padding ratio).

``--edge-cases`` additionally runs the engine-quota-controlled
edge-case accuracy via
:func:`pawn.eval_suite.diagnostics.compute_edge_case_accuracy_quota`,
which guarantees coverage for every label in
:data:`pawn.eval_suite.diagnostics.EDGE_CASE_LABELS` (in_check,
double_check, pin_restricts, ep_available, castle_legal_*,
castle_blocked_check, promotion_available, checkmate, stalemate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pawn.checkpoint import load_model, resolve_checkpoint_source
from pawn.corpus import conditioning_from_run_block
from pawn.generation import run_all_diagnostics
from pawn.jax_setup import setup_jax_caching


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_generation_jax")
    ap.add_argument("--checkpoint", required=True)
    # The gate defaults to auto-detection from the checkpoint's persisted
    # conditioning (plan §8.1: eval reads the checkpoint's own conditioning)
    # — a checkpoint trained with conditioning=["outcome"] runs the
    # diagnostics; one without it reports `_skipped`. The explicit flags
    # override the auto-detection (e.g. to force the skip path for a parity
    # check, or to run diagnostics on a checkpoint whose run block is
    # absent).
    gate = ap.add_mutually_exclusive_group(required=False)
    gate.add_argument(
        "--outcome-prefix-trained", dest="trained",
        action="store_true", default=None,
    )
    gate.add_argument(
        "--no-outcome-prefix-trained", dest="trained", action="store_false",
    )
    ap.add_argument(
        "--edge-cases", action="store_true",
        help="also run edge-case diagnostics via "
        "engine.generate_diagnostic_sets (quota-controlled coverage of "
        "all 10 labels)",
    )
    ap.add_argument(
        "--edge-per-label", type=int, default=10,
        help="per-(colour, label) game count for quota-controlled edge "
        "case sampling (default 10; raise for rarer labels)",
    )
    ap.add_argument(
        "--gen-n-per-outcome", type=int, default=16,
        help="games per outcome in `outcome_signal_test` (v1's default "
        "was 1000; we default to 16 because the JAX path doesn't yet "
        "use a KV-cached decoder — raise as throughput allows)",
    )
    ap.add_argument(
        "--gen-max-seq-len", type=int, default=32,
        help="autoregressive decode horizon for the generation suite",
    )
    ap.add_argument(
        "--cache-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="dtype for the KV cache (bf16 halves the cache footprint "
        "at production-scale n_per_outcome — paired with --compute-dtype)",
    )
    ap.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="forward-pass compute dtype (must match --cache-dtype "
        "precision when --cache-dtype is bf16/fp16)",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    # Enable the persistent compilation cache before any decode jit compiles
    # (the KV-cached generation path shares HLO across diagnostic runs).
    cache_path = setup_jax_caching()
    if cache_path is not None:
        print(f"JAX compilation cache: {cache_path}")

    import jax.numpy as jnp
    _DTYPE_MAP = {
        "float32": None,  # init_kv_cache treats None as fp32
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }
    cache_dtype = _DTYPE_MAP[args.cache_dtype]
    compute_dtype = _DTYPE_MAP[args.compute_dtype]

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    model, run_block = load_model(ckpt_path)

    # The checkpoint's own conditioning layout (plan §8.1) — used both to
    # auto-detect the outcome gate and to assemble in-distribution edge-case
    # tokens with the matching [BOS][cond…] prefix.
    conditioning = conditioning_from_run_block(run_block)
    # Auto-detect outcome conditioning from the checkpoint's run block when
    # the operator didn't pass an explicit gate flag.
    if args.trained is None:
        outcome_prefix_trained = "outcome" in conditioning
    else:
        outcome_prefix_trained = args.trained

    results = run_all_diagnostics(
        model,
        outcome_prefix_trained=outcome_prefix_trained,
        n_per_outcome=args.gen_n_per_outcome,
        max_seq_len=args.gen_max_seq_len,
        cache_dtype=cache_dtype,
        compute_dtype=compute_dtype,
    )

    if args.edge_cases:
        from pawn.eval_suite.diagnostics import compute_edge_case_accuracy_quota
        edge = compute_edge_case_accuracy_quota(
            model, per_label=args.edge_per_label, conditioning=conditioning,
        )
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
