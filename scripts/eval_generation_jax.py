"""JAX generation-diagnostics evaluator.

Runs the five §6 generation tests (`outcome_signal`, `prefix_continuation`,
`poisoned_prefix`, `impossible_task`, `improbable_task`) on a JAX checkpoint
and writes a per-run JSON report.

Usage:

    uv run python scripts/eval_generation_jax.py \\
        --checkpoint ~/.cache/huggingface/pawn-jax-converted/pawn-small \\
        --test outcome_signal --n-per-outcome 200

    # Run every test on a small corpus
    uv run python scripts/eval_generation_jax.py \\
        --checkpoint ~/.cache/huggingface/pawn-jax-converted/pawn-small \\
        --test all --n-per-outcome 100 --n-per-bucket 100

    # Fresh-init smoke (TINY)
    uv run python scripts/eval_generation_jax.py \\
        --supernet tiny --variant small \\
        --test outcome_signal --n-per-outcome 8 --batch-size 4

The diagnostics use ``pawn.generation.autoregressive_generate`` with
KV-cache enabled by default — runtime scales linearly in
``max_seq_len`` (the cache is shape-stable at ``cfg.max_seq_len``,
so the JIT trace cost is paid once per batch shape). Pass
``--no-kv-cache`` to fall back to the legacy full-recompute path.
"""

from __future__ import annotations

import argparse
import datetime
from typing import Any
import json
import os
from pathlib import Path

import chess_engine as engine
import jax
import numpy as np

from pawn.checkpoint import load_model
from pawn.config import (
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
    ModelConfig,
    validate_nested,
)
from pawn.generation import (
    impossible_task_test,
    improbable_task_test,
    outcome_signal_test,
    poisoned_prefix_test,
    prefix_continuation_test,
)
from pawn.model import PAWNModel, init_model, sliced


TESTS = (
    "outcome_signal",
    "prefix_continuation",
    "poisoned_prefix",
    "impossible_task",
    "improbable_task",
)


def _resolve_model_from_supernet(
    supernet_name: str, variant: str, key: jax.Array
) -> PAWNModel:
    if supernet_name == "tiny":
        supernet_cfg, variants = TINY_SUPERNET, TINY_VARIANTS
    else:
        supernet_cfg, variants = SUPERNET, VARIANTS
    if variant != "supernet" and variant not in variants:
        raise SystemExit(
            f"unknown variant '{variant}'; choose from "
            f"{sorted(variants.keys())} or 'supernet'."
        )
    target_cfg: ModelConfig = (
        supernet_cfg if variant == "supernet" else variants[variant]
    )
    validate_nested(target_cfg, supernet_cfg)
    supernet = init_model(supernet_cfg, key)
    return supernet if variant == "supernet" else sliced(supernet, target_cfg)


def _make_run_dir(
    base: Path, checkpoint: Path | None, logs_dir: Path | None,
) -> Path:
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    if checkpoint is not None:
        out = checkpoint.parent
        out.mkdir(parents=True, exist_ok=True)
        return out
    slug = f"jax_generation_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    out = base / slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def _generate_corpus(
    n_games: int, max_ply: int, seed: int
) -> dict[str, np.ndarray]:
    """One small held-out corpus reused by the prefix-based tests.

    The four corpus-consuming tests (prefix_continuation /
    poisoned_prefix / impossible / improbable) all need the same
    ``{move_ids, game_lengths, termination_codes}`` triple — generate
    it once and pass it through.
    """
    _ids, _t, _lm, move_ids, gls, tcs = engine.generate_clm_batch(
        n_games, max_ply, seed, False, 0.0, False,
    )
    return {
        "move_ids": move_ids,
        "game_lengths": gls,
        "termination_codes": tcs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="path to a JAX checkpoint directory; omit to use a freshly-"
             "initialised model from --supernet / --variant.",
    )
    parser.add_argument("--supernet", choices=("tiny", "supernet"), default="tiny")
    parser.add_argument("--variant", default="small")
    parser.add_argument(
        "--test", choices=(*TESTS, "all"), default="outcome_signal",
        help="which §6 test to run.",
    )
    parser.add_argument("--n-per-outcome", type=int, default=200)
    parser.add_argument("--n-per-bucket", type=int, default=100)
    parser.add_argument("--n-per-pair", type=int, default=200)
    parser.add_argument("--n-per-scenario", type=int, default=100)
    parser.add_argument("--corpus-size", type=int, default=2000)
    parser.add_argument("--corpus-max-ply", type=int, default=255)
    parser.add_argument("--corpus-seed", type=int, default=4242)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-kv-cache", dest="use_kv_cache", action="store_false",
        help="disable the KV-cache decoder and use the legacy full-recompute "
             "path (one full forward per step). Both paths are bitwise-"
             "equivalent; this flag is for parity-comparison runs.",
    )
    parser.set_defaults(use_kv_cache=True)
    args = parser.parse_args()

    if args.checkpoint is not None and not args.checkpoint.is_dir():
        raise SystemExit(f"--checkpoint {args.checkpoint} is not a directory")

    init_key = jax.random.key(args.seed)
    if args.checkpoint is not None:
        if not args.quiet:
            print(f"[gen] loading checkpoint {args.checkpoint}")
        model = load_model(args.checkpoint)
    else:
        model = _resolve_model_from_supernet(
            args.supernet, args.variant, init_key,
        )

    tests_to_run = TESTS if args.test == "all" else (args.test,)

    corpus_needed = any(
        t in tests_to_run for t in (
            "prefix_continuation", "poisoned_prefix",
            "impossible_task", "improbable_task",
        )
    )
    corpus = None
    if corpus_needed:
        if not args.quiet:
            print(
                f"[gen] generating held-out corpus "
                f"(n_games={args.corpus_size}, max_ply={args.corpus_max_ply})"
            )
        corpus = _generate_corpus(
            args.corpus_size, args.corpus_max_ply, args.corpus_seed,
        )

    results: dict[str, dict[str, Any]] = {}

    if "outcome_signal" in tests_to_run:
        results["outcome_signal"] = outcome_signal_test(
            model, n_per_outcome=args.n_per_outcome,
            batch_size=args.batch_size, seed=args.seed,
            verbose=not args.quiet, use_kv_cache=args.use_kv_cache,
        )
    if "prefix_continuation" in tests_to_run:
        assert corpus is not None
        results["prefix_continuation"] = prefix_continuation_test(
            model, corpus, n_per_bucket=args.n_per_bucket,
            batch_size=args.batch_size, seed=args.seed,
            verbose=not args.quiet, use_kv_cache=args.use_kv_cache,
        )
    if "poisoned_prefix" in tests_to_run:
        assert corpus is not None
        results["poisoned_prefix"] = poisoned_prefix_test(
            model, corpus, n_per_pair=args.n_per_pair,
            batch_size=args.batch_size, seed=args.seed,
            use_kv_cache=args.use_kv_cache,
        )
    if "impossible_task" in tests_to_run:
        assert corpus is not None
        results["impossible_task"] = impossible_task_test(
            model, corpus, n_per_scenario=args.n_per_scenario,
            batch_size=args.batch_size, seed=args.seed,
            use_kv_cache=args.use_kv_cache,
        )
    if "improbable_task" in tests_to_run:
        assert corpus is not None
        results["improbable_task"] = improbable_task_test(
            model, corpus, n_per_scenario=args.n_per_scenario,
            batch_size=args.batch_size, seed=args.seed,
            use_kv_cache=args.use_kv_cache,
        )

    out_dir = _make_run_dir(Path("logs"), args.checkpoint, args.logs_dir)
    out_path = out_dir / "generation_results.json"
    output = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "supernet": args.supernet if args.checkpoint is None else None,
        "variant": args.variant if args.checkpoint is None else None,
        "model_config": {
            "d_model": model.cfg.d_model,
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "d_ff": model.cfg.d_ff,
            "max_seq_len": model.cfg.max_seq_len,
        },
        "tests": list(tests_to_run),
        "args": {
            "n_per_outcome": args.n_per_outcome,
            "n_per_bucket": args.n_per_bucket,
            "n_per_pair": args.n_per_pair,
            "n_per_scenario": args.n_per_scenario,
            "corpus_size": args.corpus_size if corpus_needed else None,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "use_kv_cache": args.use_kv_cache,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    if not args.quiet:
        print(f"[gen] wrote {out_path}")


if __name__ == "__main__":
    main()
