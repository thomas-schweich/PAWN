"""CLI driver for ``pawn.sweep``.

Wraps the Optuna study + AdapterObjective into a one-liner for
sweeping an adapter strategy:

    uv run python scripts/sweep.py --strategy lora --n-trials 50 \
        --supernet tiny --variant base \
        --storage sqlite:///sweeps/lora.db \
        --logs-dir logs/sweeps/lora

The trial subprocess is ``scripts/train_jax_adapter.py`` (see
``pawn.sweep.TRAIN_SCRIPT``); per-trial argv is built from the
strategy's suggester function in ``pawn.sweep.SUGGEST_FNS``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pawn.sweep import SUGGEST_FNS, AdapterObjective, create_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(SUGGEST_FNS),
        help="adapter strategy to sweep",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="number of trials to run (default: 50)",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (e.g. sqlite:///sweeps/lora.db). "
        "Default: in-memory (study state lost on exit)",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help='Optuna study name (default: "pawn-<strategy>")',
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path / HF repo for the frozen backbone (use this OR "
        "--supernet + --variant; ignored for specialized_clm)",
    )
    parser.add_argument(
        "--supernet",
        default=None,
        choices=["tiny", "supernet"],
        help="which supernet shape the frozen backbone comes from "
        "(use this with --variant, OR pass --checkpoint)",
    )
    parser.add_argument("--variant", default=None, choices=["small", "base", "large"])
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/sweeps"),
        help="parent dir for per-trial run dirs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="optional per-trial timeout (seconds). Default: no timeout",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args forwarded verbatim to scripts/train_jax_adapter.py",
    )
    args = parser.parse_args()

    args.logs_dir.mkdir(parents=True, exist_ok=True)

    study = create_study(
        strategy=args.strategy,
        storage=args.storage,
        study_name=args.study_name,
    )

    objective = AdapterObjective(
        strategy=args.strategy,
        checkpoint=args.checkpoint,
        supernet=args.supernet,
        variant=args.variant,
        logs_dir=args.logs_dir,
        extra_args=args.extra,
        timeout_s=args.timeout,
    )

    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n[sweep] best trial: {study.best_trial.number}")
    print(f"[sweep] best val_loss: {study.best_trial.value}")
    print(f"[sweep] best params: {study.best_trial.params}")
    print(f"[sweep] best run_dir: {study.best_trial.user_attrs.get('run_dir')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
