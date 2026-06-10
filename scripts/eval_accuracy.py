#!/usr/bin/env python3
"""Backwards-compatibility wrapper for the v1 adapter-accuracy eval.

v1's ``eval_accuracy.py`` evaluated an *adapter* on a held-out Lichess
slice: it took a backbone ``--checkpoint`` plus an
``--adapter-checkpoint`` and ran the ADAPTED model (not the bare
backbone, not random games). In v2 the adapter checkpoint directory is
self-contained — ``model.safetensors`` holds the folded effective model
(lora / sparse / unfreeze) or the frozen backbone plus a typed
``adapter.safetensors`` sidecar (bottleneck / pure-FiLM / hybrid), which
:func:`pawn.checkpoint.load_eval_model` re-applies at eval time. So the
v2 entry point ``scripts/eval_jax.py`` already evaluates the adapted
model when ``--checkpoint`` points at an adapter checkpoint.

This wrapper preserves the v1 CLI surface: it maps ``--adapter-checkpoint``
onto v2's ``--checkpoint`` (so the *adapter* dir is what's loaded and
adapted), drops the now-folded-in v1 ``--checkpoint`` backbone arg, maps
v1's ``--max-games`` onto v2's ``--n-games``, and forwards the
within-distribution ``--elo-min`` / ``--elo-max`` / ``--min-eval-ply`` /
``--per-ply`` / ``--pgn`` flags through to ``eval_jax.py``. The v1-only
flags with no v2 analogue (``--device`` / ``--amp-dtype`` / ``--val-start``
/ ``--val-games`` / ``--prepend-outcome``) are consumed here and dropped
with a stderr note rather than forwarded verbatim — otherwise eval_jax.py's
argparse would reject the exact v1 invocation this wrapper exists to
support. A one-line DeprecationWarning points users at the v2 entry point.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _translate(argv: list[str]) -> list[str]:
    """Map the v1 adapter-accuracy CLI onto v2 ``eval_jax.py`` flags.

    The v1 surface separated the backbone (``--checkpoint``) from the
    trained adapter (``--adapter-checkpoint``). In v2 the adapter
    checkpoint is self-contained, so the adapter dir becomes v2's
    ``--checkpoint`` and the v1 backbone arg is dropped (it's folded
    into / referenced by the adapter checkpoint already).

    The v1-only flags that have no v2 analogue (``--device`` /
    ``--amp-dtype`` — JAX picks the device, fp32 is the eval dtype — and
    the ``--val-start`` / ``--val-games`` raw-game-slice knobs and
    ``--prepend-outcome``, which the v2 checkpoint config carries) are
    *consumed here* and dropped with a stderr note, so they never reach
    ``eval_jax.py``'s argparse (which would reject them with exit 2).
    ``--max-games`` (v1's game-count knob) is mapped onto v2's
    ``--n-games``. Every remaining flag the v2 entry point understands
    (``--pgn`` / ``--min-eval-ply`` / ``--per-ply`` / ``--elo-min`` /
    ``--elo-max`` / ``--batch-size`` / ...) passes straight through.
    """
    p = argparse.ArgumentParser(prog="eval_accuracy", add_help=False)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--adapter-checkpoint", default=None)
    # v1-only flags with a v2 mapping or no v2 analogue. Parsing them here
    # removes them from ``passthrough`` so a verbatim v1 invocation never
    # trips eval_jax.py's argparse on an unrecognised flag.
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--amp-dtype", default=None)
    p.add_argument("--val-start", default=None)
    p.add_argument("--val-games", default=None)
    p.add_argument("--prepend-outcome", action="store_true", default=False)
    args, passthrough = p.parse_known_args(argv)

    out: list[str] = []
    # The adapter dir is the v2 --checkpoint (it carries the adapted
    # model). Fall back to the v1 backbone --checkpoint when no adapter
    # was given (a bare-backbone accuracy run).
    eval_ckpt = args.adapter_checkpoint or args.checkpoint
    if eval_ckpt is not None:
        out += ["--checkpoint", eval_ckpt]
    if args.adapter_checkpoint is not None and args.checkpoint is not None:
        print(
            "[scripts/eval_accuracy.py] note: v2 adapter checkpoints are "
            "self-contained; ignoring the v1 backbone --checkpoint "
            f"{args.checkpoint!r} and evaluating the adapter checkpoint "
            f"{args.adapter_checkpoint!r} directly.",
            file=sys.stderr, flush=True,
        )
    # Map v1 --max-games onto v2 --n-games (the game-count knob).
    if args.max_games is not None:
        out += ["--n-games", str(args.max_games)]
    # Drop the v1-only flags that v2 doesn't model, with a note so the
    # change in behaviour is visible.
    dropped: list[str] = []
    if args.device is not None:
        dropped.append(f"--device {args.device!r}")
    if args.amp_dtype is not None:
        dropped.append(f"--amp-dtype {args.amp_dtype!r}")
    if args.val_start is not None:
        dropped.append(f"--val-start {args.val_start!r}")
    if args.val_games is not None:
        dropped.append(f"--val-games {args.val_games!r}")
    if args.prepend_outcome:
        dropped.append("--prepend-outcome")
    if dropped:
        print(
            "[scripts/eval_accuracy.py] note: dropping v1-only flags with "
            "no v2 analogue (JAX picks the device, the eval dtype is fp32, "
            "the validation slice + conditioning layout come from the "
            f"checkpoint config): {', '.join(dropped)}.",
            file=sys.stderr, flush=True,
        )
    out += passthrough
    return out


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    print(
        "[scripts/eval_accuracy.py] DeprecationWarning: forwarding to "
        "scripts/eval_jax.py (the v2 adapter-aware accuracy eval). "
        "Update invocations to use the v2 entry point directly.",
        file=sys.stderr,
        flush=True,
    )
    target = Path(__file__).parent / "eval_jax.py"
    translated = _translate(raw)
    os.execvp(sys.executable, [sys.executable, str(target), *translated])


if __name__ == "__main__":
    raise SystemExit(main())
