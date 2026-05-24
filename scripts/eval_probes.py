#!/usr/bin/env python3
"""Backwards-compatibility wrapper for the v1 linear-probes eval.

`scripts/eval_probes_jax.py` is the v2 replacement. Forwards argv +
prints a one-line DeprecationWarning so existing workflows keep
working post-framework-swap.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).parent / "eval_probes_jax.py"
    print(
        "[scripts/eval_probes.py] DeprecationWarning: forwarding to "
        f"scripts/eval_probes_jax.py. Update invocations to use the v2 "
        "entry point directly.",
        file=sys.stderr,
        flush=True,
    )
    os.execvp(sys.executable, [sys.executable, str(target), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
