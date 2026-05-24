#!/usr/bin/env python3
"""Backwards-compatibility wrapper for the v1 unified training entry point.

The v1 ``scripts/train.py`` dispatched on ``--run-type {pretrain, adapter,
cotrain}`` into a single Python process. The v2 (JAX/Equinox/Optax) stack
splits that into three dedicated entry points:

- ``scripts/train_jax.py`` — pretraining (was ``--run-type pretrain``)
- ``scripts/train_jax_adapter.py`` — adapter finetuning (was
  ``--run-type adapter``); also handles ``--strategy specialized_clm``
  for from-scratch standalone CLMs.
- *Cotrain is gone by design* — the v2 supernet's joint loss replaces
  it. See ``docs/jax_migration_plan.md`` §6 ("GONE BY DESIGN") and the
  ``--supernet`` flag on ``scripts/train_jax.py`` for the equivalent
  workflow.

This wrapper preserves the ``python scripts/train.py --run-type …``
invocation contract: the user's argv is forwarded verbatim (minus the
``--run-type`` flag) to the appropriate v2 script. A deprecation warning
is printed to stderr so workflows pick up the rename naturally without
silently breaking.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HELP_TAIL = """\

For new code, invoke the v2 entry points directly:

  python scripts/train_jax.py  --supernet {tiny,production} ...
  python scripts/train_jax_adapter.py  --strategy {lora,film,...} ...

This wrapper exists so v1 invocations (`scripts/train.py --run-type X`)
continue to work after the framework swap to JAX/Equinox/Optax.
"""


def _die(msg: str, code: int = 2) -> None:
    print(f"scripts/train.py: {msg}", file=sys.stderr)
    sys.exit(code)


def _pop_run_type(argv: list[str]) -> tuple[str | None, list[str]]:
    """Strip ``--run-type X`` (or ``--run-type=X``) from argv.

    Returns ``(run_type or None, rest_of_argv)``. v1 also allowed the
    field to live inside a JSON ``--config``; if neither CLI nor a
    detected ``--config`` reveals it, ``None`` is returned and the
    caller falls back to reading the JSON.
    """
    out: list[str] = []
    run_type: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--run-type":
            if i + 1 >= len(argv):
                _die("--run-type requires an argument")
            run_type = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--run-type="):
            run_type = arg.split("=", 1)[1]
            i += 1
            continue
        out.append(arg)
        i += 1
    return run_type, out


def _run_type_from_config(argv: list[str]) -> str | None:
    """If a ``--config <path>`` JSON exists in argv, peek for
    ``run_type``."""
    import json

    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            path = Path(argv[i + 1])
            if path.is_file():
                try:
                    return json.loads(path.read_text(encoding="utf-8")).get(
                        "run_type"
                    )
                except (json.JSONDecodeError, OSError):
                    return None
        if arg.startswith("--config="):
            path = Path(arg.split("=", 1)[1])
            if path.is_file():
                try:
                    return json.loads(path.read_text(encoding="utf-8")).get(
                        "run_type"
                    )
                except (json.JSONDecodeError, OSError):
                    return None
    return None


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    run_type, rest = _pop_run_type(list(raw_argv))
    if run_type is None:
        run_type = _run_type_from_config(rest)

    if run_type is None:
        _die(
            "v2 requires the run type to be inferable from the CLI or the "
            f"JSON config. Pass `--run-type pretrain` (or adapter / "
            f"specialized_clm).\n{_HELP_TAIL}"
        )
    # `_die` raises SystemExit, but pyright doesn't pick that up — narrow
    # explicitly so the str-keyed dict lookup below is type-clean.
    assert run_type is not None

    if run_type == "cotrain":
        _die(
            "`--run-type cotrain` is GONE BY DESIGN per "
            "docs/jax_migration_plan.md §6 — the v2 supernet's joint loss "
            "(see `scripts/train_jax.py --supernet {tiny,production}`) "
            "replaces the v1 multi-variant cotrain trainer.\n"
            f"{_HELP_TAIL}"
        )

    target_map: dict[str, str] = {
        "pretrain": "scripts/train_jax.py",
        "adapter": "scripts/train_jax_adapter.py",
        "specialized_clm": "scripts/train_jax_adapter.py",
    }
    if run_type not in target_map:
        _die(
            f"unknown --run-type {run_type!r}; expected one of "
            f"{sorted(target_map)} or 'cotrain' (rejected loudly).\n"
            f"{_HELP_TAIL}"
        )
    target = target_map[run_type]

    # The specialized_clm path threads through the adapter entry; it
    # needs `--strategy specialized_clm` if not already in argv.
    if run_type == "specialized_clm" and not any(
        a == "--strategy" or a.startswith("--strategy=") for a in rest
    ):
        rest = ["--strategy", "specialized_clm", *rest]

    print(
        f"[scripts/train.py] DeprecationWarning: forwarding to {target!r}. "
        f"Update invocations to use the v2 entry point directly.",
        file=sys.stderr,
        flush=True,
    )

    # Exec preserves the original PID + signals (SIGTERM handling lives in
    # the v2 scripts). os.execvp lets the v2 entry point own the process.
    target_path = Path(__file__).parent / Path(target).name
    os.execvp(sys.executable, [sys.executable, str(target_path), *rest])


if __name__ == "__main__":
    raise SystemExit(main())
