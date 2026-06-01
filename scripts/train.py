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
from typing import NoReturn

_USAGE = """\
usage: scripts/train.py --run-type {pretrain,adapter,specialized_clm} [...]

v1-compat wrapper. Dispatches on --run-type into the v2 entry points:

  --run-type pretrain          -> scripts/train_jax.py
  --run-type adapter           -> scripts/train_jax_adapter.py
  --run-type specialized_clm   -> scripts/train_jax_adapter.py --strategy specialized_clm
  --run-type cotrain           -> rejected (GONE BY DESIGN; use --supernet)

v1 --variant {toy,small,base,large} on a pretrain run is translated to the
v2 --supernet/--variants surface. For the full flag surface of a target, run
e.g. `scripts/train.py --run-type pretrain --help` (the wrapper forwards
--help to the resolved v2 script).
"""

_HELP_TAIL = """\

For new code, invoke the v2 entry points directly:

  python scripts/train_jax.py  --supernet {tiny,production} ...
  python scripts/train_jax_adapter.py  --strategy {lora,film,...} ...

This wrapper exists so v1 invocations (`scripts/train.py --run-type X`)
continue to work after the framework swap to JAX/Equinox/Optax.
"""


def _die(msg: str, code: int = 2) -> NoReturn:
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

    def _peek(path: Path) -> str | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        # ``payload`` is ``Any`` (json.loads); guard the lookup so a
        # non-dict JSON root or non-string ``run_type`` yields ``None``
        # rather than leaking ``Any`` past the ``-> str | None`` contract.
        if not isinstance(payload, dict):
            return None
        run_type = payload.get("run_type")
        return run_type if isinstance(run_type, str) else None

    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return _peek(Path(argv[i + 1]))
        if arg.startswith("--config="):
            return _peek(Path(arg.split("=", 1)[1]))
    return None


# v1 `--variant {toy,small,base,large,custom}` selected a single standalone
# model to pretrain. v2 trains the supernet (`--supernet {tiny,production}`)
# and restricts to a nested slice via `--variants {small,base,large}`. The
# mapping is therefore: `toy` → the tiny supernet smoke shape; `small`/
# `base`/`large` → the production supernet, narrowed to that single nested
# variant (so the v1 "pretrain just base" intent survives as the
# corresponding single-variant supernet slice).
_VARIANT_TO_SUPERNET: dict[str, str] = {
    "toy": "tiny",
    "small": "production",
    "base": "production",
    "large": "production",
}


def _pop_variant(argv: list[str]) -> tuple[str | None, list[str]]:
    """Strip ``--variant X`` (or ``--variant=X``) from argv.

    Returns ``(variant or None, rest_of_argv)``. The v2 ``train_jax.py``
    has no ``--variant`` flag (it uses ``--supernet`` + ``--variants``), so
    a verbatim-forwarded v1 ``--variant base`` would trip its argparse with
    exit 2. We pop it here and translate it in :func:`_translate_variant`.
    """
    out: list[str] = []
    variant: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--variant":
            if i + 1 >= len(argv):
                _die("--variant requires an argument")
            variant = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--variant="):
            variant = arg.split("=", 1)[1]
            i += 1
            continue
        out.append(arg)
        i += 1
    return variant, out


def _translate_variant(variant: str, argv: list[str]) -> list[str]:
    """Translate a popped v1 ``--variant`` into v2 ``--supernet``/``--variants``.

    ``argv`` is the remaining argv (``--variant`` already removed). If the
    user *also* passed the v2-native ``--supernet`` / ``--variants`` flags,
    that's an ambiguous mix of the v1 and v2 surfaces — reject it rather
    than silently pick one. ``--variant custom`` has no v2 standalone-pretrain
    home (the supernet is fixed-shape; a from-scratch custom-arch model is the
    ``specialized_clm`` adapter path), so it's rejected with that pointer.
    """
    has_supernet = any(
        a == "--supernet" or a.startswith("--supernet=") for a in argv
    )
    has_variants = any(
        a == "--variants" or a.startswith("--variants=") for a in argv
    )
    if has_supernet or has_variants:
        _die(
            f"--variant {variant!r} (v1 surface) cannot be combined with the "
            "v2-native --supernet / --variants flags. Use one surface or the "
            f"other.\n{_HELP_TAIL}"
        )
    if variant == "custom":
        _die(
            "--variant custom has no v2 supernet-pretrain analogue (the "
            "supernet is fixed-shape). Train a from-scratch custom-arch model "
            "via the specialized_clm path:\n"
            "  python scripts/train_jax_adapter.py --strategy specialized_clm "
            "--d-model … --n-layers … --n-heads … --d-ff …\n"
            f"{_HELP_TAIL}"
        )
    supernet = _VARIANT_TO_SUPERNET.get(variant)
    if supernet is None:
        _die(
            f"unknown --variant {variant!r}; expected one of "
            f"{sorted(_VARIANT_TO_SUPERNET) + ['custom']}.\n{_HELP_TAIL}"
        )
    translated = ["--supernet", supernet]
    # toy → the tiny supernet trains all three nested variants (the smoke
    # shape); small/base/large → narrow the production supernet to that one
    # nested slice so the v1 "pretrain just this variant" intent survives.
    if variant in ("small", "base", "large"):
        translated += ["--variants", variant]
    return [*translated, *argv]


def resolve_target(raw_argv: list[str]) -> tuple[str, list[str]]:
    """Resolve a v1 ``train.py`` invocation to a ``(target, argv)`` pair.

    ``target`` is the basename of the v2 entry script to exec
    (``train_jax.py`` / ``train_jax_adapter.py``); ``argv`` is the
    translated argv to pass it (``--run-type`` and ``--variant`` removed,
    v2 ``--supernet`` / ``--variants`` / ``--strategy`` synthesised as
    needed). Pure + side-effect-free apart from ``_die`` (SystemExit) on
    an unsupported invocation, so the translation is unit-testable without
    forking the v2 process. ``main`` calls this then ``os.execvp``\\ s.
    """
    run_type, rest = _pop_run_type(list(raw_argv))
    if run_type is None:
        run_type = _run_type_from_config(rest)

    if run_type is None:
        _die(
            "v2 requires the run type to be inferable from the CLI or the "
            f"JSON config. Pass `--run-type pretrain` (or adapter / "
            f"specialized_clm).\n{_HELP_TAIL}"
        )

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
    target = Path(target_map[run_type]).name

    # v1 `--variant` selected the model arch for *pretraining* only (the
    # adapter / specialized_clm paths derive their arch from the loaded
    # backbone checkpoint or explicit --d-model/... flags). Pop it here so a
    # verbatim v1 invocation never reaches the v2 argparse, then translate
    # (pretrain) or drop-with-a-note (adapter paths).
    variant, rest = _pop_variant(rest)
    if run_type == "pretrain":
        if variant is not None:
            rest = _translate_variant(variant, rest)
    elif variant is not None:
        print(
            f"[scripts/train.py] note: dropping v1 --variant {variant!r} on a "
            f"{run_type!r} run — the adapter / specialized_clm arch comes from "
            "the loaded backbone checkpoint (or explicit --d-model/... flags), "
            "not a variant preset.",
            file=sys.stderr,
            flush=True,
        )

    # The specialized_clm path threads through the adapter entry; it
    # needs `--strategy specialized_clm` if not already in argv.
    if run_type == "specialized_clm" and not any(
        a == "--strategy" or a.startswith("--strategy=") for a in rest
    ):
        rest = ["--strategy", "specialized_clm", *rest]

    return target, rest


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv

    # A bare `--help`/`-h` (no run type given) prints the wrapper's own usage
    # and exits 0 — there's no target to forward to yet. If a run type *is*
    # present, `--help` falls through and is forwarded to the resolved v2
    # script so the user sees that script's real flag surface.
    has_help = any(a in ("--help", "-h") for a in raw_argv)
    run_type_present = any(
        a == "--run-type" or a.startswith("--run-type=") for a in raw_argv
    ) or _run_type_from_config(raw_argv) is not None
    if has_help and not run_type_present:
        print(_USAGE + _HELP_TAIL)
        return 0

    target, rest = resolve_target(list(raw_argv))

    print(
        f"[scripts/train.py] DeprecationWarning: forwarding to {target!r}. "
        f"Update invocations to use the v2 entry point directly.",
        file=sys.stderr,
        flush=True,
    )

    # Exec preserves the original PID + signals (SIGTERM handling lives in
    # the v2 scripts). os.execvp lets the v2 entry point own the process.
    target_path = Path(__file__).parent / target
    os.execvp(sys.executable, [sys.executable, str(target_path), *rest])


if __name__ == "__main__":
    raise SystemExit(main())
