"""PAWN: Playstyle-Agnostic World-model Network for Chess.

A causal transformer trained on random chess games, designed as a
testbed for finetuning and augmentation methods at small scales.

This module re-exports the v2 (JAX/Equinox/Optax) top-level types so
``from pawn import ModelConfig, PAWNModel, PretrainConfig`` works as a
one-stop import. v1 names (``CLMConfig`` / ``TrainingConfig`` /
``PAWNCLM``) are kept *reachable* via PEP 562 ``__getattr__`` so old
import sites surface a precise migration ``ImportError`` instead of a
generic "module has no attribute".

The v2 imports are lightweight — `pawn.config` and `pawn.run_config`
don't pull JAX into the import graph; `pawn.model` does. Consumers
that need to stay JAX-free (the dashboard, lab MCP) should import
the specific sub-module they need rather than relying on this
top-level surface.
"""

from __future__ import annotations

from typing import Any

# v2 public types — re-exported from `pawn`. `pawn.config` and
# `pawn.run_config` are JAX-free (pydantic + stdlib only); `pawn.model`
# pulls JAX into the import graph, which would break the lightweight-
# import invariant for the sweep driver / dashboard / sentinel
# consumers. `PAWNModel` is therefore **not** eagerly imported here —
# it's reachable via `from pawn.model import PAWNModel` directly, or
# `from pawn import PAWNModel` via the PEP-562 `__getattr__` below.
from pawn.config import ModelConfig
from pawn.run_config import (
    AdapterConfig,
    BaseRunConfig,
    PretrainConfig,
    RunConfig,
    SpecializedCLMConfig,
)

__all__ = [
    "ModelConfig",
    "BaseRunConfig",
    "PretrainConfig",
    "AdapterConfig",
    "SpecializedCLMConfig",
    "RunConfig",
]


# v1 names — kept reachable via `__getattr__` (PEP 562) so import sites
# that say `from pawn import CLMConfig` surface a precise migration
# error instead of a generic "module has no attribute". The error
# message points at the v2 replacement so the fix is mechanical.
_RENAMED: dict[str, str] = {
    "CLMConfig": (
        "`pawn.CLMConfig` was renamed to `pawn.ModelConfig` in the v2 "
        "JAX/Equinox/Optax stack. Replace `from pawn import CLMConfig` "
        "with `from pawn import ModelConfig`. The v2 config drops "
        "torch-only fields (`amp_dtype`, `device`, `num_workers`, "
        "`no_compile`, `sdpa_math`) — see `pawn/config.py` for the v2 "
        "field set. See `docs/jax_migration_plan.md` for rationale."
    ),
    "TrainingConfig": (
        "`pawn.TrainingConfig` is gone — the v2 stack uses a "
        "discriminated-union `pawn.RunConfig` covering "
        "`PretrainConfig`, `AdapterConfig`, and "
        "`SpecializedCLMConfig`. For pretraining, replace `from pawn "
        "import TrainingConfig` with `from pawn import PretrainConfig`. "
        "See `pawn/run_config.py` for the v2 surface."
    ),
    "PAWNCLM": (
        "`pawn.PAWNCLM` was renamed to `pawn.PAWNModel` in the v2 "
        "JAX/Equinox/Optax stack. Replace `from pawn import PAWNCLM` "
        "with `from pawn import PAWNModel`. The v2 model is an "
        "Equinox module (not a `torch.nn.Module`); its `__call__` "
        "signature differs and there is no separate `.forward()` "
        "method. See `pawn/model.py` for the v2 surface."
    ),
}


def __getattr__(name: str) -> Any:
    if name == "PAWNModel":
        # Lazy import — kept out of `__all__` to preserve the
        # JAX-free import invariant for `pawn.config` /
        # `pawn._sentinel` / `pawn.logging` / `pawn.run_config`
        # consumers (see tests/test_jax_{config,sentinel,logging}.py).
        from pawn.model import PAWNModel  # noqa: PLC0415

        return PAWNModel
    if name in _RENAMED:
        raise ImportError(_RENAMED[name])
    raise AttributeError(f"module 'pawn' has no attribute {name!r}")
