"""PAWN training dashboard — Solara + Jupyter.

Standalone:  solara run pawn.dashboard.sol
Jupyter:     from pawn.dashboard import Dashboard, Runner; Dashboard()
CLI:         python -m pawn.dashboard --log-dir ../logs
"""

from .metrics import col, detect_run_type, load_metrics, load_runs


def __getattr__(name):
    # Lazy-load Solara components so metrics work without dashboard deps
    if name in ("Dashboard", "Runner", "Page"):
        from . import sol
        return getattr(sol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Dashboard",
    "Runner",
    "Page",
    "load_runs",
    "load_metrics",
    "detect_run_type",
    "col",
]
