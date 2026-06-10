"""PAWN training dashboard.

Light-weight v2 surface: only the metrics loader is unconditionally
importable; the Solara UI lives behind a lazy `__getattr__` so
`from pawn.dashboard.metrics import load_metrics` works without
needing the `dashboard` extra installed.

CLI:        python -m pawn.dashboard --log-dir <run-dir>
Jupyter:    from pawn.dashboard import Dashboard; Dashboard()
"""

from pawn.dashboard.metrics import (
    MetricsBundle,
    col,
    detect_run_type,
    discover_runs,
    list_trials,
    load_metrics,
    load_run_buckets,
    load_runs,
)


def __getattr__(name):
    # Solara components are lazy-loaded so the `dashboard` extra is
    # only required when the UI is actually instantiated.
    if name in ("Dashboard", "Runner", "Page"):
        from pawn.dashboard import sol

        return getattr(sol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Dashboard",
    "Runner",
    "Page",
    "MetricsBundle",
    "col",
    "detect_run_type",
    "discover_runs",
    "list_trials",
    "load_metrics",
    "load_run_buckets",
    "load_runs",
]
