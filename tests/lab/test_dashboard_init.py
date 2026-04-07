"""Tests for pawn.dashboard package __init__ — lazy-loading."""

from __future__ import annotations

import importlib

import pytest


class TestDashboardInit:
    def test_exports_metrics_helpers(self):
        import pawn.dashboard as d
        assert hasattr(d, "load_runs")
        assert hasattr(d, "load_metrics")
        assert hasattr(d, "detect_run_type")
        assert hasattr(d, "col")

    def test_all_list(self):
        import pawn.dashboard as d
        assert "load_runs" in d.__all__
        assert "load_metrics" in d.__all__
        assert "detect_run_type" in d.__all__
        assert "col" in d.__all__
        assert "Dashboard" in d.__all__
        assert "Runner" in d.__all__
        assert "Page" in d.__all__

    def test_unknown_attr_raises(self):
        import pawn.dashboard as d
        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            d.nonexistent

    def test_importing_metrics_helpers_does_not_load_solara(self):
        # Clear sol from cache, import dashboard, verify sol NOT imported
        import sys
        sys.modules.pop("pawn.dashboard.sol", None)
        # Re-import pawn.dashboard
        import pawn.dashboard
        importlib.reload(pawn.dashboard)
        # Access load_runs without triggering sol
        _ = pawn.dashboard.load_runs
        assert "pawn.dashboard.sol" not in sys.modules
