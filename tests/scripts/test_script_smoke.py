"""Smoke tests for scripts/*.py: import cleanly and --help doesn't crash.

For argparse-based scripts, we invoke ``python scripts/<name>.py --help``
via subprocess and assert exit 0. For the custom parser in train.py (which
doesn't natively support --help), we import the module and verify key
callables exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def _run_help(script_name: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Invoke ``python scripts/<name> --help`` and return the result."""
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO / ".mpl-cache"))
    env.setdefault("PAWN_ALLOW_CPU", "1")
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script_name), "--help"],
        capture_output=True, text=True, timeout=timeout, env=env,
        cwd=str(REPO),
    )


# =====================================================================
# Scripts with argparse (all support --help → exit 0)
# =====================================================================


ARGPARSE_SCRIPTS = [
    "benchmark.py",
    "benchmark_stockfish_nodes.py",
    "compute_theoretical_ceiling.py",
    "eval_accuracy.py",
    "eval_probes.py",
    "eval_vs_stockfish.py",
    "export_hf_repo.py",
    "extract_lichess_parquet.py",
    "generate_lc0_data.py",
    "generate_model_cards.py",
    "rename_shards.py",
    "run_evals_backbone.py",
]


@pytest.mark.parametrize("script", ARGPARSE_SCRIPTS)
def test_script_help_exits_cleanly(script):
    """Each argparse-based script exits 0 on --help."""
    r = _run_help(script)
    assert r.returncode == 0, (
        f"{script} --help failed (exit={r.returncode}):\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    # argparse always emits "usage:" on --help
    assert "usage:" in r.stdout, (
        f"{script} --help stdout missing 'usage:' line: {r.stdout!r}"
    )


# =====================================================================
# train.py (custom CLI parser, no --help; verify import only)
# =====================================================================


class TestTrainPyImports:
    def test_module_imports(self):
        # Import as a top-level module to avoid scripts.train package issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "scripts_train", str(SCRIPTS / "train.py"),
        )
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Verify key functions exist
        assert hasattr(mod, "_parse_cli")
        assert hasattr(mod, "_die")


# =====================================================================
# Scripts that may not have argparse (generate_lc0/stockfish, etc.)
# Run as import-level smoke tests.
# =====================================================================


def _test_script_importable(script_name: str):
    """Attempt to import a script file as a module without executing __main__."""
    import importlib.util
    path = SCRIPTS / script_name
    assert path.exists(), f"Script missing: {path}"
    # Verify it parses as valid Python via compile() — avoids running __main__.
    source = path.read_text()
    compile(source, str(path), "exec")


IMPORTABLE_SCRIPTS = [
    "train.py",
]


@pytest.mark.parametrize("script", IMPORTABLE_SCRIPTS)
def test_script_source_parses(script):
    """Script source is at least syntactically valid Python."""
    _test_script_importable(script)


# =====================================================================
# Help output exists for all CLI scripts
# =====================================================================


def test_all_argparse_scripts_are_covered():
    """Every script with argparse should be listed in ARGPARSE_SCRIPTS."""
    import re
    known = set(ARGPARSE_SCRIPTS) | set(IMPORTABLE_SCRIPTS)
    for path in sorted(SCRIPTS.glob("*.py")):
        source = path.read_text()
        # Look for argparse.ArgumentParser
        if "argparse.ArgumentParser" in source and not re.search(
            r"^\s*#.*argparse.ArgumentParser", source, re.MULTILINE,
        ):
            name = path.name
            # All scripts using argparse should be tested
            if name not in known:
                pytest.fail(
                    f"Script {name} uses argparse but is not in "
                    "ARGPARSE_SCRIPTS list"
                )
