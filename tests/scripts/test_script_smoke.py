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


def _has_pep723_inline_deps(path: Path) -> bool:
    """Return True iff the script declares PEP 723 inline dependencies.

    Such scripts (e.g. `vastai_score.py`) need to be invoked through
    `uv run --script` so the declared deps are installed in a transient
    venv. Running them with bare `python` produces a misleading
    `ModuleNotFoundError` for deps the script's shebang would have set
    up automatically.
    """
    try:
        with path.open("r") as fh:
            for _ in range(10):
                line = fh.readline()
                if not line:
                    return False
                if line.startswith("# /// script"):
                    return True
    except OSError:
        return False
    return False


def _run_help(script_name: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Invoke ``<script> --help`` and return the result.

    PEP 723 inline-deps scripts (`# /// script` block at the top) are
    dispatched through ``uv run --script`` so their declared deps resolve;
    plain scripts use the test's Python directly.
    """
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO / ".mpl-cache"))
    env.setdefault("PAWN_ALLOW_CPU", "1")
    path = SCRIPTS / script_name
    if _has_pep723_inline_deps(path):
        cmd = ["uv", "run", "--script", str(path), "--help"]
    else:
        cmd = [sys.executable, str(path), "--help"]
    return subprocess.run(
        cmd,
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
    "convert_published_checkpoints.py",
    "datagen_reconcile_tier.py",
    "datagen_with_hf_sync.py",
    "eval_accuracy.py",
    "eval_probes.py",
    "eval_vs_stockfish.py",
    "export_hf_repo.py",
    "extract_lichess_parquet.py",
    "generate_lc0_data.py",
    "generate_model_cards.py",
    "rename_shards.py",
    "run_evals_backbone.py",
    "train_jax.py",
    "vastai_score.py",
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
