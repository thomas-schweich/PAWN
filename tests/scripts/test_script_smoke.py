"""Smoke tests for scripts/*.py: ``--help`` exits 0 and emits a usage line.

After the Phase-4 PyTorch removal every shipped script uses argparse,
so the historical ``IMPORTABLE_SCRIPTS`` table for ``train.py``'s
custom parser is gone — all entry points are covered uniformly by
the argparse-help subprocess invocation.
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
    "benchmark_stockfish_nodes.py",
    "compact_stockfish_dataset.py",
    "compute_theoretical_ceiling.py",
    "convert_published_checkpoints.py",
    "datagen_reconcile_tier.py",
    "datagen_with_hf_sync.py",
    "eval_generation_jax.py",
    "eval_jax.py",
    "eval_probes_jax.py",
    "eval_vs_stockfish.py",
    "export_hf_repo.py",
    "extract_lichess_parquet.py",
    "generate_lc0_data.py",
    "generate_model_cards.py",
    "rename_shards.py",
    "train_jax.py",
    "train_jax_adapter.py",
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
# Help output exists for all CLI scripts
# =====================================================================


def test_all_argparse_scripts_are_covered():
    """Every script with argparse should be listed in ARGPARSE_SCRIPTS."""
    import re
    known = set(ARGPARSE_SCRIPTS)
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
