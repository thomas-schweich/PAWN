"""Tests for ``scripts/eval_jax.py``.

Focused on the upfront validation guards — the ``SystemExit`` paths
the script promises to fire before any corpus generation. The full
end-to-end happy path is covered by the converter-driven smoke tests
elsewhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("chess_engine")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script() -> ModuleType:
    name = "eval_jax_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / "eval_jax.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _run(args: list[str], tmp_path: Path) -> None:
    script = _load_script()
    if "--logs-dir" not in args:
        args = args + ["--logs-dir", str(tmp_path)]
    script.main(args)


def test_rejects_seq_len_exceeding_max(tmp_path: Path) -> None:
    """``--seq-len`` > model.max_seq_len fails BEFORE corpus
    generation. The TINY supernet has max_seq_len=512; request 1024
    and the script must exit immediately rather than burning the
    corpus-gen pass first. Mirrors the upfront guards in
    ``scripts/train_jax.py`` and ``scripts/train_jax_adapter.py``.
    """
    with pytest.raises(SystemExit, match="seq-len"):
        _run(
            [
                "--supernet", "tiny",
                "--variant", "base",
                "--n-games", "8",
                "--seq-len", "1024",
                "--batch-size", "2",
                "--quiet",
            ],
            tmp_path,
        )


def test_rejects_seq_len_zero(tmp_path: Path) -> None:
    """``--seq-len 0`` is caught by the positivity guard."""
    with pytest.raises(SystemExit, match="--seq-len"):
        _run(
            [
                "--supernet", "tiny",
                "--n-games", "8",
                "--seq-len", "0",
                "--batch-size", "2",
                "--quiet",
            ],
            tmp_path,
        )


def test_rejects_n_games_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--n-games"):
        _run(
            [
                "--supernet", "tiny",
                "--n-games", "0",
                "--seq-len", "16",
                "--batch-size", "2",
                "--quiet",
            ],
            tmp_path,
        )


def test_rejects_batch_size_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--batch-size"):
        _run(
            [
                "--supernet", "tiny",
                "--n-games", "8",
                "--seq-len", "16",
                "--batch-size", "0",
                "--quiet",
            ],
            tmp_path,
        )
