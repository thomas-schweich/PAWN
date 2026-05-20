"""Tests for ``scripts/train_jax_adapter.py`` — Phase-3 driver.

Mirrors ``test_train_jax.py``'s shape: upfront-validation guards
+ happy-path E2E + no-orphan-run-dir invariant.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("optax")
pytest.importorskip("chess_engine")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script() -> ModuleType:
    name = "train_jax_adapter_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(SCRIPTS / "train_jax_adapter.py")
    )
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


_GOOD_BASE = [
    "--supernet", "tiny",
    "--variant", "base",
    "--rank", "4",
    "--total-steps", "10",
    "--k", "5",
    "--batch-size", "2",
    "--seq-len", "16",
    "--warmup-steps", "1",
    "--val-frac", "0.1",
]


def test_rejects_k_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--k"):
        _run([*_GOOD_BASE, "--k", "0"][:-2] + ["--k", "0"], tmp_path)


def test_rejects_batch_size_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--batch-size"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "0", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_rejects_seq_len_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--seq-len"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "0",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_rejects_total_steps_not_multiple_of_k(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="multiple of"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--rank", "4",
                "--total-steps", "7", "--k", "3",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_rejects_rank_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--rank"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--rank", "0",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--variant"):
        _run(
            [
                "--supernet", "tiny", "--variant", "huge", "--rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_rejects_val_frac_out_of_range(tmp_path: Path) -> None:
    for v in ("0", "1", "1.5", "-0.1"):
        with pytest.raises(SystemExit, match="--val-frac"):
            _run(
                [
                    "--supernet", "tiny", "--variant", "base", "--rank", "4",
                    "--total-steps", "10", "--k", "5",
                    "--batch-size", "2", "--seq-len", "16",
                    "--warmup-steps", "1", "--val-frac", v,
                ],
                tmp_path,
            )


def test_rejects_seq_len_exceeding_max(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="seq-len"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "9999",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_rejects_bad_lr_schedule(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="LR-schedule"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "10",       # warmup == total
                "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_happy_path_writes_metrics_and_config(tmp_path: Path) -> None:
    """End-to-end smoke: 50 steps of LoRA training on TINY/base.
    Verifies config.json carries the full ModelConfig + LoRA cfg,
    metrics.jsonl has one row per chunk with finite losses + at
    least one val row, and the run dir slug pattern is stable."""
    _run(
        [
            "--supernet", "tiny", "--variant", "base", "--rank", "4",
            "--total-steps", "50", "--k", "10",
            "--batch-size", "2", "--seq-len", "16",
            "--warmup-steps", "5", "--val-frac", "0.1",
            "--val-every", "2", "--quiet",
        ],
        tmp_path,
    )
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1
    rd = runs[0]
    cfg = json.loads((rd / "config.json").read_text())
    # Config carries full per-variant ModelConfig dicts.
    assert cfg["variant"] == "base"
    assert cfg["variant_cfg"]["d_model"] == 128  # TINY_VARIANTS["base"]
    assert cfg["lora"]["rank"] == 4
    assert cfg["lora"]["targets"] == ["q", "v"]
    rows = [
        json.loads(line)
        for line in (rd / "metrics.jsonl").read_text().splitlines()
    ]
    # 50 steps / k=10 = 5 chunks; --val-every 2 + final = 3 val rows.
    assert len(rows) == 5
    val_rows = [r for r in rows if r["val_loss"] is not None]
    assert len(val_rows) >= 2
    for r in rows:
        assert math.isfinite(r["train_loss_mean"])
        assert math.isfinite(r["grad_norm_mean"])
        if r["val_loss"] is not None:
            assert math.isfinite(r["val_loss"])


def test_validation_failures_do_not_create_run_dir(tmp_path: Path) -> None:
    """No orphan ``jax_adapter_run_*`` directory should be created
    for any documented validation failure."""
    cases = [
        # k = 0
        ["--supernet", "tiny", "--variant", "base", "--rank", "4",
         "--total-steps", "10", "--k", "0",
         "--batch-size", "2", "--seq-len", "16",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # batch-size 0
        ["--supernet", "tiny", "--variant", "base", "--rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "0", "--seq-len", "16",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # seq-len exceeds max
        ["--supernet", "tiny", "--variant", "base", "--rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "2", "--seq-len", "9999",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # unknown variant
        ["--supernet", "tiny", "--variant", "huge", "--rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "2", "--seq-len", "16",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # bad LR schedule (warmup == total)
        ["--supernet", "tiny", "--variant", "base", "--rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "2", "--seq-len", "16",
         "--warmup-steps", "10", "--val-frac", "0.1"],
    ]
    for args in cases:
        with pytest.raises(SystemExit):
            _run(args, tmp_path)
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, f"validation failures leaked run dirs: {leaked}"
