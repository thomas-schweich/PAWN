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
    # Config carries full per-variant ModelConfig dicts. The
    # multi-strategy dispatch refactor moved per-strategy hyperparams
    # under ``strategy_config`` keyed by ``strategy``.
    assert cfg["variant"] == "base"
    assert cfg["variant_cfg"]["d_model"] == 128  # TINY_VARIANTS["base"]
    assert cfg["strategy"] == "lora"
    assert cfg["strategy_config"]["rank"] == 4
    assert cfg["strategy_config"]["targets"] == ["q", "v"]
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


_STRATEGY_EXTRA_ARGS: dict[str, list[str]] = {
    "lora": ["--rank", "4"],
    "film": [],
    "unfreeze": ["--n-unfreeze", "1"],
    "bottleneck": ["--bottleneck-dim", "8"],
    "hybrid": ["--rank", "4"],
    "sparse": ["--sparse-density", "0.1"],
    "rosa": ["--rank", "4"],
    "specialized_clm": [
        "--specialized-d-model", "64",
        "--specialized-n-layers", "2",
        "--specialized-n-heads", "2",
        "--specialized-d-ff", "128",
    ],
}


@pytest.mark.parametrize("strategy", list(_STRATEGY_EXTRA_ARGS))
def test_each_strategy_dispatch_runs(strategy: str, tmp_path: Path) -> None:
    """Smoke-test the per-strategy dispatch path. Each strategy
    builds its own model + adapter_filter + (optional) gradient mask
    and runs through ``init_adapter_state`` →
    ``make_adapter_train_step`` → 10 training steps + 1 val pass
    without raising. The 8th strategy (``specialized_clm``) is the
    only one that doesn't slice a supernet — its dispatch ignores
    ``--variant`` and builds a from-scratch model from the
    ``--specialized-*`` hyperparams."""
    args = [
        "--strategy", strategy,
        "--supernet", "tiny", "--variant", "base",
        "--total-steps", "10", "--k", "5",
        "--batch-size", "2", "--seq-len", "16",
        "--warmup-steps", "1", "--val-frac", "0.1",
        "--val-every", "1", "--quiet",
        *_STRATEGY_EXTRA_ARGS[strategy],
    ]
    _run(args, tmp_path)
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1
    cfg = json.loads((runs[0] / "config.json").read_text())
    assert cfg["strategy"] == strategy
    if strategy == "specialized_clm":
        # specialized_clm doesn't use the supernet/variant pipeline.
        assert cfg["variant"] == "from-scratch"
        assert cfg["variant_cfg"] is None
    else:
        assert cfg["variant"] == "base"
    # At least one training row + one val row.
    rows = [
        json.loads(line)
        for line in (runs[0] / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    for r in rows:
        assert math.isfinite(r["train_loss_mean"])
        assert math.isfinite(r["grad_norm_mean"])


def test_rosa_three_phase_writes_transition_log_and_completes(
    tmp_path: Path, capfd: pytest.CaptureFixture[str],
) -> None:
    """RoSA dispatch with --rosa-warmup-frac > 0 runs the three-phase
    schedule (LoRA warmup → mask gen → joint training) end-to-end.

    Pins:
      * The Phase 2 → 3 transition log line appears once (announces
        active-entry count + targets).
      * The training run completes without raising.
      * `metrics.jsonl` has rows for both Phase 1 and Phase 3.
    """
    _run(
        [
            "--strategy", "rosa",
            "--supernet", "tiny", "--variant", "small",
            "--rank", "4",
            "--total-steps", "20", "--k", "5",
            "--batch-size", "2", "--seq-len", "16",
            "--warmup-steps", "2", "--val-frac", "0.25",
            "--val-every", "1",
            "--rosa-warmup-frac", "0.5",
            "--rosa-top-k-frac", "0.1",
        ],
        tmp_path,
    )
    out, _err = capfd.readouterr()
    # The transition log is exactly one line for a single transition.
    assert out.count("[rosa] Phase 2 → 3 transition") == 1
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1
    rows = [
        json.loads(line)
        for line in (runs[0] / "metrics.jsonl").read_text().splitlines()
    ]
    # 4 chunks total at k=5: 2 Phase-1 + 2 Phase-3.
    assert len(rows) == 4


def test_rosa_zero_warmup_runs_single_phase(tmp_path: Path) -> None:
    """``--rosa-warmup-frac 0`` skips Phase 1 + Phase 2 entirely and
    trains jointly from step 0 — matches the C.1 dispatch behaviour
    pre-three-phase. Useful for runs that want to compare against a
    pure-joint baseline."""
    _run(
        [
            "--strategy", "rosa",
            "--supernet", "tiny", "--variant", "small",
            "--rank", "4",
            "--total-steps", "10", "--k", "5",
            "--batch-size", "2", "--seq-len", "16",
            "--warmup-steps", "1", "--val-frac", "0.5",
            "--val-every", "1",
            "--rosa-warmup-frac", "0",
        ],
        tmp_path,
    )
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1


def test_rosa_warmup_frac_too_large_rejected(tmp_path: Path) -> None:
    """``--rosa-warmup-frac`` that leaves no Phase-3 chunks is a
    user-error and surfaced upfront with a SystemExit so the user
    knows before any compute spins up *and* before any filesystem
    side-effect (no orphan ``jax_adapter_run_*`` directory).

    Codex round-2 P3: pre-fix the check fired only after
    ``run_dir.mkdir`` + ``config.json`` write, leaking an orphan
    directory on validation failure — unlike every other validation
    path."""
    with pytest.raises(SystemExit, match="Phase 3"):
        _run(
            [
                "--strategy", "rosa",
                "--supernet", "tiny", "--variant", "small",
                "--rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.5",
                "--rosa-warmup-frac", "0.95",  # > 2/2 chunks → no joint chunks
            ],
            tmp_path,
        )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, (
        f"RoSA warmup-frac validation leaked run dirs: {leaked}"
    )


def test_rejects_n_unfreeze_exceeding_n_layers_upfront(tmp_path: Path) -> None:
    """``--n-unfreeze > variant.n_layers`` is rejected upfront, before
    corpus generation and run-dir creation. Pre-fix (Codex round-3 P2),
    this fired only inside ``init_unfreeze_model`` after the corpus
    had been generated and the run dir was written.

    TINY_VARIANTS["base"].n_layers is 3, so --n-unfreeze 999 is
    invalid and must fail fast.
    """
    with pytest.raises(SystemExit, match="n-unfreeze=999"):
        _run(
            [
                "--strategy", "unfreeze",
                "--supernet", "tiny", "--variant", "base",
                "--n-unfreeze", "999",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, (
        f"upfront --n-unfreeze validation leaked run dirs: {leaked}"
    )


def test_rejects_bottleneck_dim_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="bottleneck-dim"):
        _run(
            [
                "--strategy", "bottleneck",
                "--supernet", "tiny", "--variant", "base",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
                "--bottleneck-dim", "0",
            ],
            tmp_path,
        )


def test_rejects_sparse_density_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="sparse-density"):
        _run(
            [
                "--strategy", "sparse",
                "--supernet", "tiny", "--variant", "base",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
                "--sparse-density", "1.5",
            ],
            tmp_path,
        )


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
