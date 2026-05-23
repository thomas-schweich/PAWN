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
    "--lora-rank", "4",
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
                "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
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
                "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
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
                "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
                "--total-steps", "7", "--k", "3",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )


def test_rejects_rank_zero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--lora-rank"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--lora-rank", "0",
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
                "--supernet", "tiny", "--variant", "huge", "--lora-rank", "4",
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
                    "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
                    "--total-steps", "10", "--k", "5",
                    "--batch-size", "2", "--seq-len", "16",
                    "--warmup-steps", "1", "--val-frac", v,
                ],
                tmp_path,
            )


def test_rejects_val_every_zero(tmp_path: Path) -> None:
    """``--val-every 0`` is used as a modulo divisor inside the chunk
    loop; without an upfront guard it would crash with
    ZeroDivisionError after corpus generation + JIT trace had paid
    their cost (Codex round 5 P3). The guard must fire before any
    filesystem side effect so no orphan run dir lands on disk."""
    with pytest.raises(SystemExit, match="--val-every"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
                "--val-every", "0",
            ],
            tmp_path,
        )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, (
        f"--val-every=0 validation leaked a run directory: {leaked}"
    )


def test_rejects_seq_len_exceeding_max(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="seq-len"):
        _run(
            [
                "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
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
                "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
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
            "--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
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
    # MetricsLogger schema: a ``type: "config"`` header, one
    # ``type: "train"`` row per chunk (50 / k=10 = 5 chunks), and a
    # SEPARATE ``type: "val"`` row per validation point (§8.5 — val
    # is its own record, not a column on the train row).
    config_rows = [r for r in rows if r["type"] == "config"]
    train_rows = [r for r in rows if r["type"] == "train"]
    val_rows = [r for r in rows if r["type"] == "val"]
    assert len(config_rows) == 1
    assert len(train_rows) == 5
    assert len(val_rows) >= 2
    for r in train_rows:
        assert math.isfinite(r["train_loss_mean"])
        assert math.isfinite(r["grad_norm_mean"])
    for r in val_rows:
        assert math.isfinite(r["val_loss"])


_STRATEGY_EXTRA_ARGS: dict[str, list[str]] = {
    "lora": ["--lora-rank", "4"],
    "film": [],
    "unfreeze": ["--n-unfreeze", "1"],
    "bottleneck": ["--bottleneck-dim", "8"],
    "hybrid": ["--lora-rank", "4"],
    "sparse": ["--density", "0.1"],
    "rosa": ["--lora-rank", "4"],
    "specialized_clm": [
        "--d-model", "64",
        "--n-layers", "2",
        "--n-heads", "2",
        "--d-ff", "128",
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
    ``--d-model`` / ``--n-layers`` / ``--n-heads`` / ``--d-ff``
    hyperparams (§8.3 — no ``specialized_`` prefix)."""
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
    # At least one ``type: "train"`` row + the ``type: "config"``
    # header. (val rows depend on --val-every; not asserted here.)
    rows = [
        json.loads(line)
        for line in (runs[0] / "metrics.jsonl").read_text().splitlines()
    ]
    train_rows = [r for r in rows if r["type"] == "train"]
    assert len(train_rows) >= 1
    assert any(r["type"] == "config" for r in rows)
    for r in train_rows:
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
      * `metrics.jsonl` has ``type: "train"`` rows for both Phase 1
        and Phase 3.
      * The ``step`` column is monotonically non-decreasing across
        the phase boundary (the round-2 fix preserves
        ``state.step`` across the Phase 2 → 3 re-init; without it
        Phase 3 logs jump backwards to step 0).
      * The final train row's ``step`` equals ``--total-steps`` —
        the run accounted for every training step.
    """
    _run(
        [
            "--strategy", "rosa",
            "--supernet", "tiny", "--variant", "small",
            "--lora-rank", "4",
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
    # 4 chunks total at k=5: 2 Phase-1 + 2 Phase-3 ``type: "train"``
    # rows (plus the config header + per-chunk val rows).
    train_rows = [r for r in rows if r["type"] == "train"]
    assert len(train_rows) == 4
    # Monotonic ``step`` across the phase boundary. Each train row's
    # ``step`` >= the previous train row's. The round-2 fix
    # (state._replace(step=phase1_step)) is what keeps this true
    # past chunk_i == rosa_warmup_chunks.
    step_ends = [r["step"] for r in train_rows]
    for i in range(1, len(step_ends)):
        assert step_ends[i] >= step_ends[i - 1], (
            f"non-monotonic step at train row {i}: "
            f"{step_ends[i - 1]} → {step_ends[i]}"
        )
    # Final train row accounts for every training step requested.
    assert step_ends[-1] == 20, (
        f"final step={step_ends[-1]} != --total-steps=20"
    )


def test_rosa_zero_warmup_runs_single_phase(tmp_path: Path) -> None:
    """``--rosa-warmup-frac 0`` skips Phase 1 + Phase 2 entirely and
    trains jointly from step 0 — matches the C.1 dispatch behaviour
    pre-three-phase. Useful for runs that want to compare against a
    pure-joint baseline."""
    _run(
        [
            "--strategy", "rosa",
            "--supernet", "tiny", "--variant", "small",
            "--lora-rank", "4",
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
                "--lora-rank", "4",
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


def test_film_default_preserves_output_modulation(tmp_path: Path) -> None:
    """Default `--strategy film` and `--strategy hybrid` invocations
    (no `--no-use-output-film` flag) preserve the v1-canonical
    default: ``FiLMConfig.use_output_film == True`` (§8.3 — the v2
    ``--no-film-output`` polarity flip was reverted; the CLI now
    uses a ``BooleanOptionalAction --use-output-film`` defaulting
    True)."""
    for strat in ("film", "hybrid"):
        _run(
            [
                "--strategy", strat,
                "--supernet", "tiny", "--variant", "base",
                "--lora-rank", "4",
                "--total-steps", "5", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.5",
                "--val-every", "1", "--quiet",
            ],
            tmp_path,
        )
        runs = list(tmp_path.glob("jax_adapter_run_*"))
        # Most recent run is the one we just made.
        latest = max(runs, key=lambda p: p.stat().st_mtime)
        cfg = json.loads((latest / "config.json").read_text())
        # Both film and hybrid records use_output_film / film_output
        # under different keys.
        # FiLM and Hybrid now share the ``use_output_film`` key
        # (round-6 hygiene normalisation).
        assert cfg["strategy_config"]["use_output_film"] is True


def test_film_no_output_opt_out(tmp_path: Path) -> None:
    """The opt-out flag ``--no-use-output-film`` disables output FiLM.
    Pins the user-facing toggle works as §8.3's BooleanOptionalAction
    revert advertises."""
    _run(
        [
            "--strategy", "film",
            "--supernet", "tiny", "--variant", "base",
            "--no-use-output-film",
            "--total-steps", "5", "--k", "5",
            "--batch-size", "2", "--seq-len", "16",
            "--warmup-steps", "1", "--val-frac", "0.5",
            "--val-every", "1", "--quiet",
        ],
        tmp_path,
    )
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1
    cfg = json.loads((runs[0] / "config.json").read_text())
    assert cfg["strategy_config"]["use_output_film"] is False


def test_rejects_no_op_unfreeze_upfront(tmp_path: Path) -> None:
    """``--n-unfreeze 0 --no-include-lm-head`` with default
    ``--include-embeddings=False`` leaves zero trainable parameters
    — the run would burn full compute updating nothing. Pre-fix
    (Codex round-5 P2), the script accepted this and ran. The
    upfront check rejects it with no orphan run-dir."""
    with pytest.raises(SystemExit, match="zero parameters"):
        _run(
            [
                "--strategy", "unfreeze",
                "--supernet", "tiny", "--variant", "base",
                "--n-unfreeze", "0",
                "--no-include-lm-head",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, f"no-op unfreeze validation leaked: {leaked}"


def test_rejects_rosa_top_k_frac_out_of_range(tmp_path: Path) -> None:
    """``--rosa-top-k-frac`` outside (0, 1] is rejected upfront with
    no orphan run-dir. Test-risk round 6 flagged this as the only
    driver-level guard without a pinning test."""
    for bad in ("0", "1.5", "-0.1"):
        with pytest.raises(SystemExit, match="rosa-top-k-frac"):
            _run(
                [
                    "--strategy", "rosa",
                    "--supernet", "tiny", "--variant", "base",
                    "--lora-rank", "4",
                    "--rosa-top-k-frac", bad,
                    "--total-steps", "10", "--k", "5",
                    "--batch-size", "2", "--seq-len", "16",
                    "--warmup-steps", "1", "--val-frac", "0.1",
                ],
                tmp_path,
            )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, f"rosa-top-k-frac validation leaked: {leaked}"


def test_rejects_negative_lora_alpha_upfront(tmp_path: Path) -> None:
    """``--lora-alpha`` <= 0 is invalid for LoRA / Hybrid / RoSA.
    Pre-fix (Codex round-4 P2), this raised inside
    ``LoRAConfig.__post_init__`` only after corpus generation and
    run_dir creation, leaving an orphan dir. The upfront check
    rejects negative / zero alpha and produces no run-dir."""
    for strat in ("lora", "hybrid", "rosa"):
        with pytest.raises(SystemExit, match="lora-alpha"):
            _run(
                [
                    "--strategy", strat,
                    "--supernet", "tiny", "--variant", "base",
                    "--lora-rank", "4", "--lora-alpha", "-1",
                    "--total-steps", "10", "--k", "5",
                    "--batch-size", "2", "--seq-len", "16",
                    "--warmup-steps", "1", "--val-frac", "0.1",
                ],
                tmp_path,
            )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, f"lora-alpha validation leaked: {leaked}"


def test_rejects_bottleneck_no_attn_no_ffn_upfront(tmp_path: Path) -> None:
    """``--no-adapt-attn --no-adapt-ffn`` together is a no-op adapter
    and rejected upfront, before corpus generation + run_dir
    creation. (§8.3: v1-canonical flag names — the v2
    ``--bottleneck-no-*`` prefix was reverted.)"""
    with pytest.raises(SystemExit, match="--no-adapt-attn and --no-adapt-ffn"):
        _run(
            [
                "--strategy", "bottleneck",
                "--supernet", "tiny", "--variant", "base",
                "--bottleneck-dim", "8",
                "--no-adapt-attn", "--no-adapt-ffn",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, f"bottleneck no-attn/no-ffn validation leaked: {leaked}"


def test_rejects_specialized_clm_odd_head_dim_upfront(tmp_path: Path) -> None:
    """specialized_clm with odd ``head_dim = d_model / n_heads`` is
    rejected upfront. Pre-fix (Codex round-4 P2), this raised inside
    ``ModelConfig.__post_init__`` after corpus generation +
    run_dir creation. ``d_model=65, n_heads=2`` is the smallest
    case: 65 isn't divisible by 2 at all, but ``d_model=70,
    n_heads=2`` gives ``head_dim=35`` which IS odd."""
    # d_model=70, n_heads=2 → head_dim=35 (odd, RoPE-incompatible).
    with pytest.raises(SystemExit, match="head_dim"):
        _run(
            [
                "--strategy", "specialized_clm",
                "--supernet", "tiny", "--variant", "base",
                "--d-model", "70",
                "--n-heads", "2",
                "--n-layers", "2",
                "--d-ff", "128",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "2", "--seq-len", "16",
                "--warmup-steps", "1", "--val-frac", "0.1",
            ],
            tmp_path,
        )
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, f"specialized_clm head_dim validation leaked: {leaked}"


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
                "--density", "1.5",
            ],
            tmp_path,
        )


def test_validation_failures_do_not_create_run_dir(tmp_path: Path) -> None:
    """No orphan ``jax_adapter_run_*`` directory should be created
    for any documented validation failure."""
    cases = [
        # k = 0
        ["--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
         "--total-steps", "10", "--k", "0",
         "--batch-size", "2", "--seq-len", "16",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # batch-size 0
        ["--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "0", "--seq-len", "16",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # seq-len exceeds max
        ["--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "2", "--seq-len", "9999",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # unknown variant
        ["--supernet", "tiny", "--variant", "huge", "--lora-rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "2", "--seq-len", "16",
         "--warmup-steps", "1", "--val-frac", "0.1"],
        # bad LR schedule (warmup == total)
        ["--supernet", "tiny", "--variant", "base", "--lora-rank", "4",
         "--total-steps", "10", "--k", "5",
         "--batch-size", "2", "--seq-len", "16",
         "--warmup-steps", "10", "--val-frac", "0.1"],
    ]
    for args in cases:
        with pytest.raises(SystemExit):
            _run(args, tmp_path)
    leaked = list(tmp_path.glob("jax_adapter_run_*"))
    assert not leaked, f"validation failures leaked run dirs: {leaked}"


# ---------------------------------------------------------------------------
# Lichess data path (--pgn) — the realistic adapter task
# ---------------------------------------------------------------------------


def _write_lichess_parquet(path: Path, n_games: int, plies: int) -> None:
    """Write a synthetic Lichess-schema parquet for the --pgn path."""
    import polars as pl
    from pawn.config import OUTCOME_TOKEN_BASE

    rows = [
        {
            "tokens": [((g + p) % 1900) for p in range(plies)],
            "game_length": plies,
            "outcome_token": OUTCOME_TOKEN_BASE + (g % 5),
            "white_elo": 1800 + (g % 50),
            "black_elo": 1800 + ((g * 3) % 50),
        }
        for g in range(n_games)
    ]
    pl.DataFrame(
        rows,
        schema={
            "tokens": pl.List(pl.Int32),
            "game_length": pl.Int32,
            "outcome_token": pl.Int32,
            "white_elo": pl.Int32,
            "black_elo": pl.Int32,
        },
    ).write_parquet(path)


def test_lichess_pgn_path_trains_and_tiles(tmp_path: Path) -> None:
    """``--pgn`` loads a finite Elo-filtered Lichess slice and tiles it
    across epochs to fill ``total_steps * batch_size`` game-slots — the
    realistic adapter task. Pins that the slice is loaded, the cache
    is written, and training completes through MetricsLogger."""
    pq = tmp_path / "lichess.parquet"
    # 24 games × 20 plies. With total_steps=10 k=5 batch_size=2 the
    # trainer needs 20 train-game-slots; the ~21-game train pool
    # (after the val split) is tiled to fill them.
    _write_lichess_parquet(pq, n_games=24, plies=20)

    _run(
        [
            "--strategy", "lora",
            "--supernet", "tiny", "--variant", "base",
            "--lora-rank", "4",
            "--total-steps", "10", "--k", "5",
            "--batch-size", "2", "--seq-len", "24",
            "--warmup-steps", "1",
            "--val-frac", "0.2", "--val-every", "1", "--quiet",
            "--pgn", str(pq), "--pgn-val-split", "",  # single file → carve
            "--elo-min", "1800", "--elo-max", "2000", "--min-ply", "10",
            "--cache-dir", str(tmp_path / "lcache"),
        ],
        tmp_path,
    )
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1
    cfg = json.loads((runs[0] / "config.json").read_text())
    # config.json records the data-source provenance.
    assert cfg["data_source"] == "lichess"
    assert cfg["elo_min"] == 1800
    assert cfg["elo_max"] == 2000
    assert cfg["pgn_val_split"] == ""
    # The tokenized-Lichess cache was written (one entry — single
    # split loaded since --pgn-val-split="" carves from train).
    cache_entries = [p for p in (tmp_path / "lcache").iterdir() if p.is_dir()]
    assert len(cache_entries) == 1
    assert (cache_entries[0] / ".complete").exists()
    # Training produced train + val rows through MetricsLogger.
    rows = [
        json.loads(line)
        for line in (runs[0] / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(r["type"] == "train" for r in rows) == 2  # 10 steps / k=5
    assert any(r["type"] == "val" for r in rows)
    for r in rows:
        if r["type"] == "train":
            assert math.isfinite(r["train_loss_mean"])


def test_lichess_pgn_path_uses_held_out_validation_split(tmp_path: Path) -> None:
    """When the --pgn source is a directory with split-prefixed
    parquets (the HF ``data/{split}-*.parquet`` layout), the default
    ``--pgn-val-split=validation`` reads val from the held-out shards
    rather than carving from train. Critical: the realistic adapter
    benchmark requires no leakage between train and val."""
    pq_dir = tmp_path / "lichess"
    pq_dir.mkdir()
    _write_lichess_parquet(
        pq_dir / "train-00000-of-00001.parquet", n_games=30, plies=20
    )
    # Build the held-out split by hand — distinct outcome offset so we
    # can verify it isn't accidentally sampled from train.
    import polars as pl
    from pawn.config import OUTCOME_TOKEN_BASE

    pl.DataFrame(
        [
            {
                "tokens": [((g + p) % 1900) for p in range(20)],
                "game_length": 20,
                "outcome_token": OUTCOME_TOKEN_BASE + 2,  # distinct
                "white_elo": 1850,
                "black_elo": 1850,
            }
            for g in range(8)
        ],
        schema={
            "tokens": pl.List(pl.Int32),
            "game_length": pl.Int32,
            "outcome_token": pl.Int32,
            "white_elo": pl.Int32,
            "black_elo": pl.Int32,
        },
    ).write_parquet(pq_dir / "validation-00000-of-00001.parquet")

    _run(
        [
            "--strategy", "lora",
            "--supernet", "tiny", "--variant", "base",
            "--lora-rank", "4",
            "--total-steps", "10", "--k", "5",
            "--batch-size", "2", "--seq-len", "24",
            "--warmup-steps", "1",
            "--val-frac", "0.2", "--val-every", "1", "--quiet",
            "--pgn", str(pq_dir),
            "--pgn-split", "train", "--pgn-val-split", "validation",
            "--elo-min", "1800", "--elo-max", "2000", "--min-ply", "10",
            "--cache-dir", str(tmp_path / "lcache"),
        ],
        tmp_path,
    )
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1
    cfg = json.loads((runs[0] / "config.json").read_text())
    assert cfg["pgn_split"] == "train"
    assert cfg["pgn_val_split"] == "validation"
    # Two cache entries — one per loaded split.
    cache_entries = [p for p in (tmp_path / "lcache").iterdir() if p.is_dir()]
    assert len(cache_entries) == 2


def test_lichess_pgn_path_carve_from_train_when_val_split_empty(
    tmp_path: Path,
) -> None:
    """``--pgn-val-split ""`` opts out of the held-out split and
    carves val out of train — for single-file sources with no split
    structure."""
    pq = tmp_path / "lichess.parquet"
    _write_lichess_parquet(pq, n_games=30, plies=20)
    _run(
        [
            "--strategy", "lora",
            "--supernet", "tiny", "--variant", "base",
            "--lora-rank", "4",
            "--total-steps", "10", "--k", "5",
            "--batch-size", "2", "--seq-len", "24",
            "--warmup-steps", "1",
            "--val-frac", "0.2", "--val-every", "1", "--quiet",
            "--pgn", str(pq),
            "--pgn-val-split", "",       # carve from train
            "--elo-min", "1800", "--elo-max", "2000", "--min-ply", "10",
            "--cache-dir", str(tmp_path / "lcache"),
        ],
        tmp_path,
    )
    runs = list(tmp_path.glob("jax_adapter_run_*"))
    assert len(runs) == 1
    cfg = json.loads((runs[0] / "config.json").read_text())
    assert cfg["pgn_val_split"] == ""


def test_lichess_pgn_path_rejects_oversmall_slice(tmp_path: Path) -> None:
    """A Lichess slice too small to yield even one val batch fails
    upfront with an actionable message, no orphan run dir."""
    pq = tmp_path / "lichess.parquet"
    _write_lichess_parquet(pq, n_games=3, plies=20)
    with pytest.raises(SystemExit, match="val"):
        _run(
            [
                "--strategy", "lora",
                "--supernet", "tiny", "--variant", "base",
                "--lora-rank", "4",
                "--total-steps", "10", "--k", "5",
                "--batch-size", "8", "--seq-len", "24",
                "--warmup-steps", "1",
                "--val-frac", "0.1", "--val-every", "1", "--quiet",
                "--pgn", str(pq), "--pgn-val-split", "",  # carve
                "--cache-dir", str(tmp_path / "lcache"),
            ],
            tmp_path,
        )
    assert not list(tmp_path.glob("jax_adapter_run_*"))
