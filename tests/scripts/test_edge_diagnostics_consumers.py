"""End-to-end schema agreement for the edge-case diagnostics consumers.

The v2 edge-case writer
(:func:`pawn.eval_suite.diagnostics.compute_edge_case_diagnostics_quota`,
surfaced by ``scripts/eval_generation_jax.py`` under the ``edge_cases``
key) and the two consumers — ``scripts/generate_model_cards.py`` and
:func:`pawn.eval_suite.viz.plot_diagnostic_results` — must agree on the
metric schema (v1's sampled keys) and the v2 label names (the castle
labels were spelled out from ``castle_legal_k``/``_q``). These tests pin
that agreement so a future rename or schema drift fails loudly instead of
KeyError-ing at card-generation time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_script(name: str):  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"scripts_{name}_edge", Path("scripts") / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_edge_cases() -> dict[str, dict[str, object]]:
    """A v2 ``edge_cases`` block in the exact shape
    ``eval_generation_jax`` emits: per-label sampled metrics + accuracy,
    keyed by the v2 :data:`EDGE_CASE_LABELS` labels."""
    from pawn.eval_suite.diagnostics import EDGE_CASE_LABELS, TERMINAL_LABELS

    out: dict[str, dict[str, object]] = {}
    for label in EDGE_CASE_LABELS:
        out[label] = {
            "n_positions": 7,
            "terminal": label in TERMINAL_LABELS,
            "mean_legal_rate": 0.42,
            "std_legal_rate": 0.1,
            "mean_pad_prob": 0.8,
            "mean_entropy": 1.5,
            "std_entropy": 0.3,
            "accuracy": 0.5,
        }
    return out


def test_model_card_diagnostic_names_match_v2_labels() -> None:
    """generate_model_cards.DIAGNOSTIC_NAMES keys are exactly the v2
    edge-case labels (castle labels spelled out, not v1's `_k`/`_q`)."""
    from pawn.eval_suite.diagnostics import EDGE_CASE_LABELS

    mod = _load_script("generate_model_cards")
    assert set(mod.DIAGNOSTIC_NAMES) == set(EDGE_CASE_LABELS)
    # The renamed castle labels are present under their v2 spelling.
    assert "castle_legal_kingside" in mod.DIAGNOSTIC_NAMES
    assert "castle_legal_queenside" in mod.DIAGNOSTIC_NAMES
    assert "castle_legal_k" not in mod.DIAGNOSTIC_NAMES


def test_model_card_format_diagnostic_consumes_v2_edge_output() -> None:
    """`format_diagnostic` reads the v2 edge_cases schema without KeyError
    for every label — terminal labels report pad-prob, the rest legal
    rate."""
    mod = _load_script("generate_model_cards")
    eval_results = {"diagnostics": _fake_edge_cases()}
    for label in mod.DIAGNOSTIC_NAMES:
        n, value = mod.format_diagnostic(eval_results, label)
        assert n == "7"
        if label in ("checkmate", "stalemate"):
            assert value == "80.0%"  # mean_pad_prob
        else:
            assert value == "42.0%"  # mean_legal_rate


def test_viz_plot_diagnostic_results_consumes_v2_edge_output() -> None:
    """The viz helper reads mean_legal_rate / mean_pad_prob / mean_entropy
    off the same per-label dict the writer emits — no KeyError."""
    import matplotlib

    matplotlib.use("Agg")
    from pawn.eval_suite.viz import plot_diagnostic_results

    fig = plot_diagnostic_results(_fake_edge_cases())
    assert fig is not None


def test_run_evals_backbone_wires_edge_cases_into_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_evals_backbone --edge-cases` forwards the flag to
    eval_generation_jax and surfaces the emitted `edge_cases` block under
    the top-level `diagnostics` key the model card reads."""
    mod = _load_script("run_evals_backbone")
    edge = _fake_edge_cases()

    seen_args: list[list[str]] = []

    def fake_run(cmd, check=False):  # type: ignore[no-untyped-def]
        seen_args.append(list(cmd))
        # Identify the output path for the script being invoked and write a
        # minimal artifact so the orchestrator's read-back succeeds.
        out_idx = cmd.index("--output")
        out_path = Path(cmd[out_idx + 1])
        if "eval_generation_jax.py" in cmd[1]:
            out_path.write_text(json.dumps({"edge_cases": edge}))
        else:
            out_path.write_text(json.dumps({}))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc = mod.main([
        "--checkpoints", "fake/ckpt",
        "--output-dir", str(tmp_path),
        "--edge-cases", "--edge-per-label", "3",
    ])
    assert rc == 0

    # The generation invocation carried the edge flags through.
    gen_cmd = next(c for c in seen_args if "eval_generation_jax.py" in c[1])
    assert "--edge-cases" in gen_cmd
    assert "--edge-per-label" in gen_cmd
    assert gen_cmd[gen_cmd.index("--edge-per-label") + 1] == "3"

    # eval_results.json carries the edge block under `diagnostics`.
    eval_results = json.loads(
        (tmp_path / "fake_ckpt" / "eval_results.json").read_text()
    )
    assert "diagnostics" in eval_results
    assert set(eval_results["diagnostics"]) == set(edge)


def test_run_evals_backbone_omits_diagnostics_without_edge_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --edge-cases the generation script is not asked for edge
    data and the top-level `diagnostics` key stays absent."""
    mod = _load_script("run_evals_backbone")

    def fake_run(cmd, check=False):  # type: ignore[no-untyped-def]
        out_idx = cmd.index("--output")
        Path(cmd[out_idx + 1]).write_text(json.dumps({}))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rc = mod.main([
        "--checkpoints", "fake/ckpt",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    eval_results = json.loads(
        (tmp_path / "fake_ckpt" / "eval_results.json").read_text()
    )
    assert "diagnostics" not in eval_results
