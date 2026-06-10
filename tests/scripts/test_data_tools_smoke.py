"""Smoke + behavioral tests for the data-tool scripts.

Covers the three "unchanged surface" data tools from plan §S13:

- ``scripts/generate_model_cards.py`` — v2 config-schema migration
  (``model`` / ``run`` blocks, ``step`` in training_state, v2 ``val/*``
  metric keys), v2 ``-v2`` repo targeting, and structural variant
  auto-detection from ``config.json`` (never hardcoded per-repo).
- ``scripts/compute_theoretical_ceiling.py`` — the random-game E[1/N_legal]
  ceiling computed through the Rust engine.
- ``scripts/extract_lichess_parquet.py`` — work-unit sentinel + partition
  logic.

These run on CPU and need no network: the model-card HF fetches are
redirected to local files via a monkeypatched ``hf_hub_download``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pawn.config import VARIANTS


def _load_script(name: str) -> ModuleType:
    mod_name = f"scripts_{name}_dt"
    spec = importlib.util.spec_from_file_location(
        mod_name, Path("scripts") / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses defined in the module can resolve
    # their `__module__` back to this object (extract_lichess_parquet uses
    # `@dataclass` with string annotations; the stdlib looks the class's
    # module up in `sys.modules`).
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


DATA_TOOLS = (
    "generate_model_cards",
    "compute_theoretical_ceiling",
    "extract_lichess_parquet",
)


@pytest.mark.parametrize("script_name", DATA_TOOLS)
def test_data_tool_imports_and_has_main(script_name: str) -> None:
    """Each data-tool script imports cleanly and exposes ``main``."""
    mod = _load_script(script_name)
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# generate_model_cards: v2 config schema + auto-detect + v2 repos
# ---------------------------------------------------------------------------


def test_model_cards_targets_v2_repos() -> None:
    """The card generator points at the republished ``-v2`` repos, not the
    frozen v1 PyTorch artifacts (parity item ``generate-model-cards-v1-repo
    -hardcode``)."""
    mod = _load_script("generate_model_cards")
    for key, meta in mod.VARIANT_REPOS.items():
        assert meta["repo"].endswith("-v2"), (
            f"variant {key} must target a -v2 repo, got {meta['repo']}"
        )


def test_detect_variant_is_structural_not_hardcoded() -> None:
    """Variant identity is derived from the checkpoint's ``d_model`` /
    ``n_heads`` — matching :data:`pawn.config.VARIANTS` — never assumed from
    the repo it came from (model-card auto-detect convention)."""
    mod = _load_script("generate_model_cards")
    for key, cfg in VARIANTS.items():
        block = {"d_model": cfg.d_model, "n_heads": cfg.n_heads}
        assert mod.detect_variant(block) == key

    # A size that matches no production variant is a loud error, not a
    # silently-wrong card.
    with pytest.raises(ValueError, match="does not match any known variant"):
        mod.detect_variant({"d_model": 123, "n_heads": 7})


def _write_v2_repo_files(
    repo_dir: Path,
    *,
    variant_cfg: Any,
    step: int,
    conditioning: list[str],
) -> None:
    """Materialise the four files the card generator fetches, in the v2
    layout (``model`` / ``run`` config blocks, ``step`` in training_state,
    namespaced ``val/*`` metric keys)."""
    repo_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "mask_version": 1,
        "model": {
            "d_model": variant_cfg.d_model,
            "n_layers": variant_cfg.n_layers,
            "n_heads": variant_cfg.n_heads,
            "d_ff": variant_cfg.d_ff,
            "head_dim": variant_cfg.head_dim,
            "vocab_size": variant_cfg.vocab_size,
            "max_seq_len": variant_cfg.max_seq_len,
            "tie_embeddings": variant_cfg.tie_embeddings,
        },
        "run": {
            "total_steps": 100_000,
            "batch_size": 256,
            "lr": 3e-4,
            "weight_decay": 0.01,
            # fractional warmup (the int override is None) — the card must
            # resolve the effective step count, not crash on None.
            "warmup_steps": None,
            "warmup_frac": 0.05,
            "conditioning": conditioning,
        },
    }
    (repo_dir / "config.json").write_text(json.dumps(config))

    (repo_dir / "training_state.json").write_text(json.dumps({"step": step}))

    val_record = {
        "type": "val",
        "step": step,
        "val/loss": 1.5,
        "val/accuracy": 0.42,
        "val/top5_accuracy": 0.80,
        "val/perplexity": 4.48,
        "val/legal_move_rate": 0.99,
        "val/late_legal_move_rate": 0.97,
    }
    lines = [
        json.dumps({"type": "train", "step": step - 100, "train/loss": 1.7}),
        json.dumps(val_record),
    ]
    (repo_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")

    eval_results = {
        "probes": {
            "side_to_move": {"layer_0": {"accuracy": 0.95, "best_accuracy": 0.97}},
            "legal_move_count": {
                "layer_0": {"accuracy": 0.5, "best_accuracy": 0.5, "mae": 1.2}
            },
        },
        "diagnostics": {
            "in_check": {"n_positions": 1234, "mean_legal_rate": 0.98},
            "castle_legal_kingside": {"n_positions": 567, "mean_legal_rate": 0.91},
            "checkmate": {"n_positions": 89, "mean_pad_prob": 0.75},
        },
    }
    (repo_dir / "eval_results.json").write_text(json.dumps(eval_results))


def test_build_context_reads_v2_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end ``build_context`` against the v2 ``config.json`` layout.

    This is the behavioral guard for the config-schema migration: v1 read
    ``config["model_config"]`` / ``config["training_config"]`` and
    ``training_state["global_step"]`` — all absent in v2, so the v1 script
    KeyErrors on a real v2 checkpoint. The v2 script must read ``model`` /
    ``run`` / ``step`` and the namespaced ``val/*`` metric keys, and infer
    the variant structurally.
    """
    mod = _load_script("generate_model_cards")
    base_cfg = VARIANTS["base"]
    repo_dir = tmp_path / "pawn-base-v2"
    _write_v2_repo_files(
        repo_dir, variant_cfg=base_cfg, step=65_000, conditioning=["outcome"]
    )

    def fake_hf_hub_download(repo: str, filename: str, revision: Any = None) -> str:
        return str(repo_dir / filename)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download, raising=True
    )
    # Param count needs the real safetensors blob; stub it (covered by the
    # checkpoint round-trip tests, not the card schema).
    monkeypatch.setattr(
        mod, "count_params_from_weights", lambda repo, revision=None: 35_000_000
    )
    # Point the ceiling artifact at a tmp copy so the test doesn't depend on
    # the repo-tracked one.
    ceiling = {
        "unconditional_ceiling": 0.30,
        "ceiling_ci_low_95": 0.29,
        "ceiling_ci_high_95": 0.31,
        "mean_n_legal": 30.0,
        "n_games": 20_000,
        "n_positions": 1_000_000,
        "max_ply": 512,
    }
    ceiling_path = tmp_path / "ceiling.json"
    ceiling_path.write_text(json.dumps(ceiling))
    monkeypatch.setattr(mod, "CEILING_PATH", ceiling_path)

    meta = mod.VARIANT_REPOS["base"]
    ctx = mod.build_context(meta)

    # Variant auto-detected from the model block, not the repo name.
    assert ctx["variant_key"] == "base"
    # Model block (was config["model_config"]).
    assert ctx["d_model"] == base_cfg.d_model
    assert ctx["n_layers"] == base_cfg.n_layers
    assert ctx["n_heads"] == base_cfg.n_heads
    assert ctx["vocab_size"] == base_cfg.vocab_size
    assert ctx["head_dim"] == base_cfg.head_dim
    # Run block (was config["training_config"]); fractional warmup resolved.
    assert ctx["total_steps"] == 100_000
    assert ctx["batch_size"] == 256
    assert ctx["warmup_steps"] == round(0.05 * 100_000)
    # conditioning replaces the v1 prepend_outcome bool.
    assert ctx["conditioning"] == ["outcome"]
    assert ctx["prepend_outcome"] is True
    # training_state["step"] (was "global_step").
    assert ctx["published_step"] == 65_000
    # v2 namespaced val metrics.
    assert ctx["top1"] == pytest.approx(42.0)
    assert ctx["top5"] == pytest.approx(80.0)
    assert ctx["val_loss"] == pytest.approx(1.5)
    assert ctx["legal_rate"] == pytest.approx(99.0)
    # probes + diagnostics carried through (v2 spelled-out castle key).
    probe_names = {p["name"] for p in ctx["probes"]}
    assert "Side to move" in probe_names
    diag_names = {d["name"] for d in ctx["diagnostics"]}
    assert "Castling legal (kingside)" in diag_names


def test_build_context_renders_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The migrated context renders the repo's Jinja template under
    ``StrictUndefined`` — i.e. every key the template needs is present and
    no v1-only key (``completion_rate`` / ``max_ply`` / ``global_step``)
    leaks an undefined into the card."""
    import jinja2

    mod = _load_script("generate_model_cards")
    small_cfg = VARIANTS["small"]
    repo_dir = tmp_path / "pawn-small-v2"
    # conditioning=[] exercises the "no conditioning" template branch.
    _write_v2_repo_files(
        repo_dir, variant_cfg=small_cfg, step=50_000, conditioning=[]
    )

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, revision=None: str(repo_dir / filename),
        raising=True,
    )
    monkeypatch.setattr(
        mod, "count_params_from_weights", lambda repo, revision=None: 9_000_000
    )
    ceiling_path = tmp_path / "ceiling.json"
    ceiling_path.write_text(
        json.dumps(
            {
                "unconditional_ceiling": 0.30,
                "ceiling_ci_low_95": 0.29,
                "ceiling_ci_high_95": 0.31,
                "mean_n_legal": 30.0,
                "n_games": 20_000,
                "n_positions": 1_000_000,
                "max_ply": 512,
            }
        )
    )
    monkeypatch.setattr(mod, "CEILING_PATH", ceiling_path)

    ctx = mod.build_context(mod.VARIANT_REPOS["small"])

    template_path = Path("cards/hf_model_card.md.j2")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template(template_path.name)
    card = template.render(**ctx)  # raises on any undefined key

    assert "PAWN-Small" in card
    assert "pawn-small-v2" in card
    # No v1 leftovers in the rendered card.
    assert "prepend_outcome=True" not in card
    assert "global_step" not in card


# ---------------------------------------------------------------------------
# compute_theoretical_ceiling: real engine ceiling
# ---------------------------------------------------------------------------


def test_compute_ceiling_is_valid_probability() -> None:
    """``compute_ceiling`` runs real random games through the engine and
    returns per-position 1/N_legal values. Each is a valid probability and
    the mean (the unconditional top-1 ceiling) sits strictly in (0, 1]."""
    from pawn.config import VOCAB_SIZE

    mod = _load_script("compute_theoretical_ceiling")
    result = mod.compute_ceiling(
        n_games=32, max_ply=40, seed=1234, vocab_size=VOCAB_SIZE
    )

    inv_n = result["inv_n"]
    n_legal = result["n_legal"]
    assert inv_n.size > 0
    assert np.all(inv_n > 0.0)
    assert np.all(inv_n <= 1.0)
    # Every sampled non-terminal position has at least one legal move.
    assert np.all(n_legal >= 1)
    mean_ceiling = float(inv_n.mean())
    assert 0.0 < mean_ceiling <= 1.0
    # Random chess positions have many legal moves, so the ceiling is small.
    assert mean_ceiling < 0.5


# ---------------------------------------------------------------------------
# extract_lichess_parquet: work-unit + partition logic
# ---------------------------------------------------------------------------


def test_workunit_sentinel_and_tag() -> None:
    mod = _load_script("extract_lichess_parquet")
    unit = mod.WorkUnit(kind="train", year_month="2025-01")
    assert unit.tag == "train/2025-01"
    assert unit.sentinel_path == "data/_complete/train-2025-01.done"


def test_partition_units_done_vs_to_run_and_stale() -> None:
    """A unit with its sentinel is "done"; a partial unit (shards, no
    sentinel) is queued to run and its orphan shards are flagged stale."""
    mod = _load_script("extract_lichess_parquet")
    done_unit = mod.WorkUnit(kind="train", year_month="2025-01")
    partial_unit = mod.WorkUnit(kind="holdout", year_month="2025-02")
    units = [done_unit, partial_unit]

    existing = [
        done_unit.sentinel_path,
        "data/train-2025-01-0000.parquet",
        # partial holdout: shards present, no sentinel
        "data/validation-2025-02-0000.parquet",
        "data/test-2025-02-0000.parquet",
    ]

    to_run, done, stale = mod.partition_units(units, existing, force=False)
    assert done == [done_unit]
    assert to_run == [partial_unit]
    assert "data/validation-2025-02-0000.parquet" in stale
    assert "data/test-2025-02-0000.parquet" in stale
    # The completed unit's shard is NOT marked stale.
    assert "data/train-2025-01-0000.parquet" not in stale


def test_partition_units_force_requeues_everything() -> None:
    mod = _load_script("extract_lichess_parquet")
    unit = mod.WorkUnit(kind="train", year_month="2025-01")
    existing = [unit.sentinel_path, "data/train-2025-01-0000.parquet"]

    to_run, done, stale = mod.partition_units([unit], existing, force=True)
    assert to_run == [unit]
    assert done == []
    # Force re-extract deletes the old sentinel + shard so the rerun is clean.
    assert unit.sentinel_path in stale
    assert "data/train-2025-01-0000.parquet" in stale


def test_parse_day_range() -> None:
    mod = _load_script("extract_lichess_parquet")
    assert mod.parse_day_range("1-7") == (1, 7)
    with pytest.raises(Exception):
        mod.parse_day_range("notarange")
