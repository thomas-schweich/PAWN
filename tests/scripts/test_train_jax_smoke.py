"""Smoke tests for the v2 entry-point scripts.

Confirms each script imports cleanly + responds to `--help` / arg
parsing. The full functional verification (1000-step train, real
LoRA fine-tune, etc.) runs as the S13 / S16 acceptance criterion
checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pawn.run_config import AdapterConfig, PretrainConfig

SCRIPTS = (
    "train_jax",
    "train_jax_adapter",
    "train_jax_distill",
    "eval_jax",
    "eval_parity",
    "eval_probes_jax",
    "eval_generation_jax",
    "eval_vs_stockfish",
    "sweep",
    "run_evals_backbone",
)


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_script_module_imports(script_name: str) -> None:
    """Each script under `scripts/` imports cleanly without side effects."""
    import importlib.util

    script_path = Path("scripts") / f"{script_name}.py"
    assert script_path.is_file()
    spec = importlib.util.spec_from_file_location(
        f"scripts_{script_name}", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")


def _load_train_jax():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_vsel", Path("scripts") / "train_jax.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg_for_variants(tj, variants: list[str]) -> "PretrainConfig":  # noqa: F821
    """Build a tiny PretrainConfig through the real CLI path, optionally
    restricting `--variants` to a subset."""
    extra = ["--variants", *variants] if variants else []
    args = tj._parse_args(
        ["--supernet", "tiny", "--total-steps", "1", "--local-checkpoints", *extra]
    )
    return tj._build_config(args)


# The tiny supernet config the accuracy-on-trained-width tests slice against.
def _tiny_supernet_cfg():
    from pawn.config import TINY_SUPERNET

    return TINY_SUPERNET


_TINY_SUPERNET_CFG = _tiny_supernet_cfg()


def test_train_jax_variants_flag_and_build() -> None:
    """`--variants` selects which variants train; default trains all three.

    Routes through the real CLI path (_parse_args -> _build_config ->
    build_variants) so the argparse merge + pydantic validation + spec
    construction are all exercised.
    """
    from pydantic import ValidationError

    tj = _load_train_jax()

    def cfg_for(extra: list[str]):
        args = tj._parse_args(
            ["--supernet", "tiny", "--total-steps", "1", "--local-checkpoints", *extra]
        )
        return tj._build_config(args)

    # Default (no --variants): all three, only `large` is the supernet.
    cfg = cfg_for([])
    assert cfg.variants is None
    specs = tj.build_variants(cfg)
    assert tuple(s.name for s in specs) == ("small", "base", "large")
    assert sum(bool(s.is_supernet) for s in specs) == 1

    # `--variants large`: a single is_supernet=True large variant
    # (standalone-large teacher pretrain).
    cfg = cfg_for(["--variants", "large"])
    assert cfg.variants == ("large",)
    specs = tj.build_variants(cfg)
    assert len(specs) == 1
    assert specs[0].name == "large" and specs[0].is_supernet is True

    # A subset is allowed (order preserved).
    cfg = cfg_for(["--variants", "small", "base"])
    assert cfg.variants == ("small", "base")
    assert tuple(s.name for s in tj.build_variants(cfg)) == ("small", "base")

    # Invalid name -> pydantic Literal rejection.
    with pytest.raises(ValidationError):
        cfg_for(["--variants", "huge"])
    # Duplicates and bare `--variants` (empty) -> our validators.
    with pytest.raises(ValidationError):
        cfg_for(["--variants", "large", "large"])
    with pytest.raises(ValidationError):
        cfg_for(["--variants"])


def test_widest_trained_variant_and_accuracy_model_picks_trained_width() -> None:
    """`train/accuracy` must be measured on the widest TRAINED variant.

    The supernet joint loss only updates the inner ``[:d_V, :d_V]`` slice of
    each selected variant (caf6a53). So for a ``--variants`` subset that
    excludes ``large`` the full unsliced model's outer dims stay at init —
    a full-model accuracy forward would mix trained inner dims with
    untrained outer dims and report near-random accuracy for a healthy run.
    This pins that `accuracy_model` slices down to the widest trained width
    in that case, and is a no-op when ``large`` is selected (default path).
    """
    from pawn.config import TINY_VARIANTS
    from pawn.model import init_model

    tj = _load_train_jax()
    model = init_model(_TINY_SUPERNET_CFG, key=0)

    # Default (small+base+large): widest is large = the supernet, so the
    # accuracy model is the full unsliced model (no slice — old behaviour).
    default_specs = tj.build_variants(_cfg_for_variants(tj, []))
    widest_default = tj.widest_trained_variant(default_specs)
    assert widest_default.name == "large" and widest_default.is_supernet
    assert tj.accuracy_model(model, widest_default) is model

    # Subset excluding large: widest is `base`, NOT the supernet, so the
    # accuracy model is sliced down to base's width (never reads the
    # untrained outer `[d_base:d_large]` dims).
    subset_specs = tj.build_variants(_cfg_for_variants(tj, ["small", "base"]))
    widest_subset = tj.widest_trained_variant(subset_specs)
    assert widest_subset.name == "base" and not widest_subset.is_supernet
    acc_model = tj.accuracy_model(model, widest_subset)
    assert acc_model is not model
    assert acc_model.cfg.d_model == TINY_VARIANTS["base"].d_model
    assert acc_model.cfg.d_model < _TINY_SUPERNET_CFG.d_model


def test_accuracy_model_is_invariant_to_untrained_outer_dims() -> None:
    """The accuracy metric must not depend on the untrained outer dims.

    When `large` is excluded the outer `[d_base:d_large]` weight block stays
    at init and is never updated by the joint loss. `accuracy_model` slices
    down to the widest trained width, so the accuracy forward must be
    *invariant* to any change in those outer dims — whereas the full
    unsliced forward (the old, buggy `train/accuracy` source) is NOT
    invariant: it reads the untrained outer dims and reports a value that
    swings with arbitrary init noise out there.

    We verify by perturbing only the outer block of one weight tensor and
    checking the sliced accuracy is unchanged while the full-model accuracy
    moves.
    """
    import equinox as eqx
    import jax
    import numpy as np

    from pawn.corpus import generate_corpus
    from pawn.model import init_model
    from pawn.trainer import slice_batch, top1_accuracy

    tj = _load_train_jax()

    variants = tj.build_variants(_cfg_for_variants(tj, ["small", "base"]))
    widest = tj.widest_trained_variant(variants)
    assert not widest.is_supernet
    d_trained = widest.cfg.d_model
    d_full = _TINY_SUPERNET_CFG.d_model
    assert d_trained < d_full  # there ARE untrained outer dims to perturb

    model = init_model(_TINY_SUPERNET_CFG, key=0)
    corpus = generate_corpus(n_games=8, max_ply=32, seq_len=32, seed=0)
    batch = slice_batch(corpus, np.arange(8))

    base_sliced = float(top1_accuracy(tj.accuracy_model(model, widest), batch))
    base_full = float(top1_accuracy(model, batch))

    # Perturb ONLY the untrained outer block `[d_trained:, d_trained:]` of
    # every layer's wq stack. The trained inner slice `[:d_trained, :d_trained]`
    # is untouched, so the sliced forward must be bit-identical.
    def _perturb_outer(m):  # noqa: ANN001 — local PyTree edit
        wq = m.layers.wq
        outer_mask = (
            (jax.numpy.arange(wq.shape[1])[None, :, None] >= d_trained)
            | (jax.numpy.arange(wq.shape[2])[None, None, :] >= d_trained)
        )
        new_wq = jax.numpy.where(outer_mask, wq + 5.0, wq)
        return eqx.tree_at(lambda mm: mm.layers.wq, m, new_wq)

    perturbed = _perturb_outer(model)

    pert_sliced = float(top1_accuracy(tj.accuracy_model(perturbed, widest), batch))
    pert_full = float(top1_accuracy(perturbed, batch))

    # Sliced accuracy (the fix): invariant to the untrained outer dims.
    assert pert_sliced == base_sliced
    # Full-model accuracy (the old bug): contaminated by the outer dims, so a
    # large perturbation out there moves the reported value.
    assert pert_full != base_full


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_script_help_works(script_name: str) -> None:
    """`--help` exits 0 — argparse is wired up correctly."""
    import subprocess

    result = subprocess.run(
        [sys.executable, f"scripts/{script_name}.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script_name} --help failed:\n{result.stdout}\n{result.stderr}"
    )


def _subprocess_env() -> "dict[str, str]":
    """Subprocess env for CPU-friendly script tests.

    `pawn.jax_setup.require_accelerator()` refuses to run on CPU unless
    `PAWN_ALLOW_CPU=1` is set (parity with v1). Round-3 codex P2: the
    subprocess tests below need this override or they fail before
    reaching the guard they're trying to pin on CPU-only CI.
    """
    import os
    env = os.environ.copy()
    env["PAWN_ALLOW_CPU"] = "1"
    return env


def test_train_jax_adapter_rejects_rosa_resume(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """RoSA + --resume with step > 0 must fail loudly before any
    optimizer-state damage. Round-2 test-risk: the prior fix was
    verified only by manual smoke run; this test pins the rejection.

    Constructs a minimal fake checkpoint directory with just enough
    `training_state.json` for the early `_resume_step_peek` read to
    fire, then runs the script as a subprocess and asserts the exit
    message comes from our SystemExit rather than a downstream
    cryptic shape mismatch.

    Uses an in-tree TINY_SUPERNET-shaped fake (the `--checkpoint`
    arg refers to a path on disk, but the script's early-exit code
    path doesn't load it before the RoSA guard fires)."""
    import json
    import subprocess

    fake_ckpt = tmp_path / "step_00000100"
    fake_ckpt.mkdir()
    (fake_ckpt / "training_state.json").write_text(json.dumps({"step": 100}))

    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_adapter.py",
            "--strategy", "rosa", "--rosa-mode", "rosa",
            "--supernet", "tiny", "--variant", "small",
            "--checkpoint", "thomas-schweich/pawn-small",
            "--no-pgn", "--total-steps", "6",
            "--batch-size", "4", "--seq-len", "16", "--k", "2",
            "--lora-rank", "2", "--density", "0.1",
            "--local-checkpoints", "--lr", "1e-3",
            "--resume", str(fake_ckpt),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_subprocess_env(),
    )
    combined = result.stdout + result.stderr
    # Specific guard message (not just any non-zero exit — that could
    # be an unrelated import error). Round-3 test-risk MEDIUM.
    assert "--resume is not supported for RoSA" in combined, (
        f"Expected the RoSA-resume guard message; got stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert result.returncode != 0, "RoSA --resume should fail"


def test_train_jax_adapter_rejects_conditioning_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Phase-A spec Chunk 4 / C1: an adapter run whose `--conditioning`
    disagrees with the backbone's persisted conditioning must fail loudly
    (the load-time C guard), not silently shift every move's absolute
    RoPE offset.

    Builds a tiny backbone checkpoint persisting `conditioning=["outcome"]`
    (C=2), then runs the adapter with the default empty conditioning
    (C=1) and asserts the `assert_conditioning_C` failure surfaces.
    """
    import subprocess

    from pawn.checkpoint import save_model
    from pawn.config import TINY_SUPERNET
    from pawn.model import init_model

    backbone = init_model(TINY_SUPERNET, key=0)
    ckpt_dir = tmp_path / "backbone"
    save_model(
        backbone, ckpt_dir, training_state={"step": 0},
        run_config={"conditioning": ["outcome"]},
    )

    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_adapter.py",
            "--strategy", "lora",
            "--supernet", "tiny", "--variant", "small",
            "--checkpoint", str(ckpt_dir),
            "--no-pgn", "--total-steps", "2",
            "--batch-size", "4", "--seq-len", "16", "--k", "1",
            "--lora-rank", "2",
            "--local-checkpoints", "--lr", "1e-3",
            # NB: no --conditioning → cfg.conditioning defaults to [] (C=1),
            # which disagrees with the backbone's persisted C=2.
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    combined = result.stdout + result.stderr
    assert "conditioning mismatch" in combined, (
        f"Expected the load-time C-mismatch guard; got "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.returncode != 0, "conditioning mismatch should fail"


def test_train_jax_conditioning_threaded_into_corpus(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A `--conditioning outcome` pretrain must TRAIN under C=2, not just
    record C=2 in config.json.

    Regression guard for SC-1 / A1: the trainer used to build the corpus
    with the default `conditioning=()` (C=1, BOS-only) while persisting
    `conditioning=["outcome"]` (C=2) — so the model trained at one
    absolute-RoPE offset and every eval/read path (which derives C from
    the persisted block) placed moves at a different offset. This pins
    that the train-time C and the persisted C can never diverge again:

    1. Run a tiny real pretrain with `--conditioning outcome`.
    2. The written checkpoint's run block records `conditioning=["outcome"]`.
    3. Rebuilding the corpus the way eval does (from that run block)
       lands moves at `outcome_offset[0] == C == 2` — the same C the
       trainer actually packed, because both now derive from the one
       persisted `conditioning` field.
    """
    import subprocess

    from pawn.checkpoint import load_model
    from pawn.corpus import (
        conditioning_from_run_block,
        conditioning_to_C,
        generate_corpus,
    )

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "4",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--conditioning", "outcome",
            "--checkpoint-interval", "2",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"tiny pretrain failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    ckpts = sorted(logs_dir.glob("*/step_*"))
    assert ckpts, (
        f"no checkpoint written under {logs_dir}; "
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    _model, run_block = load_model(ckpts[-1])
    # The persisted run block must record the conditioning the run used.
    conditioning = conditioning_from_run_block(run_block)
    assert conditioning == ["outcome"], (
        f"checkpoint run block conditioning={conditioning!r}, expected "
        f"['outcome'] — the trainer dropped cfg.conditioning on the floor"
    )
    C = conditioning_to_C(conditioning)
    assert C == 2

    # Rebuild the corpus exactly as eval_jax does (from the persisted
    # block). If the trainer had built C=1 while persisting C=2, this
    # eval-side corpus would place moves one absolute-RoPE slot away
    # from where the model trained. The per-game prefix width is the
    # constant C, recorded in `outcome_offset`.
    corpus = generate_corpus(
        n_games=8, max_ply=32, seq_len=32, seed=0,
        conditioning=conditioning,
    )
    assert int(corpus.outcome_offset[0]) == C, (
        f"corpus prefix width {int(corpus.outcome_offset[0])} != C={C}"
    )


def test_train_jax_adapter_rejects_resume_without_training_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """--resume against a directory missing `training_state.json` must
    fail loudly. The prior code silently treated the absent file as
    `step=0`, bypassing the RoSA guard and producing a cryptic
    downstream error. Round-2 test-risk MEDIUM."""
    import subprocess

    fake_ckpt = tmp_path / "step_00000050_no_ts"
    fake_ckpt.mkdir()
    # Intentionally do NOT write training_state.json.

    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_adapter.py",
            "--strategy", "lora",
            "--supernet", "tiny", "--variant", "small",
            "--checkpoint", "thomas-schweich/pawn-small",
            "--no-pgn", "--total-steps", "6",
            "--batch-size", "4", "--seq-len", "16", "--k", "2",
            "--lora-rank", "2",
            "--local-checkpoints", "--lr", "1e-3",
            "--resume", str(fake_ckpt),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_subprocess_env(),
    )
    combined = result.stdout + result.stderr
    # Match the guard's distinctive phrase so a generic traceback
    # mentioning the filename can't satisfy the assertion
    # (round-4 bug-detector MINOR).
    assert "--resume requires" in combined and "training_state.json" in combined, (
        f"Expected the missing-sidecar guard; got stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert result.returncode != 0, (
        "Resume against a sidecar-less dir should fail"
    )


# ---------------------------------------------------------------------------
# B1 — distillation script wiring (parity with the train_jax / adapter smokes)
# ---------------------------------------------------------------------------


def _dir_digest(root: Path) -> "dict[str, str]":
    """Map each file under ``root`` to a SHA-256 of its bytes.

    Used to prove the frozen teacher checkpoint is bit-identical before and
    after a distillation run (the teacher must receive zero gradient and is
    never re-saved). Compares every payload file, not just the count.
    """
    import hashlib

    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


def test_train_jax_distill_runs_and_roundtrips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """B1 smoke (spec smoke item 2): a tiny teacher→student distillation runs
    end-to-end, writes a student checkpoint that round-trips through
    `load_model`, and leaves the frozen teacher checkpoint bit-identical.

    Mirrors `test_train_jax_accumulation_steps_runs` /
    `test_train_jax_conditioning_threaded_into_corpus` — the distill script is
    otherwise covered only by the generic import + `--help` smoke, so the
    teacher-load → scan-loop → `_save` round-trip wiring is unexercised. This
    pins:
      * rc == 0 through the real `make_distill_scan_step` driver + `val_step`,
      * a `distill_step_*` checkpoint is written and `load_model` reads it back,
      * the teacher checkpoint dir bytes are unchanged (zero teacher grad +
        no re-save), parity with the spec's "teacher bit-identical after".
    """
    import subprocess

    from pawn.checkpoint import load_model, save_model
    from pawn.config import TINY_SUPERNET
    from pawn.model import init_model

    # A real loadable v2 teacher checkpoint (default conditioning → C=1, which
    # the distill run inherits and matches).
    teacher = init_model(TINY_SUPERNET, key=0)
    teacher_dir = tmp_path / "teacher"
    save_model(
        teacher, teacher_dir, training_state={"step": 0}, run_config={}
    )
    teacher_before = _dir_digest(teacher_dir)

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_distill.py",
            "--distill-from", str(teacher_dir),
            "--student-supernet", "tiny",
            "--no-pgn", "--objective", "mix",
            "--total-steps", "4",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--checkpoint-interval", "2", "--log-interval", "1",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"tiny distill failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    ckpts = sorted(logs_dir.glob("*/distill_step_*"))
    assert ckpts, (
        f"no student checkpoint written under {logs_dir}; "
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    # The student checkpoint round-trips through the canonical v2 loader.
    _student, run_block = load_model(ckpts[-1])
    assert run_block is not None, "student checkpoint dropped its run block"
    assert run_block.get("run_type") == "distill"

    # The frozen teacher must be bit-identical — zero gradient, never re-saved.
    teacher_after = _dir_digest(teacher_dir)
    assert teacher_after == teacher_before, (
        "teacher checkpoint bytes changed across the distill run; the teacher "
        "must be frozen (zero gradient) and is never re-saved"
    )


def test_train_jax_distill_rejects_conditioning_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """B1 / Phase-A C-guard: a distill run whose run-config conditioning
    disagrees with the teacher's persisted conditioning must fail loudly
    (the load-time C-assert), not silently shift every move's absolute RoPE
    offset. Parity with `test_train_jax_adapter_rejects_conditioning_mismatch`.

    Builds a tiny teacher persisting `conditioning=["outcome"]` (C=2), then
    runs distill with the default empty conditioning (C=1) and asserts the
    `conditioning mismatch` SystemExit at train_jax_distill.py fires.
    """
    import subprocess

    from pawn.checkpoint import save_model
    from pawn.config import TINY_SUPERNET
    from pawn.model import init_model

    teacher = init_model(TINY_SUPERNET, key=0)
    teacher_dir = tmp_path / "teacher_c2"
    save_model(
        teacher, teacher_dir, training_state={"step": 0},
        run_config={"conditioning": ["outcome"]},
    )

    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_distill.py",
            "--distill-from", str(teacher_dir),
            "--student-supernet", "tiny",
            "--no-pgn", "--objective", "mix",
            "--total-steps", "2",
            "--batch-size", "4", "--seq-len", "32", "--k", "1",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(tmp_path / "logs"),
            # NB: no --conditioning → cfg.conditioning defaults to [] (C=1),
            # which disagrees with the teacher's persisted C=2.
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    combined = result.stdout + result.stderr
    assert "conditioning mismatch" in combined, (
        f"Expected the teacher-vs-run-config C-mismatch guard; got "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.returncode != 0, "conditioning mismatch should fail"


# ---------------------------------------------------------------------------
# B2 — inert-knob parity: mate_boost / accumulation_steps / adapter cadence
# ---------------------------------------------------------------------------


def _load_adapter_module():  # type: ignore[no-untyped-def]
    """Import `scripts/train_jax_adapter.py` as a module to reach
    `_resolve_cadence` (the pure cadence/sampling resolver)."""
    import importlib.util

    script_path = Path("scripts") / "train_jax_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_adapter_cadence", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _adapter_cfg(**overrides: object):  # type: ignore[no-untyped-def]
    from typing import Any

    from pawn.run_config import AdapterConfig

    base: dict[str, Any] = dict(
        local_checkpoints=True, total_steps=100, strategy="lora", lora_rank=4,
        batch_size=4,
    )
    base.update(overrides)
    return AdapterConfig(**base)


def test_resolve_cadence_data_seed_changes_sampling_seed() -> None:
    """B2: `data_seed` is consumed — it seeds the train sampler (and the
    val sampler at `data_seed + 1`). Different `data_seed` ⇒ different seed
    drives different game orders."""
    mod = _load_adapter_module()
    c_default = mod._resolve_cadence(_adapter_cfg(), n_train_games=1000)
    assert c_default.data_seed == 0  # None → 0
    c_seeded = mod._resolve_cadence(
        _adapter_cfg(data_seed=42), n_train_games=1000
    )
    assert c_seeded.data_seed == 42
    # The seed actually changes the sampled game order.
    import numpy as np

    a = np.random.default_rng(c_default.data_seed).integers(0, 1000, size=8)
    b = np.random.default_rng(c_seeded.data_seed).integers(0, 1000, size=8)
    assert not np.array_equal(a, b)


def test_resolve_cadence_epochs_steps_per_epoch_set_budget() -> None:
    """B2: `epochs` × `steps_per_epoch` resolve the step budget (v1
    semantics). With them unset the budget is `total_steps` and the
    step-based `eval_interval` drives val (epochs/val_every are no-ops)."""
    mod = _load_adapter_module()
    # Unset steps_per_epoch → total_steps is the budget; epochs ignored.
    c_none = mod._resolve_cadence(
        _adapter_cfg(total_steps=100, epochs=7), n_train_games=1000
    )
    assert c_none.effective_total_steps == 100
    assert c_none.eval_interval == 100  # eval_interval None → log_interval

    # Explicit int steps_per_epoch → epochs × steps_per_epoch.
    c_int = mod._resolve_cadence(
        _adapter_cfg(epochs=3, steps_per_epoch=10, val_every=2),
        n_train_games=1000,
    )
    assert c_int.epoch_steps == 10
    assert c_int.effective_total_steps == 30
    assert c_int.eval_interval == 20  # val_every × epoch_steps

    # steps_per_epoch="all" → n_train_games // batch_size.
    c_all = mod._resolve_cadence(
        _adapter_cfg(epochs=2, steps_per_epoch="all", batch_size=4),
        n_train_games=400,
    )
    assert c_all.epoch_steps == 100  # 400 // 4
    assert c_all.effective_total_steps == 200


def test_resolve_cadence_val_every_changes_eval_interval() -> None:
    """B2: `val_every` is consumed — it scales the eval cadence in epoch
    units (only meaningful when steps_per_epoch defines an epoch)."""
    mod = _load_adapter_module()
    c1 = mod._resolve_cadence(
        _adapter_cfg(steps_per_epoch=10, val_every=1), n_train_games=1000
    )
    c3 = mod._resolve_cadence(
        _adapter_cfg(steps_per_epoch=10, val_every=3), n_train_games=1000
    )
    assert c1.eval_interval == 10
    assert c3.eval_interval == 30
    assert c3.eval_interval != c1.eval_interval


def test_mate_boost_threaded_into_generate_corpus() -> None:
    """B2: `mate_boost` is consumed by `generate_corpus` (it maps onto the
    engine's mate-biasing arg). A positive boost changes the generated
    corpus for a fixed seed, proving the field is no longer inert."""
    import numpy as np

    from pawn.corpus import generate_corpus

    plain = generate_corpus(
        n_games=64, max_ply=60, seq_len=64, seed=7, mate_boost=0.0
    )
    boosted = generate_corpus(
        n_games=64, max_ply=60, seq_len=64, seed=7, mate_boost=5.0
    )
    assert not np.array_equal(plain.tokens, boosted.tokens), (
        "mate_boost did not reach the engine — corpus identical to the "
        "mate_boost=0 baseline"
    )


def test_discard_ply_limit_threaded_into_generate_corpus() -> None:
    """B2: the sibling `discard_ply_limit` engine knob is likewise threaded
    through `generate_corpus` (consumed, not inert)."""
    import numpy as np

    from pawn.corpus import generate_corpus

    keep = generate_corpus(
        n_games=128, max_ply=20, seq_len=24, seed=3, discard_ply_limit=False
    )
    drop = generate_corpus(
        n_games=128, max_ply=20, seq_len=24, seed=3, discard_ply_limit=True
    )
    assert not np.array_equal(keep.tokens, drop.tokens)


def test_train_jax_accumulation_steps_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """B2 smoke: `--accumulation-steps 2` runs end-to-end (no
    NotImplementedError) and writes a checkpoint — the prefetcher now emits
    `(K, N, B, T)` batches that the accumulation kernel consumes."""
    import subprocess

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "4",
            "--accumulation-steps", "2",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--checkpoint-interval", "2",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    assert "NotImplementedError" not in (result.stdout + result.stderr), (
        f"accumulation path still raised NotImplementedError:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.returncode == 0, (
        f"accumulation_steps=2 pretrain failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    ckpts = sorted(logs_dir.glob("*/step_*"))
    assert ckpts, (
        f"no checkpoint written under {logs_dir}; "
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# Held-out validation loop + patience early-stop + pause (v1 parity). These
# drive the real `train_jax.py` loop end-to-end and read metrics.jsonl back.
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    import json as _json

    return [
        _json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_train_jax_validation_loop_emits_val_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`--val-every N` runs the held-out validation pass and writes
    `type=val` records carrying the v1 `val/*` schema (val/loss, val/top1,
    val/top5, val/perplexity, val/legal_move_rate, per-phase). Regression
    guard for the major audit gap: the v2 pretrain loop previously emitted
    NO val records at all."""
    import subprocess

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "6",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--val-every", "2", "--val-games", "8",
            "--checkpoint-interval", "6",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True, text=True, timeout=600, env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"pretrain with --val-every failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    metrics = sorted(logs_dir.glob("*/metrics.jsonl"))
    assert metrics, f"no metrics.jsonl under {logs_dir}"
    records = _read_jsonl(metrics[-1])
    val_records = [r for r in records if r.get("type") == "val"]
    assert val_records, (
        "no type=val records emitted despite --val-every 2 — the held-out "
        f"validation loop did not run. stdout={result.stdout}"
    )
    # The v1 val/* schema keys must be present on a val record.
    vr = val_records[0]
    for key in (
        "val/loss", "val/top1", "val/top5", "val/perplexity",
        "val/legal_move_rate",
    ):
        assert key in vr, f"val record missing {key}: {sorted(vr)}"
    # Sanity on value ranges.
    assert 0.0 <= vr["val/top1"] <= 1.0
    assert vr["val/top5"] >= vr["val/top1"]
    assert 0.0 <= vr["val/legal_move_rate"] <= 1.0


def test_train_jax_grad_norm_logged_by_default(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Per-step `grad_norm` is logged on every train record WITHOUT
    `--emit-grad-norms` (v1 parity — v1 always logged grad_norm). The v2
    `--emit-grad-norms` gate that suppressed it by default diverged from
    that contract."""
    import subprocess

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "4",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--log-interval", "1",
            "--checkpoint-interval", "4",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
            # NB: NO --emit-grad-norms — grad_norm must still be present.
        ],
        capture_output=True, text=True, timeout=600, env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"pretrain failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    records = _read_jsonl(sorted(logs_dir.glob("*/metrics.jsonl"))[-1])
    train_records = [r for r in records if r.get("type") == "train"]
    assert train_records, "no train records"
    assert all("grad_norm" in r for r in train_records), (
        "grad_norm missing from a train record despite v1-parity default "
        f"logging: {[sorted(r) for r in train_records]}"
    )


def test_train_jax_patience_early_stops(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`--patience N` early-stops when the held-out val loss + late-game
    legality stop improving for N consecutive evals, and records
    `reason_for_stop=patience` in schedule_health.json. A very low LR keeps
    the model from improving so patience fires well before total_steps."""
    import json as _json
    import subprocess

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "1000",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--val-every", "2", "--val-games", "8", "--patience", "1",
            "--checkpoint-interval", "2",
            # LR ~0 so val loss never improves after the first eval → the
            # patience counter trips on the second eval.
            "--local-checkpoints", "--lr", "1e-12",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True, text=True, timeout=600, env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"patience run failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    health = sorted(logs_dir.glob("*/schedule_health.json"))
    assert health, f"no schedule_health.json under {logs_dir}"
    h = _json.loads(health[-1].read_text())
    assert h["reason_for_stop"] == "patience", (
        f"expected reason_for_stop=patience, got {h['reason_for_stop']}; "
        f"stdout={result.stdout}"
    )
    # Early stop: stopped well before the 1000-step budget.
    assert h["actual_total_steps"] < 1000


def test_train_jax_pause_after_steps_checkpoints_and_stops(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`--pause-after-steps N` checkpoints at the N-step boundary and stops
    with `reason_for_stop=paused` (v1 pause primitive). The run does NOT
    reach total_steps."""
    import json as _json
    import subprocess

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "1000",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--pause-after-steps", "4",
            "--checkpoint-interval", "1000",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True, text=True, timeout=600, env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"pause run failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    health = sorted(logs_dir.glob("*/schedule_health.json"))
    assert health, f"no schedule_health.json under {logs_dir}"
    h = _json.loads(health[-1].read_text())
    assert h["reason_for_stop"] == "paused", (
        f"expected reason_for_stop=paused, got {h['reason_for_stop']}"
    )
    assert h["actual_total_steps"] < 1000
    # A checkpoint was written at the pause boundary so the run is resumable.
    assert sorted(logs_dir.glob("*/step_*")), (
        "pause did not write a resumable checkpoint"
    )


# ---------------------------------------------------------------------------
# v1 CLI-compat flags reachable only via --config JSON in v2 (config-cli
# parity workstream). Each routes through the real _parse_args ->
# _build_config path so the argparse wiring + pydantic migration both run.
# ---------------------------------------------------------------------------


def _train_jax_cfg(extra: list[str]) -> PretrainConfig:
    tj = _load_train_jax()
    args = tj._parse_args(
        ["--supernet", "tiny", "--total-steps", "1",
         "--local-checkpoints", *extra]
    )
    cfg: PretrainConfig = tj._build_config(args)
    return cfg


def test_train_jax_prepend_outcome_cli_flag() -> None:
    """`--prepend-outcome` is the v1 boolean; the CLI must accept it (v1
    users got an argparse error in v2) and the before-validator folds it
    into `conditioning=["outcome"]`."""
    with pytest.warns(DeprecationWarning, match="prepend_outcome"):
        cfg = _train_jax_cfg(["--prepend-outcome"])
    assert cfg.conditioning == ["outcome"]
    assert cfg.C == 2
    # Omitting the flag leaves the BOS-only default (C == 1, no warning).
    cfg = _train_jax_cfg([])
    assert cfg.conditioning == []
    assert cfg.C == 1


def test_train_jax_prepend_outcome_conflicts_with_conditioning() -> None:
    """Passing both `--prepend-outcome` and `--conditioning` is the
    ambiguous case the migration validator rejects."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="not both"):
        _train_jax_cfg(["--prepend-outcome", "--conditioning", "outcome"])


def test_train_jax_amp_dtype_cli_flag() -> None:
    """`--amp-dtype` is exposed directly (was --config-only in v2). The v1
    spelling `none` migrates to `float32`."""
    cfg = _train_jax_cfg(["--amp-dtype", "float16"])
    assert cfg.amp_dtype == "float16"
    with pytest.warns(DeprecationWarning, match="amp_dtype"):
        cfg = _train_jax_cfg(["--amp-dtype", "none"])
    assert cfg.amp_dtype == "float32"


def test_train_jax_max_seq_len_cli_flag_migrates_to_seq_len() -> None:
    """`--max-seq-len` is the v1 field name; the CLI accepts it and the
    before-validator folds it into `seq_len`."""
    with pytest.warns(DeprecationWarning, match="max_seq_len"):
        cfg = _train_jax_cfg(["--max-seq-len", "64"])
    assert cfg.seq_len == 64


def test_train_jax_seq_len_flag_still_works() -> None:
    """The native `--seq-len` flag is unaffected by the max_seq_len alias."""
    cfg = _train_jax_cfg(["--seq-len", "128"])
    assert cfg.seq_len == 128


def test_train_jax_core_v1_knobs_exposed_as_cli_flags() -> None:
    """v1's generic `--flag value` parser exposed every BaseRunConfig
    field; v2's explicit argparse must expose the load-bearing pretrain
    knobs as direct flags (previously --config-only). Route a
    representative spread through _parse_args -> _build_config and assert
    each lands on the config."""
    cfg = _train_jax_cfg([
        "--weight-decay", "0.01",
        "--max-grad-norm", "0.5",
        "--warmup-steps", "50",
        "--decay-frac", "0.2",
        "--cooldown-frac", "0.15",
        "--stable-lr-ratio", "0.2",
        "--wsd-decay-shape", "cosine",
        "--log-interval", "25",
        "--mate-boost", "1.5",
        "--wandb-project", "pawn-test",
    ])
    assert cfg.weight_decay == 0.01
    assert cfg.max_grad_norm == 0.5
    assert cfg.warmup_steps == 50
    assert cfg.decay_frac == 0.2
    assert cfg.cooldown_frac == 0.15
    assert cfg.stable_lr_ratio == 0.2
    assert cfg.wsd_decay_shape == "cosine"
    assert cfg.log_interval == 25
    assert cfg.mate_boost == 1.5
    assert cfg.wandb_project == "pawn-test"


class _AbortAfterSpy(Exception):
    """Sentinel raised by the corpus spy to short-circuit the pretrain loop
    once the first `generate_corpus` call has been observed."""


def test_train_jax_mate_boost_cli_flag_is_honoured_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--mate-boost` stays exposed as a pretrain flag *because the loop
    honours it* — `cfg.mate_boost` is fed into `generate_corpus` at the
    corpus-build call site. "Lands on the config" is not enough (that is the
    exact masking the gate flagged for the dropped no-op knobs), so drive
    `main()` end-to-end and assert the value the CLI parsed is the value that
    reaches `generate_corpus`. A spy records the kwarg and raises to abort
    before the (heavy) training scan; the trainer's `finally` still tears down
    the prefetch executor cleanly."""
    import os

    tj = _load_train_jax()

    seen: dict[str, float] = {}
    real_generate_corpus = tj.generate_corpus

    def _spy(*args: object, **kwargs: object) -> object:
        # Build a real (tiny) corpus so the call is faithful, capture the
        # mate_boost the call site passed, then abort the run.
        mate_boost = kwargs.get("mate_boost")
        assert isinstance(mate_boost, float)
        seen["mate_boost"] = mate_boost
        real_generate_corpus(*args, **kwargs)
        raise _AbortAfterSpy

    monkeypatch.setattr(tj, "generate_corpus", _spy)
    monkeypatch.setenv("PAWN_ALLOW_CPU", "1")
    # JAX picks up the platform from env at first device use; the spy aborts
    # before any compiled step runs, but require_accelerator() still gates.
    monkeypatch.setenv("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))

    with pytest.raises(_AbortAfterSpy):
        tj.main([
            "--supernet", "tiny", "--total-steps", "2",
            "--batch-size", "2", "--seq-len", "32", "--k", "2",
            "--mate-boost", "3.5", "--local-checkpoints", "--lr", "1e-3",
        ])

    assert seen.get("mate_boost") == 3.5, (
        "the --mate-boost CLI value did not reach generate_corpus — the flag "
        "lands on the config but is not honoured by the corpus-build path"
    )


@pytest.mark.parametrize(
    "flag,value",
    [
        # Still NOT exposed — Lichess-path / unconsumed fields with no
        # consumer in the random-game pretrain path. `--eval-interval` is the
        # v1 spelling of the validation cadence; v2 exposes `--val-every`
        # (parity with the adapter cadence knob) and maps `eval_interval`
        # → `val_every` only via `--config` JSON, so the bare flag is unknown.
        ("--eval-interval", "200"),
        ("--min-ply", "20"),
        ("--max-corpus-gb", "4.0"),
        ("--cache-dir", "/tmp/pawn-cache"),
    ],
)
def test_train_jax_lichess_only_pretrain_knobs_not_exposed_as_cli_flags(
    flag: str, value: str
) -> None:
    """`min_ply` / `cache_dir` are Lichess-path fields (v1 fed them to
    `prepare_lichess_cached`, NOT the random-game pretrain corpus), and
    `max_corpus_gb` is a v2-only soft memory cap with no consumer in either
    path. `--eval-interval` is the v1 cadence spelling that v2 superseded with
    `--val-every`. So none of these must be promoted to argparse flags —
    re-advertising them would let a user pass `--min-ply 20` and get a silent
    no-op. argparse rejects the unknown flag (`SystemExit` from `error()`).

    NOTE: `--patience`, `--pause-after-steps`, `--val-games`, and `--val-every`
    ARE now exposed and honoured by the held-out validation loop (see
    `test_train_jax_validation_and_earlystop_flags_exposed` /
    `test_train_jax_validation_loop_emits_val_records`)."""
    with pytest.raises(SystemExit):
        _train_jax_cfg([flag, value])


def test_train_jax_validation_and_earlystop_flags_exposed() -> None:
    """`--val-every`, `--val-games`, `--patience`, `--pause-after-steps` are
    now first-class pretrain flags (the held-out validation loop + early-stop
    + pause primitive landed), so the CLI path must accept them and land them
    on the config. This is the inverse of the old deferral: they were
    previously --config-JSON-only and the loop ignored them."""
    cfg = _train_jax_cfg([
        "--val-every", "50",
        "--val-games", "128",
        "--patience", "3",
        "--pause-after-steps", "200",
    ])
    assert cfg.val_every == 50
    assert cfg.val_games == 128
    assert cfg.patience == 3
    assert cfg.pause_after_steps == 200


def test_train_jax_eval_interval_maps_to_val_every_via_config_json(
    tmp_path: Path,
) -> None:
    """A verbatim v1 pretrain config spelling the validation cadence
    `eval_interval` must still drive the held-out eval — the PretrainConfig
    before-validator maps `eval_interval` → `val_every` when the latter is
    unset."""
    tj = _load_train_jax()
    config_path = tmp_path / "pretrain.json"
    config_path.write_text(
        json.dumps(
            {
                "run_type": "pretrain",
                "supernet": "tiny",
                "total_steps": 1,
                "local_checkpoints": True,
                "eval_interval": 200,
            }
        )
    )
    args = tj._parse_args(["--config", str(config_path)])
    cfg = tj._build_config(args)
    assert cfg.eval_interval == 200
    assert cfg.val_every == 200  # mapped from eval_interval


def test_train_jax_v1_pretrain_knobs_load_via_config_json(
    tmp_path: Path,
) -> None:
    """A verbatim v1 pretrain run config carrying the validation /
    early-stop / pause knobs loads cleanly through `--config` JSON and the
    fields land on the config. These are now HONOURED by the held-out
    validation loop (no longer the CLI-only deferral); this pins the
    config-schema surface stays stable for v1 configs. `eval_interval`
    drives `val_every` (the v1 cadence spelling), which `patience` requires."""
    tj = _load_train_jax()
    config_path = tmp_path / "pretrain.json"
    config_path.write_text(
        json.dumps(
            {
                "run_type": "pretrain",
                "supernet": "tiny",
                "total_steps": 1,
                "local_checkpoints": True,
                "patience": 5,
                "eval_interval": 200,
                "pause_after_steps": 999,
                "val_games": 128,
            }
        )
    )
    args = tj._parse_args(["--config", str(config_path)])
    cfg: PretrainConfig = tj._build_config(args)
    assert cfg.patience == 5
    assert cfg.eval_interval == 200
    assert cfg.val_every == 200  # mapped from eval_interval
    assert cfg.pause_after_steps == 999
    assert cfg.val_games == 128


def test_train_jax_discard_ply_limit_flag() -> None:
    """`--discard-ply-limit` is a store_true that's off unless passed
    (so an absent flag never clobbers a JSON `true`)."""
    assert _train_jax_cfg([]).discard_ply_limit is False
    assert _train_jax_cfg(["--discard-ply-limit"]).discard_ply_limit is True


def test_train_jax_rejects_hf_bucket_only_config() -> None:
    """A bucket-only config must NOT validate. v2's JAX trainer has no
    bucket-push primitive, so accepting `hf_bucket` as a sole destination
    would let a long run train to completion and persist nothing (silent
    total checkpoint loss). `_check_checkpoint_mode` rejects it up front
    with an actionable error. There is deliberately no `--hf-bucket` CLI
    flag, so the only way to set the field is a JSON config; build the
    config directly to pin the validator."""
    from pydantic import ValidationError

    from pawn.run_config import PretrainConfig

    with pytest.raises(ValidationError, match="hf_bucket autosave is not"):
        # `variant="base"` is a named preset, so no arch overrides are
        # needed — `_check_checkpoint_mode` runs regardless of arch.
        PretrainConfig(variant="base", total_steps=1, hf_bucket="ns/bkt")


def test_train_jax_local_checkpoints_actually_writes_checkpoint(
    tmp_path: Path,
) -> None:
    """The chosen checkpoint mode must actually persist a checkpoint — a
    config that merely validates is not enough (regression guard for the
    bucket-mode silent-loss trap). Run a tiny `--local-checkpoints`
    pretrain end-to-end and assert a `step_*` directory lands on disk."""
    import subprocess

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "2",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--checkpoint-interval", "2",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"local-checkpoints pretrain failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    ckpts = sorted(logs_dir.glob("*/step_*"))
    assert ckpts, (
        f"no checkpoint written under {logs_dir} despite a valid "
        f"checkpoint mode; stdout={result.stdout}\nstderr={result.stderr}"
    )
    # The atomic-save sentinel must be present — the checkpoint is
    # actually complete, not a half-written `.tmp`.
    assert (ckpts[-1] / ".complete").exists(), (
        f"checkpoint {ckpts[-1]} missing its .complete sentinel"
    )


def _adapter_cli_cfg(extra: list[str]) -> AdapterConfig:
    mod = _load_adapter_module()
    args = mod._parse_args(
        ["--strategy", "lora", "--lora-rank", "4",
         "--supernet", "tiny", "--variant", "small",
         "--total-steps", "1", "--local-checkpoints", *extra]
    )
    cfg: AdapterConfig = mod._build_config(args)
    return cfg


def test_adapter_lora_ffn_cli_flag() -> None:
    """`--lora-ffn` is exposed directly (was --config-only in v2). Default
    off; the flag sets the AdapterConfig field True."""
    assert _adapter_cli_cfg([]).lora_ffn is False
    assert _adapter_cli_cfg(["--lora-ffn"]).lora_ffn is True


def test_adapter_sparse_ffn_cli_flag() -> None:
    """`--sparse-ffn` is exposed directly. Sparse strategy needs a density;
    drive it through a sparse-strategy config."""
    mod = _load_adapter_module()
    args = mod._parse_args(
        ["--strategy", "sparse", "--density", "0.01", "--sparse-ffn",
         "--supernet", "tiny", "--variant", "small",
         "--total-steps", "1", "--local-checkpoints"]
    )
    cfg = mod._build_config(args)
    assert cfg.sparse_ffn is True


def test_adapter_sparse_targets_cli_flag() -> None:
    """`--sparse-targets` is exposed directly (was --config-only in v2)."""
    mod = _load_adapter_module()
    args = mod._parse_args(
        ["--strategy", "sparse", "--density", "0.01",
         "--sparse-targets", "qv",
         "--supernet", "tiny", "--variant", "small",
         "--total-steps", "1", "--local-checkpoints"]
    )
    cfg = mod._build_config(args)
    assert cfg.sparse_targets == "qv"


def test_adapter_bottleneck_n_hidden_cli_flag() -> None:
    """`--bottleneck-n-hidden` is exposed directly (was --config-only)."""
    mod = _load_adapter_module()
    args = mod._parse_args(
        ["--strategy", "bottleneck", "--bottleneck-dim", "8",
         "--bottleneck-n-hidden", "2",
         "--supernet", "tiny", "--variant", "small",
         "--total-steps", "1", "--local-checkpoints"]
    )
    cfg = mod._build_config(args)
    assert cfg.bottleneck_n_hidden == 2
