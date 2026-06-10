"""Tests for `pawn.sweep` / `pawn.lab` / `pawn.wandb_utils` / `pawn.dashboard.metrics`."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest.mock as mock
from pathlib import Path

import pytest
import optuna

from pawn.dashboard.metrics import MetricsBundle, discover_runs, load_metrics
from pawn.lab.runner import (
    audit_schedule_health,
    lab_launch,
    lab_schema,
    read_schedule_health,
    validate_config,
)
from pawn.sweep import (
    STRATEGY_SUGGESTERS,
    AdapterObjective,
    InProcessRoSAObjective,
    _read_best_val_loss,
    _scan_val_records,
    adapter_strategy_for,
    make_pruner,
    params_to_config_json,
    suggest_bottleneck,
    suggest_common,
    suggest_lora,
    suggest_rosa,
    suggest_rosa_retro_bottleneck,
    suggest_sparse,
    suggest_unfreeze,
)
from pawn.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
    require_wandb_available,
    wandb_available,
)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def test_strategy_suggesters_present_for_all_strategies() -> None:
    """Every adapter strategy has a `suggest_*` function in the
    dispatcher table."""
    assert set(STRATEGY_SUGGESTERS.keys()) >= {
        "lora", "film", "bottleneck", "hybrid",
        "sparse", "rosa", "rosa-retro-sparse", "rosa-retro-bottleneck",
        "unfreeze", "specialized_clm",
    }


def test_suggest_lora_returns_valid_params() -> None:
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_lora(trial)
    assert "lora_rank" in params
    assert "lora_targets" in params
    assert "lr" in params
    assert 1 <= params["lora_rank"] <= 16
    assert params["lora_targets"] in ("qkvo", "qv", "qkv")


def test_suggest_rosa_includes_v1_hyperparams() -> None:
    """Plan §6 + §10 S3: rosa_warmup_steps / mask_samples / grad_alpha
    are non-negotiable v1 hyperparameters; the sweep must search over
    them."""
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_rosa(trial)
    assert "rosa_warmup_steps" in params
    assert "mask_samples" in params
    assert "grad_alpha" in params
    assert params["grad_alpha"] in (1, 2)
    # `lora_targets` is part of the v1 RoSA search space — the LoRA warmup
    # must be free to adapt a non-default projection subset.
    assert "lora_targets" in params
    assert params["lora_targets"] in ("qkvo", "qv", "qkv")


def test_suggest_rosa_retro_bottleneck_sweeps_bottleneck_axes() -> None:
    """v1 ``suggest_retro_bottleneck`` extends the shared RoSA space with the
    Houlsby width *and* the extra-stage depth; the prior v2 code locked both
    to their RoSAConfig defaults. Both must now appear in the search space."""
    study = optuna.create_study()
    params = suggest_rosa_retro_bottleneck(study.ask())
    assert params["rosa_mode"] == "retro-bottleneck"
    assert "bottleneck_dim" in params
    assert params["bottleneck_dim"] in (4, 8, 16)
    assert "bottleneck_n_hidden" in params
    assert params["bottleneck_n_hidden"] in (0, 1, 2)
    # The shared RoSA axes are still present (it builds on suggest_rosa).
    assert "lora_targets" in params
    assert "density" in params


def test_params_to_config_json_preserves_native_types() -> None:
    """H8: suggested params round-trip through a JSON `--config` body as
    native types (bool stays bool, int stays int) — not kebab CLI flags
    the adapter argparse never registered. The body carries `run_type` and
    the resolved adapter `strategy`."""
    body = params_to_config_json(
        "bottleneck",
        {"lora_rank": 4, "use_output_film": True, "no_adapt_attn": False},
    )
    assert body["run_type"] == "adapter"
    assert body["strategy"] == "bottleneck"
    assert body["lora_rank"] == 4
    assert body["use_output_film"] is True
    assert body["no_adapt_attn"] is False  # native False preserved, not dropped
    # The body must be JSON-serialisable (this is what gets written to
    # `--config`); native bool/int survive the round-trip unchanged.
    assert json.loads(json.dumps(body)) == body


def test_rosa_ratio_maps_to_consumed_rosa_strategy() -> None:
    """H9: `rosa-ratio` is a sweep-only key with no `--strategy` of its
    own. `adapter_strategy_for` resolves it to the real `rosa` strategy,
    and its suggester emits only consumed `AdapterConfig` fields (a concrete
    `bottleneck_dim`, never the unconsumed `bottleneck_ratio` key that the
    old code emitted and `extra=forbid` rejected)."""
    assert adapter_strategy_for("rosa-ratio") == "rosa"
    # Non-alias strategies pass through unchanged.
    assert adapter_strategy_for("bottleneck") == "bottleneck"
    study = optuna.create_study()
    params = STRATEGY_SUGGESTERS["rosa-ratio"](study.ask())
    assert "bottleneck_ratio" not in params  # raw ratio is not a config field
    assert params["rosa_mode"] == "retro-bottleneck"
    assert isinstance(params["bottleneck_dim"], int)
    assert params["bottleneck_dim"] >= 1


def _load_adapter_script():
    """Import `scripts/train_jax_adapter.py` by path (scripts/ isn't a
    package). Imports JAX at module load — used only by the argparse +
    pydantic acceptance test and the GPU sweep smokes below."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_adapter_accept", Path("scripts/train_jax_adapter.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("sweep_strategy", sorted(STRATEGY_SUGGESTERS))
def test_suggested_params_accepted_by_adapter_argparse_and_pydantic(
    sweep_strategy: str, tmp_path: Path
) -> None:
    """H8/H9 acceptance gate: every `STRATEGY_SUGGESTERS` entry's suggested
    params, serialized to a `--config` JSON exactly as `AdapterObjective`
    does, are accepted by `train_jax_adapter.py`'s argparse + `AdapterConfig`
    pydantic boundary — no `SystemExit` (argparse exit-2), no
    `pydantic.ValidationError` (`extra=forbid` / required-field).

    This is the previously-broken path: kebab CLI flags for
    `bottleneck_n_hidden` / `sparse_targets` / `rosa_*` / `mask_samples` /
    `grad_alpha` were never registered (exit-2), and `bottleneck_ratio` was
    not a config field (`extra=forbid`). The JSON `--config` round-trip plus
    the `rosa-ratio`→`rosa` alias fix both regimes.
    """
    adapter = _load_adapter_script()
    study = optuna.create_study()
    # Mirror AdapterObjective: suggest params, render the config body, write
    # it to disk, and build the argv the objective would run.
    params = STRATEGY_SUGGESTERS[sweep_strategy](study.ask(), n_layers=4)
    body = params_to_config_json(sweep_strategy, params)
    config_path = tmp_path / "trial_config.json"
    config_path.write_text(json.dumps(body))
    argv = [
        "--config", str(config_path),
        "--strategy", adapter_strategy_for(sweep_strategy),
        "--logs-dir", str(tmp_path / "logs"),
        # base_args the sweep CLI supplies (scripts/sweep.py); these don't
        # collide with the suggested params and satisfy the required
        # total_steps / checkpoint-mode pydantic gates. `--batch-size` is
        # NOT supplied here: `batch_size` is now a swept axis (suggest_common),
        # so a CLI override would clobber the per-trial suggested value in the
        # config merge — exactly what scripts/sweep.py base_args also avoids.
        "--supernet", "tiny", "--variant", "base",
        "--total-steps", "10", "--log-interval", "2",
        "--no-pgn", "--seq-len", "32", "--k", "5",
        "--local-checkpoints",
    ]
    # Must not raise SystemExit (argparse) or ValidationError (pydantic).
    args = adapter._parse_args(argv)
    cfg = adapter._build_config(args)
    # The resolved strategy must be a real adapter strategy, and the config
    # must carry the suggested values (spot-check the non-default knobs).
    from pawn.adapter_trainer import STRATEGIES

    assert cfg.strategy in STRATEGIES
    assert cfg.strategy == adapter_strategy_for(sweep_strategy)
    for key, val in params.items():
        # `rosa_mode` is the sub-mode selector; every other suggested key is
        # a direct AdapterConfig field that must survive the round-trip.
        assert getattr(cfg, key) == val, (
            f"{sweep_strategy}: config dropped suggested {key}={val!r}"
        )


def test_read_best_val_loss_finds_minimum(tmp_path: Path) -> None:
    """`_read_best_val_loss` walks the dir and returns the smallest
    val_loss across all metrics.jsonl files."""
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "train", "loss": 5.0}),
        json.dumps({"type": "val", "val_loss": 3.0}),
        json.dumps({"type": "val", "val_loss": 2.0}),
        json.dumps({"type": "val", "loss": 2.5}),  # accept "loss" too
    ]))
    assert _read_best_val_loss(tmp_path) == 2.0


def test_read_best_val_loss_returns_inf_on_no_records(tmp_path: Path) -> None:
    assert _read_best_val_loss(tmp_path) == float("inf")


# ---------------------------------------------------------------------------
# Sweep — restored search-space axes (common hparams + FFN)
# ---------------------------------------------------------------------------


def test_suggest_common_has_v1_training_axes() -> None:
    """v1 ``suggest_common`` (``git show main:pawn/sweep.py:48-56``) swept
    ``batch_size`` / ``weight_decay`` / ``warmup_frac`` / ``patience`` on top
    of ``lr``; the prior v2 suggesters dropped all four. Re-add them — every
    key is an ``AdapterConfig`` field so it round-trips through ``--config``."""
    study = optuna.create_study()
    params = suggest_common(study.ask())
    assert set(params) == {
        "lr", "batch_size", "weight_decay", "warmup_frac", "patience",
    }
    assert params["batch_size"] in (32, 64, 128, 256)
    assert 0.0 <= params["weight_decay"] <= 0.1
    assert 0.0 <= params["warmup_frac"] <= 0.15
    assert 5 <= params["patience"] <= 20


def test_every_strategy_sweeps_common_training_axes() -> None:
    """Each strategy's space (except ``rosa-ratio``, which v1 deliberately
    fixes its nuisances) carries the shared ``suggest_common`` axes, not just
    ``lr`` plus the adapter-specific knobs."""
    common = {"batch_size", "weight_decay", "warmup_frac", "patience"}
    for name, suggester in STRATEGY_SUGGESTERS.items():
        if name == "rosa-ratio":
            continue  # v1 fixes weight_decay/warmup_frac/patience/batch_size
        study = optuna.create_study()
        params = suggester(study.ask(), n_layers=4)
        assert common <= set(params), f"{name} dropped {common - set(params)}"


def test_suggest_lora_sweeps_ffn_axis() -> None:
    """v1 ``suggest_lora`` swept ``lora_ffn`` (whether LoRA also targets the
    FFN projections); the field exists on ``AdapterConfig`` but the prior v2
    suggester never set it."""
    study = optuna.create_study()
    params = suggest_lora(study.ask())
    assert "lora_ffn" in params
    assert isinstance(params["lora_ffn"], bool)


def test_suggest_sparse_sweeps_ffn_axis() -> None:
    """v1 ``suggest_sparse`` swept ``sparse_ffn``."""
    study = optuna.create_study()
    params = suggest_sparse(study.ask())
    assert "sparse_ffn" in params
    assert isinstance(params["sparse_ffn"], bool)


def test_suggest_bottleneck_sweeps_both_placement_toggles() -> None:
    """v1 ``suggest_bottleneck`` swept both ``no_adapt_attn`` *and*
    ``no_adapt_ffn`` as independent booleans; the prior v2 suggester only
    swept the attn toggle.

    v2's ``BottleneckConfig`` rejects the both-disabled combination
    (no_adapt_attn=True AND no_adapt_ffn=True) — the bottleneck would touch
    nothing — whereas v1 tolerated it as a no-op. So the v2 suggester must
    still emit *both* placement booleans (parity: the placement axis is
    swept, not just attn) **without** ever producing the invalid
    both-disabled pair. It samples a single 3-way placement choice; assert
    both keys are present bools and that at least one site is always adapted
    across a generous batch of draws (every draw must be a config the v2
    ``BottleneckConfig`` accepts)."""
    study = optuna.create_study()
    seen: set[tuple[bool, bool]] = set()
    for _ in range(64):
        params = suggest_bottleneck(study.ask())
        assert "no_adapt_attn" in params
        assert "no_adapt_ffn" in params
        assert isinstance(params["no_adapt_attn"], bool)
        assert isinstance(params["no_adapt_ffn"], bool)
        # The v2 invariant the bottleneck guard enforces: never both-disabled.
        assert not (params["no_adapt_attn"] and params["no_adapt_ffn"])
        seen.add((params["no_adapt_attn"], params["no_adapt_ffn"]))
    # Both placement axes are genuinely explored (not a hardcoded constant):
    # over 64 draws we expect to see the attn-only and ffn-only placements
    # (which flip each toggle on independently) alongside the both-on default.
    assert (True, False) in seen  # ffn-only: attn site skipped
    assert (False, True) in seen  # attn-only: ffn site skipped


def test_suggest_unfreeze_registers_single_lr_axis() -> None:
    """``suggest_unfreeze`` wants a narrower LR band than the common adapter
    range, but must register exactly ONE Optuna ``lr`` parameter — not the
    common ``lr`` axis *and* a stray ``unfreeze_lr``.

    The prior code called ``suggest_common`` (registering ``"lr"`` over
    1e-5..1e-2) and then overwrote ``params["lr"]`` with a second
    ``trial.suggest_float("unfreeze_lr", ...)``. That left two effects: the
    common ``"lr"`` axis was sampled-but-discarded (a wasted search dimension
    feeding the surrogate misleading data), and ``trial.params`` /
    ``study.best_params`` carried a stray ``"unfreeze_lr"`` key that
    ``AdapterConfig`` (``extra="forbid"``) later rejects — so the best trial
    is not reproducible. Assert the Optuna parameter namespace, not just the
    returned dict (the returned ``params`` dict was correct even with the
    bug)."""
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_unfreeze(trial, n_layers=4)
    # The OPTUNA parameter namespace must carry a single `lr` axis and no
    # stray `unfreeze_lr` — this is what `study.best_params` is built from.
    assert "lr" in trial.params
    assert "unfreeze_lr" not in trial.params
    # And the narrower unfreeze LR band is what got registered under `lr`.
    assert 1e-6 <= trial.params["lr"] <= 1e-3
    # The returned dict's `lr` is the same single value (no discarded axis).
    assert params["lr"] == trial.params["lr"]
    # All common axes survive under their canonical names.
    assert {"batch_size", "weight_decay", "warmup_frac", "patience"} <= set(
        trial.params
    )


def test_unfreeze_suggested_params_reproducible_through_adapter_config() -> None:
    """The exact dict ``suggest_unfreeze`` returns must round-trip through
    ``params_to_config_json`` → ``AdapterConfig`` so the best trial can be
    re-run from its suggested params. A stray ``"unfreeze_lr"`` key — which
    the previous code leaked into ``trial.params`` and would carry into a
    reproduction body — makes ``AdapterConfig(extra="forbid")`` reject the
    config. Build the body exactly as ``AdapterObjective`` does and assert it
    validates with the narrower unfreeze LR intact."""
    from pawn.run_config import AdapterConfig

    study = optuna.create_study(direction="minimize")
    params = suggest_unfreeze(study.ask(), n_layers=4)
    # `params` is what `AdapterObjective` serialises to the `--config` body.
    body = params_to_config_json("unfreeze", params)
    body.update({"local_checkpoints": True, "total_steps": 10})
    cfg = AdapterConfig(**body)  # extra="forbid" — no stray `unfreeze_lr`
    assert cfg.strategy == "unfreeze"
    assert cfg.unfreeze_layers == params["unfreeze_layers"]
    assert 1e-6 <= cfg.lr <= 1e-3


# ---------------------------------------------------------------------------
# Sweep — pruner factory + streaming val-record reader
# ---------------------------------------------------------------------------


def test_make_pruner_builds_named_pruners() -> None:
    """``make_pruner`` mirrors v1's pruner switch — median / hyperband /
    none — so a study can opt into mid-trial pruning (plan §10 S9)."""
    assert isinstance(make_pruner("median"), optuna.pruners.MedianPruner)
    assert isinstance(make_pruner("hyperband"), optuna.pruners.HyperbandPruner)
    assert isinstance(make_pruner("none"), optuna.pruners.NopPruner)
    with pytest.raises(ValueError, match="unknown pruner"):
        make_pruner("bogus")


def test_scan_val_records_streams_incrementally(tmp_path: Path) -> None:
    """``_scan_val_records`` tails a growing metrics.jsonl: it returns only
    the new ``type=val`` ``(step, val_loss)`` rows since the carried byte
    offset, leaving a trailing partial line for the next poll. This is the
    primitive that feeds ``AdapterObjective``'s mid-trial ``trial.report``."""
    jsonl = tmp_path / "metrics.jsonl"
    jsonl.write_text("\n".join([
        json.dumps({"type": "config", "slug": "x"}),
        json.dumps({"type": "train", "step": 1, "loss": 5.0}),
        json.dumps({"type": "val", "step": 2, "val_loss": 3.0}),
    ]) + "\n")
    records, offset = _scan_val_records(jsonl, 0)
    assert records == [(2, 3.0)]
    # A second poll with no new data yields nothing and keeps the offset.
    again, offset2 = _scan_val_records(jsonl, offset)
    assert again == []
    assert offset2 == offset
    # Append a complete record + a trailing *partial* line; only the
    # complete one is returned and the partial is left for the next poll.
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "val", "step": 4, "val_loss": 2.0}) + "\n")
        fh.write('{"type": "val", "step": 6, "val_loss"')  # no newline
    records3, offset3 = _scan_val_records(jsonl, offset)
    assert records3 == [(4, 2.0)]
    # Finish the partial line; the next poll picks it up exactly once.
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(": 1.0}\n")
    records4, _ = _scan_val_records(jsonl, offset3)
    assert records4 == [(6, 1.0)]


def test_scan_val_records_zero_val_loss_not_misread(tmp_path: Path) -> None:
    """A legitimate ``val_loss=0.0`` must be reported, not dropped as falsy
    (the same `or`-fallback bug ``_read_best_val_loss`` guards against)."""
    jsonl = tmp_path / "metrics.jsonl"
    jsonl.write_text(
        json.dumps({"type": "val", "step": 1, "val_loss": 0.0, "loss": 9.0})
        + "\n"
    )
    records, _ = _scan_val_records(jsonl, 0)
    assert records == [(1, 0.0)]


# ---------------------------------------------------------------------------
# Sweep — AdapterObjective mid-trial pruning (no GPU; subprocess stubbed)
# ---------------------------------------------------------------------------


class _StubProc:
    """Minimal `subprocess.Popen`-shaped stub: it 'runs' by appending val
    records to the trial's metrics.jsonl across `wait()` polls, so the
    objective's streaming pruner has data to report without launching a
    real training subprocess."""

    def __init__(self, jsonl: Path, val_losses: list[float]) -> None:
        self._jsonl = jsonl
        self._remaining = list(enumerate(val_losses))
        self.returncode: int | None = None
        self.stdout = None
        self.stderr = None
        self._jsonl.parent.mkdir(parents=True, exist_ok=True)

    def wait(self, timeout: float | None = None) -> int:
        # Each poll emits one more val row, then exits when drained.
        if self._remaining:
            step, vl = self._remaining.pop(0)
            with self._jsonl.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"type": "val", "step": step, "val_loss": vl})
                    + "\n"
                )
            raise subprocess.TimeoutExpired(cmd="stub", timeout=timeout or 0)
        self.returncode = 0
        return 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self._remaining = []  # stop streaming; the next wait() exits cleanly
        self.returncode = 0

    def kill(self) -> None:
        self._remaining = []
        self.returncode = 0


def test_adapter_objective_prunes_mid_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E (S9): ``AdapterObjective`` reports each held-out ``val_loss`` to
    ``trial.report`` *while the subprocess runs* and aborts the subprocess
    when the study's pruner says ``should_prune()`` — true mid-trial pruning,
    not a post-hoc parse. A trial whose reported losses are far worse than a
    completed baseline is pruned before its subprocess finishes.

    No GPU: the training subprocess is stubbed by `_StubProc`, which streams
    val rows into the trial's metrics.jsonl across poll cycles.
    """
    import subprocess as _sp

    logs = tmp_path / "sweep"
    # Median pruner with no warmup/startup grace so the first bad report can
    # prune (default MedianPruner needs ≥1 prior completed trial + warmup).
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1, n_warmup_steps=0, interval_steps=1
        ),
    )

    metrics_path = logs / "trial_00000" / "run" / "metrics.jsonl"

    def fake_popen(cmd: list[str], **kw: object) -> _StubProc:
        # The trial's metrics dir is derived from trial.number; trial 0's
        # logs land under trial_00000/. The first (baseline) trial reports a
        # good curve; the second reports a bad one and must be pruned.
        nonlocal metrics_path
        return _StubProc(metrics_path, fake_popen.curve)  # type: ignore[attr-defined]

    fake_popen.curve = [1.0, 0.9, 0.8, 0.7]  # type: ignore[attr-defined]
    monkeypatch.setattr(_sp, "Popen", fake_popen)

    obj = AdapterObjective(
        strategy="lora", base_args=[], logs_dir=logs, poll_interval=0.0,
    )
    # Override the suggester so we don't need a real backbone / argparse.
    monkeypatch.setattr(
        "pawn.sweep.STRATEGY_SUGGESTERS",
        {"lora": lambda trial, **_kw: {"lr": trial.suggest_float(
            "lr", 1e-5, 1e-2, log=True)}},
    )

    # Trial 0: baseline good curve → completes, value 0.7.
    trial0 = study.ask()
    v0 = obj(trial0)
    study.tell(trial0, v0)
    assert v0 == pytest.approx(0.7)

    # Trial 1: a strictly-worse curve. Point the stub at trial 1's dir and
    # feed losses well above the baseline median so the pruner fires.
    metrics_path = logs / "trial_00001" / "run" / "metrics.jsonl"
    fake_popen.curve = [9.0, 9.0, 9.0, 9.0]  # type: ignore[attr-defined]
    trial1 = study.ask()
    with pytest.raises(optuna.TrialPruned):
        obj(trial1)


def test_adapter_objective_does_not_deadlock_on_chatty_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``AdapterObjective`` must drive a real subprocess that
    writes FAR more than one OS pipe buffer (~64 KB) of stdout/stderr without
    dead-locking.

    A previous revision launched the trial with ``stdout=PIPE/stderr=PIPE``
    and never read either stream during the poll loop (it only read stderr
    *after* the loop). Once the trainer's XLA/JAX log chatter filled the pipe
    buffer the child blocked on ``write()``, stopped flushing val records, and
    the trial hung forever (default ``timeout=None`` gave no deadline either).
    This test runs a real Python subprocess that prints ~512 KB to stdout +
    stderr *and then* writes a val record — far past the buffer limit — so a
    PIPE-without-drain implementation would hang here. The fix redirects the
    child's output to a per-trial file, which never blocks the writer.

    No GPU / training involved: the 'trainer' is a tiny inline Python script.
    """
    logs = tmp_path / "sweep"
    trial_logs = logs / "trial_00000"
    trial_logs.mkdir(parents=True, exist_ok=True)  # `__call__` makes this in prod
    metrics = trial_logs / "run" / "metrics.jsonl"
    # A standalone trainer stand-in: spew >64 KB to stdout and stderr, then
    # write a single val record to the metrics path the objective tails.
    script = tmp_path / "chatty_trainer.py"
    script.write_text(
        "import sys, json, pathlib\n"
        f"p = pathlib.Path({str(metrics)!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        # ~512 KB each — well past the ~64 KB pipe buffer.
        "blob = 'x' * 1024\n"
        "for _ in range(512):\n"
        "    sys.stdout.write(blob + '\\n'); sys.stderr.write(blob + '\\n')\n"
        "sys.stdout.flush(); sys.stderr.flush()\n"
        "p.write_text(json.dumps({'type': 'val', 'step': 1, "
        "'val_loss': 0.5}) + '\\n')\n"
    )

    obj = AdapterObjective(
        strategy="lora",
        base_args=[],
        logs_dir=logs,
        poll_interval=0.05,
        timeout=120.0,  # finite ceiling — a wedged child can't hang the study
    )
    # Drive `_run_with_pruning` directly against our chatty script (no real
    # backbone / argparse): override `__call__` to invoke the script and tail
    # the trial's metrics.jsonl exactly as the real objective would.
    monkeypatch.setattr(
        AdapterObjective,
        "__call__",
        lambda self, trial: self._run_with_pruning(
            trial, [sys.executable, str(script)], trial_logs
        ),
    )

    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    # If the implementation PIPE-deadlocks, this call never returns and the
    # test times out at the suite level; on the file-redirect path it returns
    # the val_loss the chatty script emitted.
    value = obj(trial)
    assert value == pytest.approx(0.5)
    # The captured chatter landed in the per-trial log file (proving it was
    # redirected to a file, not silently dropped or piped).
    log_text = (trial_logs / "subprocess.log").read_text()
    assert len(log_text) > 64 * 1024


# ---------------------------------------------------------------------------
# Sweep — end-to-end tiny-trial smoke (subprocess per trial; GPU-only)
# ---------------------------------------------------------------------------


def _gpu_only() -> None:
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("sweep subprocess trains via train_jax_adapter (GPU-only)")


def _local_tiny_backbone(ckpt_dir: Path) -> Path:
    """Persist a TINY_SUPERNET-shaped local backbone checkpoint so the sweep
    smokes don't depend on the (unpublished) `pawn-base-v2` HF repo — the
    spec's "local backbone" smoke setup."""
    from pawn.checkpoint import save_model
    from pawn.config import TINY_SUPERNET
    from pawn.model import init_model

    backbone = init_model(TINY_SUPERNET, key=0)
    save_model(
        backbone, ckpt_dir, training_state={"step": 0},
        run_config={"conditioning": []},
    )
    return ckpt_dir


def _run_tiny_sweep(strategy: str, logs_dir: Path, n_trials: int = 2) -> float:
    """Drive a tiny in-process Optuna sweep through `AdapterObjective`, which
    subprocesses `scripts/train_jax_adapter.py` per trial. Returns
    `study.best_value` — a finite value proves the previously-broken
    argparse/pydantic path now produces non-pruned trials (no all-prune)."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _local_tiny_backbone(logs_dir / "backbone")
    study = optuna.create_study(direction="minimize")
    base_args = [
        "--supernet", "tiny", "--variant", "small",
        "--checkpoint", str(ckpt),
        "--total-steps", "10", "--log-interval", "2",
        # `--batch-size` is swept (suggest_common), so it is supplied via the
        # per-trial config, not hardcoded here (a CLI override would clobber
        # the swept value).
        "--no-pgn", "--seq-len", "32", "--k", "5",
        "--local-checkpoints",
    ]
    obj = AdapterObjective(
        strategy=strategy, base_args=base_args, logs_dir=logs_dir, n_layers=4,
    )
    study.optimize(obj, n_trials=n_trials)
    return study.best_value


@pytest.mark.parametrize(
    "strategy", ["bottleneck", "sparse", "rosa", "rosa-ratio"]
)
def test_tiny_sweep_yields_finite_best_value(
    strategy: str, tmp_path: Path
) -> None:
    """E1 smoke: a 2-trial sweep for each previously-broken strategy
    (`bottleneck`/`sparse` had unregistered kebab flags; a `rosa` sub-mode
    had unregistered `rosa_*` flags; `rosa-ratio` emitted a non-field key)
    completes with a finite `best_value` rather than 100% pruned (H8/H9)."""
    _gpu_only()
    best = _run_tiny_sweep(strategy, tmp_path / "sweep", n_trials=2)
    assert best != float("inf")
    import math

    assert math.isfinite(best), f"{strategy}: best_value not finite: {best}"


# ---------------------------------------------------------------------------
# Sweep — InProcessRoSAObjective (real 3-phase in-process trainer; GPU-only)
# ---------------------------------------------------------------------------


def _tiny_rosa_objective(strategy: str) -> "InProcessRoSAObjective":
    """Build an `InProcessRoSAObjective` over a TINY_SUPERNET backbone + small
    random-game train/val corpora — the in-process variant that loads the
    backbone + data once and runs RoSA's 3 phases per trial."""
    from pawn.config import TINY_SUPERNET
    from pawn.corpus import generate_corpus
    from pawn.model import init_model

    backbone = init_model(TINY_SUPERNET, key=0)
    seq = 64
    train = generate_corpus(64, max_ply=seq, seq_len=seq, seed=1)
    val = generate_corpus(32, max_ply=seq, seq_len=seq, seed=2)
    return InProcessRoSAObjective(
        strategy, backbone, train, val,
        epochs=2, steps_per_epoch=4, val_batches=2,
    )


def test_inprocess_rosa_rejects_non_rosa_strategy() -> None:
    """The in-process objective only drives the RoSA strategy family
    (v1 ``_supported``); a non-RoSA strategy is a constructor error."""
    from typing import cast

    from pawn.corpus import Corpus
    from pawn.model import PAWNModel

    # The strategy guard raises before any backbone/corpus field is read, so
    # placeholder objects suffice; `cast` keeps the call type-correct without
    # paying to build a real backbone + corpora for a pure-validation test.
    dummy = object()
    with pytest.raises(ValueError, match="does not support strategy"):
        InProcessRoSAObjective(
            "lora",
            cast(PAWNModel, dummy),
            cast(Corpus, dummy),
            cast(Corpus, dummy),
        )


@pytest.mark.parametrize("strategy", ["rosa", "rosa-retro-bottleneck"])
def test_inprocess_rosa_yields_finite_best_value(
    strategy: str, tmp_path: Path
) -> None:
    """Behavioral: a 2-trial in-process RoSA sweep runs the full Phase
    1→2→3 schedule per trial and returns a finite `best_value` (the backbone
    + corpora are loaded once and shared). Proves the objective is a real
    trainer, not a dispatch shim."""
    _gpu_only()
    import math

    obj = _tiny_rosa_objective(strategy)
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.NopPruner()
    )
    study.optimize(obj, n_trials=2)
    assert math.isfinite(study.best_value), study.best_value


def test_inprocess_rosa_reports_and_prunes_mid_trial() -> None:
    """Behavioral: the in-process objective feeds each Phase-3 epoch's
    held-out `val_loss` to `trial.report` and aborts the trial on
    `should_prune()` (plan §10 S9; v1 parity). A pruner that always prunes
    raises `TrialPruned`, and the pruner's `prune` hook is observed to have
    been consulted with a recorded intermediate value — distinguishing real
    mid-trial pruning from a post-hoc result parse."""
    _gpu_only()

    class _AlwaysPrune(optuna.pruners.BasePruner):
        consulted_intermediates: list[dict[int, float]] = []

        def prune(self, study: optuna.Study, trial: object) -> bool:
            # `trial` here is a FrozenTrial carrying the intermediate values
            # reported so far; record them so the test can assert the
            # objective reported a val_loss before pruning fired.
            values = getattr(trial, "intermediate_values", {})
            _AlwaysPrune.consulted_intermediates.append(dict(values))
            return True

    obj = _tiny_rosa_objective("rosa")
    study = optuna.create_study(direction="minimize", pruner=_AlwaysPrune())
    with pytest.raises(optuna.TrialPruned):
        obj(study.ask())
    # `prune` is only reachable via `trial.should_prune()`, which the
    # objective calls right after `trial.report(val_loss, epoch)` — so a
    # recorded epoch-0 intermediate proves the report→should_prune path ran.
    assert _AlwaysPrune.consulted_intermediates, "pruner never consulted"
    assert 0 in _AlwaysPrune.consulted_intermediates[0]


def test_inprocess_rosa_forwards_max_grad_norm_to_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioral: the constructor's `max_grad_norm` reaches the optimizer.

    `_train_phases` builds `make_optimizer(sched_cfg, schedule)` and the
    clip threshold is `sched_cfg.max_grad_norm` (`make_optimizer` reads
    `cfg.max_grad_norm`). A non-default `max_grad_norm` passed to the
    objective must therefore appear on the `BaseRunConfig` handed to
    `make_optimizer` — the earlier code stored the arg but built the cfg
    without it, so every in-process RoSA trial silently clipped at the
    `BaseRunConfig` default (1.0) regardless of the caller's value.
    """
    _gpu_only()
    import optax

    import pawn.trainer as trainer_mod
    from pawn.run_config import BaseRunConfig

    captured: list[float] = []
    real_make_optimizer = trainer_mod.make_optimizer

    def spy_make_optimizer(
        cfg: BaseRunConfig, schedule: optax.Schedule
    ) -> optax.GradientTransformation:
        captured.append(cfg.max_grad_norm)
        return real_make_optimizer(cfg, schedule)

    monkeypatch.setattr(trainer_mod, "make_optimizer", spy_make_optimizer)

    from pawn.config import TINY_SUPERNET
    from pawn.corpus import generate_corpus
    from pawn.model import init_model

    backbone = init_model(TINY_SUPERNET, key=0)
    seq = 64
    train = generate_corpus(64, max_ply=seq, seq_len=seq, seed=1)
    val = generate_corpus(32, max_ply=seq, seq_len=seq, seed=2)
    obj = InProcessRoSAObjective(
        "rosa", backbone, train, val,
        epochs=1, steps_per_epoch=4, val_batches=2,
        max_grad_norm=0.5,
    )
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.NopPruner()
    )
    study.optimize(obj, n_trials=1)
    assert captured, "make_optimizer was never called"
    # Every cfg `make_optimizer` saw must carry the custom threshold, not
    # the BaseRunConfig default (1.0).
    assert all(v == 0.5 for v in captured), captured


# ---------------------------------------------------------------------------
# Lab — pydantic validation
# ---------------------------------------------------------------------------


def test_lab_schema_returns_all_run_types() -> None:
    schema = lab_schema()
    assert set(schema.keys()) == {
        "pretrain", "adapter", "specialized_clm", "distill",
    }
    # Each schema is a JSON Schema dict.
    for k, s in schema.items():
        assert "properties" in s


def test_validate_config_dispatches_by_run_type() -> None:
    from pawn.run_config import AdapterConfig, PretrainConfig

    assert isinstance(
        validate_config({
            "run_type": "pretrain", "local_checkpoints": True,
            "total_steps": 100,
        }),
        PretrainConfig,
    )
    assert isinstance(
        validate_config({
            "run_type": "adapter", "local_checkpoints": True,
            "total_steps": 100, "strategy": "lora", "lora_rank": 4,
        }),
        AdapterConfig,
    )


def test_validate_config_rejects_unknown_field() -> None:
    """The lab's pydantic boundary refuses stale field names per
    acceptance criterion 19."""
    with pytest.raises(ValueError, match="extra"):
        validate_config({
            "run_type": "pretrain",
            "local_checkpoints": True,
            "legacy_vocab": True,  # stale v1 field
        })


def test_validate_config_rejects_missing_run_type() -> None:
    with pytest.raises(ValueError, match="run_type"):
        validate_config({"local_checkpoints": True})


def test_validate_config_rejects_unknown_run_type() -> None:
    with pytest.raises(ValueError, match="unknown run_type"):
        validate_config({"run_type": "cotrain", "local_checkpoints": True})


def test_lab_launch_dry_run_validates_without_spawning() -> None:
    """`dry_run=True` validates the config but doesn't actually spawn
    the subprocess."""
    result = lab_launch(
        {
            "run_type": "adapter",
            "local_checkpoints": True,
            "total_steps": 100,
            "strategy": "lora",
            "lora_rank": 4,
        },
        dry_run=True,
    )
    assert result["status"] == "validated"
    assert result["pid"] is None
    assert result["run_type"] == "adapter"


# ---------------------------------------------------------------------------
# Dashboard — metrics loader
# ---------------------------------------------------------------------------


def test_load_metrics_splits_by_type_discriminator(tmp_path: Path) -> None:
    """The dashboard's metrics loader splits records on the `type`
    discriminator (plan §10 S9)."""
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "config", "run_type": "pretrain", "slug": "test"}),
        json.dumps({"type": "train", "step": 1, "loss": 5.0}),
        json.dumps({"type": "train", "step": 2, "loss": 4.0}),
        json.dumps({"type": "val", "step": 2, "loss": 4.5}),
    ]))
    bundle = load_metrics(run)
    assert isinstance(bundle, MetricsBundle)
    assert bundle.config is not None
    assert bundle.config["run_type"] == "pretrain"
    assert bundle.slug == "test"
    assert len(bundle.train_records) == 2
    assert len(bundle.val_records) == 1


def test_load_metrics_tolerates_malformed_lines(tmp_path: Path) -> None:
    """Partial writes / corrupted lines are skipped — the dashboard
    can tail a running file safely."""
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"type": "train", "loss": 1.0}),
        "not valid json {",
        json.dumps({"type": "train", "loss": 0.9}),
    ]))
    bundle = load_metrics(run)
    assert len(bundle.train_records) == 2  # malformed line skipped


def test_discover_runs_finds_all_metrics_jsonl(tmp_path: Path) -> None:
    """`discover_runs` finds every subdirectory with a metrics.jsonl."""
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a/metrics.jsonl").write_text("{}")
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_b/metrics.jsonl").write_text("{}")
    (tmp_path / "no_metrics").mkdir()
    runs = discover_runs(tmp_path)
    assert len(runs) == 2
    assert all(r.name in ("run_a", "run_b") for r in runs)


def test_load_metrics_handles_missing_file_gracefully(tmp_path: Path) -> None:
    bundle = load_metrics(tmp_path / "nonexistent")
    assert bundle.train_records == []
    assert bundle.val_records == []
    assert bundle.config is None


# ---------------------------------------------------------------------------
# W&B — disabled-mode no-ops
# ---------------------------------------------------------------------------


def test_init_wandb_disabled_returns_none() -> None:
    run = init_wandb(
        project="pawn", slug="test", run_config={}, enabled=False
    )
    assert run is None


def test_log_metrics_no_op_on_none() -> None:
    """`log_metrics(None, ...)` is a no-op for disabled W&B."""
    log_metrics(None, {"loss": 1.0})  # must not raise


def test_finish_wandb_no_op_on_none() -> None:
    finish_wandb(None)  # must not raise


def test_init_wandb_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PAWN_WANDB_MODE=disabled` short-circuits even when `enabled=True`."""
    monkeypatch.setenv("PAWN_WANDB_MODE", "disabled")
    run = init_wandb(
        project="pawn", slug="test", run_config={}, enabled=True
    )
    assert run is None


# ---------------------------------------------------------------------------
# W&B — gating (H7: --wandb without the extra is a hard error)
# ---------------------------------------------------------------------------


def test_require_wandb_available_errors_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--wandb` requested but the `wandb` extra absent → SystemExit with
    an actionable message. (Simulate the missing extra by patching the
    availability probe.)"""
    monkeypatch.setattr("pawn.wandb_utils.wandb_available", lambda: False)
    with pytest.raises(SystemExit, match="wandb"):
        require_wandb_available()


def test_require_wandb_available_passes_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the extra is installed, the gate is a no-op."""
    monkeypatch.setattr("pawn.wandb_utils.wandb_available", lambda: True)
    require_wandb_available()  # must not raise


def test_init_wandb_invokes_mirror_with_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the extra present, init_wandb forwards to `wandb.init` and
    `log_metrics` forwards to the run's `.log`. Uses a mocked wandb module
    so no network/login is required."""
    import sys
    import types

    # The test conftest pins PAWN_WANDB_MODE=disabled globally (so no real
    # W&B run is ever created); override to "offline" here so init_wandb
    # reaches the (mocked) wandb.init call rather than short-circuiting.
    monkeypatch.setenv("PAWN_WANDB_MODE", "offline")
    fake_run = mock.MagicMock(name="wandb_run")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = mock.MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    run = init_wandb(
        project="pawn", slug="run-xyz", run_config={"lr": 1e-3},
        git_hash="deadbeef", enabled=True,
    )
    assert run is fake_run
    fake_wandb.init.assert_called_once()  # type: ignore[attr-defined]
    _, kwargs = fake_wandb.init.call_args  # type: ignore[attr-defined]
    assert kwargs["name"] == "run-xyz"
    assert kwargs["config"]["lr"] == 1e-3
    assert kwargs["config"]["git_hash"] == "deadbeef"

    log_metrics(run, {"loss": 0.5}, step=10)
    fake_run.log.assert_called_once_with({"loss": 0.5}, step=10)
    finish_wandb(run)
    fake_run.finish.assert_called_once()


def test_init_wandb_forwards_job_type_group_and_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`init_wandb` exposes the v1 `job_type` / `group` / run-name knobs so
    a single project can separate pretrain vs adapter runs and resumed
    siblings join one group. `run_dir_name` overrides the W&B run name
    (v1 used `logger.run_dir.name`); `group` defaults to the slug."""
    import sys
    import types

    monkeypatch.setenv("PAWN_WANDB_MODE", "offline")
    fake_run = mock.MagicMock(name="wandb_run")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = mock.MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    run = init_wandb(
        project="pawn", slug="bold-fox", run_config={"lr": 1e-3},
        git_hash="deadbeef", enabled=True,
        job_type="adapter", group="sweep-42",
        run_dir_name="lora_20260530_000000_000000_bold-fox",
    )
    assert run is fake_run
    _, kwargs = fake_wandb.init.call_args  # type: ignore[attr-defined]
    assert kwargs["name"] == "lora_20260530_000000_000000_bold-fox"
    assert kwargs["group"] == "sweep-42"
    assert kwargs["job_type"] == "adapter"
    assert "job_type:adapter" in kwargs["tags"]
    assert "git:deadbeef" in kwargs["tags"]


def test_init_wandb_group_defaults_to_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an explicit `group`, resumed sibling processes still join
    one group via the slug (v1 Option-A resume)."""
    import sys
    import types

    monkeypatch.setenv("PAWN_WANDB_MODE", "offline")
    fake_run = mock.MagicMock(name="wandb_run")
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = mock.MagicMock(return_value=fake_run)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    init_wandb(project="pawn", slug="bold-fox", run_config={}, enabled=True)
    _, kwargs = fake_wandb.init.call_args  # type: ignore[attr-defined]
    assert kwargs["group"] == "bold-fox"
    assert kwargs["name"] == "bold-fox"


def test_finish_wandb_forwards_exit_code() -> None:
    """`finish_wandb` records a non-zero exit code so a crashed / SIGTERM'd
    run surfaces as failed in the W&B UI (v1 parity)."""
    fake_run = mock.MagicMock(name="wandb_run")
    finish_wandb(fake_run, exit_code=1)
    fake_run.finish.assert_called_once_with(exit_code=1)


def test_finish_wandb_default_exit_code_zero() -> None:
    fake_run = mock.MagicMock(name="wandb_run")
    finish_wandb(fake_run)
    fake_run.finish.assert_called_once_with(exit_code=0)


def test_wandb_available_returns_bool() -> None:
    assert isinstance(wandb_available(), bool)


# ---------------------------------------------------------------------------
# H7 — lab runner reads schedule_health.json + flags structural mismatch
# ---------------------------------------------------------------------------


def _write_health(run_dir: Path, **fields: object) -> None:
    base = {
        "format_version": 1,
        "schedule": "cosine",
        "should_reach_zero": True,
        "planned_total_steps": 1000,
        "actual_total_steps": 1000,
        "completion_ratio": 1.0,
        "lr_peak": 3e-4,
        "actual_final_lr": 0.0,
        "reason_for_stop": "completed",
    }
    base.update(fields)
    (run_dir / "schedule_health.json").write_text(json.dumps(base))


def test_read_schedule_health_absent_returns_none(tmp_path: Path) -> None:
    assert read_schedule_health(tmp_path) is None


def test_audit_schedule_health_clean_full_run(tmp_path: Path) -> None:
    """A `completed` run whose actual == planned is healthy: no banner."""
    _write_health(tmp_path)
    audit = audit_schedule_health(tmp_path)
    assert audit["present"] is True
    assert audit["structural_mismatch"] is False
    assert audit["banner"] is None


def test_audit_schedule_health_flags_structural_mismatch(
    tmp_path: Path,
) -> None:
    """`actual != planned` AND reason_for_stop == 'completed' is the
    structural-bug signal: the lab runner raises the flag + banner."""
    _write_health(tmp_path, actual_total_steps=500, reason_for_stop="completed")
    audit = audit_schedule_health(tmp_path)
    assert audit["structural_mismatch"] is True
    assert audit["banner"] is not None
    assert "STRUCTURAL MISMATCH" in audit["banner"]


def test_audit_schedule_health_sigterm_is_not_mismatch(tmp_path: Path) -> None:
    """A SIGTERM early exit with actual != planned is a *legitimate*
    early stop, not the structural-bug signal."""
    _write_health(tmp_path, actual_total_steps=500, reason_for_stop="sigterm")
    audit = audit_schedule_health(tmp_path)
    assert audit["structural_mismatch"] is False
    assert audit["banner"] is None


def test_audit_schedule_health_absent_file(tmp_path: Path) -> None:
    audit = audit_schedule_health(tmp_path)
    assert audit["present"] is False
    assert audit["structural_mismatch"] is False
