"""Optuna hyperparameter sweep driver for v2 adapter training.

`AdapterObjective` runs `scripts/train_jax_adapter.py` as a subprocess
per trial and parses `metrics.jsonl` for the best `val_loss`.
`InProcessRoSAObjective` is the v1 in-process variant that skips
per-trial JAX startup for big RoSA sweeps (kept for parity with v1
sweep tooling). Per-strategy `suggest_*` functions match the v1
search-space contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from pawn.adapter_trainer import AdapterTrainState
    from pawn.adapters.rosa import RoSAConfig
    from pawn.corpus import Corpus
    from pawn.model import PAWNModel
    from pawn.trainer import Batch

    # The K-step scan step built by ``make_adapter_scan_step``: it carries an
    # ``AdapterTrainState`` and consumes a ``Batch`` with a leading K axis,
    # returning the advanced state plus the per-step loss vector.
    AdapterScanStep = Callable[
        ["AdapterTrainState", "Batch"], tuple["AdapterTrainState", Any]
    ]

__all__ = [
    "AdapterObjective",
    "InProcessRoSAObjective",
    "make_pruner",
    "suggest_common",
    "suggest_lora",
    "suggest_film",
    "suggest_bottleneck",
    "suggest_hybrid",
    "suggest_sparse",
    "suggest_rosa",
    "suggest_rosa_retro_sparse",
    "suggest_rosa_retro_bottleneck",
    "suggest_rosa_ratio",
    "suggest_unfreeze",
    "suggest_specialized_clm",
    "STRATEGY_SUGGESTERS",
    "adapter_strategy_for",
    "params_to_config_json",
]


# ---------------------------------------------------------------------------
# Per-strategy suggesters — match the v1 search-space contract
# ---------------------------------------------------------------------------


# RoSA retro-bottleneck extra-(Linear+GELU)-stage depth choices — matches
# the v1 ``BOTTLENECK_N_HIDDEN_CHOICES`` search-space constant.
BOTTLENECK_N_HIDDEN_CHOICES: tuple[int, ...] = (0, 1, 2)

# Shared ``suggest_common`` categorical/range constants. ``suggest_unfreeze``
# re-suggests these axes by hand (it can't override ``lr`` after the fact —
# see that function) so the choices live here to stay in lock-step.
_BATCH_SIZE_CHOICES: list[int] = [32, 64, 128, 256]
_WEIGHT_DECAY_RANGE: tuple[float, float] = (0.0, 0.1)
_WARMUP_FRAC_RANGE: tuple[float, float] = (0.0, 0.15)
_PATIENCE_RANGE: tuple[int, int] = (5, 20)


def suggest_common(trial: optuna.Trial) -> dict[str, Any]:
    """Training-loop hyperparameters shared across every strategy — the v1
    ``suggest_common`` axes (``git show main:pawn/sweep.py:48-56``).

    Every key is an ``AdapterConfig`` field, so it round-trips through the
    ``--config`` JSON the objective writes (``params_to_config_json``). The
    prior v2 suggesters narrowed every space to just ``lr`` plus the
    adapter-specific knobs, dropping ``batch_size`` / ``weight_decay`` /
    ``warmup_frac`` / ``patience`` — the search could no longer trade off
    batch size against LR or tune the early-stop budget, both of which v1
    swept on every strategy.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", _BATCH_SIZE_CHOICES),
        "weight_decay": trial.suggest_float("weight_decay", *_WEIGHT_DECAY_RANGE),
        "warmup_frac": trial.suggest_float("warmup_frac", *_WARMUP_FRAC_RANGE),
        "patience": trial.suggest_int("patience", *_PATIENCE_RANGE),
    }


def suggest_lora(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    params = suggest_common(trial)
    params["lora_rank"] = trial.suggest_int("lora_rank", 1, 16)
    params["lora_targets"] = trial.suggest_categorical(
        "lora_targets", ["qkvo", "qv", "qkv"]
    )
    # v1 ``suggest_lora`` swept whether the LoRA adapter also targets the
    # FFN projections (``lora_ffn``); the field exists on ``AdapterConfig``.
    params["lora_ffn"] = trial.suggest_categorical("lora_ffn", [True, False])
    return params


def suggest_film(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    params = suggest_common(trial)
    params["use_output_film"] = trial.suggest_categorical(
        "use_output_film", [True, False]
    )
    return params


def suggest_bottleneck(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    params = suggest_common(trial)
    params["bottleneck_dim"] = trial.suggest_categorical(
        "bottleneck_dim", [4, 8, 16, 32, 64, 128]
    )
    params["bottleneck_n_hidden"] = trial.suggest_categorical(
        "bottleneck_n_hidden", list(BOTTLENECK_N_HIDDEN_CHOICES)
    )
    # v1 ``suggest_bottleneck`` sampled ``no_adapt_attn`` and ``no_adapt_ffn``
    # as two independent booleans, but v1's ``BottleneckConfig`` tolerated the
    # both-disabled combination (it silently produced an empty adapter set / a
    # no-op run). v2's ``BottleneckConfig.__post_init__`` (bottleneck.py:89)
    # now *rejects* (no_adapt_attn=True, no_adapt_ffn=True) — the bottleneck
    # would touch nothing — so an independent-boolean sweep can emit a config
    # that every trial fails on, and a small study can prune 100% of its
    # trials (the E1 smoke test forbids exactly that). Sweep a single
    # 3-way *placement* choice instead so at least one site is always enabled;
    # this preserves the v1 search-space intent (both placement axes are
    # explored: attn-only, ffn-only, and both) while never producing the
    # unreachable-valid both-disabled combination.
    placement = trial.suggest_categorical(
        "bottleneck_placement", ["both", "attn_only", "ffn_only"]
    )
    # attn_only ⇒ skip the FFN site; ffn_only ⇒ skip the attn site; both ⇒
    # adapt everything. (Exactly one toggle is ever True, so the guard above
    # can never fire.)
    params["no_adapt_attn"] = placement == "ffn_only"
    params["no_adapt_ffn"] = placement == "attn_only"
    return params


def suggest_hybrid(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    params = suggest_common(trial)
    params["lora_rank"] = trial.suggest_int("lora_rank", 1, 8)
    # v1 ``suggest_hybrid`` swept the LoRA projection preset too — the
    # hybrid adapter's LoRA branch adapts the same ``qkvo`` / ``qv`` / ``qkv``
    # subsets as the standalone LoRA strategy.
    params["lora_targets"] = trial.suggest_categorical(
        "lora_targets", ["qkvo", "qv", "qkv"]
    )
    params["use_output_film"] = trial.suggest_categorical(
        "use_output_film", [True, False]
    )
    return params


def suggest_sparse(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    params = suggest_common(trial)
    params["density"] = trial.suggest_float("density", 0.001, 0.1, log=True)
    params["sparse_targets"] = trial.suggest_categorical(
        "sparse_targets", ["qkvo", "qv", "qkv"]
    )
    # v1 ``suggest_sparse`` swept whether the sparse mask also covers the
    # FFN projections (``sparse_ffn``); the field exists on ``AdapterConfig``.
    params["sparse_ffn"] = trial.suggest_categorical("sparse_ffn", [True, False])
    return params


def suggest_rosa(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    params = suggest_common(trial)
    params["rosa_mode"] = "rosa"
    params["lora_rank"] = trial.suggest_int("lora_rank", 1, 8)
    # `lora_targets` is part of the v1 RoSA search space
    # (``_suggest_rosa_common``): RoSA's LoRA warmup can adapt the
    # ``qv`` / ``qkv`` projection subsets, not just the default
    # ``qkvo``, so the sweep must be free to pick the preset.
    params["lora_targets"] = trial.suggest_categorical(
        "lora_targets", ["qkvo", "qv", "qkv"]
    )
    params["density"] = trial.suggest_float("density", 0.001, 0.1, log=True)
    params["rosa_warmup_steps"] = trial.suggest_int("rosa_warmup_steps", 32, 512)
    params["mask_samples"] = trial.suggest_int("mask_samples", 8, 64)
    params["grad_alpha"] = trial.suggest_categorical("grad_alpha", [1, 2])
    return params


def suggest_rosa_retro_sparse(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    return {**suggest_rosa(trial), "rosa_mode": "retro-sparse"}


def suggest_rosa_retro_bottleneck(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    # The retro-bottleneck mode instantiates a Houlsby bottleneck in
    # Phase 3, so (per v1 ``suggest_retro_bottleneck``) the sweep extends
    # the shared RoSA space with the bottleneck width *and* the extra-stage
    # depth — the prior v2 code locked both to their RoSAConfig defaults.
    return {
        **suggest_rosa(trial),
        "rosa_mode": "retro-bottleneck",
        "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", [4, 8, 16]),
        "bottleneck_n_hidden": trial.suggest_categorical(
            "bottleneck_n_hidden", list(BOTTLENECK_N_HIDDEN_CHOICES)
        ),
    }


def suggest_rosa_ratio(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    """Sweep over the bottleneck-vs-sparse parameter split (RoSA-specific
    v1 sweep).

    H9: ``rosa-ratio`` is a *sweep-only* strategy key — it is not a valid
    adapter ``--strategy`` (``STRATEGIES`` has no ``rosa-ratio`` entry). The
    objective maps it onto the real ``rosa`` strategy via
    :func:`adapter_strategy_for`; the sub-mode this suggester sweeps is
    ``retro-bottleneck`` (the only RoSA mode that actually instantiates a
    Houlsby bottleneck), so the bottleneck-vs-sparse split is a meaningful
    knob here. The historical ``bottleneck_ratio`` ∈ (0, 1) is translated
    into the *consumed* ``bottleneck_dim`` AdapterConfig field — the prior
    code emitted a raw ``bottleneck_ratio`` key that no config or adapter
    consumed, so every trial was rejected by ``extra="forbid"``.

    ``bottleneck_dim`` is derived as ``round(ratio · _ROSA_RATIO_DIM_SPAN)``
    (clamped to ≥1) so the swept ratio maps onto a small positive integer
    bottleneck width.
    """
    ratio = trial.suggest_float("bottleneck_ratio", 0.1, 0.9)
    bottleneck_dim = max(1, round(ratio * _ROSA_RATIO_DIM_SPAN))
    return {
        "rosa_mode": "retro-bottleneck",
        "lora_rank": trial.suggest_int("lora_rank", 1, 8),
        "density": trial.suggest_float("density", 0.001, 0.1, log=True),
        "bottleneck_dim": bottleneck_dim,
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


# The integer bottleneck width that a swept ``bottleneck_ratio`` of 1.0
# would map to (``bottleneck_dim = round(ratio · span)``). Picked so the
# (0.1, 0.9) ratio range spans a small-but-non-degenerate set of widths.
_ROSA_RATIO_DIM_SPAN = 32


def suggest_unfreeze(
    trial: optuna.Trial, *, n_layers: int = 10
) -> dict[str, Any]:
    """Sample an unfreeze-layers spec.

    ``n_layers`` is the total depth of the targeted backbone. Defaults
    to 10 (production SUPERNET), but the sweep driver should pass the
    actual variant depth so the suggester doesn't propose layer indices
    that ``init_unfreeze_adapter`` will reject — e.g. for the tiny
    supernet (n_layers=4) suggesting "7,8,9" kills every trial.
    """
    max_unfrozen = min(4, n_layers)
    n = trial.suggest_int("n_layers_unfrozen", 1, max_unfrozen)
    # Always unfreeze the last N layers — a reasonable v1-style default.
    layers = ",".join(str(i) for i in range(n_layers - n, n_layers))
    # Unfreeze fine-tunes real backbone weights, so it wants a smaller LR
    # range than the common adapter band. We can't call ``suggest_common`` and
    # then override ``lr``: that would register TWO Optuna parameters (the
    # common ``"lr"`` *and* a second ``"unfreeze_lr"``) — the common ``"lr"``
    # axis would be sampled-but-discarded (a wasted search dimension feeding
    # the surrogate misleading data) and ``study.best_params`` would carry the
    # stray ``"unfreeze_lr"`` key, which ``AdapterConfig`` (``extra="forbid"``)
    # rejects → non-reproducible best trial. Instead hand-suggest each common
    # axis under its canonical name, with the narrower ``lr`` range registered
    # once under ``"lr"``.
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", _BATCH_SIZE_CHOICES),
        "weight_decay": trial.suggest_float("weight_decay", *_WEIGHT_DECAY_RANGE),
        "warmup_frac": trial.suggest_float("warmup_frac", *_WARMUP_FRAC_RANGE),
        "patience": trial.suggest_int("patience", *_PATIENCE_RANGE),
        "unfreeze_layers": layers,
    }


def suggest_specialized_clm(trial: optuna.Trial, **_kw: Any) -> dict[str, Any]:
    # v1 ``suggest_tiny`` built on ``suggest_common`` (the shared training
    # axes) before adding the architecture knobs — keep that so the
    # from-scratch standalone sweep tunes batch size / weight decay /
    # warmup / patience like every other strategy.
    params = suggest_common(trial)
    d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128, 192])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    # Round d_model up to a multiple of n_heads if needed.
    if d_model % n_heads != 0:
        d_model = ((d_model + n_heads - 1) // n_heads) * n_heads
    params["d_model"] = d_model
    params["n_layers"] = trial.suggest_int("n_layers", 2, 6)
    params["n_heads"] = n_heads
    params["d_ff"] = 4 * d_model
    return params


STRATEGY_SUGGESTERS: dict[str, Callable[..., dict[str, Any]]] = {
    "lora": suggest_lora,
    "film": suggest_film,
    "bottleneck": suggest_bottleneck,
    "hybrid": suggest_hybrid,
    "sparse": suggest_sparse,
    "rosa": suggest_rosa,
    "rosa-retro-sparse": suggest_rosa_retro_sparse,
    "rosa-retro-bottleneck": suggest_rosa_retro_bottleneck,
    "rosa-ratio": suggest_rosa_ratio,
    "unfreeze": suggest_unfreeze,
    "specialized_clm": suggest_specialized_clm,
}


# Sweep-only strategy keys that are NOT valid adapter `--strategy` values —
# they map onto a real `STRATEGIES` strategy whose `rosa_mode` the suggester
# sets explicitly (H9). `rosa-ratio` sweeps the bottleneck-vs-sparse split of
# the `rosa` strategy in its `retro-bottleneck` sub-mode.
_SWEEP_STRATEGY_ALIASES: dict[str, str] = {
    "rosa-ratio": "rosa",
}


def adapter_strategy_for(sweep_strategy: str) -> str:
    """Map a `STRATEGY_SUGGESTERS` key onto the adapter `--strategy` it runs.

    Most sweep keys are identical to the adapter strategy; sweep-only
    aliases (currently just `rosa-ratio`) resolve to the real strategy the
    `train_jax_adapter` argparse + pydantic accept (H9)."""
    return _SWEEP_STRATEGY_ALIASES.get(sweep_strategy, sweep_strategy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def params_to_config_json(strategy: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON `--config` body for one suggested trial (H8).

    The previous design rendered the suggested-params dict into kebab-cased
    CLI argv (`{"lora_rank": 4}` → `["--lora-rank", "4"]`). Many suggested
    keys (`bottleneck_n_hidden`, `sparse_targets`, `rosa_warmup_steps`,
    `mask_samples`, `grad_alpha`, …) have **no** registered argparse flag in
    `scripts/train_jax_adapter.py`, so argparse exited 2 and every trial was
    pruned. Routing the params through a temp-JSON `--config` instead lets
    them round-trip through `AdapterConfig` (pydantic) directly — the
    `--config` path in `_build_config` `json.loads`es this body and merges
    CLI flags on top.

    The returned dict carries the suggested params verbatim (every key is an
    `AdapterConfig` field — `extra="forbid"` would reject a stray one) plus
    `run_type="adapter"` and the resolved adapter `strategy` (sweep-only
    aliases like `rosa-ratio` are mapped to their real strategy via
    :func:`adapter_strategy_for`). Boolean and int values are preserved as
    native JSON types rather than stringified, so the config validates with
    the right field types.
    """
    body: dict[str, Any] = dict(params)
    body["run_type"] = "adapter"
    body["strategy"] = adapter_strategy_for(strategy)
    return body


def _read_best_val_loss(logs_dir: Path) -> float:
    """Walk `logs_dir/**/metrics.jsonl`, find the smallest `val_loss`
    on a `type=val` record."""
    best = float("inf")
    for jsonl in logs_dir.rglob("metrics.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "val":
                continue
            # `or` would silently fall back to `loss` if `val_loss` is
            # 0.0 (legitimate) or NaN-sanitised null. Use explicit
            # presence checks so a trial with `val_loss=0.0` isn't
            # mis-keyed on the training loss.
            if "val_loss" in rec:
                v = rec["val_loss"]
            elif "loss" in rec:
                v = rec["loss"]
            else:
                v = None
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if vf < best:
                best = vf
    return best


def _val_loss_from_record(rec: dict[str, Any]) -> float | None:
    """Pull the held-out ``val_loss`` (falling back to ``loss``) off one
    parsed ``type=val`` metrics record, or ``None`` if absent / non-numeric.

    Shares the explicit-presence + numeric-coercion contract of
    :func:`_read_best_val_loss` so the streaming pruning reader keys on the
    same value the final best-val parse does (a ``val_loss`` of ``0.0`` is
    legitimate and must not fall through to the training ``loss``)."""
    if rec.get("type") != "val":
        return None
    if "val_loss" in rec:
        v = rec["val_loss"]
    elif "loss" in rec:
        v = rec["loss"]
    else:
        return None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _scan_val_records(
    jsonl_path: Path, start_offset: int
) -> tuple[list[tuple[int, float]], int]:
    """Read new lines appended to ``jsonl_path`` since ``start_offset`` and
    return ``([(step, val_loss), ...], new_offset)``.

    Used by :class:`AdapterObjective` to tail the trial's ``metrics.jsonl``
    while the training subprocess is still running, so each held-out
    ``val_loss`` can be fed to ``trial.report`` for mid-trial pruning
    (plan §10 S9). The byte offset is carried across polls so each line is
    parsed exactly once; a trailing partial line (the subprocess flushing
    mid-record) is left for the next poll by rewinding to its start.
    """
    out: list[tuple[int, float]] = []
    if not jsonl_path.is_file():
        return out, start_offset
    with jsonl_path.open("r", encoding="utf-8") as fh:
        fh.seek(start_offset)
        while True:
            line_start = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.endswith("\n"):
                # Partial trailing line — rewind so the next poll re-reads
                # it once the subprocess has flushed the newline.
                fh.seek(line_start)
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            vl = _val_loss_from_record(rec)
            if vl is None:
                continue
            step = rec.get("step")
            if not isinstance(step, int):
                continue
            out.append((step, vl))
        new_offset = fh.tell()
    return out, new_offset


def make_pruner(name: str) -> optuna.pruners.BasePruner:
    """Build an Optuna pruner by name — ``median`` / ``hyperband`` / ``none``.

    Mirrors v1 ``create_study``'s pruner switch
    (``git show main:pawn/sweep.py:858-863``). ``median`` is the default for
    the v2 driver (Hyperband needs a known step budget per trial, which the
    adapter loop's early-stop makes variable). ``none`` installs a
    :class:`optuna.pruners.NopPruner` so a study can opt out of mid-trial
    pruning entirely while the objective still reports intermediates.
    """
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(
        f"unknown pruner {name!r}; expected median / hyperband / none"
    )


# ---------------------------------------------------------------------------
# Objective: subprocess-per-trial
# ---------------------------------------------------------------------------


@dataclass
class AdapterObjective:
    """Optuna objective that runs `scripts/train_jax_adapter.py` per trial.

    The subprocess writes `metrics.jsonl` to a per-trial log dir; this
    objective tails the file *while the subprocess runs*, feeding each
    held-out `val_loss` to ``trial.report(val_loss, step)`` and aborting the
    subprocess on ``trial.should_prune()`` (mid-trial pruning, plan §10 S9).
    On a clean exit it returns the best `val_loss` (Optuna minimises by
    default).
    """

    strategy: str
    base_args: list[str]  # supernet/variant/total-steps/etc.
    logs_dir: Path
    script: str = "scripts/train_jax_adapter.py"
    # Use the interpreter running the sweep (which has the `pawn` package +
    # the ROCm JAX plugin resolved) rather than a bare `python` off PATH,
    # which may resolve to a different env (§8.4-low). `field(...)` because
    # `sys.executable` is evaluated at class-definition import time.
    python: str = field(default_factory=lambda: sys.executable)
    # Per-trial wall-clock deadline (seconds). A wedged trial — one that
    # stops emitting val records but never exits — must not hang the whole
    # study, so this defaults to a finite ceiling rather than ``None``.
    # 6 h comfortably covers a real adapter trial on the target hardware;
    # the sweep CLI / lab can lower it. ``None`` opts out entirely.
    timeout: float | None = 6 * 60 * 60.0
    # Backbone depth hint forwarded to suggesters that need it
    # (currently only `suggest_unfreeze`). Defaults to the production
    # supernet depth; sweep CLI should override when targeting tiny.
    n_layers: int = 10
    # How often (seconds) to poll the trial's metrics.jsonl for new val
    # records while the subprocess runs. The val cadence is far coarser
    # than this, so a short poll keeps mid-trial pruning responsive without
    # busy-spinning.
    poll_interval: float = 1.0

    def __call__(self, trial: optuna.Trial) -> float:
        suggester = STRATEGY_SUGGESTERS.get(self.strategy)
        if suggester is None:
            raise ValueError(f"no suggester for strategy {self.strategy!r}")
        params = suggester(trial, n_layers=self.n_layers)
        trial_logs = self.logs_dir / f"trial_{trial.number:05d}"
        trial_logs.mkdir(parents=True, exist_ok=True)
        # H8: route the suggested params through a temp-JSON `--config` so
        # they round-trip through `AdapterConfig` (pydantic) rather than
        # through kebab-cased CLI flags the adapter argparse never
        # registered (which exited 2 → every trial pruned). The resolved
        # adapter `--strategy` (sweep-only aliases mapped) is set on the
        # config body *and* passed on the CLI so it wins over the config.
        config_body = params_to_config_json(self.strategy, params)
        config_path = trial_logs / "sweep_trial_config.json"
        config_path.write_text(json.dumps(config_body), encoding="utf-8")
        cmd = [
            self.python, self.script,
            "--config", str(config_path),
            "--strategy", adapter_strategy_for(self.strategy),
            "--logs-dir", str(trial_logs),
        ] + self.base_args
        return self._run_with_pruning(trial, cmd, trial_logs)

    def _run_with_pruning(
        self, trial: optuna.Trial, cmd: list[str], trial_logs: Path
    ) -> float:
        """Run the trial subprocess, streaming its ``metrics.jsonl`` to
        ``trial.report`` so the study's pruner can abort a doomed trial
        mid-run (plan §10 S9; v1 ``InProcessRoSAObjective`` reported per
        epoch). Aborts the subprocess on ``should_prune()`` and on the
        configured ``timeout``.

        The child's stdout+stderr are redirected to a per-trial
        ``subprocess.log`` file rather than ``PIPE``. The poll loop below
        never reads the child's stdout/stderr, so a ``PIPE`` would dead-lock
        the trial the moment the OS pipe buffer (~64 KB) filled with the
        trainer's XLA/JAX compilation chatter (plus the benign ROCm "sysfs
        nodes path" warning) — the child would block on ``write()`` and stop
        flushing val records, so it would neither complete nor become
        prunable. Writing to a file never blocks the writer; on failure we
        read the file's tail for the error message instead.
        """
        log_path = trial_logs / "subprocess.log"
        offsets: dict[Path, int] = {}
        deadline = (
            time.monotonic() + self.timeout if self.timeout is not None else None
        )
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                try:
                    proc.wait(timeout=self.poll_interval)
                    finished = True
                except subprocess.TimeoutExpired:
                    finished = False
                # Drain whatever val records have landed since the last poll
                # and report them. `metrics.jsonl` lands in a slug subdir of
                # `trial_logs`, so glob each poll (the file appears once the
                # trainer's MetricsLogger opens it).
                pruned = self._report_new_val_records(
                    trial, trial_logs, offsets
                )
                if pruned:
                    self._terminate(proc)
                    raise optuna.TrialPruned(
                        f"trial {trial.number} pruned mid-run"
                    )
                if finished:
                    break
                if deadline is not None and time.monotonic() > deadline:
                    self._terminate(proc)
                    raise optuna.TrialPruned(
                        f"trial {trial.number} exceeded timeout {self.timeout}s"
                    )
            # Final drain after the subprocess exits — records flushed in the
            # interval between the last poll and process exit.
            self._report_new_val_records(trial, trial_logs, offsets)
        if proc.returncode != 0:
            raise optuna.TrialPruned(
                f"trial {trial.number} failed: {self._log_tail(log_path)}"
            )
        best = _read_best_val_loss(trial_logs)
        if best == float("inf"):
            raise optuna.TrialPruned(
                f"trial {trial.number} produced no val_loss"
            )
        return best

    @staticmethod
    def _log_tail(log_path: Path, max_chars: int = 500) -> str:
        """Read the last ``max_chars`` of the trial's captured stdout/stderr
        log for the failure message (the file the child wrote to, never a
        pipe). Returns ``""`` if the log is missing/unreadable."""
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    @staticmethod
    def _report_new_val_records(
        trial: optuna.Trial, trial_logs: Path, offsets: dict[Path, int]
    ) -> bool:
        """Report every new ``val_loss`` row (across all metrics.jsonl files
        under ``trial_logs``) to ``trial.report`` and return ``True`` if the
        pruner says the trial should stop. ``offsets`` carries the per-file
        byte cursor so each record is reported exactly once across polls."""
        reported_any = False
        for jsonl in sorted(trial_logs.rglob("metrics.jsonl")):
            records, new_offset = _scan_val_records(
                jsonl, offsets.get(jsonl, 0)
            )
            offsets[jsonl] = new_offset
            for step, val_loss in records:
                trial.report(val_loss, step)
                reported_any = True
        return reported_any and trial.should_prune()

    @staticmethod
    def _terminate(proc: "subprocess.Popen[str]") -> None:
        """Stop a pruned/timed-out trial subprocess. SIGTERM first (the
        trainer handles it gracefully — finishes the chunk, saves, exits 0),
        then SIGKILL if it ignores the signal."""
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Objective: in-process (skips per-trial JAX startup for big RoSA sweeps)
# ---------------------------------------------------------------------------


# RoSA strategy keys this objective accepts (v1 ``_supported`` tuple). All map
# onto the single ``rosa`` adapter strategy with a distinct ``rosa_mode``.
_ROSA_SWEEP_STRATEGIES: tuple[str, ...] = (
    "rosa",
    "rosa-retro-sparse",
    "rosa-retro-bottleneck",
    "rosa-ratio",
)

# Sweep ``rosa_mode`` value → :class:`pawn.adapters.rosa.RoSAConfig` ``mode``.
_ROSA_MODE_MAP: dict[str, str] = {
    "rosa": "rosa",
    "retro-sparse": "retro-sparse",
    "retro-bottleneck": "retro-bottleneck",
}


class InProcessRoSAObjective:
    """In-process RoSA objective — runs the full 3-phase RoSA trainer in the
    sweep process, skipping the per-trial JAX startup cost a subprocess
    (`AdapterObjective`) pays (v1 ``InProcessRoSAObjective``,
    ``git show main:pawn/sweep.py:421-833``).

    The backbone weights and the train/val corpora are loaded **once** at
    construction and shared across every trial; each trial gets a fresh RoSA
    adapter wrapping the cached backbone. Each trial runs:

    - **Phase 1** — LoRA warm-up for ``rosa_warmup_steps`` steps.
    - **Phase 2** — gradient-magnitude mask generation over ``mask_samples``
      batches.
    - **Phase 3** — joint sparse(+LoRA/+bottleneck) training under the frozen
      mask, with **epoch-level Optuna pruning**: each held-out ``val_loss`` is
      fed to ``trial.report(val_loss, epoch)`` and the trial aborts on
      ``trial.should_prune()`` (v1 parity, plan §10 S9). Early-stops on
      ``patience``.

    Returns the best held-out ``val_loss`` (Optuna minimises by default).

    The JAX-heavy primitives are imported lazily inside the methods so the
    sweep module — and the suggester-contract tests — import without a GPU /
    the JAX plugin resolved.
    """

    def __init__(
        self,
        strategy: str,
        backbone: PAWNModel,
        train_corpus: Corpus,
        val_corpus: Corpus,
        *,
        epochs: int = 50,
        steps_per_epoch: int = 100,
        val_batches: int = 4,
        max_grad_norm: float = 1.0,
        apply_legal: bool = True,
        seed: int = 0,
    ) -> None:
        if strategy not in _ROSA_SWEEP_STRATEGIES:
            raise ValueError(
                f"InProcessRoSAObjective does not support strategy "
                f"{strategy!r}; expected one of {_ROSA_SWEEP_STRATEGIES}"
            )
        self.strategy = strategy
        self.backbone = backbone
        self.train_corpus = train_corpus
        self.val_corpus = val_corpus
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.val_batches = val_batches
        self.max_grad_norm = max_grad_norm
        self.apply_legal = apply_legal
        self.seed = seed

    # -- suggester → RoSAConfig ----------------------------------------------

    def _rosa_config(self, params: dict[str, Any]) -> RoSAConfig:
        """Translate one trial's suggested params into a
        :class:`pawn.adapters.rosa.RoSAConfig` (v1 names preserved)."""
        from pawn.adapters.rosa import RoSAConfig

        mode = _ROSA_MODE_MAP[params["rosa_mode"]]
        kwargs: dict[str, Any] = {
            "mode": mode,
            "lora_rank": int(params.get("lora_rank", 4)),
            "density": float(params.get("density", 0.01)),
        }
        # Optional v1 RoSA axes — only override the RoSAConfig defaults when
        # the suggester actually sweeps them (rosa-ratio fixes the nuisances).
        for key in (
            "rosa_warmup_steps", "mask_samples", "lora_targets",
            "sparse_targets", "bottleneck_dim", "bottleneck_n_hidden",
        ):
            if key in params:
                kwargs[key] = params[key]
        if "grad_alpha" in params:
            kwargs["grad_alpha"] = int(params["grad_alpha"])
        return RoSAConfig(**kwargs)

    # -- Optuna entry point --------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        suggester = STRATEGY_SUGGESTERS[self.strategy]
        params = suggester(trial)
        cfg = self._rosa_config(params)
        lr = float(params.get("lr", 1e-3))
        weight_decay = float(params.get("weight_decay", 0.0))
        warmup_frac = float(params.get("warmup_frac", 0.05))
        batch_size = int(params.get("batch_size", 64))
        patience = int(params.get("patience", 10))

        best = float("inf")
        patience_counter = 0
        for epoch, val_loss in self._train_phases(
            cfg, lr=lr, weight_decay=weight_decay, warmup_frac=warmup_frac,
            batch_size=batch_size,
        ):
            # Mid-trial pruning: report each epoch's held-out val_loss and
            # abort the trial if the study's pruner says so (v1 parity).
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"trial {trial.number} pruned at epoch {epoch}"
                )
            if val_loss < best:
                best = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        if best == float("inf"):
            raise optuna.TrialPruned(
                f"trial {trial.number} produced no val_loss"
            )
        return best

    # -- 3-phase trainer (yields per-epoch val_loss) -------------------------

    def _train_phases(
        self,
        cfg: RoSAConfig,
        *,
        lr: float,
        weight_decay: float,
        warmup_frac: float,
        batch_size: int,
    ) -> "Iterator[tuple[int, float]]":
        """Run the RoSA 3-phase schedule for one trial and yield
        ``(epoch, val_loss)`` after each Phase-3 epoch.

        Mirrors the orchestration in ``scripts/train_jax_adapter.py``
        (Phase 1 LoRA warmup → Phase 2 gradient-mask gen → Phase 3 joint
        training) but drives it from the cached backbone + corpora so the
        sweep pays the JAX startup once, not per trial."""
        import equinox as eqx
        import jax
        import numpy as np

        from pawn.adapter_trainer import (
            AdapterTrainState,
            generate_rosa_masks,
            make_adapter_scan_step,
            make_adapter_train_step,
            make_adapter_val_metrics,
            rosa_phase1_to_phase3,
        )
        from pawn.adapters.rosa import init_rosa_adapter
        from pawn.corpus import legal_mask_for_games
        from pawn.run_config import BaseRunConfig
        from pawn.trainer import make_lr_schedule, make_optimizer, slice_batch

        strategy = "rosa"  # the real adapter strategy all RoSA modes run as
        rng = np.random.default_rng(self.seed)
        total_steps = max(1, self.epochs * self.steps_per_epoch)

        # Forward the caller's grad-norm clip threshold — ``make_optimizer``
        # reads ``cfg.max_grad_norm`` for its global-norm clip, so omitting it
        # here would silently clip every in-process RoSA trial at the
        # ``BaseRunConfig`` default (1.0) regardless of the constructor arg.
        sched_cfg = BaseRunConfig(
            lr=lr, weight_decay=weight_decay, warmup_frac=warmup_frac,
            max_grad_norm=self.max_grad_norm, local_checkpoints=True,
        )
        schedule = make_lr_schedule(sched_cfg, total_steps)
        optimizer = make_optimizer(sched_cfg, schedule)

        need_legal = self.apply_legal

        def legal_for(
            corpus: Corpus, idx: np.ndarray
        ) -> NDArray[np.bool_] | None:
            if not need_legal:
                return None
            return legal_mask_for_games(corpus, idx)

        def make_batch(corpus: Corpus, idx: np.ndarray) -> Batch:
            batch = slice_batch(corpus, idx)
            lm = legal_for(corpus, idx)
            if lm is not None:
                batch = eqx.tree_at(
                    lambda b: b.legal_mask, batch, lm,
                    is_leaf=lambda x: x is None,
                )
            return batch

        # The cached backbone is shared across every trial, but the train
        # steps are built with ``donate="all"`` — the first scan would
        # *delete* the cached backbone's buffers (and trial N+1 would hit
        # "Array has been deleted"). Materialise an independent device copy
        # per trial so donation consumes the copy, never the shared cache.
        backbone = jax.tree.map(
            lambda x: jax.numpy.asarray(x) + 0 if eqx.is_array(x) else x,
            self.backbone,
        )

        # --- Phase 1: LoRA warmup ---
        adapter = init_rosa_adapter(backbone, cfg, key=self.seed)
        flt = self._dispatch_filter(strategy, adapter)
        state = AdapterTrainState(
            backbone=backbone,
            adapter=adapter,
            opt_state=optimizer.init(eqx.filter(adapter, flt)),
            step=jax.numpy.int32(0),
            key=jax.random.key(self.seed),
        )
        train_step = make_adapter_train_step(
            strategy, optimizer, apply_legal=self.apply_legal
        )
        scan_step = make_adapter_scan_step(train_step)
        warmup_n = min(cfg.rosa_warmup_steps, total_steps)
        state = self._run_steps(
            scan_step, state, warmup_n, batch_size, make_batch, rng
        )

        # --- Phase 2: gradient-magnitude mask generation ---
        mask_batches = [
            make_batch(
                self.train_corpus,
                rng.integers(0, self.train_corpus.n_games, size=batch_size),
            )
            for _ in range(cfg.mask_samples)
        ]
        new_sparse = generate_rosa_masks(
            state.backbone, state.adapter, mask_batches,
            apply_legal=self.apply_legal,
        )

        # --- Phase 3: re-init LoRA, install masks, flip toggles ---
        new_adapter = rosa_phase1_to_phase3(
            state.adapter, new_sparse, key=jax.random.key(self.seed + 1)
        )
        flt = self._dispatch_filter(strategy, new_adapter)
        state = AdapterTrainState(
            backbone=state.backbone,
            adapter=new_adapter,
            opt_state=optimizer.init(eqx.filter(new_adapter, flt)),
            step=state.step,
            key=state.key,
        )
        train_step = make_adapter_train_step(
            strategy, optimizer, apply_legal=self.apply_legal
        )
        scan_step = make_adapter_scan_step(train_step)
        val_metrics_fn = make_adapter_val_metrics(
            strategy, apply_legal=self.apply_legal
        )

        for epoch in range(self.epochs):
            state = self._run_steps(
                scan_step, state, self.steps_per_epoch,
                batch_size, make_batch, rng,
            )
            losses: list[float] = []
            for _ in range(self.val_batches):
                idx = rng.integers(
                    0, self.val_corpus.n_games, size=batch_size
                )
                vb = make_batch(self.val_corpus, idx)
                losses.append(
                    float(val_metrics_fn(state.backbone, state.adapter, vb).loss)
                )
            yield epoch, float(np.mean(losses))

    def _run_steps(
        self,
        scan_step: AdapterScanStep,
        state: AdapterTrainState,
        n_steps: int,
        batch_size: int,
        make_batch: Callable[[Corpus, np.ndarray], Batch],
        rng: np.random.Generator,
    ) -> AdapterTrainState:
        """Advance ``state`` by ``n_steps`` chunked train steps (chunk = K,
        capped at 8 to bound the scan's compile-time tensor width)."""
        done = 0
        k = 8
        while done < n_steps:
            chunk = min(k, n_steps - done)
            batches = self._stack_batches(
                [
                    make_batch(
                        self.train_corpus,
                        rng.integers(
                            0, self.train_corpus.n_games, size=batch_size
                        ),
                    )
                    for _ in range(chunk)
                ]
            )
            state, _ = scan_step(state, batches)
            done += chunk
        return state

    @staticmethod
    def _stack_batches(batches: Sequence[Any]) -> Any:
        """Stack a list of single ``Batch`` objects into one Batch with a
        leading K axis (the shape :func:`make_adapter_scan_step` expects)."""
        import jax

        return jax.tree.map(lambda *xs: jax.numpy.stack(xs), *batches)

    @staticmethod
    def _dispatch_filter(strategy: str, adapter: Any) -> Any:
        from pawn.adapter_trainer import dispatch_filter

        return dispatch_filter(strategy)(adapter)
