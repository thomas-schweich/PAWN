"""Standalone Optuna driver for PAWN adapter sweeps.

Subprocess-based: ``AdapterObjective`` builds CLI argv from a trial's
suggested params, runs ``scripts/train_jax_adapter.py`` as a subprocess,
parses the resulting ``metrics.jsonl``, and returns the best ``val_loss``
to Optuna. Pruning hookup via ``trial.report(val_loss, step) +
trial.should_prune()``.

This replaces v1's ``pawn/sweep.py`` (which targeted v1's
``scripts/train.py`` and used several v1-only fields removed in
``docs/jax-migration.md`` §8.4: ``rosa_mode``, ``mask_samples``,
``grad_alpha``, ``lora_ffn``, ``sparse_ffn``, ``bucket_size``,
``unfreeze_layers``). The slimmer v2 surface only handles the eight
strategies the JAX adapter trainer actually supports (LoRA / FiLM /
Unfreeze / Bottleneck / Hybrid / Sparse / RoSA / SpecializedCLM).

The lab MCP server (``pawn.lab``) drives a different code path: it uses
Optuna's ``study.ask()`` to produce candidate suggestions on demand via
the ``lab_results`` API, rather than calling ``study.optimize()`` here.
Both modes spawn ``scripts/train_jax_adapter.py`` as a subprocess; they
don't conflict.

``InProcessRoSAObjective`` (the v1 in-process objective for RoSA
sweeps that skipped per-trial JAX startup) is deferred to a future
release — the subprocess driver is the v2 starting point.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna


TRAIN_SCRIPT = "scripts/train_jax_adapter.py"

# Bottleneck inner-hidden-MLP layer counts the sweep explores.
BOTTLENECK_N_HIDDEN_CHOICES: tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------------------
# Common suggester (shared by every adapter strategy)
# ---------------------------------------------------------------------------


def suggest_common(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64, 128, 256]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_frac": trial.suggest_float("warmup_frac", 0.0, 0.15),
        "patience": trial.suggest_int("patience", 5, 20),
    }


def _suggest_target_subset(
    trial: "optuna.trial.BaseTrial", name: str
) -> list[str]:
    """Pick a subset of ``{q, k, v, o}``. v2 widened ``lora_targets`` /
    ``sparse_targets`` / ``rosa_targets`` from a 3-value Literal to
    ``list[str]`` (§8.4); the canonical 3 v1 settings (``"qkvo"``,
    ``"qv"``, ``"qkv"``) survive as named picks and are listed alongside
    a couple of singletons."""
    pick = trial.suggest_categorical(
        name,
        ["qkvo", "qkv", "qv", "q", "v"],
    )
    return list(pick)


def suggest_lora(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    params = suggest_common(trial)
    params["lora_rank"] = trial.suggest_categorical(
        "lora_rank", [2, 4, 8, 16, 32]
    )
    params["lora_alpha"] = trial.suggest_float("lora_alpha", 1.0, 32.0)
    params["lora_targets"] = _suggest_target_subset(trial, "lora_targets")
    return params


def suggest_bottleneck(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    params = suggest_common(trial)
    params["bottleneck_dim"] = trial.suggest_categorical(
        "bottleneck_dim", [4, 8, 16, 32, 64, 128]
    )
    params["bottleneck_n_hidden"] = trial.suggest_categorical(
        "bottleneck_n_hidden", list(BOTTLENECK_N_HIDDEN_CHOICES)
    )
    # No-op guard (--no-adapt-attn AND --no-adapt-ffn is invalid) is
    # enforced by AdapterConfig._check_strategy_args at validation time;
    # sweep just suggests them independently.
    params["no_adapt_attn"] = trial.suggest_categorical(
        "no_adapt_attn", [True, False]
    )
    # ...and we mask out the both-true case by retrying the categorical
    # for no_adapt_ffn when no_adapt_attn is true.
    if params["no_adapt_attn"]:
        params["no_adapt_ffn"] = False
    else:
        params["no_adapt_ffn"] = trial.suggest_categorical(
            "no_adapt_ffn", [True, False]
        )
    return params


def suggest_film(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    params = suggest_common(trial)
    # ``use_output_film`` (v1 canonical per §8.3, no polarity flip)
    params["use_output_film"] = trial.suggest_categorical(
        "use_output_film", [True, False]
    )
    return params


def suggest_sparse(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    params = suggest_common(trial)
    params["density"] = trial.suggest_float("density", 0.001, 0.1, log=True)
    params["sparse_targets"] = _suggest_target_subset(trial, "sparse_targets")
    params["sparse_hard"] = trial.suggest_categorical(
        "sparse_hard", [True, False]
    )
    return params


def suggest_hybrid(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    params = suggest_common(trial)
    params["lora_rank"] = trial.suggest_categorical("lora_rank", [2, 4, 8, 16])
    params["lora_targets"] = _suggest_target_subset(trial, "lora_targets")
    params["bottleneck_dim"] = trial.suggest_categorical(
        "bottleneck_dim", [4, 8, 16, 32, 64]
    )
    params["use_output_film"] = trial.suggest_categorical(
        "use_output_film", [True, False]
    )
    return params


def suggest_rosa(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    """RoSA in v2 is single-mode (the v1 ``retro-sparse`` /
    ``retro-bottleneck`` modes were not ported per §8.4).
    ``rosa_warmup_steps`` rescaled to ``rosa_warmup_frac``;
    ``mask_samples`` / ``grad_alpha`` removed (Phase-2 mask gen is now
    one-shot)."""
    params = suggest_common(trial)
    params["lora_rank"] = trial.suggest_categorical("lora_rank", [2, 4, 8, 16])
    params["lora_alpha"] = trial.suggest_float("lora_alpha", 1.0, 32.0)
    params["rosa_targets"] = _suggest_target_subset(trial, "rosa_targets")
    params["rosa_warmup_frac"] = trial.suggest_float(
        "rosa_warmup_frac", 0.05, 0.5
    )
    params["rosa_top_k_frac"] = trial.suggest_float(
        "rosa_top_k_frac", 0.001, 0.1, log=True
    )
    return params


def suggest_unfreeze(trial: "optuna.trial.BaseTrial") -> dict[str, Any]:
    """Unfreeze top-N layers (§8.4 — v1's ``unfreeze_layers`` explicit
    picks were rolled back to a top-N count)."""
    params = suggest_common(trial)
    params["n_unfreeze"] = trial.suggest_int("n_unfreeze", 1, 6)
    params["include_lm_head"] = trial.suggest_categorical(
        "include_lm_head", [True, False]
    )
    params["include_embeddings"] = trial.suggest_categorical(
        "include_embeddings", [True, False]
    )
    return params


def suggest_specialized_clm(
    trial: "optuna.trial.BaseTrial",
) -> dict[str, Any]:
    """From-scratch CLM. Architecture lives in the nested
    ``SpecializedCLMConfig`` — d_model / n_layers / n_heads / d_ff
    (canonical names, no ``specialized_`` prefix per §8.3)."""
    params = suggest_common(trial)
    params["d_model"] = trial.suggest_categorical(
        "d_model", [32, 64, 128, 192, 256]
    )
    params["n_layers"] = trial.suggest_int("n_layers", 1, 6)
    params["n_heads"] = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    params["d_ff"] = trial.suggest_int("d_ff", 32, 1024, step=32)
    return params


SUGGEST_FNS: dict[str, Callable[["optuna.trial.BaseTrial"], dict[str, Any]]] = {
    "lora": suggest_lora,
    "bottleneck": suggest_bottleneck,
    "film": suggest_film,
    "sparse": suggest_sparse,
    "hybrid": suggest_hybrid,
    "rosa": suggest_rosa,
    "unfreeze": suggest_unfreeze,
    "specialized_clm": suggest_specialized_clm,
}


# ---------------------------------------------------------------------------
# Subprocess objective — runs scripts/train_jax_adapter.py per trial
# ---------------------------------------------------------------------------


@dataclass
class AdapterObjective:
    """Optuna objective: build CLI argv from trial params, run the
    JAX adapter trainer as a subprocess, parse metrics.jsonl, return
    the best val_loss to Optuna.

    Pruning: between subprocess output flushes, the host can't read
    the partial metrics file mid-run. The subprocess writes one JSON
    record per chunk and flushes per-record (``pawn.logging.MetricsLogger``
    is SIGKILL-durable), so the parent can tail the file periodically
    if pruning is requested. The default ``optimize()`` flow is
    "wait for the subprocess, parse the final file" — pruning hookup
    is a future enhancement.
    """

    strategy: str
    checkpoint: str | None
    supernet: str | None
    variant: str | None
    logs_dir: Path
    extra_args: list[str] = field(default_factory=list)
    timeout_s: int | None = None

    def __call__(self, trial: "optuna.trial.BaseTrial") -> float:
        if self.strategy not in SUGGEST_FNS:
            raise ValueError(
                f"unknown strategy {self.strategy!r}; supported: "
                f"{sorted(SUGGEST_FNS)}"
            )
        params = SUGGEST_FNS[self.strategy](trial)
        return self._run(trial, params)

    def _run(
        self, trial: "optuna.trial.BaseTrial", params: dict[str, Any]
    ) -> float:
        cmd = [sys.executable, TRAIN_SCRIPT, "--strategy", self.strategy]
        if self.checkpoint:
            cmd.extend(["--checkpoint", self.checkpoint])
        if self.supernet:
            cmd.extend(["--supernet", self.supernet])
        if self.variant:
            cmd.extend(["--variant", self.variant])
        cmd.extend(["--logs-dir", str(self.logs_dir)])
        cmd.extend(self.extra_args)
        cmd.extend(_params_to_argv(params))

        trial.set_user_attr("cmd", " ".join(shlex.quote(c) for c in cmd))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout_s
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"train_jax_adapter exited {result.returncode}:\n"
                f"stderr (last 40 lines):\n"
                + "\n".join(result.stderr.splitlines()[-40:])
            )

        run_dir = _find_run_dir(self.logs_dir, result.stdout, result.stderr)
        if run_dir is None:
            raise RuntimeError(
                "could not locate run_dir from subprocess output — "
                "scripts/train_jax_adapter.py must print run_dir=<path>"
            )

        best_val = _read_best_val_loss(run_dir / "metrics.jsonl")
        if best_val is None:
            raise RuntimeError(
                f"no val rows in {run_dir / 'metrics.jsonl'} — the run "
                f"finished without emitting a single ``type: 'val'``"
            )
        trial.set_user_attr("run_dir", str(run_dir))
        return best_val


def _params_to_argv(params: dict[str, Any]) -> list[str]:
    """Translate a ``{field_name: value}`` dict into CLI argv that
    argparse understands. ``bool`` values become a presence-only flag
    (``--foo`` when True, ``--no-foo`` when False); ``list[str]`` values
    are space-joined (matching argparse's ``nargs='+'`` consumption);
    everything else is ``--foo value``."""
    argv: list[str] = []
    for k, v in params.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            argv.append(flag if v else f"--no-{k.replace('_', '-')}")
        elif isinstance(v, list):
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv.extend([flag, str(v)])
    return argv


def _find_run_dir(
    logs_dir: Path, stdout: str, stderr: str
) -> Path | None:
    """Locate the run_dir printed by the subprocess.

    The trainer's MetricsLogger writes ``run_dir=<absolute path>`` to
    its final stdout line; failing that, this falls back to the most
    recently modified ``jax_adapter_run_*`` directory under
    ``logs_dir``."""
    for line in reversed((stdout + "\n" + stderr).splitlines()):
        if "run_dir=" in line:
            tail = line.split("run_dir=", 1)[1].strip()
            if tail:
                # Trim trailing punctuation
                tail = tail.rstrip(";,)")
                candidate = Path(tail)
                if candidate.exists():
                    return candidate
    # Fallback — newest matching dir
    candidates = sorted(
        logs_dir.glob("jax_adapter_run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_best_val_loss(metrics_path: Path) -> float | None:
    """Return the minimum ``val_loss`` across all ``type: "val"`` rows
    in the metrics file, or None if no val rows exist or the file is
    missing."""
    if not metrics_path.exists():
        return None
    best: float | None = None
    with open(metrics_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "val":
                continue
            v = rec.get("val_loss")
            if v is None:
                v = rec.get("loss")
            if v is None or not isinstance(v, (int, float)):
                continue
            if best is None or v < best:
                best = float(v)
    return best


# ---------------------------------------------------------------------------
# Study helpers
# ---------------------------------------------------------------------------


def create_study(
    strategy: str,
    storage: str | None = None,
    study_name: str | None = None,
    pruner: "optuna.pruners.BasePruner | None" = None,
    sampler: "optuna.samplers.BaseSampler | None" = None,
) -> "optuna.Study":
    """Create or resume a persistent Optuna study for ``strategy``.

    ``storage`` should be an SQLite URL like
    ``sqlite:///sweeps/lora.db``; ``study_name`` defaults to
    ``"pawn-{strategy}"``."""
    import optuna  # local import — optuna is in base deps but we keep
    # this module importable for static-only consumers (lab schema, etc.)

    name = study_name or f"pawn-{strategy}"
    return optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name=name,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )


__all__ = [
    "TRAIN_SCRIPT",
    "BOTTLENECK_N_HIDDEN_CHOICES",
    "SUGGEST_FNS",
    "AdapterObjective",
    "create_study",
    "suggest_common",
    "suggest_lora",
    "suggest_bottleneck",
    "suggest_film",
    "suggest_sparse",
    "suggest_hybrid",
    "suggest_rosa",
    "suggest_unfreeze",
    "suggest_specialized_clm",
]
