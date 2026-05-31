#!/usr/bin/env python3
"""Peak-LR sweep for the standalone PAWN-large *teacher* pretrain.

The distillation-canonical ladder (plan §7) starts from a standalone
``large`` teacher (``train_jax.py --variants large``). v1 trained the
production supernet at ``bs=256``; v2 trains the teacher locally at
``bs=64`` with no accumulation, so the optimal peak LR is substantially
different and must be re-found empirically.

This driver sweeps the **peak** LR of the ``infinite`` schedule
(linear warmup → cosine cooldown to ``stable_lr_ratio*peak`` → flat
plateau → final decay). Everything except the peak is held fixed across
trials, so the comparison is apples-to-apples and transfers directly to
the long production teacher run (the schedule shape is length-invariant —
the same phase *fractions* produce the same plateau at any
``total_steps``).

Objective: pretraining trains on freshly-generated random games every
step (no data reuse), so train loss is an unbiased proxy for held-out
loss — no separate validation pass exists or is needed. Each trial is
scored by the **trailing-smoothed minimum train loss** over its proxy
budget, which lands on the plateau floor. Too-high LRs surface as a
higher floor or an outright divergence (NaN / rising loss), both of
which rank worst.

"Thorough" mode runs the wide grid, then a second finer grid placed
geometrically around the winner.

Usage (always on the GPU — never set ``PAWN_ALLOW_CPU``):

    uv run --extra rocm python scripts/sweep_pretrain_lr.py --refine

    # custom grid / budget:
    uv run --extra rocm python scripts/sweep_pretrain_lr.py \
        --grid 6e-5,1e-4,2e-4,3e-4,5e-4,1e-3 --steps 1500 --refine

The heavy training launch is injectable (``launch_fn``) so the
orchestration is unit-testable without spawning JAX.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_JAX = REPO_ROOT / "scripts" / "train_jax.py"

# The "wide" peak-LR grid (plan: bracket both scaling-law estimates AND
# above v1's 3e-4 peak). Geometric, ~1.6-1.8× spacing.
WIDE_GRID: tuple[float, ...] = (6e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3)


# ---------------------------------------------------------------------------
# Fixed-shape sweep settings (everything held constant across trials)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepSettings:
    """The schedule shape + run knobs held fixed while peak LR varies."""

    steps: int = 1500
    batch_size: int = 64
    seq_len: int = 512
    k: int = 50
    supernet: str = "production"
    # `infinite` schedule shape — the steady-state plateau sits at
    # `stable_lr_ratio * peak`. 0.5 is a balanced sustained steady state
    # with room for the final decay to capture gains.
    stable_lr_ratio: float = 0.5
    warmup_frac: float = 0.05
    cooldown_frac: float = 0.2
    decay_frac: float = 0.1
    wsd_decay_shape: str = "linear"
    conditioning: tuple[str, ...] = ()
    log_interval: int = 20
    # Trailing-mean window (in *records*, not steps) used to smooth the
    # noisy per-step loss before taking the minimum.
    smooth_window: int = 10


def base_config(settings: SweepSettings) -> dict[str, object]:
    """The fixed ``PretrainConfig`` payload (sans ``lr`` / ``total_steps``).

    ``lr`` and ``total_steps`` are supplied per-trial on the CLI (which
    overrides the JSON), so they're deliberately omitted here. Everything
    else — the ``infinite`` schedule shape, ``--variants large``, ``bs=64``
    with ``accumulation_steps=1`` — is frozen so trials differ only in peak
    LR.

    ``checkpoint_interval`` is set above ``steps`` so the only checkpoint
    written is the forced final save (train_jax saves once at
    ``next_step >= total_steps`` when ``--local-checkpoints`` is set); it
    lands inside the trial's own ``--logs-dir``.
    """
    return {
        "run_type": "pretrain",
        "supernet": settings.supernet,
        "variants": ["large"],
        "batch_size": settings.batch_size,
        "accumulation_steps": 1,
        "seq_len": settings.seq_len,
        "k": settings.k,
        "lr_schedule": "infinite",
        "stable_lr_ratio": settings.stable_lr_ratio,
        "warmup_frac": settings.warmup_frac,
        "cooldown_frac": settings.cooldown_frac,
        "decay_frac": settings.decay_frac,
        "wsd_decay_shape": settings.wsd_decay_shape,
        "conditioning": list(settings.conditioning),
        "log_interval": settings.log_interval,
        "checkpoint_interval": settings.steps + 1,
        "local_checkpoints": True,
    }


def trial_command(
    cfg_path: Path, lr: float, steps: int, logs_dir: Path
) -> list[str]:
    """The ``train_jax.py`` argv for one trial.

    ``--lr`` / ``--total-steps`` override the base config; ``--logs-dir``
    contains the trial's metrics + its single final checkpoint.
    """
    return [
        sys.executable,
        str(TRAIN_JAX),
        "--config",
        str(cfg_path),
        "--lr",
        f"{lr:.6g}",
        "--total-steps",
        str(steps),
        "--local-checkpoints",
        "--logs-dir",
        str(logs_dir),
    ]


# ---------------------------------------------------------------------------
# Scoring — parse a trial's metrics.jsonl, smooth, take the floor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialResult:
    lr: float
    plateau_lr: float
    smoothed_min: float
    final_loss: float
    n_records: int
    n_nonfinite: int
    diverged: bool


def parse_trial_losses(logs_dir: Path) -> list[tuple[int, float | None]]:
    """``(step, loss)`` for every ``type=train`` record under ``logs_dir``.

    train_jax writes ``logs_dir/<run-slug>/metrics.jsonl``; pick the most
    recently modified one (a fresh trial dir has exactly one).
    """
    candidates = sorted(
        logs_dir.glob("**/metrics.jsonl"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        return []
    rows: list[tuple[int, float | None]] = []
    for line in candidates[-1].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") != "train":
            continue
        loss = rec.get("loss")
        rows.append(
            (int(rec["step"]), float(loss) if loss is not None else None)
        )
    return rows


def _trailing_mean(values: Sequence[float], window: int) -> list[float]:
    """Causal trailing mean — element ``i`` averages ``values[max(0,i-w+1)..i]``."""
    out: list[float] = []
    acc = 0.0
    w = max(1, window)
    for i, v in enumerate(values):
        acc += v
        if i >= w:
            acc -= values[i - w]
        denom = min(i + 1, w)
        out.append(acc / denom)
    return out


def score_losses(
    rows: Sequence[tuple[int, float | None]], lr: float, *,
    stable_lr_ratio: float, smooth_window: int,
) -> TrialResult:
    """Score one trial: trailing-smoothed minimum train loss (lower = better).

    A trial is ``diverged`` if it emitted any non-finite loss, produced no
    usable records, or its loss rose overall (final smoothed ≳ initial) —
    the signatures of an LR past the stability edge.
    """
    finite = [(s, v) for s, v in rows if v is not None and math.isfinite(v)]
    n_nonfinite = sum(
        1 for _, v in rows if v is None or not math.isfinite(v)
    )
    plateau_lr = lr * stable_lr_ratio
    if not finite:
        return TrialResult(
            lr=lr, plateau_lr=plateau_lr, smoothed_min=math.inf,
            final_loss=math.inf, n_records=0, n_nonfinite=n_nonfinite,
            diverged=True,
        )
    vals = [v for _, v in finite]
    smoothed = _trailing_mean(vals, smooth_window)
    smoothed_min = min(smoothed)
    final_loss = vals[-1]
    # Rising-loss divergence: the smoothed tail ended materially above the
    # smoothed head. 5% tolerance absorbs ordinary plateau noise.
    rose = len(smoothed) >= 2 and smoothed[-1] > 1.05 * smoothed[0]
    diverged = n_nonfinite > 0 or rose
    return TrialResult(
        lr=lr, plateau_lr=plateau_lr, smoothed_min=smoothed_min,
        final_loss=final_loss, n_records=len(vals),
        n_nonfinite=n_nonfinite, diverged=diverged,
    )


# ---------------------------------------------------------------------------
# Refinement grid — geometric neighbours around the wide-pass winner
# ---------------------------------------------------------------------------


def refine_grid(
    grid: Sequence[float], winner: float, n_points: int = 2
) -> tuple[list[float], bool]:
    """Geometric refinement points around ``winner``.

    Interior winner: place points at the geometric means between the
    winner and each adjacent grid value. Endpoint winner: extend half a
    log-step beyond the grid (the optimum may lie outside) and return
    ``edge=True`` so the caller can warn.

    Returns ``(sorted_unique_points, hit_edge)``; points already in
    ``grid`` are dropped.
    """
    g = sorted(grid)
    if winner not in g:
        raise ValueError(f"winner {winner!r} not in grid {g!r}")
    i = g.index(winner)
    pts: list[float] = []
    hit_edge = False
    # Left side.
    if i > 0:
        pts.append(math.sqrt(g[i - 1] * winner))
    else:
        hit_edge = True
        step = math.sqrt(g[1] / g[0]) if len(g) >= 2 else math.sqrt(2.0)
        pts.append(winner / step)
    # Right side.
    if i < len(g) - 1:
        pts.append(math.sqrt(winner * g[i + 1]))
    else:
        hit_edge = True
        step = math.sqrt(g[-1] / g[-2]) if len(g) >= 2 else math.sqrt(2.0)
        pts.append(winner * step)
    # Optionally densify: add quarter-log-step points for n_points > 2.
    if n_points > 2 and i > 0 and i < len(g) - 1:
        pts.append(math.sqrt(g[i - 1] * pts[0]))
        pts.append(math.sqrt(pts[1] * g[i + 1]))
    existing = set(g)
    uniq = sorted({round(p, 12) for p in pts} - existing)
    return uniq, hit_edge


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class SweepReport:
    settings: SweepSettings
    wide: list[TrialResult] = field(default_factory=list)
    refined: list[TrialResult] = field(default_factory=list)
    hit_edge: bool = False

    @property
    def all_results(self) -> list[TrialResult]:
        return [*self.wide, *self.refined]

    @property
    def best(self) -> TrialResult | None:
        usable = [r for r in self.all_results if not r.diverged]
        if not usable:
            return None
        return min(usable, key=lambda r: r.smoothed_min)


LaunchFn = Callable[[list[str]], None]


def _default_launch(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _run_trial(
    settings: SweepSettings, cfg_path: Path, lr: float, logs_root: Path,
    launch_fn: LaunchFn,
) -> TrialResult:
    trial_dir = logs_root / f"lr_{lr:.3e}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    launch_fn(trial_command(cfg_path, lr, settings.steps, trial_dir))
    rows = parse_trial_losses(trial_dir)
    return score_losses(
        rows, lr, stable_lr_ratio=settings.stable_lr_ratio,
        smooth_window=settings.smooth_window,
    )


def run_sweep(
    settings: SweepSettings,
    grid: Sequence[float],
    *,
    refine: bool,
    refine_points: int,
    logs_root: Path,
    launch_fn: LaunchFn = _default_launch,
    progress: Callable[[str], None] = print,
) -> SweepReport:
    """Run the wide pass (and optional refinement) and return the report."""
    logs_root.mkdir(parents=True, exist_ok=True)
    cfg_path = logs_root / "base_config.json"
    cfg_path.write_text(
        json.dumps(base_config(settings), indent=2) + "\n", encoding="utf-8"
    )
    report = SweepReport(settings=settings)
    for n, lr in enumerate(sorted(grid), 1):
        progress(f"[wide {n}/{len(grid)}] peak_lr={lr:.3e} "
                 f"(plateau={lr * settings.stable_lr_ratio:.3e})")
        report.wide.append(
            _run_trial(settings, cfg_path, lr, logs_root, launch_fn)
        )
    best = report.best
    if refine and best is not None:
        rgrid, hit_edge = refine_grid(grid, best.lr, refine_points)
        report.hit_edge = hit_edge
        if hit_edge:
            progress("[refine] winner sits at a grid endpoint — extending "
                     "beyond the grid; the optimum may lie further out.")
        for n, lr in enumerate(rgrid, 1):
            progress(f"[refine {n}/{len(rgrid)}] peak_lr={lr:.3e} "
                     f"(plateau={lr * settings.stable_lr_ratio:.3e})")
            report.refined.append(
                _run_trial(settings, cfg_path, lr, logs_root, launch_fn)
            )
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_table(report: SweepReport) -> str:
    """A sorted-by-LR results table + the recommended teacher command."""
    lines = [
        "",
        f"{'peak_lr':>10} {'plateau_lr':>11} {'smooth_min':>11} "
        f"{'final':>9} {'recs':>5} {'status':>9}",
        "-" * 60,
    ]
    pass_of = {id(r): "wide" for r in report.wide}
    pass_of.update({id(r): "refine" for r in report.refined})
    for r in sorted(report.all_results, key=lambda r: r.lr):
        status = "DIVERGED" if r.diverged else pass_of[id(r)]
        smin = "   inf" if not math.isfinite(r.smoothed_min) else f"{r.smoothed_min:.4f}"
        fin = "   inf" if not math.isfinite(r.final_loss) else f"{r.final_loss:.4f}"
        lines.append(
            f"{r.lr:>10.3e} {r.plateau_lr:>11.3e} {smin:>11} "
            f"{fin:>9} {r.n_records:>5} {status:>9}"
        )
    best = report.best
    lines.append("-" * 60)
    if best is None:
        lines.append("NO USABLE TRIAL — every candidate diverged. Widen "
                     "downward / shorten warmup and retry.")
        return "\n".join(lines)
    s = report.settings
    lines += [
        f"\nBEST peak_lr = {best.lr:.3e}  "
        f"(plateau = {best.plateau_lr:.3e}, smooth_min_loss = "
        f"{best.smoothed_min:.4f})",
    ]
    if report.hit_edge and best.lr in {min(r.lr for r in report.all_results),
                                       max(r.lr for r in report.all_results)}:
        lines.append("⚠  winner is at the searched range's edge — consider "
                     "extending the grid in that direction.")
    cond = " ".join(s.conditioning)
    cond_flag = f" --conditioning {cond}" if cond else ""
    lines += [
        "\nRecommended teacher pretrain command (set --total-steps for the "
        "real budget):",
        f"  uv run --extra rocm python scripts/train_jax.py --variants large \\",
        f"    --supernet {s.supernet} --batch-size {s.batch_size} "
        f"--accumulation-steps 1 --seq-len {s.seq_len} --k {s.k} \\",
        f"    --lr {best.lr:.3e} --lr-schedule infinite --warmup-frac "
        f"{s.warmup_frac}{cond_flag} \\",
        f"    --total-steps <BUDGET> --local-checkpoints",
        f"  (infinite-schedule shape via --config: stable_lr_ratio="
        f"{s.stable_lr_ratio}, cooldown_frac={s.cooldown_frac}, "
        f"decay_frac={s.decay_frac})",
    ]
    return "\n".join(lines)


def write_report_json(report: SweepReport, path: Path) -> None:
    best = report.best
    payload = {
        "settings": report.settings.__dict__ | {
            "conditioning": list(report.settings.conditioning)
        },
        "hit_edge": report.hit_edge,
        "best": None if best is None else best.__dict__,
        "trials": [
            {**r.__dict__, "pass": ("refine" if r in report.refined else "wide")}
            for r in sorted(report.all_results, key=lambda r: r.lr)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_report(report: SweepReport, logs_root: Path, path: Path) -> Path | None:
    """Scatter smooth_min-loss vs peak LR (log-x), plus per-trial curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    # Left: per-trial smoothed loss curves.
    for r in sorted(report.all_results, key=lambda r: r.lr):
        trial_dir = logs_root / f"lr_{r.lr:.3e}"
        rows = parse_trial_losses(trial_dir)
        finite = [(s, v) for s, v in rows if v is not None and math.isfinite(v)]
        if not finite:
            continue
        steps = [s for s, _ in finite]
        sm = _trailing_mean([v for _, v in finite], report.settings.smooth_window)
        ax0.plot(steps, sm, label=f"{r.lr:.1e}", alpha=0.8)
    ax0.set_xlabel("step")
    ax0.set_ylabel("train loss (trailing-smoothed)")
    ax0.set_title("Loss trajectory per peak LR")
    ax0.legend(fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)
    # Right: floor vs LR.
    usable = [r for r in report.all_results if math.isfinite(r.smoothed_min)]
    if usable:
        wide_lrs = [r.lr for r in report.wide if math.isfinite(r.smoothed_min)]
        wide_y = [r.smoothed_min for r in report.wide if math.isfinite(r.smoothed_min)]
        ref_lrs = [r.lr for r in report.refined if math.isfinite(r.smoothed_min)]
        ref_y = [r.smoothed_min for r in report.refined if math.isfinite(r.smoothed_min)]
        ax1.scatter(wide_lrs, wide_y, c="tab:blue", label="wide", zorder=3)
        if ref_lrs:
            ax1.scatter(ref_lrs, ref_y, c="tab:orange", label="refine", zorder=3)
        best = report.best
        if best is not None:
            ax1.axvline(best.lr, color="tab:green", ls="--", alpha=0.7,
                        label=f"best {best.lr:.2e}")
        ax1.set_xscale("log")
    ax1.set_xlabel("peak LR")
    ax1.set_ylabel("smooth_min train loss (lower = better)")
    ax1.set_title("Loss floor vs peak LR")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.suptitle(
        f"PAWN-large teacher LR sweep — infinite schedule, "
        f"bs={report.settings.batch_size}, "
        f"stable_lr_ratio={report.settings.stable_lr_ratio}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_grid(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="sweep_pretrain_lr")
    ap.add_argument("--grid", type=_parse_grid, default=WIDE_GRID,
                    help="comma-separated peak LRs (default: the wide grid "
                         "6e-5,1e-4,2e-4,3e-4,5e-4,1e-3)")
    ap.add_argument("--steps", type=int, default=1500,
                    help="proxy budget per trial (default 1500)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--supernet", choices=("tiny", "production"),
                    default="production")
    ap.add_argument("--stable-lr-ratio", type=float, default=0.5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--cooldown-frac", type=float, default=0.2)
    ap.add_argument("--decay-frac", type=float, default=0.1)
    ap.add_argument("--conditioning", nargs="*", default=[],
                    help="control-token kinds prepended after BOS (must match "
                         "the intended teacher run; default BOS-only)")
    ap.add_argument("--refine", action="store_true",
                    help="after the wide pass, run a finer grid around the "
                         "winner")
    ap.add_argument("--refine-points", type=int, default=2)
    ap.add_argument("--logs-root", type=Path, default=Path("logs/lr_sweep"))
    ap.add_argument("--smooth-window", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the base config + per-trial commands without "
                         "launching any training")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    settings = SweepSettings(
        steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        k=args.k, supernet=args.supernet,
        stable_lr_ratio=args.stable_lr_ratio, warmup_frac=args.warmup_frac,
        cooldown_frac=args.cooldown_frac, decay_frac=args.decay_frac,
        conditioning=tuple(args.conditioning),
        smooth_window=args.smooth_window,
    )
    if args.dry_run:
        print(json.dumps(base_config(settings), indent=2))
        args.logs_root.mkdir(parents=True, exist_ok=True)
        cfg_path = args.logs_root / "base_config.json"
        cfg_path.write_text(
            json.dumps(base_config(settings), indent=2) + "\n", encoding="utf-8"
        )
        for lr in sorted(args.grid):
            trial_dir = args.logs_root / f"lr_{lr:.3e}"
            print(" ".join(trial_command(cfg_path, lr, settings.steps, trial_dir)))
        return 0

    report = run_sweep(
        settings, args.grid, refine=args.refine,
        refine_points=args.refine_points, logs_root=args.logs_root,
    )
    print(format_table(report))
    write_report_json(report, args.logs_root / "sweep_report.json")
    png = plot_report(report, args.logs_root, args.logs_root / "lr_sweep.png")
    if png is not None:
        print(f"\nPlot: {png}")
    print(f"Report: {args.logs_root / 'sweep_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
