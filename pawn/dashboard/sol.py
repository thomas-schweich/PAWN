"""PAWN Dashboard — Solara components.

Standalone:  solara run pawn.dashboard.sol
Jupyter:     from pawn.dashboard import Dashboard; Dashboard()
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time as _time
from pathlib import Path

import solara

from . import charts, theme
from .metrics import (
    detect_run_type,
    get_run_meta,
    list_trials,
    load_metrics,
    load_notes,
    load_runs,
    save_notes,
    sync_hf_metrics,
)

# ---------------------------------------------------------------------------
# Global reactive state
# ---------------------------------------------------------------------------

_default_log_dir = Path(
    os.environ.get(
        "PAWN_LOG_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "logs"),
    )
)

ALL_TRIALS = "(all)"

log_dir = solara.reactive(_default_log_dir)
selected_trial = solara.reactive(ALL_TRIALS)
selected_run = solara.reactive("")
compare_run = solara.reactive("")
metrics_tick = solara.reactive(0)
runner_output = solara.reactive("")
runner_pid = solara.reactive(0)
# Per-chart axis-log overrides. Keyed by a stable chart id (the title text);
# values are ``(x_log, y_log)`` tuples. Missing keys mean "use the chart's
# built-in default" (log-log for loss/perplexity, linear for accuracy, etc.).
chart_axis_overrides: solara.Reactive[dict[str, tuple[bool, bool]]] = solara.reactive({})

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All runs dispatch through scripts/train.py. `pawn` runs pretrain;
# adapter entries map to `--run-type adapter --strategy <strategy>`.
TRAIN_SCRIPTS: dict[str, dict[str, str | Path]] = {
    "pawn":       {"project": PROJECT_ROOT, "script": "scripts/train.py", "run_type": "pretrain"},
    "film":       {"project": PROJECT_ROOT, "script": "scripts/train.py", "run_type": "adapter", "strategy": "film"},
    "lora":       {"project": PROJECT_ROOT, "script": "scripts/train.py", "run_type": "adapter", "strategy": "lora"},
    "hybrid":     {"project": PROJECT_ROOT, "script": "scripts/train.py", "run_type": "adapter", "strategy": "hybrid"},
    "sparse":     {"project": PROJECT_ROOT, "script": "scripts/train.py", "run_type": "adapter", "strategy": "sparse"},
    "bottleneck": {"project": PROJECT_ROOT, "script": "scripts/train.py", "run_type": "adapter", "strategy": "bottleneck"},
    "tiny":       {"project": PROJECT_ROOT, "script": "scripts/train.py", "run_type": "adapter", "strategy": "specialized_clm"},
}


# ---------------------------------------------------------------------------
# Global CSS — injected once per page render
# ---------------------------------------------------------------------------

GLOBAL_CSS = f"""
:root {{
  --pawn-bg: {theme.BG};
  --pawn-surface: {theme.SURFACE};
  --pawn-surface-2: {theme.SURFACE_ELEVATED};
  --pawn-border: {theme.BORDER};
  --pawn-border-strong: {theme.BORDER_STRONG};
  --pawn-text: {theme.TEXT};
  --pawn-text-muted: {theme.TEXT_MUTED};
  --pawn-text-faint: {theme.TEXT_FAINT};
  --pawn-primary: {theme.SKY};
  --pawn-accent: {theme.ROSE};
  --pawn-success: {theme.EMERALD};
  --pawn-warning: {theme.AMBER};
}}

html, body,
.v-application,
.theme--dark.v-application,
.v-application .v-application--wrap,
.theme--dark.v-application .v-application--wrap,
.theme--dark.v-sheet,
.theme--dark.v-card,
.theme--dark.v-main,
.theme--dark .v-main__wrap {{
  background: {theme.BG} !important;
  background-color: {theme.BG} !important;
  color: var(--pawn-text) !important;
  font-family: {theme.FONT_FAMILY};
  font-feature-settings: 'cv11', 'ss01', 'ss03';
}}

.pawn-shell {{
  position: relative;
  z-index: 1;
  max-width: 1520px;
  margin: 0 auto;
  padding: 24px 32px 64px;
}}

.pawn-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 24px;
  margin-bottom: 24px;
  background: {theme.HEADER_BG};
  border: 1px solid var(--pawn-border);
  border-radius: 16px;
}}
.pawn-header .pawn-title {{
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: var(--pawn-text);
  line-height: 1.2;
}}
.pawn-header .pawn-title .pawn-brand {{
  background: linear-gradient(135deg, {theme.SKY}, {theme.CYAN});
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}
.pawn-header .pawn-subtitle {{
  font-size: 12px;
  color: var(--pawn-text-muted);
  margin-top: 2px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}}

.pawn-pulse {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: color-mix(in srgb, var(--pawn-success) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--pawn-success) 25%, transparent);
  border-radius: 999px;
  font-size: 12px;
  color: var(--pawn-success);
  font-weight: 500;
}}
.pawn-pulse.idle {{
  background: color-mix(in srgb, var(--pawn-text-muted) 8%, transparent);
  border-color: color-mix(in srgb, var(--pawn-text-muted) 22%, transparent);
  color: var(--pawn-text-muted);
}}
.pawn-pulse .dot {{
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 0 0 currentColor;
  animation: pawn-pulse 1.8s ease-out infinite;
}}
.pawn-pulse.idle .dot {{ animation: none; opacity: 0.5; }}
@keyframes pawn-pulse {{
  0%   {{ box-shadow: 0 0 0 0 color-mix(in srgb, currentColor 55%, transparent); }}
  70%  {{ box-shadow: 0 0 0 8px color-mix(in srgb, currentColor 0%, transparent); }}
  100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, currentColor 0%, transparent); }}
}}

.pawn-card {{
  background: var(--pawn-surface);
  border: 1px solid var(--pawn-border);
  border-radius: 14px;
  padding: 16px 18px 14px;
  transition: border-color 120ms ease, transform 120ms ease;
}}
.pawn-card:hover {{
  border-color: var(--pawn-border-strong);
}}

.pawn-section-title {{
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--pawn-text-muted);
  margin: 28px 4px 10px;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.pawn-section-title::after {{
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--pawn-border), transparent);
}}

.pawn-kpi {{
  background: var(--pawn-surface);
  border: 1px solid var(--pawn-border);
  border-radius: 14px;
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  position: relative;
  overflow: hidden;
}}
.pawn-kpi::before {{
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: var(--pawn-kpi-accent, var(--pawn-primary));
  opacity: 0.9;
}}
.pawn-kpi .label {{
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--pawn-text-muted);
}}
.pawn-kpi .value {{
  font-family: {theme.FONT_MONO};
  font-size: 22px;
  font-weight: 600;
  color: var(--pawn-text);
  font-feature-settings: 'tnum';
  line-height: 1.15;
}}
.pawn-kpi .sub {{
  font-size: 11px;
  color: var(--pawn-text-faint);
}}

.pawn-chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  margin: 3px 4px 3px 0;
  border-radius: 999px;
  background: rgba(148,163,184,0.07);
  border: 1px solid var(--pawn-border);
  font-family: {theme.FONT_MONO};
  font-size: 11px;
  color: var(--pawn-text-muted);
}}
.pawn-chip b {{ color: var(--pawn-text); font-weight: 500; }}

.pawn-desc {{
  color: var(--pawn-text-muted);
  font-size: 11.5px;
  line-height: 1.5;
  margin: 2px 2px 6px;
}}

.pawn-controls {{
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  padding: 12px 14px;
  background: var(--pawn-surface);
  border: 1px solid var(--pawn-border);
  border-radius: 14px;
  margin-bottom: 16px;
}}

.v-btn {{ text-transform: none !important; letter-spacing: 0 !important; }}
.v-input__slot {{ background: var(--pawn-surface-2) !important; }}
.v-label {{ color: var(--pawn-text-muted) !important; }}
.v-select__selection, .v-select__selection--comma, input {{ color: var(--pawn-text) !important; }}

.pawn-tabs {{
  display: inline-flex;
  background: var(--pawn-surface);
  border: 1px solid var(--pawn-border);
  border-radius: 12px;
  padding: 4px;
  gap: 2px;
  margin-bottom: 20px;
}}
.pawn-tab-btn.v-btn {{
  min-width: 0 !important;
  padding: 0 18px !important;
  height: 34px !important;
  border-radius: 9px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  color: var(--pawn-text-muted) !important;
  text-transform: none !important;
  transition: all 120ms ease;
}}
.pawn-tab-btn.v-btn:hover {{
  color: var(--pawn-text) !important;
  background: rgba(148,163,184,0.06) !important;
}}
.pawn-tab-btn.v-btn.active {{
  background: linear-gradient(135deg, {theme.SKY}26, {theme.ROSE}18) !important;
  color: var(--pawn-text) !important;
  box-shadow: 0 0 0 1px {theme.SKY}44 inset !important;
}}

.pawn-runner-output {{
  background: #07090f;
  border: 1px solid var(--pawn-border);
  border-radius: 10px;
  padding: 12px 14px;
  font-family: {theme.FONT_MONO};
  font-size: 11.5px;
  color: #cdd4e5;
  max-height: 360px;
  overflow-y: auto;
  white-space: pre-wrap;
  line-height: 1.55;
}}
.pawn-axis-toggles {{
  display: inline-flex;
  gap: 4px;
  margin-left: auto;
  align-items: center;
}}
.pawn-chart-head {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 4px;
}}
.pawn-chart-head .pawn-desc {{ margin: 0; }}
.pawn-axis-btn.v-btn {{
  min-width: 0 !important;
  height: 22px !important;
  padding: 0 9px !important;
  border-radius: 6px !important;
  font-family: {theme.FONT_MONO};
  font-size: 10.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0.03em !important;
  color: var(--pawn-text-muted) !important;
  background: rgba(148,163,184,0.06) !important;
  border: 1px solid var(--pawn-border) !important;
  text-transform: none !important;
  box-shadow: none !important;
  transition: all 120ms ease;
}}
.pawn-axis-btn.v-btn:hover {{
  color: var(--pawn-text) !important;
  border-color: var(--pawn-border-strong) !important;
}}
.pawn-axis-btn.v-btn.active {{
  color: {theme.SKY} !important;
  background: color-mix(in srgb, {theme.SKY} 14%, transparent) !important;
  border-color: color-mix(in srgb, {theme.SKY} 45%, transparent) !important;
}}
.pawn-cmd-preview {{
  background: #07090f;
  border: 1px solid var(--pawn-border);
  border-radius: 10px;
  padding: 10px 12px;
  font-family: {theme.FONT_MONO};
  font-size: 11.5px;
  color: {theme.SKY};
  word-break: break-all;
  line-height: 1.55;
}}
"""


@solara.component
def InjectStyle():
    solara.Style(GLOBAL_CSS)


# ---------------------------------------------------------------------------
# Plotly autoresize shim
# ---------------------------------------------------------------------------
#
# ``solara.FigurePlotly`` ships figures with ``autosize=True`` but plotly only
# honours that at the initial mount. When a chart is later placed in a wider
# container (e.g. because we filter an empty sibling card and the row collapses
# from two columns to one), plotly keeps its SVG pinned at the original 700px
# default — visually half-width even though the card spans the row. This
# script observes every ``.js-plotly-plot`` element with a ``ResizeObserver``
# and calls ``Plotly.Plots.resize(el)`` whenever the container's width changes,
# plus a ``MutationObserver`` to pick up charts added after the first render.

_PLOTLY_RESIZE_SHIM = """
(() => {
  if (window.__pawnPlotlyResizeShim) return;
  window.__pawnPlotlyResizeShim = true;

  const attach = (el) => {
    if (!el || el.__pawnResizeAttached) return;
    el.__pawnResizeAttached = true;
    const ro = new ResizeObserver(() => {
      if (window.Plotly && window.Plotly.Plots && document.contains(el)) {
        try { window.Plotly.Plots.resize(el); } catch (_) { /* noop */ }
      }
    });
    ro.observe(el);
  };

  const scan = (root) => {
    const matches = (root.querySelectorAll
      ? root.querySelectorAll('.js-plotly-plot')
      : []);
    matches.forEach(attach);
    if (root.classList && root.classList.contains('js-plotly-plot')) attach(root);
  };

  scan(document);
  const mo = new MutationObserver((muts) => {
    for (const m of muts) {
      for (const n of m.addedNodes) {
        if (n.nodeType === 1) scan(n);
      }
    }
  });
  mo.observe(document.body, { childList: true, subtree: true });
})();
"""


@solara.component
def InjectPlotlyResizeShim():
    solara.HTML(tag="script", unsafe_innerHTML=_PLOTLY_RESIZE_SHIM)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


@solara.component
def Header(run_type: str, hostname: str, auto_refresh: bool):
    status_class = "" if auto_refresh else "idle"
    status_text = "LIVE" if auto_refresh else "PAUSED"
    host_line = f"{run_type or 'unknown'} run"
    if hostname:
        host_line += f" · {hostname}"
    solara.HTML(
        tag="div",
        unsafe_innerHTML=(
            '<div class="pawn-header">'
            '  <div>'
            '    <div class="pawn-title"><span class="pawn-brand">PAWN</span> · Training Dashboard</div>'
            f'    <div class="pawn-subtitle">{host_line}</div>'
            '  </div>'
            f'  <div class="pawn-pulse {status_class}"><span class="dot"></span>{status_text}</div>'
            '</div>'
        ),
    )


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------


def _fmt_num(x, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.{max(digits - 2, 1)}f}"
    return f"{v:.{digits}f}"


def _fmt_pct(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _last(records: list[dict], key: str):
    for r in reversed(records):
        v = r.get(key)
        if v is not None:
            return v
    return None


def _best_min(records: list[dict], key: str):
    best = None
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if best is None or fv < best:
            best = fv
    return best


def _best_max(records: list[dict], key: str):
    best = None
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if best is None or fv > best:
            best = fv
    return best


def _kpi_html(label: str, value: str, sub: str = "", accent: str = theme.SKY) -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="pawn-kpi" style="--pawn-kpi-accent:{accent}">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value">{value}</div>'
        f'  {sub_html}'
        f'</div>'
    )


@solara.component
def KpiRow(run_type: str, train: list[dict], val: list[dict]):
    tiles: list[tuple[str, str, str, str]] = []  # (label, value, sub, accent)

    if run_type == "pawn":
        step = _last(train, "step")
        train_loss = _last(train, "train/loss")
        val_loss = _last(val, "val/loss")
        best_val = _best_min(val, "val/loss")
        val_acc = _last(val, "val/accuracy")
        best_acc = _best_max(val, "val/accuracy")
        completion = _last(val, "val/game_completion_rate")
        avg_ply = _last(val, "val/avg_plies_completed")
        gpu = _last(train, "mem/gpu_peak_gb")
        step_time = _last(train, "step_time")

        tiles.append(("Step", _fmt_num(step, 0), "", theme.SKY))
        tiles.append(("Train Loss", _fmt_num(train_loss), "", theme.SKY))
        tiles.append((
            "Val Loss",
            _fmt_num(val_loss),
            f"best · {_fmt_num(best_val)}" if best_val is not None else "",
            theme.ROSE,
        ))
        tiles.append((
            "Val Top-1",
            _fmt_pct(val_acc),
            f"best · {_fmt_pct(best_acc)}" if best_acc is not None else "",
            theme.EMERALD,
        ))
        if completion is not None:
            tiles.append((
                "Game Completion",
                _fmt_pct(completion),
                f"avg ply · {_fmt_num(avg_ply, 0)}" if avg_ply is not None else "",
                theme.AMBER,
            ))
        if gpu is not None:
            tiles.append((
                "GPU Peak",
                f"{float(gpu):.1f} GB",
                f"{float(step_time):.2f}s/step" if step_time is not None else "",
                theme.CYAN,
            ))
    else:
        epoch = _last(train, "epoch")
        train_loss = _last(train, "train_loss")
        val_loss = _last(val, "val_loss") or _last(train, "val_loss")
        best_val = _best_min(train, "val_loss")
        val_top1 = _last(val, "val_top1") or _last(train, "val_top1")
        best_top1 = _best_max(train, "val_top1")
        epoch_time = _last(train, "epoch_time_s") or _last(train, "epoch_time")
        gpu = _last(train, "mem/gpu_peak_gb")

        tiles.append(("Epoch", _fmt_num(epoch, 0), "", theme.SKY))
        tiles.append(("Train Loss", _fmt_num(train_loss), "", theme.SKY))
        tiles.append((
            "Val Loss",
            _fmt_num(val_loss),
            f"best · {_fmt_num(best_val)}" if best_val is not None else "",
            theme.ROSE,
        ))
        tiles.append((
            "Val Top-1",
            _fmt_pct(val_top1),
            f"best · {_fmt_pct(best_top1)}" if best_top1 is not None else "",
            theme.EMERALD,
        ))
        if gpu is not None:
            tiles.append((
                "GPU Peak",
                f"{float(gpu):.1f} GB",
                f"{float(epoch_time):.2f}s/ep" if epoch_time is not None else "",
                theme.CYAN,
            ))

    if not tiles:
        return

    inner = "".join(_kpi_html(*t) for t in tiles)
    n = len(tiles)
    solara.HTML(
        tag="div",
        unsafe_innerHTML=(
            '<div style="display:grid;'
            f'grid-template-columns:repeat({n}, minmax(0, 1fr));'
            'gap:12px;margin-bottom:18px;">'
            f'{inner}</div>'
        ),
    )


# ---------------------------------------------------------------------------
# Run selector (compact control bar)
# ---------------------------------------------------------------------------


def _label_for_run(log_dir_path: Path, run_name: str) -> str:
    """Decorate a run's relative path with its slug/variant/hostname."""
    meta = get_run_meta(log_dir_path, run_name)
    parts = []
    if meta.get("slug"):
        parts.append(meta["slug"])
    if meta.get("variant"):
        parts.append(meta["variant"])
    if meta.get("hostname"):
        parts.append(f"@ {meta['hostname']}")
    return f"{run_name} ({' / '.join(parts)})" if parts else run_name


def _runs_for_trial(all_runs: list[str], trial: str) -> list[str]:
    if trial == ALL_TRIALS:
        return all_runs
    prefix = f"{trial}/"
    return [r for r in all_runs if r.startswith(prefix) or r == trial]


@solara.component
def RunSelector(auto_refresh: bool = False, on_auto_refresh=None,
                interval: float = 10.0, on_interval=None):
    # Default "All" on when the 1-hour window is empty — otherwise the user
    # sees a populated trial dropdown but an empty run list, and selecting a
    # trial appears to do nothing.
    initial_show_all = not bool(load_runs(log_dir.value, max_age_hours=1.0))
    show_all, set_show_all = solara.use_state(initial_show_all)

    def _load():
        hours = 0 if show_all else 1.0
        raw = load_runs(log_dir.value, max_age_hours=hours)
        trials = list_trials(log_dir.value)
        return raw, trials

    all_runs, trials = solara.use_memo(
        _load,
        dependencies=[log_dir.value, metrics_tick.value, show_all],
    )

    trial_values = [ALL_TRIALS, *trials] if trials else [ALL_TRIALS]
    if selected_trial.value not in trial_values:
        selected_trial.set(ALL_TRIALS)

    filtered_runs = _runs_for_trial(all_runs, selected_trial.value)

    label_to_name: dict[str, str] = {}
    labeled: list[str] = []
    for name in filtered_runs:
        label = _label_for_run(log_dir.value, name)
        labeled.append(label)
        label_to_name[label] = name
    name_to_label = {v: k for k, v in label_to_name.items()}
    current_label = name_to_label.get(selected_run.value, "")

    if labeled and (not current_label or current_label not in labeled):
        selected_run.set(label_to_name[labeled[0]])
    elif not labeled and selected_run.value:
        selected_run.set("")

    # Comparison selector: anything in the full recent-run list, plus "(none)".
    NONE = "(none)"
    compare_label_to_name: dict[str, str] = {NONE: ""}
    compare_labels = [NONE]
    for name in all_runs:
        if name == selected_run.value:
            continue
        label = _label_for_run(log_dir.value, name)
        compare_labels.append(label)
        compare_label_to_name[label] = name
    compare_name_to_label = {v: k for k, v in compare_label_to_name.items()}
    current_compare_label = compare_name_to_label.get(compare_run.value, NONE)
    if current_compare_label not in compare_labels:
        compare_run.set("")
        current_compare_label = NONE

    def on_select(label):
        selected_run.set(label_to_name.get(label, label))

    def on_select_compare(label):
        compare_run.set(compare_label_to_name.get(label, ""))

    def on_select_trial(t):
        selected_trial.set(t)

    def _sync_hf():
        synced = sync_hf_metrics(log_dir.value)
        if synced:
            print(f"Synced {len(synced)} runs from HF")
        metrics_tick.set(metrics_tick.value + 1)

    with solara.Div(classes=["pawn-controls"]):
        if trials:
            solara.Select(
                label="Trial", value=selected_trial.value, values=trial_values,
                on_value=on_select_trial,
                style={"min-width": "180px"},
            )
        if labeled:
            solara.Select(
                label="Run", value=current_label, values=labeled, on_value=on_select,
                style={"min-width": "360px", "flex": "1"},
            )
        else:
            msg = "No recent runs (try 'All')" if not show_all else f"No runs in {log_dir.value}"
            solara.Info(msg)
        solara.Select(
            label="Compare", value=current_compare_label, values=compare_labels,
            on_value=on_select_compare,
            style={"min-width": "280px"},
        )
        solara.Button(
            "Refresh",
            on_click=lambda: metrics_tick.set(metrics_tick.value + 1),
            icon_name="mdi-refresh",
            text=True,
        )
        solara.Button("Sync HF", on_click=_sync_hf, icon_name="mdi-cloud-download", text=True)
        solara.Switch(label="All", value=show_all, on_value=set_show_all)
        if on_auto_refresh is not None:
            solara.Switch(label="Live", value=auto_refresh, on_value=on_auto_refresh)
            if auto_refresh and on_interval is not None:
                solara.Select(
                    label="",
                    value=f"{interval:.0f}s",
                    on_value=lambda v: on_interval(float(v.rstrip("s"))),
                    values=["5s", "10s", "30s", "60s"],
                    style={"max-width": "90px"},
                )


@solara.component
def NotesEditor():
    """Read/write a markdown notes file scoped to the current run's trial."""
    # Hooks must be called unconditionally (rules of hooks). We render nothing
    # when no run is selected, but the state slots still need to exist.
    run = selected_run.value
    saved, set_saved = solara.use_state("")
    draft, set_draft = solara.use_state("")
    status, set_status = solara.use_state("")

    def _reload():
        if not run:
            return None
        text = load_notes(log_dir.value, run)
        set_saved(text)
        set_draft(text)
        set_status("")
        return None

    solara.use_memo(_reload, dependencies=[log_dir.value, run, metrics_tick.value])

    if not run:
        return

    def _save():
        path = save_notes(log_dir.value, run, draft)
        set_saved(draft)
        set_status(f"Saved · {path}")

    def _discard():
        set_draft(saved)
        set_status("Discarded")

    dirty = draft != saved
    from .metrics import notes_path
    target = notes_path(log_dir.value, run)

    with solara.Div(classes=["pawn-card"], style={"margin-bottom": "18px"}):
        solara.HTML(
            tag="div",
            unsafe_innerHTML=(
                '<div class="pawn-desc" style="font-size:13px;color:var(--pawn-text);'
                'font-weight:600;letter-spacing:0.02em;">Trial Notes</div>'
                f'<div class="pawn-desc" style="margin-bottom:8px;">{target}</div>'
            ),
        )
        solara.InputTextArea(
            label="", value=draft, on_value=set_draft, rows=6,
        )
        with solara.Row(gap="8px"):
            solara.Button(
                "Save", on_click=_save, disabled=not dirty, color="primary",
                icon_name="mdi-content-save",
            )
            solara.Button(
                "Discard", on_click=_discard, disabled=not dirty, text=True,
                icon_name="mdi-restore",
            )
            if status:
                solara.HTML(
                    tag="div",
                    unsafe_innerHTML=(
                        f'<div class="pawn-desc" style="margin-left:8px;align-self:center;">{status}</div>'
                    ),
                )


# ---------------------------------------------------------------------------
# Config summary (compact chips)
# ---------------------------------------------------------------------------


_CONFIG_SKIP = {"type", "timestamp", "compiled", "hostname", "slug", "git_hash", "git_tag"}
_CONFIG_HIGHLIGHT_ORDER = (
    "variant", "run_type", "formulation",
    "model.d_model", "model.n_layers", "model.n_heads", "model.max_seq_len",
    "training.batch_size", "training.lr", "training.weight_decay",
    "training.warmup_steps", "training.total_steps", "training.patience",
    "training.eval_interval", "training.checkpoint_interval",
    "param_count",
)


def _flatten_config(config: dict) -> dict[str, str]:
    """Flatten a nested config dict to ``dot.separated`` keys.

    Prefixing with the parent key prevents silent collisions when two nested
    sections share a field name (e.g. ``model.dropout`` vs ``training.dropout``).
    """
    flat: dict[str, str] = {}
    for k, v in config.items():
        if k in _CONFIG_SKIP:
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 in _CONFIG_SKIP:
                    continue
                flat[f"{k}.{k2}"] = str(v2)
        else:
            flat[k] = str(v)
    return flat


@solara.component
def ConfigSummary(data: dict | None = None):
    if not selected_run.value:
        return
    if data is None:
        data = {}

    configs = data.get("config", [])
    config = configs[-1] if configs else {}
    flat = _flatten_config(config)
    if not flat:
        return

    # Order: highlight keys first, then everything else alphabetically.
    ordered: list[tuple[str, str]] = []
    seen = set()
    for k in _CONFIG_HIGHLIGHT_ORDER:
        if k in flat:
            ordered.append((k, flat[k]))
            seen.add(k)
    for k in sorted(flat):
        if k not in seen:
            ordered.append((k, flat[k]))

    chips = "".join(
        f'<span class="pawn-chip">{k}·<b>{v}</b></span>'
        for k, v in ordered
    )
    solara.HTML(
        tag="div",
        unsafe_innerHTML=(
            '<div class="pawn-card" style="margin-bottom:18px;">'
            '<div class="pawn-desc" style="margin-bottom:8px;">Configuration</div>'
            f'<div>{chips}</div>'
            '</div>'
        ),
    )


# ---------------------------------------------------------------------------
# Chart grid
# ---------------------------------------------------------------------------

_CHART_DESCRIPTIONS = {
    "film": {
        "loss": "Cross-entropy loss on Lichess move prediction with illegal-move masking.",
        "accuracy": "Top-1 move accuracy (train) and top-1/top-5 (val).",
        "film_gamma": "Per-layer γ deviation from identity. Large values = this layer is adapting.",
        "film_beta": "Per-layer β norm. Hidden β shifts features; output β acts as a fixed move prior.",
        "lr": "Learning rate schedule — linear warmup then cosine decay.",
        "time": "Wall-clock seconds per epoch.",
    },
    "pawn": {
        "loss": "Cross-entropy on next-token prediction. Lower = better at predicting random moves.",
        "accuracy": "Fraction of positions where the model's top-1 prediction matches the true next move.",
        "val_loss": "Validation loss and perplexity on held-out games.",
        "val_accuracy": "Validation top-1/top-5 on held-out games.",
        "lr": "Learning rate schedule — linear warmup then cosine decay.",
        "grad_norm": "L2 norm of gradients before clipping.",
        "gpu": "GPU memory footprint (peak / reserved / current).",
        "time": "Wall-clock seconds per training step.",
    },
    "lora": {
        "loss": "Cross-entropy loss on Lichess move prediction with illegal-move masking.",
        "accuracy": "Top-1 / top-5 move accuracy.",
        "lora_layer": "Mean ‖B‖₂ per layer — which depths are adapting most.",
        "lora_proj": "Mean ‖B‖₂ per projection — Q vs K vs V vs O balance.",
        "time": "Wall-clock seconds per epoch.",
    },
    "hybrid": {
        "loss": "Cross-entropy loss on Lichess move prediction with illegal-move masking.",
        "accuracy": "Top-1 / top-5 move accuracy.",
        "lora_layer": "Mean LoRA ‖B‖₂ per layer.",
        "lora_proj": "Mean LoRA ‖B‖₂ per projection type.",
        "film_gamma": "Per-layer FiLM γ deviation from identity.",
        "film_beta": "Per-layer FiLM β norm.",
        "time": "Wall-clock seconds per epoch.",
    },
}

_DEFAULT_DESCRIPTIONS = {
    "lr": "Learning rate schedule.",
    "grad_norm": "Gradient norm before clipping.",
    "gpu": "GPU memory footprint.",
    "time": "Wall-clock per step or epoch.",
    "sparse_delta": "Per-layer sparse Δ norm.",
    "adapter_up": "Bottleneck adapter up-projection norm.",
    "patience": "Consecutive evals without improvement — stops when it hits the limit.",
}


def _chart_title_text(fig) -> str:
    """Extract the plain-text title from a plotly figure.

    Chart builders wrap titles in ``<b>…</b>``; strip the markup so the
    title can double as a stable reactive-state key.
    """
    layout = getattr(fig, "layout", None)
    if layout is None:
        return ""
    title = getattr(layout, "title", None)
    text = getattr(title, "text", "") if title is not None else ""
    if not text:
        return ""
    import re as _re
    return _re.sub(r"<[^>]+>", "", text).strip()


@solara.component
def ChartCard(chart, description: str = ""):
    """A card wrapper that shows the description above the chart and renders
    per-axis log-scale toggle pills. Toggle state is keyed by the chart's
    title so it survives re-renders (including live auto-refresh ticks)."""
    chart_id = _chart_title_text(chart) or description or id(chart)

    default_x, default_y = charts.current_axis_log(chart)
    override = chart_axis_overrides.value.get(chart_id)
    x_log, y_log = override if override is not None else (default_x, default_y)
    if override is not None:
        # FigurePlotly caches the figure by object identity; mutating the
        # same instance after override doesn't trigger a re-render. Hand it
        # a fresh Figure built from the current one with the axis types
        # replaced.
        import plotly.graph_objects as _go
        chart = _go.Figure(chart)
        charts.apply_axis_log(chart, x_log, y_log)

    def _set_axes(new_x: bool, new_y: bool):
        chart_axis_overrides.set(
            {**chart_axis_overrides.value, chart_id: (new_x, new_y)}
        )

    with solara.Div(classes=["pawn-card"]):
        with solara.Div(classes=["pawn-chart-head"]):
            if description:
                solara.HTML(
                    tag="div",
                    unsafe_innerHTML=f'<div class="pawn-desc">{description}</div>',
                )
            with solara.Div(classes=["pawn-axis-toggles"]):
                solara.Button(
                    "log X",
                    on_click=lambda: _set_axes(not x_log, y_log),
                    classes=["pawn-axis-btn", *(["active"] if x_log else [])],
                    text=True,
                )
                solara.Button(
                    "log Y",
                    on_click=lambda: _set_axes(x_log, not y_log),
                    classes=["pawn-axis-btn", *(["active"] if y_log else [])],
                    text=True,
                )
        solara.FigurePlotly(chart)


@solara.component
def Section(title: str):
    solara.HTML(tag="div", unsafe_innerHTML=f'<div class="pawn-section-title">{title}</div>')


def _row(items: list[tuple]):
    """Render a row of chart cards, dropping any whose figure has no data.

    ``items`` is a list of ``(fig, description)`` tuples. Empty figures
    (e.g. end-of-epoch metrics not yet written for a mid-epoch run) are
    filtered out so the user doesn't see a blank plot card. When only one
    card survives, we render it directly (no ``Columns`` wrapper) so it
    unambiguously spans the full row width.
    """
    live = [(fig, desc) for fig, desc in items if not charts.is_empty_chart(fig)]
    if not live:
        return
    if len(live) == 1:
        fig, desc = live[0]
        ChartCard(fig, desc)
        return
    ratio = [1] * len(live)
    with solara.Columns(ratio):
        for fig, desc in live:
            ChartCard(fig, desc)


@solara.component
def MetricsCharts(data: dict | None = None, compare: dict | None = None):
    if not selected_run.value:
        return
    if data is None:
        data = {}
    if compare is None:
        compare = {}

    configs = data.get("config", [])
    config = configs[-1] if configs else {}
    run_type = detect_run_type(config)
    train = data.get("train", [])
    val = data.get("val", [])
    batch = data.get("batch", [])
    # Prefer `step` whenever records carry unique step values. Adapter runs
    # historically keyed charts on `epoch`, but the modern single-epoch
    # adapter trainer emits one record per step with `epoch` pinned at 0
    # — collapsing every chart onto one x-bucket.
    def _has_varying(records: list[dict], key: str) -> bool:
        seen = set()
        for r in records:
            v = r.get(key)
            if v is not None:
                seen.add(v)
                if len(seen) > 1:
                    return True
        return False

    if _has_varying(train, "step"):
        x_key = "step"
    elif run_type in ("bc", "film", "lora", "hybrid", "sparse", "bottleneck", "rosa"):
        x_key = "epoch"
    else:
        x_key = "step"

    c_train = compare.get("train", []) or None
    c_val = compare.get("val", []) or None
    c_batch = compare.get("batch", []) or None

    descs = _CHART_DESCRIPTIONS.get(run_type, {})

    def desc(key: str) -> str:
        return descs.get(key, _DEFAULT_DESCRIPTIONS.get(key, ""))

    def card(fig, description: str) -> tuple:
        """Adapter that keeps ``_row`` callsites unchanged across the refactor
        from thunks to ``(fig, desc)`` tuples."""
        return (fig, description)

    if run_type == "hybrid":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type, compare_records=c_train), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("LoRA Adaptation")
        _row([
            card(charts.lora_layer_chart(train, x_key, compare_records=c_train), desc("lora_layer")),
            card(charts.lora_proj_chart(train, x_key, compare_records=c_train), desc("lora_proj")),
        ])
        Section("FiLM Modulation")
        _row([
            card(charts.film_weight_chart(train, x_key, compare_records=c_train), desc("film_gamma")),
            card(charts.film_beta_chart(train, x_key, compare_records=c_train), desc("film_beta")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key,
                                 compare_train_records=c_train,
                                 compare_batch_records=c_batch), desc("lr")),
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])
    elif run_type == "rosa":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type, compare_records=c_train), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("Adapter Diagnostics")
        _row([
            card(charts.sparse_delta_chart(train, x_key, compare_records=c_train), desc("sparse_delta")),
            card(charts.bottleneck_up_chart(train, x_key, compare_records=c_train), desc("adapter_up")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key,
                                 compare_train_records=c_train,
                                 compare_batch_records=c_batch), desc("lr")),
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])
        _row([
            card(charts.gpu_chart(train, x_key, compare_records=c_train), desc("gpu")),
            card(charts.patience_chart(val, x_key, compare_records=c_val), desc("patience")),
        ])
    elif run_type == "sparse":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type, compare_records=c_train), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("Adapter Diagnostics")
        _row([
            card(charts.sparse_delta_chart(train, x_key, compare_records=c_train), desc("sparse_delta")),
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])
    elif run_type == "bottleneck":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type, compare_records=c_train), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("Adapter Diagnostics")
        _row([
            card(charts.bottleneck_up_chart(train, x_key, compare_records=c_train), desc("adapter_up")),
            card(charts.lr_chart(train, batch, x_key,
                                 compare_train_records=c_train,
                                 compare_batch_records=c_batch), desc("lr")),
        ])
        Section("Runtime")
        _row([
            card(charts.gpu_chart(train, x_key, compare_records=c_train), desc("gpu")),
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])
    elif run_type == "tiny":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type, compare_records=c_train), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key,
                                 compare_train_records=c_train,
                                 compare_batch_records=c_batch), desc("lr")),
            card(charts.gpu_chart(train, x_key, compare_records=c_train), desc("gpu")),
        ])
        _row([
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])
    elif run_type == "lora":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type, compare_records=c_train), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("LoRA Adaptation")
        _row([
            card(charts.lora_layer_chart(train, x_key, compare_records=c_train), desc("lora_layer")),
            card(charts.lora_proj_chart(train, x_key, compare_records=c_train), desc("lora_proj")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key,
                                 compare_train_records=c_train,
                                 compare_batch_records=c_batch), desc("lr")),
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])
    elif run_type == "film":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type, compare_records=c_train), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("FiLM Modulation")
        _row([
            card(charts.film_weight_chart(train, x_key, compare_records=c_train), desc("film_gamma")),
            card(charts.film_beta_chart(train, x_key, compare_records=c_train), desc("film_beta")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key,
                                 compare_train_records=c_train,
                                 compare_batch_records=c_batch), desc("lr")),
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])
    else:
        # Pretraining default
        Section("Training")
        _row([
            card(
                charts.loss_chart(train, x_key, run_type, val_records=val,
                                  compare_records=c_train, compare_val_records=c_val),
                desc("loss") + " Val loss overlaid on train; log-log scale, "
                "with a dashed power-law fit (y = A·x^b) on the last half of each series.",
            ),
            card(charts.accuracy_chart(train, x_key, run_type, compare_records=c_train), desc("accuracy")),
        ])
        Section("Validation")
        _row([
            card(
                charts.perplexity_chart(val, x_key, compare_records=c_val),
                "Validation perplexity on held-out games (log-log scale).",
            ),
            card(charts.val_accuracy_chart(val, x_key, run_type, compare_records=c_val), desc("val_accuracy")),
        ])
        patience = config.get("training", {}).get("patience", 10)
        if isinstance(patience, str):
            try:
                patience = int(patience)
            except ValueError:
                patience = 10
        if val:
            Section("Game Integrity & Stopping")
            error_desc = (
                "Log-log error rates — illegal (per-position), late illegal "
                "(per-position, plies ≥ context/2), and forfeit (per-game, any illegal). "
                "Dashed = power-law fit y = A·x^b on the last half of the series; "
                "exponent b is the log-log slope, and halve@×N is the multiplicative "
                "step ratio needed to halve the rate. Lower is better."
            )
            patience_desc = _DEFAULT_DESCRIPTIONS["patience"]
            _row([
                card(charts.error_rate_chart(val, x_key, fit=True, compare_records=c_val), error_desc),
                card(charts.patience_chart(val, x_key, patience_limit=patience,
                                           compare_records=c_val), patience_desc),
            ])
        Section("Optimizer")
        _row([
            card(charts.lr_chart(train, batch, x_key,
                                 compare_train_records=c_train,
                                 compare_batch_records=c_batch), desc("lr")),
            card(charts.grad_chart(train, x_key, compare_records=c_train), desc("grad_norm")),
        ])
        Section("Runtime")
        _row([
            card(charts.gpu_chart(train, x_key, compare_records=c_train), desc("gpu")),
            card(charts.time_chart(train, x_key, run_type, compare_records=c_train), desc("time")),
        ])


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------


@solara.component
def AutoRefresh(interval: float = 10.0):
    def setup():
        stop = threading.Event()

        def _tick_loop():
            while not stop.wait(interval):
                metrics_tick.set(metrics_tick.value + 1)

        t = threading.Thread(target=_tick_loop, daemon=True)
        t.start()
        return lambda: stop.set()

    solara.use_effect(setup, [interval])


# ---------------------------------------------------------------------------
# Top-level Dashboard component
# ---------------------------------------------------------------------------


@solara.component
def Dashboard(log_dir_override: Path | None = None):
    """Full metrics dashboard.

    Args:
        log_dir_override: Explicit log directory (handy in notebooks).
            If None, uses PAWN_LOG_DIR env var or ``<repo>/logs``.
    """
    if log_dir_override is not None and log_dir.value != log_dir_override:
        log_dir.set(log_dir_override)

    auto_refresh, set_auto_refresh = solara.use_state(True)
    interval, set_interval = solara.use_state(10.0)

    if auto_refresh:
        AutoRefresh(interval=interval)

    data = solara.use_memo(
        lambda: load_metrics(log_dir.value, selected_run.value) if selected_run.value else {},
        dependencies=[selected_run.value, metrics_tick.value],
    )
    compare_data = solara.use_memo(
        lambda: load_metrics(log_dir.value, compare_run.value) if compare_run.value else {},
        dependencies=[compare_run.value, metrics_tick.value],
    )
    configs = data.get("config", []) if data else []
    config = configs[-1] if configs else {}
    run_type = detect_run_type(config) if config else ""
    hostname = config.get("hostname", "") if config else ""
    train = data.get("train", []) if data else []
    val = data.get("val", []) if data else []

    Header(run_type=run_type, hostname=hostname, auto_refresh=auto_refresh)
    RunSelector(
        auto_refresh=auto_refresh,
        on_auto_refresh=set_auto_refresh,
        interval=interval,
        on_interval=set_interval,
    )
    if train or val:
        KpiRow(run_type, train, val)
    ConfigSummary(data=data)
    NotesEditor()
    MetricsCharts(data=data, compare=compare_data)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@solara.component
def Runner(project_root: Path | None = None):
    """Training run launcher with live output."""
    root = project_root or PROJECT_ROOT

    run_type, set_run_type = solara.use_state("bottleneck")
    checkpoint, set_checkpoint = solara.use_state("")
    pgn_path, set_pgn_path = solara.use_state("")
    lr, set_lr = solara.use_state("1e-4")
    batch_size, set_batch_size = solara.use_state("64")
    steps, set_steps = solara.use_state("50")
    extra_args, set_extra_args = solara.use_state("")

    def build_command() -> list[str]:
        info = TRAIN_SCRIPTS[run_type]
        cmd = ["uv", "run", "python", str(info["script"])]
        cmd.extend(["--run-type", str(info["run_type"])])
        if info["run_type"] == "pretrain":
            if checkpoint:
                cmd.extend(["--resume", checkpoint])
            if batch_size:
                cmd.extend(["--batch-size", batch_size])
            if steps:
                cmd.extend(["--total-steps", steps])
        else:
            cmd.extend(["--strategy", str(info["strategy"])])
            if checkpoint:
                cmd.extend(["--checkpoint", checkpoint])
            if pgn_path:
                cmd.extend(["--pgn", pgn_path])
            cmd.extend(["--lr", lr, "--batch-size", batch_size, "--epochs", steps])
        cmd.extend(["--log-dir", str(root / "logs"), "--local-checkpoints"])
        if extra_args.strip():
            cmd.extend(extra_args.strip().split())
        return cmd

    def launch():
        if runner_pid.value:
            return
        info = TRAIN_SCRIPTS[run_type]
        cmd = build_command()
        cwd = str(info["project"])
        runner_output.set(f"$ cd {cwd}\n$ {' '.join(cmd)}\n\n")

        def _stream():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=cwd,
                )
                runner_pid.set(proc.pid)
                buf: list[str] = []
                last_flush = _time.monotonic()
                for line in proc.stdout:
                    buf.append(line)
                    if _time.monotonic() - last_flush > 0.5:
                        runner_output.set(runner_output.value + "".join(buf))
                        buf.clear()
                        last_flush = _time.monotonic()
                if buf:
                    runner_output.set(runner_output.value + "".join(buf))
                proc.wait()
                runner_output.set(runner_output.value + f"\n[exited {proc.returncode}]\n")
            except Exception as exc:
                runner_output.set(runner_output.value + f"\n[error: {exc}]\n")
            finally:
                runner_pid.set(0)

        threading.Thread(target=_stream, daemon=True).start()

    def stop():
        pid = runner_pid.value
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            runner_pid.set(0)

    cmd_preview = " ".join(build_command())
    is_running = bool(runner_pid.value)

    with solara.Div(classes=["pawn-card"]):
        solara.HTML(
            tag="div",
            unsafe_innerHTML=(
                '<div class="pawn-desc" style="font-size:13px;color:var(--pawn-text);'
                'font-weight:600;margin-bottom:12px;letter-spacing:0.02em;">Launch Training Run</div>'
            ),
        )
        with solara.Columns([1, 1]):
            with solara.Column(gap="10px"):
                solara.Select(
                    label="Run Type", value=run_type, on_value=set_run_type,
                    values=list(TRAIN_SCRIPTS.keys()),
                )
                solara.InputText("Checkpoint", value=checkpoint, on_value=set_checkpoint)
                if run_type != "pawn":
                    solara.InputText("PGN File", value=pgn_path, on_value=set_pgn_path)
                solara.InputText("Learning Rate", value=lr, on_value=set_lr)
                solara.InputText("Batch Size", value=batch_size, on_value=set_batch_size)
                steps_label = "Total Steps" if run_type == "pawn" else "Epochs"
                solara.InputText(steps_label, value=steps, on_value=set_steps)
                solara.InputText("Extra Args", value=extra_args, on_value=set_extra_args)

            with solara.Column(gap="10px"):
                solara.HTML(
                    tag="div",
                    unsafe_innerHTML=f'<div class="pawn-cmd-preview">{cmd_preview}</div>',
                )
                with solara.Row(gap="8px"):
                    solara.Button(
                        "Launch", on_click=launch, disabled=is_running, color="primary",
                        icon_name="mdi-play",
                    )
                    solara.Button(
                        "Stop", on_click=stop, disabled=not is_running, color="error",
                        icon_name="mdi-stop",
                    )
                    status_class = "" if is_running else "idle"
                    status_text = f"PID {runner_pid.value}" if is_running else "IDLE"
                    solara.HTML(
                        tag="div",
                        unsafe_innerHTML=(
                            f'<div class="pawn-pulse {status_class}" style="margin-left:8px;">'
                            f'<span class="dot"></span>{status_text}</div>'
                        ),
                    )
                if runner_output.value:
                    tail = "\n".join(runner_output.value.split("\n")[-120:])
                    solara.HTML(
                        tag="div",
                        unsafe_innerHTML=f'<div class="pawn-runner-output">{_escape(tail)}</div>',
                    )


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Standalone page
# ---------------------------------------------------------------------------


@solara.component
def Page():
    """Top-level page for ``solara run pawn.dashboard.sol``."""
    solara.Title("PAWN Dashboard")
    solara.use_effect(lambda: setattr(solara.lab.theme, "dark", True), [])

    InjectStyle()
    InjectPlotlyResizeShim()

    tab, set_tab = solara.use_state(0)

    with solara.Div(classes=["pawn-shell"]):
        with solara.Div(classes=["pawn-tabs"]):
            solara.Button(
                "Training",
                on_click=lambda: set_tab(0),
                classes=["pawn-tab-btn", *(["active"] if tab == 0 else [])],
                text=True,
            )
            solara.Button(
                "Runner",
                on_click=lambda: set_tab(1),
                classes=["pawn-tab-btn", *(["active"] if tab == 1 else [])],
                text=True,
            )

        if tab == 0:
            Dashboard()
        else:
            Runner()
