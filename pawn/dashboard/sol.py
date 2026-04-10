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
from .metrics import detect_run_type, get_run_meta, load_metrics, load_runs, sync_hf_metrics

# ---------------------------------------------------------------------------
# Global reactive state
# ---------------------------------------------------------------------------

_default_log_dir = Path(
    os.environ.get(
        "PAWN_LOG_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "logs"),
    )
)

log_dir = solara.reactive(_default_log_dir)
selected_run = solara.reactive("")
metrics_tick = solara.reactive(0)
runner_output = solara.reactive("")
runner_pid = solara.reactive(0)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_SCRIPTS = {
    "pawn": {"project": PROJECT_ROOT, "script": "scripts/train.py"},
    "film": {"project": PROJECT_ROOT, "script": "scripts/legacy/train_film.py"},
    "lora": {"project": PROJECT_ROOT, "script": "scripts/legacy/train_lora.py"},
    "hybrid": {"project": PROJECT_ROOT, "script": "scripts/legacy/train_hybrid.py"},
    "sparse": {"project": PROJECT_ROOT, "script": "scripts/legacy/train_sparse.py"},
    "bottleneck": {"project": PROJECT_ROOT, "script": "scripts/legacy/train_bottleneck.py"},
    "tiny": {"project": PROJECT_ROOT, "script": "scripts/legacy/train_tiny.py"},
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
  background: rgba(52,211,153,0.08);
  border: 1px solid rgba(52,211,153,0.25);
  border-radius: 999px;
  font-size: 12px;
  color: var(--pawn-success);
  font-weight: 500;
}}
.pawn-pulse.idle {{
  background: rgba(148,163,184,0.08);
  border-color: rgba(148,163,184,0.22);
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
  0%   {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }}
  70%  {{ box-shadow: 0 0 0 8px rgba(52,211,153,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0); }}
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
        avg_ply = _last(val, "val/avg_plies_to_forfeit")
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


@solara.component
def RunSelector(auto_refresh: bool = False, on_auto_refresh=None,
                interval: float = 10.0, on_interval=None):
    show_all, set_show_all = solara.use_state(False)

    def _load():
        hours = 0 if show_all else 1.0
        raw = load_runs(log_dir.value, max_age_hours=hours)
        labeled = []
        label_to_name = {}
        for name in raw:
            meta = get_run_meta(log_dir.value, name)
            parts = []
            if meta.get("slug"):
                parts.append(meta["slug"])
            if meta.get("variant"):
                parts.append(meta["variant"])
            if meta.get("hostname"):
                parts.append(f"@ {meta['hostname']}")
            label = f"{name} ({' / '.join(parts)})" if parts else name
            labeled.append(label)
            label_to_name[label] = name
        return labeled, label_to_name

    labeled, label_to_name = solara.use_memo(
        _load,
        dependencies=[log_dir.value, metrics_tick.value, show_all],
    )
    name_to_label = {v: k for k, v in label_to_name.items()}
    current_label = name_to_label.get(selected_run.value, "")

    if labeled and (not current_label or current_label not in labeled):
        selected_run.set(label_to_name[labeled[0]])

    def on_select(label):
        selected_run.set(label_to_name.get(label, label))

    def _sync_hf():
        synced = sync_hf_metrics(log_dir.value)
        if synced:
            print(f"Synced {len(synced)} runs from HF")
        metrics_tick.set(metrics_tick.value + 1)

    with solara.Div(classes=["pawn-controls"]):
        if labeled:
            solara.Select(
                label="Run", value=current_label, values=labeled, on_value=on_select,
                style={"min-width": "360px", "flex": "1"},
            )
        else:
            msg = "No recent runs (try 'All')" if not show_all else f"No runs in {log_dir.value}"
            solara.Info(msg)
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


# ---------------------------------------------------------------------------
# Config summary (compact chips)
# ---------------------------------------------------------------------------


_CONFIG_SKIP = {"type", "timestamp", "compiled", "hostname", "slug", "git_hash", "git_tag"}
_CONFIG_HIGHLIGHT_ORDER = (
    "variant", "run_type", "d_model", "n_layers", "n_heads",
    "max_seq_len", "batch_size", "lr", "weight_decay", "warmup_steps",
    "total_steps", "patience", "eval_interval", "checkpoint_interval",
)


def _flatten_config(config: dict) -> dict[str, str]:
    flat: dict[str, str] = {}
    for k, v in config.items():
        if k in _CONFIG_SKIP:
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 in _CONFIG_SKIP:
                    continue
                flat[k2] = str(v2)
        else:
            flat[k] = str(v)
    return flat


@solara.component
def ConfigSummary():
    data = solara.use_memo(
        lambda: load_metrics(log_dir.value, selected_run.value) if selected_run.value else {},
        dependencies=[selected_run.value, metrics_tick.value],
    )
    if not selected_run.value:
        return

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


@solara.component
def ChartCard(chart, description: str = ""):
    """A card wrapper that shows the description above the chart."""
    with solara.Div(classes=["pawn-card"]):
        if description:
            solara.HTML(tag="div", unsafe_innerHTML=f'<div class="pawn-desc">{description}</div>')
        solara.FigurePlotly(chart)


@solara.component
def Section(title: str):
    solara.HTML(tag="div", unsafe_innerHTML=f'<div class="pawn-section-title">{title}</div>')


def _row(items: list):
    """Two-column row of chart cards, gracefully handling odd counts."""
    ratio = [1] * max(len(items), 1)
    with solara.Columns(ratio):
        for item in items:
            item()


@solara.component
def MetricsCharts():
    data = solara.use_memo(
        lambda: load_metrics(log_dir.value, selected_run.value) if selected_run.value else {},
        dependencies=[selected_run.value, metrics_tick.value],
    )
    if not selected_run.value:
        return

    configs = data.get("config", [])
    config = configs[-1] if configs else {}
    run_type = detect_run_type(config)
    train = data.get("train", [])
    val = data.get("val", [])
    batch = data.get("batch", [])
    x_key = "epoch" if run_type in ("bc", "film", "lora", "hybrid", "sparse", "bottleneck", "rosa") else "step"

    descs = _CHART_DESCRIPTIONS.get(run_type, {})

    def desc(key: str) -> str:
        return descs.get(key, _DEFAULT_DESCRIPTIONS.get(key, ""))

    def card(fig, description: str):
        def _render():
            ChartCard(fig, description)
        return _render

    if run_type == "hybrid":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("LoRA Adaptation")
        _row([
            card(charts.lora_layer_chart(train, x_key), desc("lora_layer")),
            card(charts.lora_proj_chart(train, x_key), desc("lora_proj")),
        ])
        Section("FiLM Modulation")
        _row([
            card(charts.film_weight_chart(train, x_key), desc("film_gamma")),
            card(charts.film_beta_chart(train, x_key), desc("film_beta")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key), desc("lr")),
            card(charts.time_chart(train, x_key, run_type), desc("time")),
        ])
    elif run_type == "rosa":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("Adapter Diagnostics")
        _row([
            card(charts.sparse_delta_chart(train, x_key), desc("sparse_delta")),
            card(charts.bottleneck_up_chart(train, x_key), desc("adapter_up")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key), desc("lr")),
            card(charts.time_chart(train, x_key, run_type), desc("time")),
        ])
        _row([
            card(charts.gpu_chart(train, x_key), desc("gpu")),
            card(charts.patience_chart(val, x_key), desc("patience")),
        ])
    elif run_type == "sparse":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("Adapter Diagnostics")
        _row([
            card(charts.sparse_delta_chart(train, x_key), desc("sparse_delta")),
            card(charts.time_chart(train, x_key, run_type), desc("time")),
        ])
    elif run_type == "bottleneck":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("Adapter Diagnostics")
        _row([
            card(charts.bottleneck_up_chart(train, x_key), desc("adapter_up")),
            card(charts.lr_chart(train, batch, x_key), desc("lr")),
        ])
        Section("Runtime")
        _row([
            card(charts.gpu_chart(train, x_key), desc("gpu")),
            card(charts.time_chart(train, x_key, run_type), desc("time")),
        ])
    elif run_type == "tiny":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key), desc("lr")),
            card(charts.gpu_chart(train, x_key), desc("gpu")),
        ])
        _row([
            card(charts.time_chart(train, x_key, run_type), desc("time")),
        ])
    elif run_type == "lora":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("LoRA Adaptation")
        _row([
            card(charts.lora_layer_chart(train, x_key), desc("lora_layer")),
            card(charts.lora_proj_chart(train, x_key), desc("lora_proj")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key), desc("lr")),
            card(charts.time_chart(train, x_key, run_type), desc("time")),
        ])
    elif run_type == "film":
        Section("Training")
        _row([
            card(charts.loss_chart(train, x_key, run_type), desc("loss")),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("FiLM Modulation")
        _row([
            card(charts.film_weight_chart(train, x_key), desc("film_gamma")),
            card(charts.film_beta_chart(train, x_key), desc("film_beta")),
        ])
        Section("Optimizer & Runtime")
        _row([
            card(charts.lr_chart(train, batch, x_key), desc("lr")),
            card(charts.time_chart(train, x_key, run_type), desc("time")),
        ])
    else:
        # Pretraining default
        Section("Training")
        _row([
            card(
                charts.loss_chart(train, x_key, run_type, val_records=val),
                desc("loss") + " Val loss overlaid on train; log scale.",
            ),
            card(charts.accuracy_chart(train, x_key, run_type), desc("accuracy")),
        ])
        Section("Validation")
        _row([
            card(
                charts.val_loss_chart(val, x_key, run_type),
                "Validation perplexity on held-out games (log scale).",
            ),
            card(charts.val_accuracy_chart(val, x_key, run_type), desc("val_accuracy")),
        ])
        patience = config.get("training", {}).get("patience", 10)
        if isinstance(patience, str):
            try:
                patience = int(patience)
            except ValueError:
                patience = 10
        Section("Game Integrity & Stopping")
        error_desc = (
            "Log-scale error rates — illegal (per-position), late illegal "
            "(second half of games), and forfeit (per-game). Dashed = log-linear "
            "fit on the last half; slope → half-life. Lower is better."
        )
        patience_desc = _DEFAULT_DESCRIPTIONS["patience"]
        _row([
            card(charts.error_rate_chart(val, x_key, fit=True), error_desc),
            card(charts.patience_chart(val, x_key, patience_limit=patience), patience_desc)
            if val else (lambda: None),
        ])
        Section("Optimizer")
        _row([
            card(charts.lr_chart(train, batch, x_key), desc("lr")),
            card(charts.grad_chart(train, x_key), desc("grad_norm")),
        ])
        Section("Runtime")
        _row([
            card(charts.gpu_chart(train, x_key), desc("gpu")),
            card(charts.time_chart(train, x_key, run_type), desc("time")),
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
    ConfigSummary()
    MetricsCharts()


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
        cmd = ["uv", "run", "python", info["script"]]
        if run_type == "pawn":
            if checkpoint:
                cmd.extend(["--resume", checkpoint])
            if batch_size:
                cmd.extend(["--batch-size", batch_size])
            if steps:
                cmd.extend(["--total-steps", steps])
        else:
            if checkpoint:
                cmd.extend(["--checkpoint", checkpoint])
            if pgn_path:
                cmd.extend(["--pgn", pgn_path])
            cmd.extend(["--lr", lr, "--batch-size", batch_size, "--epochs", steps])
        cmd.extend(["--log-dir", str(root / "logs")])
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
