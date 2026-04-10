"""Plotly chart builders for the PAWN dashboard.

All visual defaults are pulled from ``theme`` so the chart grid shares one
coherent color/typography system with the Solara shell.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from . import theme
from .theme import COLORS, LAYER_COLORS, OUTPUT_LAYER_COLOR, TRACE_WIDTH

# Legacy alias so any outside code that imported LAYOUT keeps working.
LAYOUT = theme.PLOTLY_LAYOUT


def _melt(records: list[dict], x_key: str, y_specs: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, dict]:
    """Build a long-form DataFrame + color map from records and y_specs."""
    if not records:
        return pd.DataFrame(), {}
    df = pd.DataFrame(records)
    cols_present = [(key, name, color) for key, name, color in y_specs if key in df.columns]
    if not cols_present or x_key not in df.columns:
        return pd.DataFrame(), {}
    melted = df[[x_key] + [k for k, _, _ in cols_present]].dropna(subset=[x_key])
    melted = melted.melt(id_vars=x_key, var_name="metric", value_name="value").dropna(subset=["value"])
    rename = {k: name for k, name, _ in cols_present}
    melted["metric"] = melted["metric"].map(rename)
    color_map = {name: color for _, name, color in cols_present}
    return melted, color_map


def _empty_figure(title: str, y_title: str = ""):
    """A blank themed figure used when no data matches.

    Intentionally omits an in-chart placeholder: the card description above
    the chart already tells the user what is expected, and an annotation can
    visually bleed through if the figure is later populated with traces.
    """
    fig = px.line(title=title)
    fig.update_layout(**theme.PLOTLY_LAYOUT)
    fig.update_layout(title=f"<b>{title}</b>", yaxis_title=y_title, annotations=[])
    return fig


def _style_figure(fig, title: str, y_title: str):
    """Apply theme defaults + bold title + trace polish to a figure."""
    fig.update_layout(**theme.PLOTLY_LAYOUT)
    fig.update_layout(title=f"<b>{title}</b>", yaxis_title=y_title)
    fig.update_traces(
        line=dict(width=TRACE_WIDTH),
        mode="lines",
        hovertemplate="%{y:.4f}<extra>%{meta}</extra>",
    )
    for trace in fig.data:
        trace.meta = trace.name
    return fig


def make_chart(
    records: list[dict],
    x_key: str,
    y_specs: list[tuple[str, str, str]],
    title: str = "",
    y_title: str = "",
    y_log: bool = False,
) -> px.line:
    """Create a themed Plotly line chart. y_specs: list of (key, label, color)."""
    melted, color_map = _melt(records, x_key, y_specs)
    if melted.empty:
        return _empty_figure(title, y_title)
    fig = px.line(
        melted, x=x_key, y="value", color="metric",
        color_discrete_map=color_map,
        labels={"value": y_title, x_key: x_key.replace("_", " ").title(), "metric": ""},
    )
    if y_log:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(rangemode="tozero")
    return _style_figure(fig, title, y_title)


_ADAPTER_TYPES = {"film", "lora", "hybrid", "sparse", "bottleneck", "tiny", "rosa"}


# ---------------------------------------------------------------------------
# Core training charts
# ---------------------------------------------------------------------------


def loss_chart(records: list[dict], x_key: str, run_type: str,
               val_records: list[dict] | None = None):
    """Loss chart. For pawn runs, val_records overlays val/loss on top of train/loss."""
    if run_type in _ADAPTER_TYPES | {"bc"}:
        specs = [("train_loss", "Train", COLORS["primary"]),
                 ("val_loss", "Val", COLORS["accent"])]
        return make_chart(records, x_key, specs, title="Loss", y_title="cross-entropy", y_log=True)
    if run_type == "pawn":
        combined = list(records or []) + list(val_records or [])
        specs = [
            ("train/loss", "Train", COLORS["primary"]),
            ("val/loss", "Val", COLORS["accent"]),
        ]
        return make_chart(combined, x_key, specs, title="Loss", y_title="cross-entropy", y_log=True)
    specs = [("train_loss", "Train", COLORS["primary"]),
             ("val_loss", "Val", COLORS["accent"])]
    return make_chart(records, x_key, specs, title="Loss", y_title="cross-entropy", y_log=True)


def accuracy_chart(records: list[dict], x_key: str, run_type: str):
    if run_type in _ADAPTER_TYPES:
        specs = [
            ("train_top1", "Train Top-1", COLORS["primary"]),
            ("val_top1", "Val Top-1", COLORS["accent"]),
            ("val_top5", "Val Top-5", COLORS["success"]),
        ]
    elif run_type == "bc":
        specs = [("train_acc", "Train", COLORS["primary"]), ("val_acc", "Val", COLORS["accent"])]
    elif run_type == "pawn":
        specs = [("train/accuracy", "Top-1", COLORS["primary"])]
    else:
        specs = [("train_top1", "Train Top-1", COLORS["primary"]), ("val_top1", "Val Top-1", COLORS["accent"])]
    return make_chart(records, x_key, specs, title="Accuracy", y_title="rate")


def lr_chart(train_records: list[dict], batch_records: list[dict], x_key: str):
    src = batch_records if batch_records else train_records
    x = "global_step" if batch_records else x_key
    specs = [("lr", "LR", COLORS["warning"])]
    if src and "lr_lora" in src[0]:
        specs = [
            ("lr_lora", "LoRA LR", COLORS["accent"]),
            ("lr_film", "FiLM LR", COLORS["primary"]),
        ]
    return make_chart(src, x, specs, title="Learning Rate", y_title="lr")


def grad_chart(records: list[dict], x_key: str):
    return make_chart(records, x_key, [
        ("grad_norm", "Grad Norm", COLORS["violet"]),
        ("grad_norm_mean", "Mean", COLORS["primary"]),
        ("grad_norm_max", "Max", COLORS["error"]),
    ], title="Gradient Norm", y_title="‖g‖₂", y_log=True)


def gpu_chart(records: list[dict], x_key: str):
    return make_chart(records, x_key, [
        ("mem/gpu_peak_gb", "Peak", COLORS["info"]),
        ("mem/gpu_reserved_gb", "Reserved", COLORS["warning"]),
        ("mem/gpu_current_gb", "Current", COLORS["success"]),
    ], title="GPU Memory", y_title="GB")


def time_chart(records: list[dict], x_key: str, run_type: str):
    key = "epoch_time_s" if run_type in _ADAPTER_TYPES else ("epoch_time" if run_type == "bc" else "step_time")
    label = "Epoch Time" if run_type in (_ADAPTER_TYPES | {"bc"}) else "Step Time"
    return make_chart(records, x_key, [(key, label, COLORS["slate"])], title=label, y_title="seconds")


# ---------------------------------------------------------------------------
# Adapter diagnostics
# ---------------------------------------------------------------------------


def film_weight_chart(records: list[dict], x_key: str):
    specs = [(f"film/hidden_{i}/gamma_dev", f"L{i} γ", LAYER_COLORS[i]) for i in range(8)]
    specs.append(("film/output/gamma_dev", "Out γ", OUTPUT_LAYER_COLOR))
    return make_chart(records, x_key, specs, title="FiLM γ Deviation", y_title="‖γ − 1‖₂")


def film_beta_chart(records: list[dict], x_key: str):
    specs = [(f"film/hidden_{i}/beta_norm", f"L{i} β", LAYER_COLORS[i]) for i in range(8)]
    specs.append(("film/output/beta_norm", "Out β", OUTPUT_LAYER_COLOR))
    return make_chart(records, x_key, specs, title="FiLM β Norm", y_title="‖β‖₂")


_PROJ_DASH = {"wq": "solid", "wv": "dot", "wk": "dash", "wo": "dashdot"}
_PROJ_COLORS = {
    "wq": COLORS["error"],
    "wk": COLORS["orange"],
    "wv": COLORS["primary"],
    "wo": COLORS["violet"],
}


def _detect_lora_projs(records: list[dict]) -> list[str]:
    projs: set[str] = set()
    for r in records:
        for k in r:
            if k.startswith("lora/layer"):
                parts = k.split(".")
                if len(parts) >= 2:
                    projs.add(parts[1])
    return sorted(projs)


def lora_layer_chart(records: list[dict], x_key: str):
    proj_names = _detect_lora_projs(records)
    if not proj_names or not records:
        return _empty_figure("LoRA B Norm (by layer)", "mean ‖B‖₂")
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("lora/")}
        for i in range(8):
            vals = [r.get(f"lora/layer{i}.{p}.B") for p in proj_names]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"lora_layer_{i}"] = sum(vals) / len(vals)
        augmented.append(row)
    specs = [(f"lora_layer_{i}", f"L{i}", LAYER_COLORS[i]) for i in range(8)]
    return make_chart(augmented, x_key, specs, title="LoRA B Norm · by layer", y_title="mean ‖B‖₂")


def lora_proj_chart(records: list[dict], x_key: str):
    proj_names = _detect_lora_projs(records)
    if not proj_names or not records:
        return _empty_figure("LoRA B Norm (by projection)", "mean ‖B‖₂")
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("lora/")}
        for p in proj_names:
            vals = [r.get(f"lora/layer{i}.{p}.B") for i in range(8)]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"lora_proj_{p}"] = sum(vals) / len(vals)
        augmented.append(row)
    specs = [(f"lora_proj_{p}", p.upper(), _PROJ_COLORS.get(p, COLORS["slate"])) for p in proj_names]
    fig = make_chart(augmented, x_key, specs, title="LoRA B Norm · by projection", y_title="mean ‖B‖₂")
    for trace in fig.data:
        proj = trace.name.lower()
        dash = _PROJ_DASH.get(proj)
        if dash:
            trace.line = dict(dash=dash, width=TRACE_WIDTH)
    return fig


def lora_detail_chart(records: list[dict], x_key: str, proj: str):
    specs = [(f"lora/layer{i}.{proj}.B", f"L{i}", LAYER_COLORS[i]) for i in range(8)]
    dash = _PROJ_DASH.get(proj, "solid")
    fig = make_chart(records, x_key, specs, title=f"LoRA B Norm — {proj.upper()}", y_title="‖B‖₂")
    for trace in fig.data:
        trace.line = dict(dash=dash, width=TRACE_WIDTH)
    return fig


def sparse_delta_chart(records: list[dict], x_key: str):
    projs: set[str] = set()
    for r in records:
        for k in r:
            if k.startswith("sparse/layer"):
                parts = k.split(".")
                if len(parts) >= 2:
                    projs.add(parts[1])
    proj_names = sorted(projs)
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("sparse/")}
        for i in range(8):
            vals = [r.get(f"sparse/layer{i}.{p}.delta") for p in proj_names]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"sparse_layer_{i}"] = sum(vals) / len(vals)
        augmented.append(row)
    specs = [(f"sparse_layer_{i}", f"L{i}", LAYER_COLORS[i]) for i in range(8)]
    return make_chart(augmented, x_key, specs, title="Sparse Δ Norm · by layer", y_title="mean ‖Δ‖₂")


def bottleneck_up_chart(records: list[dict], x_key: str):
    specs = []
    for i in range(8):
        specs.append((f"adapter/layer{i}.attn.up", f"L{i} attn", theme.layer_color(i, saturation=0.78)))
        specs.append((f"adapter/layer{i}.ffn.up", f"L{i} ffn", theme.layer_color(i, saturation=0.38)))
    return make_chart(records, x_key, specs, title="Adapter Up-Proj Norm · by layer", y_title="‖W_up‖₂")


# ---------------------------------------------------------------------------
# Validation + error charts
# ---------------------------------------------------------------------------


def val_loss_chart(records: list[dict], x_key: str, run_type: str):
    """Validation loss (log-scaled). Adapter runs only — pawn uses ``perplexity_chart``."""
    specs = [("val_loss", "Val Loss", COLORS["accent"])]
    return make_chart(records, x_key, specs, title="Validation Loss", y_title="loss", y_log=True)


def perplexity_chart(records: list[dict], x_key: str = "step"):
    """Standalone log-scaled perplexity chart for pawn pretraining runs."""
    specs = [("val/perplexity", "Perplexity", COLORS["warning"])]
    return make_chart(records, x_key, specs, title="Perplexity", y_title="perplexity", y_log=True)


def val_accuracy_chart(records: list[dict], x_key: str, run_type: str):
    if run_type in _ADAPTER_TYPES:
        specs = [
            ("val_top1", "Val Top-1", COLORS["primary"]),
            ("val_top5", "Val Top-5", COLORS["success"]),
        ]
    elif run_type == "pawn":
        specs = [
            ("val/accuracy", "Top-1", COLORS["primary"]),
            ("val/top5_accuracy", "Top-5", COLORS["success"]),
        ]
    else:
        specs = [
            ("val_top1", "Val Top-1", COLORS["primary"]),
            ("val_top5", "Val Top-5", COLORS["success"]),
        ]
    return make_chart(records, x_key, specs, title="Validation Accuracy", y_title="rate")


def error_rate_chart(records: list[dict], x_key: str = "step", fit: bool = False):
    """Log-scale error rates (1 - legality / completion). Lower is better.

    When ``fit=True``, a log-linear regression is added per series using the
    most recent half of the data; slope is interpretable as exponential decay.
    """
    if not records:
        return _empty_figure("Error Rates (log)", "1 − rate")

    import numpy as np
    import plotly.graph_objects as go

    df = pd.DataFrame(records)
    derived: list[dict] = []
    for row in df.to_dict("records"):
        if x_key not in row or row[x_key] is None:
            continue
        entry: dict = {x_key: row[x_key]}
        if row.get("val/legal_move_rate") is not None:
            entry["illegal"] = 1.0 - row["val/legal_move_rate"]
        if row.get("val/late_legal_move_rate") is not None:
            entry["late_illegal"] = 1.0 - row["val/late_legal_move_rate"]
        if row.get("val/game_completion_rate") is not None:
            entry["forfeit"] = 1.0 - row["val/game_completion_rate"]
        if len(entry) > 1:
            derived.append(entry)

    specs = [
        ("forfeit", "Forfeit (per-game)", COLORS["error"]),
        ("late_illegal", "Late Illegal (per-move)", COLORS["violet"]),
        ("illegal", "Illegal (per-move)", COLORS["info"]),
    ]

    fig = go.Figure()
    fit_df = pd.DataFrame(derived).sort_values(x_key) if derived else pd.DataFrame()
    n_total = len(fit_df)
    half = fit_df.iloc[n_total // 2:] if n_total >= 4 else fit_df
    x_half = half[x_key].to_numpy(dtype=float) if not half.empty else np.array([])
    x_full = fit_df[x_key].to_numpy(dtype=float) if not fit_df.empty else np.array([])

    short_labels = {"illegal": "Illegal", "late_illegal": "Late Illegal", "forfeit": "Forfeit"}

    for key, label, color in specs:
        if key not in fit_df.columns:
            continue
        y_full = fit_df[key].to_numpy(dtype=float)
        valid_full = np.isfinite(y_full)
        if valid_full.sum() == 0:
            continue
        fig.add_scatter(
            x=x_full[valid_full], y=y_full[valid_full], mode="lines",
            line=dict(color=color, width=TRACE_WIDTH),
            name=label,
            hovertemplate=f"<b>%{{y:.4f}}</b><extra>{label}</extra>",
        )

        if fit and not half.empty and key in half.columns:
            y = half[key].to_numpy(dtype=float)
            valid = np.isfinite(y) & (y > 0)
            if valid.sum() >= 3:
                log_y = np.log(y[valid])
                xs = x_half[valid]
                slope, intercept = np.polyfit(xs, log_y, 1)
                y_fit = np.exp(intercept + slope * x_full)
                half_life = np.log(2) / abs(slope) if slope != 0 else float("inf")
                if not np.isfinite(half_life) or abs(slope) < 1e-12:
                    hl_str = "~flat"
                else:
                    hl_str = f"{half_life:,.0f}"
                short = short_labels.get(key, key)
                legend_name = f"{short} fit · s={slope:+.1e}, hl={hl_str}"
                fig.add_scatter(
                    x=x_full, y=y_fit, mode="lines",
                    line=dict(color=theme.desaturate(color), dash="dash", width=1.4),
                    name=legend_name,
                    hovertemplate=(
                        f"<b>%{{y:.4f}}</b><extra>{short} fit · slope={slope:.2e}/step</extra>"
                    ),
                )

    fig.update_yaxes(type="log", title="1 − rate")
    fig.update_xaxes(title=x_key.replace("_", " ").title())
    fig.update_layout(**theme.PLOTLY_LAYOUT)
    fig.update_layout(title="<b>Error Rates</b>", hovermode="x unified")
    return fig


def patience_chart(val_records: list[dict], x_key: str = "step",
                   patience_limit: int = 10):
    """Plot patience counter from val records with a limit line."""
    import plotly.graph_objects as go

    if not val_records:
        return _empty_figure("Patience", "evals without improvement")

    has_explicit = any(r.get("patience_counter") is not None for r in val_records)

    best_loss = float("inf")
    steps: list = []
    counters: list = []
    counter = 0
    for r in val_records:
        s = r.get(x_key)
        if s is None:
            continue
        if has_explicit:
            pc = r.get("patience_counter")
            if pc is not None:
                counter = pc
            steps.append(s)
            counters.append(counter)
        else:
            vl = r.get("val/loss")
            if vl is None:
                continue
            if vl < best_loss:
                best_loss = vl
                counter = 0
            else:
                counter += 1
            steps.append(s)
            counters.append(counter)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=counters, mode="lines+markers",
        name="Patience",
        line=dict(color=COLORS["warning"], width=TRACE_WIDTH),
        marker=dict(size=5, color=COLORS["warning"]),
        hovertemplate="<b>%{y}</b><extra>Patience</extra>",
    ))
    fig.add_hline(
        y=patience_limit,
        line_dash="dash",
        line_color=COLORS["error"],
        line_width=1.2,
        annotation_text=f"limit · {patience_limit}",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["error"], size=11),
    )
    fig.update_layout(**theme.PLOTLY_LAYOUT)
    fig.update_layout(
        title="<b>Early-Stopping Patience</b>",
        xaxis_title=x_key.capitalize(),
        yaxis_title="evals without improvement",
        yaxis=dict(range=[0, patience_limit + 2]),
    )
    return fig
