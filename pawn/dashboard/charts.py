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


def current_axis_log(fig) -> tuple[bool, bool]:
    """Return whether the figure's x and y axes are currently log-scaled."""
    layout = getattr(fig, "layout", None)
    xaxis = getattr(layout, "xaxis", None) if layout is not None else None
    yaxis = getattr(layout, "yaxis", None) if layout is not None else None
    xt = getattr(xaxis, "type", None) if xaxis is not None else None
    yt = getattr(yaxis, "type", None) if yaxis is not None else None
    return (xt == "log", yt == "log")


def apply_axis_log(fig, x_log: bool, y_log: bool):
    """Force-set the axis types on a figure.

    Used by the dashboard to apply per-chart log-scale overrides chosen by
    the user, on top of whatever default the chart builder picked. Any
    explicit axis range is cleared and autorange re-enabled so a range that
    only made sense in log space (e.g. ``[-2, 0]``) doesn't leave the chart
    looking empty when the user switches back to linear.
    """
    fig.update_xaxes(type="log" if x_log else "linear", range=None, autorange=True)
    fig.update_yaxes(type="log" if y_log else "linear", range=None, autorange=True)
    return fig


def is_empty_chart(fig) -> bool:
    """True when a figure has no data traces, or every trace is empty.

    Used by the dashboard shell to skip rendering a card when the base run
    literally has no data for a metric yet (e.g. end-of-epoch metrics before
    the first epoch finishes).
    """
    data = getattr(fig, "data", None)
    if data is None or len(data) == 0:
        return True
    for t in data:
        xs = getattr(t, "x", None)
        if xs is not None and len(xs) > 0:
            return False
    return True


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


# ---------------------------------------------------------------------------
# Comparison-run overlay helpers
# ---------------------------------------------------------------------------

COMPARE_SUFFIX = " (compare)"
COMPARE_DASH = "dot"
COMPARE_WIDTH = 1.5


MIN_BASE_POINTS_FOR_COMPARE = 2


def _add_compare_traces(
    fig,
    records: list[dict] | None,
    x_key: str,
    y_specs: list[tuple[str, str, str]],
    *,
    base_labels: set[str] | None = None,
    x_log: bool = False,
):
    """Overlay step-matched traces from a comparison run.

    Only emits a trace for a spec if (a) the column exists in ``records``,
    (b) the base figure already has a trace for that spec (so we never add
    charts for comparison-only metrics), (c) the base trace has at least
    ``MIN_BASE_POINTS_FOR_COMPARE`` points (a single-point base is visually
    invisible as a line, so an overlaid comparison looks orphaned), and
    (d) at least one finite point survives the optional ``x_log`` filter.
    """
    if not records or x_key is None:
        return
    df = pd.DataFrame(records)
    if x_key not in df.columns:
        return
    if base_labels is None:
        base_labels = {t.name for t in fig.data if t.name}
    base_point_counts = {
        t.name: (len(t.x) if t.x is not None else 0)
        for t in fig.data if t.name
    }
    for key, label, color in y_specs:
        if key not in df.columns or label not in base_labels:
            continue
        if base_point_counts.get(label, 0) < MIN_BASE_POINTS_FOR_COMPARE:
            continue
        sub = df[[x_key, key]].dropna()
        if x_log:
            sub = sub[sub[x_key] > 0]
        if sub.empty:
            continue
        comp_label = f"{label}{COMPARE_SUFFIX}"
        fig.add_scatter(
            x=sub[x_key].tolist(),
            y=sub[key].tolist(),
            mode="lines",
            line=dict(color=theme.desaturate(color), dash=COMPARE_DASH, width=COMPARE_WIDTH),
            name=comp_label,
            meta=comp_label,
            hovertemplate="%{y:.4f}<extra>%{meta}</extra>",
        )


def make_chart(
    records: list[dict],
    x_key: str,
    y_specs: list[tuple[str, str, str]],
    title: str = "",
    y_title: str = "",
    y_log: bool = False,
    x_log: bool = False,
    compare_records: list[dict] | None = None,
) -> px.line:
    """Create a themed Plotly line chart. y_specs: list of (key, label, color).

    ``x_log`` drops rows with ``x <= 0`` before plotting — ``log(0)`` would
    produce a discontinuity at the start of the curve. ``y_log`` relies on
    Plotly's own handling (non-positive y is simply not drawn).

    When ``compare_records`` is provided, any y_spec that already rendered a
    base trace is overlaid with a muted dotted trace from the comparison data.
    Specs missing from the base data are *not* promoted, so the comparison can
    never introduce a metric the user isn't already looking at.
    """
    melted, color_map = _melt(records, x_key, y_specs)
    if melted.empty:
        return _empty_figure(title, y_title)
    if x_log:
        melted = melted[melted[x_key] > 0]
        if melted.empty:
            return _empty_figure(title, y_title)
    base_labels = set(color_map.keys())
    fig = px.line(
        melted, x=x_key, y="value", color="metric",
        color_discrete_map=color_map,
        labels={"value": y_title, x_key: x_key.replace("_", " ").title(), "metric": ""},
    )
    if y_log:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(rangemode="tozero")
    if x_log:
        fig.update_xaxes(type="log")
    _style_figure(fig, title, y_title)
    _add_compare_traces(fig, compare_records, x_key, y_specs,
                        base_labels=base_labels, x_log=x_log)
    return fig


_ADAPTER_TYPES = {"film", "lora", "hybrid", "sparse", "bottleneck", "tiny", "rosa"}


# ---------------------------------------------------------------------------
# Core training charts
# ---------------------------------------------------------------------------


def _loglog_fit_chart(
    df: pd.DataFrame,
    x_key: str,
    specs: list[tuple[str, str, str]],
    title: str,
    y_title: str,
    fit: bool,
    compare_df: pd.DataFrame | None = None,
):
    """Build a log-log chart, one trace per spec, with an optional power-law
    fit overlay per series.

    The fit is OLS on ``(log x, log y)`` over the most recent half of each
    series' strictly positive points; the legend carries the exponent ``b``
    and ``halve@×N`` — the multiplicative x-ratio needed to halve y.
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    if df.empty or x_key not in df.columns:
        fig.update_layout(**theme.PLOTLY_LAYOUT)
        fig.update_layout(title=f"<b>{title}</b>", yaxis_title=y_title)
        return fig

    df = df.sort_values(x_key)
    x_full = df[x_key].to_numpy(dtype=float)
    x_pos_all = np.sort(x_full[x_full > 0])

    base_labels: set[str] = set()
    base_point_counts: dict[str, int] = {}
    for key, label, color in specs:
        if key not in df.columns:
            continue
        y = df[key].to_numpy(dtype=float)
        valid = np.isfinite(y) & (y > 0) & (x_full > 0)
        if valid.sum() == 0:
            continue
        xs_v = x_full[valid]
        ys_v = y[valid]
        fig.add_scatter(
            x=xs_v, y=ys_v, mode="lines",
            line=dict(color=color, width=TRACE_WIDTH),
            name=label,
            hovertemplate=f"<b>%{{y:.4f}}</b><extra>{label}</extra>",
        )
        base_labels.add(label)
        base_point_counts[label] = int(xs_v.size)
        if not fit or xs_v.size < 4:
            continue
        half = xs_v.size // 2
        log_x = np.log(xs_v[half:])
        log_y = np.log(ys_v[half:])
        if log_x.size < 3 or np.var(log_x) == 0:
            continue
        exponent, log_prefactor = np.polyfit(log_x, log_y, 1)
        if x_pos_all.size == 0:
            continue
        y_fit = np.exp(log_prefactor) * np.power(x_pos_all, exponent)
        # `2^(-1/b)` explodes as ``b`` approaches zero — a near-zero exponent
        # is a "no meaningful decay" signal, so show ~flat rather than a
        # 10²⁰-sized ratio that drowns the legend.
        FLAT_THRESHOLD = 0.05
        if exponent < -FLAT_THRESHOLD and np.isfinite(exponent):
            ratio = 2.0 ** (-1.0 / exponent)
            ratio_str = f"×{ratio:,.1f}" if np.isfinite(ratio) else "~flat"
        else:
            ratio_str = "~flat"
        legend_name = f"{label} fit · b={exponent:+.2f}, halve@{ratio_str}"
        fig.add_scatter(
            x=x_pos_all, y=y_fit, mode="lines",
            line=dict(color=theme.desaturate(color), dash="dash", width=1.4),
            name=legend_name,
            hovertemplate=(
                f"<b>%{{y:.4f}}</b><extra>{label} fit · "
                f"y = {np.exp(log_prefactor):.2e} · x^{exponent:+.2f}</extra>"
            ),
        )

    if compare_df is not None and not compare_df.empty and x_key in compare_df.columns:
        cdf = compare_df.sort_values(x_key)
        cx = cdf[x_key].to_numpy(dtype=float)
        for key, label, color in specs:
            if key not in cdf.columns or label not in base_labels:
                continue
            if base_point_counts.get(label, 0) < MIN_BASE_POINTS_FOR_COMPARE:
                continue
            cy = cdf[key].to_numpy(dtype=float)
            valid = np.isfinite(cy) & (cy > 0) & (cx > 0)
            if valid.sum() == 0:
                continue
            comp_label = f"{label}{COMPARE_SUFFIX}"
            fig.add_scatter(
                x=cx[valid], y=cy[valid], mode="lines",
                line=dict(color=theme.desaturate(color), dash=COMPARE_DASH, width=COMPARE_WIDTH),
                name=comp_label,
                hovertemplate=f"<b>%{{y:.4f}}</b><extra>{comp_label}</extra>",
            )

    fig.update_yaxes(type="log", title=y_title)
    fig.update_xaxes(type="log", title=x_key.replace("_", " ").title())
    fig.update_layout(**theme.PLOTLY_LAYOUT)
    fig.update_layout(title=f"<b>{title}</b>", hovermode="x unified")
    return fig


def loss_chart(records: list[dict], x_key: str, run_type: str,
               val_records: list[dict] | None = None, fit: bool = True,
               compare_records: list[dict] | None = None,
               compare_val_records: list[dict] | None = None):
    """Loss chart (log-log) with an optional power-law fit overlay per series.

    For pawn runs, ``val_records`` is concatenated onto ``records`` so val/loss
    overlays on top of train/loss. The comparison arguments mirror the primary
    ones: ``compare_records`` / ``compare_val_records`` are from the comparison
    run and, if supplied, are overlaid as muted dotted traces on any series
    already present in the base figure.
    """
    if run_type in _ADAPTER_TYPES | {"bc"}:
        specs = [("train_loss", "Train", COLORS["primary"]),
                 ("val_loss", "Val", COLORS["accent"])]
        src = records
        cmp_src = compare_records
    elif run_type == "pawn":
        specs = [("train/loss", "Train", COLORS["primary"]),
                 ("val/loss", "Val", COLORS["accent"])]
        src = list(records or []) + list(val_records or [])
        cmp_src = (list(compare_records or []) + list(compare_val_records or [])) or None
    else:
        specs = [("train_loss", "Train", COLORS["primary"]),
                 ("val_loss", "Val", COLORS["accent"])]
        src = records
        cmp_src = compare_records
    if not src:
        return _empty_figure("Loss", "cross-entropy")
    return _loglog_fit_chart(
        pd.DataFrame(src), x_key, specs,
        title="Loss", y_title="cross-entropy", fit=fit,
        compare_df=pd.DataFrame(cmp_src) if cmp_src else None,
    )


def accuracy_chart(records: list[dict], x_key: str, run_type: str,
                   compare_records: list[dict] | None = None):
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
    return make_chart(records, x_key, specs, title="Accuracy", y_title="rate",
                      compare_records=compare_records)


def lr_chart(train_records: list[dict], batch_records: list[dict], x_key: str,
             compare_train_records: list[dict] | None = None,
             compare_batch_records: list[dict] | None = None):
    src = batch_records if batch_records else train_records
    x = "global_step" if batch_records else x_key
    specs = [("lr", "LR", COLORS["warning"])]
    if src and "lr_lora" in src[0]:
        specs = [
            ("lr_lora", "LoRA LR", COLORS["accent"]),
            ("lr_film", "FiLM LR", COLORS["primary"]),
        ]
    cmp_src = compare_batch_records if batch_records else compare_train_records
    return make_chart(src, x, specs, title="Learning Rate", y_title="lr",
                      compare_records=cmp_src)


def grad_chart(records: list[dict], x_key: str,
               compare_records: list[dict] | None = None):
    return make_chart(records, x_key, [
        ("grad_norm", "Grad Norm", COLORS["violet"]),
        ("grad_norm_mean", "Mean", COLORS["primary"]),
        ("grad_norm_max", "Max", COLORS["error"]),
    ], title="Gradient Norm", y_title="‖g‖₂", y_log=True,
       compare_records=compare_records)


def gpu_chart(records: list[dict], x_key: str,
              compare_records: list[dict] | None = None):
    return make_chart(records, x_key, [
        ("mem/gpu_peak_gb", "Peak", COLORS["info"]),
        ("mem/gpu_reserved_gb", "Reserved", COLORS["warning"]),
        ("mem/gpu_current_gb", "Current", COLORS["success"]),
    ], title="GPU Memory", y_title="GB", compare_records=compare_records)


def time_chart(records: list[dict], x_key: str, run_type: str,
               compare_records: list[dict] | None = None):
    key = "epoch_time_s" if run_type in _ADAPTER_TYPES else ("epoch_time" if run_type == "bc" else "step_time")
    label = "Epoch Time" if run_type in (_ADAPTER_TYPES | {"bc"}) else "Step Time"
    return make_chart(records, x_key, [(key, label, COLORS["slate"])],
                      title=label, y_title="seconds",
                      compare_records=compare_records)


# ---------------------------------------------------------------------------
# Adapter diagnostics
# ---------------------------------------------------------------------------


def film_weight_chart(records: list[dict], x_key: str,
                      compare_records: list[dict] | None = None):
    specs = [(f"film/hidden_{i}/gamma_dev", f"L{i} γ", LAYER_COLORS[i]) for i in range(8)]
    specs.append(("film/output/gamma_dev", "Out γ", OUTPUT_LAYER_COLOR))
    return make_chart(records, x_key, specs, title="FiLM γ Deviation", y_title="‖γ − 1‖₂",
                      compare_records=compare_records)


def film_beta_chart(records: list[dict], x_key: str,
                    compare_records: list[dict] | None = None):
    specs = [(f"film/hidden_{i}/beta_norm", f"L{i} β", LAYER_COLORS[i]) for i in range(8)]
    specs.append(("film/output/beta_norm", "Out β", OUTPUT_LAYER_COLOR))
    return make_chart(records, x_key, specs, title="FiLM β Norm", y_title="‖β‖₂",
                      compare_records=compare_records)


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


def _augment_lora_layer(records: list[dict], proj_names: list[str]) -> list[dict]:
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("lora/")}
        for i in range(8):
            vals = [r.get(f"lora/layer{i}.{p}.B") for p in proj_names]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"lora_layer_{i}"] = sum(vals) / len(vals)
        augmented.append(row)
    return augmented


def _augment_lora_proj(records: list[dict], proj_names: list[str]) -> list[dict]:
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("lora/")}
        for p in proj_names:
            vals = [r.get(f"lora/layer{i}.{p}.B") for i in range(8)]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"lora_proj_{p}"] = sum(vals) / len(vals)
        augmented.append(row)
    return augmented


def lora_layer_chart(records: list[dict], x_key: str,
                     compare_records: list[dict] | None = None):
    proj_names = _detect_lora_projs(records)
    if not proj_names or not records:
        return _empty_figure("LoRA B Norm (by layer)", "mean ‖B‖₂")
    augmented = _augment_lora_layer(records, proj_names)
    cmp_aug = _augment_lora_layer(compare_records, proj_names) if compare_records else None
    specs = [(f"lora_layer_{i}", f"L{i}", LAYER_COLORS[i]) for i in range(8)]
    return make_chart(augmented, x_key, specs, title="LoRA B Norm · by layer",
                      y_title="mean ‖B‖₂", compare_records=cmp_aug)


def lora_proj_chart(records: list[dict], x_key: str,
                    compare_records: list[dict] | None = None):
    proj_names = _detect_lora_projs(records)
    if not proj_names or not records:
        return _empty_figure("LoRA B Norm (by projection)", "mean ‖B‖₂")
    augmented = _augment_lora_proj(records, proj_names)
    cmp_aug = _augment_lora_proj(compare_records, proj_names) if compare_records else None
    specs = [(f"lora_proj_{p}", p.upper(), _PROJ_COLORS.get(p, COLORS["slate"])) for p in proj_names]
    fig = make_chart(augmented, x_key, specs, title="LoRA B Norm · by projection",
                     y_title="mean ‖B‖₂", compare_records=cmp_aug)
    for trace in fig.data:
        proj_key = (trace.name or "").replace(COMPARE_SUFFIX, "").lower()
        dash = _PROJ_DASH.get(proj_key)
        if dash:
            is_compare = trace.name and COMPARE_SUFFIX in trace.name
            trace.line = dict(
                dash=COMPARE_DASH if is_compare else dash,
                width=COMPARE_WIDTH if is_compare else TRACE_WIDTH,
                color=getattr(trace.line, "color", None),
            )
    return fig


def lora_detail_chart(records: list[dict], x_key: str, proj: str,
                      compare_records: list[dict] | None = None):
    specs = [(f"lora/layer{i}.{proj}.B", f"L{i}", LAYER_COLORS[i]) for i in range(8)]
    dash = _PROJ_DASH.get(proj, "solid")
    fig = make_chart(records, x_key, specs, title=f"LoRA B Norm — {proj.upper()}",
                     y_title="‖B‖₂", compare_records=compare_records)
    for trace in fig.data:
        is_compare = trace.name and COMPARE_SUFFIX in trace.name
        if is_compare:
            continue
        trace.line = dict(dash=dash, width=TRACE_WIDTH,
                          color=getattr(trace.line, "color", None))
    return fig


def _detect_sparse_projs(records: list[dict]) -> list[str]:
    projs: set[str] = set()
    for r in records:
        for k in r:
            if k.startswith("sparse/layer"):
                parts = k.split(".")
                if len(parts) >= 2:
                    projs.add(parts[1])
    return sorted(projs)


def _augment_sparse_delta(records: list[dict], proj_names: list[str]) -> list[dict]:
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("sparse/")}
        for i in range(8):
            vals = [r.get(f"sparse/layer{i}.{p}.delta") for p in proj_names]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"sparse_layer_{i}"] = sum(vals) / len(vals)
        augmented.append(row)
    return augmented


def sparse_delta_chart(records: list[dict], x_key: str,
                       compare_records: list[dict] | None = None):
    proj_names = _detect_sparse_projs(records)
    augmented = _augment_sparse_delta(records, proj_names)
    cmp_aug = None
    if compare_records:
        cmp_projs = _detect_sparse_projs(compare_records) or proj_names
        cmp_aug = _augment_sparse_delta(compare_records, cmp_projs)
    specs = [(f"sparse_layer_{i}", f"L{i}", LAYER_COLORS[i]) for i in range(8)]
    return make_chart(augmented, x_key, specs, title="Sparse Δ Norm · by layer",
                      y_title="mean ‖Δ‖₂", compare_records=cmp_aug)


def bottleneck_up_chart(records: list[dict], x_key: str,
                        compare_records: list[dict] | None = None):
    specs = []
    for i in range(8):
        specs.append((f"adapter/layer{i}.attn.up", f"L{i} attn", theme.layer_color(i, saturation=0.78)))
        specs.append((f"adapter/layer{i}.ffn.up", f"L{i} ffn", theme.layer_color(i, saturation=0.38)))
    return make_chart(records, x_key, specs, title="Adapter Up-Proj Norm · by layer",
                      y_title="‖W_up‖₂", compare_records=compare_records)


# ---------------------------------------------------------------------------
# Validation + error charts
# ---------------------------------------------------------------------------


def val_loss_chart(records: list[dict], x_key: str, run_type: str,
                   compare_records: list[dict] | None = None):
    """Validation loss (log-log). Adapter runs only — pawn uses ``perplexity_chart``."""
    specs = [("val_loss", "Val Loss", COLORS["accent"])]
    return make_chart(records, x_key, specs, title="Validation Loss", y_title="loss",
                      y_log=True, x_log=True, compare_records=compare_records)


def perplexity_chart(records: list[dict], x_key: str = "step",
                     compare_records: list[dict] | None = None):
    """Standalone log-log perplexity chart for pawn pretraining runs."""
    specs = [("val/perplexity", "Perplexity", COLORS["warning"])]
    return make_chart(records, x_key, specs, title="Perplexity", y_title="perplexity",
                      y_log=True, x_log=True, compare_records=compare_records)


def val_accuracy_chart(records: list[dict], x_key: str, run_type: str,
                       compare_records: list[dict] | None = None):
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
    return make_chart(records, x_key, specs, title="Validation Accuracy", y_title="rate",
                      compare_records=compare_records)


def _derive_error_rows(records: list[dict], x_key: str) -> list[dict]:
    derived: list[dict] = []
    for row in records:
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
    return derived


def error_rate_chart(records: list[dict], x_key: str = "step", fit: bool = False,
                     compare_records: list[dict] | None = None):
    """Log-log error rates (1 - legality / completion). Lower is better.

    When ``fit=True``, a power-law fit ``y = A * x^b`` is overlaid per series
    using the most recent half of the data. OLS runs in log-log space, so the
    exponent ``b`` is a straight-line slope on the chart. A negative exponent
    means the rate is decaying; ``x_ratio_to_halve = 2^(-1/b)`` summarises how
    many times x must multiply to halve y.
    """
    if not records:
        return _empty_figure("Error Rates", "1 − rate")

    derived = _derive_error_rows(records, x_key)
    cmp_derived = _derive_error_rows(compare_records, x_key) if compare_records else None

    specs = [
        ("forfeit", "Forfeit (per-game)", COLORS["error"]),
        ("late_illegal", "Late Illegal (per-move)", COLORS["violet"]),
        ("illegal", "Illegal (per-move)", COLORS["info"]),
    ]
    return _loglog_fit_chart(
        pd.DataFrame(derived), x_key, specs,
        title="Error Rates", y_title="1 − rate", fit=fit,
        compare_df=pd.DataFrame(cmp_derived) if cmp_derived else None,
    )


def _patience_series(val_records: list[dict], x_key: str) -> tuple[list, list]:
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
    return steps, counters


def patience_chart(val_records: list[dict], x_key: str = "step",
                   patience_limit: int = 10,
                   compare_records: list[dict] | None = None):
    """Plot patience counter from val records with a limit line."""
    import plotly.graph_objects as go

    if not val_records:
        return _empty_figure("Patience", "evals without improvement")

    steps, counters = _patience_series(val_records, x_key)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=counters, mode="lines+markers",
        name="Patience",
        line=dict(color=COLORS["warning"], width=TRACE_WIDTH),
        marker=dict(size=5, color=COLORS["warning"]),
        hovertemplate="<b>%{y}</b><extra>Patience</extra>",
    ))
    if compare_records:
        c_steps, c_counters = _patience_series(compare_records, x_key)
        if c_steps:
            comp_name = f"Patience{COMPARE_SUFFIX}"
            fig.add_trace(go.Scatter(
                x=c_steps, y=c_counters, mode="lines+markers",
                name=comp_name,
                line=dict(color=theme.desaturate(COLORS["warning"]), dash=COMPARE_DASH, width=COMPARE_WIDTH),
                marker=dict(size=4, color=theme.desaturate(COLORS["warning"])),
                hovertemplate=f"<b>%{{y}}</b><extra>{comp_name}</extra>",
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
