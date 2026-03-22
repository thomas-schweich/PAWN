"""Plotly chart builders for PAWN dashboard."""

import colorsys

import pandas as pd
import plotly.express as px

LAYOUT = dict(
    template="plotly_dark",
    height=300,
    margin=dict(l=40, r=10, t=40, b=30),
)

COLORS = {
    "blue": "#4c72b0",
    "red": "#c44e52",
    "green": "#55a868",
    "orange": "#dd8452",
    "purple": "#8172b3",
    "gold": "#ccb974",
    "brown": "#937860",
}


def _layer_color(layer_idx: int, n_layers: int = 8,
                 saturation: float = 0.70, lightness: float = 0.58) -> str:
    """Color for a layer index on a red -> purple -> blue gradient."""
    t = layer_idx / max(n_layers - 1, 1)
    hue = (2 / 3 + t * 120 / 360) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


LAYER_COLORS = [_layer_color(i) for i in range(8)]
OUTPUT_COLOR = "#ccb974"


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


def make_chart(
    records: list[dict],
    x_key: str,
    y_specs: list[tuple[str, str, str]],
    title: str = "",
    y_title: str = "",
    y_log: bool = False,
) -> px.line:
    """Create a Plotly Express line chart. y_specs: list of (key, label, color)."""
    melted, color_map = _melt(records, x_key, y_specs)
    if melted.empty:
        fig = px.line(title=title)
    else:
        fig = px.line(
            melted, x=x_key, y="value", color="metric",
            color_discrete_map=color_map,
            labels={"value": y_title, x_key: x_key.replace("_", " ").title(), "metric": ""},
        )
        fig.update_traces(
            hovertemplate="%{meta}<br>%{x}: <b>%{y:.4f}</b><extra></extra>",
        )
        for trace in fig.data:
            trace.meta = trace.name
    if y_log:
        fig.update_yaxes(type="log")
    fig.update_layout(title=title, **LAYOUT)
    return fig


_ADAPTER_TYPES = {"film", "lora", "hybrid", "sparse", "bottleneck", "tiny"}


def loss_chart(records: list[dict], x_key: str, run_type: str):
    if run_type in _ADAPTER_TYPES | {"bc"}:
        specs = [("train_loss", "Train", COLORS["blue"]), ("val_loss", "Val", COLORS["red"])]
    elif run_type == "pawn":
        specs = [("train/loss", "Loss", COLORS["blue"])]
    else:
        specs = [("train_loss", "Train", COLORS["blue"]), ("val_loss", "Val", COLORS["red"])]
    return make_chart(records, x_key, specs, title="Loss", y_title="Loss")


def accuracy_chart(records: list[dict], x_key: str, run_type: str):
    if run_type in _ADAPTER_TYPES:
        specs = [
            ("train_top1", "Train Top-1", COLORS["blue"]),
            ("val_top1", "Val Top-1", COLORS["red"]),
            ("val_top5", "Val Top-5", COLORS["green"]),
        ]
    elif run_type == "bc":
        specs = [("train_acc", "Train", COLORS["blue"]), ("val_acc", "Val", COLORS["red"])]
    elif run_type == "pawn":
        specs = [("train/accuracy", "Top-1 Accuracy", COLORS["blue"])]
    else:
        specs = [("train_top1", "Train Top-1", COLORS["blue"]), ("val_top1", "Val Top-1", COLORS["red"])]
    return make_chart(records, x_key, specs, title="Accuracy", y_title="Rate")


def lr_chart(train_records: list[dict], batch_records: list[dict], x_key: str):
    src = batch_records if batch_records else train_records
    x = "global_step" if batch_records else x_key
    specs = [("lr", "LR", COLORS["gold"])]
    if src and "lr_lora" in src[0]:
        specs = [
            ("lr_lora", "LoRA LR", COLORS["red"]),
            ("lr_film", "FiLM LR", COLORS["blue"]),
        ]
    return make_chart(src, x, specs, title="Learning Rate", y_title="LR")


def grad_chart(records: list[dict], x_key: str):
    return make_chart(records, x_key, [
        ("grad_norm", "Grad Norm", COLORS["purple"]),
        ("grad_norm_mean", "Mean", COLORS["purple"]),
        ("grad_norm_max", "Max", COLORS["red"]),
    ], title="Gradient Norm", y_title="Norm")


def gpu_chart(records: list[dict], x_key: str):
    return make_chart(records, x_key, [
        ("mem/gpu_peak_gb", "Peak", COLORS["blue"]),
        ("mem/gpu_reserved_gb", "Reserved", COLORS["orange"]),
        ("mem/gpu_current_gb", "Current", COLORS["green"]),
    ], title="GPU Memory", y_title="GB")


def time_chart(records: list[dict], x_key: str, run_type: str):
    key = "epoch_time_s" if run_type in _ADAPTER_TYPES else ("epoch_time" if run_type == "bc" else "step_time")
    label = "Epoch Time" if run_type in (_ADAPTER_TYPES | {"bc"}) else "Step Time"
    return make_chart(records, x_key, [(key, label, COLORS["brown"])], title=label, y_title="Seconds")


def film_weight_chart(records: list[dict], x_key: str):
    specs = []
    for i in range(8):
        specs.append((f"film/hidden_{i}/gamma_dev", f"L{i} \u03b3", LAYER_COLORS[i]))
    specs.append(("film/output/gamma_dev", "Out \u03b3", OUTPUT_COLOR))
    return make_chart(records, x_key, specs, title="FiLM \u03b3 Deviation from Identity", y_title="||\u03b3 - 1||\u2082")


def film_beta_chart(records: list[dict], x_key: str):
    specs = []
    for i in range(8):
        specs.append((f"film/hidden_{i}/beta_norm", f"L{i} \u03b2", LAYER_COLORS[i]))
    specs.append(("film/output/beta_norm", "Out \u03b2", OUTPUT_COLOR))
    return make_chart(records, x_key, specs, title="FiLM \u03b2 Norm", y_title="||\u03b2||\u2082")


_PROJ_DASH = {
    "wq": "solid",
    "wv": "dot",
    "wk": "dash",
    "wo": "dashdot",
}

_PROJ_COLORS = {
    "wq": "#e04040",
    "wk": "#dd8452",
    "wv": "#4c72b0",
    "wo": "#8172b3",
}


def _detect_lora_projs(records: list[dict]) -> list[str]:
    projs = set()
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
        return make_chart([], x_key, [], title="LoRA B Norm (by layer)")
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
    return make_chart(augmented, x_key, specs, title="LoRA B Norm (by layer)", y_title="mean ||B||\u2082")


def lora_proj_chart(records: list[dict], x_key: str):
    proj_names = _detect_lora_projs(records)
    if not proj_names or not records:
        return make_chart([], x_key, [], title="LoRA B Norm (by projection)")
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("lora/")}
        for p in proj_names:
            vals = [r.get(f"lora/layer{i}.{p}.B") for i in range(8)]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"lora_proj_{p}"] = sum(vals) / len(vals)
        augmented.append(row)
    specs = [(f"lora_proj_{p}", p.upper(), _PROJ_COLORS.get(p, "#888888")) for p in proj_names]
    fig = make_chart(augmented, x_key, specs, title="LoRA B Norm (by projection)", y_title="mean ||B||\u2082")
    for trace in fig.data:
        proj = trace.name.lower()
        dash = _PROJ_DASH.get(proj)
        if dash:
            trace.line = dict(dash=dash)
    return fig


def lora_detail_chart(records: list[dict], x_key: str, proj: str):
    specs = []
    dash = _PROJ_DASH.get(proj, "solid")
    for i in range(8):
        specs.append((f"lora/layer{i}.{proj}.B", f"L{i}", LAYER_COLORS[i]))
    fig = make_chart(records, x_key, specs,
                     title=f"LoRA B Norm \u2014 {proj.upper()}", y_title="||B||\u2082")
    for trace in fig.data:
        trace.line = dict(dash=dash)
    return fig


def sparse_delta_chart(records: list[dict], x_key: str):
    projs = set()
    for r in records:
        for k in r:
            if k.startswith("sparse/layer"):
                parts = k.split(".")
                if len(parts) >= 2:
                    projs.add(parts[1])
    projs = sorted(projs)
    augmented = []
    for r in records:
        row = {k: v for k, v in r.items() if not k.startswith("sparse/")}
        for i in range(8):
            vals = [r.get(f"sparse/layer{i}.{p}.delta") for p in projs]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"sparse_layer_{i}"] = sum(vals) / len(vals)
        augmented.append(row)
    specs = [(f"sparse_layer_{i}", f"L{i}", LAYER_COLORS[i]) for i in range(8)]
    return make_chart(augmented, x_key, specs, title="Sparse \u0394 Norm (by layer)", y_title="mean ||\u0394||\u2082")


def bottleneck_up_chart(records: list[dict], x_key: str):
    specs = []
    for i in range(8):
        specs.append((f"adapter/layer{i}.attn.up", f"L{i} attn", _layer_color(i, saturation=0.80)))
        specs.append((f"adapter/layer{i}.ffn.up", f"L{i} ffn", _layer_color(i, saturation=0.40)))
    return make_chart(records, x_key, specs, title="Adapter Up-Proj Norm (by layer)", y_title="||W_up||\u2082")


def val_loss_chart(records: list[dict], x_key: str, run_type: str):
    if run_type in _ADAPTER_TYPES:
        specs = [("val_loss", "Val Loss", COLORS["red"])]
    elif run_type == "pawn":
        specs = [
            ("val/loss", "Val Loss", COLORS["red"]),
            ("val/perplexity", "Perplexity", COLORS["orange"]),
        ]
    else:
        specs = [("val_loss", "Val Loss", COLORS["red"])]
    return make_chart(records, x_key, specs, title="Validation Loss", y_title="Loss")


def val_accuracy_chart(records: list[dict], x_key: str, run_type: str):
    if run_type in _ADAPTER_TYPES:
        specs = [
            ("val_top1", "Val Top-1", COLORS["blue"]),
            ("val_top5", "Val Top-5", COLORS["green"]),
        ]
    elif run_type == "pawn":
        specs = [
            ("val/accuracy", "Top-1", COLORS["blue"]),
            ("val/top5_accuracy", "Top-5", COLORS["green"]),
            ("val/legal_move_rate", "Legal Rate", COLORS["orange"]),
        ]
    else:
        specs = [
            ("val_top1", "Val Top-1", COLORS["blue"]),
            ("val_top5", "Val Top-5", COLORS["green"]),
        ]
    return make_chart(records, x_key, specs, title="Validation Accuracy", y_title="Rate")
