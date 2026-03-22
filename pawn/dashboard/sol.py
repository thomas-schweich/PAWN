"""PAWN Dashboard — Solara components.

Standalone:  solara run pawn.dashboard.sol
Jupyter:     from pawn.dashboard import Dashboard; Dashboard()
"""

import os
import signal
import subprocess
import threading
import time as _time
from pathlib import Path

import solara

from . import charts
from .metrics import detect_run_type, get_run_hostname, load_metrics, load_runs

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
    "film": {"project": PROJECT_ROOT, "script": "scripts/train_film.py"},
    "lora": {"project": PROJECT_ROOT, "script": "scripts/train_lora.py"},
    "hybrid": {"project": PROJECT_ROOT, "script": "scripts/train_hybrid.py"},
    "sparse": {"project": PROJECT_ROOT, "script": "scripts/train_sparse.py"},
    "bottleneck": {"project": PROJECT_ROOT, "script": "scripts/train_bottleneck.py"},
    "tiny": {"project": PROJECT_ROOT, "script": "scripts/train_tiny.py"},
}

# ---------------------------------------------------------------------------
# Dashboard components
# ---------------------------------------------------------------------------


@solara.component
def RunSelector(auto_refresh: bool = False, on_auto_refresh=None, interval: float = 10.0, on_interval=None):
    """Run dropdown with refresh and optional live-update controls."""
    show_all, set_show_all = solara.use_state(False)

    def _load():
        hours = 0 if show_all else 1.0
        raw = load_runs(log_dir.value, max_age_hours=hours)
        labeled = []
        label_to_name = {}
        for name in raw:
            host = get_run_hostname(log_dir.value, name)
            label = f"{name} @ {host}" if host else name
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

    with solara.Row(style={"align-items": "center", "gap": "8px", "flex-wrap": "wrap"}):
        if labeled:
            solara.Select(label="Run", value=current_label, values=labeled, on_value=on_select)
        else:
            solara.Info("No recent runs found (try Show All)" if not show_all else "No runs found in " + str(log_dir.value))
        solara.Button(
            "Refresh",
            on_click=lambda: metrics_tick.set(metrics_tick.value + 1),
            icon_name="mdi-refresh",
        )
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


def _flatten_config(config: dict) -> dict[str, str]:
    """Flatten nested config dicts into dot-separated key-value pairs."""
    flat: dict[str, str] = {}
    skip = {"type", "timestamp", "compiled"}
    for k, v in config.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 in skip:
                    continue
                flat[f"{k}.{k2}"] = str(v2)
        else:
            flat[k] = str(v)
    return flat


@solara.component
def ConfigSummary():
    """Config summary for the selected run."""
    data = solara.use_memo(
        lambda: load_metrics(log_dir.value, selected_run.value) if selected_run.value else {},
        dependencies=[selected_run.value, metrics_tick.value],
    )
    if not selected_run.value:
        return

    configs = data.get("config", [])
    config = configs[-1] if configs else {}
    run_type = detect_run_type(config)
    flat = _flatten_config(config)

    hostname = config.get("hostname", "")
    host_str = f" | **Host:** {hostname}" if hostname else ""
    solara.Markdown(f"**Run:** `{selected_run.value}` | **Type:** {run_type}{host_str}")

    if flat:
        items = list(flat.items())
        n_cols = 3
        n_rows = (len(items) + n_cols - 1) // n_cols
        cols = [items[i * n_rows:(i + 1) * n_rows] for i in range(n_cols)]

        header = "| Parameter | Value " * n_cols + "|"
        sep = "| --- | --- " * n_cols + "|"
        rows = []
        for r in range(n_rows):
            cells = ""
            for c in range(n_cols):
                if r < len(cols[c]):
                    k, v = cols[c][r]
                    cells += f"| `{k}` | `{v}` "
                else:
                    cells += "| | "
            rows.append(cells + "|")

        solara.Markdown("\n".join([header, sep, *rows]))


_CHART_DESCRIPTIONS = {
    "film": {
        "loss": "Cross-entropy loss on Lichess move prediction with illegal-move masking (train vs val)",
        "accuracy": "Move prediction accuracy (train top-1, val top-1/top-5)",
        "film_gamma": "Per-layer \u03b3 deviation from identity (||\u03b3-1||\u2082). Layers with large deviation contributed to adaptation",
        "film_beta": "Per-layer \u03b2 norm (||\u03b2||\u2082). Hidden-layer \u03b2 shifts position-dependent features; output \u03b2 acts as a fixed move prior",
        "lr": "Learning rate with linear warmup and cosine decay",
        "time": "Wall-clock time per epoch",
    },
    "pawn": {
        "loss": "Cross-entropy loss on next-token prediction (lower = better at predicting which random move was played)",
        "accuracy": "Fraction of positions where the model's top-1 prediction matches the actual next move",
        "val_loss": "Validation loss and perplexity on held-out games",
        "val_accuracy": "Validation top-1/top-5 accuracy and legal move rate (fraction of predictions that are legal chess moves)",
        "lr": "Learning rate with linear warmup and cosine decay",
        "grad_norm": "L2 norm of gradients before clipping",
        "gpu": "GPU memory usage: peak, reserved, current",
        "time": "Wall-clock time per training step",
    },
    "lora": {
        "loss": "Cross-entropy loss on Lichess move prediction with illegal-move masking",
        "accuracy": "Move prediction accuracy (train top-1, val top-1/top-5)",
        "lora_layer": "Mean LoRA B-norm per layer. Shows which layers are adapting most",
        "lora_proj": "Mean LoRA B-norm per projection type. Shows Q vs K vs V vs O adaptation",
        "time": "Wall-clock time per epoch",
    },
    "hybrid": {
        "loss": "Cross-entropy loss on Lichess move prediction with illegal-move masking",
        "accuracy": "Move prediction accuracy (train top-1, val top-1/top-5)",
        "lora_layer": "Mean LoRA B-norm per layer",
        "lora_proj": "Mean LoRA B-norm per projection type",
        "film_gamma": "Per-layer FiLM \u03b3 deviation from identity",
        "film_beta": "Per-layer FiLM \u03b2 norm",
        "time": "Wall-clock time per epoch",
    },
}

_DEFAULT_DESCRIPTIONS = {
    "lr": "Learning rate schedule",
    "grad_norm": "Gradient norm before clipping",
    "gpu": "GPU memory usage",
    "time": "Time per step/epoch",
}


@solara.component
def ChartWithInfo(chart, description: str = ""):
    """Chart with an info icon tooltip for the description."""
    if description:
        with solara.Row(style={"align-items": "center", "gap": "4px", "margin-bottom": "-8px"}):
            with solara.Tooltip(tooltip=description):
                solara.Text("\u2139", style="cursor: help; opacity: 0.5; font-size: 14px;")
    solara.FigurePlotly(chart)


@solara.component
def MetricsCharts():
    """Chart grid for the selected run, adapted to run type."""
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
    x_key = "epoch" if run_type in ("bc", "film", "lora", "hybrid", "sparse", "bottleneck") else "step"

    descs = _CHART_DESCRIPTIONS.get(run_type, {})

    def desc(key: str) -> str:
        return descs.get(key, _DEFAULT_DESCRIPTIONS.get(key, ""))

    if run_type == "hybrid":
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.loss_chart(train, x_key, run_type), desc("loss"))
            ChartWithInfo(charts.accuracy_chart(train, x_key, run_type), desc("accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.lora_layer_chart(train, x_key), desc("lora_layer"))
            ChartWithInfo(charts.lora_proj_chart(train, x_key), desc("lora_proj"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.film_weight_chart(train, x_key), desc("film_gamma"))
            ChartWithInfo(charts.film_beta_chart(train, x_key), desc("film_beta"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.lr_chart(train, batch, x_key), desc("lr"))
            ChartWithInfo(charts.time_chart(train, x_key, run_type), desc("time"))
    elif run_type == "sparse":
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.loss_chart(train, x_key, run_type), desc("loss"))
            ChartWithInfo(charts.accuracy_chart(train, x_key, run_type), desc("accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.sparse_delta_chart(train, x_key), desc("sparse_delta"))
            ChartWithInfo(charts.time_chart(train, x_key, run_type), desc("time"))
    elif run_type == "bottleneck":
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.loss_chart(train, x_key, run_type), desc("loss"))
            ChartWithInfo(charts.accuracy_chart(train, x_key, run_type), desc("accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.bottleneck_up_chart(train, x_key), desc("adapter_up"))
            ChartWithInfo(charts.lr_chart(train, batch, x_key), desc("lr"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.gpu_chart(train, x_key), desc("gpu"))
            ChartWithInfo(charts.time_chart(train, x_key, run_type), desc("time"))
    elif run_type == "tiny":
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.loss_chart(train, x_key, run_type), desc("loss"))
            ChartWithInfo(charts.accuracy_chart(train, x_key, run_type), desc("accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.lr_chart(train, batch, x_key), desc("lr"))
            ChartWithInfo(charts.gpu_chart(train, x_key), desc("gpu"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.time_chart(train, x_key, run_type), desc("time"))
    elif run_type == "lora":
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.loss_chart(train, x_key, run_type), desc("loss"))
            ChartWithInfo(charts.accuracy_chart(train, x_key, run_type), desc("accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.lora_layer_chart(train, x_key), desc("lora_layer"))
            ChartWithInfo(charts.lora_proj_chart(train, x_key), desc("lora_proj"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.lr_chart(train, batch, x_key), desc("lr"))
            ChartWithInfo(charts.time_chart(train, x_key, run_type), desc("time"))
    elif run_type == "film":
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.loss_chart(train, x_key, run_type), desc("loss"))
            ChartWithInfo(charts.accuracy_chart(train, x_key, run_type), desc("accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.film_weight_chart(train, x_key), desc("film_gamma"))
            ChartWithInfo(charts.film_beta_chart(train, x_key), desc("film_beta"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.lr_chart(train, batch, x_key), desc("lr"))
            ChartWithInfo(charts.time_chart(train, x_key, run_type), desc("time"))
    else:
        # Default: pretraining or unknown
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.loss_chart(train, x_key, run_type), desc("loss"))
            ChartWithInfo(charts.accuracy_chart(train, x_key, run_type), desc("accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.val_loss_chart(val, x_key, run_type), desc("val_loss"))
            ChartWithInfo(charts.val_accuracy_chart(val, x_key, run_type), desc("val_accuracy"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.lr_chart(train, batch, x_key), desc("lr"))
            ChartWithInfo(charts.grad_chart(train, x_key), desc("grad_norm"))
        with solara.Columns([1, 1]):
            ChartWithInfo(charts.gpu_chart(train, x_key), desc("gpu"))
            ChartWithInfo(charts.time_chart(train, x_key, run_type), desc("time"))


@solara.component
def AutoRefresh(interval: float = 10.0):
    """Invisible component that bumps metrics_tick every `interval` seconds."""
    def setup():
        stop = threading.Event()

        def _tick_loop():
            while not stop.wait(interval):
                metrics_tick.set(metrics_tick.value + 1)

        t = threading.Thread(target=_tick_loop, daemon=True)
        t.start()
        return lambda: stop.set()

    solara.use_effect(setup, [interval])


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

    solara.Markdown("## Training Metrics")
    RunSelector(
        auto_refresh=auto_refresh,
        on_auto_refresh=set_auto_refresh,
        interval=interval,
        on_interval=set_interval,
    )
    ConfigSummary()
    MetricsCharts()


# ---------------------------------------------------------------------------
# Runner components
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
            # Adapter training
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
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=cwd,
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
                runner_output.set(
                    runner_output.value + f"\n[exited {proc.returncode}]\n"
                )
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

    with solara.Card("Launch Training Run"):
        with solara.Columns([1, 1]):
            with solara.Column():
                solara.Select(
                    label="Run Type",
                    value=run_type,
                    on_value=set_run_type,
                    values=list(TRAIN_SCRIPTS.keys()),
                )
                solara.InputText(
                    "Checkpoint", value=checkpoint, on_value=set_checkpoint,
                )
                if run_type != "pawn":
                    solara.InputText(
                        "PGN File", value=pgn_path, on_value=set_pgn_path,
                    )
                solara.InputText("Learning Rate", value=lr, on_value=set_lr)
                solara.InputText("Batch Size", value=batch_size, on_value=set_batch_size)
                steps_label = "Total Steps" if run_type == "pawn" else "Epochs"
                solara.InputText(steps_label, value=steps, on_value=set_steps)
                solara.InputText("Extra Args", value=extra_args, on_value=set_extra_args)

            with solara.Column():
                solara.Markdown(f"**Command:**\n```\n{cmd_preview}\n```")
                with solara.Row():
                    solara.Button(
                        "Launch",
                        on_click=launch,
                        disabled=bool(runner_pid.value),
                        color="primary",
                    )
                    solara.Button(
                        "Stop",
                        on_click=stop,
                        disabled=not runner_pid.value,
                        color="error",
                    )
                if runner_output.value:
                    tail = runner_output.value.split("\n")[-100:]
                    solara.Markdown(f"```\n{chr(10).join(tail)}\n```")


# ---------------------------------------------------------------------------
# Standalone page
# ---------------------------------------------------------------------------


@solara.component
def Page():
    """Top-level page for ``solara run pawn.dashboard.sol``."""
    solara.Title("PAWN Dashboard")
    solara.use_effect(lambda: setattr(solara.lab.theme, "dark", True), [])

    tab, set_tab = solara.use_state(0)

    solara.Markdown("# PAWN Dashboard")
    with solara.Row(style={"gap": "4px", "margin-bottom": "16px"}):
        solara.Button(
            "Training", on_click=lambda: set_tab(0),
            color="primary" if tab == 0 else None,
            text=tab != 0,
        )
        solara.Button(
            "Runner", on_click=lambda: set_tab(1),
            color="primary" if tab == 1 else None,
            text=tab != 1,
        )

    if tab == 0:
        Dashboard()
    else:
        Runner()
