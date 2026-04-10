"""Tests for pawn.dashboard.charts — plotly chart generation.

All charts should accept empty records without crashing.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def charts():
    """Lazy-import the module to surface import errors clearly per test module."""
    import pawn.dashboard.charts as m
    return m


# =====================================================================
# Module-level constants
# =====================================================================


class TestModuleContents:
    def test_colors_dict(self, charts):
        assert isinstance(charts.COLORS, dict)
        assert "blue" in charts.COLORS
        assert "red" in charts.COLORS

    def test_layer_colors_list(self, charts):
        assert len(charts.LAYER_COLORS) == 8
        # Hex-ish color strings
        for c in charts.LAYER_COLORS:
            assert isinstance(c, str)
            assert c.startswith("#")

    def test_layer_color_helper_returns_hex(self):
        from pawn.dashboard.theme import layer_color
        c = layer_color(0)
        assert c.startswith("#")
        assert len(c) == 7


# =====================================================================
# make_chart / _melt
# =====================================================================


class TestMakeChart:
    def test_empty_records(self, charts):
        fig = charts.make_chart([], "step", [("loss", "Loss", "#123456")], title="Empty")
        assert fig is not None
        # Verify serialization
        d = fig.to_dict()
        assert isinstance(d, dict)

    def test_basic_chart(self, charts):
        records = [
            {"step": 0, "loss": 1.0},
            {"step": 1, "loss": 0.9},
            {"step": 2, "loss": 0.8},
        ]
        fig = charts.make_chart(records, "step", [("loss", "Loss", "#ff0000")], title="T")
        d = fig.to_dict()
        assert "data" in d
        assert len(d["data"]) == 1

    def test_y_log(self, charts):
        records = [{"step": 0, "loss": 0.5}, {"step": 1, "loss": 0.1}]
        fig = charts.make_chart(
            records, "step", [("loss", "Loss", "#ff0000")], y_log=True,
        )
        layout = fig.to_dict()["layout"]
        assert layout.get("yaxis", {}).get("type") == "log"

    def test_missing_y_column_returns_empty_chart(self, charts):
        records = [{"step": 0}, {"step": 1}]
        fig = charts.make_chart(records, "step", [("loss", "Loss", "#f00")])
        # No loss column → empty
        assert fig is not None

    def test_missing_x_key_returns_empty_chart(self, charts):
        records = [{"loss": 1.0}, {"loss": 0.5}]
        fig = charts.make_chart(records, "step", [("loss", "Loss", "#f00")])
        assert fig is not None


# =====================================================================
# Domain-specific charts (smoke + empty input)
# =====================================================================


EMPTY = []
SAMPLE_PAWN_TRAIN = [
    {"step": 0, "train/loss": 2.0, "train/accuracy": 0.1, "lr": 1e-4, "step_time": 0.1,
     "grad_norm": 1.5, "mem/gpu_peak_gb": 2.0, "mem/gpu_reserved_gb": 2.5, "mem/gpu_current_gb": 1.8},
    {"step": 100, "train/loss": 1.5, "train/accuracy": 0.3, "lr": 3e-4, "step_time": 0.12,
     "grad_norm": 1.2, "mem/gpu_peak_gb": 2.2, "mem/gpu_reserved_gb": 2.5, "mem/gpu_current_gb": 2.0},
]
SAMPLE_ADAPTER_TRAIN = [
    {"step": 0, "train_loss": 2.0, "val_loss": 2.1, "train_top1": 0.1, "val_top1": 0.09,
     "val_top5": 0.3, "lr": 1e-4, "epoch_time_s": 30.0, "grad_norm": 1.0},
    {"step": 1, "train_loss": 1.5, "val_loss": 1.7, "train_top1": 0.2, "val_top1": 0.18,
     "val_top5": 0.45, "lr": 3e-4, "epoch_time_s": 28.0, "grad_norm": 0.8},
]
SAMPLE_VAL = [
    {"step": 100, "val/loss": 2.0, "val/accuracy": 0.2, "val/top5_accuracy": 0.4,
     "val/legal_move_rate": 0.95, "val/perplexity": 7.4},
    {"step": 200, "val/loss": 1.7, "val/accuracy": 0.25, "val/top5_accuracy": 0.48,
     "val/legal_move_rate": 0.97, "val/perplexity": 5.5},
]


class TestLossChart:
    def test_empty(self, charts):
        fig = charts.loss_chart(EMPTY, "step", "pawn")
        assert fig is not None

    def test_pawn(self, charts):
        fig = charts.loss_chart(SAMPLE_PAWN_TRAIN, "step", "pawn")
        assert fig is not None
        # Titles are wrapped in <b>...</b> for bold rendering.
        assert "Loss" in fig.to_dict()["layout"]["title"]["text"]

    def test_adapter(self, charts):
        fig = charts.loss_chart(SAMPLE_ADAPTER_TRAIN, "step", "lora")
        assert fig is not None

    def test_bc(self, charts):
        fig = charts.loss_chart(SAMPLE_ADAPTER_TRAIN, "step", "bc")
        assert fig is not None


class TestAccuracyChart:
    def test_empty(self, charts):
        fig = charts.accuracy_chart(EMPTY, "step", "pawn")
        assert fig is not None

    def test_pawn(self, charts):
        fig = charts.accuracy_chart(SAMPLE_PAWN_TRAIN, "step", "pawn")
        assert fig is not None

    def test_adapter_shows_top1_top5(self, charts):
        fig = charts.accuracy_chart(SAMPLE_ADAPTER_TRAIN, "step", "lora")
        d = fig.to_dict()
        # At least one trace
        assert len(d["data"]) >= 1


class TestLrChart:
    def test_empty(self, charts):
        fig = charts.lr_chart(EMPTY, EMPTY, "step")
        assert fig is not None

    def test_basic(self, charts):
        fig = charts.lr_chart(SAMPLE_PAWN_TRAIN, [], "step")
        assert fig is not None

    def test_hybrid_with_dual_lr(self, charts):
        records = [
            {"step": 0, "lr_lora": 1e-3, "lr_film": 1e-4},
            {"step": 1, "lr_lora": 5e-4, "lr_film": 5e-5},
        ]
        fig = charts.lr_chart(records, [], "step")
        d = fig.to_dict()
        assert len(d["data"]) >= 1


class TestGradChart:
    def test_empty(self, charts):
        fig = charts.grad_chart(EMPTY, "step")
        assert fig is not None

    def test_basic(self, charts):
        fig = charts.grad_chart(SAMPLE_PAWN_TRAIN, "step")
        assert fig is not None


class TestGpuChart:
    def test_empty(self, charts):
        fig = charts.gpu_chart(EMPTY, "step")
        assert fig is not None

    def test_basic(self, charts):
        fig = charts.gpu_chart(SAMPLE_PAWN_TRAIN, "step")
        assert fig is not None


class TestTimeChart:
    def test_empty_pawn(self, charts):
        fig = charts.time_chart(EMPTY, "step", "pawn")
        assert fig is not None

    def test_empty_adapter(self, charts):
        fig = charts.time_chart(EMPTY, "step", "lora")
        assert fig is not None

    def test_pawn_shows_step_time(self, charts):
        fig = charts.time_chart(SAMPLE_PAWN_TRAIN, "step", "pawn")
        layout = fig.to_dict()["layout"]
        assert "Step Time" in layout["title"]["text"]

    def test_adapter_shows_epoch_time(self, charts):
        fig = charts.time_chart(SAMPLE_ADAPTER_TRAIN, "step", "lora")
        layout = fig.to_dict()["layout"]
        assert "Epoch Time" in layout["title"]["text"]


class TestFilmCharts:
    def test_film_weight_empty(self, charts):
        fig = charts.film_weight_chart(EMPTY, "step")
        assert fig is not None

    def test_film_beta_empty(self, charts):
        fig = charts.film_beta_chart(EMPTY, "step")
        assert fig is not None

    def test_film_weight_with_data(self, charts):
        records = [
            {"step": 0, **{f"film/hidden_{i}/gamma_dev": 0.1 for i in range(8)},
             "film/output/gamma_dev": 0.2},
            {"step": 1, **{f"film/hidden_{i}/gamma_dev": 0.05 for i in range(8)},
             "film/output/gamma_dev": 0.15},
        ]
        fig = charts.film_weight_chart(records, "step")
        assert fig is not None


class TestLoraCharts:
    def test_lora_layer_empty(self, charts):
        fig = charts.lora_layer_chart(EMPTY, "step")
        assert fig is not None

    def test_lora_proj_empty(self, charts):
        fig = charts.lora_proj_chart(EMPTY, "step")
        assert fig is not None

    def test_lora_detail_empty(self, charts):
        fig = charts.lora_detail_chart(EMPTY, "step", "wq")
        assert fig is not None

    def test_lora_layer_with_data(self, charts):
        records = [
            {"step": 0, **{f"lora/layer{i}.wq.B": 0.5 for i in range(8)}},
            {"step": 1, **{f"lora/layer{i}.wq.B": 0.6 for i in range(8)}},
        ]
        fig = charts.lora_layer_chart(records, "step")
        assert fig is not None

    def test_lora_proj_with_data(self, charts):
        records = [
            {"step": 0, **{f"lora/layer0.{p}.B": 0.5 for p in ["wq", "wv", "wk"]}},
        ]
        fig = charts.lora_proj_chart(records, "step")
        assert fig is not None

    def test_detect_lora_projs(self, charts):
        records = [
            {"lora/layer0.wq.B": 0.5, "lora/layer0.wv.B": 0.5},
            {"lora/layer1.wk.B": 0.5},
        ]
        projs = charts._detect_lora_projs(records)
        assert set(projs) == {"wq", "wv", "wk"}

    def test_detect_lora_projs_empty(self, charts):
        assert charts._detect_lora_projs([]) == []


class TestSparseCharts:
    def test_sparse_delta_empty(self, charts):
        fig = charts.sparse_delta_chart(EMPTY, "step")
        assert fig is not None

    def test_sparse_delta_with_data(self, charts):
        records = [
            {"step": 0, **{f"sparse/layer{i}.wq.delta": 0.1 for i in range(8)}},
        ]
        fig = charts.sparse_delta_chart(records, "step")
        assert fig is not None


class TestBottleneckCharts:
    def test_bottleneck_up_empty(self, charts):
        fig = charts.bottleneck_up_chart(EMPTY, "step")
        assert fig is not None

    def test_bottleneck_up_with_data(self, charts):
        records = [
            {"step": 0, **{f"adapter/layer{i}.attn.up": 0.3 for i in range(8)},
             **{f"adapter/layer{i}.ffn.up": 0.4 for i in range(8)}},
        ]
        fig = charts.bottleneck_up_chart(records, "step")
        assert fig is not None


class TestValCharts:
    def test_val_loss_empty_pawn(self, charts):
        fig = charts.val_loss_chart(EMPTY, "step", "pawn")
        assert fig is not None

    def test_val_loss_empty_adapter(self, charts):
        fig = charts.val_loss_chart(EMPTY, "step", "lora")
        assert fig is not None

    def test_val_loss_pawn_with_data(self, charts):
        fig = charts.val_loss_chart(SAMPLE_VAL, "step", "pawn")
        assert fig is not None

    def test_val_accuracy_empty(self, charts):
        fig = charts.val_accuracy_chart(EMPTY, "step", "pawn")
        assert fig is not None

    def test_val_accuracy_pawn_with_data(self, charts):
        fig = charts.val_accuracy_chart(SAMPLE_VAL, "step", "pawn")
        assert fig is not None


class TestPatienceChart:
    def test_empty(self, charts):
        fig = charts.patience_chart([], x_key="step", patience_limit=10)
        assert fig is not None

    def test_with_improvements(self, charts):
        val = [
            {"step": 10, "val/loss": 2.0},
            {"step": 20, "val/loss": 1.8},  # improvement → 0
            {"step": 30, "val/loss": 1.9},  # +1
            {"step": 40, "val/loss": 2.0},  # +2
            {"step": 50, "val/loss": 1.5},  # improvement → 0
        ]
        fig = charts.patience_chart(val, x_key="step", patience_limit=5)
        d = fig.to_dict()
        assert len(d["data"]) >= 1
        # Patience counter trace y-values should start at 0
        y = d["data"][0]["y"]
        assert y[0] == 0

    def test_patience_counter_increments(self, charts):
        val = [
            {"step": 10, "val/loss": 1.0},  # first → 0
            {"step": 20, "val/loss": 1.1},  # no improvement → 1
            {"step": 30, "val/loss": 1.2},  # no improvement → 2
        ]
        fig = charts.patience_chart(val, x_key="step", patience_limit=5)
        d = fig.to_dict()
        y = list(d["data"][0]["y"])
        assert y == [0, 1, 2]

    def test_missing_val_loss_skipped(self, charts):
        val = [
            {"step": 10, "val/loss": 2.0},
            {"step": 20},  # no val/loss, skipped
            {"step": 30, "val/loss": 1.9},
        ]
        fig = charts.patience_chart(val, x_key="step", patience_limit=5)
        d = fig.to_dict()
        assert len(d["data"]) >= 1
