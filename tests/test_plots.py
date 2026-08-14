"""Tests for evaluation plotting utilities (Phase 4).

Validates that all plot functions generate valid image files,
respect custom arguments, and handle edge cases gracefully.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from st_dssm.plots import (
    plot_horizon_metrics,
    plot_metric_comparison,
    plot_prediction_intervals,
)


class TestPlotHorizonMetrics:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "horizon.png")
        horizons = list(range(1, 13))
        values = [float(h) * 0.5 for h in horizons]

        result = plot_horizon_metrics(
            horizons, values, "MAE", out, unit="mph", title="Test Horizon MAE"
        )
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_length_mismatch_raises(self, tmp_path):
        out = str(tmp_path / "bad.png")
        with pytest.raises(ValueError, match="same length"):
            plot_horizon_metrics([1, 2], [1.0], "MAE", out)


class TestPlotPredictionIntervals:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "pi.png")
        T = 50
        y_true = np.sin(np.linspace(0, 10, T))
        mu = y_true + 0.1
        lower = mu - 0.5
        upper = mu + 0.5

        result = plot_prediction_intervals(
            y_true, mu, lower, upper, out, sensor_idx=42, title="Sensor 42 PI"
        )
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_with_horizon_range(self, tmp_path):
        out = str(tmp_path / "pi_sliced.png")
        T = 100
        y_true = np.ones(T)
        mu = np.ones(T)
        lower = np.zeros(T)
        upper = np.ones(T) * 2

        result = plot_prediction_intervals(
            y_true, mu, lower, upper, out, horizon_range=(10, 30)
        )
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


class TestPlotMetricComparison:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "comparison.png")
        results = {
            "Persistence": [2.5, 3.0, 3.5],
            "ST-GCN": [2.0, 2.3, 2.7],
            "ST-DSSM": [1.8, 2.1, 2.4],
        }

        result = plot_metric_comparison(
            results,
            "MAE",
            out,
            group_labels=["15 min", "30 min", "60 min"],
            unit="mph",
            title="Model MAE Comparison",
        )
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_empty_dict_raises(self, tmp_path):
        out = str(tmp_path / "bad.png")
        with pytest.raises(ValueError, match="must not be empty"):
            plot_metric_comparison({}, "MAE", out)

    def test_mismatched_group_lengths_raises(self, tmp_path):
        out = str(tmp_path / "bad.png")
        results = {
            "M1": [1.0, 2.0],
            "M2": [1.0, 2.0, 3.0],
        }
        with pytest.raises(ValueError, match="expected 2"):
            plot_metric_comparison(results, "MAE", out)
