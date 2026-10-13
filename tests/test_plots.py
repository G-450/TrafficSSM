"""Tests for evaluation plotting utilities (Phase 4).

Validates that all plot functions generate valid image files, execute headlessly,
respect physical units, and handle edge cases gracefully.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from st_dssm.plots import (
    plot_calibration_curve,
    plot_horizon_curves,
    plot_horizon_metrics,
    plot_interval_width_vs_horizon,
    plot_metric_comparison,
    plot_prediction_intervals,
)


class TestPlots:
    def test_plot_horizon_metrics_creates_file(self, tmp_path):
        out = str(tmp_path / "horizon_mae.png")
        horizons = list(range(1, 13))
        values = [2.0 + float(h) * 0.1 for h in horizons]

        result = plot_horizon_metrics(
            horizons, values, "MAE", out, unit="mph", title="Test MAE by Horizon"
        )
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert len(plt.get_fignums()) == 0  # Figure was closed

    def test_plot_prediction_intervals_creates_file(self, tmp_path):
        out = str(tmp_path / "pi_sensor.png")
        T = 50
        y_true = 50.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, T))
        mu = y_true + 1.0
        lower = mu - 5.0
        upper = mu + 5.0

        result = plot_prediction_intervals(
            y_true, mu, lower, upper, out, sensor_idx="Sensor-42", unit="mph"
        )
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert len(plt.get_fignums()) == 0

    def test_plot_calibration_curve_creates_file(self, tmp_path):
        out = str(tmp_path / "reliability.png")
        nominal = [0.50, 0.80, 0.90, 0.95, 0.99]
        empirical = [0.52, 0.79, 0.88, 0.94, 0.98]

        result = plot_calibration_curve(nominal, empirical, out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert len(plt.get_fignums()) == 0

    def test_plot_interval_width_vs_horizon_creates_file(self, tmp_path):
        out = str(tmp_path / "width_vs_horizon.png")
        horizons = list(range(1, 13))
        widths = [4.0 + float(h) * 0.3 for h in horizons]

        result = plot_interval_width_vs_horizon(horizons, widths, out, unit="mph")
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert len(plt.get_fignums()) == 0

    def test_plot_metric_comparison_creates_file(self, tmp_path):
        out = str(tmp_path / "model_comp.png")
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
        assert len(plt.get_fignums()) == 0

    def test_plot_horizon_curves_creates_file(self, tmp_path):
        out = str(tmp_path / "horizon_curves.png")
        minutes = [5 * h for h in range(1, 13)]
        series = {
            "Deterministic ST-GCN": ([1.4 + 0.1 * i for i in range(12)], [0.0] * 12),
            "ST-DSSM": ([1.5 + 0.1 * i for i in range(12)], [0.01 * i for i in range(12)]),
        }

        result = plot_horizon_curves(
            minutes,
            series,
            "MAE",
            out,
            unit="mph",
            title="MAE by horizon",
        )
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert len(plt.get_fignums()) == 0

    def test_plot_horizon_curves_draws_reference_level(self, tmp_path):
        out = str(tmp_path / "picp_curves.png")
        minutes = [5, 10, 15]
        series = {"ST-DSSM": ([0.96, 0.96, 0.97], [0.001, 0.001, 0.002])}

        plot_horizon_curves(
            minutes,
            series,
            "PICP",
            out,
            unit="ratio",
            reference_level=0.95,
            reference_label="nominal 95%",
        )
        assert os.path.getsize(out) > 0
        assert len(plt.get_fignums()) == 0

    def test_plot_horizon_curves_rejects_mismatched_series(self, tmp_path):
        out = str(tmp_path / "bad_curves.png")
        with pytest.raises(ValueError, match="expected 3"):
            plot_horizon_curves([5, 10, 15], {"m": ([1.0, 2.0], [0.0, 0.0])}, "MAE", out)

    def test_plot_horizon_curves_rejects_empty_series(self, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            plot_horizon_curves([5], {}, "MAE", str(tmp_path / "empty.png"))

    def test_length_mismatch_raises(self, tmp_path):
        out = str(tmp_path / "bad.png")
        with pytest.raises(ValueError, match="same length"):
            plot_horizon_metrics([1, 2], [1.0], "MAE", out)
