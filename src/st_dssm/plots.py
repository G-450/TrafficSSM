"""Evaluation plotting utilities for traffic forecasting.

All plots save to files and return the output path.
No interactive ``plt.show()`` calls. Uses headless Agg backend.

Functions
---------
plot_horizon_metrics : per-horizon bar or line chart with minute annotations.
plot_prediction_intervals : time-series with observed speeds, mean, and confidence band.
plot_calibration_curve : reliability diagram comparing empirical vs nominal coverage.
plot_interval_width_vs_horizon : interval width across forecast horizons in minutes.
plot_metric_comparison : grouped bar chart comparing multiple models or conditions.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Visual Style and Color Palette
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "axes.grid": True,
    "grid.color": "#2a3a5e",
    "grid.alpha": 0.5,
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "font.size": 11,
}

_PALETTE = [
    "#00d2ff",  # cyan
    "#ff6b6b",  # coral
    "#7bed9f",  # mint
    "#ffa502",  # amber
    "#a29bfe",  # lavender
    "#fd79a8",  # pink
]


def _apply_style() -> None:
    """Apply the project's consistent plot style."""
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# Per-horizon metric chart
# ---------------------------------------------------------------------------


def plot_horizon_metrics(
    horizons: list[int],
    values: list[float],
    metric_name: str,
    output_path: str,
    *,
    unit: str = "mph",
    title: str | None = None,
    minutes_per_step: int = 5,
) -> str:
    """Bar chart showing a metric across forecast horizons (in steps and minutes).

    Parameters
    ----------
    horizons : list of horizon step indices (e.g. [1, 2, ..., 12]).
    values : metric values corresponding to each horizon.
    metric_name : name for the y-axis label (e.g. 'MAE').
    output_path : file path to save the figure.
    unit : unit string appended to the y-axis label (default 'mph').
    title : optional figure title.
    minutes_per_step : time resolution in minutes per step (default 5).

    Returns
    -------
    The output file path.
    """
    if len(horizons) != len(values):
        raise ValueError("horizons and values must have the same length.")

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    x_labels = [f"H{h} ({h * minutes_per_step}m)" for h in horizons]
    bars = ax.bar(
        range(len(horizons)),
        values,
        color=_PALETTE[0],
        edgecolor=_PALETTE[0],
        alpha=0.85,
        width=0.65,
    )

    ylabel = metric_name if not unit else f"{metric_name} ({unit})"
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{metric_name} across Forecast Horizons", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Prediction intervals time-series
# ---------------------------------------------------------------------------


def plot_prediction_intervals(
    y_true: np.ndarray,
    mu: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: str,
    *,
    sensor_idx: int | str = 0,
    horizon_range: tuple[int, int] | None = None,
    unit: str = "mph",
    title: str | None = None,
) -> str:
    """Time-series plot showing ground-truth traffic speed, predicted mean, and confidence band.

    Parameters
    ----------
    y_true : true values [T] (1-D array or sliceable).
    mu : predicted means (same shape).
    lower : lower interval bounds (same shape).
    upper : upper interval bounds (same shape).
    output_path : file path to save the figure.
    sensor_idx : sensor identifier or index for the plot title.
    horizon_range : optional (start, end) time index slice.
    unit : physical unit string (default 'mph').
    title : optional custom title.

    Returns
    -------
    The output file path.
    """
    _apply_style()

    yt = np.asarray(y_true).ravel()
    m = np.asarray(mu).ravel()
    lo = np.asarray(lower).ravel()
    hi = np.asarray(upper).ravel()

    if horizon_range is not None:
        s, e = horizon_range
        yt = yt[s:e]
        m = m[s:e]
        lo = lo[s:e]
        hi = hi[s:e]

    t = np.arange(len(yt))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        t,
        lo,
        hi,
        alpha=0.25,
        color=_PALETTE[0],
        label="95% Prediction Interval",
    )
    ax.plot(t, m, color=_PALETTE[0], linewidth=1.8, label="Predicted Mean")
    ax.plot(t, yt, color=_PALETTE[1], linewidth=1.2, alpha=0.85, label="Observed Speed")

    ax.set_xlabel("Time Step (5-min intervals)")
    ax.set_ylabel(f"Speed ({unit})")
    ax.set_title(
        title or f"Traffic Speed Forecast & Prediction Interval — Sensor {sensor_idx}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.7)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Calibration / Reliability Diagram
# ---------------------------------------------------------------------------


def plot_calibration_curve(
    nominal_levels: list[float],
    empirical_coverages: list[float],
    output_path: str,
    *,
    title: str | None = None,
) -> str:
    """Reliability diagram comparing empirical coverage to nominal confidence levels.

    Parameters
    ----------
    nominal_levels : list of nominal confidence levels (e.g. [0.5, 0.8, 0.9, 0.95, 0.99]).
    empirical_coverages : empirical coverage values corresponding to nominal levels.
    output_path : file path to save the figure.
    title : optional title.

    Returns
    -------
    The output file path.
    """
    if len(nominal_levels) != len(empirical_coverages):
        raise ValueError("nominal_levels and empirical_coverages must have the same length.")

    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot diagonal ideal reference
    ax.plot([0, 1], [0, 1], "--", color="#a0a0a0", linewidth=1.5, label="Ideal Calibration")

    # Plot empirical curve
    ax.plot(
        nominal_levels,
        empirical_coverages,
        "o-",
        color=_PALETTE[0],
        linewidth=2.0,
        markersize=7,
        label="Observed Coverage (PICP)",
    )

    ax.set_xlabel("Nominal Coverage Probability")
    ax.set_ylabel("Empirical Coverage Probability")
    ax.set_title(title or "Uncertainty Calibration (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.set_xlim([0.45, 1.02])
    ax.set_ylim([0.45, 1.02])
    ax.legend(loc="lower right", framealpha=0.7)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Interval Width vs Horizon
# ---------------------------------------------------------------------------


def plot_interval_width_vs_horizon(
    horizons: list[int],
    widths: list[float],
    output_path: str,
    *,
    unit: str = "mph",
    title: str | None = None,
    minutes_per_step: int = 5,
) -> str:
    """Plot prediction interval width (MPIW) as a function of forecast horizon.

    Parameters
    ----------
    horizons : list of horizon steps (e.g. [1, 2, ..., 12]).
    widths : mean interval widths for each horizon.
    output_path : file path to save figure.
    unit : physical unit string.
    title : optional title.
    minutes_per_step : cadence in minutes.

    Returns
    -------
    The output file path.
    """
    if len(horizons) != len(widths):
        raise ValueError("horizons and widths must have the same length.")

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    minutes = [h * minutes_per_step for h in horizons]
    ax.plot(minutes, widths, "s-", color=_PALETTE[3], linewidth=2.0, markersize=6)

    ax.set_xlabel("Forecast Horizon (minutes)")
    ax.set_ylabel(f"Mean Prediction Interval Width ({unit})")
    ax.set_title(title or "95% Prediction Interval Width vs Horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(minutes)

    for m, w in zip(minutes, widths):
        ax.text(m, w + 0.02 * max(widths), f"{w:.2f}", ha="center", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Model Comparison Grouped Bar Chart
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    results_dict: dict[str, list[float]],
    metric_name: str,
    output_path: str,
    *,
    group_labels: list[str] | None = None,
    unit: str = "mph",
    title: str | None = None,
) -> str:
    """Grouped bar chart comparing multiple models on a metric.

    Parameters
    ----------
    results_dict : ``{model_name: [value_per_group]}``.
    metric_name : metric label on y-axis.
    output_path : file path to save figure.
    group_labels : labels for each group (e.g. ['15m', '30m', '60m']).
    unit : unit string.
    title : optional title.

    Returns
    -------
    The output file path.
    """
    if not results_dict:
        raise ValueError("results_dict must not be empty.")

    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_groups = len(next(iter(results_dict.values())))

    for name, vals in results_dict.items():
        if len(vals) != n_groups:
            raise ValueError(
                f"Model '{name}' has {len(vals)} values but expected {n_groups}."
            )

    if group_labels is None:
        group_labels = [str(i + 1) for i in range(n_groups)]

    _apply_style()
    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 6))

    x = np.arange(n_groups)
    total_width = 0.8
    bar_width = total_width / n_models

    for i, (model, vals) in enumerate(results_dict.items()):
        offset = (i - n_models / 2 + 0.5) * bar_width
        color = _PALETTE[i % len(_PALETTE)]
        ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=model,
            color=color,
            edgecolor=color,
            alpha=0.85,
        )

    ylabel = metric_name if not unit else f"{metric_name} ({unit})"
    ax.set_xlabel("Horizon / Group")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{metric_name} Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(loc="upper right", framealpha=0.7)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
