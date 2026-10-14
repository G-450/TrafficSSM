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
plot_horizon_curves : multi-model horizon curves with across-seed spread bands.
plot_reliability_comparison : reliability diagram overlaying raw and calibrated arms.
plot_pit_histogram : probability-integral-transform histogram against the uniform.
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


# ---------------------------------------------------------------------------
# Multi-model horizon curves with across-seed spread
# ---------------------------------------------------------------------------


def plot_horizon_curves(
    horizon_minutes: list[int],
    series: dict[str, tuple[list[float], list[float]]],
    metric_name: str,
    output_path: str,
    *,
    unit: str = "mph",
    title: str | None = None,
    reference_level: float | None = None,
    reference_label: str | None = None,
) -> str:
    """Compare models across forecast horizons, shading the across-seed spread.

    Parameters
    ----------
    horizon_minutes : horizon of each point in minutes (x-axis).
    series : ``{model_name: (means, stds)}``; ``stds`` may be all-zero for a
        single-seed or deterministic model, in which case no band is drawn.
    metric_name : metric label on the y-axis.
    output_path : file path to save the figure.
    unit : unit string appended to the y-axis label.
    title : optional title.
    reference_level : optional horizontal reference line (e.g. nominal coverage).
    reference_label : legend label for the reference line.

    Returns
    -------
    The output file path.
    """
    if not series:
        raise ValueError("series must not be empty.")

    n = len(horizon_minutes)
    for name, (means, stds) in series.items():
        if len(means) != n or len(stds) != n:
            raise ValueError(
                f"Model '{name}' has {len(means)} means and {len(stds)} stds but expected {n} of each."
            )

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.asarray(horizon_minutes, dtype=float)
    for i, (name, (means, stds)) in enumerate(series.items()):
        color = _PALETTE[i % len(_PALETTE)]
        mean_arr = np.asarray(means, dtype=float)
        std_arr = np.asarray(stds, dtype=float)
        ax.plot(x, mean_arr, marker="o", markersize=4, color=color, label=name, linewidth=1.8)
        if np.any(std_arr > 0):
            ax.fill_between(
                x,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

    if reference_level is not None:
        ax.axhline(
            reference_level,
            color="#e0e0e0",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=reference_label or f"reference = {reference_level:g}",
        )

    ylabel = metric_name if not unit else f"{metric_name} ({unit})"
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{metric_name} across forecast horizons", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(loc="best", framealpha=0.7)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Reliability comparison (raw vs calibrated)
# ---------------------------------------------------------------------------


def plot_reliability_comparison(
    nominal_levels: list[float],
    arms: dict[str, list[float]],
    output_path: str,
    *,
    title: str | None = None,
    gap_panel: bool = True,
) -> str:
    """Reliability diagram overlaying several arms against the ideal diagonal.

    Phase 13 uses this to read the raw and temperature-scaled forecasts side by
    side (ADR-0008 requires both to be reported).

    Parameters
    ----------
    nominal_levels : nominal coverage levels (x-axis), ascending.
    arms : ``{label: empirical_coverages}``, one entry per curve.
    output_path : file path to save the figure.
    title : optional title.
    gap_panel : when True, add a lower panel of signed coverage gaps, which
        resolves differences the diagonal plot compresses near 1.0.

    Returns
    -------
    The output file path.
    """
    if not arms:
        raise ValueError("arms must not be empty.")
    n = len(nominal_levels)
    for label, coverages in arms.items():
        if len(coverages) != n:
            raise ValueError(
                f"Arm '{label}' has {len(coverages)} coverages but expected {n}."
            )

    _apply_style()
    if gap_panel:
        fig, (ax, ax_gap) = plt.subplots(
            2, 1, figsize=(7.5, 9), height_ratios=[2.2, 1], sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax_gap = None

    x = np.asarray(nominal_levels, dtype=float)
    ax.plot([0, 1], [0, 1], "--", color="#a0a0a0", linewidth=1.5, label="Ideal calibration")

    for i, (label, coverages) in enumerate(arms.items()):
        color = _PALETTE[i % len(_PALETTE)]
        y = np.asarray(coverages, dtype=float)
        ax.plot(x, y, "o-", color=color, linewidth=2.0, markersize=6, label=label)
        if ax_gap is not None:
            ax_gap.plot(x, y - x, "o-", color=color, linewidth=1.8, markersize=5, label=label)

    lo = float(min(x.min(), min(min(v) for v in arms.values())))
    ax.set_xlim([max(0.0, lo - 0.05), 1.02])
    ax.set_ylim([max(0.0, lo - 0.05), 1.02])
    ax.set_ylabel("Empirical coverage (PICP)")
    ax.set_title(title or "Uncertainty calibration (reliability diagram)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.7)

    if ax_gap is not None:
        ax_gap.axhline(0.0, color="#a0a0a0", linestyle="--", linewidth=1.5)
        ax_gap.set_xlabel("Nominal coverage probability")
        ax_gap.set_ylabel("Coverage gap\n(empirical - nominal)")
    else:
        ax.set_xlabel("Nominal coverage probability")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# PIT histogram
# ---------------------------------------------------------------------------


def plot_pit_histogram(
    densities: list[float],
    output_path: str,
    *,
    edges: list[float] | None = None,
    title: str | None = None,
    label: str | None = None,
) -> str:
    """Histogram of probability-integral-transform values against the uniform.

    Bars are normalized so a calibrated forecast sits on the 1.0 reference
    line; a central peak means over-dispersion and a U shape under-dispersion.

    Parameters
    ----------
    densities : per-bin densities normalized to a calibrated level of 1.0.
    output_path : file path to save the figure.
    edges : optional ``len(densities) + 1`` bin edges; defaults to equal-width
        bins over [0, 1].
    title : optional title.
    label : optional legend label for the bars.

    Returns
    -------
    The output file path.
    """
    if not densities:
        raise ValueError("densities must not be empty.")
    if edges is not None and len(edges) != len(densities) + 1:
        raise ValueError(
            f"edges must have {len(densities) + 1} entries, got {len(edges)}."
        )

    bin_edges = (
        np.asarray(edges, dtype=float)
        if edges is not None
        else np.linspace(0.0, 1.0, len(densities) + 1)
    )
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(
        centers,
        np.asarray(densities, dtype=float),
        width=widths * 0.92,
        color=_PALETTE[0],
        alpha=0.85,
        label=label or "Observed PIT density",
    )
    ax.axhline(1.0, color="#a0a0a0", linestyle="--", linewidth=1.5, label="Calibrated (uniform)")

    ax.set_xlabel(r"PIT value  $\Phi((y - \hat{\mu}) / \hat{\sigma})$")
    ax.set_ylabel("Density relative to uniform")
    ax.set_title(title or "Probability integral transform", fontsize=14, fontweight="bold")
    ax.set_xlim([bin_edges[0], bin_edges[-1]])
    ax.legend(loc="best", framealpha=0.7)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
