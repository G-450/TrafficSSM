"""Evaluation plotting utilities.

All plots save to files and return the output path.
No interactive ``plt.show()`` calls.  Uses a consistent visual style.

Functions
---------
plot_horizon_metrics : per-horizon bar chart for a single metric.
plot_prediction_intervals : time-series with confidence bands.
plot_metric_comparison : grouped bar chart comparing models.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
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
# Per-horizon bar chart
# ---------------------------------------------------------------------------

def plot_horizon_metrics(
    horizons: list[int],
    values: list[float],
    metric_name: str,
    output_path: str,
    *,
    unit: str = "",
    title: str | None = None,
) -> str:
    """Bar chart showing a metric across forecast horizons.

    Parameters
    ----------
    horizons : list of horizon indices (e.g. [1, 2, ..., 12]).
    values : metric values corresponding to each horizon.
    metric_name : name for the y-axis label (e.g. 'MAE (mph)').
    output_path : file path to save the figure.
    unit : optional unit string appended to the y-axis label.
    title : optional figure title; defaults to metric_name.

    Returns
    -------
    The output path (for chaining).
    """
    if len(horizons) != len(values):
        raise ValueError("horizons and values must have the same length.")

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        horizons, values,
        color=_PALETTE[0], edgecolor=_PALETTE[0], alpha=0.85,
        width=0.7,
    )

    ylabel = metric_name if not unit else f"{metric_name} ({unit})"
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or metric_name, fontsize=14, fontweight="bold")
    ax.set_xticks(horizons)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Prediction-interval time-series plot
# ---------------------------------------------------------------------------

def plot_prediction_intervals(
    y_true: np.ndarray,
    mu: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: str,
    *,
    sensor_idx: int = 0,
    horizon_range: tuple[int, int] | None = None,
    title: str | None = None,
) -> str:
    """Time-series plot with prediction mean and confidence band.

    Parameters
    ----------
    y_true : true values, shape [T] or sliceable 1-D.
    mu : predicted means (same shape).
    lower : lower interval bounds (same shape).
    upper : upper interval bounds (same shape).
    output_path : file path to save the figure.
    sensor_idx : sensor index (for labelling only).
    horizon_range : optional (start, end) to slice the time axis.
    title : optional figure title.

    Returns
    -------
    The output path.
    """
    _apply_style()

    if horizon_range is not None:
        s, e = horizon_range
        y_true = y_true[s:e]
        mu = mu[s:e]
        lower = lower[s:e]
        upper = upper[s:e]

    t = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        t, lower, upper,
        alpha=0.25, color=_PALETTE[0], label="95% PI",
    )
    ax.plot(t, mu, color=_PALETTE[0], linewidth=1.5, label="Predicted mean")
    ax.plot(t, y_true, color=_PALETTE[1], linewidth=1.0, alpha=0.8, label="Observed")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(
        title or f"Prediction Intervals — Sensor {sensor_idx}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Model comparison grouped bar chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    results_dict: dict[str, list[float]],
    metric_name: str,
    output_path: str,
    *,
    group_labels: list[str] | None = None,
    unit: str = "",
    title: str | None = None,
) -> str:
    """Grouped bar chart comparing multiple models on a metric.

    Parameters
    ----------
    results_dict : ``{model_name: [value_per_group]}``.
        All value lists must have the same length.
    metric_name : y-axis label.
    output_path : file path to save the figure.
    group_labels : labels for each group (e.g. horizon indices).
    unit : optional unit string.
    title : optional figure title.

    Returns
    -------
    The output path.
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
            x + offset, vals,
            width=bar_width, label=model,
            color=color, edgecolor=color, alpha=0.85,
        )

    ylabel = metric_name if not unit else f"{metric_name} ({unit})"
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{metric_name} Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(loc="upper right", framealpha=0.7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
