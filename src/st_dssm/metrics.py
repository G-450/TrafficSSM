"""Evaluation metrics for probabilistic traffic forecasting.

All metric functions operate on numpy arrays and are model-independent.
Metrics are defined per the project's research protocol
(docs/RESEARCH_AND_EXPERIMENTS.md) and ADR-0008.

Functions
---------
mae, rmse : point-forecast accuracy
gaussian_nll, gaussian_crps : distributional quality (proper scores)
picp, mpiw : prediction-interval diagnostics
prediction_interval : compute interval bounds from Gaussian parameters
inverse_transform_predictions : convert normalized predictions to raw traffic units
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class MetricError(Exception):
    """Raised when metric inputs violate requirements."""


def _validate_arrays(*arrays: np.ndarray) -> None:
    """Reject non-finite values in any input array."""
    for i, arr in enumerate(arrays):
        if not np.all(np.isfinite(arr)):
            raise MetricError(
                f"Array at position {i} contains NaN or Inf values. "
                "All metric inputs must be finite."
            )


def _validate_positive(arr: np.ndarray, name: str) -> None:
    """Reject non-positive values (used for sigma/scale)."""
    if np.any(arr <= 0):
        raise MetricError(f"{name} must be strictly positive everywhere.")


def _validate_shape_match(*arrays: np.ndarray) -> None:
    """Ensure all arrays share the same shape."""
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise MetricError(
            f"Shape mismatch among metric inputs: {shapes}"
        )


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : array of true values (any shape).
    y_pred : array of predicted means (same shape).

    Returns
    -------
    Scalar MAE averaged over all elements.
    """
    _validate_arrays(y_true, y_pred)
    _validate_shape_match(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : array of true values.
    y_pred : array of predicted means (same shape).

    Returns
    -------
    Scalar RMSE.
    """
    _validate_arrays(y_true, y_pred)
    _validate_shape_match(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Distributional metrics (Gaussian)
# ---------------------------------------------------------------------------

def gaussian_nll(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Negative log-likelihood under a Gaussian predictive distribution.

    NLL = 0.5 * ln(2π) + ln(σ) + 0.5 * ((y − μ) / σ)²

    Parameters
    ----------
    y_true : observed values.
    mu : predicted means (same shape).
    sigma : predicted standard deviations (same shape, strictly positive).

    Returns
    -------
    Scalar mean NLL averaged over all elements.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")

    nll = (
        0.5 * np.log(2.0 * np.pi)
        + np.log(sigma)
        + 0.5 * ((y_true - mu) / sigma) ** 2
    )
    return float(np.mean(nll))


def gaussian_crps(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Closed-form CRPS for Gaussian predictive distribution.

    CRPS(N(μ,σ²), y) = σ · [ z·(2Φ(z)−1) + 2φ(z) − 1/√π ]

    where z = (y − μ) / σ, Φ is the standard-normal CDF,
    and φ is the standard-normal PDF.

    Parameters
    ----------
    y_true : observed values.
    mu : predicted means (same shape).
    sigma : predicted standard deviations (same shape, strictly positive).

    Returns
    -------
    Scalar mean CRPS averaged over all elements.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")

    z = (y_true - mu) / sigma
    crps_per_element = sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_per_element))


# ---------------------------------------------------------------------------
# Prediction-interval metrics
# ---------------------------------------------------------------------------

def prediction_interval(
    mu: np.ndarray,
    sigma: np.ndarray,
    nominal: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric Gaussian prediction-interval bounds.

    Parameters
    ----------
    mu : predicted means.
    sigma : predicted standard deviations (strictly positive).
    nominal : nominal coverage probability (default 0.95).

    Returns
    -------
    (lower, upper) arrays with the same shape as mu.
    """
    _validate_arrays(mu, sigma)
    _validate_shape_match(mu, sigma)
    _validate_positive(sigma, "sigma")
    if not 0.0 < nominal < 1.0:
        raise MetricError(f"nominal must be in (0, 1), got {nominal}")

    z = norm.ppf(1.0 - (1.0 - nominal) / 2.0)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return lower, upper


def picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Prediction Interval Coverage Probability.

    Fraction of true values falling within [lower, upper].

    Parameters
    ----------
    y_true : observed values.
    lower : lower interval bounds (same shape).
    upper : upper interval bounds (same shape).

    Returns
    -------
    Scalar coverage in [0, 1].
    """
    _validate_arrays(y_true, lower, upper)
    _validate_shape_match(y_true, lower, upper)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width.

    Parameters
    ----------
    lower : lower interval bounds.
    upper : upper interval bounds (same shape).

    Returns
    -------
    Scalar mean width.
    """
    _validate_arrays(lower, upper)
    _validate_shape_match(lower, upper)
    widths = upper - lower
    if np.any(widths < 0):
        raise MetricError("upper must be >= lower everywhere.")
    return float(np.mean(widths))


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------

def inverse_transform_predictions(
    mu: np.ndarray,
    sigma: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized Gaussian predictions to raw traffic units.

    Since X_norm = (X − mean) / std, we have:
        mu_raw   = mu_norm   * std + mean
        sigma_raw = sigma_norm * std        (scale transforms linearly)

    Parameters
    ----------
    mu : normalized predicted means, shape [..., N, 1] or [..., N].
    sigma : normalized predicted std devs (same shape, strictly positive).
    scaler_mean : per-sensor training means, shape [N].
    scaler_std : per-sensor training std devs, shape [N].

    Returns
    -------
    (mu_raw, sigma_raw) in original traffic units (e.g. mph).
    """
    _validate_arrays(mu, sigma, scaler_mean, scaler_std)
    _validate_positive(sigma, "sigma")
    _validate_positive(scaler_std, "scaler_std")

    # Broadcast scaler params to match prediction shape
    if mu.ndim > 1 and mu.shape[-1] == 1:
        # Shape [..., N, 1]
        s_mean = scaler_mean.reshape(*([1] * (mu.ndim - 2)), -1, 1)
        s_std = scaler_std.reshape(*([1] * (mu.ndim - 2)), -1, 1)
    elif mu.ndim >= 2:
        # Shape [..., N]
        s_mean = scaler_mean.reshape(*([1] * (mu.ndim - 1)), -1)
        s_std = scaler_std.reshape(*([1] * (mu.ndim - 1)), -1)
    else:
        s_mean = scaler_mean
        s_std = scaler_std

    mu_raw = mu * s_std + s_mean
    sigma_raw = sigma * s_std
    return mu_raw, sigma_raw
