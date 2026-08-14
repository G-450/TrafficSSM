"""Evaluation metrics for probabilistic traffic forecasting.

All metric functions operate on numpy arrays and are model-independent.
Metrics are defined per the project's research protocol
(docs/RESEARCH_AND_EXPERIMENTS.md), ADR-0007, and ADR-0008.

Functions
---------
mae, rmse, mape : point-forecast accuracy
gaussian_nll, gaussian_crps : distributional quality (proper scores)
picp, mpiw : prediction-interval diagnostics
prediction_interval : compute interval bounds from Gaussian parameters
calibration_curve : evaluate empirical coverage across multiple nominal levels
inverse_transform_predictions : convert normalized predictions/variance/bounds to raw traffic units
evaluate_metrics_by_horizon : evaluate point & probabilistic metrics per horizon and aggregate
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
        if not isinstance(arr, np.ndarray):
            raise MetricError(f"Expected numpy ndarray at position {i}, got {type(arr)}")
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
        raise MetricError(f"Shape mismatch among metric inputs: {shapes}")


def _apply_mask(
    arr: np.ndarray, mask: np.ndarray | None
) -> np.ndarray:
    """Filter an array by a boolean or binary mask, flattening to 1D."""
    if mask is None:
        return arr.ravel().astype(np.float64)

    if mask.shape != arr.shape:
        raise MetricError(
            f"Mask shape {mask.shape} does not match array shape {arr.shape}."
        )

    bool_mask = mask.astype(bool)
    if not np.any(bool_mask):
        raise MetricError(
            "Evaluation mask is completely empty (no valid observations to evaluate)."
        )

    return arr[bool_mask].astype(np.float64)


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean Absolute Error in physical units (mph).

    Parameters
    ----------
    y_true : array of true values (any shape).
    y_pred : array of predicted means (same shape).
    mask : optional boolean or binary observation mask (same shape).

    Returns
    -------
    Scalar MAE averaged over valid elements.
    """
    _validate_arrays(y_true, y_pred)
    _validate_shape_match(y_true, y_pred)

    yt = _apply_mask(y_true, mask)
    yp = _apply_mask(y_pred, mask)
    return float(np.mean(np.abs(yt - yp)))


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Root Mean Squared Error in physical units (mph).

    Parameters
    ----------
    y_true : array of true values.
    y_pred : array of predicted means (same shape).
    mask : optional boolean or binary observation mask (same shape).

    Returns
    -------
    Scalar RMSE averaged over valid elements.
    """
    _validate_arrays(y_true, y_pred)
    _validate_shape_match(y_true, y_pred)

    yt = _apply_mask(y_true, mask)
    yp = _apply_mask(y_pred, mask)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 1.0,
) -> float:
    """Mean Absolute Percentage Error (percentage, 0-100%).

    Following standard traffic forecasting protocol, observations where
    ``y_true < threshold`` (default 1.0 mph) are excluded to prevent division
    by zero or extreme percentage inflation from near-zero speeds.

    Parameters
    ----------
    y_true : array of true values.
    y_pred : array of predicted means (same shape).
    mask : optional observation mask (same shape).
    threshold : minimum speed threshold for denominator (default 1.0).

    Returns
    -------
    Scalar MAPE as a percentage (e.g. 5.2 for 5.2%).
    """
    _validate_arrays(y_true, y_pred)
    _validate_shape_match(y_true, y_pred)

    combined_mask = (y_true >= threshold)
    if mask is not None:
        if mask.shape != y_true.shape:
            raise MetricError(
                f"Mask shape {mask.shape} does not match array shape {y_true.shape}."
            )
        combined_mask = combined_mask & mask.astype(bool)

    if not np.any(combined_mask):
        raise MetricError(
            f"No valid targets with y_true >= {threshold} found for MAPE calculation."
        )

    yt = y_true[combined_mask].astype(np.float64)
    yp = y_pred[combined_mask].astype(np.float64)
    return float(np.mean(np.abs((yt - yp) / yt)) * 100.0)


# ---------------------------------------------------------------------------
# Distributional metrics (Gaussian)
# ---------------------------------------------------------------------------


def gaussian_nll(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Negative log-likelihood under a Gaussian predictive distribution.

    NLL = 0.5 * ln(2π) + ln(σ) + 0.5 * ((y − μ) / σ)²

    Parameters
    ----------
    y_true : observed values.
    mu : predicted means (same shape).
    sigma : predicted standard deviations (same shape, strictly positive).
    mask : optional observation mask (same shape).

    Returns
    -------
    Scalar mean NLL averaged over valid elements.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")

    yt = _apply_mask(y_true, mask)
    m = _apply_mask(mu, mask)
    s = _apply_mask(sigma, mask)

    nll = 0.5 * np.log(2.0 * np.pi) + np.log(s) + 0.5 * ((yt - m) / s) ** 2
    return float(np.mean(nll))


def gaussian_crps(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray | None = None,
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
    mask : optional observation mask (same shape).

    Returns
    -------
    Scalar mean CRPS averaged over valid elements.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")

    yt = _apply_mask(y_true, mask)
    m = _apply_mask(mu, mask)
    s = _apply_mask(sigma, mask)

    z = (yt - m) / s
    crps_per_element = s * (
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

    z = float(norm.ppf(1.0 - (1.0 - nominal) / 2.0))
    lower = mu - z * sigma
    upper = mu + z * sigma
    return lower, upper


def picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Prediction Interval Coverage Probability.

    Fraction of true values falling within [lower, upper].

    Parameters
    ----------
    y_true : observed values.
    lower : lower interval bounds (same shape).
    upper : upper interval bounds (same shape).
    mask : optional observation mask (same shape).

    Returns
    -------
    Scalar coverage in [0, 1].
    """
    _validate_arrays(y_true, lower, upper)
    _validate_shape_match(y_true, lower, upper)

    yt = _apply_mask(y_true, mask)
    lo = _apply_mask(lower, mask)
    hi = _apply_mask(upper, mask)

    covered = (yt >= lo) & (yt <= hi)
    return float(np.mean(covered))


def mpiw(
    lower: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean Prediction Interval Width in physical units (mph).

    Parameters
    ----------
    lower : lower interval bounds.
    upper : upper interval bounds (same shape).
    mask : optional observation mask (same shape).

    Returns
    -------
    Scalar mean width.
    """
    _validate_arrays(lower, upper)
    _validate_shape_match(lower, upper)

    lo = _apply_mask(lower, mask)
    hi = _apply_mask(upper, mask)

    widths = hi - lo
    if np.any(widths < 0):
        raise MetricError("upper must be >= lower everywhere.")
    return float(np.mean(widths))


def calibration_curve(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nominal_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95, 0.99),
    mask: np.ndarray | None = None,
) -> dict[float, dict[str, float | int]]:
    """Compute empirical coverage and width across multiple nominal confidence levels.

    Parameters
    ----------
    y_true : ground-truth values.
    mu : predicted means.
    sigma : predicted standard deviations (strictly positive).
    nominal_levels : tuple of nominal confidence levels in (0, 1).
    mask : optional observation mask.

    Returns
    -------
    Dictionary mapping nominal level -> {'picp': float, 'mpiw': float, 'valid_count': int}.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")

    results = {}
    valid_count = int(np.sum(mask.astype(bool))) if mask is not None else int(y_true.size)

    for nom in nominal_levels:
        lo, hi = prediction_interval(mu, sigma, nominal=nom)
        cov = picp(y_true, lo, hi, mask=mask)
        w = mpiw(lo, hi, mask=mask)
        results[float(nom)] = {
            "picp": cov,
            "mpiw": w,
            "valid_count": valid_count,
        }

    return results


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------


def inverse_transform_predictions(
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    scaler_mean: np.ndarray | None = None,
    scaler_std: np.ndarray | None = None,
    *,
    var: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """Convert normalized predictions, scales, variances, and bounds to raw traffic units.

    Mathematical rules:
        - Mean / Location:   ``mu_raw = mu_norm * std + mean``
        - Standard deviation: ``sigma_raw = sigma_norm * std``  (scale transforms linearly without mean)
        - Variance:          ``var_raw = var_norm * (std ** 2)``
        - Lower / Upper:     ``bound_raw = bound_norm * std + mean``

    Parameters
    ----------
    mu : normalized predicted means, shape [..., N, 1] or [..., N].
    sigma : normalized predicted std devs (same shape, strictly positive).
    scaler_mean : per-sensor training means, shape [N].
    scaler_std : per-sensor training std devs, shape [N] (strictly positive).
    var : optional normalized variances (same shape).
    lower : optional normalized lower bounds (same shape).
    upper : optional normalized upper bounds (same shape).

    Returns
    -------
    If called with (mu, sigma, scaler_mean, scaler_std) only, returns ``(mu_raw, sigma_raw)``.
    If called with extended arguments, returns a dict with transformed arrays.
    """
    if scaler_mean is None or scaler_std is None:
        raise MetricError("scaler_mean and scaler_std are required.")

    _validate_arrays(scaler_mean, scaler_std)
    _validate_positive(scaler_std, "scaler_std")

    # Find reference array for shape broadcasting
    ref_arr = mu if mu is not None else (sigma if sigma is not None else lower)
    if ref_arr is None:
        raise MetricError("At least one prediction array (mu, sigma, var, lower) must be provided.")

    _validate_arrays(ref_arr)
    N = len(scaler_mean)

    # Determine sensor axis and reshape scaler parameters
    if ref_arr.ndim > 1 and ref_arr.shape[-1] == 1:
        if ref_arr.shape[-2] != N:
            raise MetricError(
                f"Sensor dimension mismatch: array has {ref_arr.shape[-2]} sensors, "
                f"but scaler has {N} sensors."
            )
        s_mean = scaler_mean.reshape(*([1] * (ref_arr.ndim - 2)), N, 1)
        s_std = scaler_std.reshape(*([1] * (ref_arr.ndim - 2)), N, 1)
    elif ref_arr.ndim >= 1:
        if ref_arr.shape[-1] != N:
            raise MetricError(
                f"Sensor dimension mismatch: array has {ref_arr.shape[-1]} sensors, "
                f"but scaler has {N} sensors."
            )
        s_mean = scaler_mean.reshape(*([1] * (ref_arr.ndim - 1)), N)
        s_std = scaler_std.reshape(*([1] * (ref_arr.ndim - 1)), N)
    else:
        s_mean = scaler_mean
        s_std = scaler_std

    results: dict[str, np.ndarray] = {}

    if mu is not None:
        _validate_arrays(mu)
        results["mu"] = mu * s_std + s_mean

    if sigma is not None:
        _validate_arrays(sigma)
        _validate_positive(sigma, "sigma")
        results["sigma"] = sigma * s_std

    if var is not None:
        _validate_arrays(var)
        _validate_positive(var, "var")
        results["var"] = var * (s_std ** 2)

    if lower is not None:
        _validate_arrays(lower)
        results["lower"] = lower * s_std + s_mean

    if upper is not None:
        _validate_arrays(upper)
        results["upper"] = upper * s_std + s_mean

    # Backward-compatible tuple return if only mu and sigma were provided
    if len(results) == 2 and "mu" in results and "sigma" in results and var is None and lower is None and upper is None:
        return results["mu"], results["sigma"]

    return results


# ---------------------------------------------------------------------------
# Per-horizon and subset evaluation
# ---------------------------------------------------------------------------


def evaluate_metrics_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    nominal_pi: float = 0.95,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Evaluate point and probabilistic metrics overall and across each horizon.

    Expects inputs of shape ``[S, H, N, 1]`` where H is the forecast horizon (e.g. 12 steps).

    Parameters
    ----------
    y_true : ground truth in physical traffic units [S, H, N, 1].
    y_pred : predicted means in physical traffic units [S, H, N, 1].
    sigma : predicted std devs in physical traffic units [S, H, N, 1] (optional).
    mask : binary observation mask [S, H, N, 1] (optional).
    nominal_pi : nominal coverage for interval metrics (default 0.95).

    Returns
    -------
    (overall_dict, per_horizon_list) containing metrics and valid counts.
    """
    _validate_arrays(y_true, y_pred)
    _validate_shape_match(y_true, y_pred)
    if y_true.ndim != 4 or y_true.shape[-1] != 1:
        raise MetricError(f"Expected 4D array [S, H, N, 1], got shape {y_true.shape}")

    H = y_true.shape[1]

    # Compute overall
    overall: dict[str, float | int] = {
        "MAE": mae(y_true, y_pred, mask=mask),
        "RMSE": rmse(y_true, y_pred, mask=mask),
        "MAPE": mape(y_true, y_pred, mask=mask),
        "valid_count": int(np.sum(mask.astype(bool))) if mask is not None else int(y_true.size),
    }

    if sigma is not None:
        _validate_arrays(sigma)
        _validate_shape_match(y_true, sigma)
        lo, hi = prediction_interval(y_pred, sigma, nominal=nominal_pi)
        overall["NLL"] = gaussian_nll(y_true, y_pred, sigma, mask=mask)
        overall["CRPS"] = gaussian_crps(y_true, y_pred, sigma, mask=mask)
        overall["PICP"] = picp(y_true, lo, hi, mask=mask)
        overall["MPIW"] = mpiw(lo, hi, mask=mask)

    per_horizon = []
    for h in range(H):
        yt_h = y_true[:, h : h + 1, :, :]
        yp_h = y_pred[:, h : h + 1, :, :]
        m_h = mask[:, h : h + 1, :, :] if mask is not None else None

        h_dict: dict[str, float | int] = {
            "horizon_step": h + 1,
            "horizon_minutes": (h + 1) * 5,
            "MAE": mae(yt_h, yp_h, mask=m_h),
            "RMSE": rmse(yt_h, yp_h, mask=m_h),
            "MAPE": mape(yt_h, yp_h, mask=m_h),
            "valid_count": int(np.sum(m_h.astype(bool))) if m_h is not None else int(yt_h.size),
        }

        if sigma is not None:
            sig_h = sigma[:, h : h + 1, :, :]
            lo_h, hi_h = prediction_interval(yp_h, sig_h, nominal=nominal_pi)
            h_dict["NLL"] = gaussian_nll(yt_h, yp_h, sig_h, mask=m_h)
            h_dict["CRPS"] = gaussian_crps(yt_h, yp_h, sig_h, mask=m_h)
            h_dict["PICP"] = picp(yt_h, lo_h, hi_h, mask=m_h)
            h_dict["MPIW"] = mpiw(lo_h, hi_h, mask=m_h)

        per_horizon.append(h_dict)

    return overall, per_horizon
