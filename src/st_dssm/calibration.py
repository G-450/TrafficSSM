"""Calibration diagnostics and the secondary temperature-scaling analysis.

Phase 13 diagnostics for the Gaussian predictive distribution, per ADR-0008:

- Interval calibration: empirical coverage against nominal level, summarized
  by an interval expected calibration error (``interval_ece``).
- Probabilistic calibration: the probability integral transform (PIT) of the
  observations, which is uniform on [0, 1] exactly when the predictive
  distribution is calibrated, summarized by ``pit_ece``.
- Temperature scaling: one positive scalar shared across sensors and horizons,
  fitted on the validation split by NLL and frozen before test evaluation.
  ADR-0008 admits this only as a separately labelled secondary analysis, and
  it may never alter predictive means, so ``apply_temperature`` takes only a
  scale and returns only a scale.

All functions operate on numpy arrays and are model-independent. Scaling the
predictive standard deviation is scale-equivariant, so a temperature fitted on
normalized predictions is identical to one fitted in physical units.

Functions
---------
reliability_table : empirical vs nominal coverage across levels
interval_ece : coverage-gap summary of a reliability table
pit_values, pit_ece, pit_histogram : probabilistic calibration diagnostics
fit_temperature : validation-NLL-optimal scalar sigma multiplier
apply_temperature : scale a predictive standard deviation by that scalar
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .metrics import (
    _apply_mask,
    _validate_arrays,
    _validate_positive,
    _validate_shape_match,
    gaussian_nll,
    mpiw,
    picp,
    prediction_interval,
)

#: Nominal coverage levels reported by the Phase 13 reliability diagnostics.
CANONICAL_LEVELS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

#: Bin count for the PIT histogram and ``pit_ece``.
PIT_BINS: int = 20


class CalibrationError(Exception):
    """Raised when calibration inputs or fitted parameters violate requirements."""


def _validate_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    """Reject empty, out-of-range or duplicated nominal levels."""
    if len(levels) == 0:
        raise CalibrationError("At least one nominal level is required.")
    out = tuple(float(level) for level in levels)
    for level in out:
        if not 0.0 < level < 1.0:
            raise CalibrationError(f"Nominal level {level} must lie strictly inside (0, 1).")
    if len(set(out)) != len(out):
        raise CalibrationError(f"Nominal levels must be unique, got {out}.")
    return out


# ---------------------------------------------------------------------------
# Interval calibration
# ---------------------------------------------------------------------------


def reliability_table(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    levels: tuple[float, ...] = CANONICAL_LEVELS,
    mask: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Tabulate empirical coverage against nominal coverage, level by level.

    This is the reliability diagram in machine-readable form: a calibrated
    model puts ``empirical_coverage`` on the diagonal at every level.

    Parameters
    ----------
    y_true : ground-truth values.
    mu : predicted means.
    sigma : predicted standard deviations (strictly positive).
    levels : nominal coverage levels in (0, 1).
    mask : optional observation mask selecting valid targets.

    Returns
    -------
    One row per level, ascending, with the nominal level, the empirical
    coverage (PICP), the mean interval width (MPIW), the signed coverage gap
    (empirical minus nominal) and the valid observation count.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")
    ordered = sorted(_validate_levels(levels))

    valid_count = int(_apply_mask(y_true, mask).size)

    rows: list[dict[str, float | int]] = []
    for level in ordered:
        lo, hi = prediction_interval(mu, sigma, nominal=level)
        coverage = picp(y_true, lo, hi, mask=mask)
        rows.append(
            {
                "nominal_level": level,
                "empirical_coverage": coverage,
                "mean_interval_width": mpiw(lo, hi, mask=mask),
                "coverage_gap": coverage - level,
                "valid_count": valid_count,
            }
        )
    return rows


def interval_ece(rows: list[dict[str, float | int]]) -> float:
    """Summarize a reliability table as a mean absolute coverage gap.

    The interval expected calibration error is the unweighted mean of
    ``|empirical_coverage - nominal_level|`` over the tabulated levels. Levels
    are weighted equally because they are chosen by the protocol rather than
    populated by the data, so there is no bin occupancy to weight by. The value
    is 0 for a perfectly calibrated forecast and is reported as a ratio.

    Parameters
    ----------
    rows : output of :func:`reliability_table`.

    Returns
    -------
    Mean absolute coverage gap across levels.
    """
    if not rows:
        raise CalibrationError("Cannot summarize an empty reliability table.")
    gaps = [abs(float(row["coverage_gap"])) for row in rows]
    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# Probabilistic calibration (PIT)
# ---------------------------------------------------------------------------


def pit_values(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Probability integral transform of the observations under the forecast.

    Returns ``Phi((y - mu) / sigma)``, flattened over the valid targets. These
    values are uniform on [0, 1] exactly when the predictive distribution is
    calibrated; a peaked histogram means over-dispersion and a U-shaped one
    means under-dispersion.

    Parameters
    ----------
    y_true : ground-truth values.
    mu : predicted means.
    sigma : predicted standard deviations (strictly positive).
    mask : optional observation mask selecting valid targets.

    Returns
    -------
    1D array of PIT values over the valid targets.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")

    flat_true = _apply_mask(y_true, mask)
    flat_mu = _apply_mask(mu, mask)
    flat_sigma = _apply_mask(sigma, mask)

    return norm.cdf((flat_true - flat_mu) / flat_sigma)


def pit_histogram(
    pit: np.ndarray, *, bins: int = PIT_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Bin PIT values into equal-width densities over [0, 1].

    Parameters
    ----------
    pit : PIT values from :func:`pit_values`.
    bins : number of equal-width bins.

    Returns
    -------
    ``(densities, edges)``, where ``densities`` are the per-bin observed
    frequencies normalized so that a calibrated forecast gives 1.0 in every
    bin, and ``edges`` has ``bins + 1`` entries.
    """
    if bins < 2:
        raise CalibrationError(f"PIT histogram needs at least 2 bins, got {bins}.")
    if pit.size == 0:
        raise CalibrationError("Cannot histogram an empty PIT array.")

    counts, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    # Normalize against the uniform expectation so 1.0 is the calibrated level
    # at any bin count.
    densities = counts.astype(np.float64) / (pit.size / bins)
    return densities, edges


def pit_ece(pit: np.ndarray, *, bins: int = PIT_BINS) -> float:
    """Total-variation distance between the PIT histogram and the uniform.

    Computed as ``0.5 * sum |observed_fraction - 1/bins|`` over the bins, so it
    is 0 for a calibrated forecast and at most 1 for a fully displaced one, and
    is comparable across bin counts.

    Parameters
    ----------
    pit : PIT values from :func:`pit_values`.
    bins : number of equal-width bins.

    Returns
    -------
    Total-variation distance to the uniform distribution, as a ratio.
    """
    densities, _ = pit_histogram(pit, bins=bins)
    fractions = densities / bins
    return float(0.5 * np.sum(np.abs(fractions - 1.0 / bins)))


# ---------------------------------------------------------------------------
# Temperature scaling (secondary analysis, ADR-0008)
# ---------------------------------------------------------------------------


def fit_temperature(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """Fit the scalar sigma multiplier that minimizes Gaussian NLL.

    One positive scalar shared across sensors and horizons, as ADR-0008
    requires. For ``sigma -> T * sigma`` the Gaussian NLL is

        NLL(T) = const + log T + mean(r^2) / (2 T^2),    r = (y - mu) / sigma

    which is strictly convex in ``log T`` with the stationary point
    ``T* = sqrt(mean(r^2))``. The closed form is therefore the exact global
    minimizer and needs no search; ``T* > 1`` means the raw forecast was
    over-confident. Fit this on the validation split only, then freeze it.

    Parameters
    ----------
    y_true : validation ground-truth values.
    mu : validation predicted means.
    sigma : validation predicted standard deviations (strictly positive).
    mask : optional observation mask selecting valid targets.

    Returns
    -------
    The fitted temperature, strictly positive.
    """
    _validate_arrays(y_true, mu, sigma)
    _validate_shape_match(y_true, mu, sigma)
    _validate_positive(sigma, "sigma")

    flat_true = _apply_mask(y_true, mask)
    flat_mu = _apply_mask(mu, mask)
    flat_sigma = _apply_mask(sigma, mask)

    residuals = (flat_true - flat_mu) / flat_sigma
    temperature = float(np.sqrt(np.mean(residuals**2)))

    if not np.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationError(
            f"Fitted temperature {temperature} is not strictly positive and finite. "
            "The validation residuals are degenerate."
        )
    return temperature


def apply_temperature(sigma: np.ndarray, temperature: float) -> np.ndarray:
    """Scale a predictive standard deviation by a frozen temperature.

    Takes and returns only a scale, so predictive means cannot be altered by
    calibration (ADR-0008).

    Parameters
    ----------
    sigma : predicted standard deviations (strictly positive).
    temperature : the frozen scalar from :func:`fit_temperature`.

    Returns
    -------
    ``temperature * sigma``, as float64.
    """
    _validate_arrays(sigma)
    _validate_positive(sigma, "sigma")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationError(
            f"Temperature must be strictly positive and finite, got {temperature}."
        )
    return sigma.astype(np.float64, copy=False) * float(temperature)


def calibration_report(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    temperature: float | None = None,
    levels: tuple[float, ...] = CANONICAL_LEVELS,
    bins: int = PIT_BINS,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute the full Phase 13 diagnostic set for one set of predictions.

    Parameters
    ----------
    y_true : ground-truth values.
    mu : predicted means.
    sigma : raw predicted standard deviations (strictly positive).
    temperature : optional frozen temperature; when given, the report also
        carries the calibrated arm under ``calibrated``.
    levels : nominal coverage levels for the reliability table.
    bins : PIT histogram bin count.
    mask : optional observation mask selecting valid targets.

    Returns
    -------
    ``{"raw": {...}}``, plus ``"calibrated"`` and ``"temperature"`` when a
    temperature is supplied. Each arm carries ``nll``, ``interval_ece``,
    ``pit_ece``, the reliability table and the PIT histogram densities.
    """

    def _arm(scale: np.ndarray) -> dict[str, object]:
        rows = reliability_table(y_true, mu, scale, levels=levels, mask=mask)
        pit = pit_values(y_true, mu, scale, mask=mask)
        densities, edges = pit_histogram(pit, bins=bins)
        return {
            "nll": gaussian_nll(y_true, mu, scale, mask=mask),
            "interval_ece": interval_ece(rows),
            "pit_ece": pit_ece(pit, bins=bins),
            "reliability": rows,
            "pit_densities": [float(d) for d in densities],
            "pit_edges": [float(e) for e in edges],
        }

    report: dict[str, object] = {"raw": _arm(sigma)}
    if temperature is not None:
        report["temperature"] = float(temperature)
        report["calibrated"] = _arm(apply_temperature(sigma, temperature))
    return report
