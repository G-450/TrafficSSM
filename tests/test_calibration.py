"""Phase 13 calibration diagnostics and temperature-scaling tests.

The diagnostics are validated against synthetic forecasts whose calibration is
known by construction: a draw from the predicted distribution is calibrated,
and a known inflation or deflation of sigma is mis-calibrated by exactly that
factor, which temperature scaling must recover.
"""

from __future__ import annotations

import numpy as np
import pytest

from st_dssm.calibration import (
    CANONICAL_LEVELS,
    CalibrationError,
    apply_temperature,
    calibration_report,
    fit_temperature,
    interval_ece,
    pit_ece,
    pit_histogram,
    pit_values,
    reliability_table,
)
from st_dssm.metrics import MetricError, gaussian_nll

SHAPE = (400, 12, 40, 1)


def _calibrated_forecast(seed: int = 2026, sigma_level: float = 2.5):
    """Observations drawn from exactly the predicted distribution."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(55.0, 6.0, size=SHAPE).astype(np.float64)
    sigma = np.full(SHAPE, sigma_level, dtype=np.float64)
    y_true = mu + rng.normal(0.0, sigma_level, size=SHAPE)
    return y_true, mu, sigma


# ---------------------------------------------------------------------------
# Reliability table and interval ECE
# ---------------------------------------------------------------------------


def test_reliability_table_is_near_diagonal_for_a_calibrated_forecast():
    y_true, mu, sigma = _calibrated_forecast()
    rows = reliability_table(y_true, mu, sigma)

    assert [row["nominal_level"] for row in rows] == sorted(CANONICAL_LEVELS)
    for row in rows:
        assert abs(row["coverage_gap"]) < 0.01, row
    assert interval_ece(rows) < 0.005


def test_reliability_table_detects_over_and_under_dispersion():
    y_true, mu, sigma = _calibrated_forecast()

    # Sigma inflated: intervals too wide, so coverage overshoots nominal.
    over = reliability_table(y_true, mu, sigma * 2.0)
    assert all(row["coverage_gap"] > 0 for row in over[:-1])

    # Sigma deflated: intervals too narrow, so coverage undershoots nominal.
    under = reliability_table(y_true, mu, sigma * 0.5)
    assert all(row["coverage_gap"] < 0 for row in under)

    assert interval_ece(over) > 0.05
    assert interval_ece(under) > 0.2


def test_reliability_widths_increase_with_nominal_level():
    y_true, mu, sigma = _calibrated_forecast()
    rows = reliability_table(y_true, mu, sigma)
    widths = [row["mean_interval_width"] for row in rows]
    assert widths == sorted(widths)


def test_reliability_table_honours_the_mask():
    y_true, mu, sigma = _calibrated_forecast()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, :, :10, :] = True

    rows = reliability_table(y_true, mu, sigma, mask=mask)
    assert all(row["valid_count"] == int(mask.sum()) for row in rows)

    subset = reliability_table(
        y_true[:, :, :10, :], mu[:, :, :10, :], sigma[:, :, :10, :]
    )
    for masked_row, subset_row in zip(rows, subset, strict=True):
        assert masked_row["empirical_coverage"] == pytest.approx(
            subset_row["empirical_coverage"]
        )


def test_reliability_table_rejects_invalid_levels():
    y_true, mu, sigma = _calibrated_forecast()
    for bad in [(), (0.0,), (1.0,), (-0.1,), (1.5,), (0.9, 0.9)]:
        with pytest.raises(CalibrationError):
            reliability_table(y_true, mu, sigma, levels=bad)


def test_interval_ece_rejects_an_empty_table():
    with pytest.raises(CalibrationError):
        interval_ece([])


def test_reliability_table_rejects_non_positive_sigma():
    y_true, mu, sigma = _calibrated_forecast()
    sigma = sigma.copy()
    sigma[0, 0, 0, 0] = 0.0
    with pytest.raises(MetricError):
        reliability_table(y_true, mu, sigma)


# ---------------------------------------------------------------------------
# PIT diagnostics
# ---------------------------------------------------------------------------


def test_pit_values_are_uniform_for_a_calibrated_forecast():
    y_true, mu, sigma = _calibrated_forecast()
    pit = pit_values(y_true, mu, sigma)

    assert pit.shape == (int(np.prod(SHAPE)),)
    assert np.all((pit >= 0.0) & (pit <= 1.0))
    assert pit.mean() == pytest.approx(0.5, abs=0.01)
    # Variance of U(0, 1) is 1/12.
    assert pit.var() == pytest.approx(1.0 / 12.0, abs=0.005)
    assert pit_ece(pit) < 0.02


def test_pit_histogram_is_flat_for_a_calibrated_forecast():
    y_true, mu, sigma = _calibrated_forecast()
    densities, edges = pit_histogram(pit_values(y_true, mu, sigma))

    assert len(edges) == len(densities) + 1
    assert edges[0] == 0.0 and edges[-1] == 1.0
    assert np.allclose(densities, 1.0, atol=0.12)


def test_pit_ece_separates_over_from_under_dispersion():
    y_true, mu, sigma = _calibrated_forecast()

    calibrated = pit_ece(pit_values(y_true, mu, sigma))
    over_dispersed = pit_ece(pit_values(y_true, mu, sigma * 2.0))
    under_dispersed = pit_ece(pit_values(y_true, mu, sigma * 0.5))

    assert calibrated < over_dispersed
    assert calibrated < under_dispersed
    # A displaced histogram cannot exceed total variation 1.
    assert 0.0 <= over_dispersed <= 1.0
    assert 0.0 <= under_dispersed <= 1.0


def test_pit_histogram_rejects_degenerate_arguments():
    y_true, mu, sigma = _calibrated_forecast()
    pit = pit_values(y_true, mu, sigma)
    with pytest.raises(CalibrationError):
        pit_histogram(pit, bins=1)
    with pytest.raises(CalibrationError):
        pit_histogram(np.array([]))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def test_fitted_temperature_is_one_for_a_calibrated_forecast():
    y_true, mu, sigma = _calibrated_forecast()
    assert fit_temperature(y_true, mu, sigma) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("factor", [0.25, 0.5, 2.0, 4.0])
def test_fitted_temperature_recovers_a_known_sigma_distortion(factor):
    """A forecast whose sigma is off by 1/factor is corrected by factor."""
    y_true, mu, sigma = _calibrated_forecast()
    distorted = sigma / factor
    assert fit_temperature(y_true, mu, distorted) == pytest.approx(factor, rel=0.03)


def test_closed_form_temperature_matches_a_grid_search_on_nll():
    """The analytic minimizer must beat every point of a dense NLL grid."""
    y_true, mu, sigma = _calibrated_forecast(seed=2027)
    distorted = sigma * 0.4

    fitted = fit_temperature(y_true, mu, distorted)
    best_nll = gaussian_nll(y_true, mu, apply_temperature(distorted, fitted))

    grid = np.linspace(0.5 * fitted, 1.5 * fitted, 201)
    grid_nlls = [gaussian_nll(y_true, mu, apply_temperature(distorted, t)) for t in grid]

    assert best_nll <= min(grid_nlls) + 1e-9
    assert grid[int(np.argmin(grid_nlls))] == pytest.approx(fitted, rel=0.01)


def test_temperature_scaling_improves_nll_on_a_miscalibrated_forecast():
    y_true, mu, sigma = _calibrated_forecast()
    distorted = sigma * 3.0

    raw_nll = gaussian_nll(y_true, mu, distorted)
    fitted = fit_temperature(y_true, mu, distorted)
    calibrated_nll = gaussian_nll(y_true, mu, apply_temperature(distorted, fitted))

    assert calibrated_nll < raw_nll
    assert fitted == pytest.approx(1.0 / 3.0, rel=0.03)


def test_temperature_fitted_on_a_mask_uses_only_those_targets():
    y_true, mu, sigma = _calibrated_forecast()
    # Inflate the residuals outside the mask; the fit must ignore them.
    y_true = y_true.copy()
    y_true[:, :, 10:, :] += 50.0

    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, :, :10, :] = True

    assert fit_temperature(y_true, mu, sigma, mask=mask) == pytest.approx(1.0, abs=0.03)


def test_temperature_is_scale_equivariant():
    """Fitting on normalized or physical units gives the same temperature."""
    y_true, mu, sigma = _calibrated_forecast()
    scale = 7.5
    distorted = sigma * 2.0

    normalized = fit_temperature(y_true, mu, distorted)
    physical = fit_temperature(y_true * scale, mu * scale, distorted * scale)
    assert normalized == pytest.approx(physical, rel=1e-12)


def test_apply_temperature_scales_only_the_scale():
    _, _, sigma = _calibrated_forecast()
    scaled = apply_temperature(sigma, 1.7)

    assert np.allclose(scaled, sigma * 1.7)
    assert scaled.shape == sigma.shape
    # The input array must not be mutated in place.
    assert np.allclose(sigma, 2.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_apply_temperature_rejects_non_positive_or_non_finite(bad):
    _, _, sigma = _calibrated_forecast()
    with pytest.raises(CalibrationError):
        apply_temperature(sigma, bad)


def test_fit_temperature_rejects_degenerate_residuals():
    """Zero residuals everywhere give T* = 0, which is not a valid scale."""
    mu = np.full(SHAPE, 55.0)
    sigma = np.full(SHAPE, 2.0)
    with pytest.raises(CalibrationError):
        fit_temperature(mu.copy(), mu, sigma)


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------


def test_calibration_report_carries_both_arms_and_improves_the_calibrated_one():
    y_true, mu, sigma = _calibrated_forecast()
    distorted = sigma * 2.5
    temperature = fit_temperature(y_true, mu, distorted)

    report = calibration_report(y_true, mu, distorted, temperature=temperature)

    assert set(report) == {"raw", "calibrated", "temperature"}
    assert report["temperature"] == pytest.approx(temperature)

    raw, calibrated = report["raw"], report["calibrated"]
    assert calibrated["nll"] < raw["nll"]
    assert calibrated["interval_ece"] < raw["interval_ece"]
    assert calibrated["pit_ece"] < raw["pit_ece"]
    assert len(raw["reliability"]) == len(CANONICAL_LEVELS)
    assert len(raw["pit_edges"]) == len(raw["pit_densities"]) + 1


def test_calibration_report_without_a_temperature_has_only_the_raw_arm():
    y_true, mu, sigma = _calibrated_forecast()
    report = calibration_report(y_true, mu, sigma)
    assert set(report) == {"raw"}
