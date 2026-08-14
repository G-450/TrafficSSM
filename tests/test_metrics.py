"""Tests for the evaluation metrics library (Phase 4).

Uses known analytical/synthetic cases to validate every metric.
No real dataset is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from st_dssm.metrics import (
    MetricError,
    gaussian_crps,
    gaussian_nll,
    inverse_transform_predictions,
    mae,
    mpiw,
    picp,
    prediction_interval,
    rmse,
)

# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_constant_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # errors are [1, 1, 1], mean = 1
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, -1.0, 2.0, -2.0])
        # |errors| = [1, 1, 2, 2], mean = 1.5
        assert mae(y_true, y_pred) == pytest.approx(1.5)

    def test_multidimensional(self):
        y_true = np.ones((2, 3, 4))
        y_pred = np.zeros((2, 3, 4))
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_rejects_nan(self):
        y = np.array([1.0, np.nan, 3.0])
        with pytest.raises(MetricError):
            mae(y, y)

    def test_rejects_inf(self):
        y = np.array([1.0, np.inf, 3.0])
        with pytest.raises(MetricError):
            mae(y, y)

    def test_shape_mismatch(self):
        with pytest.raises(MetricError):
            mae(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # squared errors: 9, 16; mean: 12.5; sqrt: 3.5355...
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_rmse_ge_mae(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


# ---------------------------------------------------------------------------
# Gaussian NLL
# ---------------------------------------------------------------------------

class TestGaussianNLL:
    def test_standard_normal_at_mean(self):
        """NLL of N(0,1) at y=0 should be 0.5*ln(2π) ≈ 0.9189."""
        y = np.array([0.0])
        mu = np.array([0.0])
        sigma = np.array([1.0])
        expected = 0.5 * np.log(2 * np.pi)
        assert gaussian_nll(y, mu, sigma) == pytest.approx(expected, rel=1e-6)

    def test_known_nll(self):
        """NLL of N(0,1) at y=1: 0.5*ln(2π) + 0.5*1 = 0.5*ln(2π) + 0.5."""
        y = np.array([1.0])
        mu = np.array([0.0])
        sigma = np.array([1.0])
        expected = 0.5 * np.log(2 * np.pi) + 0.5
        assert gaussian_nll(y, mu, sigma) == pytest.approx(expected, rel=1e-6)

    def test_larger_sigma_at_mean_reduces_nll_peak(self):
        """NLL at the mean decreases for larger sigma (wider distribution)
        only when the penalty from ln(sigma) doesn't dominate. For sigma=2
        at mean: 0.5*ln(2π) + ln(2) ≈ 1.612, which is > 0.919 for sigma=1.
        So NLL increases with sigma at the mean."""
        y = np.array([0.0])
        mu = np.array([0.0])
        nll_s1 = gaussian_nll(y, mu, np.array([1.0]))
        nll_s2 = gaussian_nll(y, mu, np.array([2.0]))
        assert nll_s2 > nll_s1  # ln(sigma) term dominates at the mean

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(MetricError):
            gaussian_nll(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        with pytest.raises(MetricError):
            gaussian_nll(np.array([0.0]), np.array([0.0]), np.array([-1.0]))

    def test_batch(self):
        y = np.array([0.0, 1.0])
        mu = np.array([0.0, 0.0])
        sigma = np.array([1.0, 1.0])
        nll_0 = 0.5 * np.log(2 * np.pi)
        nll_1 = 0.5 * np.log(2 * np.pi) + 0.5
        expected = (nll_0 + nll_1) / 2
        assert gaussian_nll(y, mu, sigma) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Gaussian CRPS
# ---------------------------------------------------------------------------

class TestGaussianCRPS:
    def test_standard_normal_at_mean(self):
        """CRPS of N(0,1) at y=0.

        z=0, Φ(0)=0.5, φ(0)=1/√(2π)
        CRPS = 1 * [0*(2*0.5-1) + 2/√(2π) - 1/√π]
             = (√2 - 1) / √π  ≈  0.2337
        """
        y = np.array([0.0])
        mu = np.array([0.0])
        sigma = np.array([1.0])
        expected = (np.sqrt(2) - 1) / np.sqrt(np.pi)
        assert gaussian_crps(y, mu, sigma) == pytest.approx(expected, rel=1e-5)

    def test_crps_nonnegative(self):
        """CRPS is always non-negative."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(50)
        mu = rng.standard_normal(50)
        sigma = np.abs(rng.standard_normal(50)) + 0.1
        assert gaussian_crps(y, mu, sigma) >= 0.0

    def test_crps_zero_for_delta(self):
        """As sigma → 0, CRPS → |y - mu|."""
        y = np.array([3.0])
        mu = np.array([1.0])
        sigma = np.array([1e-8])
        # CRPS should approach |3-1| = 2
        assert gaussian_crps(y, mu, sigma) == pytest.approx(2.0, abs=1e-4)

    def test_crps_scales_with_sigma(self):
        """CRPS is proportional to sigma (for fixed z)."""
        y = np.array([0.0])
        mu = np.array([0.0])
        crps_1 = gaussian_crps(y, mu, np.array([1.0]))
        crps_2 = gaussian_crps(y, mu, np.array([2.0]))
        assert crps_2 == pytest.approx(2.0 * crps_1, rel=1e-5)

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(MetricError):
            gaussian_crps(np.array([0.0]), np.array([0.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Prediction Interval
# ---------------------------------------------------------------------------

class TestPredictionInterval:
    def test_95_interval(self):
        mu = np.array([0.0])
        sigma = np.array([1.0])
        lower, upper = prediction_interval(mu, sigma, 0.95)
        z95 = norm.ppf(0.975)
        assert lower[0] == pytest.approx(-z95, rel=1e-6)
        assert upper[0] == pytest.approx(z95, rel=1e-6)

    def test_wider_sigma_wider_interval(self):
        mu = np.array([5.0])
        _, upper1 = prediction_interval(mu, np.array([1.0]), 0.95)
        _, upper2 = prediction_interval(mu, np.array([2.0]), 0.95)
        assert upper2[0] > upper1[0]

    def test_invalid_nominal(self):
        with pytest.raises(MetricError):
            prediction_interval(np.array([0.0]), np.array([1.0]), 0.0)
        with pytest.raises(MetricError):
            prediction_interval(np.array([0.0]), np.array([1.0]), 1.0)


# ---------------------------------------------------------------------------
# PICP
# ---------------------------------------------------------------------------

class TestPICP:
    def test_full_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        assert picp(y, lower, upper) == pytest.approx(1.0)

    def test_zero_coverage(self):
        y = np.array([100.0, 200.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        assert picp(y, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self):
        y = np.array([5.0, 15.0, 25.0, 35.0])
        lower = np.array([0.0, 10.0, 20.0, 40.0])
        upper = np.array([10.0, 20.0, 30.0, 50.0])
        # 5 in [0,10] ✓, 15 in [10,20] ✓, 25 in [20,30] ✓, 35 in [40,50] ✗
        assert picp(y, lower, upper) == pytest.approx(0.75)

    def test_boundary_inclusion(self):
        """Values exactly at bounds are counted as covered."""
        y = np.array([0.0, 10.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        assert picp(y, lower, upper) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MPIW
# ---------------------------------------------------------------------------

class TestMPIW:
    def test_known_width(self):
        lower = np.array([0.0, 5.0])
        upper = np.array([10.0, 15.0])
        assert mpiw(lower, upper) == pytest.approx(10.0)

    def test_zero_width(self):
        val = np.array([1.0, 2.0, 3.0])
        assert mpiw(val, val) == pytest.approx(0.0)

    def test_rejects_inverted_bounds(self):
        with pytest.raises(MetricError):
            mpiw(np.array([10.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------

class TestInverseTransform:
    def test_round_trip_2d(self):
        """inverse(transform(x)) ≈ x for [T, N] data."""
        rng = np.random.default_rng(42)
        raw = rng.uniform(20, 80, size=(100, 5))
        s_mean = np.mean(raw, axis=0)
        s_std = np.std(raw, axis=0)

        norm_mu = (raw - s_mean) / s_std
        # Use a constant sigma in normalized space
        norm_sigma = np.full_like(norm_mu, 0.5)

        mu_raw, sigma_raw = inverse_transform_predictions(
            norm_mu, norm_sigma, s_mean, s_std,
        )
        np.testing.assert_allclose(mu_raw, raw, rtol=1e-6)
        expected_sigma = 0.5 * s_std
        np.testing.assert_allclose(
            sigma_raw[0], expected_sigma, rtol=1e-6,
        )

    def test_4d_shape(self):
        """Works with canonical [B, H, N, 1] shape."""
        B, H, N = 2, 12, 5
        mu = np.ones((B, H, N, 1))
        sigma = np.ones((B, H, N, 1)) * 0.5
        s_mean = np.zeros(N)
        s_std = np.ones(N) * 2.0

        mu_raw, sigma_raw = inverse_transform_predictions(
            mu, sigma, s_mean, s_std,
        )
        assert mu_raw.shape == (B, H, N, 1)
        assert sigma_raw.shape == (B, H, N, 1)
        # mu_raw = 1 * 2 + 0 = 2
        np.testing.assert_allclose(mu_raw, 2.0)
        # sigma_raw = 0.5 * 2 = 1
        np.testing.assert_allclose(sigma_raw, 1.0)
