"""Tests for evaluation metrics library and inverse transformation (Phase 4).

Validates all point and probabilistic metrics using known analytical and
synthetic test cases. Covers edge cases, numerical stability, masking,
zero-handling, and inverse transformation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from st_dssm.metrics import (
    MetricError,
    calibration_curve,
    evaluate_metrics_by_horizon,
    gaussian_crps,
    gaussian_nll,
    inverse_transform_predictions,
    mae,
    mape,
    mpiw,
    picp,
    prediction_interval,
    rmse,
)

# ---------------------------------------------------------------------------
# Point Forecast: MAE
# ---------------------------------------------------------------------------


class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_constant_error(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 22.0, 32.0])
        assert mae(y_true, y_pred) == pytest.approx(2.0)

    def test_known_hand_calculated(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.5, -2.5, 3.0, -1.0])
        # |errors| = [1.5, 2.5, 3.0, 1.0], sum = 8.0, mean = 2.0
        assert mae(y_true, y_pred) == pytest.approx(2.0)

    def test_multidimensional_4d(self):
        y_true = np.ones((2, 12, 5, 1)) * 50.0
        y_pred = np.ones((2, 12, 5, 1)) * 48.0
        assert mae(y_true, y_pred) == pytest.approx(2.0)

    def test_with_mask(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 25.0, 30.0, 99.0])
        # Mask out index 3: valid errors are |12-10|=2, |25-20|=5, |30-30|=0 -> mean = 7/3
        mask = np.array([1, 1, 1, 0])
        assert mae(y_true, y_pred, mask=mask) == pytest.approx(7.0 / 3.0)

    def test_rejects_non_binary_mask(self):
        y = np.array([10.0, 20.0])
        bad_mask = np.array([-1.0, 2.0])
        with pytest.raises(MetricError, match="strictly binary"):
            mae(y, y, mask=bad_mask)

    def test_empty_mask_raises(self):
        y = np.array([1.0, 2.0])
        mask = np.array([0, 0])
        with pytest.raises(MetricError, match="completely empty"):
            mae(y, y, mask=mask)

    def test_rejects_nan_and_inf(self):
        with pytest.raises(MetricError, match="NaN or Inf"):
            mae(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
        with pytest.raises(MetricError, match="NaN or Inf"):
            mae(np.array([1.0, 2.0]), np.array([1.0, np.inf]))

    def test_shape_mismatch(self):
        with pytest.raises(MetricError, match="Shape mismatch"):
            mae(np.ones(5), np.ones(6))


# ---------------------------------------------------------------------------
# Point Forecast: RMSE
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_hand_calculated(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # squared errors: 9, 16; mean = 12.5; sqrt(12.5) ≈ 3.5355339
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_rmse_ge_mae_property(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(30, 70, size=100)
        y_pred = rng.uniform(30, 70, size=100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    def test_with_mask(self):
        y_true = np.array([0.0, 0.0, 100.0])
        y_pred = np.array([3.0, 4.0, 0.0])
        mask = np.array([1, 1, 0])  # ignore third
        assert rmse(y_true, y_pred, mask=mask) == pytest.approx(np.sqrt(12.5))


# ---------------------------------------------------------------------------
# Point Forecast: MAPE
# ---------------------------------------------------------------------------


class TestMAPE:
    def test_perfect_prediction(self):
        y = np.array([50.0, 60.0, 70.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_known_hand_calculated(self):
        y_true = np.array([50.0, 100.0])
        y_pred = np.array([55.0, 90.0])
        # errors: |55-50|/50 = 0.10 (10%), |90-100|/100 = 0.10 (10%) -> mean = 10%
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_threshold_filters_zero(self):
        # 0.0 speed should be filtered out by threshold=1.0 without dividing by zero
        y_true = np.array([0.0, 50.0])
        y_pred = np.array([5.0, 55.0])
        # Only index 1 is evaluated: |55-50|/50 = 10%
        assert mape(y_true, y_pred, threshold=1.0) == pytest.approx(10.0)

    def test_all_below_threshold_raises(self):
        y_true = np.array([0.0, 0.5])
        y_pred = np.array([1.0, 1.0])
        with pytest.raises(MetricError, match="No valid targets"):
            mape(y_true, y_pred, threshold=1.0)


# ---------------------------------------------------------------------------
# Probabilistic: Gaussian NLL
# ---------------------------------------------------------------------------


class TestGaussianNLL:
    def test_standard_normal_at_mean(self):
        """At y=mu=0 with sigma=1, NLL = 0.5*ln(2*pi) ≈ 0.9189385."""
        y = np.array([0.0])
        mu = np.array([0.0])
        sigma = np.array([1.0])
        expected = 0.5 * np.log(2.0 * np.pi)
        assert gaussian_nll(y, mu, sigma) == pytest.approx(expected, rel=1e-6)

    def test_known_hand_calculated(self):
        """At y=2, mu=0, sigma=2:
        z = (2-0)/2 = 1.
        NLL = 0.5*ln(2*pi) + ln(2) + 0.5*(1)^2 ≈ 0.9189385 + 0.693147 + 0.5 = 2.1120857
        """
        y = np.array([2.0])
        mu = np.array([0.0])
        sigma = np.array([2.0])
        expected = 0.5 * np.log(2.0 * np.pi) + np.log(2.0) + 0.5
        assert gaussian_nll(y, mu, sigma) == pytest.approx(expected, rel=1e-6)

    def test_rejects_non_positive_sigma(self):
        with pytest.raises(MetricError, match="strictly positive"):
            gaussian_nll(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        with pytest.raises(MetricError, match="strictly positive"):
            gaussian_nll(np.array([0.0]), np.array([0.0]), np.array([-1.0]))

    def test_with_mask(self):
        y = np.array([0.0, 100.0])
        mu = np.array([0.0, 0.0])
        sigma = np.array([1.0, 1.0])
        mask = np.array([1, 0])
        expected = 0.5 * np.log(2.0 * np.pi)
        assert gaussian_nll(y, mu, sigma, mask=mask) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Probabilistic: Gaussian CRPS
# ---------------------------------------------------------------------------


class TestGaussianCRPS:
    def test_standard_normal_at_mean(self):
        """CRPS of N(0,1) at y=0 is (sqrt(2) - 1) / sqrt(pi) ≈ 0.2336949."""
        y = np.array([0.0])
        mu = np.array([0.0])
        sigma = np.array([1.0])
        expected = (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi)
        assert gaussian_crps(y, mu, sigma) == pytest.approx(expected, rel=1e-5)

    def test_crps_always_nonnegative(self):
        rng = np.random.default_rng(123)
        y = rng.normal(50, 10, size=100)
        mu = rng.normal(50, 10, size=100)
        sigma = rng.uniform(1.0, 15.0, size=100)
        assert gaussian_crps(y, mu, sigma) >= 0.0

    def test_crps_proportional_to_sigma_scaling(self):
        """CRPS scales linearly with sigma when (y-mu)/sigma is invariant."""
        y1, mu1, s1 = np.array([0.0]), np.array([0.0]), np.array([1.0])
        y2, mu2, s2 = np.array([0.0]), np.array([0.0]), np.array([3.5])
        crps1 = gaussian_crps(y1, mu1, s1)
        crps2 = gaussian_crps(y2, mu2, s2)
        assert crps2 == pytest.approx(3.5 * crps1, rel=1e-5)

    def test_crps_approaches_mae_as_sigma_approaches_zero(self):
        """As sigma -> 0, Gaussian CRPS -> |y - mu| (deterministic MAE)."""
        y = np.array([55.0])
        mu = np.array([50.0])
        sigma = np.array([1e-7])
        assert gaussian_crps(y, mu, sigma) == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Prediction Interval & Diagnostics: PICP, MPIW, Calibration
# ---------------------------------------------------------------------------


class TestPredictionIntervalAndDiagnostics:
    def test_95_nominal_interval(self):
        mu = np.array([50.0])
        sigma = np.array([5.0])
        lower, upper = prediction_interval(mu, sigma, nominal=0.95)
        z95 = float(norm.ppf(0.975))
        assert lower[0] == pytest.approx(50.0 - z95 * 5.0, rel=1e-6)
        assert upper[0] == pytest.approx(50.0 + z95 * 5.0, rel=1e-6)

    def test_picp_coverage(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        lower = np.array([5.0, 15.0, 25.0, 50.0])
        upper = np.array([15.0, 25.0, 35.0, 60.0])
        # 10 in [5,15] (yes), 20 in [15,25] (yes), 30 in [25,35] (yes), 40 in [50,60] (no)
        assert picp(y_true, lower, upper) == pytest.approx(0.75)

    def test_picp_rejects_crossed_bounds(self):
        y_true = np.array([10.0, 20.0])
        lower = np.array([15.0, 25.0])
        upper = np.array([5.0, 20.0])  # upper < lower
        with pytest.raises(MetricError, match="upper must be >= lower"):
            picp(y_true, lower, upper)

    def test_mpiw_width(self):
        lower = np.array([10.0, 20.0])
        upper = np.array([15.0, 30.0])
        # widths = [5.0, 10.0], mean = 7.5
        assert mpiw(lower, upper) == pytest.approx(7.5)

    def test_calibration_curve_multiple_levels(self):
        y_true = np.zeros(1000)
        mu = np.zeros(1000)
        sigma = np.ones(1000)
        curve = calibration_curve(y_true, mu, sigma, nominal_levels=(0.50, 0.90, 0.95))
        assert 0.50 in curve
        assert 0.90 in curve
        assert 0.95 in curve
        # Since y=mu=0 is in the center of all symmetric intervals around 0:
        assert curve[0.50]["picp"] == pytest.approx(1.0)
        assert curve[0.95]["mpiw"] > curve[0.50]["mpiw"]


# ---------------------------------------------------------------------------
# Inverse Transformation
# ---------------------------------------------------------------------------


class TestInverseTransform:
    def test_mean_and_sigma_inversion_4d(self):
        S, H, N = 2, 12, 3
        norm_mu = np.zeros((S, H, N, 1))
        norm_sigma = np.ones((S, H, N, 1)) * 0.5
        scaler_mean = np.array([50.0, 60.0, 70.0], dtype=np.float32)
        scaler_std = np.array([5.0, 10.0, 8.0], dtype=np.float32)

        mu_raw, sigma_raw = inverse_transform_predictions(
            norm_mu, norm_sigma, scaler_mean, scaler_std
        )
        assert mu_raw.shape == (S, H, N, 1)
        assert sigma_raw.shape == (S, H, N, 1)

        # mu_raw = 0 * std + mean = mean
        np.testing.assert_allclose(mu_raw[0, 0, :, 0], [50.0, 60.0, 70.0])
        # sigma_raw = 0.5 * std (NO mean added!)
        np.testing.assert_allclose(sigma_raw[0, 0, :, 0], [2.5, 5.0, 4.0])

    def test_zero_variance_is_valid(self):
        """Zero variance is mathematically valid and must not be rejected."""
        norm_var = np.array([[0.0, 4.0]])
        scaler_mean = np.array([50.0, 60.0])
        scaler_std = np.array([5.0, 2.0])

        res = inverse_transform_predictions(
            var=norm_var,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        assert isinstance(res, dict)
        # 0.0 * 25 = 0.0, 4.0 * 4 = 16.0
        np.testing.assert_allclose(res["var"][0], [0.0, 16.0])

    def test_variance_and_bounds_inversion(self):
        norm_mu = np.array([[1.0, 2.0]])
        norm_var = np.array([[4.0, 9.0]])  # var_norm
        norm_lo = np.array([[0.0, 1.0]])
        norm_hi = np.array([[2.0, 3.0]])
        scaler_mean = np.array([50.0, 60.0])
        scaler_std = np.array([5.0, 2.0])

        res = inverse_transform_predictions(
            mu=norm_mu,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            var=norm_var,
            lower=norm_lo,
            upper=norm_hi,
        )
        assert isinstance(res, dict)
        # var_raw = var_norm * std^2: 4*25 = 100, 9*4 = 36
        np.testing.assert_allclose(res["var"][0], [100.0, 36.0])
        # lower_raw = 0*5 + 50 = 50, 1*2 + 60 = 62
        np.testing.assert_allclose(res["lower"][0], [50.0, 62.0])

    def test_multi_array_shape_mismatch_raises(self):
        """Mismatched shapes between provided prediction arrays must be rejected."""
        mu = np.zeros((2, 12, 5, 1))
        bad_sigma = np.ones((1, 12, 5, 1))  # S=1 != 2
        scaler_mean = np.zeros(5)
        scaler_std = np.ones(5)

        with pytest.raises(MetricError, match="Shape mismatch"):
            inverse_transform_predictions(mu, bad_sigma, scaler_mean, scaler_std)

    def test_sensor_dimension_mismatch_raises(self):
        mu = np.zeros((1, 12, 5, 1))
        sigma = np.ones((1, 12, 5, 1))
        bad_scaler_mean = np.zeros(4)  # 4 != 5
        bad_scaler_std = np.ones(4)
        with pytest.raises(MetricError, match="Sensor dimension mismatch"):
            inverse_transform_predictions(mu, sigma, bad_scaler_mean, bad_scaler_std)


# ---------------------------------------------------------------------------
# Per-Horizon and Aggregate Evaluator
# ---------------------------------------------------------------------------


class TestEvaluateMetricsByHorizon:
    def test_evaluate_by_horizon_shape_and_keys(self):
        S, H, N = 4, 12, 3
        y_true = np.ones((S, H, N, 1)) * 60.0
        y_pred = np.ones((S, H, N, 1)) * 58.0
        sigma = np.ones((S, H, N, 1)) * 2.0
        mask = np.ones((S, H, N, 1))

        overall, per_horizon = evaluate_metrics_by_horizon(
            y_true, y_pred, sigma=sigma, mask=mask, nominal_pi=0.95, cadence_minutes=5
        )

        assert overall["MAE"] == pytest.approx(2.0)
        assert overall["RMSE"] == pytest.approx(2.0)
        assert overall["valid_count"] == S * H * N
        assert len(per_horizon) == 12

        for h, h_data in enumerate(per_horizon):
            assert h_data["horizon_step"] == h + 1
            assert h_data["horizon_minutes"] == (h + 1) * 5
            assert h_data["MAE"] == pytest.approx(2.0)
            assert h_data["valid_count"] == S * N

    def test_custom_cadence_minutes(self):
        S, H, N = 2, 4, 2
        y_true = np.ones((S, H, N, 1)) * 50.0
        y_pred = np.ones((S, H, N, 1)) * 50.0
        _, per_horizon = evaluate_metrics_by_horizon(
            y_true, y_pred, cadence_minutes=15
        )
        assert per_horizon[0]["horizon_minutes"] == 15
        assert per_horizon[1]["horizon_minutes"] == 30
        assert per_horizon[2]["horizon_minutes"] == 45
        assert per_horizon[3]["horizon_minutes"] == 60
