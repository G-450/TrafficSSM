import numpy as np
import pytest

from st_dssm.imputation import (
    ImputationError,
    apply_causal_forward_fill,
    fit_fallback_statistics,
)


def test_fit_fallback_statistics_success():
    training = np.array([
        [1.0, 2.0],
        [3.0, np.nan],
        [5.0, 4.0]
    ])
    fallback = fit_fallback_statistics(training)
    np.testing.assert_array_equal(fallback, np.array([3.0, 3.0]))


def test_fit_fallback_statistics_all_nan_fails():
    training = np.array([
        [1.0, np.nan],
        [3.0, np.nan]
    ])
    with pytest.raises(ImputationError, match="no valid observations"):
        fit_fallback_statistics(training)


def test_apply_causal_forward_fill_success():
    values = np.array([
        [np.nan, 2.0],
        [1.0, np.nan],
        [np.nan, np.nan]
    ])
    fallback = np.array([0.0, 0.0])
    
    repaired = apply_causal_forward_fill(values, fallback)
    
    expected = np.array([
        [0.0, 2.0], # sensor 0 used fallback, sensor 1 used observation
        [1.0, 2.0], # sensor 0 observed, sensor 1 ffilled
        [1.0, 2.0]  # both ffilled
    ])
    np.testing.assert_array_equal(repaired, expected)
    
    # Original array should not be mutated
    assert np.isnan(values[0, 0])


def test_apply_causal_forward_fill_unresolved_fails():
    values = np.array([
        [np.nan, 2.0],
        [1.0, 3.0]
    ])
    fallback = np.array([np.nan, 0.0]) # Missing fallback for sensor 0
    
    with pytest.raises(ImputationError, match="Unresolved missing values remain"):
        apply_causal_forward_fill(values, fallback)


def test_apply_causal_forward_fill_wrong_fallback_shape():
    values = np.array([[1.0, 2.0]])
    fallback = np.array([1.0])
    with pytest.raises(ValueError, match="Fallback stats must match"):
        apply_causal_forward_fill(values, fallback)
