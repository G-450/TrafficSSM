import numpy as np
import pytest

from st_dssm.missingness import extract_native_missingness


def test_extract_native_missingness_success():
    values = np.array([
        [1.0, 0.0, np.nan],
        [2.0, 3.0, 0.0]
    ])
    
    values_with_nan, observed_mask, stats = extract_native_missingness(values)
    
    # Check values_with_nan replaced 0.0 with np.nan
    assert np.isnan(values_with_nan[0, 1])
    assert np.isnan(values_with_nan[0, 2])
    assert np.isnan(values_with_nan[1, 2])
    assert values_with_nan[0, 0] == 1.0
    assert values_with_nan[1, 0] == 2.0
    assert values_with_nan[1, 1] == 3.0
    
    # Check observed mask
    expected_mask = np.array([
        [True, False, False],
        [True, True, False]
    ])
    np.testing.assert_array_equal(observed_mask, expected_mask)
    
    # Check stats
    assert stats.total_count == 6
    assert stats.observed_count == 3
    assert stats.missing_count == 3
    assert stats.zero_count == 2
    assert stats.nan_count == 1
    assert stats.invalid_infinity_count == 0
    np.testing.assert_array_equal(stats.per_sensor_missing, np.array([0, 1, 2]))


def test_extract_native_missingness_infinity_rejected():
    values = np.array([
        [1.0, np.inf, -np.inf],
    ])
    
    _values_with_nan, observed_mask, stats = extract_native_missingness(values)
    
    # Infinities are NOT observed, and NOT native missing
    expected_mask = np.array([[True, False, False]])
    np.testing.assert_array_equal(observed_mask, expected_mask)
    
    # Verify infinities are replaced by NaN in the values array
    assert np.isnan(_values_with_nan[0, 1])
    assert np.isnan(_values_with_nan[0, 2])
    
    assert stats.invalid_infinity_count == 2
    assert stats.missing_count == 0
    assert stats.observed_count == 1


def test_extract_native_missingness_non_numeric():
    values = np.array([["1.0", "0.0"]])
    with pytest.raises(ValueError, match="numeric"):
        extract_native_missingness(values)
