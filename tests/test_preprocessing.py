import numpy as np
import pytest
from st_dssm.preprocessing import calculate_split_boundaries, PerSensorScaler, generate_windows, PreprocessingError

def test_calculate_split_boundaries_success():
    train_end, val_end, test_end = calculate_split_boundaries(100)
    assert train_end == 70
    assert val_end == 80
    assert test_end == 100

def test_calculate_split_boundaries_insufficient():
    with pytest.raises(PreprocessingError, match="Insufficient samples"):
        calculate_split_boundaries(9)


def test_per_sensor_scaler_fit_transform():
    scaler = PerSensorScaler()
    data = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    scaler.fit(data)
    
    assert scaler.mean_.shape == (2,)
    assert scaler.std_.shape == (2,)
    
    expected_mean = np.array([3.0, 4.0])
    # population std dev of [1,3,5] is sqrt(((1-3)**2 + (3-3)**2 + (5-3)**2)/3) = sqrt(8/3) ~ 1.63299
    expected_std = np.array([np.sqrt(8/3), np.sqrt(8/3)])
    
    np.testing.assert_allclose(scaler.mean_, expected_mean)
    np.testing.assert_allclose(scaler.std_, expected_std)
    
    transformed = scaler.transform(data)
    expected_transformed = (data - expected_mean) / expected_std
    np.testing.assert_allclose(transformed, expected_transformed)
    
    inverse = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(inverse, data)

def test_per_sensor_scaler_zero_variance():
    scaler = PerSensorScaler()
    data = np.array([
        [1.0, 2.0],
        [1.0, 4.0],
        [1.0, 6.0]
    ])
    scaler.fit(data)
    
    assert scaler.mean_[0] == 1.0
    assert scaler.std_[0] == 1.0 # explicitly handled
    
    transformed = scaler.transform(data)
    assert np.all(transformed[:, 0] == 0.0)
    
    inverse = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(inverse, data)

def test_generate_windows_success():
    T = 20
    N = 3
    L = 5
    H = 3
    S = T - L - H + 1 # 20 - 5 - 3 + 1 = 13
    
    values = np.arange(T * N).reshape((T, N))
    masks = np.ones((T, N), dtype=bool)
    masks[1, 0] = False
    
    X, Y, X_mask, Y_mask = generate_windows(values, masks, input_length=L, forecast_horizon=H)
    
    assert X.shape == (S, L, N, 1)
    assert Y.shape == (S, H, N, 1)
    assert X_mask.shape == (S, L, N, 1)
    assert Y_mask.shape == (S, H, N, 1)
    
    # Check first window
    np.testing.assert_array_equal(X[0, :, :, 0], values[0:5])
    np.testing.assert_array_equal(Y[0, :, :, 0], values[5:8])
    np.testing.assert_array_equal(X_mask[0, :, :, 0], masks[0:5])
    
    # Check last window
    np.testing.assert_array_equal(X[-1, :, :, 0], values[-8:-3]) # values[12:17]
    np.testing.assert_array_equal(Y[-1, :, :, 0], values[-3:])  # values[17:20]

def test_generate_windows_too_short():
    values = np.zeros((10, 2))
    masks = np.ones((10, 2), dtype=bool)
    with pytest.raises(PreprocessingError, match="Partition too short"):
        generate_windows(values, masks, input_length=12, forecast_horizon=12)
