import numpy as np


class PreprocessingError(Exception):
    pass


def calculate_split_boundaries(total_samples: int) -> tuple[int, int, int]:
    """
    Calculates deterministic chronological split boundaries for a sequence.
    Follows ADR-0002: 70% train, 10% validation, 20% test.
    Rounding rule: train bound is int(total * 0.7), val bound is int(total * 0.8).
    
    Args:
        total_samples: The number of total chronological timestamps.
        
    Returns:
        tuple containing (train_end, val_end, test_end) indices.
        train goes from [0, train_end)
        val goes from [train_end, val_end)
        test goes from [val_end, test_end) (where test_end == total_samples)
    """
    if total_samples < 10:
        raise PreprocessingError(f"Insufficient samples for splitting: {total_samples}")
    
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.8)
    test_end = total_samples
    
    if train_end == 0 or (val_end - train_end) == 0 or (test_end - val_end) == 0:
        raise PreprocessingError("Partition calculation resulted in an empty split.")
        
    return train_end, val_end, test_end


class PerSensorScaler:
    """
    Per-sensor z-score standardization scaler.
    Fits only on training data. Applies frozen parameters to all partitions.
    Uses population standard deviation (ddof=0).
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False

    def fit(self, training_data: np.ndarray) -> None:
        """
        Fits mean and standard deviation per sensor.
        Args:
            training_data: Array of shape [T, N] (time, sensors).
        """
        if training_data.ndim != 2:
            raise ValueError(f"Expected 2D array [T, N], got {training_data.ndim}D")
            
        # Use population std dev
        self.mean_ = np.mean(training_data, axis=0)
        self.std_ = np.std(training_data, axis=0, ddof=0)
        
        # Explicit handling of zero or near-zero variance
        zero_variance_mask = np.isclose(self.std_, 0.0, atol=1e-7)
        self.std_[zero_variance_mask] = 1.0
        
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.std_).all():
            raise PreprocessingError("Fitted scaler parameters are non-finite.")
            
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms data using frozen fit parameters.
        Args:
            data: Array of shape [..., N] or [..., N, 1]
        """
        if not self.is_fitted:
            raise PreprocessingError("Scaler is not fitted.")
            
        # Support broadcasting to expected shapes
        sensor_axis = data.shape[-1] if data.ndim == 2 else data.shape[-2]
        if sensor_axis != len(self.mean_):
            raise ValueError(f"Sensor axis dimension mismatch: expected {len(self.mean_)}, got {sensor_axis}")

        # If data is [..., N, 1], reshape parameters to broadcast correctly
        if data.ndim > 2 and data.shape[-1] == 1:
            mean = self.mean_.reshape(*([1] * (data.ndim - 2)), len(self.mean_), 1)
            std = self.std_.reshape(*([1] * (data.ndim - 2)), len(self.std_), 1)
        elif data.ndim == 2:
            mean = self.mean_
            std = self.std_
        else:
            raise ValueError("Unsupported data shape for transformation.")

        transformed = (data - mean) / std
        return transformed

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transforms data back to original scale.
        """
        if not self.is_fitted:
            raise PreprocessingError("Scaler is not fitted.")

        sensor_axis = data.shape[-1] if data.ndim == 2 else data.shape[-2]
        if sensor_axis != len(self.mean_):
            raise ValueError(f"Sensor axis dimension mismatch: expected {len(self.mean_)}, got {sensor_axis}")

        if data.ndim > 2 and data.shape[-1] == 1:
            mean = self.mean_.reshape(*([1] * (data.ndim - 2)), len(self.mean_), 1)
            std = self.std_.reshape(*([1] * (data.ndim - 2)), len(self.std_), 1)
        elif data.ndim == 2:
            mean = self.mean_
            std = self.std_
        else:
            raise ValueError("Unsupported data shape for inverse transformation.")

        original = (data * std) + mean
        return original


def generate_windows(
    values: np.ndarray, 
    masks: np.ndarray, 
    input_length: int = 12, 
    forecast_horizon: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates rolling windows within a single split partition.
    
    Args:
        values: The [T, N] array (scaled for inputs, unscaled for targets typically, 
                but we pass the scaled array here, targets can be inverse transformed later).
        masks: The [T, N] observation mask boolean array.
        input_length: L
        forecast_horizon: H
        
    Returns:
        tuple containing (X, Y, X_mask, Y_mask)
        X: [S, L, N, 1]
        Y: [S, H, N, 1]
        X_mask: [S, L, N, 1]
        Y_mask: [S, H, N, 1]
    """
    total_time, num_sensors = values.shape
    num_samples = total_time - input_length - forecast_horizon + 1
    
    if num_samples <= 0:
        raise PreprocessingError(f"Partition too short to generate windows (T={total_time}, L={input_length}, H={forecast_horizon})")

    # Expand dims to include feature channel C=1
    values_expanded = np.expand_dims(values, axis=-1)
    masks_expanded = np.expand_dims(masks, axis=-1)
    
    # Pre-allocate arrays for canonical shapes
    X = np.empty((num_samples, input_length, num_sensors, 1), dtype=values.dtype)
    Y = np.empty((num_samples, forecast_horizon, num_sensors, 1), dtype=values.dtype)
    X_mask = np.empty((num_samples, input_length, num_sensors, 1), dtype=masks.dtype)
    Y_mask = np.empty((num_samples, forecast_horizon, num_sensors, 1), dtype=masks.dtype)
    
    # Fill sliding windows
    for i in range(num_samples):
        input_end = i + input_length
        target_end = input_end + forecast_horizon
        
        X[i] = values_expanded[i:input_end]
        Y[i] = values_expanded[input_end:target_end]
        
        X_mask[i] = masks_expanded[i:input_end]
        Y_mask[i] = masks_expanded[input_end:target_end]
        
    return X, Y, X_mask, Y_mask
