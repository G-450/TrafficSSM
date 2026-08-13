import numpy as np
import pandas as pd


class DataValidationError(Exception):
    pass


def validate_time_series(df: pd.DataFrame, expected_sensors: list[str]) -> None:
    """
    Validates the time-series dataframe according to the data contract.
    """
    # Check dimensionality
    expected_sensor_count = len(expected_sensors)
    if df.shape[1] != expected_sensor_count:
        raise DataValidationError(f"Expected {expected_sensor_count} sensors, got {df.shape[1]}")
    
    # Check sensor IDs match expected exactly
    actual_sensors = df.columns.astype(str).tolist()
    if actual_sensors != expected_sensors:
        raise DataValidationError("Time-series sensor IDs do not exactly match expected sensors in order.")
    
    # Check timestamps (index)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("Time-series index must be a DatetimeIndex.")
    
    if df.index.tz is None:
        raise DataValidationError("Time-series index must have an explicit timezone (or explicitly UTC).")
    
    if not df.index.is_monotonic_increasing:
        raise DataValidationError("Time-series timestamps are not strictly increasing.")
    
    if df.index.has_duplicates:
        raise DataValidationError("Time-series timestamps contain duplicates.")
    
    # Check five-minute cadence
    time_diffs = df.index.to_series().diff().dropna()
    expected_diff = pd.Timedelta(minutes=5)
    if not (time_diffs == expected_diff).all():
        bad_diffs = time_diffs[time_diffs != expected_diff]
        raise DataValidationError(f"Time-series contains non-five-minute intervals. E.g., at {bad_diffs.index[0]}")
    
    # Check data types and values
    if not pd.api.types.is_numeric_dtype(df.values):
        raise DataValidationError("Time-series values must be numeric.")
    
    if np.isinf(df.values).any():
        raise DataValidationError("Time-series contains positive or negative infinity.")


def validate_graph(adj_mx: np.ndarray, graph_sensor_ids: list[str], ts_sensor_ids: list[str]) -> None:
    """
    Validates the adjacency matrix and graph sensor alignment.
    """
    # Check adjacency matrix shape
    sensor_count = len(ts_sensor_ids)
    if adj_mx.shape != (sensor_count, sensor_count):
        raise DataValidationError(f"Expected adjacency matrix shape ({sensor_count}, {sensor_count}), got {adj_mx.shape}")
    
    # Check numeric and finite
    if not np.issubdtype(adj_mx.dtype, np.number):
        raise DataValidationError("Adjacency matrix must be numeric.")
    
    if not np.isfinite(adj_mx).all():
        raise DataValidationError("Adjacency matrix contains non-finite values.")
    
    # Check uniqueness
    if len(set(graph_sensor_ids)) != len(graph_sensor_ids):
        raise DataValidationError("Graph sensor IDs contain duplicates.")
    
    # Check alignment with time series
    if graph_sensor_ids != ts_sensor_ids:
        raise DataValidationError("Graph sensor IDs do not exactly match time-series sensor IDs in order.")
