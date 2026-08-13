from dataclasses import dataclass

import numpy as np


@dataclass
class MissingnessStats:
    total_count: int
    observed_count: int
    missing_count: int
    zero_count: int
    nan_count: int
    invalid_infinity_count: int
    per_sensor_missing: np.ndarray


def extract_native_missingness(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, MissingnessStats]:
    """
    Extracts native missingness from the raw values array.
    Dataset-native missingness is defined as 0.0 or NaN.
    
    Args:
        values: The raw values array [T, N]
        
    Returns:
        tuple containing:
            - values_with_nan: The values array with 0.0 replaced by np.nan
            - native_mask: The observation mask (True=observed, False=missing)
            - stats: MissingnessStats
    """
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Values must be numeric.")
        
    invalid_infinity_mask = np.isinf(values)
    invalid_infinity_count = int(invalid_infinity_mask.sum())
    
    # 0.0 and NaN are native missing
    zero_mask = (values == 0.0)
    nan_mask = np.isnan(values)
    missing_mask = zero_mask | nan_mask
    
    observed_mask = ~missing_mask & ~invalid_infinity_mask
    
    stats = MissingnessStats(
        total_count=int(values.size),
        observed_count=int(observed_mask.sum()),
        missing_count=int(missing_mask.sum()),
        zero_count=int(zero_mask.sum()),
        nan_count=int(nan_mask.sum()),
        invalid_infinity_count=invalid_infinity_count,
        per_sensor_missing=missing_mask.sum(axis=0),
    )
    
    # Replace 0.0 with NaN for standard downstream processing
    values_with_nan = np.copy(values)
    values_with_nan[zero_mask] = np.nan
    
    if values_with_nan.shape != observed_mask.shape:
        raise ValueError(f"Mask shape {observed_mask.shape} does not match values shape {values_with_nan.shape}")
        
    return values_with_nan, observed_mask, stats
