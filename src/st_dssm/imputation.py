import numpy as np


class ImputationError(Exception):
    pass


def fit_fallback_statistics(training_values: np.ndarray) -> np.ndarray:
    """
    Fits per-sensor fallback statistics using strictly training data.
    The fallback is used for leading gaps before any observation.
    
    Args:
        training_values: The values array [T_train, N] with np.nan for missing
        
    Returns:
        fallback_stats: Array of shape [N] with the mean for each sensor.
    """
    if not np.issubdtype(training_values.dtype, np.number):
        raise ValueError("Values must be numeric.")
        
    import warnings
    # Ignore NaNs when computing the mean
    # If a sensor is entirely NaN in training, np.nanmean returns NaN, which we can catch later
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fallback = np.nanmean(training_values, axis=0)
        
    if np.isnan(fallback).any():
        raise ImputationError("Some sensors have no valid observations in the training set for fallback statistics.")
        
    return fallback


def apply_causal_forward_fill(values: np.ndarray, fallback_stats: np.ndarray) -> np.ndarray:
    """
    Applies causal forward filling to a dataset.
    
    Rules:
    1. Fill only from past observations.
    2. No backward filling.
    3. Leading gaps use explicit fallback statistics.
    4. Must not mutate original array.
    5. Fails if any values remain unresolved (e.g. if fallback_stats is incomplete).
    
    Args:
        values: The values array [T, N] with np.nan for missing
        fallback_stats: Pre-computed fallback statistics [N] strictly from training data
        
    Returns:
        repaired_values: The repaired array with no missing values.
    """
    if values.shape[1] != fallback_stats.shape[0]:
        raise ValueError("Fallback stats must match the number of sensors.")
        
    repaired = np.copy(values)
    _T, _N = repaired.shape
    
    # Forward fill using pandas for convenience, as it supports causal ffill
    import pandas as pd
    df = pd.DataFrame(repaired)
    
    # ffill() strictly uses previous values
    df_filled = df.ffill()
    
    # Fill remaining leading gaps with fallback
    df_filled = df_filled.fillna(pd.Series(fallback_stats))
    
    repaired_filled = df_filled.values
    
    # Verify no missing remaining
    if np.isnan(repaired_filled).any():
        raise ImputationError("Unresolved missing values remain after causal repair and fallback.")
        
    return repaired_filled
