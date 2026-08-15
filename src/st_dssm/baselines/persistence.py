"""Historical persistence forecasting baseline.

Implements a non-parametric baseline that repeats the last observed value
across all forecast horizons per ADR-0008.
"""

from __future__ import annotations

import numpy as np
import torch


class HistoricalPersistence:
    """Historical Persistence baseline.

    Repeats the last time-step observation across all forecast horizons H:
        hat{Y}[:, h, :, :] = X[:, -1, :, 0:1] for h in 0..H-1
    """

    def __init__(self, forecast_horizon: int = 12) -> None:
        if forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}")
        self.forecast_horizon = forecast_horizon

    def predict(
        self,
        x: np.ndarray | torch.Tensor,
        forecast_horizon: int | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Generate persistence predictions.

        Args:
            x: Input array or tensor of shape [B, L, N, C] or [L, N, C] or [B, L, N].
               Channel 0 is assumed to be the speed value.
            forecast_horizon: Optional override for forecast horizon H.

        Returns:
            Forecast of shape [B, H, N, 1] matching input type.
        """
        h = forecast_horizon or self.forecast_horizon
        if h < 1:
            raise ValueError(f"forecast_horizon must be >= 1, got {h}")

        is_torch = isinstance(x, torch.Tensor)

        if is_torch:
            return self._predict_torch(x, h)
        if isinstance(x, np.ndarray):
            return self._predict_numpy(x, h)
        raise TypeError(f"Input must be numpy.ndarray or torch.Tensor, got {type(x)}")

    def _predict_numpy(self, x: np.ndarray, h: int) -> np.ndarray:
        if x.ndim == 3:
            # [B, L, N] -> [B, L, N, 1]
            x = np.expand_dims(x, axis=-1)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D numpy array, got shape {x.shape}")

        if x.shape[1] < 1:
            raise ValueError("Input sequence length L must be >= 1.")

        # Extract last time step, first channel (value)
        last_step = x[:, -1:, :, 0:1]  # Shape: [B, 1, N, 1]
        # Repeat along horizon axis
        return np.repeat(last_step, h, axis=1)

    def _predict_torch(self, x: torch.Tensor, h: int) -> torch.Tensor:
        if x.dim() == 3:
            # [B, L, N] -> [B, L, N, 1]
            x = x.unsqueeze(-1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D torch Tensor, got shape {x.shape}")

        if x.shape[1] < 1:
            raise ValueError("Input sequence length L must be >= 1.")

        last_step = x[:, -1:, :, 0:1]  # Shape: [B, 1, N, 1]
        return last_step.repeat(1, h, 1, 1)

    def __call__(
        self,
        x: np.ndarray | torch.Tensor,
        forecast_horizon: int | None = None,
    ) -> np.ndarray | torch.Tensor:
        return self.predict(x, forecast_horizon)
