"""Training utilities, loss functions, seed management, and early stopping.

Implements the deterministic training loop, MaskedMAELoss, and early stopping
governed by ADR-0008 (15-epoch patience, 1e-4 min_delta, validation MAE selection).
"""

from __future__ import annotations

import os
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn, optim


def set_seed(seed: int = 2026) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for exact reproducibility.

    Canonical project seeds: 2026, 2027, 2028 (per ADR-0008).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MaskedMAELoss(nn.Module):
    """Masked Mean Absolute Error loss.

    Computes MAE exclusively over observed target values where mask == 1:
        Loss = sum(|y_pred - y_true| * mask) / sum(mask)
    Returns NaN if no valid targets are observed (sum(mask) == 0) to prevent false zero loss.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked MAE.

        Args:
            y_pred: Predicted values [B, H, N, 1].
            y_true: Ground truth target values [B, H, N, 1].
            mask: Binary observation mask [B, H, N, 1]. If None, all elements are used.

        Returns:
            Scalar loss tensor (NaN if mask has zero valid entries).
        """
        diff = torch.abs(y_pred - y_true)
        if mask is not None:
            mask = mask.to(dtype=diff.dtype, device=diff.device)
            valid_count = torch.sum(mask)
            if valid_count <= 0:
                return torch.tensor(float("nan"), dtype=diff.dtype, device=diff.device)
            return torch.sum(diff * mask) / valid_count

        return torch.mean(diff)


class EarlyStopping:
    """Early stopping handler adhering to ADR-0008.

    Rule: Stop after 15 consecutive epochs without an improvement of at least
    1e-4 in the selection metric (validation MAE); always retain the best checkpoint.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.best_state: dict[str, Any] | None = None
        self.best_epoch: int = 0

    def step(self, current_score: float, model: nn.Module, epoch: int) -> bool:
        """Evaluate current epoch score and determine whether to stop.

        Args:
            current_score: Metric value (e.g. validation MAE).
            model: Current PyTorch model to snapshot best weights.
            epoch: Current epoch index (1-based).

        Returns:
            True if training should stop, False otherwise.
        """
        if not np.isfinite(current_score):
            raise ValueError(f"Non-finite validation score encountered: {current_score}")

        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self._save_checkpoint(model)
            return False

        if self.mode == "min":
            improvement = self.best_score - current_score
        else:
            improvement = current_score - self.best_score

        if improvement >= self.min_delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _save_checkpoint(self, model: nn.Module) -> None:
        self.best_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best snapshot weights into the given model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    epoch: int = 0,
    val_loss: float = 0.0,
    config: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint safely using atomic temporary file replacement."""
    dir_name = os.path.dirname(checkpoint_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config or {},
    }

    fd, tmp_path = tempfile.mkstemp(dir=dir_name or None, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load training checkpoint into model securely."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
