"""Unit tests for training utilities (MaskedMAELoss, EarlyStopping, checkpointing, and seeding)."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from st_dssm.training import (
    EarlyStopping,
    MaskedMAELoss,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


class TestTrainingUtilities:
    def test_masked_mae_loss_unmasked(self):
        loss_fn = MaskedMAELoss()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
        true = torch.tensor([1.5, 2.5, 2.0, 5.0]).reshape(1, 4, 1, 1)

        loss = loss_fn(pred, true)
        expected = torch.mean(torch.abs(pred - true))
        assert loss.item() == pytest.approx(expected.item())

    def test_masked_mae_loss_with_mask(self):
        loss_fn = MaskedMAELoss()
        pred = torch.tensor([1.0, 2.0, 3.0, 100.0]).reshape(1, 4, 1, 1)
        true = torch.tensor([1.5, 2.0, 4.0, 0.0]).reshape(1, 4, 1, 1)
        # Mask out the extreme error at index 3
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0]).reshape(1, 4, 1, 1)

        loss = loss_fn(pred, true, mask)
        # Expected error on first 3 elements: (0.5 + 0.0 + 1.0) / 3 = 0.5
        assert loss.item() == pytest.approx(0.5, abs=1e-4)

    def test_masked_mae_loss_all_zero_mask(self):
        loss_fn = MaskedMAELoss()
        pred = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1)
        true = torch.tensor([2.0, 3.0]).reshape(1, 2, 1, 1)
        mask = torch.zeros(1, 2, 1, 1)

        loss = loss_fn(pred, true, mask)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_early_stopping_patience_and_improvement(self):
        patience = 3
        min_delta = 1e-3
        es = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
        model = nn.Linear(4, 2)

        # Epoch 1: initial score
        stop = es.step(1.0, model, epoch=1)
        assert not stop
        assert es.best_score == 1.0
        assert es.best_epoch == 1
        assert es.counter == 0

        # Epoch 2: improvement >= 1e-3
        stop = es.step(0.99, model, epoch=2)
        assert not stop
        assert es.best_score == 0.99
        assert es.best_epoch == 2
        assert es.counter == 0

        # Epoch 3: insufficient improvement (< 1e-3)
        stop = es.step(0.9895, model, epoch=3)
        assert not stop
        assert es.counter == 1

        # Epoch 4: worse score
        stop = es.step(1.10, model, epoch=4)
        assert not stop
        assert es.counter == 2

        # Epoch 5: 3rd consecutive failed improvement -> triggers stop
        stop = es.step(1.05, model, epoch=5)
        assert stop
        assert es.counter == 3
        assert es.best_epoch == 2

    def test_early_stopping_restores_best_weights(self):
        es = EarlyStopping(patience=2, min_delta=1e-4, mode="min")
        model = nn.Linear(2, 2, bias=False)

        # Initialize weights to 1.0
        with torch.no_grad():
            model.weight.fill_(1.0)
        es.step(0.5, model, epoch=1)

        # Modify weights to 2.0 in worse epoch
        with torch.no_grad():
            model.weight.fill_(2.0)
        es.step(0.6, model, epoch=2)

        # Restore weights
        es.restore_best_weights(model)
        assert torch.allclose(model.weight, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))

    def test_early_stopping_non_finite_guard(self):
        es = EarlyStopping(patience=5)
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="Non-finite validation score"):
            es.step(float("nan"), model, epoch=1)

    def test_set_seed_reproducibility(self):
        set_seed(2026)
        r1 = torch.randn(5)
        set_seed(2026)
        r2 = torch.randn(5)
        torch.testing.assert_close(r1, r2)

    def test_save_and_load_checkpoint(self, tmp_path):
        ckpt_path = str(tmp_path / "model.pt")
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        save_checkpoint(ckpt_path, model, optimizer, epoch=5, val_loss=0.42)
        assert os.path.exists(ckpt_path)

        new_model = nn.Linear(3, 2)
        new_opt = torch.optim.Adam(new_model.parameters(), lr=0.01)
        loaded = load_checkpoint(ckpt_path, new_model, new_opt)

        assert loaded["epoch"] == 5
        assert loaded["val_loss"] == 0.42
        for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=False):
            torch.testing.assert_close(p1, p2)
