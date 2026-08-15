"""Unit tests for baseline models (Historical Persistence and Deterministic ST-GCN)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from st_dssm.baselines.persistence import HistoricalPersistence
from st_dssm.baselines.st_gcn import (
    CausalGatedTemporalConv,
    DeterministicSTGCN,
    STGCNBlock,
    count_trainable_parameters,
    get_capacity_report,
)
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
)
from st_dssm.training import MaskedMAELoss, set_seed


class TestHistoricalPersistence:
    def test_numpy_prediction_4d(self):
        s, l, n, c = 4, 12, 5, 1
        x = np.random.normal(0, 1, (s, l, n, c)).astype(np.float32)
        persistence = HistoricalPersistence(forecast_horizon=12)

        pred = persistence.predict(x)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (s, 12, n, 1)

        # Each horizon must exactly equal the last historical step
        for h in range(12):
            np.testing.assert_allclose(pred[:, h, :, :], x[:, -1, :, :])

    def test_numpy_prediction_3d(self):
        s, l, n = 3, 10, 4
        x = np.random.normal(0, 1, (s, l, n)).astype(np.float32)
        persistence = HistoricalPersistence(forecast_horizon=6)

        pred = persistence.predict(x)
        assert pred.shape == (s, 6, n, 1)
        for h in range(6):
            np.testing.assert_allclose(pred[:, h, :, 0], x[:, -1, :])

    def test_torch_prediction(self):
        s, l, n, c = 2, 8, 3, 2
        x = torch.randn(s, l, n, c)
        persistence = HistoricalPersistence(forecast_horizon=4)

        pred = persistence(x)
        assert isinstance(pred, torch.Tensor)
        assert pred.shape == (s, 4, n, 1)
        # Should extract channel 0
        for h in range(4):
            torch.testing.assert_close(pred[:, h, :, 0], x[:, -1, :, 0])

    def test_invalid_input(self):
        persistence = HistoricalPersistence(forecast_horizon=12)
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            persistence.predict(np.ones((5, 5)))

        with pytest.raises(ValueError, match="sequence length L must be >= 1"):
            persistence.predict(np.zeros((2, 0, 3, 1)))

        with pytest.raises(TypeError, match="must be numpy.ndarray or torch.Tensor"):
            persistence.predict([1, 2, 3])  # type: ignore


class TestCausalGatedTemporalConv:
    def test_strict_causality_no_future_leakage(self):
        """Verify that altering future time steps in X does NOT affect past/present outputs."""
        b, t, n, c_in, c_out = 1, 6, 2, 2, 4
        conv = CausalGatedTemporalConv(in_channels=c_in, out_channels=c_out, kernel_size=3)
        conv.eval()

        x1 = torch.randn(b, t, n, c_in)
        # Create x2 identical to x1 up to time step t=3, but modified at t=4, 5
        x2 = x1.clone()
        x2[:, 4:, :, :] = torch.randn(b, 2, n, c_in) * 100.0

        with torch.no_grad():
            out1 = conv(x1)
            out2 = conv(x2)

        # Outputs up to index 3 must be exactly identical
        torch.testing.assert_close(out1[:, :4, :, :], out2[:, :4, :, :])


class TestSTGCNBlock:
    def test_block_forward_and_residual(self):
        b, t, n, c_in, hidden = 2, 12, 4, 2, 16
        adj = np.eye(n, dtype=np.float32)
        norm_lap = calculate_normalized_laplacian(adj)
        scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
        cheb_poly = torch.tensor(compute_chebyshev_polynomials(scaled_lap, k=3), dtype=torch.float32)

        block = STGCNBlock(in_channels=c_in, hidden_channels=hidden, kernel_size=3, cheb_k=3)
        x = torch.randn(b, t, n, c_in)

        out = block(x, cheb_poly)
        assert out.shape == (b, t, n, hidden)
        assert torch.isfinite(out).all()


class TestDeterministicSTGCN:
    def test_full_forward_and_backward(self):
        b, l, h, n, c_in, hidden = 2, 12, 12, 6, 2, 32
        adj = np.eye(n, dtype=np.float32)
        for i in range(n - 1):
            adj[i, i + 1] = 0.5
            adj[i + 1, i] = 0.5

        norm_lap = calculate_normalized_laplacian(adj)
        scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
        cheb_poly = torch.tensor(compute_chebyshev_polynomials(scaled_lap, k=3), dtype=torch.float32)

        model = DeterministicSTGCN(
            num_nodes=n,
            in_channels=c_in,
            hidden_channels=hidden,
            input_length=l,
            forecast_horizon=h,
            cheb_k=3,
        )

        x = torch.randn(b, l, n, c_in, requires_grad=True)
        pred = model(x, cheb_poly)

        assert pred.shape == (b, h, n, 1)

        loss = pred.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for parameter {name}"

    def test_capacity_report(self):
        model = DeterministicSTGCN(
            num_nodes=325,
            in_channels=2,
            hidden_channels=64,
            input_length=12,
            forecast_horizon=12,
        )
        n_params = count_trainable_parameters(model)
        assert n_params > 10_000

        report = get_capacity_report(model, "ST-GCN", reference_param_count=n_params)
        assert report["capacity_ratio_valid"] is True
        assert report["capacity_ratio_vs_reference"] == pytest.approx(1.0)

    def test_overfit_one_batch(self):
        """Sanity test verifying model optimization capacity on a single fixed batch."""
        set_seed(42)
        b, l, h, n = 2, 12, 12, 4
        adj = np.eye(n, dtype=np.float32)
        norm_lap = calculate_normalized_laplacian(adj)
        scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
        cheb_poly = torch.tensor(compute_chebyshev_polynomials(scaled_lap, k=3), dtype=torch.float32)

        model = DeterministicSTGCN(
            num_nodes=n,
            in_channels=2,
            hidden_channels=16,
            input_length=l,
            forecast_horizon=h,
            cheb_k=3,
            dropout=0.0,
        )

        x = torch.randn(b, l, n, 2)
        y = torch.randn(b, h, n, 1)
        mask = torch.ones(b, h, n, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = MaskedMAELoss()

        initial_loss = None
        final_loss = None

        for step in range(60):
            optimizer.zero_grad()
            pred = model(x, cheb_poly)
            loss = criterion(pred, y, mask)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert initial_loss is not None and final_loss is not None
        assert final_loss < initial_loss * 0.2, f"Expected overfit: initial {initial_loss:.4f} -> final {final_loss:.4f}"
