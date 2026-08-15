"""Unit tests for graph Laplacian operators and Chebyshev graph convolutions (src/st_dssm/graph.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from st_dssm.graph import (
    ChebConv,
    GraphError,
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
)


class TestGraphOperators:
    def test_normalized_laplacian_identity_graph(self):
        # 3 isolated nodes with self-loops
        adj = np.eye(3, dtype=np.float32)
        lap = calculate_normalized_laplacian(adj)
        # For self-loops with degree 1: D^(-1/2) W D^(-1/2) = I -> L = I - I = 0
        np.testing.assert_allclose(lap, np.zeros((3, 3)), atol=1e-6)

    def test_normalized_laplacian_path_graph(self):
        # Path graph 0 - 1 - 2
        adj = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        lap = calculate_normalized_laplacian(adj)
        assert lap.shape == (3, 3)
        assert np.isfinite(lap).all()

        # Diagonal entries should be 1.0 for connected nodes
        assert np.allclose(np.diag(lap), 1.0)
        # Symmetric check
        np.testing.assert_allclose(lap, lap.T, atol=1e-6)

        # Eigenvalues must be in [0, 2]
        eigvals = np.linalg.eigvalsh(lap)
        assert np.all(eigvals >= -1e-6)
        assert np.all(eigvals <= 2.0 + 1e-6)

    def test_normalized_laplacian_invalid_inputs(self):
        with pytest.raises(GraphError, match="must be square 2D array"):
            calculate_normalized_laplacian(np.ones((3, 4)))

        with pytest.raises(GraphError, match="non-finite"):
            calculate_normalized_laplacian(np.array([[1.0, np.nan], [0.0, 1.0]]))

    def test_scaled_laplacian_bounds(self):
        adj = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        lap = calculate_normalized_laplacian(adj)
        scaled_lap, lambda_max = calculate_scaled_laplacian(lap)

        assert lambda_max > 0.0
        # Scaled laplacian eigenvalues must be in [-1, 1]
        scaled_eigvals = np.linalg.eigvalsh(scaled_lap)
        assert np.all(scaled_eigvals >= -1.0 - 1e-5)
        assert np.all(scaled_eigvals <= 1.0 + 1e-5)

    def test_chebyshev_polynomial_recurrence(self):
        adj = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        lap = calculate_normalized_laplacian(adj)
        scaled_lap, _ = calculate_scaled_laplacian(lap)

        k = 4
        cheb = compute_chebyshev_polynomials(scaled_lap, k=k)
        assert cheb.shape == (k, 3, 3)

        # T_0 = I
        np.testing.assert_allclose(cheb[0], np.eye(3), atol=1e-6)
        # T_1 = L_tilde
        np.testing.assert_allclose(cheb[1], scaled_lap, atol=1e-6)
        # T_2 = 2 * L_tilde^2 - I
        t2_expected = 2.0 * (scaled_lap @ scaled_lap) - np.eye(3)
        np.testing.assert_allclose(cheb[2], t2_expected, atol=1e-5)
        # T_3 = 2 * L_tilde * T_2 - T_1
        t3_expected = 2.0 * (scaled_lap @ t2_expected) - scaled_lap
        np.testing.assert_allclose(cheb[3], t3_expected, atol=1e-5)

    def test_chebyshev_invalid_k(self):
        with pytest.raises(GraphError, match="must be >= 1"):
            compute_chebyshev_polynomials(np.eye(3), k=0)


class TestChebConv:
    def test_chebconv_forward_4d(self):
        b, t, n, c_in, c_out, k = 2, 4, 5, 3, 8, 3
        conv = ChebConv(in_channels=c_in, out_channels=c_out, k=k)

        x = torch.randn(b, t, n, c_in)
        cheb = torch.randn(k, n, n)

        out = conv(x, cheb)
        assert out.shape == (b, t, n, c_out)
        assert torch.isfinite(out).all()

    def test_chebconv_forward_3d(self):
        b, n, c_in, c_out, k = 2, 5, 3, 8, 3
        conv = ChebConv(in_channels=c_in, out_channels=c_out, k=k, bias=False)

        x = torch.randn(b, n, c_in)
        cheb = torch.randn(k, n, n)

        out = conv(x, cheb)
        assert out.shape == (b, n, c_out)
        assert conv.bias is None

    def test_chebconv_backward(self):
        b, t, n, c_in, c_out, k = 2, 3, 4, 2, 4, 3
        conv = ChebConv(in_channels=c_in, out_channels=c_out, k=k)

        x = torch.randn(b, t, n, c_in, requires_grad=True)
        cheb = torch.randn(k, n, n)

        out = conv(x, cheb)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert conv.weights.grad is not None
        assert conv.bias.grad is not None
