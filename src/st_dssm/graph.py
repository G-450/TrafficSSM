"""Graph operators and Chebyshev polynomial convolutions for spatial-temporal modeling.

Implements normalized graph Laplacian calculation, scaled Laplacian computation,
Chebyshev polynomial recurrence, and the PyTorch ChebConv layer per ADR-0007.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import torch
from torch import nn


class GraphError(Exception):
    """Raised for errors in graph structure or operator computations."""


def calculate_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """Calculate the symmetric normalized graph Laplacian L = I - D^(-1/2) W D^(-1/2).

    Args:
        adj: Square adjacency matrix W of shape [N, N].

    Returns:
        Normalized Laplacian matrix L of shape [N, N].
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise GraphError(f"Adjacency matrix must be square 2D array, got shape {adj.shape}")

    if not np.isfinite(adj).all():
        raise GraphError("Adjacency matrix contains non-finite values (NaN or Inf).")

    # Symmetric check/enforcement if needed
    adj = adj.astype(np.float64)
    row_sum = np.sum(adj, axis=1)

    # Invert square root of degrees
    d_inv_sqrt = np.zeros_like(row_sum)
    non_zero = row_sum > 1e-12
    d_inv_sqrt[non_zero] = 1.0 / np.sqrt(row_sum[non_zero])

    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    n = adj.shape[0]
    identity = np.eye(n, dtype=np.float64)

    # L = I - D^(-1/2) W D^(-1/2)
    normalized_laplacian = identity - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return normalized_laplacian.astype(np.float32)


def calculate_scaled_laplacian(
    laplacian: np.ndarray,
    lambda_max: float | None = None,
) -> tuple[np.ndarray, float]:
    """Scale Laplacian to [-1, 1] range: L_tilde = (2 / lambda_max) * L - I.

    Args:
        laplacian: Normalized graph Laplacian matrix L of shape [N, N].
        lambda_max: Maximum eigenvalue. If None, it is computed numerically.

    Returns:
        tuple (scaled_laplacian, computed_lambda_max)
    """
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise GraphError(f"Laplacian must be square 2D array, got shape {laplacian.shape}")

    n = laplacian.shape[0]
    if lambda_max is None:
        try:
            # Use scipy eigsh on symmetric matrix for efficiency
            eigvals = scipy.sparse.linalg.eigsh(
                sp.csr_matrix(laplacian.astype(np.float64)),
                k=1,
                which="LM",
                return_eigenvectors=False,
            )
            lambda_max = float(eigvals[0])
        except Exception:  # noqa: BLE001
            # Fallback to dense eigenvalue decomposition
            eigvals = np.linalg.eigvalsh(laplacian.astype(np.float64))
            lambda_max = float(np.max(eigvals))

    if lambda_max is None or lambda_max <= 1e-6:
        # For disconnected/zero Laplacians, default to standard upper bound lambda_max = 2.0
        lambda_max = 2.0

    identity = np.eye(n, dtype=np.float32)
    scaled_laplacian = (2.0 / lambda_max) * laplacian - identity
    return scaled_laplacian.astype(np.float32), lambda_max


def compute_chebyshev_polynomials(
    scaled_laplacian: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Compute Chebyshev polynomials basis matrices [T_0, T_1, ..., T_{k-1}] of shape [K, N, N].

    Recurrence relation:
        T_0(L_tilde) = I
        T_1(L_tilde) = L_tilde
        T_k(L_tilde) = 2 * L_tilde * T_{k-1}(L_tilde) - T_{k-2}(L_tilde) for k >= 2

    Args:
        scaled_laplacian: Scaled Laplacian L_tilde of shape [N, N].
        k: Chebyshev polynomial order K (default 3 per ADR-0007).

    Returns:
        Chebyshev basis tensor of shape [K, N, N].
    """
    if k < 1:
        raise GraphError(f"Chebyshev order K must be >= 1, got {k}")

    n = scaled_laplacian.shape[0]
    cheb_polynomials = [np.eye(n, dtype=np.float32)]

    if k > 1:
        cheb_polynomials.append(scaled_laplacian.astype(np.float32))

    for i in range(2, k):
        t_next = 2.0 * (scaled_laplacian @ cheb_polynomials[i - 1]) - cheb_polynomials[i - 2]
        cheb_polynomials.append(t_next.astype(np.float32))

    return np.stack(cheb_polynomials, axis=0)


class ChebConv(nn.Module):
    """Chebyshev Graph Convolutional layer (ChebNet).

    Computes:
        Z = sum_{k=0}^{K-1} (T_k @ X) @ W_k + bias
    where X is shaped [B, T, N, C_in] or [B, N, C_in].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"Chebyshev order K must be >= 1, got {k}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        # Parameter weights for each Chebyshev basis: shape [K, C_in, C_out]
        self.weights = nn.Parameter(torch.empty(k, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of Chebyshev convolution.

        Args:
            x: Input tensor of shape [B, T, N, C_in] or [B, N, C_in].
            cheb_polynomials: Chebyshev basis tensor of shape [K, N, N].

        Returns:
            Output tensor with channels transformed to C_out:
            [B, T, N, C_out] or [B, N, C_out].
        """
        has_time_dim = x.dim() == 4
        if not has_time_dim and x.dim() != 3:
            raise ValueError(f"Expected 3D [B, N, C] or 4D [B, T, N, C] input, got {x.shape}")

        if cheb_polynomials.dim() != 3 or cheb_polynomials.shape[0] != self.k:
            raise ValueError(
                f"Expected cheb_polynomials of shape [{self.k}, N, N], got {cheb_polynomials.shape}"
            )

        if has_time_dim:
            # x: [B, T, N, C_in]
            # T_k: [K, N, N], weights: [K, C_in, C_out]
            # output: sum_k (T_k @ x @ W_k)
            # einsum: 'knm,btmc,kco->btno'
            out = torch.einsum("knm,btmc,kco->btno", cheb_polynomials, x, self.weights)
        else:
            # x: [B, N, C_in]
            # einsum: 'knm,bmc,kco->bno'
            out = torch.einsum("knm,bmc,kco->bno", cheb_polynomials, x, self.weights)

        if self.bias is not None:
            out = out + self.bias

        return out
