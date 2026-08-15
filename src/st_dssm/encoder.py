"""Canonical Spatial-Temporal Encoder module for ST-DSSM.

Implements the 2-block ST-GCN encoder with Causal Gated Temporal Convolutions,
Chebyshev Graph Convolutions (K=3), LayerNorm, Dropout, and Context Projection
per ADR-0007.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from st_dssm.graph import ChebConv


class CausalGatedTemporalConv(nn.Module):
    """Causal Gated Temporal Convolution (GLU).

    Applies 1D temporal convolution with left-padding to strictly preserve causality
    and prevent future information leakage:
        Output = Conv1D_filter(X) * sigmoid(Conv1D_gate(X))

    Shape:
        Input:  [B, T, N, C_in]
        Output: [B, T, N, C_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        # Generates 2 * out_channels for GLU (split into filter and gate)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=(kernel_size, 1),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, N, C_in].

        Returns:
            Output tensor [B, T, N, C_out].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, T, N, C], got shape {x.shape}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.in_channels}, got {x.shape[-1]}"
            )

        # Permute to [B, C_in, T, N] for 2D convolution
        x_perm = x.permute(0, 3, 1, 2)

        # Causal left padding on time dimension: (pad_left_N, pad_right_N, pad_left_T, pad_right_T)
        if self.padding > 0:
            x_perm = F.pad(x_perm, (0, 0, self.padding, 0))

        # Conv2d output: [B, 2 * out_channels, T, N]
        conv_out = self.conv(x_perm)

        # Split into filter and gate activations along channel axis
        p, q = torch.chunk(conv_out, 2, dim=1)

        # Gated activation: P * sigmoid(Q)
        gated = p * torch.sigmoid(q)
        gated = self.dropout(gated)

        # Permute back to [B, T, N, C_out]
        return gated.permute(0, 2, 3, 1)


class SpatialTemporalBlock(nn.Module):
    """Canonical Spatial-Temporal Residual Block.

    Structure per ADR-0007:
        1. Causal Gated Temporal Conv (K_t=3)
        2. Chebyshev Graph Conv (K=3)
        3. Causal Gated Temporal Conv (K_t=3)
        4. Layer Normalization, Dropout (0.1), and Residual Connection.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        kernel_size: int = 3,
        cheb_k: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes

        # 1. Temporal Conv 1
        self.tconv1 = CausalGatedTemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2. Chebyshev Graph Conv
        self.gconv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            k=cheb_k,
            bias=True,
        )
        self.relu = nn.ReLU()

        # 3. Temporal Conv 2
        self.tconv2 = CausalGatedTemporalConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 4. Normalization and Dropout
        self.norm = nn.LayerNorm([num_nodes, hidden_channels])
        self.dropout = nn.Dropout(dropout)

        # Residual shortcut projection when input/output channels differ
        if in_channels != hidden_channels:
            self.residual = nn.Linear(in_channels, hidden_channels)
        else:
            self.residual = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through spatial-temporal block.

        Args:
            x: Input tensor [B, T, N, C_in].
            cheb_polynomials: Chebyshev basis tensor [K, N, N].

        Returns:
            Output tensor [B, T, N, hidden_channels].
        """
        res = self.residual(x)

        # 1. First temporal convolution
        h = self.tconv1(x)  # [B, T, N, hidden_channels]

        # 2. Spatial graph convolution + non-linearity
        h = self.gconv(h, cheb_polynomials)  # [B, T, N, hidden_channels]
        h = self.relu(h)

        # 3. Second temporal convolution
        h = self.tconv2(h)  # [B, T, N, hidden_channels]

        # 4. Residual addition + LayerNorm + Dropout
        h = self.norm(h + res)
        h = self.dropout(h)
        return h


class SpatialTemporalEncoder(nn.Module):
    """Canonical Spatial-Temporal Encoder for ST-DSSM.

    Encodes input historical traffic observations and binary observation masks
    [B, L=12, N=325, 2] into rich spatial-temporal context representations [B, L, N, 64]
    per ADR-0007.

    Architecture:
        - 2 SpatialTemporalBlock modules (64 channels throughout)
        - Final context projection linear layer (64 -> 64)
        - Strict temporal causality and input validation guards.
    """

    def __init__(
        self,
        num_nodes: int = 325,
        in_channels: int = 2,
        hidden_channels: int = 64,
        input_length: int = 12,
        cheb_k: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.input_length = input_length
        self.cheb_k = cheb_k

        # 2 Spatial-Temporal Residual Blocks
        self.block1 = SpatialTemporalBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_nodes=num_nodes,
            kernel_size=kernel_size,
            cheb_k=cheb_k,
            dropout=dropout,
        )
        self.block2 = SpatialTemporalBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_nodes=num_nodes,
            kernel_size=kernel_size,
            cheb_k=cheb_k,
            dropout=dropout,
        )

        # Final context projection per node and time step
        self.context_proj = nn.Linear(hidden_channels, hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
    ) -> torch.Tensor:
        """Encode historical traffic observations into spatial-temporal context.

        Args:
            x: Input tensor of shape [B, L=12, N=325, C=2]
               Channel 0: normalized speed observations.
               Channel 1: binary observation mask.
            cheb_polynomials: Chebyshev basis tensor of shape [K=3, N=325, N=325].

        Returns:
            Context tensor of shape [B, L=12, N=325, 64].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor [B, L, N, C], got shape {x.shape}")

        _b, l_in, n, c = x.shape
        if l_in != self.input_length:
            raise ValueError(f"Input length mismatch: expected {self.input_length}, got {l_in}")
        if n != self.num_nodes:
            raise ValueError(f"Number of nodes mismatch: expected {self.num_nodes}, got {n}")
        if c != self.in_channels:
            raise ValueError(f"Input channel mismatch: expected {self.in_channels}, got {c}")

        if (
            cheb_polynomials.shape[0] != self.cheb_k
            or cheb_polynomials.shape[1] != n
            or cheb_polynomials.shape[2] != n
        ):
            raise ValueError(
                f"Chebyshev basis mismatch: expected [{self.cheb_k}, {n}, {n}], got {cheb_polynomials.shape}"
            )

        # Finite input guard
        if not torch.isfinite(x).all():
            raise ValueError("Input tensor x contains non-finite values (NaN or Inf).")

        # Pass through 2 spatial-temporal blocks
        h1 = self.block1(x, cheb_polynomials)  # [B, L, N, 64]
        h2 = self.block2(h1, cheb_polynomials)  # [B, L, N, 64]

        # Final context projection
        context = self.context_proj(h2)  # [B, L, N, 64]
        return context

    def get_context_summary(self, context: torch.Tensor) -> torch.Tensor:
        """Extract sequence summary context at the final time step t = L - 1.

        Args:
            context: Spatial-temporal context tensor [B, L, N, 64].

        Returns:
            Summary context tensor [B, N, 64].
        """
        if context.dim() != 4 or context.shape[1] != self.input_length:
            raise ValueError(
                f"Expected context shape [B, {self.input_length}, N, 64], got {context.shape}"
            )
        return context[:, -1, :, :]  # [B, N, 64]


def count_trainable_parameters(model: nn.Module) -> int:
    """Calculate total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_encoder_capacity_report(
    encoder: nn.Module,
    reference_param_count: int | None = None,
) -> dict[str, Any]:
    """Generate capacity and parameter report for SpatialTemporalEncoder per ADR-0008.

    Args:
        encoder: SpatialTemporalEncoder instance.
        reference_param_count: Optional reference parameter count for capacity ratio check.

    Returns:
        Dictionary report containing parameter counts and architectural metadata.
    """
    trainable_params = count_trainable_parameters(encoder)
    total_params = sum(p.numel() for p in encoder.parameters())

    report: dict[str, Any] = {
        "model_name": "spatial_temporal_encoder",
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "num_nodes": getattr(encoder, "num_nodes", None),
        "in_channels": getattr(encoder, "in_channels", None),
        "hidden_channels": getattr(encoder, "hidden_channels", None),
        "input_length": getattr(encoder, "input_length", None),
        "cheb_k": getattr(encoder, "cheb_k", None),
    }

    if reference_param_count is not None and reference_param_count > 0:
        ratio = trainable_params / reference_param_count
        report["capacity_ratio_vs_reference"] = float(ratio)
        report["capacity_ratio_valid"] = 0.5 <= ratio <= 2.0
    else:
        report["capacity_ratio_vs_reference"] = None
        report["capacity_ratio_valid"] = False

    return report
