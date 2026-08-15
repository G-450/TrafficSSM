"""Deterministic Spatial-Temporal Graph Convolutional Network (ST-GCN) Baseline.

Implements the capacity-controlled deterministic graph baseline per ADR-0007 and ADR-0008:
- Input shape: [B, L, N, 2] (normalized value channel and binary observation mask).
- 2 Causal Spatial-Temporal Residual Blocks with 64 hidden channels throughout.
- Causal Gated Temporal Convolutions (kernel size 3, left padding 2).
- Chebyshev Graph Convolutions of order K=3 over scaled Laplacian.
- Layer Normalization, Dropout (0.1), and residual projections.
- Direct 12-horizon linear point-forecast head trained with MAE.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from st_dssm.graph import ChebConv


class CausalGatedTemporalConv(nn.Module):
    """Causal Gated Temporal Convolution (Gated TCN).

    Preserves temporal length L using left-padding of (kernel_size - 1) only,
    preventing any future target leakage.

    Activation:
        out = tanh(P) * sigmoid(Q)
    where P and Q are the two halves of the projected channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        # 2 * out_channels for gated activation (GLU style)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=(kernel_size, 1),
            padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, N, C_in].

        Returns:
            Output tensor of shape [B, T, N, C_out].
        """
        # Permute to PyTorch Conv2d format: [B, C_in, T, N]
        x_perm = x.permute(0, 3, 1, 2)

        # Causal left padding along time dimension (pad_left, pad_right, pad_top, pad_bottom)
        # For Conv2d on (T, N): time is dim 2, nodes is dim 3
        # Padding order for (N, T): (0, 0, self.padding, 0)
        x_padded = F.pad(x_perm, (0, 0, self.padding, 0))

        conv_out = self.conv(x_padded)  # [B, 2 * C_out, T, N]

        # Split into P and Q for gated activation
        p = conv_out[:, : self.out_channels, :, :]
        q = conv_out[:, self.out_channels :, :, :]

        gated = torch.tanh(p) * torch.sigmoid(q)

        # Permute back to [B, T, N, C_out]
        return gated.permute(0, 2, 3, 1)


class STGCNBlock(nn.Module):
    """Spatial-Temporal Residual Block.

    Applies:
    1. Causal gated temporal convolution (kernel size 3)
    2. Chebyshev graph convolution (K=3)
    3. Causal gated temporal convolution (kernel size 3)
    4. Layer Normalization, Dropout (0.1), and residual shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        cheb_k: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # 1. Temporal Conv 1
        self.tconv1 = CausalGatedTemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        # 2. Graph Spatial Conv
        self.sconv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            k=cheb_k,
        )

        # 3. Temporal Conv 2
        self.tconv2 = CausalGatedTemporalConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        # 4. LayerNorm and Dropout
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

        # Residual shortcut projection if dimensions change
        if in_channels != hidden_channels:
            self.residual = nn.Linear(in_channels, hidden_channels)
        else:
            self.residual = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, N, C_in].
            cheb_polynomials: Chebyshev basis [K, N, N].

        Returns:
            Output tensor [B, T, N, hidden_channels].
        """
        residual = self.residual(x)

        # TCN 1 -> ReLU / identity (activation is inside Gated TCN)
        out = self.tconv1(x)

        # Spatial ChebConv -> ReLU
        out = F.relu(self.sconv(out, cheb_polynomials))

        # TCN 2
        out = self.tconv2(out)

        # LayerNorm over channels
        out = self.layer_norm(out)

        # Dropout and Residual
        out = self.dropout(out)
        out = out + residual

        return out


class DeterministicSTGCN(nn.Module):
    """Canonical Capacity-Controlled Deterministic ST-GCN Baseline.

    Specification:
    - Input: [B, L, N, 2] (normalized values and binary observation mask)
    - 2 Causal ST-Blocks with 64 hidden channels throughout
    - Final linear point-forecast head for H horizons -> [B, H, N, 1]
    """

    def __init__(
        self,
        num_nodes: int = 325,
        in_channels: int = 2,
        hidden_channels: int = 64,
        input_length: int = 12,
        forecast_horizon: int = 12,
        cheb_k: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.cheb_k = cheb_k

        # 2 Causal ST-Blocks
        self.block1 = STGCNBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
            cheb_k=cheb_k,
            dropout=dropout,
        )
        self.block2 = STGCNBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
            cheb_k=cheb_k,
            dropout=dropout,
        )

        # Final context projection: [B, L, N, hidden_channels] -> 64-dim context per node
        self.context_proj = nn.Linear(hidden_channels, hidden_channels)

        # Direct multi-horizon linear forecast head
        # Maps the temporal context at the final historical step (or combined sequence) to H horizons
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, forecast_horizon),
        )

    def forward(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of deterministic ST-GCN.

        Args:
            x: Input tensor of shape [B, L, N, C_in] (e.g. C_in=2).
            cheb_polynomials: Chebyshev polynomials tensor of shape [K, N, N].

        Returns:
            Forecast predictions of shape [B, H, N, 1].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor [B, L, N, C], got {x.shape}")

        _b, l_in, n, c = x.shape
        if l_in != self.input_length:
            raise ValueError(f"Input length mismatch: expected {self.input_length}, got {l_in}")
        if n != self.num_nodes:
            raise ValueError(f"Number of nodes mismatch: expected {self.num_nodes}, got {n}")
        if c != self.in_channels:
            raise ValueError(f"Input channel mismatch: expected {self.in_channels}, got {c}")

        if cheb_polynomials.shape[0] != self.cheb_k or cheb_polynomials.shape[1] != n or cheb_polynomials.shape[2] != n:
            raise ValueError(
                f"Chebyshev basis mismatch: expected [{self.cheb_k}, {n}, {n}], got {cheb_polynomials.shape}"
            )

        # Pass through Block 1 & Block 2
        h1 = self.block1(x, cheb_polynomials)  # [B, L, N, 64]
        h2 = self.block2(h1, cheb_polynomials)  # [B, L, N, 64]

        # Context projection
        context = self.context_proj(h2)  # [B, L, N, 64]

        # Use the context at the last historical time step t=L-1
        last_context = context[:, -1, :, :]  # [B, N, 64]

        # Direct multi-horizon forecast head: [B, N, 64] -> [B, N, H]
        forecast = self.forecast_head(last_context)  # [B, N, H]

        # Permute to canonical shape: [B, H, N, 1]
        forecast = forecast.permute(0, 2, 1).unsqueeze(-1)  # [B, H, N, 1]
        return forecast


def count_trainable_parameters(model: nn.Module) -> int:
    """Calculate the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_capacity_report(
    model: nn.Module,
    model_name: str = "deterministic_st_gcn",
    reference_param_count: int | None = None,
) -> dict[str, Any]:
    """Generate a capacity and parameter summary report per ADR-0008.

    Args:
        model: PyTorch model.
        model_name: Name of the model architecture.
        reference_param_count: Trainable parameters of target model for ratio calculation.

    Returns:
        Dictionary report containing parameter counts and capacity ratio.
    """
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())

    report: dict[str, Any] = {
        "model_name": model_name,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "hidden_channels": getattr(model, "hidden_channels", None),
        "forecast_horizon": getattr(model, "forecast_horizon", None),
    }

    if reference_param_count is not None and reference_param_count > 0:
        ratio = trainable_params / reference_param_count
        report["capacity_ratio_vs_reference"] = float(ratio)
        report["capacity_ratio_valid"] = 0.5 <= ratio <= 2.0
    else:
        report["capacity_ratio_vs_reference"] = None
        report["capacity_ratio_valid"] = False

    return report
