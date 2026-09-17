"""Probabilistic Gaussian Forecast Head for ST-DSSM — Phase 7.

Implements the autoregressive 12-step decoder that produces per-horizon
Gaussian predictive distributions from the latent state and encoder
context summary per ADR-0007.

Architecture
------------
At each horizon step h:
    input:  concat[z, context_summary, horizon_embed_h, prev_mu_h]
    output: mu_h (unconstrained), sigma_h = softplus(clamp(log_raw, -8, 5)) + 1e-4

Teacher forcing (training only):
    At each step, with probability teacher_force_ratio the ground-truth
    y_target[:, h, :, :] is used as prev_mu instead of the previous prediction.
    At val/test teacher_force_ratio must be 0.0 (autoregressive only).

Canonical hyperparameters (ADR-0007):
    latent_dim   = 32
    context_dim  = 64
    hidden_dim   = 128
    horizon      = 12
    num_nodes    = 325
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Bounds on log-scale per ADR-0007
_LOG_SIGMA_MIN: float = -8.0
_LOG_SIGMA_MAX: float = 5.0
# Hard lower bound on sigma to guarantee strict positivity
_SIGMA_FLOOR: float = 1e-4


class GaussianForecastHead(nn.Module):
    """Autoregressive Gaussian decoder for multi-step probabilistic forecasting.

    Decodes a latent state and encoder context into H-step Gaussian predictive
    distributions per ADR-0007:
        sigma = softplus(clamp(log_sigma_raw, -8, 5)) + 1e-4

    Shape contract
    --------------
    z:               [B, N, latent_dim]   — latent state (prior or posterior sample)
    context_summary: [B, N, context_dim]  — encoder context at the last history step
    y_target:        [B, H, N, 1]         — ground truth (teacher forcing, training only)
    output:          (mu, sigma) each [B, H, N, 1]
    """

    def __init__(
        self,
        latent_dim: int = 32,
        context_dim: int = 64,
        hidden_dim: int = 128,
        horizon: int = 12,
        num_nodes: int = 325,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")

        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_nodes = num_nodes

        # Learnable per-horizon position embeddings [H, hidden_dim]
        self.horizon_embed = nn.Embedding(horizon, hidden_dim)

        # MLP decoder: [z, context_summary, horizon_embed, prev_mu] -> (mu_raw, log_sigma_raw)
        decoder_in_dim = latent_dim + context_dim + hidden_dim + 1
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # outputs (mu_raw, log_sigma_raw)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for linear layers; normal for embeddings."""
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.horizon_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        z: torch.Tensor,
        context_summary: torch.Tensor,
        y_target: torch.Tensor | None = None,
        teacher_force_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively decode H-step Gaussian forecasts.

        Args:
            z: Latent state tensor [B, N, latent_dim].
            context_summary: Encoder context summary [B, N, context_dim].
            y_target: Ground-truth forecast targets [B, H, N, 1]. Required
                when teacher_force_ratio > 0.0; ignored at val/test.
            teacher_force_ratio: Float in [0, 1]. Probability of using the
                ground-truth previous output as prev_mu instead of the
                predicted value. Must be 0.0 at val/test.

        Returns:
            (mu, sigma): Tensors of shape [B, H, N, 1].
                mu:    Unconstrained predicted mean.
                sigma: Strictly positive predicted std dev (>= 1e-4).

        Raises:
            ValueError: On shape mismatch, non-finite inputs, or missing
                y_target when teacher_force_ratio > 0.0.
        """
        # --- Input validation ---
        if z.dim() != 3:
            raise ValueError(
                f"Expected z of shape [B, N, latent_dim], got shape {z.shape}"
            )
        if context_summary.dim() != 3:
            raise ValueError(
                f"Expected context_summary of shape [B, N, context_dim], got shape {context_summary.shape}"
            )

        B, N, ld = z.shape
        if ld != self.latent_dim:
            raise ValueError(
                f"z latent_dim mismatch: expected {self.latent_dim}, got {ld}"
            )
        if context_summary.shape != (B, N, self.context_dim):
            raise ValueError(
                f"context_summary shape mismatch: expected {(B, N, self.context_dim)}, "
                f"got {context_summary.shape}"
            )
        if not torch.isfinite(z).all():
            raise ValueError("z contains non-finite values (NaN or Inf).")
        if not torch.isfinite(context_summary).all():
            raise ValueError("context_summary contains non-finite values (NaN or Inf).")
        if not 0.0 <= teacher_force_ratio <= 1.0:
            raise ValueError(
                f"teacher_force_ratio must be in [0, 1], got {teacher_force_ratio}"
            )
        if teacher_force_ratio > 0.0:
            if y_target is None:
                raise ValueError(
                    "y_target must be provided when teacher_force_ratio > 0.0."
                )
            if y_target.shape != (B, self.horizon, N, 1):
                raise ValueError(
                    f"y_target shape mismatch: expected {(B, self.horizon, N, 1)}, "
                    f"got {y_target.shape}"
                )

        device = z.device
        dtype = z.dtype

        # Pre-compute all horizon embeddings: [H, hidden_dim]
        h_idx = torch.arange(self.horizon, device=device)
        h_embeds = self.horizon_embed(h_idx)  # [H, hidden_dim]

        mu_steps: list[torch.Tensor] = []
        sigma_steps: list[torch.Tensor] = []

        # Initial previous predicted mean: zeros [B, N, 1]
        prev_mu = torch.zeros(B, N, 1, device=device, dtype=dtype)

        for h in range(self.horizon):
            # Broadcast horizon embedding: [hidden_dim] -> [B, N, hidden_dim]
            h_emb = h_embeds[h].unsqueeze(0).unsqueeze(0).expand(B, N, -1)

            # Build decoder input: [B, N, latent_dim + context_dim + hidden_dim + 1]
            dec_in = torch.cat([z, context_summary, h_emb, prev_mu], dim=-1)

            # MLP forward: flatten (B, N) -> [B*N, decoder_in], apply, restore
            out_flat = self.decoder(dec_in.reshape(B * N, -1))  # [B*N, 2]
            out = out_flat.reshape(B, N, 2)

            mu_h = out[..., 0:1]         # [B, N, 1] — unconstrained
            log_sigma_raw = out[..., 1:2]  # [B, N, 1]

            # Clamp then convert to strictly positive sigma per ADR-0007
            log_sigma = torch.clamp(log_sigma_raw, _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
            sigma_h = F.softplus(log_sigma) + _SIGMA_FLOOR

            # Finite output guard — reject non-finite outputs immediately
            if not torch.isfinite(mu_h).all() or not torch.isfinite(sigma_h).all():
                raise ValueError(
                    f"Non-finite forecast output at horizon step {h + 1}: "
                    f"mu finite={torch.isfinite(mu_h).all().item()}, "
                    f"sigma finite={torch.isfinite(sigma_h).all().item()}."
                )

            mu_steps.append(mu_h)
            sigma_steps.append(sigma_h)

            # Determine prev_mu for next step
            if h < self.horizon - 1:
                if teacher_force_ratio > 0.0 and y_target is not None:
                    # Stochastic teacher forcing: use ground truth with probability teacher_force_ratio
                    use_teacher = torch.rand(1, device=device).item() < teacher_force_ratio
                    if use_teacher:
                        prev_mu = y_target[:, h, :, :]  # [B, N, 1]
                    else:
                        prev_mu = mu_h.detach()
                else:
                    prev_mu = mu_h.detach()

        # Stack along horizon dimension: list of [B, N, 1] -> [B, H, N, 1]
        mu = torch.stack(mu_steps, dim=1)
        sigma = torch.stack(sigma_steps, dim=1)

        return mu, sigma

    def sample_prediction(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Draw samples from the Gaussian predictive distribution.

        Uses the reparameterization trick:
            y = mu + sigma * eps,  eps ~ N(0, I)

        Args:
            mu: Predicted means [B, H, N, 1].
            sigma: Predicted std devs [B, H, N, 1] (strictly positive).
            n_samples: Number of samples to draw per prediction.

        Returns:
            Samples tensor of shape [n_samples, B, H, N, 1].

        Raises:
            ValueError: On shape mismatch, non-positive sigma, or non-finite inputs.
        """
        if mu.shape != sigma.shape:
            raise ValueError(
                f"mu and sigma must have the same shape, "
                f"got {mu.shape} vs {sigma.shape}"
            )
        if not torch.isfinite(mu).all():
            raise ValueError("mu contains non-finite values.")
        if not torch.isfinite(sigma).all():
            raise ValueError("sigma contains non-finite values.")
        if (sigma <= 0).any():
            raise ValueError("sigma must be strictly positive everywhere.")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        # Reparameterization: [n_samples, B, H, N, 1]
        eps = torch.randn(n_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
