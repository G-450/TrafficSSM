"""Deep State Space Model (DSSM) for ST-DSSM — Phase 8.

Integrates the SpatialTemporalEncoder (Phase 6) with a GRU-based prior
transition network, bidirectional GRU recognition network, and
GaussianForecastHead (Phase 7) per ADR-0007.

Latent state
------------
    z_t in R^{latent_dim} per sensor node, modelled as diagonal Gaussian.
    latent_dim = 32 (canonical per ADR-0007).

Components
----------
    Prior:      p(z_t | z_{t-1}, c_t)            — causal GRU transition
    Posterior:  q(z_t | context_{1:L}, x, m)     — Bi-GRU recognition, training only
    Emission:   p(y_{1:H} | z_L, context_summary) — GaussianForecastHead (Phase 7)

Objective
---------
    negative ELBO = recon_nll + beta * KL(q || p)

    beta:       0 -> 1 linearly over first 20 training epochs (passed in by caller).
    recon_nll:  masked Gaussian NLL over observed forecast targets.
    KL:         closed-form diagonal Gaussian KL, mean over B, L, N, latent_dim.
    Non-finite loss raises ValueError immediately (no silent NaN training).

Test / Val contract
-------------------
    forward_predict samples exclusively from the prior; no posterior is computed,
    no teacher forcing is applied, no targets are accessed.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from st_dssm.encoder import SpatialTemporalEncoder
from st_dssm.forecast_head import GaussianForecastHead

# Bounds on log-scale per ADR-0007
_LOG_SIGMA_MIN: float = -8.0
_LOG_SIGMA_MAX: float = 5.0
_SIGMA_FLOOR: float = 1e-4


# ---------------------------------------------------------------------------
# Internal sub-networks
# ---------------------------------------------------------------------------


class _TransitionNet(nn.Module):
    """Prior transition p(z_t | z_{t-1}, c_t) as a GRU-based diagonal Gaussian.

    All B*N nodes are processed in parallel by collapsing the batch and node
    dimensions. The GRU maintains a hidden state h_t that accumulates temporal
    information causally (left to right only).

    Input per step:  concat[z_{t-1}, c_t]   — [B*N, latent_dim + context_dim]
    GRU hidden:                              — [B*N, hidden_dim]
    Output per step: (mu_p_t, sigma_p_t)    — [B, N, latent_dim] each
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.gru_cell = nn.GRUCell(
            input_size=latent_dim + context_dim,
            hidden_size=hidden_dim,
        )
        # Projects GRU hidden state -> (mu_p, log_sigma_p_raw), dim = 2 * latent_dim
        self.out_proj = nn.Linear(hidden_dim, 2 * latent_dim)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        context: torch.Tensor,
        z0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run causal prior forward through L history steps.

        Args:
            context: Encoder context [B, L, N, context_dim].
            z0:      Initial latent state [B, N, latent_dim] (typically zeros).

        Returns:
            (mu_p, sigma_p, z_L):
                mu_p:    Prior means    [B, L, N, latent_dim]  (strictly positive)
                sigma_p: Prior std devs [B, L, N, latent_dim]  (strictly positive)
                z_L:     Last sampled latent [B, N, latent_dim] (reparameterized)
        """
        B, L, N, _ = context.shape

        # Collapse B, N for parallel GRU: [B*N, hidden_dim]
        h = torch.zeros(B * N, self.hidden_dim, device=context.device, dtype=context.dtype)
        z_prev = z0.reshape(B * N, self.latent_dim)

        mu_p_list: list[torch.Tensor] = []
        sigma_p_list: list[torch.Tensor] = []
        z_last = z_prev  # will be overwritten each step

        for t in range(L):
            c_t = context[:, t, :, :].reshape(B * N, self.context_dim)
            gru_in = torch.cat([z_prev, c_t], dim=-1)
            h = self.gru_cell(gru_in, h)

            raw = self.out_proj(h)  # [B*N, 2*latent_dim]
            mu_p_t = raw[:, : self.latent_dim]
            log_s_t = torch.clamp(
                raw[:, self.latent_dim :], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX
            )
            sigma_p_t = F.softplus(log_s_t) + _SIGMA_FLOOR

            # Reparameterization sample for prior chain: z_t ~ p(z_t | ...)
            eps = torch.randn_like(sigma_p_t)
            z_t = mu_p_t + sigma_p_t * eps

            mu_p_list.append(mu_p_t.reshape(B, N, self.latent_dim))
            sigma_p_list.append(sigma_p_t.reshape(B, N, self.latent_dim))

            z_prev = z_t
            z_last = z_t

        mu_p = torch.stack(mu_p_list, dim=1)     # [B, L, N, latent_dim]
        sigma_p = torch.stack(sigma_p_list, dim=1)
        z_L = z_last.reshape(B, N, self.latent_dim)
        return mu_p, sigma_p, z_L


class _RecognitionNet(nn.Module):
    """Posterior q(z_t | context_{1:L}, x_{1:L}, m_{1:L}) via Bidirectional GRU.

    Used exclusively during training. The bi-GRU reads the full history
    window in both directions to produce per-timestep posterior parameters.
    Unobserved speed entries are zeroed before being fed to the recognition
    network so no values behind the mask are visible.

    Input per step:  concat[context_t, masked_speed_t, mask_t]
                     [B*N, L, context_dim + 2]
    Output per step: (mu_q, sigma_q) each [B, L, N, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Input: context_dim + speed channel (1) + mask channel (1)
        rec_in_dim = context_dim + 2
        self.bigru = nn.GRU(
            input_size=rec_in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # Projects bi-GRU output (2 * hidden_dim) -> (mu_q, log_sigma_q_raw)
        self.out_proj = nn.Linear(2 * hidden_dim, 2 * latent_dim)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        context: torch.Tensor,
        x_speed: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameterize per-step posterior distributions.

        Args:
            context:  Encoder context [B, L, N, context_dim].
            x_speed:  Normalized speed channel [B, L, N, 1].
            obs_mask: Binary observation mask [B, L, N, 1].

        Returns:
            (mu_q, sigma_q): each [B, L, N, latent_dim].
        """
        B, L, N, _ = context.shape

        # Zero-out unobserved speed entries before feeding to recognition net
        masked_speed = x_speed * obs_mask  # [B, L, N, 1]

        # Concatenate inputs: [B, L, N, context_dim + 2]
        rec_in = torch.cat([context, masked_speed, obs_mask], dim=-1)

        # Collapse B and N for parallel GRU: [B*N, L, context_dim + 2]
        rec_in_flat = rec_in.reshape(B * N, L, self.context_dim + 2)

        # Bi-GRU forward: [B*N, L, 2*hidden_dim]
        bigru_out, _ = self.bigru(rec_in_flat)

        # Project per-step: [B*N, L, 2*latent_dim]
        raw = self.out_proj(bigru_out)

        # Restore spatial dimensions and split into (mu_q, sigma_q)
        mu_q = raw[:, :, : self.latent_dim].reshape(B, L, N, self.latent_dim)
        log_s_q = torch.clamp(
            raw[:, :, self.latent_dim :], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX
        ).reshape(B, L, N, self.latent_dim)
        sigma_q = F.softplus(log_s_q) + _SIGMA_FLOOR

        return mu_q, sigma_q


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _diagonal_gaussian_kl(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL(q || p) for diagonal Gaussians.

    KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))
        = 0.5 * [ log(sigma_p^2/sigma_q^2)
                  + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2
                  - 1 ]

    Returns the mean KL over all elements (B, L, N, latent_dim).
    The returned value is always >= 0 for valid sigma inputs.
    """
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    kl = 0.5 * (
        2.0 * torch.log(sigma_p)
        - 2.0 * torch.log(sigma_q)
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )
    return kl.mean()


def _masked_gaussian_nll(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked Gaussian NLL, averaged over observed target elements.

    NLL_elem = 0.5 * log(2*pi) + log(sigma) + 0.5 * ((y - mu) / sigma)^2

    Args:
        y_true:   Ground truth [B, H, N, 1].
        mu:       Predicted means [B, H, N, 1].
        sigma:    Predicted std devs [B, H, N, 1] (strictly positive).
        obs_mask: Binary mask [B, H, N, 1] (1 = observed target).

    Returns:
        Scalar mean NLL over observed targets.
        Returns 0.0 tensor if mask has no valid entries (degenerate case).
    """
    nll_elem = (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(sigma)
        + 0.5 * ((y_true - mu) / sigma) ** 2
    )
    bool_mask = obs_mask.bool()
    valid_count = bool_mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
    return nll_elem[bool_mask].mean()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GaussianDSSM(nn.Module):
    """Uncertainty-Aware Spatial-Temporal Deep State Space Model (ST-DSSM).

    Implements the full ST-DSSM architecture per ADR-0007:

        1. ST-GCN Encoder (Phase 6)    context: [B, L, N, 64]
        2. Prior TransitionNet          p(z_t | z_{t-1}, c_t), causal GRU
        3. Recognition Net              q(z_t | context, x, m), Bi-GRU, training only
        4. Gaussian Forecast Head (P7)  autoregressive 12-step decoder
        5. ELBO                         recon_nll + beta * KL(q || p)

    Canonical hyperparameters (ADR-0007):
        latent_dim   = 32
        context_dim  = 64
        hidden_dim   = 128
        horizon      = 12
        num_nodes    = 325
        input_length = 12
    """

    def __init__(
        self,
        num_nodes: int = 325,
        in_channels: int = 2,
        latent_dim: int = 32,
        context_dim: int = 64,
        hidden_dim: int = 128,
        horizon: int = 12,
        input_length: int = 12,
        cheb_k: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if input_length < 1:
            raise ValueError(f"input_length must be >= 1, got {input_length}")

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.input_length = input_length

        # Phase 6: Spatial-Temporal Encoder
        self.encoder = SpatialTemporalEncoder(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=context_dim,
            input_length=input_length,
            cheb_k=cheb_k,
            dropout=dropout,
        )

        # Phase 8: Prior transition network
        self.transition = _TransitionNet(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
        )

        # Phase 8: Recognition (posterior) network — training only
        self.recognition = _RecognitionNet(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
        )

        # Phase 7: Probabilistic Gaussian Forecast Head
        self.forecast_head = GaussianForecastHead(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            num_nodes=num_nodes,
        )

    # ------------------------------------------------------------------
    # Forward: training
    # ------------------------------------------------------------------

    def forward_train(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
        y_target: torch.Tensor,
        obs_mask_hist: torch.Tensor,
        obs_mask_fore: torch.Tensor,
        teacher_force_ratio: float = 0.0,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass — computes predictions and ELBO components.

        Args:
            x:                  History input [B, L, N, 2]
                                  channel 0: normalized speed
                                  channel 1: binary observation mask
            cheb_polynomials:   Chebyshev basis [K, N, N].
            y_target:           Forecast ground truth [B, H, N, 1].
            obs_mask_hist:      Binary observation mask for history [B, L, N, 1].
            obs_mask_fore:      Binary observation mask for forecast [B, H, N, 1].
            teacher_force_ratio: Float in [0, 1]. Decays 1.0 -> 0.0 over the
                first 50% of training epochs (caller manages schedule).
            beta:               KL annealing weight. Increases 0 -> 1 over the
                first 20 training epochs, then fixed at 1 (caller manages).

        Returns:
            (mu_pred, sigma_pred, kl, recon_nll):
                mu_pred:    Predicted means  [B, H, N, 1]
                sigma_pred: Predicted stds   [B, H, N, 1]  (>= 1e-4)
                kl:         Scalar KL divergence (>= 0)
                recon_nll:  Scalar masked reconstruction NLL

        Raises:
            ValueError: On non-finite ELBO, shape mismatch, or invalid inputs.
        """
        self._validate_train_inputs(x, cheb_polynomials, y_target, obs_mask_hist, obs_mask_fore)

        B = x.shape[0]
        N = self.num_nodes
        device = x.device
        dtype = x.dtype

        # 1. Encode history -> context [B, L, N, context_dim]
        context = self.encoder(x, cheb_polynomials)

        # 2. Recognition: posterior q(z_t | context, x_speed, obs_mask_hist)
        x_speed = x[:, :, :, 0:1]  # [B, L, N, 1] — speed channel
        mu_q, sigma_q = self.recognition(context, x_speed, obs_mask_hist)
        # mu_q, sigma_q: [B, L, N, latent_dim]

        # 3. Prior: p(z_t | z_{t-1}, c_t) — causal forward through L steps
        z0 = torch.zeros(B, N, self.latent_dim, device=device, dtype=dtype)
        mu_p, sigma_p, _ = self.transition(context, z0)
        # mu_p, sigma_p: [B, L, N, latent_dim]

        # 4. KL(q || p) — mean over all elements, always >= 0 for valid sigmas
        kl = _diagonal_gaussian_kl(mu_q, sigma_q, mu_p, sigma_p)

        # 5. Sample z_L from posterior at the last history step (reparameterization)
        eps_L = torch.randn_like(sigma_q[:, -1, :, :])
        z_L = mu_q[:, -1, :, :] + sigma_q[:, -1, :, :] * eps_L  # [B, N, latent_dim]

        # 6. Context summary at last history step: [B, N, context_dim]
        context_summary = self.encoder.get_context_summary(context)

        # 7. Autoregressive forecast (with teacher forcing during training)
        mu_pred, sigma_pred = self.forecast_head(
            z=z_L,
            context_summary=context_summary,
            y_target=y_target,
            teacher_force_ratio=teacher_force_ratio,
        )
        # mu_pred, sigma_pred: [B, H, N, 1]

        # 8. Masked reconstruction NLL over observed forecast targets
        recon_nll = _masked_gaussian_nll(y_target, mu_pred, sigma_pred, obs_mask_fore)

        # 9. ELBO = recon_nll + beta * KL
        elbo = recon_nll + beta * kl

        # 10. Hard finite guard — non-finite ELBO must not propagate silently
        if not torch.isfinite(elbo):
            raise ValueError(
                f"Non-finite ELBO: elbo={elbo.item():.6g}, "
                f"recon_nll={recon_nll.item():.6g}, "
                f"kl={kl.item():.6g}, beta={beta:.4f}. "
                "Inspect model parameters and gradient norms."
            )

        return mu_pred, sigma_pred, kl, recon_nll

    # ------------------------------------------------------------------
    # Forward: inference (val / test)
    # ------------------------------------------------------------------

    def forward_predict(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward pass — prior only, no posterior, no teacher forcing.

        Used exclusively for validation and test evaluation per ADR-0007.
        Targets are never accessed in this path.

        Args:
            x:                History input [B, L, N, 2].
            cheb_polynomials: Chebyshev basis [K, N, N].

        Returns:
            (mu_pred, sigma_pred): each [B, H, N, 1].

        Raises:
            ValueError: On non-finite outputs or shape mismatch.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected x of shape [B, L, N, 2], got {x.shape}"
            )
        if x.shape[1] != self.input_length:
            raise ValueError(
                f"Input length mismatch: expected {self.input_length}, got {x.shape[1]}"
            )
        if x.shape[2] != self.num_nodes:
            raise ValueError(
                f"Node count mismatch: expected {self.num_nodes}, got {x.shape[2]}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values (NaN or Inf).")

        B = x.shape[0]
        N = self.num_nodes
        device = x.device
        dtype = x.dtype

        # 1. Encode
        context = self.encoder(x, cheb_polynomials)

        # 2. Prior forward — sample z_L from prior chain (no posterior)
        z0 = torch.zeros(B, N, self.latent_dim, device=device, dtype=dtype)
        _mu_p, _sigma_p, z_L = self.transition(context, z0)

        # 3. Context summary
        context_summary = self.encoder.get_context_summary(context)

        # 4. Autoregressive forecast — no teacher forcing, no targets accessed
        mu_pred, sigma_pred = self.forecast_head(
            z=z_L,
            context_summary=context_summary,
            y_target=None,
            teacher_force_ratio=0.0,
        )

        # 5. Finite output guard
        if not torch.isfinite(mu_pred).all():
            raise ValueError("Non-finite values in predicted mu.")
        if not torch.isfinite(sigma_pred).all():
            raise ValueError("Non-finite values in predicted sigma.")

        return mu_pred, sigma_pred

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_capacity_report(self) -> dict[str, Any]:
        """Report trainable parameter counts per sub-module.

        Returns:
            Dictionary with total and per-component trainable parameter counts.
        """
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "model_name": "gaussian_dssm",
            "total_trainable_parameters": _count(self),
            "encoder_parameters": _count(self.encoder),
            "transition_parameters": _count(self.transition),
            "recognition_parameters": _count(self.recognition),
            "forecast_head_parameters": _count(self.forecast_head),
        }

    def _validate_train_inputs(
        self,
        x: torch.Tensor,
        cheb_polynomials: torch.Tensor,
        y_target: torch.Tensor,
        obs_mask_hist: torch.Tensor,
        obs_mask_fore: torch.Tensor,
    ) -> None:
        """Strict shape and finiteness validation for all training inputs."""
        if x.dim() != 4:
            raise ValueError(
                f"Expected x of shape [B, L, N, 2], got {x.shape}"
            )
        B, L, N, C = x.shape
        if L != self.input_length:
            raise ValueError(
                f"Input length mismatch: expected {self.input_length}, got {L}"
            )
        if N != self.num_nodes:
            raise ValueError(
                f"Node count mismatch: expected {self.num_nodes}, got {N}"
            )
        if C != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.in_channels}, got {C}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values (NaN or Inf).")

        if y_target.shape != (B, self.horizon, N, 1):
            raise ValueError(
                f"y_target shape mismatch: expected {(B, self.horizon, N, 1)}, "
                f"got {y_target.shape}"
            )
        if not torch.isfinite(y_target).all():
            raise ValueError("y_target contains non-finite values.")

        if obs_mask_hist.shape != (B, L, N, 1):
            raise ValueError(
                f"obs_mask_hist shape mismatch: expected {(B, L, N, 1)}, "
                f"got {obs_mask_hist.shape}"
            )
        if obs_mask_fore.shape != (B, self.horizon, N, 1):
            raise ValueError(
                f"obs_mask_fore shape mismatch: expected {(B, self.horizon, N, 1)}, "
                f"got {obs_mask_fore.shape}"
            )
        if cheb_polynomials.dim() != 3:
            raise ValueError(
                f"cheb_polynomials must be 3D [K, N, N], got {cheb_polynomials.dim()}D."
            )
