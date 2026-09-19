"""Unit and integration tests for Phase 7 — Probabilistic Gaussian Forecast Head.

Tests
-----
- Output shapes [B, H, N, 1] for mu and sigma
- sigma strictly positive everywhere
- Finite outputs (no NaN or Inf)
- Log-sigma clamping holds at extreme inputs
- Teacher forcing path vs. autoregressive path produce different outputs
- Gradient flow through all trainable parameters
- Single-batch overfit: loss decreases over iterations
- sample_prediction shape and reparameterization correctness
- Invalid argument rejection (wrong shapes, negative sigma, missing y_target)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from st_dssm.forecast_head import (
    GaussianForecastHead,
    _LOG_SIGMA_MAX,
    _LOG_SIGMA_MIN,
    _SIGMA_FLOOR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_head() -> GaussianForecastHead:
    """Small GaussianForecastHead for fast testing."""
    return GaussianForecastHead(
        latent_dim=8,
        context_dim=16,
        hidden_dim=32,
        horizon=4,
        num_nodes=5,
    )


@pytest.fixture
def canonical_inputs(small_head: GaussianForecastHead):
    """Canonical synthetic inputs matching small_head dimensions."""
    torch.manual_seed(0)
    B, N = 2, small_head.num_nodes
    ld = small_head.latent_dim
    cd = small_head.context_dim
    H = small_head.horizon

    z = torch.randn(B, N, ld)
    ctx = torch.randn(B, N, cd)
    y = torch.randn(B, H, N, 1)
    return z, ctx, y, B, N, H


# ---------------------------------------------------------------------------
# Shape and value correctness
# ---------------------------------------------------------------------------

class TestForecastHeadShapes:
    def test_output_shapes_no_teacher_forcing(self, small_head, canonical_inputs):
        z, ctx, y, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)
        assert mu.shape == (B, H, N, 1)
        assert sigma.shape == (B, H, N, 1)

    def test_output_shapes_with_teacher_forcing(self, small_head, canonical_inputs):
        z, ctx, y, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx, y_target=y, teacher_force_ratio=1.0)
        assert mu.shape == (B, H, N, 1)
        assert sigma.shape == (B, H, N, 1)

    def test_sigma_strictly_positive(self, small_head, canonical_inputs):
        z, ctx, y, B, N, H = canonical_inputs
        _, sigma = small_head(z, ctx)
        assert (sigma > 0).all(), "sigma contains non-positive values."
        assert (sigma >= _SIGMA_FLOOR).all(), (
            f"sigma below floor {_SIGMA_FLOOR}: min={sigma.min().item()}"
        )

    def test_outputs_finite(self, small_head, canonical_inputs):
        z, ctx, _, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)
        assert torch.isfinite(mu).all(), "mu contains non-finite values."
        assert torch.isfinite(sigma).all(), "sigma contains non-finite values."


# ---------------------------------------------------------------------------
# Log-sigma clamping
# ---------------------------------------------------------------------------

class TestLogSigmaClamping:
    def test_log_sigma_clamp_holds_at_max(self, small_head):
        """Extreme positive raw outputs must be clamped to log_sigma_max."""
        B, N = 1, small_head.num_nodes
        z = torch.randn(B, N, small_head.latent_dim)
        ctx = torch.randn(B, N, small_head.context_dim)

        # Force decoder output towards +infinity via very large z and ctx
        with torch.no_grad():
            # Override last linear bias to produce huge log_sigma_raw
            last_linear = small_head.decoder[-1]
            last_linear.bias.fill_(1e6)

        _, sigma = small_head(z, ctx)
        # softplus(_LOG_SIGMA_MAX) + _SIGMA_FLOOR is the max achievable sigma
        max_sigma = F.softplus(torch.tensor(_LOG_SIGMA_MAX)) + _SIGMA_FLOOR
        assert (sigma <= max_sigma + 1e-5).all(), (
            f"sigma exceeded expected clamp bound. max={sigma.max().item()}"
        )

    def test_log_sigma_clamp_holds_at_min(self, small_head):
        """Extreme negative raw outputs must be clamped to log_sigma_min."""
        B, N = 1, small_head.num_nodes
        z = torch.randn(B, N, small_head.latent_dim)
        ctx = torch.randn(B, N, small_head.context_dim)

        with torch.no_grad():
            last_linear = small_head.decoder[-1]
            last_linear.bias.fill_(-1e6)

        _, sigma = small_head(z, ctx)
        min_sigma = F.softplus(torch.tensor(_LOG_SIGMA_MIN)) + _SIGMA_FLOOR
        assert (sigma >= min_sigma - 1e-5).all(), (
            f"sigma below expected clamp floor. min={sigma.min().item()}"
        )


# ---------------------------------------------------------------------------
# Teacher forcing
# ---------------------------------------------------------------------------

class TestTeacherForcing:
    def test_teacher_forcing_full_differs_from_autoregressive(self, small_head, canonical_inputs):
        """With teacher_force_ratio=1.0 and ratio=0.0 outputs must differ (in general)."""
        z, ctx, y, B, N, H = canonical_inputs
        small_head.eval()

        torch.manual_seed(1)
        mu_tf, _ = small_head(z, ctx, y_target=y, teacher_force_ratio=1.0)

        torch.manual_seed(1)
        mu_ar, _ = small_head(z, ctx, teacher_force_ratio=0.0)

        # With H=4 steps, teacher forcing changes prev_mu for steps 2..4,
        # so outputs should not be identical.
        assert not torch.allclose(mu_tf, mu_ar), (
            "Teacher-forced and autoregressive outputs are unexpectedly identical."
        )

    def test_teacher_forcing_zero_is_fully_autoregressive(self, small_head, canonical_inputs):
        """With ratio=0.0, y_target is never used; same output regardless of y_target."""
        z, ctx, y, B, N, H = canonical_inputs
        small_head.eval()

        torch.manual_seed(2)
        mu1, sig1 = small_head(z, ctx, y_target=y, teacher_force_ratio=0.0)

        torch.manual_seed(2)
        mu2, sig2 = small_head(z, ctx, y_target=None, teacher_force_ratio=0.0)

        assert torch.allclose(mu1, mu2, atol=1e-6), (
            "Autoregressive outputs differ when y_target is and is not provided with ratio=0."
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradient_flow_all_parameters(self, small_head, canonical_inputs):
        """All trainable parameters must receive finite gradients."""
        z, ctx, y, B, N, H = canonical_inputs
        small_head.train()

        mu, sigma = small_head(z, ctx)
        loss = mu.sum() + sigma.sum()
        loss.backward()

        for name, param in small_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter '{name}'."
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in parameter '{name}'."
                )


# ---------------------------------------------------------------------------
# Overfit single batch
# ---------------------------------------------------------------------------

class TestSingleBatchOverfit:
    def test_loss_decreases_on_single_batch(self, small_head):
        """NLL loss should strictly decrease when overfitting a single batch."""
        torch.manual_seed(42)
        B, N = 2, small_head.num_nodes
        H = small_head.horizon

        z = torch.randn(B, N, small_head.latent_dim)
        ctx = torch.randn(B, N, small_head.context_dim)
        y = torch.randn(B, H, N, 1)

        small_head.train()
        optimizer = torch.optim.Adam(small_head.parameters(), lr=5e-3)

        initial_loss: float | None = None
        final_loss: float | None = None

        for _ in range(80):
            optimizer.zero_grad()
            mu, sigma = small_head(z, ctx, y_target=y, teacher_force_ratio=1.0)
            # Gaussian NLL loss
            loss = (
                torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2
            ).mean()
            loss.backward()
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss is not None and initial_loss is not None
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# sample_prediction
# ---------------------------------------------------------------------------

class TestSamplePrediction:
    def test_sample_shape(self, small_head, canonical_inputs):
        z, ctx, _, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)

        n_samples = 5
        samples = small_head.sample_prediction(mu, sigma, n_samples=n_samples)
        assert samples.shape == (n_samples, B, H, N, 1)

    def test_sample_single(self, small_head, canonical_inputs):
        z, ctx, _, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)
        samples = small_head.sample_prediction(mu, sigma, n_samples=1)
        assert samples.shape == (1, B, H, N, 1)

    def test_samples_finite(self, small_head, canonical_inputs):
        z, ctx, _, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)
        samples = small_head.sample_prediction(mu, sigma, n_samples=3)
        assert torch.isfinite(samples).all()

    def test_sample_distribution_mean(self, small_head, canonical_inputs):
        """Large-sample mean should converge toward mu."""
        z, ctx, _, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)

        samples = small_head.sample_prediction(mu, sigma, n_samples=2000)
        sample_mean = samples.mean(dim=0)  # [B, H, N, 1]
        assert torch.allclose(sample_mean, mu, atol=0.15), (
            f"Sample mean deviates too far from mu. "
            f"Max diff: {(sample_mean - mu).abs().max().item():.4f}"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_z_wrong_dim(self, small_head):
        B, N = 2, small_head.num_nodes
        with pytest.raises(ValueError, match="Expected z of shape"):
            small_head(torch.randn(B, N), torch.randn(B, N, small_head.context_dim))

    def test_z_wrong_latent_dim(self, small_head):
        B, N = 2, small_head.num_nodes
        with pytest.raises(ValueError, match="z latent_dim mismatch"):
            small_head(
                torch.randn(B, N, small_head.latent_dim + 1),
                torch.randn(B, N, small_head.context_dim),
            )

    def test_context_summary_wrong_shape(self, small_head):
        B, N = 2, small_head.num_nodes
        with pytest.raises(ValueError, match="context_summary shape mismatch"):
            small_head(
                torch.randn(B, N, small_head.latent_dim),
                torch.randn(B, N + 1, small_head.context_dim),
            )

    def test_non_finite_z(self, small_head):
        B, N = 2, small_head.num_nodes
        z = torch.randn(B, N, small_head.latent_dim)
        z[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite values"):
            small_head(z, torch.randn(B, N, small_head.context_dim))

    def test_non_finite_context_summary(self, small_head):
        B, N = 2, small_head.num_nodes
        ctx = torch.randn(B, N, small_head.context_dim)
        ctx[0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite values"):
            small_head(torch.randn(B, N, small_head.latent_dim), ctx)

    def test_teacher_forcing_without_y_target(self, small_head):
        B, N = 2, small_head.num_nodes
        with pytest.raises(ValueError, match="y_target must be provided"):
            small_head(
                torch.randn(B, N, small_head.latent_dim),
                torch.randn(B, N, small_head.context_dim),
                y_target=None,
                teacher_force_ratio=0.5,
            )

    def test_teacher_forcing_y_target_wrong_shape(self, small_head):
        B, N, H = 2, small_head.num_nodes, small_head.horizon
        with pytest.raises(ValueError, match="y_target shape mismatch"):
            small_head(
                torch.randn(B, N, small_head.latent_dim),
                torch.randn(B, N, small_head.context_dim),
                y_target=torch.randn(B, H + 1, N, 1),
                teacher_force_ratio=0.5,
            )

    def test_invalid_teacher_force_ratio(self, small_head):
        B, N = 2, small_head.num_nodes
        with pytest.raises(ValueError, match="teacher_force_ratio must be in"):
            small_head(
                torch.randn(B, N, small_head.latent_dim),
                torch.randn(B, N, small_head.context_dim),
                teacher_force_ratio=1.5,
            )

    def test_sample_non_positive_sigma(self, small_head, canonical_inputs):
        z, ctx, _, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)
        bad_sigma = sigma.clone()
        bad_sigma[0, 0, 0, 0] = -0.1
        with pytest.raises(ValueError, match="strictly positive"):
            small_head.sample_prediction(mu, bad_sigma)

    def test_sample_shape_mismatch(self, small_head, canonical_inputs):
        z, ctx, _, B, N, H = canonical_inputs
        mu, sigma = small_head(z, ctx)
        with pytest.raises(ValueError, match="same shape"):
            small_head.sample_prediction(mu, sigma[:, :1, :, :])

    def test_invalid_constructor_args(self):
        with pytest.raises(ValueError, match="latent_dim"):
            GaussianForecastHead(latent_dim=0)
        with pytest.raises(ValueError, match="horizon"):
            GaussianForecastHead(horizon=0)
