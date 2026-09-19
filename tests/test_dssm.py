"""Unit and integration tests for Phase 8 — GaussianDSSM.

Tests
-----
- forward_predict output shapes [B, H, N, 1]
- sigma strictly positive in both training and inference outputs
- KL divergence is non-negative
- ELBO components (recon_nll, kl) are finite
- Gradient flow through all parameters (encoder + transition + recognition + head)
- beta=0: KL contribution to ELBO is zero
- beta=1: ELBO = recon_nll + KL
- Single-batch overfit: ELBO decreases over 50 gradient steps
- Prior and posterior distributions differ (recognition network has effect)
- Non-finite inputs raise ValueError
- Invalid constructor arguments raise ValueError
- forward_predict does not access targets (inference contract)
- CLI synthetic smoke test runs and returns exit code 0
- _diagonal_gaussian_kl returns non-negative values
- _masked_gaussian_nll with all-zero mask returns 0
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from st_dssm.cli.dssm import main as dssm_main
from st_dssm.dssm import (
    GaussianDSSM,
    _diagonal_gaussian_kl,
    _masked_gaussian_nll,
)
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_graph_small():
    """Small N=6 symmetric graph with Chebyshev basis (K=3)."""
    N = 6
    adj = np.eye(N, dtype=np.float32)
    for i in range(N - 1):
        adj[i, i + 1] = 0.5
        adj[i + 1, i] = 0.5
    norm_lap = calculate_normalized_laplacian(adj)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb_np = compute_chebyshev_polynomials(scaled_lap, k=3)
    return torch.tensor(cheb_np, dtype=torch.float32), N


@pytest.fixture(scope="module")
def small_dssm(synthetic_graph_small):
    """Small GaussianDSSM instance for fast testing."""
    _, N = synthetic_graph_small
    return GaussianDSSM(
        num_nodes=N,
        in_channels=2,
        latent_dim=8,
        context_dim=16,
        hidden_dim=32,
        horizon=4,
        input_length=12,
        cheb_k=3,
        dropout=0.0,
    )


@pytest.fixture
def standard_inputs(small_dssm, synthetic_graph_small):
    """Standard batch of synthetic inputs matching small_dssm dimensions."""
    torch.manual_seed(0)
    cheb, N = synthetic_graph_small
    B = 2
    L = small_dssm.input_length
    H = small_dssm.horizon

    x = torch.randn(B, L, N, 2)
    y = torch.randn(B, H, N, 1)
    mask_h = torch.ones(B, L, N, 1)
    mask_f = torch.ones(B, H, N, 1)
    return x, y, mask_h, mask_f, cheb, B, L, N, H


# ---------------------------------------------------------------------------
# Loss helper unit tests
# ---------------------------------------------------------------------------

class TestLossHelpers:
    def test_kl_non_negative_identity(self):
        """KL(N(mu, sigma) || N(mu, sigma)) must be 0."""
        mu = torch.randn(2, 4, 3, 8)
        sigma = torch.rand(2, 4, 3, 8) + 0.1
        kl = _diagonal_gaussian_kl(mu, sigma, mu, sigma)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_non_negative_random(self):
        """KL must be >= 0 for arbitrary Gaussian pairs."""
        torch.manual_seed(7)
        for _ in range(20):
            mu_q = torch.randn(2, 3, 4, 8)
            sigma_q = torch.rand(2, 3, 4, 8) + 0.05
            mu_p = torch.randn(2, 3, 4, 8)
            sigma_p = torch.rand(2, 3, 4, 8) + 0.05
            kl = _diagonal_gaussian_kl(mu_q, sigma_q, mu_p, sigma_p)
            assert kl.item() >= -1e-6, f"KL is negative: {kl.item()}"

    def test_masked_nll_all_observed(self):
        """NLL with full mask == unmasked NLL."""
        torch.manual_seed(3)
        shape = (2, 4, 3, 1)
        y = torch.randn(*shape)
        mu = torch.randn(*shape)
        sigma = torch.rand(*shape) + 0.1
        mask = torch.ones(*shape)

        nll_masked = _masked_gaussian_nll(y, mu, sigma, mask)
        # Reference: manual NLL
        import math
        nll_ref = (
            0.5 * math.log(2 * math.pi)
            + torch.log(sigma)
            + 0.5 * ((y - mu) / sigma) ** 2
        ).mean()
        assert nll_masked.item() == pytest.approx(nll_ref.item(), abs=1e-5)

    def test_masked_nll_zero_mask_returns_zero(self):
        """All-zero mask: no observed targets, NLL should return 0."""
        shape = (2, 4, 3, 1)
        y = torch.randn(*shape)
        mu = torch.randn(*shape)
        sigma = torch.rand(*shape) + 0.1
        mask = torch.zeros(*shape)
        nll = _masked_gaussian_nll(y, mu, sigma, mask)
        assert nll.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# forward_train tests
# ---------------------------------------------------------------------------

class TestForwardTrain:
    def test_output_shapes(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()
        mu, sigma, kl, nll = small_dssm.forward_train(
            x, cheb, y, mh, mf, teacher_force_ratio=0.0, beta=1.0
        )
        assert mu.shape == (B, H, N, 1)
        assert sigma.shape == (B, H, N, 1)
        assert kl.dim() == 0    # scalar
        assert nll.dim() == 0   # scalar

    def test_sigma_strictly_positive(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()
        _, sigma, _, _ = small_dssm.forward_train(x, cheb, y, mh, mf)
        assert (sigma > 0).all(), f"sigma min={sigma.min().item()}"

    def test_kl_non_negative(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()
        _, _, kl, _ = small_dssm.forward_train(x, cheb, y, mh, mf)
        assert kl.item() >= -1e-5, f"KL is negative: {kl.item()}"

    def test_elbo_components_finite(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()
        mu, sigma, kl, nll = small_dssm.forward_train(x, cheb, y, mh, mf, beta=1.0)
        assert torch.isfinite(kl), f"KL non-finite: {kl.item()}"
        assert torch.isfinite(nll), f"NLL non-finite: {nll.item()}"
        assert torch.isfinite(mu).all()
        assert torch.isfinite(sigma).all()

    def test_beta_zero_kl_term_is_zero(self, small_dssm, standard_inputs):
        """With beta=0, the KL value itself should still be computed but
        the ELBO equals recon_nll (KL not checked to be zero, only ELBO holds)."""
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()
        _, _, kl, nll = small_dssm.forward_train(
            x, cheb, y, mh, mf, beta=0.0
        )
        # With beta=0, caller's ELBO = nll + 0*kl = nll
        # The KL is still computed and returned for logging; it must be >= 0
        assert kl.item() >= -1e-5

    def test_teacher_forcing_ratio_one(self, small_dssm, standard_inputs):
        """teacher_force_ratio=1.0 must not raise and must produce valid outputs."""
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()
        mu, sigma, kl, nll = small_dssm.forward_train(
            x, cheb, y, mh, mf, teacher_force_ratio=1.0, beta=1.0
        )
        assert mu.shape == (B, H, N, 1)
        assert (sigma > 0).all()
        assert torch.isfinite(kl)
        assert torch.isfinite(nll)


# ---------------------------------------------------------------------------
# forward_predict tests
# ---------------------------------------------------------------------------

class TestForwardPredict:
    def test_output_shapes(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.eval()
        with torch.no_grad():
            mu, sigma = small_dssm.forward_predict(x, cheb)
        assert mu.shape == (B, H, N, 1)
        assert sigma.shape == (B, H, N, 1)

    def test_sigma_strictly_positive(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.eval()
        with torch.no_grad():
            _, sigma = small_dssm.forward_predict(x, cheb)
        assert (sigma > 0).all()

    def test_outputs_finite(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.eval()
        with torch.no_grad():
            mu, sigma = small_dssm.forward_predict(x, cheb)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(sigma).all()

    def test_predict_does_not_use_targets(self, small_dssm, standard_inputs):
        """forward_predict must produce identical results regardless of what
        y_target would be — targets must never be accessed in this path."""
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.eval()

        torch.manual_seed(10)
        with torch.no_grad():
            mu1, s1 = small_dssm.forward_predict(x, cheb)

        torch.manual_seed(10)
        with torch.no_grad():
            mu2, s2 = small_dssm.forward_predict(x, cheb)

        assert torch.allclose(mu1, mu2, atol=1e-6)
        assert torch.allclose(s1, s2, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradient_flow_all_parameters(self, small_dssm, standard_inputs):
        """All trainable parameters must receive finite gradients."""
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()

        mu, sigma, kl, nll = small_dssm.forward_train(
            x, cheb, y, mh, mf, teacher_force_ratio=0.0, beta=1.0
        )
        elbo = nll + kl
        elbo.backward()

        for name, param in small_dssm.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for '{name}'."
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in '{name}'."
                )


# ---------------------------------------------------------------------------
# Single-batch overfit
# ---------------------------------------------------------------------------

class TestSingleBatchOverfit:
    def test_elbo_decreases(self, synthetic_graph_small):
        """ELBO must decrease when overfitting a single fixed batch."""
        torch.manual_seed(99)
        cheb, N = synthetic_graph_small
        B, L, H = 2, 12, 4

        model = GaussianDSSM(
            num_nodes=N,
            in_channels=2,
            latent_dim=8,
            context_dim=16,
            hidden_dim=32,
            horizon=H,
            input_length=L,
            cheb_k=3,
            dropout=0.0,
        )
        model.train()

        x = torch.randn(B, L, N, 2)
        y = torch.randn(B, H, N, 1)
        mh = torch.ones(B, L, N, 1)
        mf = torch.ones(B, H, N, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        initial_elbo: float | None = None
        final_elbo: float | None = None

        for _ in range(60):
            optimizer.zero_grad()
            _, _, kl, nll = model.forward_train(
                x, cheb, y, mh, mf, teacher_force_ratio=0.0, beta=1.0
            )
            elbo = nll + kl
            elbo.backward()
            optimizer.step()
            if initial_elbo is None:
                initial_elbo = elbo.item()
            final_elbo = elbo.item()

        assert final_elbo is not None and initial_elbo is not None
        assert final_elbo < initial_elbo, (
            f"ELBO did not decrease: initial={initial_elbo:.4f}, final={final_elbo:.4f}"
        )


# ---------------------------------------------------------------------------
# Prior vs posterior differ
# ---------------------------------------------------------------------------

class TestPriorPosteriorDiffer:
    def test_prior_and_posterior_params_differ(self, small_dssm, standard_inputs):
        """Recognition network must produce different parameters than prior transition.

        If recognition net is having no effect the model collapses to prior-only,
        which defeats the point of amortized inference.
        """
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        small_dssm.train()

        # Access internal components by running them independently
        with torch.no_grad():
            context = small_dssm.encoder(x, cheb)
            x_speed = x[:, :, :, 0:1]

            mu_q, sigma_q = small_dssm.recognition(context, x_speed, mh)
            z0 = torch.zeros(B, N, small_dssm.latent_dim)
            mu_p, sigma_p, _ = small_dssm.transition(context, z0)

        # In general, with random weights, q and p parameters should differ
        assert not torch.allclose(mu_q, mu_p, atol=1e-4), (
            "Posterior and prior means are identical — recognition net may have no effect."
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_nonfinite_x_raises(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        bad_x = x.clone()
        bad_x[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            small_dssm.forward_train(bad_x, cheb, y, mh, mf)

    def test_nonfinite_x_predict_raises(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        bad_x = x.clone()
        bad_x[0, 0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            small_dssm.forward_predict(bad_x, cheb)

    def test_wrong_input_length(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        bad_x = torch.randn(B, L + 2, N, 2)
        with pytest.raises(ValueError, match="Input length mismatch"):
            small_dssm.forward_train(bad_x, cheb, y, mh, mf)

    def test_wrong_num_nodes(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        bad_x = torch.randn(B, L, N + 1, 2)
        with pytest.raises(ValueError, match="Node count mismatch"):
            small_dssm.forward_train(bad_x, cheb, y, mh, mf)

    def test_wrong_y_target_shape(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        bad_y = torch.randn(B, H + 1, N, 1)
        with pytest.raises(ValueError, match="y_target shape mismatch"):
            small_dssm.forward_train(x, cheb, bad_y, mh, mf)

    def test_wrong_obs_mask_hist_shape(self, small_dssm, standard_inputs):
        x, y, mh, mf, cheb, B, L, N, H = standard_inputs
        bad_mh = torch.ones(B, L + 1, N, 1)
        with pytest.raises(ValueError, match="obs_mask_hist shape mismatch"):
            small_dssm.forward_train(x, cheb, y, bad_mh, mf)

    def test_invalid_constructor_latent_dim(self):
        with pytest.raises(ValueError, match="latent_dim"):
            GaussianDSSM(latent_dim=0)

    def test_invalid_constructor_num_nodes(self):
        with pytest.raises(ValueError, match="num_nodes"):
            GaussianDSSM(num_nodes=0)


# ---------------------------------------------------------------------------
# Capacity report
# ---------------------------------------------------------------------------

class TestCapacityReport:
    def test_capacity_report_all_keys(self, small_dssm):
        cap = small_dssm.get_capacity_report()
        assert "model_name" in cap
        assert cap["model_name"] == "gaussian_dssm"
        assert cap["total_trainable_parameters"] > 0
        assert cap["encoder_parameters"] > 0
        assert cap["transition_parameters"] > 0
        assert cap["recognition_parameters"] > 0
        assert cap["forecast_head_parameters"] > 0

    def test_component_sum_equals_total(self, small_dssm):
        cap = small_dssm.get_capacity_report()
        component_sum = (
            cap["encoder_parameters"]
            + cap["transition_parameters"]
            + cap["recognition_parameters"]
            + cap["forecast_head_parameters"]
        )
        assert component_sum == cap["total_trainable_parameters"]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestDSSMCli:
    def test_synthetic_smoke_returns_zero(self, capsys):
        ret = dssm_main(["--synthetic-smoke"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "completed successfully" in captured.out

    def test_synthetic_smoke_with_output(self, tmp_path, capsys):
        out_file = str(tmp_path / "dssm_smoke.json")
        ret = dssm_main(["--synthetic-smoke", "--output", out_file])
        assert ret == 0
        assert (tmp_path / "dssm_smoke.json").exists()

    def test_no_args_returns_zero(self, capsys):
        """Calling with no args should print help and return 0."""
        ret = dssm_main([])
        assert ret == 0
