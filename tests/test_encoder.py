"""Unit and integration tests for Phase 6 Spatial-Temporal Encoder.

Tests:
- Causal gated temporal convolution shape and strict causality
- Spatial-temporal residual block execution
- Canonical SpatialTemporalEncoder forward pass and context shapes
- Strict temporal causality (zero future leakage)
- Dimension, basis, and non-finite input validation
- Full gradient flow backpropagation
- Single-batch optimization / overfit capability
- Parameter capacity reporting
- CLI runner integration
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from st_dssm.cli.encoder import main as encoder_main
from st_dssm.encoder import (
    CausalGatedTemporalConv,
    SpatialTemporalBlock,
    SpatialTemporalEncoder,
    count_trainable_parameters,
    get_encoder_capacity_report,
)
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
)


@pytest.fixture
def synthetic_graph():
    """Create a 5-node synthetic symmetric graph and Chebyshev basis."""
    n = 5
    adj = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 0.5
        adj[i + 1, i] = 0.5

    norm_lap = calculate_normalized_laplacian(adj)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb_poly_np = compute_chebyshev_polynomials(scaled_lap, k=3)
    return torch.tensor(cheb_poly_np, dtype=torch.float32), n


def test_causal_gated_temporal_conv_shape_and_causality():
    """Test CausalGatedTemporalConv preserves length and obeys causality."""
    b, t, n, c_in, c_out = 2, 10, 4, 3, 16
    conv = CausalGatedTemporalConv(in_channels=c_in, out_channels=c_out, kernel_size=3)
    x1 = torch.randn(b, t, n, c_in)

    out1 = conv(x1)
    assert out1.shape == (b, t, n, c_out)

    # Causality test: alter inputs at t >= 4
    x2 = x1.clone()
    x2[:, 4:, :, :] = torch.randn(b, t - 4, n, c_in) * 100.0

    out2 = conv(x2)
    # Output at t <= 3 must be identical
    assert torch.allclose(out1[:, :4, :, :], out2[:, :4, :, :], atol=1e-5)


def test_causal_gated_temporal_conv_invalid_input():
    """Test error handling in CausalGatedTemporalConv."""
    with pytest.raises(ValueError, match="kernel_size must be >= 1"):
        CausalGatedTemporalConv(in_channels=2, out_channels=16, kernel_size=0)

    conv = CausalGatedTemporalConv(in_channels=2, out_channels=16, kernel_size=3)
    with pytest.raises(ValueError, match="Expected 4D input"):
        conv(torch.randn(2, 10, 4))

    with pytest.raises(ValueError, match="Input channel mismatch"):
        conv(torch.randn(2, 10, 4, 5))


def test_spatial_temporal_block_shape_and_residual(synthetic_graph):
    """Test SpatialTemporalBlock execution and shape preservation."""
    cheb_poly, n = synthetic_graph
    b, t, c_in, hidden = 2, 12, 2, 32
    block = SpatialTemporalBlock(
        in_channels=c_in,
        hidden_channels=hidden,
        num_nodes=n,
        kernel_size=3,
        cheb_k=3,
        dropout=0.1,
    )
    x = torch.randn(b, t, n, c_in)
    out = block(x, cheb_poly)

    assert out.shape == (b, t, n, hidden)
    assert torch.isfinite(out).all()


def test_spatial_temporal_encoder_forward_shape(synthetic_graph):
    """Test full SpatialTemporalEncoder forward pass and output shape."""
    cheb_poly, n = synthetic_graph
    b, l, c_in, hidden = 4, 12, 2, 64
    encoder = SpatialTemporalEncoder(
        num_nodes=n,
        in_channels=c_in,
        hidden_channels=hidden,
        input_length=l,
        cheb_k=3,
    )

    x = torch.randn(b, l, n, c_in)
    context = encoder(x, cheb_poly)

    assert context.shape == (b, l, n, hidden)
    assert torch.isfinite(context).all()


def test_spatial_temporal_encoder_strict_causality(synthetic_graph):
    """Test that future changes in input never leak into past/present encoder representations."""
    cheb_poly, n = synthetic_graph
    l, c_in, hidden = 12, 2, 64
    encoder = SpatialTemporalEncoder(
        num_nodes=n,
        in_channels=c_in,
        hidden_channels=hidden,
        input_length=l,
        cheb_k=3,
    )
    encoder.eval()

    x1 = torch.randn(1, l, n, c_in)
    x2 = x1.clone()
    # Modify steps from t=5 to t=11
    x2[:, 5:, :, :] = torch.randn(1, l - 5, n, c_in) * 20.0

    with torch.no_grad():
        c1 = encoder(x1, cheb_poly)
        c2 = encoder(x2, cheb_poly)

    # Representations for t=0,1,2,3,4 must be strictly identical
    assert torch.allclose(c1[:, :5, :, :], c2[:, :5, :, :], atol=1e-5), (
        "Future time steps leaked into past encoder context!"
    )


def test_spatial_temporal_encoder_context_summary(synthetic_graph):
    """Test get_context_summary extraction."""
    cheb_poly, n = synthetic_graph
    encoder = SpatialTemporalEncoder(num_nodes=n, in_channels=2, hidden_channels=64, input_length=12)

    x = torch.randn(2, 12, n, 2)
    context = encoder(x, cheb_poly)
    summary = encoder.get_context_summary(context)

    assert summary.shape == (2, n, 64)
    assert torch.allclose(summary, context[:, -1, :, :])

    with pytest.raises(ValueError, match="Expected context shape"):
        encoder.get_context_summary(torch.randn(2, 6, n, 64))

    # Non-finite context guard
    context_nan = context.clone()
    context_nan[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        encoder.get_context_summary(context_nan)


def test_spatial_temporal_encoder_input_validation(synthetic_graph):
    """Test strict input validation and non-finite guards."""
    cheb_poly, n = synthetic_graph
    encoder = SpatialTemporalEncoder(num_nodes=n, in_channels=2, hidden_channels=64, input_length=12)

    # Non-4D input
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        encoder(torch.randn(2, 12, n), cheb_poly)

    # Wrong length
    with pytest.raises(ValueError, match="Input length mismatch"):
        encoder(torch.randn(2, 8, n, 2), cheb_poly)

    # Wrong node count
    with pytest.raises(ValueError, match="Number of nodes mismatch"):
        encoder(torch.randn(2, 12, n + 1, 2), cheb_poly)

    # Wrong channels
    with pytest.raises(ValueError, match="Input channel mismatch"):
        encoder(torch.randn(2, 12, n, 3), cheb_poly)

    # Wrong chebyshev basis (mismatched K)
    with pytest.raises(ValueError, match="Chebyshev basis mismatch"):
        encoder(torch.randn(2, 12, n, 2), cheb_poly[:2, :, :])

    # Non-3D chebyshev basis (e.g. 2D tensor)
    with pytest.raises(ValueError, match="Expected 3D Chebyshev basis"):
        encoder(torch.randn(2, 12, n, 2), torch.randn(n, n))

    # Non-finite values
    x_nan = torch.randn(2, 12, n, 2)
    x_nan[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        encoder(x_nan, cheb_poly)


def test_spatial_temporal_encoder_gradient_flow(synthetic_graph):
    """Test full backward gradient flow across all trainable parameters."""
    cheb_poly, n = synthetic_graph
    encoder = SpatialTemporalEncoder(num_nodes=n, in_channels=2, hidden_channels=32, input_length=12)
    encoder.train()

    x = torch.randn(2, 12, n, 2, requires_grad=True)
    out = encoder(x, cheb_poly)
    loss = out.sum()
    loss.backward()

    # Verify input received gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Verify all model parameters received gradients
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} gradient contains NaN/Inf!"


def test_spatial_temporal_encoder_overfit_one_batch(synthetic_graph):
    """Test that encoder can overfit a single batch fixture across seeds to a hard threshold."""
    cheb_poly, n = synthetic_graph

    # Test optimization stability and convergence across multiple distinct seeds
    for seed in (42, 123, 777):
        torch.manual_seed(seed)
        np.random.seed(seed)

        encoder = SpatialTemporalEncoder(
            num_nodes=n,
            in_channels=2,
            hidden_channels=32,
            input_length=12,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1.5e-2)

        x = torch.randn(2, 12, n, 2)
        target = torch.randn(2, 12, n, 32)

        final_loss = None

        for _ in range(120):
            optimizer.zero_grad()
            pred = encoder(x, cheb_poly)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert final_loss is not None
        # Hard convergence threshold verifying true overfit optimization capacity
        assert final_loss < 0.1, (
            f"Seed {seed}: Encoder failed to reach hard overfit threshold (< 0.1), got {final_loss:.4f}"
        )


def test_encoder_capacity_report(synthetic_graph):
    """Test parameter count and capacity report generation."""
    _, n = synthetic_graph
    encoder = SpatialTemporalEncoder(num_nodes=n, in_channels=2, hidden_channels=64, input_length=12)

    params = count_trainable_parameters(encoder)
    assert params > 0

    report = get_encoder_capacity_report(encoder, reference_param_count=params)
    assert report["trainable_parameters"] == params
    assert report["capacity_ratio_valid"] is True
    assert report["capacity_ratio_vs_reference"] == pytest.approx(1.0)

    report_no_ref = get_encoder_capacity_report(encoder)
    assert report_no_ref["capacity_ratio_valid"] is False


def test_encoder_cli_synthetic_smoke(capsys):
    """Test st-dssm-encoder CLI synthetic smoke option."""
    ret = encoder_main(["--synthetic-smoke"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Synthetic encoder verification completed successfully." in captured.out


def test_spatial_temporal_encoder_node_permutation_equivariance():
    """Test that the encoder is strictly node-permutation equivariant per ADR-0007.

    Permuting node indices in both input X and graph adjacency A produces an identically
    permuted context output representation:
        Encoder(P X, P Cheb P^T) == P Encoder(X, Cheb)
    """
    n = 6
    torch.manual_seed(123)
    np.random.seed(123)

    adj = np.random.uniform(0.1, 0.9, size=(n, n)).astype(np.float32)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 1.0)

    perm = np.random.permutation(n)
    p_mat = np.eye(n, dtype=np.float32)[perm]
    adj_perm = p_mat @ adj @ p_mat.T

    norm_lap = calculate_normalized_laplacian(adj)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb = torch.tensor(compute_chebyshev_polynomials(scaled_lap, k=3), dtype=torch.float32)

    norm_lap_perm = calculate_normalized_laplacian(adj_perm)
    scaled_lap_perm, _ = calculate_scaled_laplacian(norm_lap_perm)
    cheb_perm = torch.tensor(compute_chebyshev_polynomials(scaled_lap_perm, k=3), dtype=torch.float32)

    encoder = SpatialTemporalEncoder(
        num_nodes=n,
        in_channels=2,
        hidden_channels=32,
        input_length=12,
        dropout=0.0,
    )
    encoder.eval()

    x = torch.randn(2, 12, n, 2)
    x_perm = x[:, :, perm, :]

    with torch.no_grad():
        out = encoder(x, cheb)
        out_perm = encoder(x_perm, cheb_perm)

    expected_out_perm = out[:, :, perm, :]
    assert torch.allclose(out_perm, expected_out_perm, atol=1e-5), (
        f"Node-permutation equivariance broken! Max diff: {torch.max(torch.abs(out_perm - expected_out_perm)).item()}"
    )


def test_encoder_cli_bare_output_filename(tmp_path, monkeypatch):
    """Test CLI runs cleanly when output path is a bare filename in current working directory."""
    monkeypatch.chdir(tmp_path)
    bare_file = "bare_report.json"
    ret = encoder_main(["--synthetic-smoke", "--output", bare_file])
    assert ret == 0
    assert (tmp_path / bare_file).exists()


def test_encoder_cli_unrecognized_graph_and_missing_metadata(tmp_path):
    """Test CLI fails fast on unrecognized graph names or missing metadata."""
    # 1. Unrecognized graph name
    fake_graph = tmp_path / "malicious.pkl"
    fake_graph.write_text("fake")

    from st_dssm.cli.encoder import verify_encoder_on_dataset
    from st_dssm.graph import GraphError

    with pytest.raises(ValueError, match="Unrecognized graph adjacency file"):
        verify_encoder_on_dataset(adj_mx_path=str(fake_graph))

    # 2. Asymmetric graph without symmetrization raises GraphError
    import pickle

    from st_dssm.io import save_processed_artifact

    artifact_dir = str(tmp_path / "asym_art")
    s, l, n = 2, 12, 4
    save_processed_artifact(
        artifact_dir,
        {
            "test_X": np.random.randn(s, l, n, 1).astype(np.float32),
            "test_X_mask": np.ones((s, l, n, 1), dtype=np.float32),
            "scaler_means": np.zeros(n, dtype=np.float32),
            "scaler_stds": np.ones(n, dtype=np.float32),
        },
        {"schema_version": "1.0", "sensor_ids": ["s1", "s2", "s3", "s4"]},
    )
    asym_graph = str(tmp_path / "adj_mx_bay.pkl")
    asym_adj = np.eye(n, dtype=np.float32)
    asym_adj[0, 1] = 1.0  # Asymmetric
    with open(asym_graph, "wb") as f:
        pickle.dump((["s1", "s2", "s3", "s4"], {}, asym_adj), f)

    with pytest.raises(GraphError, match="Adjacency matrix must be symmetric"):
        verify_encoder_on_dataset(
            artifact_dir=artifact_dir,
            adj_mx_path=asym_graph,
            allow_unverified_graph=True,
            symmetrize_graph=False,
        )
