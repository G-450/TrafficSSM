"""Command-line interface for ST-DSSM Phase 8 DSSM integration smoke test.

Provides an end-to-end synthetic verification of the GaussianDSSM:
    - forward_train: encoder + recognition + prior + forecast head + ELBO
    - forward_predict: encoder + prior + forecast head (no posterior, no teacher forcing)
    - Backward gradient flow through all components
    - Parameter capacity report

Entry point: st-dssm-dssm  (registered in pyproject.toml)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from st_dssm.dssm import GaussianDSSM
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
)
from st_dssm.training import set_seed


def run_synthetic_dssm_smoke(
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run end-to-end synthetic verification of the GaussianDSSM.

    Verifies:
        1. forward_train output shapes [B, H, N, 1]
        2. sigma strictly positive in predictions
        3. KL >= 0
        4. ELBO components are finite
        5. Backward gradient flow through all sub-modules
        6. forward_predict output shapes and sigma positivity
        7. Parameter capacity report

    Args:
        device: Compute device. Defaults to CPU.

    Returns:
        Dictionary report of verification results.
    """
    print("Running GaussianDSSM Synthetic Smoke Test...", flush=True)
    dev = device or torch.device("cpu")
    set_seed(42)

    # Synthetic model dimensions (smaller than canonical for speed)
    B, L, N, H = 2, 12, 6, 12
    in_channels = 2
    latent_dim = 8
    context_dim = 16
    hidden_dim = 32

    # Build synthetic symmetric graph and Chebyshev basis
    adj = np.eye(N, dtype=np.float32)
    for i in range(N - 1):
        adj[i, i + 1] = 0.5
        adj[i + 1, i] = 0.5

    norm_lap = calculate_normalized_laplacian(adj)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb_np = compute_chebyshev_polynomials(scaled_lap, k=3)
    cheb = torch.tensor(cheb_np, dtype=torch.float32, device=dev)

    # Instantiate DSSM with small synthetic dims
    model = GaussianDSSM(
        num_nodes=N,
        in_channels=in_channels,
        latent_dim=latent_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        horizon=H,
        input_length=L,
        cheb_k=3,
        dropout=0.0,
    ).to(dev)
    model.train()

    # Synthetic inputs
    x = torch.randn(B, L, N, in_channels, device=dev)
    y_target = torch.randn(B, H, N, 1, device=dev)
    obs_mask_hist = torch.ones(B, L, N, 1, device=dev)
    obs_mask_fore = torch.ones(B, H, N, 1, device=dev)

    # --- 1. forward_train ---
    mu_pred, sigma_pred, kl, recon_nll = model.forward_train(
        x=x,
        cheb_polynomials=cheb,
        y_target=y_target,
        obs_mask_hist=obs_mask_hist,
        obs_mask_fore=obs_mask_fore,
        teacher_force_ratio=0.5,
        beta=1.0,
    )

    # Shape checks
    assert mu_pred.shape == (B, H, N, 1), f"mu_pred shape wrong: {mu_pred.shape}"
    assert sigma_pred.shape == (B, H, N, 1), f"sigma_pred shape wrong: {sigma_pred.shape}"
    print("  [PASSED] forward_train output shapes correct.", flush=True)

    # Sigma strictly positive
    assert (sigma_pred > 0).all(), "sigma_pred contains non-positive values."
    print("  [PASSED] sigma_pred strictly positive.", flush=True)

    # KL >= 0
    assert kl.item() >= 0.0, f"KL is negative: {kl.item():.6f}"
    print(f"  [PASSED] KL >= 0 (KL = {kl.item():.4f}).", flush=True)

    # ELBO components finite
    assert torch.isfinite(recon_nll), f"recon_nll is non-finite: {recon_nll.item()}"
    assert torch.isfinite(kl), f"KL is non-finite: {kl.item()}"
    print(
        f"  [PASSED] ELBO components finite "
        f"(recon_nll={recon_nll.item():.4f}, kl={kl.item():.4f}).",
        flush=True,
    )

    # --- 2. Gradient flow ---
    elbo = recon_nll + kl
    elbo.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter '{name}'."
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient in parameter '{name}'."
            )
    print("  [PASSED] Gradient flow through all parameters.", flush=True)

    # --- 3. forward_predict (inference, prior only) ---
    model.eval()
    with torch.no_grad():
        mu_p, sigma_p = model.forward_predict(x=x, cheb_polynomials=cheb)

    assert mu_p.shape == (B, H, N, 1), f"forward_predict mu shape wrong: {mu_p.shape}"
    assert sigma_p.shape == (B, H, N, 1), f"forward_predict sigma shape wrong: {sigma_p.shape}"
    assert (sigma_p > 0).all(), "forward_predict sigma contains non-positive values."
    assert torch.isfinite(mu_p).all(), "forward_predict mu contains non-finite values."
    assert torch.isfinite(sigma_p).all(), "forward_predict sigma contains non-finite values."
    print("  [PASSED] forward_predict shapes, sigma positivity, and finiteness.", flush=True)

    # --- 4. Capacity report ---
    cap = model.get_capacity_report()
    assert cap["total_trainable_parameters"] > 0, "No trainable parameters found."
    print(
        f"  [PASSED] Capacity: {cap['total_trainable_parameters']} total trainable params "
        f"(encoder={cap['encoder_parameters']}, "
        f"transition={cap['transition_parameters']}, "
        f"recognition={cap['recognition_parameters']}, "
        f"forecast_head={cap['forecast_head_parameters']}).",
        flush=True,
    )

    print("GaussianDSSM synthetic smoke test completed successfully.", flush=True)

    return {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "capacity_report": cap,
        "kl_value": float(kl.item()),
        "recon_nll_value": float(recon_nll.item()),
        "shapes_verified": True,
        "gradient_flow_verified": True,
        "inference_prior_only_verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for st-dssm-dssm."""
    parser = argparse.ArgumentParser(
        description="ST-DSSM Phase 8 GaussianDSSM integration verification."
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run end-to-end synthetic DSSM integration smoke test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write verification summary JSON.",
    )

    args = parser.parse_args(argv)

    if not args.synthetic_smoke:
        parser.print_help()
        print(
            "\nNote: Only --synthetic-smoke is available in Phase 8. "
            "Full dataset training is Phase 9.",
            flush=True,
        )
        return 0

    try:
        report = run_synthetic_dssm_smoke()

        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output}", flush=True)

        return 0

    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"DSSM verification error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
