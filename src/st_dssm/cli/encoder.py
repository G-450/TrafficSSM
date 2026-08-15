"""Command line interface for Spatial-Temporal Encoder verification (st-dssm-encoder).

Provides commands to inspect encoder architecture, test parameter capacity,
run synthetic smoke tests, and execute end-to-end verification on canonical dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from st_dssm.data import EXPECTED_FILES, ChecksumMismatchError, compute_md5
from st_dssm.encoder import (
    SpatialTemporalEncoder,
    get_encoder_capacity_report,
)
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
    symmetrize_adjacency,
)
from st_dssm.io import load_and_validate_artifact
from st_dssm.training import set_seed
from st_dssm.validator import validate_graph


def run_synthetic_encoder_smoke(device: torch.device | None = None) -> dict[str, Any]:
    """Run end-to-end synthetic verification of the SpatialTemporalEncoder.

    Verifies:
        1. Forward shape integrity [B, 12, N, 2] -> [B, 12, N, 64]
        2. Strict temporal causality (no future leakage)
        3. Backward gradient flow across all trainable parameters
        4. Parameter capacity reporting

    Returns:
        Dictionary report of verification results.
    """
    print("Running SpatialTemporalEncoder Synthetic Smoke Test...", flush=True)
    dev = device or torch.device("cpu")
    set_seed(42)

    b, l, n, c_in, hidden = 4, 12, 6, 2, 64
    x_synth = torch.randn(b, l, n, c_in, device=dev)

    # 1. Synthetic symmetric graph
    adj = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 0.5
        adj[i + 1, i] = 0.5

    norm_lap = calculate_normalized_laplacian(adj)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb_poly_np = compute_chebyshev_polynomials(scaled_lap, k=3)
    cheb_poly = torch.tensor(cheb_poly_np, dtype=torch.float32, device=dev)

    # 2. Instantiate encoder
    encoder = SpatialTemporalEncoder(
        num_nodes=n,
        in_channels=c_in,
        hidden_channels=hidden,
        input_length=l,
        cheb_k=3,
        dropout=0.1,
    ).to(dev)

    # 3. Forward pass
    context = encoder(x_synth, cheb_poly)
    assert context.shape == (b, l, n, hidden), f"Expected shape {(b, l, n, hidden)}, got {context.shape}"
    assert torch.isfinite(context).all(), "Context contains non-finite values"
    print("  [PASSED] Forward pass shape and finiteness check.", flush=True)

    # 4. Context summary
    summary = encoder.get_context_summary(context)
    assert summary.shape == (b, n, hidden), f"Expected summary shape {(b, n, hidden)}, got {summary.shape}"
    print("  [PASSED] Context sequence summary extraction check.", flush=True)

    # 5. Strict causality verification
    x1 = torch.randn(1, l, n, c_in, device=dev)
    x2 = x1.clone()
    # Modify future time steps t >= 4
    x2[:, 4:, :, :] = torch.randn(1, l - 4, n, c_in, device=dev) * 50.0

    encoder.eval()
    with torch.no_grad():
        c1 = encoder(x1, cheb_poly)
        c2 = encoder(x2, cheb_poly)

    # Steps 0..3 must match exactly
    assert torch.allclose(c1[:, :4, :, :], c2[:, :4, :, :], atol=1e-5), "Causality violation detected!"
    print("  [PASSED] Strict temporal causality check (zero future leakage).", flush=True)

    # 6. Backward gradient flow check
    encoder.train()
    out = encoder(x_synth, cheb_poly)
    loss = out.sum()
    loss.backward()

    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    print("  [PASSED] Full gradient backpropagation check.", flush=True)

    # 7. Capacity report
    cap = get_encoder_capacity_report(encoder)
    print(f"  [PASSED] Parameter count: {cap['trainable_parameters']} trainable parameters.", flush=True)
    print("Synthetic encoder verification completed successfully.", flush=True)

    return {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "capacity_report": cap,
        "causality_verified": True,
        "gradient_flow_verified": True,
    }


def verify_encoder_on_dataset(
    artifact_dir: str = "data/processed",
    adj_mx_path: str = "data/raw/adj_mx_bay.pkl",
    split: str = "test",
    batch_size: int = 64,
    device: torch.device | None = None,
    allow_unverified_graph: bool = False,
    symmetrize_graph: bool = True,
) -> dict[str, Any]:
    """Verify SpatialTemporalEncoder execution on canonical preprocessed dataset.

    Args:
        artifact_dir: Path to preprocessed Phase 3 artifact.
        adj_mx_path: Path to graph adjacency pickle file.
        split: Partition to verify ('test', 'val', 'train').
        batch_size: Batch size for inference.
        device: Compute device.
        allow_unverified_graph: Whether to skip checksum check for custom/synthetic test graphs.
        symmetrize_graph: Whether to apply canonical symmetrization (W + W.T)/2 per ADR-0009.

    Returns:
        Dictionary report of verification.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verifying SpatialTemporalEncoder on canonical dataset ({split} split, device={dev})...", flush=True)

    # 1. Load and validate artifact
    arrays, metadata = load_and_validate_artifact(artifact_dir)
    x_key = f"{split}_X" if f"{split}_X" in arrays else f"x_{split}"
    x_mask_key = f"{split}_X_mask" if f"{split}_X_mask" in arrays else f"x_{split}_mask"

    x = arrays[x_key]
    x_mask = arrays[x_mask_key]
    num_samples, seq_len, num_nodes, _ = x.shape

    # 2. Load and validate graph
    if not os.path.exists(adj_mx_path):
        raise FileNotFoundError(f"Required graph adjacency file not found at: {adj_mx_path}")

    fname = os.path.basename(adj_mx_path)
    if not allow_unverified_graph:
        if fname not in EXPECTED_FILES:
            raise ValueError(
                f"Unrecognized graph adjacency file: '{fname}'. Expected canonical pinned graph '{list(EXPECTED_FILES.keys())}'."
            )

        observed_md5 = compute_md5(adj_mx_path)
        expected_md5 = EXPECTED_FILES[fname]
        if observed_md5 != expected_md5:
            raise ChecksumMismatchError(
                f"MD5 checksum mismatch for {fname}: expected {expected_md5}, got {observed_md5}"
            )

    with open(adj_mx_path, "rb") as f:
        sensor_ids_raw, _, adj_mx = pickle.load(f, encoding="latin1")

    graph_sensor_ids = [str(sid) for sid in sensor_ids_raw]
    expected_sensor_ids = metadata.get("sensor_ids", [])
    if not expected_sensor_ids:
        raise ValueError(
            f"Sensor metadata 'sensor_ids' is missing or empty in dataset artifact metadata at {artifact_dir}."
        )
    validate_graph(adj_mx, graph_sensor_ids, expected_sensor_ids)

    # Explicit symmetrization per ADR-0009
    if symmetrize_graph and not np.allclose(adj_mx, adj_mx.T, atol=1e-5):
        print("Note: Applying canonical symmetrization (W + W.T)/2 per ADR-0009 for Chebyshev spectral convolution.", flush=True)
        adj_mx = symmetrize_adjacency(adj_mx, method="average")

    norm_lap = calculate_normalized_laplacian(adj_mx)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb_poly_np = compute_chebyshev_polynomials(scaled_lap, k=3)
    cheb_poly = torch.tensor(cheb_poly_np, dtype=torch.float32, device=dev)

    # 3. Instantiate encoder
    encoder = SpatialTemporalEncoder(
        num_nodes=num_nodes,
        in_channels=2,
        hidden_channels=64,
        input_length=seq_len,
        cheb_k=3,
        dropout=0.1,
    ).to(dev)

    cap = get_encoder_capacity_report(encoder)
    print(f"Canonical Encoder Capacity: {cap['trainable_parameters']} trainable parameters.", flush=True)

    # 4. Stream evaluation
    encoder.eval()
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(x_mask))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_batches = len(loader)
    print(f"Encoding {num_samples} samples across {total_batches} batches...", flush=True)

    context_shapes = []
    with torch.no_grad():
        for bx, bxm in loader:
            bx_in = torch.cat([bx, bxm.float()], dim=-1).to(dev)
            context = encoder(bx_in, cheb_poly)
            context_shapes.append(context.shape)

    assert len(context_shapes) == total_batches
    print(f"Successfully processed all {total_batches} batches. Context shape per batch: {context_shapes[0]}", flush=True)

    return {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "num_samples": num_samples,
        "seq_len": seq_len,
        "num_nodes": num_nodes,
        "context_dimension": 64,
        "symmetrize_graph_applied": bool(symmetrize_graph),
        "capacity_report": cap,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for st-dssm-encoder."""
    parser = argparse.ArgumentParser(
        description="ST-DSSM Phase 6 Spatial-Temporal Encoder verification."
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run end-to-end synthetic encoder smoke test.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="data/processed",
        help="Path to Phase 3 processed artifact directory.",
    )
    parser.add_argument(
        "--adj-mx-path",
        type=str,
        default="data/raw/adj_mx_bay.pkl",
        help="Path to graph adjacency pickle file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to verify ('test', 'val', 'train').",
    )
    parser.add_argument(
        "--symmetrize-graph",
        action="store_true",
        default=True,
        help="Apply canonical (W + W.T)/2 symmetrization for Chebyshev convolution per ADR-0009.",
    )
    parser.add_argument(
        "--no-symmetrize-graph",
        dest="symmetrize_graph",
        action="store_false",
        help="Do not symmetrize graph adjacency; requires graph to be strictly symmetric.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write verification summary JSON.",
    )

    args = parser.parse_args(argv)

    try:
        if args.synthetic_smoke:
            report = run_synthetic_encoder_smoke()
        else:
            report = verify_encoder_on_dataset(
                artifact_dir=args.artifact_dir,
                adj_mx_path=args.adj_mx_path,
                split=args.split,
                symmetrize_graph=args.symmetrize_graph,
            )

        if args.output:
            if out_dir := os.path.dirname(args.output):
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output}", flush=True)

        return 0

    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"Encoder verification error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
