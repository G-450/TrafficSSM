"""Command line interface for training and evaluating deterministic baselines (st-dssm-baseline).

Implements end-to-end execution for:
- Historical Persistence baseline
- Deterministic ST-GCN baseline (capacity-controlled graph baseline)
Adheres to ADR-0007 and ADR-0008.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from st_dssm.baselines.persistence import HistoricalPersistence
from st_dssm.baselines.st_gcn import DeterministicSTGCN, get_capacity_report
from st_dssm.cli.evaluate import run_evaluation
from st_dssm.data import EXPECTED_FILES, ChecksumMismatchError, compute_md5
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
    symmetrize_adjacency,
)
from st_dssm.io import load_and_validate_artifact
from st_dssm.training import (
    EarlyStopping,
    MaskedMAELoss,
    save_checkpoint,
    set_seed,
)
from st_dssm.validator import validate_graph


def run_persistence_baseline(
    config: dict[str, Any],
    output_dir: str = "artifacts/results",
    save_plots: bool = True,
) -> dict[str, Any]:
    """Execute Historical Persistence baseline prediction and evaluation."""
    print("Running Historical Persistence Baseline...")
    artifact_dir = config.get("artifact_dir", "data/processed")
    split = config.get("split", "test")
    forecast_horizon = int(config.get("forecast_horizon", 12))

    arrays, _metadata = load_and_validate_artifact(artifact_dir)

    x_key = f"{split}_X" if f"{split}_X" in arrays else f"x_{split}"
    x_data = arrays[x_key]  # [S, L, N, 1]

    persistence = HistoricalPersistence(forecast_horizon=forecast_horizon)
    y_pred = persistence.predict(x_data, forecast_horizon=forecast_horizon)

    os.makedirs(output_dir, exist_ok=True)
    run_id = f"baseline-persistence-{split}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    predictions_path = os.path.join(output_dir, f"{run_id}_predictions.npz")

    np.savez_compressed(predictions_path, predictions=y_pred)
    print(f"Predictions saved to: {predictions_path}")

    # Evaluate using Phase 4 evaluation pipeline
    eval_config = {
        "schema_version": "1.0",
        "run_id": run_id,
        "model_name": "historical_persistence",
        "artifact_dir": artifact_dir,
        "predictions_path": predictions_path,
        "split": split,
        "cadence_minutes": int(config.get("cadence_minutes", 5)),
        "output_dir": output_dir,
    }

    manifest, manifest_path = run_evaluation(eval_config, output_dir=output_dir, save_plots=save_plots)
    return {"run_id": run_id, "manifest_path": manifest_path, "manifest": manifest}


def train_and_eval_st_gcn(
    config: dict[str, Any],
    output_dir: str = "artifacts/results",
    checkpoint_dir: str = "artifacts/checkpoints",
    save_plots: bool = True,
) -> dict[str, Any]:
    """Train and evaluate the deterministic ST-GCN baseline."""
    print("Executing Deterministic ST-GCN Baseline Pipeline...")

    seed = int(config.get("seed", 2026))
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {device}")

    # 1. Load Dataset Artifact
    artifact_dir = config.get("artifact_dir", "data/processed")
    arrays, _metadata = load_and_validate_artifact(artifact_dir)

    train_x = arrays["train_X"]
    train_x_mask = arrays["train_X_mask"]
    train_y = arrays["train_Y"]
    train_y_mask = arrays["train_Y_mask"]

    val_x = arrays["val_X"]
    val_x_mask = arrays["val_X_mask"]
    val_y = arrays["val_Y"]
    val_y_mask = arrays["val_Y_mask"]

    split = config.get("split", "test")
    test_x_key = f"{split}_X" if f"{split}_X" in arrays else f"x_{split}"
    test_x_mask_key = f"{split}_X_mask" if f"{split}_X_mask" in arrays else f"x_{split}_mask"
    test_x = arrays[test_x_key]
    test_x_mask = arrays[test_x_mask_key]

    # 2. Graph & Chebyshev Polynomials
    adj_mx_path = config.get("adj_mx_path", "data/raw/adj_mx_bay.pkl")
    cheb_k = int(config.get("cheb_k", 3))

    if not os.path.exists(adj_mx_path):
        raise FileNotFoundError(
            f"Required graph adjacency file not found at: {adj_mx_path}. "
            "Please acquire the canonical dataset using 'st-dssm-provenance --download'."
        )

    # Verify checksum if filename matches pinned canonical release
    fname = os.path.basename(adj_mx_path)
    if fname in EXPECTED_FILES:
        observed_md5 = compute_md5(adj_mx_path)
        expected_md5 = EXPECTED_FILES[fname]
        if observed_md5 != expected_md5:
            raise ChecksumMismatchError(
                f"MD5 checksum mismatch for {fname}: expected {expected_md5}, got {observed_md5}"
            )

    with open(adj_mx_path, "rb") as f:
        sensor_ids_graph_raw, _, adj_mx = pickle.load(f, encoding="latin1")

    graph_sensor_ids = [str(sid) for sid in sensor_ids_graph_raw]
    expected_sensor_ids = _metadata.get("sensor_ids", [])
    if expected_sensor_ids:
        validate_graph(adj_mx, graph_sensor_ids, expected_sensor_ids)

    # Symmetrize if directed (as standard in spectral graph convolutions)
    if not np.allclose(adj_mx, adj_mx.T, atol=1e-5):
        print("Note: Raw graph adjacency is directed; applying canonical symmetrization (W + W.T)/2 for Chebyshev spectral convolution.", flush=True)
        adj_mx = symmetrize_adjacency(adj_mx, method="average")

    norm_lap = calculate_normalized_laplacian(adj_mx)
    scaled_lap, _lambda_max = calculate_scaled_laplacian(norm_lap)
    cheb_poly_np = compute_chebyshev_polynomials(scaled_lap, k=cheb_k)
    cheb_poly = torch.tensor(cheb_poly_np, dtype=torch.float32, device=device)

    # 3. Model instantiation & Capacity Report
    num_nodes = train_x.shape[2]
    hidden_channels = int(config.get("hidden_channels", 64))
    forecast_horizon = int(config.get("forecast_horizon", 12))
    dropout = float(config.get("dropout", 0.1))

    model = DeterministicSTGCN(
        num_nodes=num_nodes,
        in_channels=2,
        hidden_channels=hidden_channels,
        input_length=train_x.shape[1],
        forecast_horizon=forecast_horizon,
        cheb_k=cheb_k,
        dropout=dropout,
    ).to(device)

    capacity_report = get_capacity_report(model, "deterministic_st_gcn")
    print(f"Model capacity: {capacity_report['trainable_parameters']} trainable parameters.", flush=True)

    # 4. DataLoaders using zero-copy tensor views and batch-level channel concatenation
    batch_size = int(config.get("batch_size", 64))
    train_dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_x_mask),
        torch.from_numpy(train_y),
        torch.from_numpy(train_y_mask),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_x),
        torch.from_numpy(val_x_mask),
        torch.from_numpy(val_y),
        torch.from_numpy(val_y_mask),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Training loop
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MaskedMAELoss()

    max_epochs = int(config.get("max_epochs", 100))
    patience = int(config.get("patience", 15))
    min_delta = float(config.get("min_delta", 1e-4))
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")

    run_id = f"baseline-st_gcn-{split}-seed{seed}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")

    print(f"Beginning training (Max epochs={max_epochs}, Patience={patience}, Min delta={min_delta})...", flush=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_abs_err = 0.0
        total_train_valid_count = 0.0

        for bx, bxm, by, bym in train_loader:
            bx_in = torch.cat([bx, bxm.float()], dim=-1).to(device)
            by = by.to(device)
            bym = bym.to(device)

            optimizer.zero_grad()
            pred = model(bx_in, cheb_poly)
            loss = criterion(pred, by, bym)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                diff = torch.abs(pred - by) * bym
                total_train_abs_err += torch.sum(diff).item()
                total_train_valid_count += torch.sum(bym).item()

        avg_train_loss = total_train_abs_err / max(1.0, total_train_valid_count)

        # Validation with mathematically exact global sum over valid targets
        model.eval()
        total_val_abs_err = 0.0
        total_val_valid_count = 0.0
        with torch.no_grad():
            for bx, bxm, by, bym in val_loader:
                bx_in = torch.cat([bx, bxm.float()], dim=-1).to(device)
                by = by.to(device)
                bym = bym.to(device)

                pred = model(bx_in, cheb_poly)
                diff = torch.abs(pred - by) * bym
                total_val_abs_err += torch.sum(diff).item()
                total_val_valid_count += torch.sum(bym).item()

        avg_val_loss = total_val_abs_err / max(1.0, total_val_valid_count)

        should_stop = early_stopping.step(avg_val_loss, model, epoch)
        print(f"Epoch {epoch:03d} | Train MAE (norm): {avg_train_loss:.4f} | Val MAE (norm): {avg_val_loss:.4f} | Best Val: {early_stopping.best_score:.4f} (Ep {early_stopping.best_epoch})", flush=True)

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best weights from epoch {early_stopping.best_epoch}.", flush=True)
            break

    # Restore best checkpoint
    early_stopping.restore_best_weights(model)
    save_checkpoint(
        best_ckpt_path,
        model,
        optimizer,
        epoch=early_stopping.best_epoch,
        val_loss=early_stopping.best_score or 0.0,
        config=config,
    )
    print(f"Best checkpoint saved to: {best_ckpt_path}", flush=True)

    # 6. Evaluation on target split
    model.eval()
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_x_mask))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_preds = []

    with torch.no_grad():
        for bx, bxm in test_loader:
            bx_in = torch.cat([bx, bxm.float()], dim=-1).to(device)
            pred = model(bx_in, cheb_poly)
            test_preds.append(pred.cpu().numpy())

    y_pred = np.concatenate(test_preds, axis=0)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, f"{run_id}_predictions.npz")
    np.savez_compressed(predictions_path, predictions=y_pred)
    print(f"Predictions saved to: {predictions_path}")

    # 7. Run evaluation and create manifest
    eval_config = {
        "schema_version": "1.0",
        "run_id": run_id,
        "model_name": "deterministic_st_gcn",
        "artifact_dir": artifact_dir,
        "predictions_path": predictions_path,
        "split": split,
        "cadence_minutes": int(config.get("cadence_minutes", 5)),
        "output_dir": output_dir,
    }

    manifest, manifest_path = run_evaluation(eval_config, output_dir=output_dir, save_plots=save_plots)
    return {
        "run_id": run_id,
        "manifest_path": manifest_path,
        "checkpoint_path": best_ckpt_path,
        "manifest": manifest,
        "capacity_report": capacity_report,
    }


def run_synthetic_baseline_smoke(output_dir: str = "artifacts/results") -> None:
    """Run an end-to-end synthetic baseline smoke test for both Persistence and ST-GCN."""
    print("Running Synthetic Baseline Smoke Test (Persistence & ST-GCN)...")
    set_seed(42)

    S, L, H, N = 16, 12, 12, 5
    x_synth = np.random.normal(0.0, 1.0, (S, L, N, 1)).astype(np.float32)
    x_mask = np.ones((S, L, N, 1), dtype=np.float32)
    y_synth = np.random.normal(0.0, 1.0, (S, H, N, 1)).astype(np.float32)
    y_mask = np.ones((S, H, N, 1), dtype=np.float32)

    # 1. Persistence Test
    persistence = HistoricalPersistence(forecast_horizon=H)
    p_pred = persistence.predict(x_synth)
    assert p_pred.shape == (S, H, N, 1)
    np.testing.assert_allclose(p_pred[:, 0, :, :], x_synth[:, -1, :, :])
    print("  [PASSED] Synthetic Persistence Prediction check.")

    # 2. ST-GCN Forward & Backward Test
    adj_synth = np.eye(N, dtype=np.float32)
    for i in range(N - 1):
        adj_synth[i, i + 1] = 0.5
        adj_synth[i + 1, i] = 0.5

    norm_lap = calculate_normalized_laplacian(adj_synth)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb_poly_np = compute_chebyshev_polynomials(scaled_lap, k=3)
    cheb_poly = torch.tensor(cheb_poly_np, dtype=torch.float32)

    model = DeterministicSTGCN(
        num_nodes=N,
        in_channels=2,
        hidden_channels=16,
        input_length=L,
        forecast_horizon=H,
        cheb_k=3,
    )

    in_tensor = torch.tensor(
        np.concatenate([x_synth, x_mask], axis=-1), dtype=torch.float32
    )
    target_tensor = torch.tensor(y_synth, dtype=torch.float32)
    mask_tensor = torch.tensor(y_mask, dtype=torch.float32)

    out = model(in_tensor, cheb_poly)
    assert out.shape == (S, H, N, 1)
    print("  [PASSED] Synthetic ST-GCN Forward pass shape check.")

    criterion = MaskedMAELoss()
    loss = criterion(out, target_tensor, mask_tensor)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
    print("  [PASSED] Synthetic ST-GCN Gradient flow check.")

    # Check capacity reporting
    cap = get_capacity_report(model, "synthetic_st_gcn")
    assert cap["trainable_parameters"] > 0
    print(f"  [PASSED] Synthetic capacity report: {cap['trainable_parameters']} parameters.")
    print("Synthetic baseline smoke test completed successfully.")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for st-dssm-baseline."""
    parser = argparse.ArgumentParser(
        description="ST-DSSM Phase 5 Baseline Runner (Persistence & ST-GCN)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["persistence", "st_gcn"],
        default="persistence",
        help="Baseline model architecture to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to baseline YAML configuration file.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run end-to-end synthetic baseline smoke test.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="data/processed",
        help="Path to Phase 3 processed artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/results",
        help="Directory to save predictions, run manifests, and plots.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Directory to save trained model checkpoints.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate ('test', 'val', 'train').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--adj-mx-path",
        type=str,
        default=None,
        help="Path to graph adjacency pickle file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs for learned models.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Generate evaluation plots.",
    )
    parser.add_argument(
        "--no-plots",
        dest="save_plots",
        action="store_false",
        help="Disable evaluation plot generation.",
    )

    args = parser.parse_args(argv)

    try:
        if args.synthetic_smoke:
            run_synthetic_baseline_smoke(output_dir=args.output_dir)
            return 0

        config: dict[str, Any] = {}
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        # Merge CLI arguments into config
        config.setdefault("artifact_dir", args.artifact_dir)
        config.setdefault("output_dir", args.output_dir)
        config.setdefault("checkpoint_dir", args.checkpoint_dir)
        config.setdefault("split", args.split)
        config.setdefault("seed", args.seed)
        config.setdefault("max_epochs", args.epochs)
        if args.adj_mx_path:
            config["adj_mx_path"] = args.adj_mx_path

        model_type = config.get("model", args.model)

        out_dir = config.get("output_dir", args.output_dir)
        ckpt_dir = config.get("checkpoint_dir", args.checkpoint_dir)

        if model_type == "persistence":
            run_persistence_baseline(
                config=config,
                output_dir=out_dir,
                save_plots=args.save_plots,
            )
        elif model_type == "st_gcn":
            train_and_eval_st_gcn(
                config=config,
                output_dir=out_dir,
                checkpoint_dir=ckpt_dir,
                save_plots=args.save_plots,
            )
        else:
            print(f"Unknown baseline model: {model_type}", file=sys.stderr)
            return 1

        return 0

    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"Baseline runner error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
