"""Command line interface for training and evaluating deterministic baselines (st-dssm-baseline).

Implements end-to-end execution for:
- Historical Persistence baseline
- Deterministic ST-GCN baseline (capacity-controlled graph baseline)
Adheres to ADR-0007 and ADR-0008.
"""

from __future__ import annotations

import argparse
import glob
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
from st_dssm.mask_generator import MaskGenerator, resolve_mask_generator
from st_dssm.mask_generator import condition_tag as build_condition_tag
from st_dssm.training import (
    EarlyStopping,
    MaskedMAELoss,
    load_training_state,
    save_checkpoint,
    save_training_state,
    set_seed,
)
from st_dssm.validator import validate_graph


def _resolve_condition(
    config: dict[str, Any], num_nodes: int, seed: int | None, trains: bool = False
) -> tuple[MaskGenerator | None, bool, str]:
    """Resolve the run's masking condition and report it.

    Returns ``(generator, apply_to_training, run_id_tag)``. ``generator`` is None
    for the normal (0%) condition, in which case nothing about the run changes.

    ``trains`` says whether the caller fits a model. A masked-training ablation
    run is a different model from the primary condition run, so it earns its own
    run-identity suffix (ADR-0011); for a model that never trains, the flag has
    nothing to act on and is reported as ignored rather than tagged.
    """
    masking_cfg = config.get("masking") or {}
    generator = resolve_mask_generator(masking_cfg, num_nodes, seed)
    apply_to_training = bool(masking_cfg.get("apply_to_training", False))
    if generator is None:
        return None, apply_to_training, ""

    if apply_to_training and not trains:
        print(
            "Note: masking.apply_to_training has no effect on a model that is not "
            "fitted; this run is the primary condition.",
            flush=True,
        )
        apply_to_training = False

    withheld = int((~generator.node_mask).sum())
    print(
        f"Experimental mask {generator.mask_id}: {withheld} of {generator.num_nodes} "
        f"sensors withheld from model inputs (checksum {generator.checksum[:12]}).",
        flush=True,
    )
    return generator, apply_to_training, build_condition_tag(generator, apply_to_training)


def _mask_eval_config(
    generator: MaskGenerator | None, apply_to_training: bool
) -> dict[str, Any]:
    """The masking block the evaluator needs to report target groups."""
    if generator is None:
        return {}
    return {
        "masking": {
            "missing_ratio": generator.missing_ratio,
            "seed": generator.seed,
            "apply_to_training": apply_to_training,
        }
    }


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
    x_mask_key = f"{split}_X_mask" if f"{split}_X_mask" in arrays else f"x_{split}_mask"
    x_data = arrays[x_key]  # [S, L, N, 1]

    # Experimental node masking (Phase 11). Persistence reads the last observed
    # input step, so a withheld sensor's input is 0.0 after normalization and the
    # baseline predicts its training mean. That is the honest consequence of the
    # condition and is not special-cased.
    mask_generator, mask_training, condition_tag = _resolve_condition(
        config, x_data.shape[2], config.get("seed")
    )
    if mask_generator is not None:
        x_mask = arrays.get(x_mask_key)
        if x_mask is None:
            raise KeyError(
                f"Input mask '{x_mask_key}' not found in artifact; a masked condition "
                "needs the native observation mask."
            )
        x_data, _ = mask_generator.apply_mask(x_data, x_mask, copy=False)

    persistence = HistoricalPersistence(forecast_horizon=forecast_horizon)
    y_pred = persistence.predict(x_data, forecast_horizon=forecast_horizon)

    os.makedirs(output_dir, exist_ok=True)
    run_id = f"baseline-persistence-{split}{condition_tag}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
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
        "seed": config.get("seed"),
        **_mask_eval_config(mask_generator, mask_training),
    }

    manifest, manifest_path = run_evaluation(eval_config, output_dir=output_dir, save_plots=save_plots)
    return {"run_id": run_id, "manifest_path": manifest_path, "manifest": manifest}


def train_and_eval_st_gcn(
    config: dict[str, Any],
    output_dir: str = "artifacts/results",
    checkpoint_dir: str = "artifacts/checkpoints",
    save_plots: bool = True,
    resume_path: str | None = None,
    predict_from: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate the deterministic ST-GCN baseline.

    ``resume_path`` may be a training state saved by an interrupted run, or
    "auto" to continue the newest unfinished run for this split and seed.
    ``predict_from`` evaluates an existing ``_best.pt`` checkpoint instead of
    training, which is how a missingness condition is run (ADR-0004).
    """
    print("Executing Deterministic ST-GCN Baseline Pipeline...")

    if predict_from is not None and resume_path is not None:
        raise ValueError(
            "--predict-from evaluates a finished checkpoint and --resume continues an "
            "unfinished run; pass only one."
        )

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

    # 1b. Experimental node masking (Phase 11). Inputs only: targets stay intact
    # so the evaluator still scores every sensor (ADR-0004). Masking is in place
    # because a full split of inputs is hundreds of megabytes.
    mask_generator, mask_training, condition_tag = _resolve_condition(
        config, train_x.shape[2], seed, trains=True
    )
    if mask_generator is not None:
        test_x, test_x_mask = mask_generator.apply_mask(test_x, test_x_mask, copy=False)
        if mask_training:
            print("Masking training and validation inputs under the same mask.", flush=True)
            train_x, train_x_mask = mask_generator.apply_mask(train_x, train_x_mask, copy=False)
            val_x, val_x_mask = mask_generator.apply_mask(val_x, val_x_mask, copy=False)
        else:
            print(
                "Training and validation inputs are unmasked; the condition applies "
                "at inference only (ADR-0004).",
                flush=True,
            )

    # 2. Graph & Chebyshev Polynomials
    adj_mx_path = config.get("adj_mx_path", "data/raw/adj_mx_bay.pkl")
    cheb_k = int(config.get("cheb_k", 3))

    if not os.path.exists(adj_mx_path):
        raise FileNotFoundError(
            f"Required graph adjacency file not found at: {adj_mx_path}. "
            "Please acquire the canonical dataset using 'st-dssm-provenance --download'."
        )

    # Verify checksum against pinned canonical release
    allow_unverified_graph = bool(config.get("allow_unverified_graph", False))
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
        sensor_ids_graph_raw, _, adj_mx = pickle.load(f, encoding="latin1")

    graph_sensor_ids = [str(sid) for sid in sensor_ids_graph_raw]
    expected_sensor_ids = _metadata.get("sensor_ids", [])
    if not expected_sensor_ids:
        raise ValueError(
            f"Sensor metadata 'sensor_ids' is missing or empty in dataset artifact metadata at {artifact_dir}."
        )
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
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    if predict_from is None:
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

    # The run identity carries the masking condition, so runs under different
    # conditions neither overwrite each other's outputs nor resume each other.
    run_prefix = f"baseline-st_gcn-{split}-seed{seed}{condition_tag}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{run_prefix}-{timestamp}"
    start_epoch = 1
    if resume_path == "auto":
        # A run's _last.pt state is deleted once the run completes.
        candidates = sorted(glob.glob(os.path.join(checkpoint_dir, f"{run_prefix}-*_last.pt")))
        resume_path = candidates[-1] if candidates else None
        if resume_path is None:
            print("No unfinished run to resume; starting a new run.", flush=True)
    if resume_path:
        print(f"Resuming from training state: {resume_path}", flush=True)
        state = load_training_state(resume_path, model, optimizer, early_stopping, map_location=device)
        run_id = state["run_id"]
        start_epoch = state["epoch"] + 1
        print(f"Resumed at epoch {state['epoch']} with best val MAE: {early_stopping.best_score}", flush=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")
    last_state_path: str | None = os.path.join(checkpoint_dir, f"{run_id}_last.pt")

    if predict_from is not None:
        # Evaluate an existing checkpoint instead of training. Nothing is fitted
        # here, so no checkpoint is written and no training state is kept.
        from st_dssm.training import load_checkpoint

        if not os.path.exists(predict_from):
            raise FileNotFoundError(f"Checkpoint not found at: {predict_from}")
        checkpoint = load_checkpoint(predict_from, model, map_location=device)
        best_ckpt_path = predict_from
        last_state_path = None
        run_id = f"{run_prefix}-predict-{timestamp}"
        early_stopping.best_epoch = int(checkpoint.get("epoch", 0))
        early_stopping.best_score = checkpoint.get("val_loss")
        print(
            f"Evaluating checkpoint {predict_from} (epoch "
            f"{early_stopping.best_epoch}) without training.",
            flush=True,
        )
    else:
        print(f"Beginning training (Max epochs={max_epochs}, Patience={patience}, Min delta={min_delta})...", flush=True)

    last_epoch = 0 if predict_from is not None else start_epoch - 1
    if predict_from is not None:
        epochs_to_run: range = range(0)
    else:
        epochs_to_run = range(start_epoch, max_epochs + 1) if not early_stopping.early_stop else range(0)
    for epoch in epochs_to_run:
        last_epoch = epoch
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

        save_training_state(last_state_path, model, optimizer, epoch, early_stopping, run_id, config)

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best weights from epoch {early_stopping.best_epoch}.", flush=True)
            break

    # Restore best checkpoint
    if predict_from is None:
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
        "seed": seed,
        **_mask_eval_config(mask_generator, mask_training),
    }

    # Record training provenance so each result traces back to its run settings.
    # This is passed into the evaluator rather than patched in afterwards, so the
    # manifest is complete the first and only time it is written.
    eval_config["training_config"] = config
    provenance = {
        "training_seed": seed,
        "checkpoint_path": best_ckpt_path,
        "selection_metric": "val_mae",
        "best_epoch": int(early_stopping.best_epoch or 0),
        "total_epochs": last_epoch,
        "trainable_parameters": int(capacity_report["trainable_parameters"]),
    }

    manifest, manifest_path = run_evaluation(
        eval_config,
        output_dir=output_dir,
        save_plots=save_plots,
        provenance=provenance,
    )

    # The run is complete; drop its resume state so "--resume auto" won't pick it up.
    if last_state_path is not None and os.path.exists(last_state_path):
        os.remove(last_state_path)

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
        default=None,
        help="Baseline model architecture to run. [default: persistence]",
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
        default=None,
        help="Path to Phase 3 processed artifact directory. [default: data/processed]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save predictions, run manifests, and plots. [default: artifacts/results]",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save trained model checkpoints. [default: artifacts/checkpoints]",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split to evaluate ('test', 'val', 'train'). [default: test]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. [default: 2026]",
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
        default=None,
        help="Maximum training epochs for learned models. [default: 100]",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Training state to resume from, or 'auto' to continue the newest unfinished run for this split and seed.",
    )
    parser.add_argument(
        "--predict-from",
        type=str,
        default=None,
        help="Evaluate this '_best.pt' checkpoint instead of training (st_gcn only). "
        "The primary way to run a missingness condition (ADR-0004).",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Withhold this fraction of sensors from the model's input history "
        "(e.g. 0.2 for the primary condition). 0.0 is the normal condition.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help="Seed of the experimental mask (canonical: 2026, 2027, 2028). "
        "[default: the run's seed]",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default=None,
        help="Published mask artifact (experiments/masks/*.json) to use instead of "
        "regenerating the mask from ratio and seed. Its checksum is verified.",
    )
    parser.add_argument(
        "--mask-training",
        action="store_true",
        default=None,
        help="Also withhold the same sensors during training and validation. By "
        "default a condition applies at inference only (ADR-0004).",
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
            run_synthetic_baseline_smoke(output_dir=args.output_dir or "artifacts/results")
            return 0

        config: dict[str, Any] = {}
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        # Merge CLI arguments into config: an explicitly passed flag always wins,
        # then the YAML value, then the hardcoded default. (Previously the YAML
        # value always won, so e.g. --seed was ignored whenever the config set one.)
        defaults: dict[str, Any] = {
            "model": ("model", "persistence"),
            "artifact_dir": ("artifact_dir", "data/processed"),
            "output_dir": ("output_dir", "artifacts/results"),
            "checkpoint_dir": ("checkpoint_dir", "artifacts/checkpoints"),
            "split": ("split", "test"),
            "seed": ("seed", 2026),
            "epochs": ("max_epochs", 100),
        }
        for arg_name, (config_key, default) in defaults.items():
            cli_value = getattr(args, arg_name)
            if cli_value is not None:
                config[config_key] = cli_value
            else:
                config.setdefault(config_key, default)
        if args.adj_mx_path:
            config["adj_mx_path"] = args.adj_mx_path

        # Masking condition (Phase 11). CLI flags override the config block.
        masking = dict(config.get("masking") or {})
        if args.mask_ratio is not None:
            masking["missing_ratio"] = args.mask_ratio
        if args.mask_seed is not None:
            masking["seed"] = args.mask_seed
        if args.mask_path is not None:
            masking["mask_path"] = args.mask_path
        if args.mask_training is not None:
            masking["apply_to_training"] = args.mask_training
        if masking:
            config["masking"] = masking

        model_type = config["model"]
        out_dir = config["output_dir"]
        ckpt_dir = config["checkpoint_dir"]

        if model_type == "persistence":
            if args.predict_from:
                print(
                    "--predict-from does not apply to the persistence baseline, "
                    "which fits no parameters.",
                    file=sys.stderr,
                )
                return 1
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
                resume_path=args.resume,
                predict_from=args.predict_from,
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
