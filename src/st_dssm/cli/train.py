"""Command line interface for training the ST-DSSM model (st-dssm-train).

Implements Phase 9 Training operations, including:
- YAML configuration loading
- Checkpoint selection on validation NLL
- Beta annealing schedule
- Teacher forcing decay schedule
- Canonical seeding
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

from st_dssm.cli.evaluate import run_evaluation
from st_dssm.data import EXPECTED_FILES, ChecksumMismatchError, compute_md5
from st_dssm.dssm import GaussianDSSM, _masked_gaussian_nll
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
    symmetrize_adjacency,
)
from st_dssm.io import load_and_validate_artifact
from st_dssm.result_schema import save_run_manifest
from st_dssm.training import (
    EarlyStopping,
    save_checkpoint,
    set_seed,
)
from st_dssm.validator import validate_graph


def train_step(
    model: GaussianDSSM,
    optimizer: optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cheb_poly: torch.Tensor,
    device: torch.device,
    teacher_force_ratio: float,
    beta: float,
    micro_batch_size: int | None = None,
) -> tuple[float, float, float]:
    """Run one optimizer step on a batch, optionally split into micro-batches.

    Splitting only bounds peak GPU memory: each micro-batch loss is weighted so
    the accumulated gradient equals the full-batch gradient. The reconstruction
    NLL is a mean over observed targets, so it is weighted by each micro-batch's
    share of observed targets; the KL is a mean over all elements, so it is
    weighted by each micro-batch's share of samples.

    Returns:
        (elbo, nll, kl) for the full batch.
    """
    bx, bxm, by, bym = batch
    batch_len = bx.shape[0]
    step = micro_batch_size or batch_len
    total_obs = float(bym.sum())

    optimizer.zero_grad()
    batch_nll = 0.0
    batch_kl = 0.0
    for start in range(0, batch_len, step):
        sl = slice(start, start + step)
        x_in = torch.cat([bx[sl], bxm[sl].float()], dim=-1).to(device)
        y = by[sl].to(device)
        y_mask = bym[sl].to(device)

        _mu, _sigma, kl, nll = model.forward_train(
            x=x_in,
            cheb_polynomials=cheb_poly,
            y_target=y,
            obs_mask_hist=bxm[sl].to(device),
            obs_mask_fore=y_mask,
            teacher_force_ratio=teacher_force_ratio,
            beta=beta,
        )

        nll_weight = float(bym[sl].sum()) / total_obs if total_obs > 0 else 0.0
        kl_weight = bx[sl].shape[0] / batch_len
        loss = nll_weight * nll + beta * kl_weight * kl
        loss.backward()

        batch_nll += nll_weight * nll.item()
        batch_kl += kl_weight * kl.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return batch_nll + beta * batch_kl, batch_nll, batch_kl


def train_st_dssm(
    config: dict[str, Any],
    output_dir: str = "artifacts/results",
    checkpoint_dir: str = "artifacts/checkpoints",
    save_plots: bool = True,
    resume_path: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate the ST-DSSM."""
    print("Executing ST-DSSM Training Pipeline...")

    # Configuration extraction
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
    model_cfg = config.get("model", {})
    cheb_k = int(model_cfg.get("cheb_k", 3))

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

    # 3. Model instantiation
    num_nodes = train_x.shape[2]
    
    model = GaussianDSSM(
        num_nodes=num_nodes,
        in_channels=int(model_cfg.get("in_channels", 2)),
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        context_dim=int(model_cfg.get("context_dim", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        horizon=int(model_cfg.get("horizon", 12)),
        input_length=int(model_cfg.get("input_length", 12)),
        cheb_k=cheb_k,
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    cap_report = model.get_capacity_report()
    print(f"Model capacity: {cap_report['total_trainable_parameters']} trainable parameters.", flush=True)

    # 4. DataLoaders
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    micro_batch_size = train_cfg.get("micro_batch_size")
    if micro_batch_size is not None:
        micro_batch_size = int(micro_batch_size)
        if not 0 < micro_batch_size <= batch_size:
            raise ValueError(
                f"training.micro_batch_size must be in [1, batch_size={batch_size}], got {micro_batch_size}"
            )
        print(f"Gradient accumulation: batch size {batch_size} in micro-batches of {micro_batch_size}.", flush=True)
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
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = int(train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("patience", 15))
    min_delta = float(train_cfg.get("min_delta", 1e-4))
    
    schedules_cfg = config.get("schedules", {})
    beta_anneal_epochs = int(schedules_cfg.get("beta_anneal_epochs", 20))
    tf_decay_epochs = int(schedules_cfg.get("teacher_force_decay_epochs", max_epochs // 2))

    # EarlyStopping tracks validation NLL (minimization)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")

    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        from st_dssm.training import load_checkpoint
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        early_stopping.best_score = ckpt.get("val_loss", None)
        early_stopping.best_epoch = ckpt.get("epoch", 0)
        early_stopping.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Resumed at epoch {start_epoch-1} with best val NLL: {early_stopping.best_score}")

    run_id = f"st_dssm-{split}-seed{seed}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")

    print(f"Beginning training (Max epochs={max_epochs}, Beta Anneal={beta_anneal_epochs}, TF Decay={tf_decay_epochs})...", flush=True)

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        last_epoch = epoch
        # Calculate schedules
        beta = min(epoch / max(1, beta_anneal_epochs), 1.0)
        tf_ratio = max(1.0 - (epoch / max(1, tf_decay_epochs)), 0.0)

        model.train()
        total_train_elbo = 0.0
        total_train_nll = 0.0
        total_train_kl = 0.0
        batch_count = 0

        for batch in train_loader:
            elbo, nll, kl = train_step(
                model,
                optimizer,
                batch,
                cheb_poly,
                device,
                teacher_force_ratio=tf_ratio,
                beta=beta,
                micro_batch_size=micro_batch_size,
            )
            total_train_elbo += elbo
            total_train_nll += nll
            total_train_kl += kl
            batch_count += 1

        avg_train_elbo = total_train_elbo / batch_count
        avg_train_nll = total_train_nll / batch_count
        avg_train_kl = total_train_kl / batch_count

        # Validation (compute Gaussian NLL using predictive distribution)
        model.eval()
        total_val_nll = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for bx, bxm, by, bym in val_loader:
                bx_in = torch.cat([bx, bxm.float()], dim=-1).to(device)
                by = by.to(device)
                bym = bym.to(device)

                # Use forward_predict per validation contract
                mu_pred, sigma_pred = model.forward_predict(bx_in, cheb_poly)
                
                nll = _masked_gaussian_nll(by, mu_pred, sigma_pred, bym)
                total_val_nll += nll.item()
                val_batch_count += 1

        avg_val_nll = total_val_nll / max(1, val_batch_count)

        should_stop = early_stopping.step(avg_val_nll, model, epoch)
        print(f"Epoch {epoch:03d} | beta: {beta:.2f} | TF: {tf_ratio:.2f} | Train ELBO: {avg_train_elbo:.4f} (NLL {avg_train_nll:.4f}, KL {avg_train_kl:.4f}) | Val NLL: {avg_val_nll:.4f} | Best Val NLL: {early_stopping.best_score:.4f}", flush=True)

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
    test_sigmas = []

    with torch.no_grad():
        for bx, bxm in test_loader:
            bx_in = torch.cat([bx, bxm.float()], dim=-1).to(device)
            # Draw predictions (mean is returned as mu_pred)
            mu_pred, sigma_pred = model.forward_predict(bx_in, cheb_poly)
            test_preds.append(mu_pred.cpu().numpy())
            test_sigmas.append(sigma_pred.cpu().numpy())

    y_pred = np.concatenate(test_preds, axis=0)
    y_sigma = np.concatenate(test_sigmas, axis=0)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, f"{run_id}_predictions.npz")
    # Key names must match what run_evaluation reads, or the uncertainty
    # metrics (NLL, CRPS, PICP, MPIW) are silently skipped.
    np.savez_compressed(predictions_path, predictions=y_pred, sigma=y_sigma)
    print(f"Predictions saved to: {predictions_path}")

    # 7. Run evaluation and create manifest
    eval_config = {
        "schema_version": "1.0",
        "run_id": run_id,
        "model_name": "st_dssm",
        "artifact_dir": artifact_dir,
        "predictions_path": predictions_path,
        "split": split,
        "cadence_minutes": int(config.get("cadence_minutes", 5)),
        "output_dir": output_dir,
        "training_config": config,
    }

    manifest, manifest_path = run_evaluation(eval_config, output_dir=output_dir, save_plots=save_plots)

    # Record training provenance so each result traces back to its run settings.
    manifest.training_seed = seed
    manifest.checkpoint_path = best_ckpt_path
    manifest.selection_metric = "val_nll"
    manifest.best_epoch = int(early_stopping.best_epoch or 0)
    manifest.total_epochs = last_epoch
    manifest.trainable_parameters = int(cap_report["total_trainable_parameters"])
    save_run_manifest(manifest, manifest_path, overwrite=True, atomic=True)

    return {
        "run_id": run_id,
        "manifest_path": manifest_path,
        "checkpoint_path": best_ckpt_path,
        "manifest": manifest,
        "capacity_report": cap_report,
    }


def run_synthetic_train_smoke(output_dir: str = "artifacts/results") -> None:
    """Run an end-to-end synthetic training smoke test for ST-DSSM."""
    print("Running Synthetic ST-DSSM Train Smoke Test...")
    set_seed(42)

    S, L, H, N = 16, 12, 12, 5
    x_synth = np.random.normal(0.0, 1.0, (S, L, N, 1)).astype(np.float32)
    x_mask = np.ones((S, L, N, 1), dtype=np.float32)
    y_synth = np.random.normal(0.0, 1.0, (S, H, N, 1)).astype(np.float32)
    y_mask = np.ones((S, H, N, 1), dtype=np.float32)

    adj_synth = np.eye(N, dtype=np.float32)
    for i in range(N - 1):
        adj_synth[i, i + 1] = 0.5
        adj_synth[i + 1, i] = 0.5

    norm_lap = calculate_normalized_laplacian(adj_synth)
    scaled_lap, _ = calculate_scaled_laplacian(norm_lap)
    cheb_poly_np = compute_chebyshev_polynomials(scaled_lap, k=3)
    cheb_poly = torch.tensor(cheb_poly_np, dtype=torch.float32)

    model = GaussianDSSM(
        num_nodes=N,
        in_channels=2,
        latent_dim=4,
        context_dim=8,
        hidden_dim=16,
        horizon=H,
        input_length=L,
        cheb_k=3,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    in_tensor = torch.tensor(
        np.concatenate([x_synth, x_mask], axis=-1), dtype=torch.float32
    )
    target_tensor = torch.tensor(y_synth, dtype=torch.float32)
    mask_tensor = torch.tensor(y_mask, dtype=torch.float32)

    model.train()
    optimizer.zero_grad()
    _mu, _sigma, kl, nll = model.forward_train(
        x=in_tensor,
        cheb_polynomials=cheb_poly,
        y_target=target_tensor,
        obs_mask_hist=mask_tensor[:, :L, :, :],
        obs_mask_fore=mask_tensor,
        teacher_force_ratio=1.0,
        beta=1.0
    )
    loss = nll + kl
    loss.backward()
    optimizer.step()

    # Save a temporary checkpoint for resume tests
    from st_dssm.training import save_checkpoint
    os.makedirs(output_dir, exist_ok=True)
    save_checkpoint(
        os.path.join(output_dir, "synthetic_ckpt.pt"),
        model,
        optimizer,
        epoch=1,
        val_loss=nll.item(),
    )

    assert not torch.isnan(loss), "Loss is NaN"
    print("  [PASSED] Synthetic ST-DSSM training step check.")
    
    model.eval()
    with torch.no_grad():
        _mu_pred, _sigma_pred = model.forward_predict(in_tensor, cheb_poly)
    
    assert _mu_pred.shape == (S, H, N, 1)
    print("  [PASSED] Synthetic ST-DSSM eval step check.")
    print("Synthetic train smoke test completed successfully.")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for st-dssm-train."""
    parser = argparse.ArgumentParser(description="ST-DSSM Phase 9 Training operations.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run synthetic smoke test.")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Path to Phase 3 processed artifact directory. [default: data/processed]")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save predictions, run manifests, and plots. [default: artifacts/results]")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save trained model checkpoints. [default: artifacts/checkpoints]")
    parser.add_argument("--split", type=str, default=None, help="Split to evaluate ('test', 'val', 'train'). [default: test]")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. [default: 2026]")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--adj-mx-path", type=str, default=None, help="Path to graph adjacency pickle file.")
    parser.add_argument("--device", type=str, default=None, help="Compute device. [default: cuda if available, else cpu]")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Generate evaluation plots.")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false", help="Disable evaluation plot generation.")

    args = parser.parse_args(argv)

    try:
        if args.synthetic_smoke:
            run_synthetic_train_smoke(output_dir=args.output_dir or "artifacts/results")
            return 0

        config: dict[str, Any] = {}
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        # Merge CLI arguments into config: an explicitly-passed CLI flag always
        # wins, falling back to the YAML config value, then a hardcoded default.
        # Using `None` argparse defaults (rather than comparing against the
        # hardcoded default value) lets a CLI flag override a config value even
        # when the CLI value happens to equal the hardcoded default.
        config["artifact_dir"] = args.artifact_dir if args.artifact_dir is not None else config.get("artifact_dir", "data/processed")
        config["output_dir"] = args.output_dir if args.output_dir is not None else config.get("output_dir", "artifacts/results")
        config["checkpoint_dir"] = args.checkpoint_dir if args.checkpoint_dir is not None else config.get("checkpoint_dir", "artifacts/checkpoints")
        config["split"] = args.split if args.split is not None else config.get("split", "test")
        config["seed"] = args.seed if args.seed is not None else config.get("seed", 2026)

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        config["device"] = args.device if args.device is not None else config.get("device", default_device)
        if args.adj_mx_path:
            config["adj_mx_path"] = args.adj_mx_path

        out_dir = config["output_dir"]
        ckpt_dir = config["checkpoint_dir"]

        train_st_dssm(
            config=config,
            output_dir=out_dir,
            checkpoint_dir=ckpt_dir,
            save_plots=args.save_plots,
            resume_path=args.resume,
        )

        return 0

    except RuntimeError as e:
        import traceback

        print(f"Trainer runner error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
