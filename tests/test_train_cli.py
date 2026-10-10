"""Tests for the st-dssm-train CLI."""

from __future__ import annotations

import os
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from st_dssm.cli.train import main, train_st_dssm, train_step
from st_dssm.dssm import GaussianDSSM
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
)
from st_dssm.io import save_processed_artifact
from st_dssm.result_schema import load_run_manifest


def test_train_cli_synthetic_smoke(tmp_path: os.PathLike[str]) -> None:
    """Test that the synthetic smoke test runs successfully."""
    # Use tmp_path to capture any output if it's written locally
    output_dir = os.path.join(tmp_path, "results")
    
    # Run the synthetic smoke test
    exit_code = main(["--synthetic-smoke", "--output-dir", output_dir])
    
    # Assert successful exit
    assert exit_code == 0
    assert os.path.exists(os.path.join(output_dir, "synthetic_ckpt.pt"))

def test_train_cli_resume(tmp_path: os.PathLike[str]) -> None:
    """Test that we can resume from a checkpoint."""
    output_dir = os.path.join(tmp_path, "results")
    ckpt_path = os.path.join(output_dir, "synthetic_ckpt.pt")
    
    # Run once to create the checkpoint
    exit_code = main(["--synthetic-smoke", "--output-dir", output_dir])
    assert exit_code == 0
    assert os.path.exists(ckpt_path)
    
    # Create a minimal config to load the synthetic data and resume
    # We can't really run real train_st_dssm without real data, 
    # but we can just test if the parser handles --resume.
    # Actually, train_st_dssm with dummy config fails on dataset load,
    # so we will just verify the parser parses it correctly, or we can mock load_and_validate_artifact.
    # A simple parser check for now:
    from unittest.mock import patch
    
    with patch("st_dssm.cli.train.train_st_dssm") as mock_train:
        main(["--resume", ckpt_path])
        mock_train.assert_called_once()
        assert mock_train.call_args[1]["resume_path"] == ckpt_path


def test_train_cli_flag_overrides_config_value_matching_hardcoded_default(
    tmp_path: os.PathLike[str],
) -> None:
    """An explicit CLI flag must override a config value even when the CLI
    value happens to equal the argparse default (regression test: previously
    the merge logic compared the parsed CLI value against the hardcoded
    default to decide whether the flag was "explicitly passed", so a config
    value could not be overridden back to that default via the CLI)."""
    config_path = os.path.join(tmp_path, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"seed": 99, "split": "val"}, f)

    with patch("st_dssm.cli.train.train_st_dssm") as mock_train:
        # --seed 2026 and (implicitly) --split test both match the CLI's
        # hardcoded defaults, but must still win over the config's seed=99/split=val.
        main(["--config", config_path, "--seed", "2026", "--split", "test"])
        mock_train.assert_called_once()
        called_config = mock_train.call_args[1]["config"]
        assert called_config["seed"] == 2026
        assert called_config["split"] == "test"


def _grads_after_step(micro_batch_size: int | None) -> tuple[list[torch.Tensor], tuple[float, float, float]]:
    """One train_step on a fixed synthetic batch with all sampling noise disabled."""
    torch.manual_seed(0)
    n_nodes, length = 5, 12
    adj = np.eye(n_nodes, dtype=np.float32)
    for i in range(n_nodes - 1):
        adj[i, i + 1] = adj[i + 1, i] = 0.5
    scaled_lap, _ = calculate_scaled_laplacian(calculate_normalized_laplacian(adj))
    cheb = torch.tensor(compute_chebyshev_polynomials(scaled_lap, k=3), dtype=torch.float32)

    model = GaussianDSSM(
        num_nodes=n_nodes, in_channels=2, latent_dim=4, context_dim=8, hidden_dim=8,
        horizon=length, input_length=length, cheb_k=3, dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    gen = torch.Generator().manual_seed(1)
    x = torch.randn(8, length, n_nodes, 1, generator=gen)
    x_mask = torch.rand(8, length, n_nodes, 1, generator=gen) > 0.2
    y = torch.randn(8, length, n_nodes, 1, generator=gen)
    # Uneven observed-target counts per sample exercise the NLL weighting.
    y_mask = torch.rand(8, length, n_nodes, 1, generator=gen) > torch.linspace(0.1, 0.7, 8).view(8, 1, 1, 1)

    with patch("torch.randn_like", side_effect=torch.zeros_like):
        losses = train_step(
            model, optimizer, (x, x_mask, y, y_mask), cheb, torch.device("cpu"),
            teacher_force_ratio=1.0, beta=0.5, micro_batch_size=micro_batch_size,
        )
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return grads, losses


@pytest.mark.parametrize("micro_batch_size", [4, 3])
def test_train_step_micro_batches_match_full_batch(micro_batch_size: int) -> None:
    """Gradient accumulation must reproduce the full-batch gradient and losses,
    including when the last micro-batch is smaller than the others."""
    full_grads, full_losses = _grads_after_step(None)
    micro_grads, micro_losses = _grads_after_step(micro_batch_size)

    assert len(full_grads) == len(micro_grads) > 0
    for g_full, g_micro in zip(full_grads, micro_grads):
        torch.testing.assert_close(g_micro, g_full, rtol=1e-4, atol=1e-6)
    assert micro_losses == pytest.approx(full_losses, rel=1e-5)


def _tiny_training_config(tmp_path, micro_batch_size: int) -> dict:
    """Write a 4-sensor artifact and graph to tmp_path and return a CPU training config."""
    rng = np.random.default_rng(0)
    n, length = 4, 12
    sensor_ids = [f"s{i}" for i in range(n)]
    arrays = {}
    for split, count in [("train", 16), ("val", 8), ("test", 8)]:
        arrays[f"{split}_X"] = rng.normal(size=(count, length, n, 1)).astype(np.float32)
        arrays[f"{split}_X_mask"] = np.ones((count, length, n, 1), dtype=bool)
        arrays[f"{split}_Y"] = rng.normal(size=(count, length, n, 1)).astype(np.float32)
        arrays[f"{split}_Y_mask"] = np.ones((count, length, n, 1), dtype=bool)
    arrays["scaler_means"] = np.full(n, 60.0, dtype=np.float32)
    arrays["scaler_stds"] = np.full(n, 5.0, dtype=np.float32)
    artifact_dir = str(tmp_path / "artifact")
    save_processed_artifact(artifact_dir, arrays, {"schema_version": "1.0", "sensor_ids": sensor_ids})

    adj = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 0.5
    adj_path = str(tmp_path / "adj.pkl")
    with open(adj_path, "wb") as f:
        pickle.dump((sensor_ids, {}, adj), f)

    return {
        "seed": 7,
        "device": "cpu",
        "artifact_dir": artifact_dir,
        "adj_mx_path": adj_path,
        "allow_unverified_graph": True,
        "split": "test",
        "model": {"latent_dim": 4, "context_dim": 8, "hidden_dim": 8, "dropout": 0.0},
        "training": {"max_epochs": 2, "batch_size": 8, "micro_batch_size": micro_batch_size},
    }


def test_train_st_dssm_end_to_end_records_uncertainty_and_provenance(tmp_path) -> None:
    """A full train_st_dssm run must produce a manifest with the probabilistic
    metrics and the training settings (regression test: predictions were saved
    under a key the evaluator did not read, so NLL/CRPS/PICP were silently lost)."""
    result = train_st_dssm(
        config=_tiny_training_config(tmp_path, micro_batch_size=4),
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_plots=False,
    )

    manifest = load_run_manifest(result["manifest_path"])
    for metric in ("MAE", "NLL", "CRPS", "PICP", "MPIW"):
        assert metric in manifest.overall_metrics
    assert manifest.training_seed == 7
    assert manifest.selection_metric == "val_nll"
    assert manifest.total_epochs == 2
    assert 1 <= manifest.best_epoch <= 2
    assert os.path.exists(manifest.checkpoint_path)
    assert manifest.resolved_config["training_config"]["training"]["micro_batch_size"] == 4


def test_train_st_dssm_rejects_micro_batch_larger_than_batch(tmp_path) -> None:
    with pytest.raises(ValueError, match="micro_batch_size"):
        train_st_dssm(
            config=_tiny_training_config(tmp_path, micro_batch_size=16),
            output_dir=str(tmp_path / "results"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            save_plots=False,
        )


def test_interrupted_run_resumes_to_identical_result(tmp_path) -> None:
    """A run killed mid-training and continued with resume_path="auto" must end
    with exactly the weights and metrics of an uninterrupted run."""
    import st_dssm.cli.train as train_module

    def run(out: str, resume_path: str | None = None) -> dict:
        config = _tiny_training_config(tmp_path / out, micro_batch_size=4)
        config["training"]["max_epochs"] = 3
        return train_st_dssm(
            config=config,
            output_dir=str(tmp_path / out / "results"),
            checkpoint_dir=str(tmp_path / out / "ckpt"),
            save_plots=False,
            resume_path=resume_path,
        )

    reference = run("straight")

    real_save = train_module.save_training_state

    def save_then_crash(path, model, optimizer, epoch, *args, **kwargs):
        real_save(path, model, optimizer, epoch, *args, **kwargs)
        if epoch == 2:
            raise KeyboardInterrupt  # simulate the laptop being unplugged

    with (
        patch.object(train_module, "save_training_state", side_effect=save_then_crash),
        pytest.raises(KeyboardInterrupt),
    ):
        run("resumed")
    last_states = list((tmp_path / "resumed" / "ckpt").glob("*_last.pt"))
    assert len(last_states) == 1

    resumed = run("resumed", resume_path="auto")

    assert resumed["run_id"] == last_states[0].name.removesuffix("_last.pt")
    assert not last_states[0].exists()  # cleaned up after completion
    ref_weights = torch.load(reference["checkpoint_path"], weights_only=False)["model_state_dict"]
    res_weights = torch.load(resumed["checkpoint_path"], weights_only=False)["model_state_dict"]
    for name, tensor in ref_weights.items():
        torch.testing.assert_close(res_weights[name], tensor, rtol=0, atol=0)
    assert resumed["manifest"].overall_metrics == reference["manifest"].overall_metrics
    assert resumed["manifest"].total_epochs == 3


def test_resume_auto_without_unfinished_run_starts_fresh(tmp_path) -> None:
    config = _tiny_training_config(tmp_path, micro_batch_size=4)
    config["training"]["max_epochs"] = 1
    result = train_st_dssm(
        config=config,
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_plots=False,
        resume_path="auto",
    )
    assert result["manifest"].total_epochs == 1
    assert not list((tmp_path / "ckpt").glob("*_last.pt"))
