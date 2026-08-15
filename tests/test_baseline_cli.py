"""Integration tests for baseline CLI runner (st-dssm-baseline)."""

from __future__ import annotations

import os
import pickle

import numpy as np
import yaml

from st_dssm.cli.baseline import main
from st_dssm.io import save_processed_artifact
from st_dssm.result_schema import load_run_manifest


class TestBaselineCLI:
    def test_cli_synthetic_smoke(self, tmp_path):
        out_dir = str(tmp_path / "smoke_results")
        exit_code = main(["--synthetic-smoke", "--output-dir", out_dir])
        assert exit_code == 0

    def test_cli_persistence_with_artifact(self, tmp_path):
        # 1. Create a dummy Phase 3 artifact
        artifact_dir = str(tmp_path / "artifact")
        s, l, h, n = 8, 12, 12, 4
        arrays = {
            "test_X": np.ones((s, l, n, 1), dtype=np.float32) * 0.5,
            "test_X_mask": np.ones((s, l, n, 1), dtype=np.float32),
            "test_Y": np.ones((s, h, n, 1), dtype=np.float32) * 0.5,
            "test_Y_mask": np.ones((s, h, n, 1), dtype=np.float32),
            "scaler_means": np.array([50.0, 55.0, 60.0, 65.0], dtype=np.float32),
            "scaler_stds": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        }
        metadata = {
            "schema_version": "1.0",
            "sensor_ids": ["s1", "s2", "s3", "s4"],
            "splits": {"train": 0, "val": 0, "test": s},
        }
        save_processed_artifact(artifact_dir, arrays, metadata)

        out_dir = str(tmp_path / "persistence_results")
        exit_code = main([
            "--model", "persistence",
            "--artifact-dir", artifact_dir,
            "--output-dir", out_dir,
            "--no-plots",
        ])
        assert exit_code == 0

        # Verify predictions file exists
        pred_files = [f for f in os.listdir(out_dir) if f.endswith("_predictions.npz")]
        assert len(pred_files) == 1
        with np.load(os.path.join(out_dir, pred_files[0])) as data:
            assert "predictions" in data
            assert data["predictions"].shape == (s, h, n, 1)

        # Verify manifest file exists and matches metrics
        manifest_files = [f for f in os.listdir(out_dir) if f.endswith("_manifest.json")]
        assert len(manifest_files) == 1
        manifest = load_run_manifest(os.path.join(out_dir, manifest_files[0]))
        assert manifest.model_name == "historical_persistence"
        assert "MAE" in manifest.overall_metrics
        assert manifest.run_status == "complete"

    def test_cli_st_gcn_with_artifact(self, tmp_path):
        # 1. Create a dummy Phase 3 artifact with train, val, and test splits
        artifact_dir = str(tmp_path / "st_gcn_artifact")
        s_train, s_val, s_test = 8, 4, 4
        l, h, n = 12, 12, 4

        arrays = {
            "train_X": np.random.randn(s_train, l, n, 1).astype(np.float32),
            "train_X_mask": np.ones((s_train, l, n, 1), dtype=np.float32),
            "train_Y": np.random.randn(s_train, h, n, 1).astype(np.float32),
            "train_Y_mask": np.ones((s_train, h, n, 1), dtype=np.float32),
            "val_X": np.random.randn(s_val, l, n, 1).astype(np.float32),
            "val_X_mask": np.ones((s_val, l, n, 1), dtype=np.float32),
            "val_Y": np.random.randn(s_val, h, n, 1).astype(np.float32),
            "val_Y_mask": np.ones((s_val, h, n, 1), dtype=np.float32),
            "test_X": np.random.randn(s_test, l, n, 1).astype(np.float32),
            "test_X_mask": np.ones((s_test, l, n, 1), dtype=np.float32),
            "test_Y": np.random.randn(s_test, h, n, 1).astype(np.float32),
            "test_Y_mask": np.ones((s_test, h, n, 1), dtype=np.float32),
            "scaler_means": np.array([50.0, 55.0, 60.0, 65.0], dtype=np.float32),
            "scaler_stds": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        }
        metadata = {
            "schema_version": "1.0",
            "sensor_ids": ["s1", "s2", "s3", "s4"],
            "splits": {"train": s_train, "val": s_val, "test": s_test},
        }
        save_processed_artifact(artifact_dir, arrays, metadata)

        # 2. Create dummy adjacency matrix
        adj_path = str(tmp_path / "adj.pkl")
        with open(adj_path, "wb") as f:
            pickle.dump((["s1", "s2", "s3", "s4"], {}, np.eye(n, dtype=np.float32)), f)

        # 3. Create config
        config_path = str(tmp_path / "st_gcn_config.yaml")
        out_dir = str(tmp_path / "st_gcn_results")
        ckpt_dir = str(tmp_path / "st_gcn_ckpts")

        config_data = {
            "schema_version": "1.0",
            "model": "st_gcn",
            "artifact_dir": artifact_dir,
            "adj_mx_path": adj_path,
            "split": "test",
            "seed": 2026,
            "hidden_channels": 16,
            "cheb_k": 3,
            "dropout": 0.0,
            "batch_size": 4,
            "lr": 0.01,
            "max_epochs": 2,
            "patience": 2,
            "output_dir": out_dir,
            "checkpoint_dir": ckpt_dir,
            "allow_unverified_graph": True,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # 4. Run baseline CLI
        exit_code = main([
            "--config", config_path,
            "--no-plots",
        ])
        assert exit_code == 0

        # Verify checkpoint
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        assert len(ckpts) == 1

        # Verify predictions
        preds = [f for f in os.listdir(out_dir) if f.endswith("_predictions.npz")]
        assert len(preds) == 1

        # Verify manifest
        manifests = [f for f in os.listdir(out_dir) if f.endswith("_manifest.json")]
        assert len(manifests) == 1
        loaded = load_run_manifest(os.path.join(out_dir, manifests[0]))
        assert loaded.model_name == "deterministic_st_gcn"
        assert loaded.run_status == "complete"
        assert "MAE" in loaded.overall_metrics

    def test_cli_st_gcn_missing_graph_fails_explicitly(self, tmp_path):
        config_path = str(tmp_path / "missing_graph_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "model": "st_gcn",
                "adj_mx_path": str(tmp_path / "non_existent_graph.pkl"),
            }, f)
        exit_code = main(["--config", config_path])
        assert exit_code == 1
