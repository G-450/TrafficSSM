"""Tests for evaluation CLI command (st-dssm-evaluate).

Tests synthetic smoke mode, canonical artifact evaluation mode, configuration routing,
output directory creation, manifest creation, and plot generation.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from st_dssm.cli.evaluate import main, run_synthetic_smoke
from st_dssm.io import save_processed_artifact
from st_dssm.result_schema import load_run_manifest


class TestEvaluateCLI:
    def test_run_synthetic_smoke_generates_manifest_and_plots(self, tmp_path):
        out_dir = str(tmp_path / "smoke_results")
        manifest, manifest_path = run_synthetic_smoke(out_dir, save_plots=True)

        assert os.path.exists(manifest_path)
        assert manifest.run_status == "complete"
        assert len(manifest.metrics) > 0
        assert manifest.manifest_checksum != ""

        # Check that plots exist
        plots_dir = os.path.join(out_dir, "plots")
        assert os.path.exists(plots_dir)
        plot_files = os.listdir(plots_dir)
        assert len(plot_files) >= 3

        # Verify loaded manifest
        loaded = load_run_manifest(manifest_path)
        assert loaded.run_id == manifest.run_id
        assert loaded.model_name == "synthetic_smoke_evaluator"

    def test_main_cli_synthetic_smoke(self, tmp_path):
        out_dir = str(tmp_path / "cli_results")
        exit_code = main(["--synthetic-smoke", "--output-dir", out_dir])
        assert exit_code == 0
        assert os.path.exists(out_dir)

    def test_main_cli_with_config_synthetic_mode(self, tmp_path):
        out_dir = str(tmp_path / "config_results")
        config_path = "configs/evaluation/synthetic.yaml"
        exit_code = main(["--config", config_path, "--output-dir", out_dir])
        assert exit_code == 0

    def test_run_evaluation_with_canonical_artifact(self, tmp_path):
        """Test evaluation routing with canonical Phase 3 artifact and prediction files."""
        # 1. Create a dummy Phase 3 artifact
        artifact_dir = str(tmp_path / "phase3_artifact")
        S, H, N = 10, 12, 4
        arrays = {
            "x_test": np.zeros((S, 12, N, 1), dtype=np.float32),
            "x_test_mask": np.ones((S, 12, N, 1), dtype=np.float32),
            "y_test": np.ones((S, H, N, 1), dtype=np.float32) * 0.5,
            "y_test_mask": np.ones((S, H, N, 1), dtype=np.float32),
        }
        metadata = {
            "schema_version": "1.0",
            "scaler": {
                "mean": [50.0, 55.0, 60.0, 65.0],
                "std": [5.0, 6.0, 7.0, 8.0],
            },
            "sensor_ids": ["s1", "s2", "s3", "s4"],
            "splits": {"train": 100, "val": 20, "test": 10},
        }
        save_processed_artifact(artifact_dir, arrays, metadata)

        # 2. Create a dummy prediction NPZ
        pred_path = str(tmp_path / "predictions.npz")
        np.savez_compressed(
            pred_path,
            mu=np.ones((S, H, N, 1), dtype=np.float32) * 0.4,
            sigma=np.ones((S, H, N, 1), dtype=np.float32) * 0.2,
        )

        # 3. Create evaluation YAML config
        config_path = str(tmp_path / "eval_config.yaml")
        out_dir = str(tmp_path / "eval_results")
        config_data = {
            "schema_version": "1.0",
            "artifact_dir": artifact_dir,
            "predictions_path": pred_path,
            "split": "test",
            "model_name": "test_st_gcn",
            "output_dir": out_dir,
            "cadence_minutes": 5,
            "intervals": {"primary_level": 0.95},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # 4. Run evaluation via CLI
        exit_code = main(["--config", config_path, "--output-dir", out_dir])
        assert exit_code == 0

        # Verify saved manifest
        manifests = [f for f in os.listdir(out_dir) if f.endswith("_manifest.json")]
        assert len(manifests) == 1
        manifest_file = os.path.join(out_dir, manifests[0])
        loaded = load_run_manifest(manifest_file)
        assert loaded.model_name == "test_st_gcn"
        assert loaded.split_evaluated == "test"
        assert loaded.sensor_count == N
        assert loaded.sample_count == S
        assert loaded.run_status == "complete"
        assert "MAE" in loaded.overall_metrics
        assert "NLL" in loaded.overall_metrics
        assert len(loaded.per_horizon_metrics) == H
