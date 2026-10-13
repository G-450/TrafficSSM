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


def test_git_dirty_ignores_untracked_files(tmp_path, monkeypatch) -> None:
    """Untracked files (such as earlier run outputs) must not mark a run's code as dirty."""
    import subprocess

    from st_dssm.cli.evaluate import _get_git_commit

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "code.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "run_output.json").write_text("{}", encoding="utf-8")
    _, dirty = _get_git_commit()
    assert dirty is False

    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    _, dirty = _get_git_commit()
    assert dirty is True


class TestEvaluationProvenance:
    """A saved manifest must be a complete record the first time it is written.

    The manifest used to be saved and then patched with training provenance, so a
    run killed in between (e.g. during plot rendering) left a manifest reporting
    seed ``None`` and ``0`` epochs for a result that was otherwise valid.
    """

    @staticmethod
    def _artifact_and_predictions(tmp_path):
        artifact_dir = str(tmp_path / "artifact")
        S, H, N = 6, 12, 3
        arrays = {
            "test_X": np.zeros((S, 12, N, 1), dtype=np.float32),
            "test_X_mask": np.ones((S, 12, N, 1), dtype=np.float32),
            "test_Y": np.ones((S, H, N, 1), dtype=np.float32) * 0.5,
            "test_Y_mask": np.ones((S, H, N, 1), dtype=np.float32),
        }
        metadata = {
            "schema_version": "1.0",
            "scaler": {"mean": [50.0, 55.0, 60.0], "std": [5.0, 6.0, 7.0]},
            "sensor_ids": ["s1", "s2", "s3"],
            "splits": {"train": 60, "val": 12, "test": S},
        }
        save_processed_artifact(artifact_dir, arrays, metadata)

        pred_path = str(tmp_path / "preds.npz")
        np.savez_compressed(
            pred_path,
            mu=np.ones((S, H, N, 1), dtype=np.float32) * 0.4,
            sigma=np.ones((S, H, N, 1), dtype=np.float32) * 0.2,
        )
        return artifact_dir, pred_path

    def test_provenance_is_recorded_in_the_first_manifest_write(self, tmp_path):
        from st_dssm.cli.evaluate import run_evaluation

        artifact_dir, pred_path = self._artifact_and_predictions(tmp_path)
        out_dir = str(tmp_path / "results")
        config = {
            "run_id": "st_dssm-test-seed2026-fixture",
            "model_name": "st_dssm",
            "artifact_dir": artifact_dir,
            "predictions_path": pred_path,
            "split": "test",
            "output_dir": out_dir,
        }
        provenance = {
            "training_seed": 2026,
            "checkpoint_path": "ckpt/best.pt",
            "selection_metric": "val_nll",
            "best_epoch": 58,
            "total_epochs": 60,
            "trainable_parameters": 397058,
        }

        # save_plots=False proves the provenance is already in the file that the
        # evaluator writes, independently of anything that happens afterwards.
        _manifest, manifest_path = run_evaluation(
            config, output_dir=out_dir, save_plots=False, provenance=provenance
        )

        loaded = load_run_manifest(manifest_path)
        assert loaded.training_seed == 2026
        assert loaded.checkpoint_path == "ckpt/best.pt"
        assert loaded.selection_metric == "val_nll"
        assert loaded.best_epoch == 58
        assert loaded.total_epochs == 60
        assert loaded.trainable_parameters == 397058
        assert loaded.run_status == "complete"

    def test_unknown_provenance_field_is_rejected(self, tmp_path):
        import pytest

        from st_dssm.cli.evaluate import run_evaluation
        from st_dssm.metrics import MetricError

        artifact_dir, pred_path = self._artifact_and_predictions(tmp_path)
        config = {
            "run_id": "r",
            "model_name": "st_dssm",
            "artifact_dir": artifact_dir,
            "predictions_path": pred_path,
            "split": "test",
        }
        with pytest.raises(MetricError, match="Unknown provenance field"):
            run_evaluation(
                config,
                output_dir=str(tmp_path / "out"),
                save_plots=False,
                provenance={"traning_seed": 2026},
            )

    def test_manifest_records_data_config_and_package_provenance(self, tmp_path):
        from st_dssm.cli.evaluate import run_evaluation

        artifact_dir, pred_path = self._artifact_and_predictions(tmp_path)
        out_dir = str(tmp_path / "results")
        config = {
            "run_id": "run-prov",
            "model_name": "st_dssm",
            "artifact_dir": artifact_dir,
            "predictions_path": pred_path,
            "split": "test",
        }
        _manifest, manifest_path = run_evaluation(config, output_dir=out_dir, save_plots=False)

        loaded = load_run_manifest(manifest_path)
        assert loaded.config_hash != ""
        assert loaded.dataset_manifest_checksum != ""
        assert loaded.split_artifact_checksum != ""
        assert "numpy" in loaded.package_versions
        assert "python" in loaded.package_versions

    def test_provenance_can_be_read_from_a_checkpoint(self, tmp_path):
        from torch import nn

        from st_dssm.cli.evaluate import provenance_from_checkpoint
        from st_dssm.training import save_checkpoint

        model = nn.Linear(4, 3)  # 4*3 weights + 3 biases = 15 parameters
        ckpt = str(tmp_path / "model_best.pt")
        save_checkpoint(ckpt, model, epoch=42, val_loss=0.1234, config={"seed": 2027})

        provenance = provenance_from_checkpoint(ckpt)
        assert provenance["training_seed"] == 2027
        assert provenance["best_epoch"] == 42
        assert provenance["trainable_parameters"] == 15
        assert provenance["checkpoint_path"] == ckpt

    def test_cli_regenerates_a_manifest_in_place_with_provenance(self, tmp_path):
        artifact_dir, pred_path = self._artifact_and_predictions(tmp_path)
        out_dir = str(tmp_path / "results")

        exit_code = main(
            [
                "--artifact-dir", artifact_dir,
                "--predictions-path", pred_path,
                "--split", "test",
                "--output-dir", out_dir,
                "--run-id", "st_dssm-test-seed2028-original",
                "--model-name", "st_dssm",
                "--training-seed", "2028",
                "--selection-metric", "val_nll",
                "--best-epoch", "60",
                "--total-epochs", "60",
                "--trainable-parameters", "397058",
                "--no-plots",
            ]
        )
        assert exit_code == 0

        # The run id is preserved, so the original run's manifest is replaced
        # rather than a second manifest being created alongside it.
        manifest_path = os.path.join(out_dir, "st_dssm-test-seed2028-original_manifest.json")
        loaded = load_run_manifest(manifest_path)
        assert loaded.run_id == "st_dssm-test-seed2028-original"
        assert loaded.model_name == "st_dssm"
        assert loaded.training_seed == 2028
        assert loaded.best_epoch == 60
        assert loaded.total_epochs == 60
        assert loaded.trainable_parameters == 397058
