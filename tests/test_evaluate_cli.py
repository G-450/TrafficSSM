"""Tests for evaluation CLI command (st-dssm-evaluate).

Tests synthetic smoke mode, output directory creation, manifest creation,
and plot generation.
"""

from __future__ import annotations

import os

from st_dssm.cli.evaluate import main, run_synthetic_smoke
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

    def test_main_cli_with_config(self, tmp_path):
        out_dir = str(tmp_path / "config_results")
        config_path = "configs/evaluation/synthetic.yaml"
        exit_code = main(["--config", config_path, "--output-dir", out_dir])
        assert exit_code == 0
