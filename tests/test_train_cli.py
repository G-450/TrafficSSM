"""Tests for the st-dssm-train CLI."""

from __future__ import annotations

import os

from st_dssm.cli.train import main


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
    import argparse
    from unittest.mock import patch
    
    with patch("st_dssm.cli.train.train_st_dssm") as mock_train:
        main(["--resume", ckpt_path])
        mock_train.assert_called_once()
        assert mock_train.call_args[1]["resume_path"] == ckpt_path
