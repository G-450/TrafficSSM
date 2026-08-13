import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from st_dssm.cli.preprocess import generate_pipeline, main


@pytest.fixture
def mock_dataset_loaders():
    from unittest.mock import mock_open
    with patch("st_dssm.cli.preprocess.verify_dataset") as mock_verify, \
         patch("h5py.File") as mock_h5py, \
         patch("pickle.load") as mock_pickle, \
         patch("st_dssm.cli.preprocess.validate_time_series"), \
         patch("st_dssm.cli.preprocess.validate_graph"), \
         patch("builtins.open", mock_open()):
         
        mock_verify.return_value = {"status": "PASSED"}
        
        # Setup h5py mock for time series
        mock_file = MagicMock()
        mock_h5py.return_value.__enter__.return_value = mock_file
        
        speed_group = MagicMock()
        mock_file.__getitem__.return_value = speed_group
        
        # Small dataset: 100 timesteps, 5 sensors
        # 0.0s for missingness
        values = np.random.rand(100, 5)
        values[5, 2] = 0.0 # native missingness
        values[10, 3] = np.nan
        
        speed_group.__getitem__.side_effect = lambda k: {
            "block0_values": values,
            "block0_items": [f"sensor_{i}".encode() for i in range(5)],
            "axis1": pd.date_range("2026-01-01", periods=100, freq="5min").astype(np.int64)
        }[k]
        
        # Setup pickle mock for graph
        sensor_ids = [f"sensor_{i}" for i in range(5)]
        adj_mx = np.eye(5)
        mock_pickle.return_value = (sensor_ids, {}, adj_mx)
        
        yield
        
def test_generate_pipeline_success(mock_dataset_loaders):
    with tempfile.TemporaryDirectory() as tmpdir:
        arrays, metadata = generate_pipeline("dummy/data", tmpdir, input_length=5, forecast_horizon=5)
        
        assert "train_X" in arrays
        assert "scaler_means" in arrays
        
        assert metadata["preprocessing"]["split_raw_observation_counts"]["train"] == 70
        assert metadata["preprocessing"]["split_raw_observation_counts"]["val"] == 10
        assert metadata["preprocessing"]["split_raw_observation_counts"]["test"] == 20
        
        # For L=5, H=5, split 70 -> S = 70 - 5 - 5 + 1 = 61
        assert arrays["train_X"].shape == (61, 5, 5, 1)

@patch("st_dssm.cli.preprocess.generate_pipeline")
@patch("st_dssm.cli.preprocess.save_processed_artifact")
@patch("st_dssm.cli.preprocess.load_and_validate_artifact")
def test_cli_generate_success(mock_load, mock_save, mock_gen):
    mock_gen.return_value = ({}, {})
    
    with patch.object(sys, 'argv', ['preprocess', 'generate', '--force']):
        main()


@patch("st_dssm.cli.preprocess.load_and_validate_artifact")
def test_cli_validate_success(mock_load):
    mock_load.return_value = (
        {"train_X": np.zeros((10, 12, 325, 1)), "val_X": np.zeros((10, 12, 325, 1)), "test_X": np.zeros((10, 12, 325, 1))}, 
        {"schema_version": "1.0", "source_dataset_manifest": {"source": {"record": "mock"}}}
    )
    
    with patch.object(sys, 'argv', ['preprocess', 'validate']):
        main()


@patch("st_dssm.cli.preprocess.generate_pipeline")
@patch("st_dssm.cli.preprocess.load_and_validate_artifact")
def test_cli_smoke_success(mock_load, mock_gen):
    mock_meta = {
        "schema_version": "1.0",
        "arrays": {},
        "environment": "remove_me",
        "creation_timestamp_utc": "remove_me"
    }
    mock_arrays = {"train_X": np.zeros((10, 12, 325, 1))}
    
    mock_load.return_value = (mock_arrays, mock_meta)
    mock_gen.return_value = (mock_arrays, mock_meta)
    
    with patch.object(sys, 'argv', ['preprocess', 'smoke']):
        main()
