import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from st_dssm.cli.validate import main


@patch("st_dssm.cli.validate.verify_dataset")
@patch("st_dssm.cli.validate.h5py.File")
@patch("builtins.open")
@patch("st_dssm.cli.validate.pickle.load")
@patch("st_dssm.cli.validate.validate_time_series")
@patch("st_dssm.cli.validate.validate_graph")
@patch("st_dssm.cli.validate.extract_native_missingness")
@patch("st_dssm.cli.validate.os.makedirs")
@patch("st_dssm.cli.validate.json.dump")
def test_cli_validate_success(
    mock_json_dump, mock_makedirs, mock_extract, mock_val_graph, mock_val_ts, 
    mock_pickle, mock_open, mock_h5py, mock_verify
):
    # Setup mocks
    mock_verify.return_value = {"status": "PASSED"}
    
    # Mock h5py File
    mock_file = MagicMock()
    mock_h5py.return_value.__enter__.return_value = mock_file
    
    speed_group = MagicMock()
    mock_file.__getitem__.return_value = speed_group
    
    # Mock data inside speed group
    speed_group.__getitem__.side_effect = lambda key: {
        "block0_values": MagicMock(__getitem__=lambda s, k: np.zeros((10, 2))),
        "block0_items": MagicMock(__getitem__=lambda s, k: [b"1", b"2"]),
        "axis1": MagicMock(__getitem__=lambda s, k: np.array([0]*10))
    }[key]
    
    mock_pickle.return_value = ([b"1", b"2"], {}, MagicMock())
    
    stats_mock = MagicMock()
    stats_mock.total_count = 20
    stats_mock.missing_count = 2
    mock_extract.return_value = (MagicMock(), MagicMock(), stats_mock)
    
    with patch.object(sys, 'argv', ['validate', '--data-dir', 'dummy']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
        
    mock_verify.assert_called_once()
    mock_h5py.assert_called_once()
    mock_val_ts.assert_called_once()
    mock_val_graph.assert_called_once()
    mock_extract.assert_called_once()
    mock_json_dump.assert_called_once()


@patch("st_dssm.cli.validate.verify_dataset")
def test_cli_validate_verify_fails(mock_verify):
    mock_verify.side_effect = Exception("Checksum error")
    
    with patch.object(sys, 'argv', ['validate', '--data-dir', 'dummy']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
