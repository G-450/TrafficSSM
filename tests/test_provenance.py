import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from st_dssm.cli.provenance import main
from st_dssm.data import (
    EXPECTED_FILES,
    ChecksumMismatchError,
    DataProvenanceError,
    StructuralValidationError,
    verify_dataset,
)


@pytest.fixture
def mock_dependencies():
    with patch("st_dssm.data.compute_md5") as mock_md5, \
         patch("os.path.getsize", return_value=1024), \
         patch("os.path.exists", return_value=True), \
         patch("pandas.read_hdf") as mock_read_hdf, \
         patch("builtins.open", mock_open()), \
         patch("pickle.load") as mock_pickle_load, \
         patch("h5py.File") as mock_h5py:
        
        # Default successful setup
        mock_md5.side_effect = lambda f: EXPECTED_FILES[os.path.basename(f)]
        
        # Time series mock via h5py
        # h5py mock was already yielded. Let's customize it when it's called with pems_bay_path.
        # However, the h5py mock is a context manager.
        mock_file = MagicMock()
        mock_h5py.return_value.__enter__.return_value = mock_file
        
        # We need mock_file["speed"]["block0_values"].shape to be (10, 325)
        # And mock_file["speed"]["block0_items"][:] to be bytes or strings
        speed_group = MagicMock()
        mock_file.__getitem__.return_value = speed_group
        
        block0_values = MagicMock()
        block0_values.shape = (52116, 325)
        
        block0_items = MagicMock()
        block0_items.__getitem__.return_value = [str(i).encode('utf-8') for i in range(325)]
        
        def speed_getitem(key):
            if key == "block0_values":
                return block0_values
            if key == "block0_items":
                return block0_items
            return MagicMock()
            
        speed_group.__getitem__.side_effect = speed_getitem
        
        mock_file.keys.return_value = ["sensor_id"]
        sensor_id_group = MagicMock()
        sensor_id_group.__getitem__.return_value = [str(i).encode('utf-8') for i in range(325)]
        
        def file_getitem(key):
            if key == "speed":
                return speed_group
            if key == "sensor_id":
                return sensor_id_group
            return MagicMock()
            
        mock_file.__getitem__.side_effect = file_getitem
        
        # Graph mock
        sensor_ids = [str(i) for i in range(325)]
        adj_mx = MagicMock()
        adj_mx.shape = (325, 325)
        mock_pickle_load.return_value = (sensor_ids, {}, adj_mx)

        yield {
            "md5": mock_md5,
            "read_hdf": mock_read_hdf,
            "pickle_load": mock_pickle_load,
            "h5py": mock_h5py,
        }

def test_verify_dataset_success(mock_dependencies):
    manifest = verify_dataset("dummy/dir")
    assert manifest["verification_status"] == "PASSED"
    assert manifest["structure"]["sensor_count"] == 325
    assert manifest["structure"]["sensor_ids_exact_match"] is True

def test_verify_dataset_missing_file(mock_dependencies):
    with patch("os.path.exists", return_value=False), pytest.raises(FileNotFoundError):
        verify_dataset("dummy/dir")

def test_verify_dataset_checksum_mismatch(mock_dependencies):
    mock_dependencies["md5"].side_effect = lambda f: "wrong_md5"
    with pytest.raises(ChecksumMismatchError):
        verify_dataset("dummy/dir")

def test_verify_dataset_wrong_sensor_count(mock_dependencies):
    mock_file = mock_dependencies["h5py"].return_value.__enter__.return_value
    mock_file["speed"]["block0_values"].shape = (52116, 100)
    with pytest.raises(StructuralValidationError, match="Expected 52116 rows and 325 sensors"):
        verify_dataset("dummy/dir")

def test_verify_dataset_non_unique_sensors(mock_dependencies):
    mock_file = mock_dependencies["h5py"].return_value.__enter__.return_value
    cols = [str(i).encode('utf-8') for i in range(324)] + [b"0"]
    mock_file["speed"]["block0_items"].__getitem__.return_value = cols
    with pytest.raises(StructuralValidationError, match="not unique"):
        verify_dataset("dummy/dir")

def test_verify_dataset_graph_shape_mismatch(mock_dependencies):
    sensor_ids = [str(i) for i in range(325)]
    adj_mx = MagicMock()
    adj_mx.shape = (325, 100)
    mock_dependencies["pickle_load"].return_value = (sensor_ids, {}, adj_mx)
    with pytest.raises(StructuralValidationError, match="Expected graph shape"):
        verify_dataset("dummy/dir")

def test_verify_dataset_sensor_order_mismatch(mock_dependencies):
    # Time series has 0..324
    # Graph has 324..0
    sensor_ids = [str(i) for i in reversed(range(325))]
    adj_mx = MagicMock()
    adj_mx.shape = (325, 325)
    mock_dependencies["pickle_load"].return_value = (sensor_ids, {}, adj_mx)
    with pytest.raises(StructuralValidationError, match="Exact equality and ordering"):
        verify_dataset("dummy/dir")

@patch("st_dssm.cli.provenance.verify_dataset")
@patch("st_dssm.cli.provenance.yaml.safe_load")
@patch("builtins.open", mock_open())
def test_cli_success(mock_yaml, mock_verify, tmp_path):
    mock_yaml.return_value = {
        "data_dir": str(tmp_path / "data/raw"),
        "manifest_path": str(tmp_path / "manifest.json")
    }
    mock_verify.return_value = {"status": "PASSED"}
    
    with patch.object(sys, 'argv', ['provenance', '--config', 'dummy.yaml']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0

@patch("st_dssm.cli.provenance.verify_dataset")
@patch("st_dssm.cli.provenance.yaml.safe_load")
@patch("builtins.open", mock_open())
def test_cli_failure(mock_yaml, mock_verify, tmp_path):
    mock_yaml.return_value = {
        "data_dir": str(tmp_path / "data/raw"),
        "manifest_path": str(tmp_path / "manifest.json")
    }
    mock_verify.side_effect = DataProvenanceError("Failure")
    
    with patch.object(sys, 'argv', ['provenance', '--config', 'dummy.yaml']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
