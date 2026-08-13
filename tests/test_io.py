import numpy as np
import tempfile
import pytest
import os
from st_dssm.io import save_processed_artifact, load_and_validate_artifact, ArtifactError

def test_io_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        arrays = {
            "train_X": np.ones((10, 12, 325, 1), dtype=np.float32),
            "train_Y": np.zeros((10, 12, 325, 1), dtype=np.float32),
            "train_X_mask": np.ones((10, 12, 325, 1), dtype=bool),
            "scaler_means": np.zeros((325,), dtype=np.float32),
            "scaler_stds": np.ones((325,), dtype=np.float32),
        }
        metadata = {
            "schema_version": "1.0",
            "sensor_ids": [f"sensor_{i}" for i in range(325)],
        }
        
        save_processed_artifact(tmpdir, arrays, metadata)
        
        loaded_arrays, loaded_meta = load_and_validate_artifact(tmpdir)
        
        assert loaded_meta["schema_version"] == "1.0"
        for name in arrays:
            assert name in loaded_arrays
            np.testing.assert_array_equal(arrays[name], loaded_arrays[name])
            
def test_io_corruption_missing_array():
    with tempfile.TemporaryDirectory() as tmpdir:
        arrays = {
            "train_X": np.ones((10, 12, 325, 1), dtype=np.float32),
            "scaler_means": np.zeros((325,), dtype=np.float32),
        }
        metadata = {
            "schema_version": "1.0",
            "sensor_ids": [f"sensor_{i}" for i in range(325)],
        }
        
        save_processed_artifact(tmpdir, arrays, metadata)
        
        # Corrupt the npz
        npz_path = os.path.join(tmpdir, "processed_arrays.npz")
        # Save a new npz without scaler_means
        np.savez_compressed(npz_path, train_X=arrays["train_X"])
        
        with pytest.raises(ArtifactError, match="Missing array in NPZ"):
            load_and_validate_artifact(tmpdir)

def test_io_corruption_checksum_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        arrays = {
            "train_X": np.ones((10, 12, 325, 1), dtype=np.float32),
        }
        metadata = {
            "schema_version": "1.0",
            "sensor_ids": [f"sensor_{i}" for i in range(325)],
        }
        
        save_processed_artifact(tmpdir, arrays, metadata)
        
        # Corrupt the npz data
        npz_path = os.path.join(tmpdir, "processed_arrays.npz")
        np.savez_compressed(npz_path, train_X=np.zeros((10, 12, 325, 1), dtype=np.float32))
        
        with pytest.raises(ArtifactError, match="Checksum mismatch"):
            load_and_validate_artifact(tmpdir)
