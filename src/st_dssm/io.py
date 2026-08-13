import hashlib
import json
import os
import tempfile

import numpy as np


class ArtifactError(Exception):
    pass

def _compute_array_checksum(arr: np.ndarray) -> str:
    """Computes a deterministic MD5 checksum for a numpy array's data buffer."""
    # Ensure contiguous buffer
    arr_c = np.ascontiguousarray(arr)
    return hashlib.md5(arr_c.data.tobytes()).hexdigest()

def save_processed_artifact(
    output_dir: str,
    arrays: dict[str, np.ndarray],
    metadata: dict
) -> None:
    """
    Saves the processed artifact as an NPZ file and a metadata JSON file safely using atomic replacement.
    
    Args:
        output_dir: The directory to store the artifact.
        arrays: Dictionary of array names to numpy arrays.
        metadata: Dictionary containing reproducibility metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "processed_arrays.npz")
    meta_path = os.path.join(output_dir, "processed_metadata.json")

    # Add array checksums to metadata
    array_metadata = {}
    for name, arr in arrays.items():
        array_metadata[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "checksum": _compute_array_checksum(arr)
        }
    metadata["arrays"] = array_metadata

    # Write using temporary files and atomic rename to prevent partial writes
    fd_npz, tmp_npz_path = tempfile.mkstemp(dir=output_dir, suffix=".npz")
    os.close(fd_npz)
    
    fd_meta, tmp_meta_path = tempfile.mkstemp(dir=output_dir, suffix=".json")
    os.close(fd_meta)
    
    try:
        np.savez_compressed(tmp_npz_path, **arrays)
        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            
        # Atomic replace
        os.replace(tmp_npz_path, npz_path)
        os.replace(tmp_meta_path, meta_path)
    finally:
        # Cleanup temps if anything failed
        if os.path.exists(tmp_npz_path):
            os.remove(tmp_npz_path)
        if os.path.exists(tmp_meta_path):
            os.remove(tmp_meta_path)

def load_and_validate_artifact(artifact_dir: str) -> tuple[dict[str, np.ndarray], dict]:
    """
    Loads and rigorously validates the processed artifact against its metadata.
    
    Args:
        artifact_dir: The directory containing processed_arrays.npz and processed_metadata.json.
        
    Returns:
        tuple containing (arrays_dict, metadata_dict).
    """
    npz_path = os.path.join(artifact_dir, "processed_arrays.npz")
    meta_path = os.path.join(artifact_dir, "processed_metadata.json")
    
    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise ArtifactError(f"Artifact files not found in {artifact_dir}")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    if metadata.get("schema_version") != "1.0":
        raise ArtifactError(f"Unsupported schema version: {metadata.get('schema_version')}")
        
    expected_arrays = metadata.get("arrays", {})
    loaded_arrays = {}
    
    with np.load(npz_path) as data:
        for name, expected_meta in expected_arrays.items():
            if name not in data:
                raise ArtifactError(f"Missing array in NPZ: {name}")
                
            arr = data[name]
            loaded_arrays[name] = arr
            
            # Validate shape
            if list(arr.shape) != expected_meta["shape"]:
                raise ArtifactError(f"Shape mismatch for {name}: expected {expected_meta['shape']}, got {list(arr.shape)}")
                
            # Validate dtype
            if str(arr.dtype) != expected_meta["dtype"]:
                raise ArtifactError(f"Dtype mismatch for {name}: expected {expected_meta['dtype']}, got {arr.dtype!s}")
                
            # Validate checksum
            actual_checksum = _compute_array_checksum(arr)
            if actual_checksum != expected_meta["checksum"]:
                raise ArtifactError(f"Checksum mismatch for {name}: expected {expected_meta['checksum']}, got {actual_checksum}")
                
            # If array is expected to be finite and float, check it
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                # For masks or target data, it might contain NaN natively if we didn't fill targets. 
                # But the requirements say: "repaired outputs required for scaling/windowing must be finite"
                # We will strictly check that scaled X arrays are finite.
                if (name.endswith("_X") or name in ["scaler_means", "scaler_stds"]):
                    raise ArtifactError(f"Array {name} contains non-finite values.")
                            
    # Verify exact sensor ID alignment
    expected_sensor_ids = metadata.get("sensor_ids", [])
    if not expected_sensor_ids:
        raise ArtifactError("Sensor IDs not found in metadata.")
        
    # Check shape alignment across keys
    # Window arrays should have length of expected_sensor_ids at axis=2 (or -2)
    sensor_count = len(expected_sensor_ids)
    for name, arr in loaded_arrays.items():
        if name in ["train_X", "train_Y", "val_X", "val_Y", "test_X", "test_Y", "train_X_mask", "train_Y_mask", "val_X_mask", "val_Y_mask", "test_X_mask", "test_Y_mask"]:
            if arr.shape[2] != sensor_count:
                raise ArtifactError(f"Array {name} sensor axis mismatch: expected {sensor_count}, got {arr.shape[2]}")
        elif name in ["scaler_means", "scaler_stds", "repair_fallback_stats"] and arr.shape[0] != sensor_count:
            raise ArtifactError(f"Array {name} shape mismatch: expected {sensor_count}, got {arr.shape[0]}")
                
    return loaded_arrays, metadata
