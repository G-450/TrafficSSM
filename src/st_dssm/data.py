import hashlib
import os
import pickle
import platform
from datetime import datetime, timezone

import h5py


class DataProvenanceError(Exception):
    pass


class ChecksumMismatchError(DataProvenanceError):
    pass


class StructuralValidationError(DataProvenanceError):
    pass


EXPECTED_FILES = {
    "pems-bay.h5": "bfa47e6ee7cc2d665e9b62c0c95c0b41",
    "adj_mx_bay.pkl": "55d25daceac847312b7748c59ded6f77",
    "pems-bay-meta.h5": "fbde3d20ac413e29f304d0cb528cfdc2",
}

ZENODO_URLS = {
    "pems-bay.h5": "https://zenodo.org/records/4263971/files/pems-bay.h5",
    "adj_mx_bay.pkl": "https://zenodo.org/records/4263971/files/adj_mx_bay.pkl",
    "pems-bay-meta.h5": "https://zenodo.org/records/4263971/files/pems-bay-meta.h5",
}


def compute_md5(file_path: str) -> str:
    """Stream-computes the MD5 checksum of a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_dataset(data_dir: str) -> dict:
    """
    Verifies the PEMS-BAY dataset checksums and structural integrity.
    Returns the dataset manifest dictionary if successful.
    """
    observed_checksums = {}
    file_sizes = {}
    
    # 1. Verify Checksums First
    for filename, expected_md5 in EXPECTED_FILES.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {filename}")
        
        observed_md5 = compute_md5(file_path)
        observed_checksums[filename] = observed_md5
        file_sizes[filename] = os.path.getsize(file_path)
        
        if observed_md5 != expected_md5:
            raise ChecksumMismatchError(
                f"MD5 mismatch for {filename}: expected {expected_md5}, got {observed_md5}"
            )

    # 2. Inspect Dataset Structure (only after checksum passes)
    pems_bay_path = os.path.join(data_dir, "pems-bay.h5")
    adj_mx_path = os.path.join(data_dir, "adj_mx_bay.pkl")
    meta_path = os.path.join(data_dir, "pems-bay-meta.h5")

    # Time series
    with h5py.File(pems_bay_path, "r") as f:
        group = f["speed"]
        df_shape = group["block0_values"].shape
        cols_raw = group["block0_items"][:]
        sensor_ids_ts = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cols_raw]
    
    if df_shape != (52116, 325):
        raise StructuralValidationError(f"Expected 52116 rows and 325 sensors, got {df_shape}")
    
    if len(set(sensor_ids_ts)) != len(sensor_ids_ts):
        raise StructuralValidationError("Time series sensor IDs are not unique.")

    # Graph
    with open(adj_mx_path, "rb") as f:
        sensor_ids_graph, _sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
    
    graph_shape = adj_mx.shape
    if graph_shape != (325, 325):
        raise StructuralValidationError(f"Expected graph shape (325, 325), got {graph_shape}")
    
    # Strict matching
    sensor_ids_graph = [str(sid) for sid in sensor_ids_graph]
    ts_sensor_ids_str = [str(sid) for sid in sensor_ids_ts]
    
    if ts_sensor_ids_str != sensor_ids_graph:
        raise StructuralValidationError("Exact equality and ordering of time-series sensor IDs and graph sensor IDs failed.")

    # Metadata
    with h5py.File(meta_path, 'r') as f:
        meta_keys = list(f.keys())
    
    # 3. Create Manifest
    manifest = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "record": "Zenodo PEMS-BAY release 4263971",
            "doi": "10.5281/zenodo.4263971",
            "version": "v2",
            "urls": ZENODO_URLS,
            "license": "Creative Commons Attribution 4.0 International",
        },
        "files": {
            name: {
                "byte_size": file_sizes[name],
                "expected_md5": EXPECTED_FILES[name],
                "observed_md5": observed_checksums[name],
            } for name in EXPECTED_FILES
        },
        "structure": {
            "time_series_shape": list(df_shape),
            "sensor_count": len(sensor_ids_ts),
            "graph_shape": list(graph_shape),
            "node_count": len(sensor_ids_graph),
            "sensor_ids_exact_match": True,
            "sensor_ids_unique": True,
            "metadata_identifiers": meta_keys,
        },
        "semantics": {
            "native_missingness": ["0.0", "NaN"],
        },
        "verification_status": "PASSED",
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
    }
    return manifest
