import argparse
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone

import h5py
import pandas as pd

from st_dssm.data import (
    verify_dataset,
)
from st_dssm.imputation import apply_causal_forward_fill, fit_fallback_statistics
from st_dssm.io import (
    _compute_array_checksum,
    load_and_validate_artifact,
    save_processed_artifact,
)
from st_dssm.missingness import extract_native_missingness
from st_dssm.preprocessing import (
    PerSensorScaler,
    calculate_split_boundaries,
    generate_windows,
)
from st_dssm.validator import validate_graph, validate_time_series


def _get_git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return rev.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def generate_pipeline(data_dir: str, output_dir: str, input_length: int = 12, forecast_horizon: int = 12) -> tuple[dict, dict]:
    """Runs the full canonical reproducible preprocessing pipeline and returns (arrays, metadata)."""
    # 1. Verify Phase 1 manifest and raw-file checksums
    print("Verifying Phase 1 canonical dataset...")
    manifest = verify_dataset(data_dir)
    
    pems_bay_path = os.path.join(data_dir, "pems-bay.h5")
    adj_mx_path = os.path.join(data_dir, "adj_mx_bay.pkl")

    # 2. Run the complete Phase 2 data gate
    print("Running Phase 2 data gate...")
    with h5py.File(pems_bay_path, "r") as f:
        group = f["speed"]
        values = group["block0_values"][:]
        cols_raw = group["block0_items"][:]
        idx_raw = group["axis1"][:]
    
    sensor_ids = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cols_raw]
    # PEMS-BAY timestamps are stored as nanoseconds since epoch in UTC, but the canonical data 
    # might not have tz explicitly set in the h5 file. We coerce to UTC.
    timestamps = pd.to_datetime(idx_raw).tz_localize("UTC")
    df = pd.DataFrame(values, index=timestamps, columns=sensor_ids)
    
    # NEW: Resample to strict 5-minute intervals to pad missing DST gap
    df = df.asfreq("5min")
    
    # Update values and timestamps from resampled df
    values = df.values
    timestamps = df.index
    
    with open(adj_mx_path, "rb") as f:
        sensor_ids_graph_raw, _, adj_mx = pickle.load(f, encoding='latin1')
    graph_sensor_ids = [str(sid) for sid in sensor_ids_graph_raw]
    
    validate_time_series(df, sensor_ids)
    validate_graph(adj_mx, graph_sensor_ids, sensor_ids)
    
    # 3. Load canonical time series and sensor ordering
    # 4. Construct and preserve the native-observation mask before repairing values
    print("Extracting native missingness...")
    values_with_nan, native_mask, missingness_stats = extract_native_missingness(values)
    
    # 5. Determine chronological split boundaries
    print("Calculating chronological split boundaries...")
    total_samples = len(timestamps)
    train_end, val_end, test_end = calculate_split_boundaries(total_samples)
    
    # 7. Fit every learned preprocessing parameter using only the training partition
    print("Fitting repair fallback statistics (training only)...")
    train_values_with_nan = values_with_nan[:train_end]
    fallback_stats = fit_fallback_statistics(train_values_with_nan)
    
    # 8. Apply frozen repair parameters to all partitions
    print("Applying causal forward fill with learned fallbacks...")
    imputed_values = apply_causal_forward_fill(values_with_nan, fallback_stats)
    
    # Split imputed values and masks
    train_imputed = imputed_values[:train_end]
    val_imputed = imputed_values[train_end:val_end]
    test_imputed = imputed_values[val_end:test_end]
    
    train_mask = native_mask[:train_end]
    val_mask = native_mask[train_end:val_end]
    test_mask = native_mask[val_end:test_end]
    
    # Fit scaling parameters using only the training partition
    print("Fitting PerSensorScaler (training only)...")
    scaler = PerSensorScaler()
    scaler.fit(train_imputed)
    
    # Apply frozen scaling parameters to all partitions
    print("Applying frozen scaler to all partitions...")
    train_scaled = scaler.transform(train_imputed)
    val_scaled = scaler.transform(val_imputed)
    test_scaled = scaler.transform(test_imputed)
    
    # 9. Generate windows independently inside each split
    print(f"Generating windows (L={input_length}, H={forecast_horizon})...")
    train_X, train_Y, train_X_mask, train_Y_mask = generate_windows(train_scaled, train_mask, input_length, forecast_horizon)
    val_X, val_Y, val_X_mask, val_Y_mask = generate_windows(val_scaled, val_mask, input_length, forecast_horizon)
    test_X, test_Y, test_X_mask, test_Y_mask = generate_windows(test_scaled, test_mask, input_length, forecast_horizon)
    
    arrays = {
        "train_X": train_X, "train_Y": train_Y, "train_X_mask": train_X_mask, "train_Y_mask": train_Y_mask,
        "val_X": val_X, "val_Y": val_Y, "val_X_mask": val_X_mask, "val_Y_mask": val_Y_mask,
        "test_X": test_X, "test_Y": test_Y, "test_X_mask": test_X_mask, "test_Y_mask": test_Y_mask,
        "scaler_means": scaler.mean_,
        "scaler_stds": scaler.std_,
        "repair_fallback_stats": fallback_stats,
    }
    
    # 10. Complete reproducibility metadata
    metadata = {
        "schema_version": "1.0",
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _get_git_revision(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_dataset_manifest": manifest,
        "configuration": {
            "input_length": input_length,
            "forecast_horizon": forecast_horizon,
        },
        "preprocessing": {
            "split_boundaries": {
                "train_end": train_end,
                "val_end": val_end,
                "test_end": test_end,
            },
            "split_timestamp_ranges": {
                "train": [timestamps[0].isoformat(), timestamps[train_end - 1].isoformat()],
                "val": [timestamps[train_end].isoformat(), timestamps[val_end - 1].isoformat()],
                "test": [timestamps[val_end].isoformat(), timestamps[test_end - 1].isoformat()],
            },
            "split_raw_observation_counts": {
                "train": train_end,
                "val": val_end - train_end,
                "test": test_end - val_end,
            },
            "split_window_counts": {
                "train": train_X.shape[0],
                "val": val_X.shape[0],
                "test": test_X.shape[0],
            },
            "missingness_semantics": "0.0 and NaN are dataset-native missing observations.",
            "missingness_counts": {
                "total_count": missingness_stats.total_count,
                "zero_count": missingness_stats.zero_count,
                "nan_count": missingness_stats.nan_count,
            },
            "scaler_fit_scope": "train_only",
            "repair_method": "causal_forward_fill_with_train_fallback",
            "repair_fit_scope": "train_only",
            "feature_names": ["speed"],
            "feature_units": ["mph"],
        },
        "sensor_ids": sensor_ids,
        "arrays": {} # Populated inside save_processed_artifact
    }
    
    return arrays, metadata


def cmd_generate(args):
    try:
        arrays, metadata = generate_pipeline(args.data_dir, args.output_dir, args.input_length, args.forecast_horizon)
        if not args.force and os.path.exists(os.path.join(args.output_dir, "processed_arrays.npz")):
            print(f"Error: Artifact already exists at {args.output_dir}. Use --force to overwrite.", file=sys.stderr)
            sys.exit(1)
            
        print(f"Persisting artifacts to {args.output_dir}...")
        save_processed_artifact(args.output_dir, arrays, metadata)
        
        # 11. Reload the artifact and validate it against its manifest
        print("Reloading and validating generated artifact...")
        load_and_validate_artifact(args.output_dir)
        print("Phase 3 canonical artifact generated and validated successfully.")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"Error during generation: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_validate(args):
    try:
        print(f"Validating artifact in {args.artifact_dir}...")
        loaded_arrays, metadata = load_and_validate_artifact(args.artifact_dir)
        print("Artifact validated successfully:")
        print(f"  Schema version: {metadata.get('schema_version')}")
        print(f"  Source record: {metadata['source_dataset_manifest']['source']['record']}")
        print(f"  Train windows: {loaded_arrays['train_X'].shape[0]}")
        print(f"  Val windows: {loaded_arrays['val_X'].shape[0]}")
        print(f"  Test windows: {loaded_arrays['test_X'].shape[0]}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"Error validating artifact: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_smoke(args):
    try:
        print(f"Loading existing artifact from {args.artifact_dir} for smoke regeneration comparison...")
        orig_arrays, orig_metadata = load_and_validate_artifact(args.artifact_dir)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Regenerating pipeline into temporary directory {temp_dir}...")
            new_arrays, new_metadata = generate_pipeline(args.data_dir, temp_dir, args.input_length, args.forecast_horizon)
            
            # Compare strictly
            # Exclude volatile fields
            def filter_metadata(m):
                # Copy without volatile keys
                out = dict(m)
                out.pop("creation_timestamp_utc", None)
                if "source_dataset_manifest" in out:
                    sm = dict(out["source_dataset_manifest"])
                    sm.pop("timestamp_utc", None)
                    out["source_dataset_manifest"] = sm
                if "environment" in out:
                    out.pop("environment", None)
                if "arrays" in out:
                    out.pop("arrays", None)
                return out
                
            orig_meta_filtered = filter_metadata(orig_metadata)
            new_meta_filtered = filter_metadata(new_metadata)
            
            if orig_meta_filtered != new_meta_filtered:
                print("Error: Smoke test failed. Deterministic metadata comparison mismatch.", file=sys.stderr)
                sys.exit(1)
                
            # Compare array structures and checksums
            for name, arr in orig_arrays.items():
                if name not in new_arrays:
                    print(f"Error: Array {name} missing in newly generated artifact.", file=sys.stderr)
                    sys.exit(1)
                
                new_arr = new_arrays[name]
                if arr.shape != new_arr.shape:
                    print(f"Error: Shape mismatch for {name}: orig {arr.shape} != new {new_arr.shape}", file=sys.stderr)
                    sys.exit(1)
                
                if arr.dtype != new_arr.dtype:
                    print(f"Error: Dtype mismatch for {name}: orig {arr.dtype} != new {new_arr.dtype}", file=sys.stderr)
                    sys.exit(1)
                    
                orig_checksum = _compute_array_checksum(arr)
                new_checksum = _compute_array_checksum(new_arr)
                
                if orig_checksum != new_checksum:
                    print(f"Error: Checksum mismatch for array {name}.", file=sys.stderr)
                    sys.exit(1)
                    
            print("Smoke test passed: Regenerated artifact is structurally and deterministically identical.")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"Error during smoke test: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ST-DSSM Phase 3 Reproducible Preprocessing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_gen = subparsers.add_parser("generate", help="Generate the Phase 3 processed artifact from the verified canonical source.")
    parser_gen.add_argument("--data-dir", default="data/raw", help="Directory containing raw PEMS-BAY files.")
    parser_gen.add_argument("--output-dir", default="data/processed", help="Directory to save the processed artifact.")
    parser_gen.add_argument("--input-length", type=int, default=12, help="Window input length L")
    parser_gen.add_argument("--forecast-horizon", type=int, default=12, help="Window forecast horizon H")
    parser_gen.add_argument("--force", action="store_true", help="Overwrite existing artifact if present.")
    
    parser_val = subparsers.add_parser("validate", help="Validate an existing processed artifact.")
    parser_val.add_argument("--artifact-dir", default="data/processed", help="Directory containing the processed artifact.")
    
    parser_smoke = subparsers.add_parser("smoke", help="Perform a deterministic smoke regeneration and metadata comparison.")
    parser_smoke.add_argument("--data-dir", default="data/raw", help="Directory containing raw PEMS-BAY files.")
    parser_smoke.add_argument("--artifact-dir", default="data/processed", help="Directory containing the processed artifact to compare against.")
    parser_smoke.add_argument("--input-length", type=int, default=12, help="Window input length L")
    parser_smoke.add_argument("--forecast-horizon", type=int, default=12, help="Window forecast horizon H")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "smoke":
        cmd_smoke(args)

if __name__ == "__main__":
    main()
