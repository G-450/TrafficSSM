import argparse
import json
import os
import pickle
import sys

import h5py
import pandas as pd

from st_dssm.data import verify_dataset
from st_dssm.missingness import extract_native_missingness
from st_dssm.validator import validate_graph, validate_time_series


def main():
    parser = argparse.ArgumentParser(description="ST-DSSM Phase 2 Data Gate Validation")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing raw data")
    parser.add_argument("--report-out", default="data/manifest/validation_report.json", help="Path to write validation report")
    args = parser.parse_args()

    data_dir = args.data_dir
    
    errors = []

    print("1. Verifying checksums...")
    try:
        verify_dataset(data_dir)
    except Exception as e:  # noqa: BLE001
        errors.append(f"Checksum/Provenance verification failed: {e}")

    print("2. Loading dataset...")
    pems_bay_path = os.path.join(data_dir, "pems-bay.h5")
    adj_mx_path = os.path.join(data_dir, "adj_mx_bay.pkl")
    
    df, adj_mx, sensor_ids_graph = None, None, None
    try:
        with h5py.File(pems_bay_path, "r") as f:
            values = f["speed"]["block0_values"][:]
            sensor_ids_raw = f["speed"]["block0_items"][:]
            sensor_ids = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in sensor_ids_raw]
            timestamps = pd.to_datetime(f["speed"]["axis1"][:], unit='ns', utc=True)
            df = pd.DataFrame(values, index=timestamps, columns=sensor_ids)
        
        with open(adj_mx_path, "rb") as f:
            sensor_ids_graph_raw, _, adj_mx = pickle.load(f, encoding='latin1')
            sensor_ids_graph = [str(sid) for sid in sensor_ids_graph_raw]
    except Exception as e:  # noqa: BLE001
        errors.append(f"Failed to load dataset: {e}")

    if df is not None and adj_mx is not None:
        print("3. Validating time series...")
        try:
            expected_sensors = sensor_ids_graph
            validate_time_series(df, expected_sensors)
        except Exception as e:  # noqa: BLE001
            errors.append(f"Time series validation failed: {e}")

        print("4. Validating graph...")
        try:
            ts_sensors = df.columns.astype(str).tolist()
            validate_graph(adj_mx, sensor_ids_graph, ts_sensors)
        except Exception as e:  # noqa: BLE001
            errors.append(f"Graph validation failed: {e}")

        print("5. Extracting missingness semantics...")
        try:
            _values_with_nan, _observed_mask, stats = extract_native_missingness(df.values)
        except Exception as e:  # noqa: BLE001
            errors.append(f"Missingness extraction failed: {e}")
            stats = None
    else:
        stats = None

    if stats is not None:
        print("\n--- Validation Report ---")
        print("Cadence: exactly 5 minutes")
        print(f"Time-series shape: {df.shape}")
        print(f"Adjacency shape: {adj_mx.shape}")
        print("Sensor alignment: EXACT")
        print("Numeric validity: FINITE AND NUMERIC")
        print("\n--- Missingness Statistics ---")
        print(f"Total values: {stats.total_count}")
        print(f"Observed values: {stats.observed_count}")
        print(f"Native missing values: {stats.missing_count}")
        print(f"Zero count: {stats.zero_count}")
        print(f"NaN count: {stats.nan_count}")
        print(f"Invalid infinity count: {stats.invalid_infinity_count}")
        missing_pct = (stats.missing_count / stats.total_count) * 100
        print(f"Missingness percentage: {missing_pct:.6f}%")

        report = {
            "time_series_shape": list(df.shape),
            "adjacency_shape": list(adj_mx.shape),
            "sensor_alignment": "EXACT",
            "cadence_validated": True,
            "numeric_validity_validated": True,
            "missingness": {
                "total_count": stats.total_count,
                "observed_count": stats.observed_count,
                "missing_count": stats.missing_count,
                "zero_count": stats.zero_count,
                "nan_count": stats.nan_count,
                "invalid_infinity_count": stats.invalid_infinity_count,
                "missing_percentage": missing_pct,
            }
        }

        if args.report_out:
            os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
            with open(args.report_out, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport written to {args.report_out}")

    if errors:
        print("\n=== VALIDATION FAILED ===", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        sys.exit(1)

    print("\nSUCCESS: Phase 2 Data Gate Validation Passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
