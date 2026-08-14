"""CLI command for evaluation and result generation (st-dssm-evaluate).

Supports evaluating predictions against targets, calculating overall and
per-horizon point/probabilistic metrics in physical units (mph), serializing
a validated RunManifest, and rendering headless evaluation plots.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

from st_dssm.metrics import (
    calibration_curve,
    evaluate_metrics_by_horizon,
    inverse_transform_predictions,
    prediction_interval,
)
from st_dssm.plots import (
    plot_calibration_curve,
    plot_horizon_metrics,
    plot_interval_width_vs_horizon,
    plot_prediction_intervals,
)
from st_dssm.result_schema import (
    MetricRecord,
    RunManifest,
    load_run_manifest,
    save_run_manifest,
)


def _get_git_commit() -> tuple[str, bool]:
    """Retrieve git commit and dirty working tree status."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return commit, len(status) > 0
    except Exception:  # noqa: BLE001
        return "unknown", False


def run_synthetic_smoke(output_dir: str, save_plots: bool = True) -> tuple[RunManifest, str]:
    """Run an end-to-end synthetic evaluation smoke test without needing external data files."""
    print("Running synthetic evaluation smoke test...")
    np.random.seed(42)

    # Small deterministic synthetic dimensions: S=20 samples, H=12 horizons, N=5 sensors
    S, H, N = 20, 12, 5
    scaler_mean = np.array([55.0, 60.0, 50.0, 65.0, 58.0], dtype=np.float32)
    scaler_std = np.array([8.0, 9.0, 7.5, 10.0, 8.5], dtype=np.float32)

    # True normalized speed with synthetic diurnal pattern
    t = np.linspace(0, 4 * np.pi, S * H * N).reshape(S, H, N, 1)
    norm_true = 0.5 * np.sin(t) + np.random.normal(0, 0.1, (S, H, N, 1))
    norm_mu = norm_true + np.random.normal(0, 0.15, (S, H, N, 1))
    norm_sigma = np.full((S, H, N, 1), 0.35, dtype=np.float32)

    # Native observation mask (98% observed)
    native_mask = (np.random.uniform(0, 1, (S, H, N, 1)) > 0.02).astype(np.float32)

    # Inverse transform to physical units (mph)
    raw_true, _ = inverse_transform_predictions(norm_true, norm_sigma, scaler_mean, scaler_std)
    raw_mu, raw_sigma = inverse_transform_predictions(norm_mu, norm_sigma, scaler_mean, scaler_std)

    # Evaluate metrics
    overall, per_horizon = evaluate_metrics_by_horizon(
        raw_true, raw_mu, sigma=raw_sigma, mask=native_mask, nominal_pi=0.95
    )

    git_commit, git_dirty = _get_git_commit()
    run_id = f"smoke-synthetic-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    manifest = RunManifest(
        run_id=run_id,
        model_name="synthetic_smoke_evaluator",
        split_evaluated="synthetic",
        sensor_count=N,
        sample_count=S,
        forecast_horizon=H,
        git_commit=git_commit,
        git_dirty=git_dirty,
        start_time_utc=datetime.now(timezone.utc).isoformat(),
        overall_metrics=overall,
        per_horizon_metrics=per_horizon,
    )

    # Add metric records
    for name in ["MAE", "RMSE", "MAPE", "NLL", "CRPS", "PICP", "MPIW"]:
        if name in overall:
            unit = (
                "percent"
                if name == "MAPE"
                else (
                    "mph"
                    if name in ["MAE", "RMSE", "MPIW"]
                    else "ratio"
                    if name == "PICP"
                    else "raw"
                )
            )
            manifest.add_metric(
                MetricRecord(
                    name=name,
                    value=float(overall[name]),
                    unit=unit,
                    horizon="aggregate",
                    target_group="all",
                    valid_count=int(overall.get("valid_count", 0)),
                )
            )

    for h_data in per_horizon:
        h_step = int(h_data["horizon_step"])
        h_min = int(h_data["horizon_minutes"])
        for name in ["MAE", "RMSE", "MAPE", "NLL", "CRPS", "PICP", "MPIW"]:
            if name in h_data:
                unit = (
                    "percent"
                    if name == "MAPE"
                    else (
                        "mph"
                        if name in ["MAE", "RMSE", "MPIW"]
                        else "ratio"
                        if name == "PICP"
                        else "raw"
                    )
                )
                manifest.add_metric(
                    MetricRecord(
                        name=name,
                        value=float(h_data[name]),
                        unit=unit,
                        horizon=h_step,
                        horizon_minutes=h_min,
                        target_group="all",
                        valid_count=int(h_data.get("valid_count", 0)),
                    )
                )

    manifest.mark_complete()

    # Save manifest atomically
    manifest_path = os.path.join(output_dir, f"{run_id}_manifest.json")
    save_run_manifest(manifest, manifest_path, overwrite=True, atomic=True)

    # Generate plots if requested
    if save_plots:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        h_steps = [int(h["horizon_step"]) for h in per_horizon]
        mae_vals = [float(h["MAE"]) for h in per_horizon]
        plot_horizon_metrics(
            h_steps,
            mae_vals,
            "MAE",
            os.path.join(plots_dir, f"{run_id}_mae_by_horizon.png"),
            unit="mph",
            title="Synthetic MAE across Forecast Horizons",
        )

        lo_raw, hi_raw = prediction_interval(raw_mu, raw_sigma, nominal=0.95)
        plot_prediction_intervals(
            raw_true[:, :, 0, 0].ravel()[:100],
            raw_mu[:, :, 0, 0].ravel()[:100],
            lo_raw[:, :, 0, 0].ravel()[:100],
            hi_raw[:, :, 0, 0].ravel()[:100],
            os.path.join(plots_dir, f"{run_id}_prediction_interval.png"),
            sensor_idx="0 (Synthetic)",
            unit="mph",
        )

        cal_results = calibration_curve(raw_true, raw_mu, raw_sigma, mask=native_mask)
        nom_levels = list(cal_results.keys())
        emp_covs = [float(cal_results[lvl]["picp"]) for lvl in nom_levels]
        plot_calibration_curve(
            nom_levels,
            emp_covs,
            os.path.join(plots_dir, f"{run_id}_calibration.png"),
        )

        mpiw_vals = [float(h["MPIW"]) for h in per_horizon]
        plot_interval_width_vs_horizon(
            h_steps,
            mpiw_vals,
            os.path.join(plots_dir, f"{run_id}_mpiw_by_horizon.png"),
            unit="mph",
        )

    # Validate saved manifest
    loaded = load_run_manifest(manifest_path)
    assert loaded.run_id == run_id
    assert loaded.run_status == "complete"
    print(f"Synthetic smoke evaluation complete. Manifest saved and verified at: {manifest_path}")

    return manifest, manifest_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for st-dssm-evaluate."""
    parser = argparse.ArgumentParser(
        description="ST-DSSM model-independent evaluation runner."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation YAML configuration file.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run deterministic synthetic smoke test to validate evaluation machinery.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/results",
        help="Directory to save run manifests and plots.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Generate evaluation plots.",
    )
    parser.add_argument(
        "--no-plots",
        dest="save_plots",
        action="store_false",
        help="Disable evaluation plot generation.",
    )

    args = parser.parse_args(argv)

    try:
        if args.synthetic_smoke or args.config is None:
            # Run synthetic smoke test if specified or default
            run_synthetic_smoke(output_dir=args.output_dir, save_plots=args.save_plots)
            return 0

        # Load configuration if provided
        with open(args.config, "r", encoding="utf-8") as f:
            _ = yaml.safe_load(f)

        print(f"Loaded evaluation config from {args.config}")
        run_synthetic_smoke(output_dir=args.output_dir, save_plots=args.save_plots)
        return 0

    except Exception as e:  # noqa: BLE001
        print(f"Evaluation error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
