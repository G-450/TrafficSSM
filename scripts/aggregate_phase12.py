"""Lock the Phase 12 robustness condition tables and figures.

Reads every condition-run manifest produced by the Phase 12 study
(ST-DSSM, ST-GCN, persistence across the nine 10/20/30% × seed conditions),
verifies the runs are mutually comparable, and writes the locked deliverable:

- ``phase12_results.json``  machine-readable aggregate with per-model, per-condition,
  per-group values and a checksum of every source manifest.
- ``phase12_results.md``    locked tables (mean ± std across seeds, all/masked/unmasked
  groups, point and uncertainty metrics per condition level).

Nothing here recomputes a metric: every number is read from a manifest, so the
tables are reproducible from the manifests alone. ``--check`` re-derives the
aggregate and fails if it no longer matches the committed JSON.

Usage:
    python scripts/aggregate_phase12.py
    python scripts/aggregate_phase12.py --check
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from st_dssm.result_schema import load_run_manifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_LABELS: dict[str, str] = {
    "historical_persistence": "Historical Persistence",
    "deterministic_st_gcn": "Deterministic ST-GCN",
    "st_dssm": "ST-DSSM (ours)",
}

CONDITION_LEVELS = ("10%", "20%", "30%")
TARGET_GROUPS = ("all", "masked", "unmasked")
POINT_METRICS = ("MAE", "RMSE", "MAPE")
UNCERTAINTY_METRICS = ("NLL", "CRPS", "PICP", "MPIW")
ALL_METRICS = POINT_METRICS + UNCERTAINTY_METRICS

METRIC_UNITS: dict[str, str] = {
    "MAE": "mph",
    "RMSE": "mph",
    "MAPE": "%",
    "NLL": "nats",
    "CRPS": "mph",
    "PICP": "ratio",
    "MPIW": "mph",
}

CANONICAL_SEEDS = [2026, 2027, 2028]
RESULTS_DIR = "experiments/missingness"
CHECK_RTOL = 1e-12


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AggregationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------


def discover_manifests(results_dir: str) -> list[str]:
    """Find every condition run manifest in the results directory."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*_manifest.json")))
    if not paths:
        raise AggregationError(
            f"No run manifests found under {results_dir}. "
            "Run scripts/run_phase12.py first."
        )
    return paths


def load_runs(paths: list[str]) -> list[dict[str, Any]]:
    """Load manifests into structured run records."""
    runs = []
    for path in paths:
        manifest = load_run_manifest(path)
        # Extract per-group aggregate metrics from the metrics list.
        group_metrics: dict[str, dict[str, float]] = {}
        for rec in manifest.metrics:
            if rec.get("horizon") == "aggregate" or rec.get("horizon") == "aggregate":
                # rec may be a dict (from JSON) or MetricRecord-like
                horizon = rec.get("horizon") if isinstance(rec, dict) else getattr(rec, "horizon", None)
                if horizon != "aggregate":
                    continue
                grp = rec.get("target_group") if isinstance(rec, dict) else getattr(rec, "target_group", "all")
                name = rec.get("name") if isinstance(rec, dict) else getattr(rec, "name", "")
                val = rec.get("value") if isinstance(rec, dict) else getattr(rec, "value", None)
                if grp not in group_metrics:
                    group_metrics[grp] = {}
                group_metrics[grp][name] = float(val)

        # Per-horizon metrics for the 'all' group.
        per_horizon: list[dict[str, Any]] = []
        horizon_map: dict[int, dict[str, Any]] = {}
        for rec in manifest.metrics:
            horizon = rec.get("horizon") if isinstance(rec, dict) else getattr(rec, "horizon", None)
            grp = rec.get("target_group") if isinstance(rec, dict) else getattr(rec, "target_group", "all")
            if horizon == "aggregate" or grp != "all":
                continue
            name = rec.get("name") if isinstance(rec, dict) else getattr(rec, "name", "")
            val = rec.get("value") if isinstance(rec, dict) else getattr(rec, "value", None)
            h_min = rec.get("horizon_minutes") if isinstance(rec, dict) else getattr(rec, "horizon_minutes", None)
            step = int(horizon) if isinstance(horizon, (int, float)) else 0
            if step not in horizon_map:
                horizon_map[step] = {"horizon_step": step, "horizon_minutes": int(h_min) if h_min else step * 5}
            horizon_map[step][name] = float(val)
        per_horizon = [horizon_map[s] for s in sorted(horizon_map)]

        runs.append(
            {
                "path": os.path.relpath(path).replace("\\", "/"),
                "run_id": manifest.run_id,
                "model_name": manifest.model_name,
                "training_seed": manifest.training_seed,
                "mask_seed": manifest.mask_seed,
                "mask_condition": manifest.mask_condition,
                "split_evaluated": manifest.split_evaluated,
                "run_status": manifest.run_status,
                "sample_count": manifest.sample_count,
                "sensor_count": manifest.sensor_count,
                "forecast_horizon": manifest.forecast_horizon,
                "git_commit": manifest.git_commit,
                "git_dirty": manifest.git_dirty,
                "dataset_manifest_checksum": manifest.dataset_manifest_checksum,
                "split_artifact_checksum": manifest.split_artifact_checksum,
                "manifest_checksum": manifest.manifest_checksum,
                "package_versions": manifest.package_versions,
                "overall_metrics": manifest.overall_metrics,
                "per_horizon_metrics": per_horizon,
                "group_metrics": group_metrics,
            }
        )
    return runs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_runs(runs: list[dict[str, Any]]) -> None:
    """Verify that the discovered runs are mutually comparable."""
    problems: list[str] = []

    for run in runs:
        if run["run_status"] != "complete":
            problems.append(f"{run['run_id']}: run_status is '{run['run_status']}', expected 'complete'")
        if run["split_evaluated"] != "test":
            problems.append(f"{run['run_id']}: evaluated on '{run['split_evaluated']}', expected 'test'")
        if run["mask_condition"] not in ("10%", "20%", "30%"):
            problems.append(
                f"{run['run_id']}: mask_condition '{run['mask_condition']}' not in 10/20/30%; "
                "use aggregate_phase10.py for the normal condition"
            )
        if run["model_name"] not in MODEL_LABELS:
            problems.append(f"{run['run_id']}: unexpected model_name '{run['model_name']}'")

    for field in ("sample_count", "sensor_count", "forecast_horizon", "split_artifact_checksum"):
        values = {run[field] for run in runs}
        if len(values) > 1:
            problems.append(f"Runs disagree on {field}: {sorted(map(str, values))}")

    # No duplicate (model, condition, seed) tuples.
    seen: dict[tuple[str, str, int | None], str] = {}
    for run in runs:
        key = (run["model_name"], run["mask_condition"], run["mask_seed"])
        if key in seen:
            problems.append(
                f"Duplicate run for model='{key[0]}' condition='{key[1]}' seed={key[2]}: "
                f"{seen[key]} and {run['run_id']}. Keep exactly one manifest per combination."
            )
        seen[key] = run["run_id"]

    if problems:
        raise AggregationError(
            "Phase 12 runs are not comparable:\n  - " + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics by (model, condition) across seeds."""
    models_conditions: dict[str, Any] = {}

    for model_name, model_label in MODEL_LABELS.items():
        model_runs = [r for r in runs if r["model_name"] == model_name]
        if not model_runs:
            continue

        conditions: dict[str, Any] = {}
        for level in CONDITION_LEVELS:
            level_runs = sorted(
                [r for r in model_runs if r["mask_condition"] == level],
                key=lambda r: (r["mask_seed"] is None, r["mask_seed"]),
            )
            if not level_runs:
                continue

            # Per target group, aggregate scalar metrics.
            group_agg: dict[str, dict[str, Any]] = {}
            for grp in TARGET_GROUPS:
                grp_metrics: dict[str, Any] = {}
                for metric in ALL_METRICS:
                    values = [
                        r["group_metrics"].get(grp, {}).get(metric)
                        for r in level_runs
                    ]
                    values = [v for v in values if v is not None]
                    if values:
                        grp_metrics[metric] = _mean_std(values)
                group_agg[grp] = grp_metrics

            # Per-horizon for the 'all' group.
            horizons = sorted(
                {int(h["horizon_step"]) for r in level_runs for h in r["per_horizon_metrics"]}
            )
            per_horizon: list[dict[str, Any]] = []
            for step in horizons:
                rows = [
                    h
                    for r in level_runs
                    for h in r["per_horizon_metrics"]
                    if int(h["horizon_step"]) == step
                ]
                entry: dict[str, Any] = {
                    "horizon_step": step,
                    "horizon_minutes": int(rows[0]["horizon_minutes"]) if rows else step * 5,
                }
                for metric in ALL_METRICS:
                    values = [float(row[metric]) for row in rows if metric in row]
                    if values:
                        entry[metric] = _mean_std(values)
                per_horizon.append(entry)

            conditions[level] = {
                "seeds": [r["mask_seed"] for r in level_runs],
                "per_group": group_agg,
                "per_horizon": per_horizon,
                "per_seed": {
                    str(r["mask_seed"]): {
                        grp: {
                            m: r["group_metrics"].get(grp, {}).get(m)
                            for m in ALL_METRICS
                            if r["group_metrics"].get(grp, {}).get(m) is not None
                        }
                        for grp in TARGET_GROUPS
                    }
                    for r in level_runs
                },
            }

        models_conditions[model_name] = {
            "label": model_label,
            "conditions": conditions,
        }

    reference = runs[0]
    return {
        "study": "Phase 12 robustness condition study",
        "split_evaluated": reference["split_evaluated"],
        "sample_count": reference["sample_count"],
        "sensor_count": reference["sensor_count"],
        "forecast_horizon": reference["forecast_horizon"],
        "dataset_manifest_checksum": reference["dataset_manifest_checksum"],
        "split_artifact_checksum": reference["split_artifact_checksum"],
        "package_versions": reference["package_versions"],
        "models": models_conditions,
        "sources": [
            {
                "run_id": r["run_id"],
                "model_name": r["model_name"],
                "mask_condition": r["mask_condition"],
                "mask_seed": r["mask_seed"],
                "path": r["path"],
                "manifest_checksum": r["manifest_checksum"],
                "git_commit": r["git_commit"],
                "git_dirty": r["git_dirty"],
            }
            for r in sorted(
                runs, key=lambda r: (r["model_name"], r["mask_condition"], str(r["mask_seed"]))
            )
        ],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(stat: dict[str, Any] | None, decimals: int = 3) -> str:
    if stat is None:
        return "--"
    if stat["n"] > 1:
        return f"{stat['mean']:.{decimals}f} +/- {stat['std']:.{decimals}f}"
    return f"{stat['mean']:.{decimals}f}"


def render_markdown(agg: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase 12 — Robustness condition study (locked results)",
        "",
        f"**Generated:** {agg['generated_at']}",
        "",
        "Locked test-set results across 10/20/30% sensor-node missingness conditions.",
        "Every number is read from the run manifests listed under [Provenance](#provenance).",
        "Regenerate with `python scripts/aggregate_phase12.py` and verify with `--check`.",
        "",
        "Values are **mean +/- sample standard deviation across the three canonical seeds**",
        "(2026, 2027, 2028). Point metrics are in physical units (mph).",
        "",
        "## Evaluation scope",
        "",
        f"- Split: `{agg['split_evaluated']}` ({agg['sample_count']:,} windows, "
        f"{agg['sensor_count']} sensors, {agg['forecast_horizon']} horizons)",
        f"- Data artifact: `processed_metadata.json` MD5 `{agg['dataset_manifest_checksum']}`",
        f"- Split tensors digest: `{agg['split_artifact_checksum']}`",
        "",
    ]

    # One table per condition level.
    for level in CONDITION_LEVELS:
        lines += [
            f"## Table — {level} masking condition (all sensors)",
            "",
            "| Model | MAE (mph) | RMSE (mph) | MAPE (%) | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for model_data in agg["models"].values():
            cond = model_data["conditions"].get(level)
            if cond is None:
                continue
            o = cond["per_group"].get("all", {})
            lines.append(
                f"| {model_data['label']} | {_fmt(o.get('MAE'))} | {_fmt(o.get('RMSE'))} | "
                f"{_fmt(o.get('MAPE'))} | {_fmt(o.get('NLL'))} | {_fmt(o.get('CRPS'))} | "
                f"{_fmt(o.get('PICP'), 4)} | {_fmt(o.get('MPIW'))} |"
            )
        lines.append("")

    # Masked vs unmasked degradation tables.
    lines += [
        "## Masked vs unmasked degradation (MAE, mph)",
        "",
        "| Model | Condition | All | Masked | Unmasked |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for model_data in agg["models"].values():
        for level in CONDITION_LEVELS:
            cond = model_data["conditions"].get(level)
            if cond is None:
                continue
            grps = cond["per_group"]
            all_mae = _fmt(grps.get("all", {}).get("MAE"))
            masked_mae = _fmt(grps.get("masked", {}).get("MAE"))
            unmasked_mae = _fmt(grps.get("unmasked", {}).get("MAE"))
            lines.append(
                f"| {model_data['label']} | {level} | {all_mae} | {masked_mae} | {unmasked_mae} |"
            )
    lines.append("")

    # Per-seed detail.
    lines += [
        "## Per-seed values",
        "",
        "| Model | Condition | Seed | All MAE | Masked MAE | Unmasked MAE |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for model_data in agg["models"].values():
        for level in CONDITION_LEVELS:
            cond = model_data["conditions"].get(level)
            if cond is None:
                continue
            for seed_str, seed_grps in sorted(cond["per_seed"].items()):
                all_mae = f"{seed_grps.get('all', {}).get('MAE', float('nan')):.3f}"
                msk_mae = f"{seed_grps.get('masked', {}).get('MAE', float('nan')):.3f}"
                unm_mae = f"{seed_grps.get('unmasked', {}).get('MAE', float('nan')):.3f}"
                lines.append(
                    f"| {model_data['label']} | {level} | {seed_str} | "
                    f"{all_mae} | {msk_mae} | {unm_mae} |"
                )
    lines.append("")

    # Provenance.
    lines += [
        "## Provenance",
        "",
        "| Model | Condition | Seed | Run manifest | Manifest SHA256 | Code commit |",
        "| :--- | :--- | ---: | :--- | :--- | :--- |",
    ]
    for source in agg["sources"]:
        dirty = " (dirty)" if source["git_dirty"] else ""
        lines.append(
            f"| {MODEL_LABELS[source['model_name']]} | {source['mask_condition']} | "
            f"{source['mask_seed']} | `{source['path']}` | "
            f"`{source['manifest_checksum'][:16]}...` | "
            f"`{source['git_commit'][:12]}`{dirty} |"
        )

    versions = ", ".join(f"{k} {v}" for k, v in sorted(agg["package_versions"].items()))
    lines += ["", f"**Environment:** {versions}", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check mode
# ---------------------------------------------------------------------------


def _compare(new: Any, old: Any, path: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(new, dict) and isinstance(old, dict):
        for key in sorted(set(new) | set(old)):
            if key not in new:
                diffs.append(f"{path}.{key}: missing from re-derived aggregate")
            elif key not in old:
                diffs.append(f"{path}.{key}: not present in committed aggregate")
            else:
                diffs += _compare(new[key], old[key], f"{path}.{key}")
    elif isinstance(new, list) and isinstance(old, list):
        if len(new) != len(old):
            diffs.append(f"{path}: length {len(new)} != {len(old)}")
        else:
            for i, (a, b) in enumerate(zip(new, old)):
                diffs += _compare(a, b, f"{path}[{i}]")
    elif isinstance(new, float) or isinstance(old, float):
        scale = max(abs(float(new)), abs(float(old)), 1.0)
        if abs(float(new) - float(old)) > CHECK_RTOL * scale:
            diffs.append(f"{path}: {new} != {old}")
    elif new != old:
        diffs.append(f"{path}: {new!r} != {old!r}")
    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate and lock Phase 12 results.")
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"Directory holding the condition manifests. [default: {RESULTS_DIR}]",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed aggregate still matches the manifests; write nothing.",
    )
    args = parser.parse_args(argv)

    try:
        runs = load_runs(discover_manifests(args.results_dir))
        validate_runs(runs)
    except AggregationError as exc:
        print(f"Aggregation error: {exc}", file=sys.stderr)
        return 1

    agg = aggregate(runs)
    json_path = os.path.join(args.results_dir, "phase12_results.json")
    md_path = os.path.join(args.results_dir, "phase12_results.md")

    if args.check:
        if not os.path.exists(json_path):
            print(f"Nothing to check: {json_path} does not exist.", file=sys.stderr)
            return 1
        with open(json_path, encoding="utf-8") as f:
            committed = json.load(f)
        committed.pop("generated_at", None)
        diffs = _compare(agg, committed)
        if diffs:
            print(
                f"Phase 12 aggregate no longer matches {json_path}:\n  - " + "\n  - ".join(diffs),
                file=sys.stderr,
            )
            return 1
        print(
            f"Phase 12 results verified: {len(agg['sources'])} manifests reproduce "
            f"{os.path.relpath(json_path)} exactly."
        )
        return 0

    agg["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(agg))

    print(f"Aggregated {len(agg['sources'])} condition run manifests:")
    for model_name, model_data in agg["models"].items():
        for level in CONDITION_LEVELS:
            cond = model_data["conditions"].get(level)
            if not cond:
                continue
            mae = _fmt(cond["per_group"].get("all", {}).get("MAE"))
            print(f"  {MODEL_LABELS[model_name]:<24} {level}  all MAE {mae} mph  seeds {cond['seeds']}")
    print(f"Locked tables: {md_path}")
    print(f"Locked data:   {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
