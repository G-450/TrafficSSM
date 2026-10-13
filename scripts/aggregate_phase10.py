"""Lock the Phase 10 normal-condition tables and figures.

Reads every run manifest produced by the Phase 10 study (ST-DSSM across the
canonical seeds, plus the Phase 5 deterministic baselines at 0% masking),
verifies the runs are mutually comparable, and writes the locked deliverable:

- ``phase10_results.json``  machine-readable aggregate with per-seed values and
  the checksum of every source manifest.
- ``phase10_results.md``    the locked tables, mean +/- standard deviation across seeds.
- ``plots/phase10_*.png``   model comparison figures across forecast horizons.

Nothing here recomputes a metric: every number is read from a manifest, so the
tables are reproducible from the manifests alone. ``--check`` re-derives the
aggregate and fails if it no longer matches the committed JSON, which is the
Phase 10 gate ("tables and figures are reproducible from manifests").

Usage:
    python scripts/aggregate_phase10.py
    python scripts/aggregate_phase10.py --check
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

from st_dssm.plots import plot_horizon_curves
from st_dssm.result_schema import load_run_manifest

#: Display order and labels for the models compared in the normal-condition study.
MODEL_LABELS = {
    "historical_persistence": "Historical Persistence",
    "deterministic_st_gcn": "Deterministic ST-GCN",
    "st_dssm": "ST-DSSM (ours)",
}

POINT_METRICS = ("MAE", "RMSE", "MAPE")
UNCERTAINTY_METRICS = ("NLL", "CRPS", "PICP", "MPIW")
ALL_METRICS = POINT_METRICS + UNCERTAINTY_METRICS

METRIC_UNITS = {
    "MAE": "mph",
    "RMSE": "mph",
    "MAPE": "%",
    "NLL": "nats",
    "CRPS": "mph",
    "PICP": "ratio",
    "MPIW": "mph",
}

#: Horizons highlighted in the headline table (minutes).
HEADLINE_HORIZONS = (15, 30, 60)

#: Nominal coverage of the prediction intervals used for PICP/MPIW.
NOMINAL_COVERAGE = 0.95

#: Tolerance for --check. Re-deriving the aggregate from unchanged manifests is
#: plain arithmetic over the same stored doubles, so only true drift exceeds this.
CHECK_RTOL = 1e-12


class AggregationError(Exception):
    """Raised when the discovered runs are missing, duplicated or not comparable."""


def discover_manifests(results_dir: str) -> list[str]:
    """Find every run manifest belonging to the Phase 10 study."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*_manifest.json"))) + sorted(
        glob.glob(os.path.join(results_dir, "baselines", "*_manifest.json"))
    )
    if not paths:
        raise AggregationError(
            f"No run manifests found under {results_dir}. Run scripts/run_phase10.py first."
        )
    return paths


def load_runs(paths: list[str]) -> list[dict[str, Any]]:
    """Load manifests into comparable run records, keyed by model and seed."""
    runs = []
    for path in paths:
        manifest = load_run_manifest(path)
        runs.append(
            {
                "path": os.path.relpath(path).replace("\\", "/"),
                "run_id": manifest.run_id,
                "model_name": manifest.model_name,
                "training_seed": manifest.training_seed,
                "split_evaluated": manifest.split_evaluated,
                "mask_condition": manifest.mask_condition,
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
                "trainable_parameters": manifest.trainable_parameters,
                "best_epoch": manifest.best_epoch,
                "total_epochs": manifest.total_epochs,
                "selection_metric": manifest.selection_metric,
                "checkpoint_path": manifest.checkpoint_path,
                "overall_metrics": manifest.overall_metrics,
                "per_horizon_metrics": manifest.per_horizon_metrics,
            }
        )
    return runs


def validate_runs(runs: list[dict[str, Any]], expected_seeds: list[int]) -> None:
    """Refuse to aggregate runs that are not mutually comparable.

    The normal-condition study compares models on one split, one data artifact and
    one masking condition. Anything else must stop the aggregation rather than be
    averaged into a table that looks authoritative.
    """
    problems = []

    for run in runs:
        if run["run_status"] != "complete":
            problems.append(f"{run['run_id']}: run_status is '{run['run_status']}', expected 'complete'")
        if run["split_evaluated"] != "test":
            problems.append(f"{run['run_id']}: evaluated on '{run['split_evaluated']}', expected 'test'")
        if run["mask_condition"] != "0%":
            problems.append(
                f"{run['run_id']}: mask_condition '{run['mask_condition']}', expected '0%' "
                "(the normal condition)"
            )
        if run["model_name"] not in MODEL_LABELS:
            problems.append(f"{run['run_id']}: unexpected model_name '{run['model_name']}'")

    for field in ("sample_count", "sensor_count", "forecast_horizon", "split_artifact_checksum"):
        values = {run[field] for run in runs}
        if len(values) > 1:
            problems.append(f"Runs disagree on {field}: {sorted(map(str, values))}")

    # One manifest per (model, seed): duplicates would silently bias the mean.
    seen: dict[tuple[str, int | None], str] = {}
    for run in runs:
        key = (run["model_name"], run["training_seed"])
        if key in seen:
            problems.append(
                f"Duplicate run for model '{key[0]}' seed {key[1]}: {seen[key]} and {run['run_id']}. "
                "Keep exactly one manifest per model and seed."
            )
        seen[key] = run["run_id"]

    # Trained models must cover exactly the locked seeds.
    for model in ("deterministic_st_gcn", "st_dssm"):
        seeds = sorted(r["training_seed"] for r in runs if r["model_name"] == model)
        if seeds != sorted(expected_seeds):
            problems.append(
                f"Model '{model}' has seeds {seeds}, expected {sorted(expected_seeds)}"
            )

    if problems:
        raise AggregationError(
            "Phase 10 runs are not comparable:\n  - " + "\n  - ".join(problems)
        )


def _mean_std(values: list[float]) -> dict[str, Any]:
    """Mean with sample standard deviation; std is 0.0 for a single run."""
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-model overall and per-horizon metrics across seeds."""
    models: dict[str, Any] = {}

    for model_name, model_label in MODEL_LABELS.items():
        model_runs = sorted(
            (r for r in runs if r["model_name"] == model_name),
            key=lambda r: (r["training_seed"] is None, r["training_seed"]),
        )
        if not model_runs:
            continue

        overall: dict[str, Any] = {}
        for metric in ALL_METRICS:
            values = [
                float(r["overall_metrics"][metric])
                for r in model_runs
                if metric in r["overall_metrics"]
            ]
            if values:
                overall[metric] = _mean_std(values)

        horizons = sorted({int(h["horizon_step"]) for r in model_runs for h in r["per_horizon_metrics"]})
        per_horizon = []
        for step in horizons:
            rows = [
                h
                for r in model_runs
                for h in r["per_horizon_metrics"]
                if int(h["horizon_step"]) == step
            ]
            entry: dict[str, Any] = {
                "horizon_step": step,
                "horizon_minutes": int(rows[0]["horizon_minutes"]),
            }
            for metric in ALL_METRICS:
                values = [float(row[metric]) for row in rows if metric in row]
                if values:
                    entry[metric] = _mean_std(values)
            per_horizon.append(entry)

        models[model_name] = {
            "label": model_label,
            "seeds": [r["training_seed"] for r in model_runs],
            "trainable_parameters": model_runs[0]["trainable_parameters"],
            "selection_metric": model_runs[0]["selection_metric"],
            "best_epochs": [r["best_epoch"] for r in model_runs],
            "total_epochs": [r["total_epochs"] for r in model_runs],
            "overall": overall,
            "per_horizon": per_horizon,
            "per_seed_overall": {
                str(r["training_seed"]): {
                    m: float(r["overall_metrics"][m])
                    for m in ALL_METRICS
                    if m in r["overall_metrics"]
                }
                for r in model_runs
            },
        }

    reference = runs[0]
    return {
        "study": "Phase 10 normal-condition study",
        "split_evaluated": reference["split_evaluated"],
        "mask_condition": reference["mask_condition"],
        "nominal_coverage": NOMINAL_COVERAGE,
        "sample_count": reference["sample_count"],
        "sensor_count": reference["sensor_count"],
        "forecast_horizon": reference["forecast_horizon"],
        "valid_target_count": int(reference["overall_metrics"].get("valid_count", 0)),
        "dataset_manifest_checksum": reference["dataset_manifest_checksum"],
        "split_artifact_checksum": reference["split_artifact_checksum"],
        "package_versions": reference["package_versions"],
        "models": models,
        "sources": [
            {
                "run_id": r["run_id"],
                "model_name": r["model_name"],
                "training_seed": r["training_seed"],
                "path": r["path"],
                "manifest_checksum": r["manifest_checksum"],
                "git_commit": r["git_commit"],
                "git_dirty": r["git_dirty"],
                "checkpoint_path": r["checkpoint_path"].replace("\\", "/"),
            }
            for r in sorted(runs, key=lambda r: (r["model_name"], str(r["training_seed"])))
        ],
    }


def _fmt(stat: dict[str, Any] | None, decimals: int = 3, separator: str = " ± ") -> str:
    """Format a mean +/- std cell; a single run shows the mean alone.

    ``separator`` is ASCII-only for console output: the Windows console encodes
    cp1252 and cannot print "±".
    """
    if stat is None:
        return "--"
    if stat["n"] > 1:
        return f"{stat['mean']:.{decimals}f}{separator}{stat['std']:.{decimals}f}"
    return f"{stat['mean']:.{decimals}f}"


def _horizon_entry(model: dict[str, Any], minutes: int) -> dict[str, Any] | None:
    for entry in model["per_horizon"]:
        if entry["horizon_minutes"] == minutes:
            return entry
    return None


def render_markdown(agg: dict[str, Any], figures: list[str]) -> str:
    """Render the locked Phase 10 tables."""
    models = agg["models"]
    lines: list[str] = []

    lines += [
        "# Phase 10 — Normal-condition study (locked results)",
        "",
        f"**Generated:** {agg['generated_at']}",
        "",
        "Locked test-set results at 0% masking. Every number is read from the run",
        "manifests listed under [Provenance](#provenance); regenerate this file with",
        "`python scripts/aggregate_phase10.py` and verify it with `--check`.",
        "",
        "Values are **mean ± sample standard deviation across the three canonical seeds**",
        "(2026, 2027, 2028). Historical Persistence is deterministic and has no seed, so it",
        "shows a single value. Point metrics are in physical units (mph); PICP and MPIW are",
        f"for the {agg['nominal_coverage']:.0%} nominal prediction interval.",
        "",
        "## Evaluation scope",
        "",
        (
            f"- Split: `{agg['split_evaluated']}` ({agg['sample_count']:,} windows, "
            f"{agg['sensor_count']} sensors, {agg['forecast_horizon']} horizons)"
        ),
        f"- Observed targets scored: {agg['valid_target_count']:,}",
        f"- Masking condition: {agg['mask_condition']} (normal condition)",
        f"- Data artifact: `processed_metadata.json` MD5 `{agg['dataset_manifest_checksum']}`",
        f"- Split tensors digest: `{agg['split_artifact_checksum']}`",
        "",
        "## Table 1 — Overall test metrics",
        "",
        "| Model | Params | MAE (mph) | RMSE (mph) | MAPE (%) | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, model in models.items():
        o = model["overall"]
        params = f"{model['trainable_parameters']:,}" if model["trainable_parameters"] else "--"
        lines.append(
            f"| {model['label']} | {params} | {_fmt(o.get('MAE'))} | {_fmt(o.get('RMSE'))} | "
            f"{_fmt(o.get('MAPE'))} | {_fmt(o.get('NLL'))} | {_fmt(o.get('CRPS'))} | "
            f"{_fmt(o.get('PICP'), 4)} | {_fmt(o.get('MPIW'))} |"
        )
        del name

    lines += [
        "",
        "## Table 2 — Point accuracy by horizon",
        "",
        "| Model | " + " | ".join(f"{m} min MAE / RMSE" for m in HEADLINE_HORIZONS) + " |",
        "| :--- | " + " | ".join(["---:"] * len(HEADLINE_HORIZONS)) + " |",
    ]
    for model in models.values():
        cells = []
        for minutes in HEADLINE_HORIZONS:
            entry = _horizon_entry(model, minutes)
            if entry is None:
                cells.append("--")
            else:
                cells.append(f"{_fmt(entry.get('MAE'), 2)} / {_fmt(entry.get('RMSE'), 2)}")
        lines.append(f"| {model['label']} | " + " | ".join(cells) + " |")

    probabilistic = {
        name: model
        for name, model in models.items()
        if "NLL" in model["overall"]
    }
    if probabilistic:
        lines += [
            "",
            "## Table 3 — Uncertainty quality by horizon",
            "",
            f"Nominal interval coverage is {agg['nominal_coverage']:.0%}; PICP closer to that",
            "value is better calibrated, and MPIW is the mean interval width.",
            "",
        ]
        for model in probabilistic.values():
            lines += [
                f"**{model['label']}**",
                "",
                "| Horizon | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
            for entry in model["per_horizon"]:
                lines.append(
                    f"| {entry['horizon_minutes']} min | {_fmt(entry.get('NLL'))} | "
                    f"{_fmt(entry.get('CRPS'))} | {_fmt(entry.get('PICP'), 4)} | "
                    f"{_fmt(entry.get('MPIW'), 2)} |"
                )
            lines.append("")

    lines += ["## Figures", ""]
    for figure in figures:
        lines.append(f"- `{figure}`")

    lines += [
        "",
        "## Per-seed values",
        "",
        "| Model | Seed | Best epoch | Epochs run | " + " | ".join(ALL_METRICS) + " |",
        "| :--- | ---: | ---: | ---: | " + " | ".join(["---:"] * len(ALL_METRICS)) + " |",
    ]
    for model in models.values():
        for seed, best, total in zip(model["seeds"], model["best_epochs"], model["total_epochs"]):
            values = model["per_seed_overall"][str(seed)]
            cells = [
                f"{values[m]:.4f}" if m in values else "--" for m in ALL_METRICS
            ]
            seed_label = str(seed) if seed is not None else "--"
            epoch_label = str(best) if total else "--"
            total_label = str(total) if total else "--"
            lines.append(
                f"| {model['label']} | {seed_label} | {epoch_label} | {total_label} | "
                + " | ".join(cells)
                + " |"
            )

    lines += [
        "",
        "## Provenance",
        "",
        "| Model | Seed | Run manifest | Manifest SHA256 | Code commit |",
        "| :--- | ---: | :--- | :--- | :--- |",
    ]
    for source in agg["sources"]:
        seed_label = str(source["training_seed"]) if source["training_seed"] is not None else "--"
        dirty = " (dirty)" if source["git_dirty"] else ""
        lines.append(
            f"| {MODEL_LABELS[source['model_name']]} | {seed_label} | `{source['path']}` | "
            f"`{source['manifest_checksum'][:16]}…` | `{source['git_commit'][:12]}`{dirty} |"
        )

    versions = ", ".join(f"{k} {v}" for k, v in sorted(agg["package_versions"].items()))
    lines += ["", f"**Environment:** {versions}", ""]

    return "\n".join(lines)


def render_figures(agg: dict[str, Any], output_dir: str) -> list[str]:
    """Write the model-comparison figures and return their relative paths."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    models = agg["models"]
    minutes = [e["horizon_minutes"] for e in next(iter(models.values()))["per_horizon"]]
    written = []

    def series_for(metric: str) -> dict[str, tuple[list[float], list[float]]]:
        out = {}
        for model in models.values():
            entries = [e for e in model["per_horizon"] if metric in e]
            if len(entries) == len(minutes):
                out[model["label"]] = (
                    [e[metric]["mean"] for e in entries],
                    [e[metric]["std"] for e in entries],
                )
        return out

    for metric, filename, reference in (
        ("MAE", "phase10_mae_by_horizon.png", None),
        ("RMSE", "phase10_rmse_by_horizon.png", None),
        ("PICP", "phase10_picp_by_horizon.png", NOMINAL_COVERAGE),
        ("MPIW", "phase10_mpiw_by_horizon.png", None),
        ("CRPS", "phase10_crps_by_horizon.png", None),
    ):
        series = series_for(metric)
        if not series:
            continue
        path = os.path.join(plots_dir, filename)
        plot_horizon_curves(
            minutes,
            series,
            metric,
            path,
            unit=METRIC_UNITS[metric],
            title=f"Phase 10 normal condition: {metric} by horizon (test split)",
            reference_level=reference,
            reference_label=f"nominal {NOMINAL_COVERAGE:.0%}" if reference else None,
        )
        written.append(os.path.relpath(path, output_dir).replace("\\", "/"))

    return written


def _compare(new: Any, old: Any, path: str = "") -> list[str]:
    """Recursively diff two aggregate structures, tolerating float noise."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate and lock Phase 10 results.")
    parser.add_argument(
        "--results-dir",
        default="experiments/normal",
        help="Directory holding the Phase 10 run manifests. [default: experiments/normal]",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028],
        help="Seeds every trained model must cover. [default: 2026 2027 2028]",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed aggregate still matches the manifests; write nothing.",
    )
    parser.add_argument(
        "--no-figures",
        dest="figures",
        action="store_false",
        help="Skip figure rendering.",
    )
    args = parser.parse_args(argv)

    try:
        runs = load_runs(discover_manifests(args.results_dir))
        validate_runs(runs, args.seeds)
    except AggregationError as exc:
        print(f"Aggregation error: {exc}", file=sys.stderr)
        return 1

    agg = aggregate(runs)
    json_path = os.path.join(args.results_dir, "phase10_results.json")
    md_path = os.path.join(args.results_dir, "phase10_results.md")

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
                f"Phase 10 aggregate no longer matches {json_path}:\n  - " + "\n  - ".join(diffs),
                file=sys.stderr,
            )
            return 1
        print(
            f"Phase 10 results verified: {len(agg['sources'])} manifests reproduce "
            f"{os.path.relpath(json_path)} exactly."
        )
        return 0

    figures = render_figures(agg, args.results_dir) if args.figures else []
    agg["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(agg, figures))

    print(f"Aggregated {len(agg['sources'])} run manifests from {args.results_dir}:")
    for name, model in agg["models"].items():
        mae = model["overall"].get("MAE")
        print(f"  {MODEL_LABELS[name]:<24} MAE {_fmt(mae, separator=' +/- ')} mph over seeds {model['seeds']}")
    print(f"Locked tables:  {md_path}")
    print(f"Locked data:    {json_path}")
    for figure in figures:
        print(f"Figure:         {os.path.join(args.results_dir, figure)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
