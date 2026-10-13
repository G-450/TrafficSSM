"""Lock the Phase 12 robustness condition tables and figures.

Reads every condition-run manifest produced by the Phase 12 study
(ST-DSSM, ST-GCN, persistence across the nine 10/20/30% × seed conditions),
verifies the runs are mutually comparable, and writes the locked deliverable:

- ``phase12_results.json``  machine-readable aggregate with per-model, per-condition,
  per-group values and a checksum of every source manifest.
- ``phase12_results.md``    locked tables (mean ± std across seeds, all/masked/unmasked
  groups, point and uncertainty metrics per condition level).

Per ADR-0011 a condition applies at inference, so the primary tables hold only
runs whose inputs were masked at inference. A masked-training ablation run
(``masking.apply_to_training: true``) is a different model and is reported in its
own labelled section, never averaged into a primary cell.

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


def _field(rec: Any, name: str, default: Any = None) -> Any:
    """Read a field from a metric record, whether it is a dict or an object."""
    if isinstance(rec, dict):
        return rec.get(name, default)
    return getattr(rec, name, default)


def _masked_training(manifest: Any) -> bool:
    """Whether this run also withheld the sensors during training (ADR-0011).

    False for every primary condition run, so the locked primary tables are
    unaffected by the field's introduction.
    """
    masking = (getattr(manifest, "resolved_config", None) or {}).get("masking") or {}
    return bool(masking.get("apply_to_training", False))


def load_runs(paths: list[str]) -> list[dict[str, Any]]:
    """Load manifests into structured run records."""
    runs = []
    for path in paths:
        manifest = load_run_manifest(path)
        # Extract per-group aggregate metrics from the metrics list.
        group_metrics: dict[str, dict[str, float]] = {}
        for rec in manifest.metrics:
            if _field(rec, "horizon") != "aggregate":
                continue
            grp = _field(rec, "target_group", "all")
            group_metrics.setdefault(grp, {})[_field(rec, "name", "")] = float(
                _field(rec, "value")
            )

        # Per-horizon metrics for the 'all' group.
        per_horizon: list[dict[str, Any]] = []
        horizon_map: dict[int, dict[str, Any]] = {}
        for rec in manifest.metrics:
            horizon = _field(rec, "horizon")
            if horizon == "aggregate" or _field(rec, "target_group", "all") != "all":
                continue
            h_min = _field(rec, "horizon_minutes")
            step = int(horizon) if isinstance(horizon, (int, float)) else 0
            if step not in horizon_map:
                horizon_map[step] = {
                    "horizon_step": step,
                    "horizon_minutes": int(h_min) if h_min else step * 5,
                }
            horizon_map[step][_field(rec, "name", "")] = float(_field(rec, "value"))
        per_horizon = [horizon_map[s] for s in sorted(horizon_map)]

        runs.append(
            {
                "path": os.path.relpath(path).replace("\\", "/"),
                "run_id": manifest.run_id,
                "model_name": manifest.model_name,
                "training_seed": manifest.training_seed,
                "mask_seed": manifest.mask_seed,
                "mask_condition": manifest.mask_condition,
                "masked_training": _masked_training(manifest),
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

    # No duplicate (model, condition, seed, arm) tuples. The arm matters: a
    # masked-training ablation run shares model, condition and seed with its
    # primary counterpart but is a different model (ADR-0011).
    seen: dict[tuple[str, str, int | None, bool], str] = {}
    for run in runs:
        key = (
            run["model_name"],
            run["mask_condition"],
            run["mask_seed"],
            run["masked_training"],
        )
        if key in seen:
            arm = "masked-training ablation" if key[3] else "primary"
            problems.append(
                f"Duplicate {arm} run for model='{key[0]}' condition='{key[1]}' "
                f"seed={key[2]}: {seen[key]} and {run['run_id']}. "
                "Keep exactly one manifest per combination."
            )
        seen[key] = run["run_id"]

    # Dirty provenance. A locked table has to be traceable to a committed tree,
    # so a run evaluated from uncommitted code is not admissible evidence.
    dirty = sorted(run["run_id"] for run in runs if run["git_dirty"])
    if dirty:
        problems.append(
            "Runs were produced from an uncommitted working tree (git_dirty): "
            + ", ".join(dirty)
            + ". Re-run them from a committed tree."
        )

    # Every (model, arm, condition) present must cover all three canonical seeds.
    # Without this a missing run still renders as "mean +/- std" over whatever
    # seeds survived, which reads as a complete cell.
    coverage: dict[tuple[str, bool, str], set[int | None]] = {}
    for run in runs:
        key = (run["model_name"], run["masked_training"], run["mask_condition"])
        coverage.setdefault(key, set()).add(run["mask_seed"])
    for (model_name, masked_training, level), seeds in sorted(
        coverage.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        missing = sorted(set(CANONICAL_SEEDS) - seeds)
        if missing:
            arm = " (masked-training ablation)" if masked_training else ""
            problems.append(
                f"{model_name}{arm} at {level} is missing seed(s) "
                f"{missing}; a condition cell needs all of {CANONICAL_SEEDS}"
            )

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


def _aggregate_arm(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one arm's runs by (model, condition) across seeds."""
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

    return models_conditions


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the primary condition runs, with any ablation arm beside them.

    Per ADR-0011 a condition applies at inference. A masked-training ablation run
    is therefore a different model and never enters a primary cell; it is
    aggregated into its own ``ablations.masked_training`` block instead.
    """
    primary_runs = [r for r in runs if not r["masked_training"]]
    ablation_runs = [r for r in runs if r["masked_training"]]

    if not primary_runs:
        raise AggregationError(
            "No primary condition runs found: every manifest sets "
            "masking.apply_to_training. The primary tables are the deliverable; "
            "the ablation is reported against them."
        )

    models_conditions = _aggregate_arm(primary_runs)

    reference = primary_runs[0]
    aggregate_out: dict[str, Any] = {
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
                "masked_training": r["masked_training"],
                "path": r["path"],
                "manifest_checksum": r["manifest_checksum"],
                "git_commit": r["git_commit"],
                "git_dirty": r["git_dirty"],
            }
            for r in sorted(
                runs,
                key=lambda r: (
                    r["masked_training"],
                    r["model_name"],
                    r["mask_condition"],
                    str(r["mask_seed"]),
                ),
            )
        ],
    }

    if ablation_runs:
        # Reported beside the primary tables, never inside them. The ablation
        # answers whether training under the mask changes the representation, so
        # it is only meaningful against the primary cell at the same condition.
        aggregate_out["ablations"] = {
            "masked_training": {
                "label": "Masked-training ablation (apply_to_training: true)",
                "note": (
                    "Trained and validated with the same sensors withheld, so these "
                    "runs are a different model from the primary condition runs and "
                    "must not be compared head-to-head with other models (ADR-0011)."
                ),
                "models": _aggregate_arm(ablation_runs),
            }
        }

    return aggregate_out


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
        (
            f"- Split: `{agg['split_evaluated']}` ({agg['sample_count']:,} windows, "
            f"{agg['sensor_count']} sensors, {agg['forecast_horizon']} horizons)"
        ),
        f"- Data artifact: `processed_metadata.json` MD5 `{agg['dataset_manifest_checksum']}`",
        f"- Split tensors digest: `{agg['split_artifact_checksum']}`",
        "",
    ]

    # One table per condition level.
    for level in CONDITION_LEVELS:
        lines += [
            f"## Table — {level} masking condition (all sensors, primary arm)",
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

    # Masked-training ablation, reported against its primary counterpart.
    ablation = (agg.get("ablations") or {}).get("masked_training")
    if ablation:
        lines += [
            f"## Ablation — {ablation['label']}",
            "",
            ablation["note"],
            "",
            "`Primary` repeats the inference-only cell from the tables above so the",
            "two arms can be read side by side. `Delta` is ablation minus primary:",
            "negative means training under the mask helped.",
            "",
            "| Model | Condition | Group | Primary MAE | Ablation MAE | Delta |",
            "| :--- | :--- | :--- | ---: | ---: | ---: |",
        ]
        for model_name, model_data in ablation["models"].items():
            primary_model = agg["models"].get(model_name, {}).get("conditions", {})
            for level in CONDITION_LEVELS:
                cond = model_data["conditions"].get(level)
                if cond is None:
                    continue
                for grp in TARGET_GROUPS:
                    abl = cond["per_group"].get(grp, {}).get("MAE")
                    if abl is None:
                        continue
                    pri = (
                        primary_model.get(level, {})
                        .get("per_group", {})
                        .get(grp, {})
                        .get("MAE")
                    )
                    delta = (
                        f"{abl['mean'] - pri['mean']:+.3f}" if pri is not None else "--"
                    )
                    lines.append(
                        f"| {model_data['label']} | {level} | {grp} | "
                        f"{_fmt(pri)} | {_fmt(abl)} | {delta} |"
                    )
        lines.append("")

    # Provenance.
    lines += [
        "## Provenance",
        "",
        "| Model | Arm | Condition | Seed | Run manifest | Manifest SHA256 | Code commit |",
        "| :--- | :--- | :--- | ---: | :--- | :--- | :--- |",
    ]
    for source in agg["sources"]:
        dirty = " (dirty)" if source["git_dirty"] else ""
        arm = "masked-training" if source.get("masked_training") else "primary"
        lines.append(
            f"| {MODEL_LABELS[source['model_name']]} | {arm} | "
            f"{source['mask_condition']} | "
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
    ablation = (agg.get("ablations") or {}).get("masked_training")
    if ablation:
        print("Masked-training ablation (reported separately, ADR-0011):")
        for model_name, model_data in ablation["models"].items():
            for level in CONDITION_LEVELS:
                cond = model_data["conditions"].get(level)
                if not cond:
                    continue
                mae = _fmt(cond["per_group"].get("all", {}).get("MAE"))
                print(
                    f"  {MODEL_LABELS[model_name]:<24} {level}  all MAE {mae} mph  "
                    f"seeds {cond['seeds']}"
                )
    print(f"Locked tables: {md_path}")
    print(f"Locked data:   {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
