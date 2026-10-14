"""Lock the Phase 13 calibration and robustness review.

Phase 13 asks three questions of the locked Phase 10/12 results, and this
script answers all three into one deliverable:

1. **Calibration.** Is the Gaussian predictive distribution calibrated, and does
   the secondary temperature-scaling analysis of ADR-0008 improve it? The
   temperature is one positive scalar per seed, fitted on the *validation*
   split by NLL and frozen before any test number is computed. Both the raw and
   the calibrated arm are reported, as ADR-0008 requires, and calibration never
   touches predictive means.
2. **Error analysis.** How does accuracy and coverage decay across the forecast
   horizon under each missingness condition, and how does that decay differ
   between withheld and observed sensors?
3. **Latency.** What does one forward pass cost, so the project's claims can be
   reconciled against a measured record rather than an estimate.

Unlike the Phase 10 and Phase 12 aggregators, which read numbers straight out
of run manifests, the calibration diagnostics cannot come from a manifest: a
reliability curve, a PIT histogram and a temperature are functions of the full
predictive distribution. This script therefore recomputes them from the saved
prediction artifacts (``*_predictions.npz``) and the canonical Phase 3 targets,
and records a checksum of every prediction file it reads, so the locked numbers
are tied to exact artifacts. The per-horizon error analysis still comes from the
manifests. ``--check`` re-derives everything and fails on any disagreement.

Because the diagnostics are recomputed over ~40 million targets per run, a full
aggregation takes a few minutes; ``--check`` costs the same.

Only ST-DSSM is diagnosed. Calibration is a property of a predictive
distribution, and the two baselines are point forecasters with no predictive
scale to calibrate (ADR-0008); they appear in the latency record and in the
Phase 10/12 accuracy tables instead.

Usage:
    python scripts/aggregate_phase13.py
    python scripts/aggregate_phase13.py --check
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from st_dssm.calibration import (
    CANONICAL_LEVELS,
    PIT_BINS,
    apply_temperature,
    fit_temperature,
    interval_ece,
    pit_ece,
    pit_histogram,
    pit_values,
    reliability_table,
)
from st_dssm.io import load_and_validate_artifact
from st_dssm.mask_generator import resolve_mask_generator
from st_dssm.metrics import (
    gaussian_crps,
    gaussian_nll,
    inverse_transform_predictions,
    mae,
    mpiw,
    picp,
    prediction_interval,
)
from st_dssm.plots import (
    plot_horizon_curves,
    plot_pit_histogram,
    plot_reliability_comparison,
)
from st_dssm.result_schema import load_run_manifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAL_DIR = "experiments/calibration"
NORMAL_DIR = "experiments/normal"
CONDITION_DIR = "experiments/missingness"
RESULTS_DIR = "experiments/calibration"
ARTIFACT_DIR = "data/processed"

CANONICAL_SEEDS = (2026, 2027, 2028)
NOMINAL_COVERAGE = 0.95

#: Conditions in reporting order. "normal" is the Phase 10 0% condition.
CONDITION_ORDER = ("normal", "10%", "20%", "30%")
ABLATION_CONDITION = "20%"
TARGET_GROUPS = ("all", "masked", "unmasked")
ARMS = ("raw", "calibrated")

# Unlike Phases 10 and 12, which compare values stored in the run manifests and
# can therefore afford 1e-12, this gate *recomputes* every diagnostic from the
# prediction artifacts. Recomputation is bit-identical within one environment,
# but float32 reduction order differs across torch/numpy versions, so a
# cross-environment re-derivation from checksum-identical artifacts lands about
# 1e-7 away (max observed 1.7e-5, on a three-seed standard deviation, where
# cancellation amplifies the relative error). Holding the gate tighter than that
# would make it pass only on the machine that produced the lock, which is not a
# reproducibility guarantee. This tolerance verifies every value to four
# significant figures — the precision at which the deliverable reports them —
# while still catching any change large enough to alter a reported number.
CHECK_RTOL = 1e-4


class AggregationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _sha256(path: str) -> str:
    """Checksum a prediction artifact, which is too large to inline."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_path(manifest_path: str) -> str:
    """The prediction artifact beside a run manifest."""
    path = manifest_path.replace("_manifest.json", "_predictions.npz")
    if not os.path.exists(path):
        raise AggregationError(
            f"Prediction artifact missing for {os.path.relpath(manifest_path)}:\n"
            f"  expected {os.path.relpath(path)}\n"
            "Phase 13 recomputes calibration from the predictive distribution, so the "
            "*_predictions.npz files must be present. They are git-ignored; regenerate "
            "them with scripts/run_phase10.py and scripts/run_phase12.py."
        )
    return path


def _masked_training(manifest: Any) -> bool:
    masking = (getattr(manifest, "resolved_config", None) or {}).get("masking") or {}
    return bool(masking.get("apply_to_training", False))


def discover_runs(
    val_dir: str = VAL_DIR,
    normal_dir: str = NORMAL_DIR,
    condition_dir: str = CONDITION_DIR,
) -> dict[str, list[dict[str, Any]]]:
    """Collect the validation, normal and condition ST-DSSM runs.

    Returns ``{"validation": [...], "test": [...]}``. Every record carries the
    manifest, the prediction artifact and the condition identity needed to pair
    it with a frozen temperature.
    """
    groups: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}

    patterns = [
        ("validation", os.path.join(val_dir, "st_dssm-val-seed*_manifest.json")),
        ("test", os.path.join(normal_dir, "st_dssm-test-seed*_manifest.json")),
        ("test", os.path.join(condition_dir, "st_dssm-test-seed*_manifest.json")),
    ]

    for kind, pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            manifest = load_run_manifest(path)
            if manifest.model_name != "st_dssm":
                continue
            if manifest.run_status != "complete":
                raise AggregationError(
                    f"{os.path.relpath(path)} has run_status "
                    f"'{manifest.run_status}'; Phase 13 aggregates completed runs only."
                )
            pred_path = _prediction_path(path)
            masked_training = _masked_training(manifest)
            condition = manifest.mask_condition or "normal"
            if condition == "0%":
                condition = "normal"
            groups[kind].append(
                {
                    "manifest_path": os.path.relpath(path).replace("\\", "/"),
                    "prediction_path": os.path.relpath(pred_path).replace("\\", "/"),
                    "prediction_checksum": _sha256(pred_path),
                    "manifest_checksum": manifest.manifest_checksum,
                    "run_id": manifest.run_id,
                    "training_seed": manifest.training_seed,
                    "mask_seed": manifest.mask_seed,
                    "condition": condition,
                    "masked_training": masked_training,
                    "arm": "masked-training" if masked_training else "primary",
                    "split": manifest.split_evaluated,
                    "git_commit": manifest.git_commit,
                    "masking_config": (manifest.resolved_config or {}).get("masking"),
                    "metrics": manifest.metrics,
                    "sample_count": manifest.sample_count,
                    "sensor_count": manifest.sensor_count,
                    "dataset_manifest_checksum": manifest.dataset_manifest_checksum,
                    "split_artifact_checksum": manifest.split_artifact_checksum,
                    "package_versions": manifest.package_versions,
                }
            )

    if not groups["validation"]:
        raise AggregationError(
            f"No validation runs found under {val_dir}. ADR-0008 requires the "
            "temperature to be fitted on the validation split; generate the runs "
            "with scripts/run_phase13.py."
        )
    if not groups["test"]:
        raise AggregationError(
            f"No ST-DSSM test runs found under {normal_dir} or {condition_dir}."
        )

    val_seeds = sorted(r["training_seed"] for r in groups["validation"])
    if val_seeds != sorted(CANONICAL_SEEDS):
        raise AggregationError(
            f"Validation runs cover seeds {val_seeds}, expected {list(CANONICAL_SEEDS)}. "
            "A temperature must be fitted for every canonical seed."
        )
    return groups


# ---------------------------------------------------------------------------
# Targets and predictions
# ---------------------------------------------------------------------------


class _Targets:
    """Canonical targets in physical units, loaded once per split."""

    def __init__(self, artifact_dir: str):
        self._arrays, self._metadata = load_and_validate_artifact(artifact_dir)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def sensor_ids(self) -> list[str]:
        return [str(s) for s in self._metadata.get("sensor_ids", [])]

    def get(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y_true_mph, observation_mask)`` for a split."""
        if split not in self._cache:
            key = f"{split}_Y"
            if key not in self._arrays:
                raise AggregationError(
                    f"Split '{split}' not present in {ARTIFACT_DIR}: "
                    f"no '{key}' array."
                )
            y_true = inverse_transform_predictions(
                mu=self._arrays[key],
                scaler_mean=np.asarray(self._arrays["scaler_means"], dtype=np.float32),
                scaler_std=np.asarray(self._arrays["scaler_stds"], dtype=np.float32),
            )
            self._cache[split] = (y_true, self._arrays[f"{split}_Y_mask"])
        return self._cache[split]

    def scaler(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._arrays["scaler_means"], dtype=np.float32),
            np.asarray(self._arrays["scaler_stds"], dtype=np.float32),
        )


def load_predictions(
    run: dict[str, Any], targets: _Targets
) -> tuple[np.ndarray, np.ndarray]:
    """Load one run's predictive mean and scale in physical units (mph)."""
    scaler_mean, scaler_std = targets.scaler()
    with np.load(run["prediction_path"]) as data:
        if "predictions" in data:
            mu_norm = data["predictions"]
        elif "mu" in data:
            mu_norm = data["mu"]
        else:
            raise AggregationError(
                f"{run['prediction_path']} has no 'predictions' or 'mu' array."
            )
        if "sigma" not in data:
            raise AggregationError(
                f"{run['prediction_path']} has no 'sigma' array; a calibration "
                "diagnostic needs the predictive scale."
            )
        sigma_norm = data["sigma"]

    return inverse_transform_predictions(
        mu=mu_norm, sigma=sigma_norm, scaler_mean=scaler_mean, scaler_std=scaler_std
    )


def group_masks(
    run: dict[str, Any], targets: _Targets, shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Boolean evaluation masks for the all/masked/unmasked target groups.

    The experimental mask withholds model *inputs*; every target stays available
    to the evaluator (ADR-0004), so the groups partition the same predictions.
    The normal condition has one group.
    """
    _y, native_mask = targets.get(run["split"])
    base = native_mask.astype(bool, copy=False)
    groups: dict[str, np.ndarray] = {"all": base}

    generator = resolve_mask_generator(
        run.get("masking_config"), shape[-2], run.get("training_seed")
    )
    if generator is not None:
        groups.update(generator.node_group_masks(native_mask, shape))
    return groups


def _flatten(arrays: tuple[np.ndarray, ...], mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Select a group's valid targets once, so metrics need no further masking."""
    return tuple(arr[mask] for arr in arrays)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _arm_diagnostics(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    with_pit_histogram: bool,
) -> dict[str, Any]:
    """Diagnostics for one set of (already flattened) predictions."""
    rows = reliability_table(y_true, mu, sigma, levels=CANONICAL_LEVELS)
    lo, hi = prediction_interval(mu, sigma, nominal=NOMINAL_COVERAGE)
    pit = pit_values(y_true, mu, sigma)

    out: dict[str, Any] = {
        "NLL": gaussian_nll(y_true, mu, sigma),
        "CRPS": gaussian_crps(y_true, mu, sigma),
        "PICP": picp(y_true, lo, hi),
        "MPIW": mpiw(lo, hi),
        "interval_ECE": interval_ece(rows),
        "PIT_ECE": pit_ece(pit, bins=PIT_BINS),
        "reliability": [
            {
                "nominal_level": float(row["nominal_level"]),
                "empirical_coverage": float(row["empirical_coverage"]),
                "coverage_gap": float(row["coverage_gap"]),
                "mean_interval_width": float(row["mean_interval_width"]),
            }
            for row in rows
        ],
    }
    if with_pit_histogram:
        densities, edges = pit_histogram(pit, bins=PIT_BINS)
        out["pit_densities"] = [float(d) for d in densities]
        out["pit_edges"] = [float(e) for e in edges]
    return out


def diagnose_run(
    run: dict[str, Any],
    targets: _Targets,
    temperature: float | None,
) -> dict[str, Any]:
    """Compute raw and calibrated diagnostics for every target group of a run."""
    y_true, _ = targets.get(run["split"])
    mu, sigma = load_predictions(run, targets)
    if mu.shape != y_true.shape:
        raise AggregationError(
            f"{run['prediction_path']} has shape {mu.shape} but the "
            f"{run['split']} targets have shape {y_true.shape}."
        )

    masks = group_masks(run, targets, y_true.shape)
    result: dict[str, Any] = {"groups": {}}

    for group, mask in masks.items():
        if not mask.any():
            continue
        g_true, g_mu, g_sigma = _flatten((y_true, mu, sigma), mask)
        # MAE is reported alongside so a calibration row can be read next to the
        # accuracy it belongs to without opening another file; it is identical
        # in both arms because calibration cannot move the mean.
        arms = {
            "raw": _arm_diagnostics(
                g_true, g_mu, g_sigma, with_pit_histogram=(group == "all")
            )
        }
        if temperature is not None:
            arms["calibrated"] = _arm_diagnostics(
                g_true,
                g_mu,
                apply_temperature(g_sigma, temperature),
                with_pit_histogram=(group == "all"),
            )
        result["groups"][group] = {
            "valid_count": int(mask.sum()),
            "MAE": mae(g_true, g_mu),
            **arms,
        }

    if temperature is not None:
        result["temperature"] = float(temperature)
    return result


def fit_temperatures(
    val_runs: list[dict[str, Any]], targets: _Targets
) -> dict[int, dict[str, Any]]:
    """Fit and record one frozen temperature per seed on the validation split.

    ADR-0008: one positive scalar shared across sensors and horizons, selected
    by validation NLL, frozen before test evaluation. Nothing in this function
    may read a test split.
    """
    fitted: dict[int, dict[str, Any]] = {}
    for run in sorted(val_runs, key=lambda r: r["training_seed"]):
        if run["split"] != "val":
            raise AggregationError(
                f"{run['run_id']} is a '{run['split']}' run; a temperature may "
                "only be fitted on the validation split (ADR-0008)."
            )
        y_true, native_mask = targets.get("val")
        mu, sigma = load_predictions(run, targets)
        mask = native_mask.astype(bool, copy=False)
        g_true, g_mu, g_sigma = _flatten((y_true, mu, sigma), mask)

        temperature = fit_temperature(g_true, g_mu, g_sigma)
        raw = _arm_diagnostics(g_true, g_mu, g_sigma, with_pit_histogram=False)
        calibrated = _arm_diagnostics(
            g_true, g_mu, apply_temperature(g_sigma, temperature), with_pit_histogram=False
        )

        fitted[int(run["training_seed"])] = {
            "temperature": temperature,
            "run_id": run["run_id"],
            "valid_count": int(mask.sum()),
            "validation_raw": {k: raw[k] for k in ("NLL", "interval_ECE", "PIT_ECE", "PICP")},
            "validation_calibrated": {
                k: calibrated[k] for k in ("NLL", "interval_ECE", "PIT_ECE", "PICP")
            },
        }
    return fitted


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def _stat(values: list[float]) -> dict[str, Any]:
    """Mean, sample standard deviation and count across seeds."""
    clean = [float(v) for v in values]
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
        "values": clean,
    }


SCALAR_DIAGNOSTICS = ("NLL", "CRPS", "PICP", "MPIW", "interval_ECE", "PIT_ECE")


def _aggregate_condition(diagnosed: list[dict[str, Any]]) -> dict[str, Any]:
    """Average one condition's per-seed diagnostics across seeds."""
    seeds = sorted(d["seed"] for d in diagnosed)
    per_group: dict[str, Any] = {}

    for group in TARGET_GROUPS:
        present = [d for d in diagnosed if group in d["diagnostics"]["groups"]]
        if not present:
            continue
        group_out: dict[str, Any] = {
            "valid_count": present[0]["diagnostics"]["groups"][group]["valid_count"],
            "MAE": _stat([d["diagnostics"]["groups"][group]["MAE"] for d in present]),
        }
        for arm in ARMS:
            if arm not in present[0]["diagnostics"]["groups"][group]:
                continue
            arm_rows = [d["diagnostics"]["groups"][group][arm] for d in present]
            arm_out: dict[str, Any] = {
                metric: _stat([row[metric] for row in arm_rows])
                for metric in SCALAR_DIAGNOSTICS
            }
            # Reliability is averaged level by level so the locked curve is the
            # across-seed mean, matching how every other table reports.
            arm_out["reliability"] = [
                {
                    "nominal_level": float(level),
                    "empirical_coverage": _stat(
                        [row["reliability"][i]["empirical_coverage"] for row in arm_rows]
                    ),
                    "coverage_gap": _stat(
                        [row["reliability"][i]["coverage_gap"] for row in arm_rows]
                    ),
                    "mean_interval_width": _stat(
                        [row["reliability"][i]["mean_interval_width"] for row in arm_rows]
                    ),
                }
                for i, level in enumerate(sorted(CANONICAL_LEVELS))
            ]
            if "pit_densities" in arm_rows[0]:
                n_bins = len(arm_rows[0]["pit_densities"])
                arm_out["pit_densities"] = [
                    statistics.fmean([row["pit_densities"][i] for row in arm_rows])
                    for i in range(n_bins)
                ]
                arm_out["pit_edges"] = arm_rows[0]["pit_edges"]
            group_out[arm] = arm_out
        per_group[group] = group_out

    return {"seeds": seeds, "per_group": per_group}


def _per_horizon_from_manifests(
    diagnosed: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Per-horizon error curves, read from the run manifests, not recomputed.

    The horizon decay is already a locked manifest quantity, so Phase 13 reads
    it rather than deriving a second version of it.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for group in TARGET_GROUPS:
        by_horizon: dict[int, dict[str, list[float]]] = {}
        for entry in diagnosed:
            for rec in entry["metrics"]:
                horizon = rec.horizon if hasattr(rec, "horizon") else rec.get("horizon")
                if horizon == "aggregate":
                    continue
                rec_group = (
                    rec.target_group if hasattr(rec, "target_group") else rec.get("target_group")
                ) or "all"
                if rec_group != group:
                    continue
                name = rec.name if hasattr(rec, "name") else rec.get("name")
                value = rec.value if hasattr(rec, "value") else rec.get("value")
                step = int(horizon)
                by_horizon.setdefault(step, {}).setdefault(name, []).append(float(value))
        if not by_horizon:
            continue
        out[group] = [
            {
                "horizon_step": step,
                "horizon_minutes": step * 5,
                **{name: _stat(values) for name, values in sorted(by_horizon[step].items())},
            }
            for step in sorted(by_horizon)
        ]
    return out


def _run_environments(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group the source runs by the package versions they were executed under."""
    buckets: dict[str, dict[str, Any]] = {}
    for run in runs:
        versions = run["package_versions"] or {}
        entry = buckets.setdefault(
            json.dumps(versions, sort_keys=True),
            {"package_versions": versions, "run_ids": []},
        )
        entry["run_ids"].append(run["run_id"])
    for entry in buckets.values():
        entry["run_ids"].sort()
        entry["run_count"] = len(entry["run_ids"])
    return [buckets[key] for key in sorted(buckets)]


def aggregate(
    runs: dict[str, list[dict[str, Any]]],
    targets: _Targets,
    latency: dict[str, Any],
) -> dict[str, Any]:
    """Build the full Phase 13 aggregate."""
    temperatures = fit_temperatures(runs["validation"], targets)

    primary: dict[str, list[dict[str, Any]]] = {}
    ablation: dict[str, list[dict[str, Any]]] = {}

    for run in sorted(runs["test"], key=lambda r: (r["condition"], r["training_seed"])):
        seed = int(run["training_seed"])
        if seed not in temperatures:
            raise AggregationError(
                f"{run['run_id']} was trained with seed {seed}, which has no "
                "validation-fitted temperature. ADR-0008 forbids reusing another "
                "seed's calibration."
            )
        temperature = temperatures[seed]["temperature"]
        print(
            f"  diagnosing {run['run_id']} "
            f"({run['condition']}, {run['arm']}, T={temperature:.4f})",
            flush=True,
        )
        entry = {
            "seed": seed,
            "run_id": run["run_id"],
            "metrics": run["metrics"],
            "diagnostics": diagnose_run(run, targets, temperature),
        }
        bucket = ablation if run["masked_training"] else primary
        bucket.setdefault(run["condition"], []).append(entry)

    conditions = {
        condition: {
            **_aggregate_condition(entries),
            "per_horizon": _per_horizon_from_manifests(entries),
        }
        for condition, entries in primary.items()
    }
    ablations = {
        condition: {
            **_aggregate_condition(entries),
            "per_horizon": _per_horizon_from_manifests(entries),
        }
        for condition, entries in ablation.items()
    }

    missing = [c for c in CONDITION_ORDER if c not in conditions]
    if missing:
        raise AggregationError(
            f"No ST-DSSM runs found for condition(s) {missing}. Phase 13 reviews "
            "the normal condition and all three missingness levels."
        )

    reference = runs["test"][0]
    return {
        "study": "Phase 13 calibration and robustness review",
        "model": "st_dssm",
        "protocol": {
            "temperature_scaling": (
                "One positive scalar per seed, shared across sensors and horizons, "
                "fitted on the validation split by Gaussian NLL and frozen before "
                "test evaluation (ADR-0008). Predictive means are never altered."
            ),
            "nominal_levels": sorted(CANONICAL_LEVELS),
            "primary_coverage_level": NOMINAL_COVERAGE,
            "pit_bins": PIT_BINS,
            "target_groups": list(TARGET_GROUPS),
            "seeds": list(CANONICAL_SEEDS),
        },
        "temperatures": {
            str(seed): record for seed, record in sorted(temperatures.items())
        },
        "temperature_summary": _stat(
            [record["temperature"] for record in temperatures.values()]
        ),
        "conditions": conditions,
        "ablations": {"masked_training": ablations} if ablations else {},
        "latency": latency,
        "dataset_manifest_checksum": reference["dataset_manifest_checksum"],
        "split_artifact_checksum": reference["split_artifact_checksum"],
        # The source runs were not all executed under one installed stack (a
        # CUDA build of torch was reinstalled mid-study), so every distinct
        # version set is recorded with the runs that used it rather than
        # reporting one run's versions as if they covered the whole study.
        "run_environments": _run_environments(
            runs["validation"] + runs["test"]
        ),
        "sources": [
            {
                "run_id": run["run_id"],
                "kind": kind,
                "condition": run["condition"],
                "arm": run["arm"],
                "seed": run["training_seed"],
                "split": run["split"],
                "manifest_path": run["manifest_path"],
                "manifest_checksum": run["manifest_checksum"],
                "prediction_path": run["prediction_path"],
                "prediction_checksum": run["prediction_checksum"],
                "git_commit": run["git_commit"],
            }
            for kind in ("validation", "test")
            for run in sorted(
                runs[kind], key=lambda r: (r["condition"], r["arm"], r["training_seed"])
            )
        ],
    }


# ---------------------------------------------------------------------------
# Latency record
# ---------------------------------------------------------------------------


def load_latency(results_dir: str) -> dict[str, Any]:
    """Fold the measured latency records into the deliverable.

    Both the accelerator and the CPU-fallback record are included when present:
    ADR-0008 permits a CPU fallback, so a claim about feasibility needs both.
    """
    records: dict[str, Any] = {}
    for label, filename in (
        ("accelerator", "latency_record.json"),
        ("cpu", "latency_record_cpu.json"),
    ):
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        record.pop("measured_at_utc", None)
        records[label] = record

    if not records:
        raise AggregationError(
            f"No latency record under {results_dir}. Phase 13 requires measured "
            "latency; run scripts/measure_latency.py (and --device cpu for the "
            "fallback record)."
        )
    return records


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _f(stat: dict[str, Any] | None, decimals: int = 4) -> str:
    """Format a mean +/- std cell."""
    if stat is None:
        return "n/a"
    return f"{stat['mean']:.{decimals}f} +/- {stat['std']:.{decimals}f}"


def _delta(new: dict[str, Any], old: dict[str, Any], decimals: int = 4) -> str:
    return f"{new['mean'] - old['mean']:+.{decimals}f}"


def render_markdown(agg: dict[str, Any]) -> str:
    lines: list[str] = []
    levels = agg["protocol"]["nominal_levels"]

    lines += [
        "# Phase 13 — Calibration and robustness review (locked results)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "Calibration diagnostics, per-horizon error analysis and measured inference",
        "latency for the locked ST-DSSM model, under the normal condition and all",
        "three missingness conditions. Every number here is recomputed from the",
        "prediction artifacts and run manifests listed under [Provenance](#provenance);",
        "`python scripts/aggregate_phase13.py --check` re-derives them.",
        "",
        "Only ST-DSSM is diagnosed: calibration is a property of a predictive",
        "distribution and the two baselines are point forecasters (ADR-0008). They",
        "appear in the latency record below and in the Phase 10/12 accuracy tables.",
        "",
        "## Temperature scaling (secondary analysis, ADR-0008)",
        "",
        agg["protocol"]["temperature_scaling"],
        "",
        "`T > 1` means the raw forecast was over-confident in the NLL sense and the",
        "fit widens the predictive scale. The fit sees the validation split only.",
        "",
        "| Seed | Fitted T | Val NLL raw | Val NLL calibrated | Val interval ECE raw | Val interval ECE calibrated |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, record in agg["temperatures"].items():
        raw, cal = record["validation_raw"], record["validation_calibrated"]
        lines.append(
            f"| {seed} | {record['temperature']:.4f} | {raw['NLL']:.4f} | "
            f"{cal['NLL']:.4f} | {raw['interval_ECE']:.4f} | {cal['interval_ECE']:.4f} |"
        )
    summary = agg["temperature_summary"]
    lines += [
        "",
        f"Across-seed temperature: **{summary['mean']:.4f} +/- {summary['std']:.4f}**.",
        "",
    ]

    # --- Calibration per condition ---
    lines += [
        "## Calibration under each condition (test set)",
        "",
        "Values are **mean +/- sample standard deviation across the three canonical",
        f"seeds** {agg['protocol']['seeds']}. `interval ECE` is the mean absolute gap",
        f"between empirical and nominal coverage over levels {levels}; `PIT ECE` is the",
        "total-variation distance between the PIT histogram and the uniform. Both are",
        "0 for a perfectly calibrated forecast. MAE is identical in both arms because",
        "temperature scaling cannot move a predictive mean.",
        "",
    ]
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond:
            continue
        label = "Normal (0% missing)" if condition == "normal" else f"{condition} missing"
        lines += [
            f"### {label}",
            "",
            "| Group | Arm | NLL | CRPS (mph) | PICP@95% | MPIW (mph) | interval ECE | PIT ECE |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for group in TARGET_GROUPS:
            grp = cond["per_group"].get(group)
            if not grp:
                continue
            for arm in ARMS:
                if arm not in grp:
                    continue
                a = grp[arm]
                lines.append(
                    f"| {group} | {arm} | {_f(a['NLL'])} | {_f(a['CRPS'], 3)} | "
                    f"{_f(a['PICP'])} | {_f(a['MPIW'], 3)} | {_f(a['interval_ECE'])} | "
                    f"{_f(a['PIT_ECE'])} |"
                )
        lines.append("")

    # --- Reliability curves ---
    lines += [
        "## Reliability by nominal level",
        "",
        "Signed coverage gap (empirical minus nominal) for the `all` target group.",
        "A positive gap means the interval is too wide at that level.",
        "",
        "| Condition | Arm | " + " | ".join(f"{level:.0%}" for level in levels) + " |",
        "| :--- | :--- | " + " | ".join(["---:"] * len(levels)) + " |",
    ]
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond or "all" not in cond["per_group"]:
            continue
        for arm in ARMS:
            if arm not in cond["per_group"]["all"]:
                continue
            gaps = cond["per_group"]["all"][arm]["reliability"]
            cells = " | ".join(f"{row['coverage_gap']['mean']:+.4f}" for row in gaps)
            lines.append(f"| {condition} | {arm} | {cells} |")
    lines.append("")

    # --- Error analysis ---
    lines += [
        "## Error analysis: horizon decay under each condition",
        "",
        "Per-horizon values are read from the run manifests, not recomputed here.",
        "MAE in mph for the `all` target group.",
        "",
        "| Condition | 15 min | 30 min | 45 min | 60 min |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond:
            continue
        rows = {r["horizon_minutes"]: r for r in cond["per_horizon"].get("all", [])}
        cells = []
        for minutes in (15, 30, 45, 60):
            row = rows.get(minutes)
            cells.append(_f(row["MAE"], 3) if row and "MAE" in row else "n/a")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "### Withheld vs observed sensors",
        "",
        "The experimental mask withholds model inputs; every target is still scored",
        "(ADR-0004), so a condition splits into withheld (`masked`) and observed",
        "(`unmasked`) sensor groups over one set of predictions.",
        "",
        "| Condition | masked MAE | unmasked MAE | ratio | masked PICP@95% | unmasked PICP@95% |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond:
            continue
        masked = cond["per_group"].get("masked")
        unmasked = cond["per_group"].get("unmasked")
        if not masked or not unmasked:
            continue
        ratio = masked["MAE"]["mean"] / unmasked["MAE"]["mean"]
        lines.append(
            f"| {condition} | {_f(masked['MAE'], 3)} | {_f(unmasked['MAE'], 3)} | "
            f"{ratio:.2f}x | {_f(masked['raw']['PICP'])} | {_f(unmasked['raw']['PICP'])} |"
        )
    lines.append("")

    lines += [
        "### Coverage decay across the horizon",
        "",
        "Empirical coverage of the nominal 95% interval, by horizon and target group.",
        "Coverage is **worst at the shortest horizon** under missingness, which inverts",
        "the usual intuition: at 5 minutes the predictive scale is normally very small,",
        "because the next step is close to the last observation, so withholding that",
        "observation leaves a confident interval with nothing to justify it. By 60",
        "minutes the scale is wide regardless and the missing input matters less.",
        "",
        "| Condition | Group | 5 min | 15 min | 30 min | 60 min |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond:
            continue
        for group in TARGET_GROUPS:
            rows = {
                row["horizon_minutes"]: row
                for row in cond["per_horizon"].get(group, [])
            }
            if not rows or "PICP" not in next(iter(rows.values())):
                continue
            cells = []
            for minutes in (5, 15, 30, 60):
                row = rows.get(minutes)
                cells.append(
                    f"{row['PICP']['mean']:.3f}" if row and "PICP" in row else "n/a"
                )
            lines.append(
                f"| {condition} | {group} | " + " | ".join(cells) + " |"
            )
    lines.append("")

    # --- Ablation ---
    ablations = (agg.get("ablations") or {}).get("masked_training") or {}
    if ablations:
        lines += [
            "## Masked-training ablation (apply_to_training: true)",
            "",
            "These runs retrain with the same sensors withheld, so they are a",
            "different model from a primary condition run and are reported separately",
            "(ADR-0011). `Primary` repeats the inference-only cell above.",
            "",
            "| Condition | Group | Arm | Primary NLL | Ablation NLL | Primary interval ECE | Ablation interval ECE |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
        ]
        for condition, cond in sorted(ablations.items()):
            primary = agg["conditions"].get(condition, {}).get("per_group", {})
            for group in TARGET_GROUPS:
                grp = cond["per_group"].get(group)
                base = primary.get(group)
                if not grp or not base:
                    continue
                for arm in ARMS:
                    if arm not in grp or arm not in base:
                        continue
                    lines.append(
                        f"| {condition} | {group} | {arm} | {_f(base[arm]['NLL'])} | "
                        f"{_f(grp[arm]['NLL'])} | {_f(base[arm]['interval_ECE'])} | "
                        f"{_f(grp[arm]['interval_ECE'])} |"
                    )
        lines.append("")

    # --- Latency ---
    lines += [
        "## Inference latency",
        "",
        "One forward pass producing 12 horizons for all 325 sensors from one window",
        "of history. Measured with warmup and, on CUDA, device synchronization around",
        "every timed region; the median and p95 are reported because latency",
        "distributions are right-skewed. Training cost is recorded in the Phase 10/12",
        "run manifests, not here.",
        "",
    ]
    for label, record in agg["latency"].items():
        env = record["environment"]
        proto = record["protocol"]
        lines += [
            f"### {label} — {env['device_name']} ({env['device']})",
            "",
            (
                f"torch {env['torch']}, batch size {proto['batch_size']}, "
                f"{proto['warmup_batches']} warmup + {proto['timed_batches']} timed batches."
            ),
            "",
            "| Model | Parameters | median ms/window | p95 ms/window | ms/batch | windows/s |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for model in record["models"]:
            lines.append(
                f"| {model['model_name']} | {model['trainable_parameters']:,} | "
                f"{model['per_window_median_ms']:.4f} | {model['per_window_p95_ms']:.4f} | "
                f"{model['per_batch_median_ms']:.2f} | {model['windows_per_second']:,.1f} |"
            )
        lines.append("")

    # --- Provenance ---
    lines += [
        "## Provenance",
        "",
        "Phase 13 recomputes calibration from the predictive distribution, so both",
        "the manifest and the prediction artifact are checksummed. Prediction",
        "artifacts are git-ignored; regenerate them with the Phase 10/12/13 runners.",
        "",
        "| Kind | Condition | Arm | Seed | Run manifest | Manifest SHA256 | Predictions SHA256 | Code commit |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for source in agg["sources"]:
        lines.append(
            f"| {source['kind']} | {source['condition']} | {source['arm']} | "
            f"{source['seed']} | `{source['manifest_path']}` | "
            f"`{(source['manifest_checksum'] or '')[:16]}...` | "
            f"`{source['prediction_checksum'][:16]}...` | "
            f"`{(source['git_commit'] or '')[:12]}` |"
        )

    environments = agg["run_environments"]
    lines += [""]
    if len(environments) == 1:
        versions = ", ".join(
            f"{k} {v}" for k, v in sorted(environments[0]["package_versions"].items())
        )
        lines += [f"**Run environment:** {versions}", ""]
    else:
        lines += [
            (
                "**Run environments:** the source runs were not all executed under "
                "one installed stack, so each is recorded with the runs that used it."
            ),
            "",
        ]
        for i, env in enumerate(environments, start=1):
            versions = ", ".join(
                f"{k} {v}" for k, v in sorted(env["package_versions"].items())
            )
            lines += [f"{i}. {versions} — {env['run_count']} run(s):", ""]
            lines += [f"   - `{run_id}`" for run_id in env["run_ids"]]
            lines += [""]
    lines += [
        "Calibration diagnostics were recomputed in the environment recorded in the",
        "latency record above, which may differ from the environment of the original",
        "runs; the recomputation reads saved predictions and does not re-run a model.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def render_figures(agg: dict[str, Any], output_dir: str) -> list[str]:
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    written: list[str] = []
    levels = agg["protocol"]["nominal_levels"]

    # Reliability, raw vs calibrated, one figure per condition.
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond or "all" not in cond["per_group"]:
            continue
        arms = {}
        for arm in ARMS:
            if arm not in cond["per_group"]["all"]:
                continue
            label = "raw" if arm == "raw" else "temperature-scaled"
            arms[label] = [
                row["empirical_coverage"]["mean"]
                for row in cond["per_group"]["all"][arm]["reliability"]
            ]
        if not arms:
            continue
        tag = "normal" if condition == "normal" else f"node{condition.rstrip('%')}"
        label = "normal condition" if condition == "normal" else f"{condition} missing"
        written.append(
            plot_reliability_comparison(
                levels,
                arms,
                os.path.join(plots_dir, f"phase13_reliability_{tag}.png"),
                title=f"ST-DSSM calibration, {label} (3-seed mean)",
            )
        )

        densities = cond["per_group"]["all"]["raw"].get("pit_densities")
        if densities:
            written.append(
                plot_pit_histogram(
                    densities,
                    os.path.join(plots_dir, f"phase13_pit_{tag}.png"),
                    edges=cond["per_group"]["all"]["raw"].get("pit_edges"),
                    title=f"ST-DSSM PIT, {label} (3-seed mean)",
                    label="observed PIT density",
                )
            )

    # Horizon decay across conditions, one figure per metric.
    for metric, unit, reference in (
        ("MAE", "mph", None),
        ("NLL", "nats", None),
        ("PICP", "ratio", NOMINAL_COVERAGE),
    ):
        series: dict[str, tuple[list[float], list[float]]] = {}
        minutes: list[int] = []
        for condition in CONDITION_ORDER:
            cond = agg["conditions"].get(condition)
            if not cond:
                continue
            rows = cond["per_horizon"].get("all", [])
            if not rows or metric not in rows[0]:
                continue
            minutes = [row["horizon_minutes"] for row in rows]
            label = "normal" if condition == "normal" else f"{condition} missing"
            series[label] = (
                [row[metric]["mean"] for row in rows],
                [row[metric]["std"] for row in rows],
            )
        if not series or not minutes:
            continue
        written.append(
            plot_horizon_curves(
                minutes,
                series,
                metric,
                os.path.join(plots_dir, f"phase13_degradation_{metric.lower()}.png"),
                unit=unit,
                title=f"ST-DSSM {metric} decay across horizons by condition",
                reference_level=reference,
                reference_label=f"nominal {NOMINAL_COVERAGE:.0%}" if reference else None,
            )
        )

    # Withheld vs observed sensors, MAE across horizons.
    series = {}
    minutes = []
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond:
            continue
        for group in ("masked", "unmasked"):
            rows = cond["per_horizon"].get(group, [])
            if not rows or "MAE" not in rows[0]:
                continue
            minutes = [row["horizon_minutes"] for row in rows]
            series[f"{condition} {group}"] = (
                [row["MAE"]["mean"] for row in rows],
                [row["MAE"]["std"] for row in rows],
            )
    if series and minutes:
        written.append(
            plot_horizon_curves(
                minutes,
                series,
                "MAE",
                os.path.join(plots_dir, "phase13_masked_vs_unmasked_mae.png"),
                unit="mph",
                title="ST-DSSM MAE at withheld vs observed sensors",
            )
        )

    return written


# ---------------------------------------------------------------------------
# Check
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
    parser = argparse.ArgumentParser(
        description="Aggregate and lock the Phase 13 calibration and robustness review."
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"Where the locked deliverable is written. [default: {RESULTS_DIR}]",
    )
    parser.add_argument(
        "--artifact-dir",
        default=ARTIFACT_DIR,
        help=f"Canonical Phase 3 data artifact. [default: {ARTIFACT_DIR}]",
    )
    parser.add_argument(
        "--val-dir",
        default=VAL_DIR,
        help=f"Validation runs used to fit the temperature. [default: {VAL_DIR}]",
    )
    parser.add_argument(
        "--normal-dir",
        default=NORMAL_DIR,
        help=f"Phase 10 normal-condition test runs. [default: {NORMAL_DIR}]",
    )
    parser.add_argument(
        "--condition-dir",
        default=CONDITION_DIR,
        help=f"Phase 12 missingness condition runs. [default: {CONDITION_DIR}]",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed aggregate still matches the artifacts; write nothing.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild the tables and figures from the committed JSON without "
        "recomputing any diagnostic. The markdown and figures are pure functions "
        "of that aggregate, so this is for presentation changes only; it can "
        "neither change a locked number nor detect drift (use --check for that).",
    )
    args = parser.parse_args(argv)

    json_path = os.path.join(args.results_dir, "phase13_results.json")
    md_path = os.path.join(args.results_dir, "phase13_results.md")

    if args.render_only:
        if args.check:
            print(
                "[error] --render-only recomputes nothing, so it cannot verify "
                "anything; use --check on its own.",
                file=sys.stderr,
            )
            return 2
        if not os.path.exists(json_path):
            print(f"Nothing to render: {json_path} does not exist.", file=sys.stderr)
            return 1
        with open(json_path, encoding="utf-8") as f:
            committed = json.load(f)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(committed))
        figures = render_figures(committed, args.results_dir)
        print(f"Re-rendered {md_path} and {len(figures)} figures from {json_path}.")
        print("No diagnostic was recomputed; run --check to verify the lock.")
        return 0

    try:
        runs = discover_runs(args.val_dir, args.normal_dir, args.condition_dir)
        targets = _Targets(args.artifact_dir)
        latency = load_latency(args.results_dir)
        print(
            f"Diagnosing {len(runs['test'])} test runs against "
            f"{len(runs['validation'])} validation-fitted temperatures:"
        )
        agg = aggregate(runs, targets, latency)
    except AggregationError as exc:
        print(f"Aggregation error: {exc}", file=sys.stderr)
        return 1

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
                f"Phase 13 aggregate no longer matches {json_path}:\n  - "
                + "\n  - ".join(diffs),
                file=sys.stderr,
            )
            return 1
        print(
            f"Phase 13 results verified: {len(agg['sources'])} runs reproduce "
            f"{os.path.relpath(json_path)} to within {CHECK_RTOL:g} relative "
            "(four significant figures; see the CHECK_RTOL comment)."
        )
        return 0

    agg["generated_at"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(args.results_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(agg))
    figures = render_figures(agg, args.results_dir)

    summary = agg["temperature_summary"]
    print(
        f"\nFitted temperature across seeds: {summary['mean']:.4f} +/- {summary['std']:.4f}"
    )
    for condition in CONDITION_ORDER:
        cond = agg["conditions"].get(condition)
        if not cond:
            continue
        grp = cond["per_group"]["all"]
        print(
            f"  {condition:<8} raw NLL {_f(grp['raw']['NLL'])}  "
            f"interval ECE {_f(grp['raw']['interval_ECE'])} -> "
            f"{_f(grp['calibrated']['interval_ECE'])} calibrated"
        )
    print(f"Locked tables: {md_path}")
    print(f"Locked data:   {json_path}")
    print(f"Locked figures: {len(figures)} under {os.path.join(args.results_dir, 'plots')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
