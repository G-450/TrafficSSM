"""Tests for scripts/aggregate_phase10.py.

The locked Phase 10 tables must be reproducible from the run manifests alone, and
the aggregator must refuse to average runs that are not comparable rather than
produce a table that looks authoritative.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from st_dssm.result_schema import RunManifest, save_run_manifest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(
    results_dir: Path,
    model_name: str,
    seed: int | None,
    mae: float,
    *,
    with_uncertainty: bool = False,
    split: str = "test",
    mask_condition: str = "0%",
    run_id: str | None = None,
) -> Path:
    """Write a minimal but schema-valid run manifest for the aggregator to read."""
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"{model_name}-{split}-seed{seed}"
    manifest = RunManifest(
        run_id=run_id,
        model_name=model_name,
        split_evaluated=split,
        sample_count=10403,
        sensor_count=325,
        forecast_horizon=12,
        training_seed=seed,
        mask_condition=mask_condition,
        trainable_parameters=397058 if model_name == "st_dssm" else 109260,
        best_epoch=58,
        total_epochs=60,
        selection_metric="val_nll" if model_name == "st_dssm" else "val_mae",
        split_artifact_checksum="split-digest",
        dataset_manifest_checksum="dataset-digest",
        package_versions={"python": "3.12.10", "numpy": "1.26.4"},
    )

    overall = {"MAE": mae, "RMSE": mae * 2, "MAPE": mae * 2.3, "valid_count": 40569096}
    per_horizon = []
    for step in range(1, 13):
        row = {
            "horizon_step": step,
            "horizon_minutes": step * 5,
            "MAE": mae + 0.05 * step,
            "RMSE": (mae + 0.05 * step) * 2,
            "MAPE": (mae + 0.05 * step) * 2.3,
            "valid_count": 3380758,
        }
        if with_uncertainty:
            row.update(
                {
                    "NLL": 2.0 + 0.01 * step,
                    "CRPS": 0.6 + 0.1 * step,
                    "PICP": 0.96,
                    "MPIW": 5.0 + step,
                }
            )
        per_horizon.append(row)
    if with_uncertainty:
        overall.update({"NLL": 2.1, "CRPS": 1.43, "PICP": 0.96, "MPIW": 11.0})

    manifest.overall_metrics = overall
    manifest.per_horizon_metrics = per_horizon
    manifest.mark_complete()

    path = results_dir / f"{run_id}_manifest.json"
    save_run_manifest(manifest, str(path))
    return path


def _complete_study(tmp_path: Path) -> Path:
    """A full Phase 10 study: three seeds per trained model plus the persistence run."""
    results = tmp_path / "normal"
    for i, seed in enumerate((2026, 2027, 2028)):
        _write_run(results, "st_dssm", seed, 1.93 + 0.01 * i, with_uncertainty=True)
        _write_run(results / "baselines", "deterministic_st_gcn", seed, 1.77 + 0.01 * i)
    _write_run(
        results / "baselines",
        "historical_persistence",
        None,
        2.17,
        run_id="baseline-persistence-test",
    )
    return results


def _figures_in(tables: str) -> list[str]:
    """Pull the figure paths out of the rendered markdown."""
    lines = tables.splitlines()
    start = lines.index("## Figures")
    figures = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("- "):
            figures.append(line[2:].strip().strip("`"))
    return figures


def test_aggregate_locks_tables_figures_and_machine_readable_results(tmp_path) -> None:
    aggregate_phase10 = _load_script("aggregate_phase10")
    results = _complete_study(tmp_path)

    assert aggregate_phase10.main(["--results-dir", str(results)]) == 0

    payload = json.loads((results / "phase10_results.json").read_text(encoding="utf-8"))
    assert sorted(payload["models"]) == [
        "deterministic_st_gcn",
        "historical_persistence",
        "st_dssm",
    ]
    assert payload["models"]["st_dssm"]["seeds"] == [2026, 2027, 2028]
    assert payload["models"]["st_dssm"]["overall"]["MAE"]["n"] == 3
    assert payload["models"]["st_dssm"]["overall"]["MAE"]["mean"] == pytest.approx(1.94)
    assert payload["models"]["st_dssm"]["overall"]["MAE"]["std"] > 0

    # A deterministic single-run model reports no spread.
    assert payload["models"]["historical_persistence"]["overall"]["MAE"]["std"] == 0.0
    # Point-only models carry no uncertainty metrics.
    assert "NLL" not in payload["models"]["deterministic_st_gcn"]["overall"]
    # Every source manifest is listed with its checksum, for audit.
    assert len(payload["sources"]) == 7
    assert all(source["manifest_checksum"] for source in payload["sources"])

    tables = (results / "phase10_results.md").read_text(encoding="utf-8")
    assert "Phase 10" in tables
    assert "ST-DSSM (ours)" in tables
    assert "1.940 ±" in tables  # mean +/- std cell
    assert "dataset-digest" in tables

    figures = _figures_in(tables)
    assert figures
    for figure in figures:
        assert (results / figure).exists()


def test_aggregate_check_passes_on_unchanged_manifests_and_fails_on_drift(tmp_path) -> None:
    aggregate_phase10 = _load_script("aggregate_phase10")
    results = _complete_study(tmp_path)
    assert aggregate_phase10.main(["--results-dir", str(results), "--no-figures"]) == 0

    # Re-deriving the aggregate from the same manifests reproduces it exactly.
    assert aggregate_phase10.main(["--results-dir", str(results), "--check"]) == 0

    # A changed result no longer matches the locked tables.
    manifest_path = results / "st_dssm-test-seed2026_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["overall_metrics"]["MAE"] = 99.0
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert aggregate_phase10.main(["--results-dir", str(results), "--check"]) == 1


def test_check_without_a_locked_aggregate_fails(tmp_path) -> None:
    aggregate_phase10 = _load_script("aggregate_phase10")
    results = _complete_study(tmp_path)
    assert aggregate_phase10.main(["--results-dir", str(results), "--check"]) == 1


def test_aggregate_requires_every_locked_seed(tmp_path) -> None:
    aggregate_phase10 = _load_script("aggregate_phase10")
    results = tmp_path / "partial"
    _write_run(results, "st_dssm", 2026, 1.93, with_uncertainty=True)
    _write_run(results, "st_dssm", 2027, 1.94, with_uncertainty=True)
    for seed in (2026, 2027, 2028):
        _write_run(results / "baselines", "deterministic_st_gcn", seed, 1.77)

    assert aggregate_phase10.main(["--results-dir", str(results), "--no-figures"]) == 1


def test_aggregate_rejects_a_different_masking_condition(tmp_path) -> None:
    aggregate_phase10 = _load_script("aggregate_phase10")
    results = tmp_path / "mixed"
    for seed in (2026, 2027):
        _write_run(results, "st_dssm", seed, 1.93, with_uncertainty=True)
    _write_run(results, "st_dssm", 2028, 2.5, with_uncertainty=True, mask_condition="20%")
    for seed in (2026, 2027, 2028):
        _write_run(results / "baselines", "deterministic_st_gcn", seed, 1.77)

    assert aggregate_phase10.main(["--results-dir", str(results), "--no-figures"]) == 1


def test_aggregate_rejects_duplicate_runs_for_one_seed(tmp_path) -> None:
    aggregate_phase10 = _load_script("aggregate_phase10")
    results = _complete_study(tmp_path)
    _write_run(
        results,
        "st_dssm",
        2026,
        1.80,
        with_uncertainty=True,
        run_id="st_dssm-test-seed2026-rerun",
    )

    assert aggregate_phase10.main(["--results-dir", str(results), "--no-figures"]) == 1


def test_aggregate_rejects_an_empty_results_directory(tmp_path) -> None:
    aggregate_phase10 = _load_script("aggregate_phase10")
    assert aggregate_phase10.main(["--results-dir", str(tmp_path / "nothing"), "--no-figures"]) == 1
