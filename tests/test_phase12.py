"""Tests for Phase 12 — Robustness condition study.

Gate: ``scripts/aggregate_phase12.py --check`` re-derives the locked aggregate
from the condition manifests and fails on any drift; this test suite validates
the aggregator's logic against minimal synthetic fixtures.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from st_dssm.result_schema import MetricRecord, RunManifest, save_run_manifest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_condition_run(
    results_dir: Path,
    model_name: str,
    mask_condition: str,
    mask_seed: int | None,
    training_seed: int | None,
    mae_all: float,
    mae_masked: float,
    mae_unmasked: float,
    *,
    with_uncertainty: bool = False,
) -> Path:
    """Write a minimal schema-valid condition run manifest."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ratio_label = mask_condition.replace("%", "")
    run_id = f"{model_name}-test-seed{training_seed}-node{ratio_label}-seed{mask_seed}"
    manifest = RunManifest(
        run_id=run_id,
        model_name=model_name,
        split_evaluated="test",
        sample_count=10403,
        sensor_count=325,
        forecast_horizon=12,
        training_seed=training_seed,
        mask_seed=mask_seed,
        mask_condition=mask_condition,
        split_artifact_checksum="split-digest",
        dataset_manifest_checksum="dataset-digest",
        package_versions={"python": "3.12.10", "numpy": "1.26.4"},
    )

    # Add aggregate metrics for each target group.
    for grp, mae in [("all", mae_all), ("masked", mae_masked), ("unmasked", mae_unmasked)]:
        manifest.add_metric(MetricRecord(
            name="MAE", value=mae, unit="mph",
            horizon="aggregate", target_group=grp,
        ))
        manifest.add_metric(MetricRecord(
            name="RMSE", value=mae * 2.0, unit="mph",
            horizon="aggregate", target_group=grp,
        ))
        manifest.add_metric(MetricRecord(
            name="MAPE", value=mae * 2.3, unit="percent",
            horizon="aggregate", target_group=grp,
        ))
        if with_uncertainty:
            manifest.add_metric(MetricRecord(
                name="NLL", value=2.1, unit="raw",
                horizon="aggregate", target_group=grp,
            ))
            manifest.add_metric(MetricRecord(
                name="CRPS", value=1.43, unit="mph",
                horizon="aggregate", target_group=grp,
            ))
            manifest.add_metric(MetricRecord(
                name="PICP", value=0.96, unit="ratio",
                horizon="aggregate", target_group=grp,
            ))
            manifest.add_metric(MetricRecord(
                name="MPIW", value=11.0, unit="mph",
                horizon="aggregate", target_group=grp,
            ))

    # Add per-horizon metrics for the 'all' group.
    for step in range(1, 13):
        manifest.add_metric(MetricRecord(
            name="MAE", value=mae_all + 0.05 * step, unit="mph",
            horizon=step, horizon_minutes=step * 5, target_group="all",
        ))
        manifest.add_metric(MetricRecord(
            name="RMSE", value=(mae_all + 0.05 * step) * 2.0, unit="mph",
            horizon=step, horizon_minutes=step * 5, target_group="all",
        ))
        manifest.add_metric(MetricRecord(
            name="MAPE", value=(mae_all + 0.05 * step) * 2.3, unit="percent",
            horizon=step, horizon_minutes=step * 5, target_group="all",
        ))

    manifest.overall_metrics = {"MAE": mae_all, "RMSE": mae_all * 2.0, "MAPE": mae_all * 2.3, "valid_count": 40569096}
    manifest.per_horizon_metrics = [
        {"horizon_step": s, "horizon_minutes": s * 5, "MAE": mae_all + 0.05 * s}
        for s in range(1, 13)
    ]
    manifest.mark_complete()

    path = results_dir / f"{run_id}_manifest.json"
    save_run_manifest(manifest, str(path))
    return path


def _complete_condition_study(tmp_path: Path) -> Path:
    """Write a full 9-condition × 3-model (27-run) synthetic study."""
    results = tmp_path / "missingness"
    for seed in (2026, 2027, 2028):
        for ratio, level in [(10, "10%"), (20, "20%"), (30, "30%")]:
            # ST-DSSM (probabilistic)
            _write_condition_run(
                results, "st_dssm", level, seed, seed,
                mae_all=2.0 + ratio * 0.01 + seed * 0.0001,
                mae_masked=2.5 + ratio * 0.02,
                mae_unmasked=1.9 + ratio * 0.005,
                with_uncertainty=True,
            )
            # ST-GCN (point only)
            _write_condition_run(
                results, "deterministic_st_gcn", level, seed, seed,
                mae_all=1.9 + ratio * 0.01 + seed * 0.0001,
                mae_masked=2.4 + ratio * 0.02,
                mae_unmasked=1.8 + ratio * 0.005,
            )
            # Persistence (no training seed)
            _write_condition_run(
                results, "historical_persistence", level, seed, None,
                mae_all=2.3 + ratio * 0.01,
                mae_masked=3.0 + ratio * 0.02,
                mae_unmasked=2.1 + ratio * 0.005,
            )
    return results


# ---------------------------------------------------------------------------
# Aggregate script tests
# ---------------------------------------------------------------------------


def test_aggregate_produces_json_and_markdown(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)

    assert agg.main(["--results-dir", str(results)]) == 0

    json_path = results / "phase12_results.json"
    md_path = results / "phase12_results.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert sorted(payload["models"]) == [
        "deterministic_st_gcn",
        "historical_persistence",
        "st_dssm",
    ]
    assert len(payload["sources"]) == 27
    assert all(src["manifest_checksum"] for src in payload["sources"])


def test_aggregate_condition_levels_are_present(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    assert agg.main(["--results-dir", str(results)]) == 0

    payload = json.loads((results / "phase12_results.json").read_text(encoding="utf-8"))
    st_dssm = payload["models"]["st_dssm"]["conditions"]
    assert sorted(st_dssm) == ["10%", "20%", "30%"]
    for level in ("10%", "20%", "30%"):
        cond = st_dssm[level]
        assert sorted(cond["seeds"]) == [2026, 2027, 2028]
        assert "all" in cond["per_group"]
        assert "masked" in cond["per_group"]
        assert "unmasked" in cond["per_group"]
        assert "MAE" in cond["per_group"]["all"]
        assert "mean" in cond["per_group"]["all"]["MAE"]


def test_aggregate_mean_std_across_seeds(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    assert agg.main(["--results-dir", str(results)]) == 0

    payload = json.loads((results / "phase12_results.json").read_text(encoding="utf-8"))
    mae_stat = payload["models"]["st_dssm"]["conditions"]["20%"]["per_group"]["all"]["MAE"]
    assert mae_stat["n"] == 3
    assert mae_stat["std"] > 0


def test_aggregate_check_passes_on_unchanged_manifests(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    assert agg.main(["--results-dir", str(results)]) == 0
    assert agg.main(["--results-dir", str(results), "--check"]) == 0


def test_aggregate_check_fails_on_drift(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    assert agg.main(["--results-dir", str(results)]) == 0

    json_path = results / "phase12_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    # Corrupt one value in the locked file.
    payload["models"]["st_dssm"]["conditions"]["20%"]["per_group"]["all"]["MAE"]["mean"] += 99.0
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert agg.main(["--results-dir", str(results), "--check"]) != 0


def test_aggregate_rejects_duplicate_run_for_same_condition_and_seed(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    # Write a second manifest for the same (model, condition, seed) combination but
    # under a different filename (simulating a re-run that produces a new timestamp).
    import json as _json, dataclasses
    from st_dssm.result_schema import RunManifest
    dup = RunManifest(
        run_id="st_dssm-test-seed2026-node20-seed2026-predict-99999999-999999",
        model_name="st_dssm",
        split_evaluated="test",
        sample_count=10403,
        sensor_count=325,
        forecast_horizon=12,
        mask_seed=2026,
        training_seed=2026,
        mask_condition="20%",
        split_artifact_checksum="split-digest",
        dataset_manifest_checksum="dataset-digest",
        package_versions={"python": "3.12.10"},
    )
    dup.mark_complete()
    raw = dataclasses.asdict(dup)
    dup_path = results / "st_dssm-test-seed2026-node20-seed2026-predict-99999999-999999_manifest.json"
    dup_path.write_text(_json.dumps(raw), encoding="utf-8")
    assert agg.main(["--results-dir", str(results)]) != 0


def test_aggregate_rejects_normal_condition_manifests(tmp_path: Path) -> None:
    """Normal-condition (0%) runs belong in aggregate_phase10, not here."""
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    _write_condition_run(
        results, "historical_persistence", "0%", 2026, None,
        mae_all=2.17, mae_masked=2.17, mae_unmasked=2.17,
    )
    assert agg.main(["--results-dir", str(results)]) != 0


def test_aggregate_rejects_incomplete_run(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    # Write a manifest with run_status != 'complete'.
    manifest = RunManifest(
        run_id="st_dssm-test-seed2026-node20-seed2026-incomplete",
        model_name="st_dssm",
        split_evaluated="test",
        sample_count=10403,
        sensor_count=325,
        forecast_horizon=12,
        mask_seed=2026,
        mask_condition="20%",
        split_artifact_checksum="split-digest",
        dataset_manifest_checksum="dataset-digest",
        package_versions={"python": "3.12.10"},
    )
    # Do not call mark_complete() — leave run_status as 'incomplete'.
    path = results / "st_dssm-test-seed2026-node20-seed2026-incomplete_manifest.json"
    # Force-write by patching run_status after construction.
    import json as _json, dataclasses
    raw = dataclasses.asdict(manifest)
    raw["run_status"] = "incomplete"
    path.write_text(_json.dumps(raw), encoding="utf-8")

    assert agg.main(["--results-dir", str(results)]) != 0


def test_aggregate_markdown_contains_all_condition_levels(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase12")
    results = _complete_condition_study(tmp_path)
    assert agg.main(["--results-dir", str(results)]) == 0

    tables = (results / "phase12_results.md").read_text(encoding="utf-8")
    assert "10%" in tables
    assert "20%" in tables
    assert "30%" in tables
    assert "masked" in tables.lower()
    assert "unmasked" in tables.lower()
    assert "dataset-digest" in tables


def test_run_phase12_dry_run_lists_all_conditions(tmp_path: Path, monkeypatch) -> None:
    """Dry run should report 27 planned executions (9 × 3 models)."""
    run12 = _load_script("run_phase12")
    monkeypatch.setattr(run12, "OUTPUT_DIR", str(tmp_path / "missingness"))
    monkeypatch.setattr(run12, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(run12, "CONFIG_DIR", str(tmp_path / "configs"))

    # Create fake config and checkpoint files so the script can resolve them.
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    import yaml
    for seed in (2026, 2027, 2028):
        for ratio in (10, 20, 30):
            cfg = cfg_dir / f"node{ratio}-seed{seed}.yaml"
            cfg.write_text(yaml.dump({"seed": seed, "masking": {"missing_ratio": ratio / 100}}))
        (ckpt_dir / f"baseline-st_gcn-test-seed{seed}-20260101-000000_best.pt").write_bytes(b"")
        (ckpt_dir / f"st_dssm-test-seed{seed}-20260101-000000_best.pt").write_bytes(b"")

    import io
    captured = io.StringIO()
    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        run12.main(["--dry-run"])
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    # 9 persistence + 9 st_gcn + 9 st_dssm = 27 planned runs (all shown as [dry] or [skip]).
    dry_lines = [l for l in output.splitlines() if "[dry]" in l]
    assert len(dry_lines) == 27, f"Expected 27 dry-run lines, got {len(dry_lines)}:\n{output}"
