"""Tests for Phase 13 — Calibration and robustness review.

Gate: ``scripts/aggregate_phase13.py --check`` re-derives the locked aggregate
from the prediction artifacts and run manifests and fails on any drift. This
suite validates the aggregator against a small synthetic study, and in
particular the ADR-0008 constraints that the temperature is fitted on the
validation split only and that calibration never moves a predictive mean.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from st_dssm.io import save_processed_artifact
from st_dssm.result_schema import MetricRecord, RunManifest, save_run_manifest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

N_SENSORS = 10
LENGTH = 12
WINDOWS = 8
SEEDS = (2026, 2027, 2028)
CONDITIONS = (("normal", None), ("10%", 0.1), ("20%", 0.2), ("30%", 0.3))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_artifact(artifact_dir: Path) -> None:
    """Write a small Phase-3-shaped artifact with val and test splits."""
    rng = np.random.default_rng(13)
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        shape = (WINDOWS, LENGTH, N_SENSORS, 1)
        arrays[f"{split}_X"] = rng.normal(size=shape).astype(np.float32)
        arrays[f"{split}_X_mask"] = np.ones(shape, dtype=bool)
        arrays[f"{split}_Y"] = rng.normal(size=shape).astype(np.float32)
        arrays[f"{split}_Y_mask"] = np.ones(shape, dtype=bool)
    arrays["scaler_means"] = np.full(N_SENSORS, 60.0, dtype=np.float32)
    arrays["scaler_stds"] = np.full(N_SENSORS, 5.0, dtype=np.float32)

    save_processed_artifact(
        str(artifact_dir),
        arrays,
        {"schema_version": "1.0", "sensor_ids": [f"s{i}" for i in range(N_SENSORS)]},
    )


def _write_run(
    results_dir: Path,
    *,
    split: str,
    seed: int,
    condition: str,
    ratio: float | None,
    masked_training: bool = False,
    sigma_scale: float = 1.0,
    run_status: str = "complete",
    write_predictions: bool = True,
    package_versions: dict[str, str] | None = None,
) -> Path:
    """Write a schema-valid run manifest beside a synthetic prediction artifact."""
    results_dir.mkdir(parents=True, exist_ok=True)

    if condition == "normal":
        run_id = f"st_dssm-{split}-seed{seed}-predict-2026"
    else:
        label = condition.rstrip("%")
        run_id = f"st_dssm-{split}-seed{seed}-node{label}-seed{seed}"
        run_id += "-masktrain-2026" if masked_training else "-predict-2026"

    masking = None
    if ratio is not None:
        masking = {
            "missing_ratio": ratio,
            "seed": seed,
            "apply_to_training": masked_training,
        }

    manifest = RunManifest(
        run_id=run_id,
        model_name="st_dssm",
        split_evaluated=split,
        sample_count=WINDOWS,
        sensor_count=N_SENSORS,
        forecast_horizon=LENGTH,
        training_seed=seed,
        mask_seed=seed if ratio is not None else None,
        mask_condition=condition if condition != "normal" else None,
        split_artifact_checksum="split-digest",
        dataset_manifest_checksum="dataset-digest",
        package_versions=package_versions or {"python": "3.12.10", "numpy": "1.26.4"},
        resolved_config={"masking": masking} if masking else {},
    )
    for step in range(1, LENGTH + 1):
        for group in ("all", "masked", "unmasked"):
            if group != "all" and ratio is None:
                continue
            manifest.add_metric(
                MetricRecord(
                    name="MAE",
                    value=1.5 + 0.05 * step,
                    unit="mph",
                    horizon=step,
                    horizon_minutes=step * 5,
                    target_group=group,
                )
            )
            # Coverage rises with the horizon, as it does in the real study.
            manifest.add_metric(
                MetricRecord(
                    name="PICP",
                    value=0.84 + 0.01 * step,
                    unit="ratio",
                    horizon=step,
                    horizon_minutes=step * 5,
                    target_group=group,
                )
            )
    manifest.add_metric(
        MetricRecord(name="MAE", value=1.8, unit="mph", horizon="aggregate", target_group="all")
    )
    if run_status == "complete":
        manifest.mark_complete()
    else:
        manifest.run_status = run_status

    path = results_dir / f"{run_id}_manifest.json"
    save_run_manifest(manifest, str(path))

    if write_predictions:
        rng = np.random.default_rng(seed + int((ratio or 0) * 100))
        shape = (WINDOWS, LENGTH, N_SENSORS, 1)
        np.savez_compressed(
            results_dir / f"{run_id}_predictions.npz",
            predictions=rng.normal(size=shape).astype(np.float32),
            sigma=np.full(shape, 0.5 * sigma_scale, dtype=np.float32),
        )
    return path


def _write_latency(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "study": "Phase 13 inference latency",
        "measured_at_utc": "2026-10-14T00:00:00+00:00",
        "protocol": {"batch_size": 64, "warmup_batches": 10, "timed_batches": 30},
        "environment": {
            "device": "cuda",
            "device_name": "synthetic",
            "torch": "2.12.1+cu130",
        },
        "models": [
            {
                "model_name": "ST-DSSM (ours)",
                "trainable_parameters": 397058,
                "per_window_median_ms": 1.6,
                "per_window_p95_ms": 1.7,
                "per_batch_median_ms": 103.0,
                "windows_per_second": 621.0,
            }
        ],
    }
    (results_dir / "latency_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def _study(tmp_path: Path, **overrides) -> dict[str, Path]:
    """Build a complete synthetic Phase 13 study on disk."""
    artifact = tmp_path / "processed"
    val = tmp_path / "calibration"
    normal = tmp_path / "normal"
    condition = tmp_path / "missingness"

    _write_artifact(artifact)
    _write_latency(val)

    for seed in SEEDS:
        _write_run(val, split="val", seed=seed, condition="normal", ratio=None)
        for cond, ratio in CONDITIONS:
            target = normal if cond == "normal" else condition
            _write_run(target, split="test", seed=seed, condition=cond, ratio=ratio)

    if overrides.get("with_ablation"):
        for seed in SEEDS:
            _write_run(
                condition,
                split="test",
                seed=seed,
                condition="20%",
                ratio=0.2,
                masked_training=True,
                sigma_scale=1.4,
            )

    return {
        "artifact": artifact,
        "val": val,
        "normal": normal,
        "condition": condition,
    }


def _args(dirs: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--artifact-dir", str(dirs["artifact"]),
        "--results-dir", str(dirs["val"]),
        "--val-dir", str(dirs["val"]),
        "--normal-dir", str(dirs["normal"]),
        "--condition-dir", str(dirs["condition"]),
        *extra,
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_produces_json_markdown_and_figures(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)

    assert agg.main(_args(dirs)) == 0

    json_path = dirs["val"] / "phase13_results.json"
    md_path = dirs["val"] / "phase13_results.md"
    assert json_path.exists()
    assert md_path.exists()
    assert list((dirs["val"] / "plots").glob("phase13_*.png"))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model"] == "st_dssm"
    assert sorted(payload["conditions"]) == ["10%", "20%", "30%", "normal"]
    # 3 validation runs + 4 conditions x 3 seeds.
    assert len(payload["sources"]) == 15
    assert all(src["prediction_checksum"] for src in payload["sources"])


def test_temperature_is_fitted_per_seed_on_validation_only(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    payload = json.loads((dirs["val"] / "phase13_results.json").read_text(encoding="utf-8"))
    assert sorted(payload["temperatures"]) == [str(s) for s in SEEDS]
    for seed in SEEDS:
        record = payload["temperatures"][str(seed)]
        assert record["temperature"] > 0.0
        assert "val" in record["run_id"]
        assert "validation_raw" in record and "validation_calibrated" in record

    # Every temperature source must be a validation run.
    val_sources = [s for s in payload["sources"] if s["kind"] == "validation"]
    assert len(val_sources) == 3
    assert all(s["split"] == "val" for s in val_sources)


def test_calibrated_arm_leaves_mae_untouched(tmp_path: Path) -> None:
    """ADR-0008: calibration may not alter predictive means."""
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    payload = json.loads((dirs["val"] / "phase13_results.json").read_text(encoding="utf-8"))
    for cond in payload["conditions"].values():
        for group in cond["per_group"].values():
            # MAE is stored once per group, outside both arms, precisely because
            # it cannot differ between them.
            assert "MAE" in group
            assert "MAE" not in group["raw"]
            assert "MAE" not in group["calibrated"]
            # MPIW must scale with the temperature, so the arms do differ.
            assert group["raw"]["MPIW"]["mean"] != group["calibrated"]["MPIW"]["mean"]


def test_calibrated_arm_improves_validation_nll(tmp_path: Path) -> None:
    """The fitted temperature is the NLL minimizer, so it cannot hurt val NLL."""
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    payload = json.loads((dirs["val"] / "phase13_results.json").read_text(encoding="utf-8"))
    for record in payload["temperatures"].values():
        assert (
            record["validation_calibrated"]["NLL"] <= record["validation_raw"]["NLL"] + 1e-12
        )


def test_reliability_and_pit_are_recorded_for_every_condition(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    payload = json.loads((dirs["val"] / "phase13_results.json").read_text(encoding="utf-8"))
    levels = payload["protocol"]["nominal_levels"]
    for cond in payload["conditions"].values():
        for arm in ("raw", "calibrated"):
            arm_data = cond["per_group"]["all"][arm]
            assert len(arm_data["reliability"]) == len(levels)
            assert [r["nominal_level"] for r in arm_data["reliability"]] == levels
            assert len(arm_data["pit_densities"]) == payload["protocol"]["pit_bins"]
            assert len(arm_data["pit_edges"]) == payload["protocol"]["pit_bins"] + 1
            assert arm_data["interval_ECE"]["mean"] >= 0.0
            assert 0.0 <= arm_data["PIT_ECE"]["mean"] <= 1.0


def test_masked_and_unmasked_groups_split_only_masked_conditions(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    payload = json.loads((dirs["val"] / "phase13_results.json").read_text(encoding="utf-8"))
    # The normal condition withholds nothing, so it has one group.
    assert sorted(payload["conditions"]["normal"]["per_group"]) == ["all"]
    for level in ("10%", "20%", "30%"):
        groups = payload["conditions"][level]["per_group"]
        assert sorted(groups) == ["all", "masked", "unmasked"]
        # The groups partition the valid targets of the 'all' group.
        assert (
            groups["masked"]["valid_count"] + groups["unmasked"]["valid_count"]
            == groups["all"]["valid_count"]
        )


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------


def test_check_passes_on_unchanged_artifacts(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0
    assert agg.main(_args(dirs, "--check")) == 0


def test_check_fails_when_a_prediction_artifact_changes(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    # Re-scale one run's sigma: the mean is untouched, so only a diagnostic that
    # actually reads the predictive distribution can notice.
    target = next(dirs["normal"].glob("*seed2026*_predictions.npz"))
    with np.load(target) as data:
        payload = {k: data[k] for k in data.files}
    payload["sigma"] = payload["sigma"] * 2.0
    np.savez_compressed(target, **payload)

    assert agg.main(_args(dirs, "--check")) == 1


def test_check_fails_on_hand_edited_results(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    json_path = dirs["val"] / "phase13_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["conditions"]["20%"]["per_group"]["all"]["raw"]["NLL"]["mean"] = 1.0
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert agg.main(_args(dirs, "--check")) == 1


def test_check_without_a_committed_aggregate_fails(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs, "--check")) == 1


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_missing_prediction_artifact_is_refused_with_guidance(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    next(dirs["condition"].glob("*node20*_predictions.npz")).unlink()

    with pytest.raises(agg.AggregationError, match="Prediction artifact missing"):
        agg.discover_runs(str(dirs["val"]), str(dirs["normal"]), str(dirs["condition"]))


def test_incomplete_seed_coverage_is_refused(tmp_path: Path) -> None:
    """A temperature must exist for every canonical seed (ADR-0008)."""
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    for path in dirs["val"].glob("*seed2028*"):
        path.unlink()

    with pytest.raises(agg.AggregationError, match="expected"):
        agg.discover_runs(str(dirs["val"]), str(dirs["normal"]), str(dirs["condition"]))


def test_missing_validation_runs_are_refused(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    for path in dirs["val"].glob("st_dssm-val-*"):
        path.unlink()

    with pytest.raises(agg.AggregationError, match="No validation runs"):
        agg.discover_runs(str(dirs["val"]), str(dirs["normal"]), str(dirs["condition"]))


def test_incomplete_run_is_refused(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    _write_run(
        dirs["condition"],
        split="test",
        seed=2026,
        condition="30%",
        ratio=0.3,
        run_status="incomplete",
    )

    with pytest.raises(agg.AggregationError, match="run_status"):
        agg.discover_runs(str(dirs["val"]), str(dirs["normal"]), str(dirs["condition"]))


def test_fitting_a_temperature_on_a_test_split_is_refused(tmp_path: Path) -> None:
    """ADR-0008 forbids a temperature that has seen the test split."""
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    runs = agg.discover_runs(str(dirs["val"]), str(dirs["normal"]), str(dirs["condition"]))
    targets = agg._Targets(str(dirs["artifact"]))

    leaked = [dict(run, split="test") for run in runs["validation"]]
    with pytest.raises(agg.AggregationError, match="validation split"):
        agg.fit_temperatures(leaked, targets)


def test_missing_condition_is_refused(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    for path in dirs["condition"].glob("*node30*"):
        path.unlink()

    assert agg.main(_args(dirs)) == 1


def test_missing_latency_record_is_refused(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    (dirs["val"] / "latency_record.json").unlink()

    with pytest.raises(agg.AggregationError, match="No latency record"):
        agg.load_latency(str(dirs["val"]))


# ---------------------------------------------------------------------------
# Ablation arm and provenance
# ---------------------------------------------------------------------------


def test_ablation_is_reported_separately_from_primary_cells(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path, with_ablation=True)
    assert agg.main(_args(dirs)) == 0

    payload = json.loads((dirs["val"] / "phase13_results.json").read_text(encoding="utf-8"))
    ablation = payload["ablations"]["masked_training"]
    assert sorted(ablation) == ["20%"]
    # The primary 20% cell must still average exactly three runs.
    assert payload["conditions"]["20%"]["seeds"] == list(SEEDS)
    assert ablation["20%"]["seeds"] == list(SEEDS)
    # The ablation's wider sigma must not have leaked into the primary cell.
    assert (
        payload["conditions"]["20%"]["per_group"]["all"]["raw"]["MPIW"]["mean"]
        != ablation["20%"]["per_group"]["all"]["raw"]["MPIW"]["mean"]
    )
    assert len(payload["sources"]) == 18


def test_ablation_section_renders_in_markdown(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path, with_ablation=True)
    assert agg.main(_args(dirs)) == 0

    markdown = (dirs["val"] / "phase13_results.md").read_text(encoding="utf-8")
    assert "Masked-training ablation" in markdown
    assert "ADR-0011" in markdown


def test_every_distinct_run_environment_is_recorded(tmp_path: Path) -> None:
    """A study spanning two installed stacks must not claim one environment."""
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    _write_run(
        dirs["condition"],
        split="test",
        seed=2026,
        condition="20%",
        ratio=0.2,
        masked_training=True,
        package_versions={"python": "3.12.10", "numpy": "2.5.2"},
    )
    assert agg.main(_args(dirs)) == 0

    payload = json.loads((dirs["val"] / "phase13_results.json").read_text(encoding="utf-8"))
    environments = payload["run_environments"]
    assert len(environments) == 2
    assert sum(env["run_count"] for env in environments) == len(payload["sources"])

    markdown = (dirs["val"] / "phase13_results.md").read_text(encoding="utf-8")
    assert "Run environments:" in markdown


def test_markdown_reports_both_arms_and_the_latency_record(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    markdown = (dirs["val"] / "phase13_results.md").read_text(encoding="utf-8")
    assert "Temperature scaling (secondary analysis, ADR-0008)" in markdown
    assert "| all | raw |" in markdown
    assert "| all | calibrated |" in markdown
    assert "Inference latency" in markdown
    assert "ST-DSSM (ours)" in markdown
    for level in ("normal", "10%", "20%", "30%"):
        assert level in markdown


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _runner(tmp_path: Path, monkeypatch):
    """Load run_phase13 pointed at a temporary checkpoint and output tree."""
    runner = _load_script("run_phase13")
    monkeypatch.setattr(runner, "OUTPUT_DIR", str(tmp_path / "calibration"))
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for seed in SEEDS:
        (ckpt_dir / f"st_dssm-test-seed{seed}-20260101-000000_best.pt").write_bytes(b"")
    return runner


def _capture(fn) -> str:
    import io
    import sys as _sys

    buffer = io.StringIO()
    old = _sys.stdout
    _sys.stdout = buffer
    try:
        fn()
    finally:
        _sys.stdout = old
    return buffer.getvalue()


def test_run_phase13_dry_run_plans_one_pass_per_seed(tmp_path: Path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    output = _capture(lambda: runner.main(["--dry-run"]))

    dry_lines = [line for line in output.splitlines() if "[dry]" in line]
    # Three validation passes plus the two latency measurements.
    assert len(dry_lines) == 5, output
    assert output.count("--split val") == 3
    assert "--predict-from" in output


def test_run_phase13_passes_each_seed_its_own_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """A seed's temperature may only be fitted on that seed's own predictions."""
    runner = _runner(tmp_path, monkeypatch)
    for run in runner.build_run_list(list(SEEDS)):
        checkpoint = runner._find_checkpoint(run["seed"])
        assert f"seed{run['seed']}" in checkpoint
        assert str(run["seed"]) in run["cmd_template"]


def test_run_phase13_skips_completed_validation_runs(tmp_path: Path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    out_dir = tmp_path / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "st_dssm-val-seed2026-predict-20261014-000000_manifest.json").write_text(
        "{}", encoding="utf-8"
    )

    output = _capture(lambda: runner.main(["--dry-run", "--skip-latency"]))
    assert "[skip]  st_dssm/val/seed2026" in output
    assert len([line for line in output.splitlines() if "[dry]" in line]) == 2


def test_run_phase13_reports_a_missing_checkpoint(tmp_path: Path, monkeypatch) -> None:
    runner = _load_script("run_phase13")
    monkeypatch.setattr(runner, "OUTPUT_DIR", str(tmp_path / "calibration"))
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()

    assert runner.main(["--dry-run", "--skip-latency"]) == 1


# ---------------------------------------------------------------------------
# Coverage-decay section and --render-only
# ---------------------------------------------------------------------------


def test_coverage_decay_section_reports_every_group(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    markdown = (dirs["val"] / "phase13_results.md").read_text(encoding="utf-8")
    assert "### Coverage decay across the horizon" in markdown
    # The normal condition has one group; a masked condition has three.
    assert "| normal | all |" in markdown
    assert "| 20% | masked |" in markdown
    assert "| 20% | unmasked |" in markdown


def test_render_only_rebuilds_from_the_committed_aggregate(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    md_path = dirs["val"] / "phase13_results.md"
    json_path = dirs["val"] / "phase13_results.json"
    before = md_path.read_text(encoding="utf-8")
    md_path.unlink()

    assert agg.main(_args(dirs, "--render-only")) == 0
    after = md_path.read_text(encoding="utf-8")

    # Identical but for the generation timestamp line.
    strip = lambda text: [
        line for line in text.splitlines() if not line.startswith("**Generated:**")
    ]
    assert strip(after) == strip(before)
    # The locked aggregate must be untouched by a render.
    assert json.loads(json_path.read_text(encoding="utf-8"))["model"] == "st_dssm"


def test_render_only_does_not_recompute_or_detect_drift(tmp_path: Path) -> None:
    """A render reads the committed JSON, so it cannot notice changed artifacts."""
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0

    target = next(dirs["normal"].glob("*seed2026*_predictions.npz"))
    with np.load(target) as data:
        payload = {k: data[k] for k in data.files}
    payload["sigma"] = payload["sigma"] * 3.0
    np.savez_compressed(target, **payload)

    # A render still succeeds against the stale lock; only --check fails.
    assert agg.main(_args(dirs, "--render-only")) == 0
    assert agg.main(_args(dirs, "--check")) == 1


def test_render_only_refuses_to_be_combined_with_check(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs)) == 0
    assert agg.main(_args(dirs, "--render-only", "--check")) == 2


def test_render_only_without_a_committed_aggregate_fails(tmp_path: Path) -> None:
    agg = _load_script("aggregate_phase13")
    dirs = _study(tmp_path)
    assert agg.main(_args(dirs, "--render-only")) == 1
