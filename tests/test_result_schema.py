"""Tests for result schema and run manifest module (Phase 4).

Covers MetricRecord validation, RunManifest lifecycle, deterministic checksums,
atomic serialization, and schema validation error handling.
"""

from __future__ import annotations

import json

import pytest

from st_dssm.result_schema import (
    MetricRecord,
    RunManifest,
    SchemaValidationError,
    load_run_manifest,
    save_run_manifest,
)


class TestMetricRecord:
    def test_valid_record_with_minutes_and_count(self):
        rec = MetricRecord(
            name="MAE",
            value=2.35,
            unit="mph",
            horizon=3,
            horizon_minutes=15,
            target_group="all",
            aggregation="mean",
            valid_count=1500,
        )
        assert rec.name == "MAE"
        assert rec.value == 2.35
        assert rec.horizon == 3
        assert rec.horizon_minutes == 15
        assert rec.valid_count == 1500

    def test_all_valid_metric_names(self):
        for name in ("MAE", "RMSE", "MAPE", "NLL", "CRPS", "PICP", "MPIW"):
            rec = MetricRecord(name=name, value=0.0, unit="raw", horizon="aggregate")
            assert rec.name == name

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown metric name"):
            MetricRecord(name="SMAPE_INVALID", value=0.0, unit="raw", horizon=1)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="unit must be"):
            MetricRecord(name="MAE", value=0.0, unit="kilometers_per_hour", horizon=1)

    def test_invalid_count_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            MetricRecord(name="MAE", value=0.0, unit="raw", horizon=1, valid_count=-5)


class TestRunManifest:
    def test_default_values_and_status(self):
        m = RunManifest(run_id="test-run-1", model_name="test_model")
        assert m.schema_version == "1.0"
        assert m.run_status == "incomplete"
        assert len(m.metrics) == 0

    def test_add_metric(self):
        m = RunManifest(run_id="test-run-1", model_name="test_model")
        rec = MetricRecord(name="RMSE", value=3.4, unit="mph", horizon="aggregate", valid_count=100)
        m.add_metric(rec)
        assert len(m.metrics) == 1
        assert m.metrics[0]["name"] == "RMSE"
        assert m.metrics[0]["value"] == 3.4

    def test_mark_complete(self):
        m = RunManifest(run_id="test-run-1", model_name="test_model")
        m.mark_complete()
        assert m.run_status == "complete"
        assert m.end_time_utc != ""

    def test_mark_failed(self):
        m = RunManifest(run_id="test-run-1", model_name="test_model")
        m.mark_failed("Numerical divergence")
        assert m.run_status == "failed"
        assert m.resolved_config.get("failure_reason") == "Numerical divergence"

    def test_checksum_computation_is_deterministic(self):
        m1 = RunManifest(run_id="test-run-1", model_name="model_a", training_seed=2026)
        m2 = RunManifest(run_id="test-run-1", model_name="model_a", training_seed=2026)
        c1 = m1.compute_checksum()
        c2 = m2.compute_checksum()
        assert len(c1) == 64  # sha256 hex string
        assert c1 == c2


class TestSaveLoadManifest:
    def test_atomic_save_and_load_round_trip(self, tmp_path):
        m = RunManifest(
            run_id="run-2026-phase4",
            model_name="evaluation_fixture",
            training_seed=2026,
            mask_condition="20%",
            git_commit="abc1234",
        )
        rec = MetricRecord(name="MAE", value=2.75, unit="mph", horizon="aggregate", valid_count=5000)
        m.add_metric(rec)
        m.mark_complete()

        path = str(tmp_path / "manifest.json")
        saved_path = save_run_manifest(m, path, overwrite=True, atomic=True)
        assert saved_path == path

        loaded = load_run_manifest(path)
        assert loaded.run_id == "run-2026-phase4"
        assert loaded.model_name == "evaluation_fixture"
        assert loaded.run_status == "complete"
        assert loaded.manifest_checksum != ""
        assert len(loaded.metrics) == 1
        assert loaded.metrics[0]["value"] == pytest.approx(2.75)

    def test_overwrite_protection(self, tmp_path):
        m = RunManifest(run_id="r1", model_name="m1")
        path = str(tmp_path / "protected_manifest.json")
        save_run_manifest(m, path)

        with pytest.raises(FileExistsError, match="already exists"):
            save_run_manifest(m, path, overwrite=False)

    def test_load_wrong_version_raises(self, tmp_path):
        path = str(tmp_path / "bad_version.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "99.0", "run_id": "r1", "model_name": "m1"}, f)

        with pytest.raises(SchemaValidationError, match="schema version"):
            load_run_manifest(path)

    def test_save_missing_model_name_raises(self, tmp_path):
        m = RunManifest(run_id="r1", model_name="")
        with pytest.raises(SchemaValidationError, match="model_name"):
            save_run_manifest(m, str(tmp_path / "invalid.json"))
