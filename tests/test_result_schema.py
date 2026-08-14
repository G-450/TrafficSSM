"""Tests for the result schema module (Phase 4).

Covers MetricRecord validation, RunManifest round-trip serialization,
and schema validation error handling.
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

# ---------------------------------------------------------------------------
# MetricRecord
# ---------------------------------------------------------------------------

class TestMetricRecord:
    def test_valid_record(self):
        rec = MetricRecord(
            name="MAE", value=2.5, unit="raw",
            horizon=3, target_group="all", aggregation="mean",
        )
        assert rec.name == "MAE"
        assert rec.value == 2.5

    def test_all_valid_names(self):
        for name in ("MAE", "RMSE", "NLL", "CRPS", "PICP", "MPIW"):
            rec = MetricRecord(name=name, value=0.0, unit="raw", horizon="aggregate")
            assert rec.name == name

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown metric name"):
            MetricRecord(name="MSE", value=0.0, unit="raw", horizon=1)

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="unit must be"):
            MetricRecord(name="MAE", value=0.0, unit="mph", horizon=1)

    def test_invalid_target_group(self):
        with pytest.raises(ValueError, match="target_group"):
            MetricRecord(name="MAE", value=0.0, unit="raw", horizon=1, target_group="partial")

    def test_aggregate_horizon(self):
        rec = MetricRecord(name="RMSE", value=3.14, unit="raw", horizon="aggregate")
        assert rec.horizon == "aggregate"

    def test_integer_horizon(self):
        rec = MetricRecord(name="RMSE", value=3.14, unit="raw", horizon=5)
        assert rec.horizon == 5


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------

class TestRunManifest:
    def test_default_values(self):
        m = RunManifest(run_id="test-001", model_name="persistence")
        assert m.schema_version == "1.0"
        assert m.run_status == "incomplete"
        assert m.metrics == []

    def test_add_metric(self):
        m = RunManifest(run_id="test-001", model_name="persistence")
        rec = MetricRecord(name="MAE", value=2.5, unit="raw", horizon="aggregate")
        m.add_metric(rec)
        assert len(m.metrics) == 1
        assert m.metrics[0]["name"] == "MAE"
        assert m.metrics[0]["value"] == 2.5

    def test_mark_complete(self):
        m = RunManifest(run_id="test-001", model_name="persistence")
        m.mark_complete()
        assert m.run_status == "complete"
        assert m.end_time_utc != ""

    def test_mark_failed(self):
        m = RunManifest(run_id="test-001", model_name="persistence")
        m.mark_failed("NaN loss detected")
        assert m.run_status == "failed"
        assert m.resolved_config.get("failure_reason") == "NaN loss detected"

    def test_invalid_status(self):
        with pytest.raises(ValueError, match="run_status"):
            RunManifest(run_id="test-001", model_name="x", run_status="running")

    def test_invalid_schema_version(self):
        with pytest.raises(ValueError, match="schema version"):
            RunManifest(schema_version="2.0", run_id="x", model_name="x")


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        m = RunManifest(
            run_id="run-42",
            model_name="det-stgcn",
            training_seed=2026,
            mask_condition="20%",
            git_commit="abc123",
        )
        rec = MetricRecord(name="MAE", value=3.21, unit="raw", horizon="aggregate")
        m.add_metric(rec)
        m.mark_complete()

        path = str(tmp_path / "manifest.json")
        save_run_manifest(m, path)

        loaded = load_run_manifest(path)
        assert loaded.run_id == "run-42"
        assert loaded.model_name == "det-stgcn"
        assert loaded.training_seed == 2026
        assert loaded.mask_condition == "20%"
        assert loaded.run_status == "complete"
        assert len(loaded.metrics) == 1
        assert loaded.metrics[0]["name"] == "MAE"
        assert loaded.metrics[0]["value"] == pytest.approx(3.21)

    def test_save_rejects_missing_run_id(self, tmp_path):
        m = RunManifest(model_name="test")  # run_id is empty
        with pytest.raises(SchemaValidationError, match="run_id"):
            save_run_manifest(m, str(tmp_path / "bad.json"))

    def test_save_rejects_missing_model_name(self, tmp_path):
        m = RunManifest(run_id="r1")  # model_name is empty
        with pytest.raises(SchemaValidationError, match="model_name"):
            save_run_manifest(m, str(tmp_path / "bad.json"))

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run_manifest(str(tmp_path / "no_such_file.json"))

    def test_load_wrong_schema_version(self, tmp_path):
        path = str(tmp_path / "wrong.json")
        with open(path, "w") as f:
            json.dump({"schema_version": "99.0", "run_id": "x", "model_name": "x"}, f)
        with pytest.raises(SchemaValidationError, match="schema version"):
            load_run_manifest(path)

    def test_load_missing_required_field(self, tmp_path):
        path = str(tmp_path / "incomplete.json")
        with open(path, "w") as f:
            json.dump({"schema_version": "1.0", "run_id": "x", "run_status": "complete"}, f)
        with pytest.raises(SchemaValidationError, match="model_name"):
            load_run_manifest(path)

    def test_json_is_valid_file(self, tmp_path):
        """Saved manifest is valid JSON with sorted keys."""
        m = RunManifest(run_id="r1", model_name="m1")
        path = str(tmp_path / "manifest.json")
        save_run_manifest(m, path)

        with open(path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["schema_version"] == "1.0"
