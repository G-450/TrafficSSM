"""Machine-readable result and run-manifest schemas for evaluation.

Every training or evaluation run produces a versioned JSON manifest with
complete metadata for independent reproduction (see
docs/QUALITY_AND_REPRODUCIBILITY.md and docs/DATA_CONTRACT.md).

Classes
-------
MetricRecord : a single metric measurement with full context.
RunManifest  : complete reproducibility and results record for one run.
EvaluationResult : alias for RunManifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricRecord:
    """A single metric measurement with full context.

    Attributes
    ----------
    name : metric identifier (e.g. 'MAE', 'RMSE', 'MAPE', 'NLL', 'CRPS', 'PICP', 'MPIW').
    value : scalar metric value (must be finite).
    unit : 'mph', 'raw', 'normalized', 'ratio', or 'percent'.
    horizon : forecast horizon step index (1-based integer) or 'aggregate'.
    horizon_minutes : forecast horizon in elapsed minutes (e.g. 5..60) or None.
    target_group : 'all', 'masked', or 'unmasked'.
    aggregation : how the value was computed (e.g. 'mean').
    valid_count : number of genuine observations evaluated.
    """

    name: str
    value: float
    unit: str
    horizon: str | int
    horizon_minutes: int | None = None
    target_group: str = "all"
    aggregation: str = "mean"
    valid_count: int = 0

    def __post_init__(self) -> None:
        valid_names = {"MAE", "RMSE", "MAPE", "NLL", "CRPS", "PICP", "MPIW"}
        if self.name not in valid_names:
            raise ValueError(
                f"Unknown metric name '{self.name}'. Valid: {valid_names}"
            )
        valid_units = {"raw", "normalized", "mph", "percent", "ratio"}
        if self.unit not in valid_units:
            raise ValueError(
                f"unit must be one of {valid_units}, got '{self.unit}'"
            )
        valid_groups = {"all", "masked", "unmasked"}
        if self.target_group not in valid_groups:
            raise ValueError(
                f"target_group must be one of {valid_groups}, got '{self.target_group}'"
            )
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"value must be numeric, got {type(self.value)}")
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite (not NaN or Inf), got {self.value}")
        if not isinstance(self.valid_count, int) or self.valid_count < 0:
            raise ValueError(f"valid_count must be a non-negative integer, got {self.valid_count}")


@dataclass
class RunManifest:
    """Complete reproducibility and evaluation record for one run.

    Follows docs/QUALITY_AND_REPRODUCIBILITY.md and ADR-0008 requirements.
    """

    # --- Identification ---
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""

    # --- Timing ---
    start_time_utc: str = ""
    end_time_utc: str = ""
    run_status: str = "incomplete"  # 'complete', 'incomplete', 'failed'

    # --- Code provenance ---
    git_commit: str = ""
    git_dirty: bool = False

    # --- Configuration ---
    config_hash: str = ""
    resolved_config: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)

    # --- Data & splits ---
    dataset_manifest_checksum: str = ""
    split_artifact_checksum: str = ""
    split_evaluated: str = "test"  # 'train', 'val', 'test', 'synthetic'
    sensor_count: int = 325
    sample_count: int = 0
    forecast_horizon: int = 12

    # --- Seeds ---
    training_seed: int | None = None
    mask_seed: int | None = None

    # --- Mask ---
    mask_condition: str = "0%"  # '0%', '10%', '20%', '30%'
    mask_sensor_ids: list[str] = field(default_factory=list)
    mask_checksum: str = ""

    # --- Model ---
    model_name: str = ""
    trainable_parameters: int = 0
    distribution_family: str = "gaussian"

    # --- Training & Checkpoints ---
    checkpoint_path: str = ""
    selection_metric: str = ""
    best_epoch: int = 0
    total_epochs: int = 0

    # --- Evaluation results ---
    metrics: list[dict[str, Any]] = field(default_factory=list)
    overall_metrics: dict[str, Any] = field(default_factory=dict)
    per_horizon_metrics: list[dict[str, Any]] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)

    # --- Integrity Checksum ---
    manifest_checksum: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {self.schema_version}. Expected: {SCHEMA_VERSION}"
            )
        valid_statuses = {"complete", "incomplete", "failed"}
        if self.run_status not in valid_statuses:
            raise ValueError(
                f"run_status must be one of {valid_statuses}, got '{self.run_status}'"
            )

    def add_metric(self, record: MetricRecord) -> None:
        """Append a validated MetricRecord to the manifest."""
        self.metrics.append(asdict(record))

    def mark_complete(self) -> None:
        """Set the run status to complete with current UTC timestamp."""
        self.end_time_utc = datetime.now(timezone.utc).isoformat()
        self.run_status = "complete"

    def mark_failed(self, reason: str = "") -> None:
        """Set the run status to failed."""
        self.end_time_utc = datetime.now(timezone.utc).isoformat()
        self.run_status = "failed"
        self.resolved_config["failure_reason"] = reason

    def compute_checksum(self) -> str:
        """Compute deterministic SHA256 checksum of manifest contents excluding the checksum field."""
        data = asdict(self)
        data["manifest_checksum"] = ""
        canonical_str = json.dumps(data, indent=2, sort_keys=True, default=str)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


EvaluationResult = RunManifest


# ---------------------------------------------------------------------------
# Serialization & Validation
# ---------------------------------------------------------------------------


class SchemaValidationError(Exception):
    """Raised when a manifest fails schema validation."""


_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "run_status",
    "model_name",
)


def save_run_manifest(
    manifest: RunManifest,
    path: str,
    overwrite: bool = True,
    atomic: bool = True,
) -> str:
    """Serialize a RunManifest to JSON with validation, atomic write, and checksum.

    Parameters
    ----------
    manifest : the manifest to save.
    path : output JSON file path.
    overwrite : if False, raises FileExistsError if file already exists.
    atomic : if True, writes via temporary file and renames atomically.

    Returns
    -------
    Path to saved manifest.

    Raises
    ------
    FileExistsError : if file exists and overwrite is False.
    SchemaValidationError : if required fields are missing or empty.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Manifest file already exists at {path} and overwrite=False.")

    manifest.manifest_checksum = manifest.compute_checksum()
    data = asdict(manifest)

    # Validate required fields are non-empty
    for field_name in _REQUIRED_FIELDS:
        val = data.get(field_name)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            raise SchemaValidationError(
                f"Required field '{field_name}' is missing or empty."
            )

    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)

    json_str = json.dumps(data, indent=2, sort_keys=True, default=str)

    if atomic:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=out_dir, delete=False, suffix=".tmp"
        ) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(json_str)
        try:
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

    return path


def load_run_manifest(path: str) -> RunManifest:
    """Deserialize a RunManifest from JSON with strict schema validation.

    Parameters
    ----------
    path : path to the JSON manifest file.

    Returns
    -------
    RunManifest instance.

    Raises
    ------
    SchemaValidationError : if the file is invalid or has wrong schema version.
    FileNotFoundError : if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Schema version check
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"Unsupported schema version: {data.get('schema_version')}. "
            f"Expected: {SCHEMA_VERSION}"
        )

    # Validate required fields
    for field_name in _REQUIRED_FIELDS:
        val = data.get(field_name)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            raise SchemaValidationError(
                f"Required field '{field_name}' is missing or empty in loaded manifest."
            )

    # Validate each metric record through MetricRecord instantiation
    raw_metrics = data.get("metrics", [])
    validated_metrics = []
    for i, m in enumerate(raw_metrics):
        try:
            rec = MetricRecord(**m)
            validated_metrics.append(asdict(rec))
        except Exception as err:
            raise SchemaValidationError(f"Invalid metric record at index {i}: {err}") from err

    data["metrics"] = validated_metrics

    # Construct RunManifest from dict, ignoring unknown keys
    known_fields = {f.name for f in RunManifest.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}

    return RunManifest(**filtered)
