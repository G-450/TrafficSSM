"""Machine-readable result and run-manifest schemas.

Every training or evaluation run must produce a JSON manifest with
sufficient metadata for independent reproduction (see
docs/QUALITY_AND_REPRODUCIBILITY.md).

Classes
-------
MetricRecord : a single metric measurement with context.
RunManifest  : complete reproducibility record for one run.
"""

from __future__ import annotations

import json
import os
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
    name : metric identifier (e.g. 'MAE', 'RMSE', 'NLL', 'CRPS', 'PICP', 'MPIW').
    value : scalar metric value.
    unit : 'raw' (traffic units) or 'normalized'.
    horizon : forecast horizon index (1-based) or 'aggregate'.
    target_group : 'all', 'masked', or 'unmasked'.
    aggregation : how the value was computed (e.g. 'mean', 'median').
    """

    name: str
    value: float
    unit: str
    horizon: str | int
    target_group: str = "all"
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        valid_names = {"MAE", "RMSE", "NLL", "CRPS", "PICP", "MPIW"}
        if self.name not in valid_names:
            raise ValueError(
                f"Unknown metric name '{self.name}'. Valid: {valid_names}"
            )
        if self.unit not in ("raw", "normalized"):
            raise ValueError(
                f"unit must be 'raw' or 'normalized', got '{self.unit}'"
            )
        if self.target_group not in ("all", "masked", "unmasked"):
            raise ValueError(
                f"target_group must be 'all', 'masked', or 'unmasked', "
                f"got '{self.target_group}'"
            )
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"value must be numeric, got {type(self.value)}")


@dataclass
class RunManifest:
    """Complete reproducibility record for one training or evaluation run.

    Follows docs/QUALITY_AND_REPRODUCIBILITY.md requirements.
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

    # --- Data ---
    dataset_manifest_checksum: str = ""
    split_artifact_checksum: str = ""

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

    # --- Training ---
    checkpoint_path: str = ""
    selection_metric: str = ""
    best_epoch: int = 0
    total_epochs: int = 0

    # --- Results ---
    metrics: list[dict[str, Any]] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {self.schema_version}"
            )
        valid_statuses = {"complete", "incomplete", "failed"}
        if self.run_status not in valid_statuses:
            raise ValueError(
                f"run_status must be one of {valid_statuses}, "
                f"got '{self.run_status}'"
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


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """Raised when a manifest fails schema validation."""


_REQUIRED_FIELDS = (
    "schema_version", "run_id", "run_status", "model_name",
)


def save_run_manifest(manifest: RunManifest, path: str) -> None:
    """Serialize a RunManifest to JSON with validation.

    Parameters
    ----------
    manifest : the manifest to save.
    path : output JSON file path.

    Raises
    ------
    SchemaValidationError : if required fields are missing or empty.
    """
    data = asdict(manifest)

    # Validate required fields are non-empty
    for field_name in _REQUIRED_FIELDS:
        val = data.get(field_name)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            raise SchemaValidationError(
                f"Required field '{field_name}' is missing or empty."
            )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def load_run_manifest(path: str) -> RunManifest:
    """Deserialize a RunManifest from JSON with schema validation.

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

    # Construct RunManifest from dict, ignoring unknown keys
    known_fields = {f.name for f in RunManifest.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}

    return RunManifest(**filtered)
