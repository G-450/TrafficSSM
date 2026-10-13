"""Phase 11 — the missingness mechanism wired through training and evaluation.

The Phase 11 gate has two halves, and this module tests both:

* **No target leakage.** A withheld sensor's history must not reach the model in
  any form. The decisive test is invariance: corrupt the withheld sensors' input
  values and the predictions must be bit-identical, which can only hold if those
  values never entered a forward pass. Targets are never masked — the evaluator
  scores every sensor (ADR-0004) — so this module also pins that a condition run
  leaves the target arrays and the all-group valid counts untouched.
* **Masks are reproducible and fair.** The same ratio and seed must produce the
  same withheld sensors for every model, the published mask artifacts must
  re-derive from their seeds, and each run manifest must record which sensors it
  withheld.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from st_dssm.cli.baseline import run_persistence_baseline
from st_dssm.cli.evaluate import main as evaluate_main
from st_dssm.cli.train import train_st_dssm
from st_dssm.io import load_and_validate_artifact, save_processed_artifact
from st_dssm.mask_generator import MaskGenerator, resolve_mask_generator
from st_dssm.result_schema import load_run_manifest

N_SENSORS = 10
LENGTH = 12
MASK_RATIO = 0.3  # 3 of 10 sensors withheld
RUN_SEED = 7
MASKS_DIR = Path("experiments/masks")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_artifact(artifact_dir: Path, corrupt_nodes: np.ndarray | None = None) -> list[str]:
    """Write a small Phase-3-shaped artifact; optionally poison some sensors' inputs.

    ``corrupt_nodes`` replaces those sensors' input values with a value no
    genuine normalized observation takes. If any of it reaches the model, a
    prediction changes.
    """
    rng = np.random.default_rng(0)
    sensor_ids = [f"s{i}" for i in range(N_SENSORS)]
    arrays: dict[str, np.ndarray] = {}
    for split, count in [("train", 16), ("val", 8), ("test", 8)]:
        shape = (count, LENGTH, N_SENSORS, 1)
        arrays[f"{split}_X"] = rng.normal(size=shape).astype(np.float32)
        arrays[f"{split}_X_mask"] = np.ones(shape, dtype=bool)
        arrays[f"{split}_Y"] = rng.normal(size=shape).astype(np.float32)
        arrays[f"{split}_Y_mask"] = np.ones(shape, dtype=bool)
    arrays["scaler_means"] = np.full(N_SENSORS, 60.0, dtype=np.float32)
    arrays["scaler_stds"] = np.full(N_SENSORS, 5.0, dtype=np.float32)

    if corrupt_nodes is not None:
        for split in ("train", "val", "test"):
            arrays[f"{split}_X"][:, :, corrupt_nodes, :] = 999.0

    save_processed_artifact(
        str(artifact_dir),
        arrays,
        {"schema_version": "1.0", "sensor_ids": sensor_ids},
    )
    return sensor_ids


def _training_config(tmp_path: Path, artifact_dir: Path, mask_ratio: float | None) -> dict:
    """A CPU training config over the small artifact, optionally under a condition."""
    sensor_ids = [f"s{i}" for i in range(N_SENSORS)]
    adj = np.eye(N_SENSORS, dtype=np.float32)
    for i in range(N_SENSORS - 1):
        adj[i, i + 1] = adj[i + 1, i] = 0.5
    adj_path = tmp_path / "adj.pkl"
    with open(adj_path, "wb") as f:
        pickle.dump((sensor_ids, {}, adj), f)

    config: dict = {
        "seed": RUN_SEED,
        "device": "cpu",
        "artifact_dir": str(artifact_dir),
        "adj_mx_path": str(adj_path),
        "allow_unverified_graph": True,
        "split": "test",
        "model": {"latent_dim": 4, "context_dim": 8, "hidden_dim": 8, "dropout": 0.0},
        "training": {"max_epochs": 1, "batch_size": 8},
    }
    if mask_ratio is not None:
        config["masking"] = {"missing_ratio": mask_ratio, "seed": RUN_SEED}
    return config


def _predictions_of(output_dir: Path, run_id: str) -> np.ndarray:
    with np.load(output_dir / f"{run_id}_predictions.npz") as payload:
        return payload["predictions"]


def _records(manifest, group: str, horizon="aggregate") -> dict[str, dict]:
    return {
        record["name"]: record
        for record in manifest.metrics
        if record["target_group"] == group and record["horizon"] == horizon
    }


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------


def test_withheld_sensors_are_zeroed_at_every_input_step() -> None:
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)
    withheld = np.flatnonzero(~generator.node_mask)
    x = np.full((4, LENGTH, N_SENSORS, 1), 7.5, dtype=np.float32)
    native = np.ones_like(x, dtype=bool)

    x_masked, combined = generator.apply_mask(x, native)

    # Every input step of a withheld sensor, not just the last one (ADR-0004).
    assert np.all(x_masked[:, :, withheld, :] == 0.0)
    assert not combined[:, :, withheld, :].any()
    # Observed sensors are untouched and still flagged observed.
    observed = np.flatnonzero(generator.node_mask)
    assert np.all(x_masked[:, :, observed, :] == 7.5)
    assert combined[:, :, observed, :].all()


def test_apply_mask_in_place_matches_the_copying_path() -> None:
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, LENGTH, N_SENSORS, 1)).astype(np.float32)
    native = rng.random((3, LENGTH, N_SENSORS, 1)) > 0.1

    copied_x, copied_mask = generator.apply_mask(x.copy(), native.copy())
    inplace_x, inplace_mask = generator.apply_mask(x, native, copy=False)

    assert np.array_equal(copied_x, inplace_x)
    assert np.array_equal(copied_mask, inplace_mask)
    assert inplace_x is x  # the caller's array was reused, not duplicated


def test_apply_mask_preserves_a_binary_float_mask_dtype() -> None:
    """Artifacts store boolean masks, but a binary float mask means the same thing."""
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)
    x = np.ones((2, LENGTH, N_SENSORS, 1), dtype=np.float32)
    native = np.ones((2, LENGTH, N_SENSORS, 1), dtype=np.float32)

    x_masked, combined = generator.apply_mask(x, native)

    assert combined.dtype == np.float32
    assert set(np.unique(combined)).issubset({0.0, 1.0})
    withheld = np.flatnonzero(~generator.node_mask)
    assert np.all(combined[:, :, withheld, :] == 0.0)
    assert np.all(x_masked[:, :, withheld, :] == 0.0)


def test_target_groups_partition_the_observed_targets() -> None:
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)
    rng = np.random.default_rng(2)
    shape = (5, LENGTH, N_SENSORS, 1)
    native = rng.random(shape) > 0.1

    groups = generator.node_group_masks(native, shape)

    assert not (groups["masked"] & groups["unmasked"]).any()
    assert np.array_equal(groups["masked"] | groups["unmasked"], native)


# ---------------------------------------------------------------------------
# Published mask artifacts
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MASKS_DIR.exists(), reason="canonical masks not published")
def test_published_masks_re_derive_from_their_seeds() -> None:
    """Every published mask must reproduce from its recorded seed and ratio.

    ``MaskGenerator.load`` regenerates the mask and compares both the checksum
    and the stored indices, so a hand-edited artifact fails here.
    """
    artifacts = sorted(MASKS_DIR.glob("node*.json"))
    assert artifacts, "no mask artifacts found"
    for path in artifacts:
        generator = MaskGenerator.load(path)
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert generator.mask_id == path.stem
        assert len(stored["masked_indices"]) == round(
            generator.num_nodes * generator.missing_ratio
        )
        if "masked_sensor_ids" in stored:
            assert len(stored["masked_sensor_ids"]) == len(stored["masked_indices"])


@pytest.mark.skipif(not MASKS_DIR.exists(), reason="canonical masks not published")
def test_published_masks_are_nested_across_levels() -> None:
    """Per ADR-0010 a sensor withheld at 10% is also withheld at 20% and 30%."""
    for seed in (2026, 2027, 2028):
        paths = [MASKS_DIR / f"node{level}-seed{seed}.json" for level in (10, 20, 30)]
        if not all(path.exists() for path in paths):
            pytest.skip(f"masks for seed {seed} not published")
        withheld = [set(json.loads(p.read_text(encoding="utf-8"))["masked_indices"]) for p in paths]
        assert withheld[0] < withheld[1] < withheld[2]


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


def test_zero_ratio_is_the_normal_condition() -> None:
    assert resolve_mask_generator({}, N_SENSORS, RUN_SEED) is None
    assert resolve_mask_generator(None, N_SENSORS, RUN_SEED) is None
    assert resolve_mask_generator({"missing_ratio": 0.0}, N_SENSORS, RUN_SEED) is None


def test_mask_seed_defaults_to_the_run_seed() -> None:
    generator = resolve_mask_generator({"missing_ratio": MASK_RATIO}, N_SENSORS, RUN_SEED)
    assert generator is not None
    assert generator.seed == RUN_SEED


def test_a_condition_without_a_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="masking.seed is required"):
        resolve_mask_generator({"missing_ratio": MASK_RATIO}, N_SENSORS, None)


def test_a_mask_artifact_contradicting_the_config_is_rejected(tmp_path: Path) -> None:
    """A run must not score one condition against another condition's mask."""
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=0.2, seed=2026)
    path = tmp_path / "mask.json"
    generator.save(path)

    with pytest.raises(ValueError, match="contradicts mask artifact"):
        resolve_mask_generator(
            {"mask_path": str(path), "missing_ratio": 0.3}, N_SENSORS, 2026
        )
    with pytest.raises(ValueError, match="contradicts mask artifact"):
        resolve_mask_generator(
            {"mask_path": str(path), "missing_ratio": 0.2, "seed": 2027}, N_SENSORS, None
        )
    with pytest.raises(ValueError, match="covers"):
        resolve_mask_generator({"mask_path": str(path)}, N_SENSORS + 1, 2026)


def test_a_hand_edited_mask_artifact_is_rejected(tmp_path: Path) -> None:
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=2026)
    path = tmp_path / "mask.json"
    generator.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["masked_indices"] = [0, 1, 2]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match the mask regenerated"):
        MaskGenerator.load(path)


# ---------------------------------------------------------------------------
# The leakage gate
# ---------------------------------------------------------------------------


def test_withheld_sensor_values_cannot_reach_predictions(tmp_path: Path) -> None:
    """The Phase 11 leakage gate.

    Two artifacts differ only in the withheld sensors' input values: one holds
    genuine normalized speeds, the other 999.0. Under the condition both runs
    must produce bit-identical predictions, which is only possible if no withheld
    value reached a forward pass. The same checkpoint is evaluated in both runs,
    so nothing but the masked-away inputs differs.
    """
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)
    withheld = np.flatnonzero(~generator.node_mask)

    clean_dir = tmp_path / "clean"
    poisoned_dir = tmp_path / "poisoned"
    _write_artifact(clean_dir)
    _write_artifact(poisoned_dir, corrupt_nodes=withheld)

    # Train once under the normal condition: this checkpoint is the model under test.
    trained = train_st_dssm(
        config=_training_config(tmp_path, clean_dir, mask_ratio=None),
        output_dir=str(tmp_path / "train_results"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_plots=False,
    )
    checkpoint = trained["checkpoint_path"]

    runs = {}
    for name, artifact_dir in (("clean", clean_dir), ("poisoned", poisoned_dir)):
        out_dir = tmp_path / f"{name}_results"
        runs[name] = train_st_dssm(
            config=_training_config(tmp_path, artifact_dir, mask_ratio=MASK_RATIO),
            output_dir=str(out_dir),
            checkpoint_dir=str(tmp_path / f"{name}_ckpt"),
            save_plots=False,
            predict_from=checkpoint,
        )
        runs[name]["predictions"] = _predictions_of(out_dir, runs[name]["run_id"])

    assert np.array_equal(runs["clean"]["predictions"], runs["poisoned"]["predictions"])
    # And the scored result is identical, not merely close.
    assert (
        runs["clean"]["manifest"].overall_metrics
        == runs["poisoned"]["manifest"].overall_metrics
    )


def test_a_condition_run_leaves_the_targets_intact(tmp_path: Path) -> None:
    """Masking is applied to inputs in place, so pin that targets never move."""
    artifact_dir = tmp_path / "artifact"
    _write_artifact(artifact_dir)
    before, _ = load_and_validate_artifact(str(artifact_dir))
    targets = {key: array.copy() for key, array in before.items() if "_Y" in key}

    train_st_dssm(
        config=_training_config(tmp_path, artifact_dir, mask_ratio=MASK_RATIO),
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_plots=False,
    )

    after, _ = load_and_validate_artifact(str(artifact_dir))
    for key, array in targets.items():
        assert np.array_equal(after[key], array), f"{key} changed on disk"


def test_the_all_group_still_scores_every_sensor(tmp_path: Path) -> None:
    """A condition changes the model's inputs, never the evaluated target set."""
    artifact_dir = tmp_path / "artifact"
    _write_artifact(artifact_dir)

    manifests = {}
    for name, ratio in (("normal", None), ("masked", MASK_RATIO)):
        result = train_st_dssm(
            config=_training_config(tmp_path, artifact_dir, mask_ratio=ratio),
            output_dir=str(tmp_path / f"{name}_results"),
            checkpoint_dir=str(tmp_path / f"{name}_ckpt"),
            save_plots=False,
        )
        manifests[name] = result["manifest"]

    assert (
        manifests["masked"].overall_metrics["valid_count"]
        == manifests["normal"].overall_metrics["valid_count"]
    )


# ---------------------------------------------------------------------------
# Reporting and provenance
# ---------------------------------------------------------------------------


def test_a_condition_run_records_its_mask_and_both_target_groups(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    sensor_ids = _write_artifact(artifact_dir)
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)

    result = train_st_dssm(
        config=_training_config(tmp_path, artifact_dir, mask_ratio=MASK_RATIO),
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_plots=False,
    )
    manifest = load_run_manifest(result["manifest_path"])

    # Provenance: which sensors were withheld, under which seed, verifiably.
    assert manifest.mask_condition == "30%"
    assert manifest.mask_seed == RUN_SEED
    assert manifest.mask_checksum == generator.checksum
    assert manifest.mask_sensor_ids == generator.get_masked_sensor_ids(sensor_ids)

    # Reporting: all / masked / unmasked, aggregate and per horizon.
    all_group = _records(manifest, "all")
    masked = _records(manifest, "masked")
    unmasked = _records(manifest, "unmasked")
    assert set(masked) == set(unmasked) == set(all_group)
    assert "MAE" in masked and "NLL" in masked
    assert (
        masked["MAE"]["valid_count"] + unmasked["MAE"]["valid_count"]
        == all_group["MAE"]["valid_count"]
    )
    for horizon in range(1, LENGTH + 1):
        per_h = _records(manifest, "masked", horizon=horizon)
        assert "MAE" in per_h, f"no masked-group metrics for horizon {horizon}"

    # The run identity carries the condition, so conditions cannot collide.
    assert result["run_id"].startswith(f"st_dssm-test-seed{RUN_SEED}-{generator.mask_id}")


def test_the_normal_condition_reports_only_the_all_group(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_artifact(artifact_dir)

    result = train_st_dssm(
        config=_training_config(tmp_path, artifact_dir, mask_ratio=None),
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_plots=False,
    )
    manifest = result["manifest"]

    assert manifest.mask_condition == "0%"
    assert manifest.mask_seed is None
    assert manifest.mask_sensor_ids == []
    assert {record["target_group"] for record in manifest.metrics} == {"all"}


def test_every_model_withholds_the_same_sensors(tmp_path: Path) -> None:
    """Fairness: one ratio and seed means one set of withheld sensors."""
    artifact_dir = tmp_path / "artifact"
    _write_artifact(artifact_dir)
    masking = {"missing_ratio": MASK_RATIO, "seed": RUN_SEED}

    dssm = train_st_dssm(
        config=_training_config(tmp_path, artifact_dir, mask_ratio=MASK_RATIO),
        output_dir=str(tmp_path / "dssm_results"),
        checkpoint_dir=str(tmp_path / "dssm_ckpt"),
        save_plots=False,
    )["manifest"]

    persistence = run_persistence_baseline(
        config={
            "artifact_dir": str(artifact_dir),
            "split": "test",
            "seed": RUN_SEED,
            "masking": masking,
        },
        output_dir=str(tmp_path / "persistence_results"),
        save_plots=False,
    )["manifest"]

    assert persistence.mask_checksum == dssm.mask_checksum
    assert persistence.mask_sensor_ids == dssm.mask_sensor_ids
    assert persistence.mask_condition == dssm.mask_condition == "30%"


def test_persistence_predicts_the_training_mean_for_withheld_sensors(tmp_path: Path) -> None:
    """A withheld sensor has no history, so persistence falls back to the scaler
    mean (a zero normalized input). This is reported, not corrected for."""
    artifact_dir = tmp_path / "artifact"
    _write_artifact(artifact_dir)
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)
    withheld = np.flatnonzero(~generator.node_mask)

    result = run_persistence_baseline(
        config={
            "artifact_dir": str(artifact_dir),
            "split": "test",
            "seed": RUN_SEED,
            "masking": {"missing_ratio": MASK_RATIO, "seed": RUN_SEED},
        },
        output_dir=str(tmp_path / "results"),
        save_plots=False,
    )
    predictions = _predictions_of(tmp_path / "results", result["run_id"])

    assert np.all(predictions[:, :, withheld, :] == 0.0)
    observed = np.flatnonzero(generator.node_mask)
    assert np.any(predictions[:, :, observed, :] != 0.0)


def test_predict_from_runs_no_epochs_and_writes_no_checkpoint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_artifact(artifact_dir)

    trained = train_st_dssm(
        config=_training_config(tmp_path, artifact_dir, mask_ratio=None),
        output_dir=str(tmp_path / "train_results"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        save_plots=False,
    )

    predict_ckpt_dir = tmp_path / "predict_ckpt"
    result = train_st_dssm(
        config=_training_config(tmp_path, artifact_dir, mask_ratio=MASK_RATIO),
        output_dir=str(tmp_path / "predict_results"),
        checkpoint_dir=str(predict_ckpt_dir),
        save_plots=False,
        predict_from=trained["checkpoint_path"],
    )
    manifest = result["manifest"]

    assert manifest.total_epochs == 0  # nothing was fitted in this run
    assert manifest.checkpoint_path == trained["checkpoint_path"]
    assert not list(predict_ckpt_dir.glob("*.pt"))  # no new weights, no resume state
    assert "-predict-" in result["run_id"]


def test_evaluate_cli_mask_flags_add_the_target_groups(tmp_path: Path) -> None:
    """Group reporting is available to a bare re-evaluation of saved predictions."""
    artifact_dir = tmp_path / "artifact"
    sensor_ids = _write_artifact(artifact_dir)
    arrays, _ = load_and_validate_artifact(str(artifact_dir))

    predictions_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        predictions=arrays["test_Y"] * 0.5,
        sigma=np.full_like(arrays["test_Y"], 0.3),
    )

    out_dir = tmp_path / "results"
    exit_code = evaluate_main(
        [
            "--artifact-dir", str(artifact_dir),
            "--predictions-path", str(predictions_path),
            "--split", "test",
            "--output-dir", str(out_dir),
            "--run-id", "masked-eval",
            "--model-name", "st_dssm",
            "--mask-ratio", str(MASK_RATIO),
            "--mask-seed", str(RUN_SEED),
            "--no-plots",
        ]
    )
    assert exit_code == 0

    manifest = load_run_manifest(out_dir / "masked-eval_manifest.json")
    generator = MaskGenerator(num_nodes=N_SENSORS, missing_ratio=MASK_RATIO, seed=RUN_SEED)
    assert manifest.mask_sensor_ids == generator.get_masked_sensor_ids(sensor_ids)
    assert {record["target_group"] for record in manifest.metrics} == {
        "all",
        "masked",
        "unmasked",
    }
    # The evaluator masks nothing itself: the all-group metrics are unchanged by
    # the flags, which only add the per-group breakdown.
    assert manifest.overall_metrics["valid_count"] == int(arrays["test_Y_mask"].sum())
