import json

import numpy as np
import pytest

from st_dssm.mask_generator import MaskGenerator, generate_canonical_masks
from st_dssm.result_schema import RunManifest


def test_mask_generator_initialization():
    gen = MaskGenerator(num_nodes=100, missing_ratio=0.2, seed=42)
    assert gen.num_nodes == 100
    assert gen.missing_ratio == 0.2
    assert gen.seed == 42

    # 20% of 100 is 20 missing, so 80 observed
    assert np.sum(gen.node_mask) == 80
    assert np.sum(~gen.node_mask) == 20


def test_mask_generator_invalid_ratio():
    with pytest.raises(ValueError):
        MaskGenerator(num_nodes=100, missing_ratio=-0.1, seed=42)

    with pytest.raises(ValueError):
        MaskGenerator(num_nodes=100, missing_ratio=1.5, seed=42)


def test_mask_generator_reproducibility():
    gen1 = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2026)
    gen2 = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2026)
    gen3 = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2027)

    np.testing.assert_array_equal(gen1.node_mask, gen2.node_mask)
    assert not np.array_equal(gen1.node_mask, gen3.node_mask)


def test_get_masked_sensor_ids():
    gen = MaskGenerator(num_nodes=5, missing_ratio=0.4, seed=42)  # 2 missing
    sensor_ids = ["A", "B", "C", "D", "E"]

    # Mask vector based on seed 42 will pick specific indices
    masked_ids = gen.get_masked_sensor_ids(sensor_ids)

    assert len(masked_ids) == 2
    assert all(sid in sensor_ids for sid in masked_ids)

    with pytest.raises(ValueError):
        gen.get_masked_sensor_ids(["A", "B"])


def test_apply_mask():
    gen = MaskGenerator(num_nodes=3, missing_ratio=0.0, seed=42)
    # Force mask for testing: 0 is missing, 1, 2 are observed
    gen.node_mask = np.array([False, True, True])

    x = np.ones((2, 3, 1), dtype=np.float32) * 5.0
    x_mask_native = np.ones((2, 3, 1), dtype=bool)

    # Introduce native missingness
    x_mask_native[0, 1, 0] = False

    x_masked, x_mask_combined = gen.apply_mask(x, x_mask_native)

    # Check combined mask
    # Node 0 is missing experimentally
    # Node 1 is missing natively at time 0
    assert not x_mask_combined[0, 0, 0]  # Exp masked
    assert not x_mask_combined[1, 0, 0]  # Exp masked

    assert not x_mask_combined[0, 1, 0]  # Natively masked
    assert x_mask_combined[1, 1, 0]  # Observed

    assert x_mask_combined[0, 2, 0]  # Observed
    assert x_mask_combined[1, 2, 0]  # Observed

    # Check values zeroed out
    assert x_masked[0, 0, 0] == 0.0
    assert x_masked[1, 0, 0] == 0.0
    assert x_masked[0, 1, 0] == 0.0
    assert x_masked[1, 1, 0] == 5.0
    assert x_masked[0, 2, 0] == 5.0


def test_apply_mask_shape_mismatch():
    gen = MaskGenerator(num_nodes=3, missing_ratio=0.2, seed=42)
    x = np.ones((2, 3, 1))
    x_mask = np.ones((2, 4, 1), dtype=bool)

    with pytest.raises(ValueError):
        gen.apply_mask(x, x_mask)

    x2 = np.ones((2, 4, 1))
    x_mask2 = np.ones((2, 4, 1), dtype=bool)

    with pytest.raises(ValueError):
        gen.apply_mask(x2, x_mask2)


def test_leakage_probe():
    gen = MaskGenerator(num_nodes=5, missing_ratio=0.4, seed=42)  # 2 missing
    x = np.ones((2, 5, 1), dtype=np.float32)

    # Fill masked sensors with 999.0
    masked_indices = np.where(~gen.node_mask)[0]
    for idx in masked_indices:
        x[:, idx, :] = 999.0

    x_original = x.copy()

    x_mask_native = np.ones((2, 5, 1), dtype=bool)

    x_masked, x_mask_combined = gen.apply_mask(x, x_mask_native, node_axis=-2)

    assert 999.0 not in x_masked
    assert np.all(x_mask_combined[:, masked_indices, :] == False)
    assert np.array_equal(x, x_original)


def test_save_load_round_trip(tmp_path):
    gen = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2026)
    path = tmp_path / "mask.json"
    gen.save(path)

    loaded_gen = MaskGenerator.load(path)
    np.testing.assert_array_equal(gen.node_mask, loaded_gen.node_mask)
    assert gen.checksum == loaded_gen.checksum

    # Edit checksum and check error
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["checksum"] = "bad_checksum"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        MaskGenerator.load(path)


def test_canonical_counts():
    gen10 = MaskGenerator(num_nodes=325, missing_ratio=0.1, seed=42)
    assert np.sum(~gen10.node_mask) == 32

    gen20 = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=42)
    assert np.sum(~gen20.node_mask) == 65

    gen30 = MaskGenerator(num_nodes=325, missing_ratio=0.3, seed=42)
    assert np.sum(~gen30.node_mask) == 98


def test_fairness():
    gen1 = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2026)
    gen2 = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2026)

    assert gen1.checksum == gen2.checksum
    assert gen1.mask_id == gen2.mask_id


def test_explicit_node_axis():
    gen = MaskGenerator(num_nodes=3, missing_ratio=0.0, seed=42)
    x = np.ones((2, 3, 1))  # nodes are at axis -2 (or 1)
    x_mask = np.ones((2, 3, 1), dtype=bool)

    # wrong axis: -1 (size 1)
    with pytest.raises(ValueError, match="Expected 3 nodes along axis -1, got 1"):
        gen.apply_mask(x, x_mask, node_axis=-1)

    # invalid axis: -5
    with pytest.raises(ValueError, match="Invalid node_axis"):
        gen.apply_mask(x, x_mask, node_axis=-5)


def test_masks_are_nested_for_same_seed():
    masked = {
        ratio: set(np.flatnonzero(~MaskGenerator(325, ratio, seed=2026).node_mask))
        for ratio in (0.1, 0.2, 0.3)
    }
    assert masked[0.1] < masked[0.2] < masked[0.3]


def test_load_rejects_edited_masked_indices(tmp_path):
    gen = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2026)
    path = tmp_path / "mask.json"
    gen.save(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    unmasked = int(np.flatnonzero(gen.node_mask)[0])
    data["masked_indices"][-1] = unmasked
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="do not match"):
        MaskGenerator.load(path)


def test_fill_manifest():
    sensor_ids = [f"s{i}" for i in range(325)]
    gen = MaskGenerator(num_nodes=325, missing_ratio=0.2, seed=2026)
    manifest = RunManifest()

    gen.fill_manifest(manifest, sensor_ids)

    assert manifest.mask_seed == 2026
    assert manifest.mask_condition == "20%"
    assert manifest.mask_sensor_ids == gen.get_masked_sensor_ids(sensor_ids)
    assert len(manifest.mask_sensor_ids) == 65
    assert manifest.mask_checksum == gen.checksum


def test_generate_canonical_masks(tmp_path):
    sensor_ids = [f"s{i}" for i in range(325)]
    generate_canonical_masks(sensor_ids, out_dir=str(tmp_path))

    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(files) == 9
    assert "node20-seed2026.json" in files

    for name in files:
        loaded = MaskGenerator.load(tmp_path / name)
        with open(tmp_path / name, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["masked_sensor_ids"] == loaded.get_masked_sensor_ids(sensor_ids)
