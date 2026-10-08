import numpy as np
import pytest

from st_dssm.mask_generator import MaskGenerator


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
