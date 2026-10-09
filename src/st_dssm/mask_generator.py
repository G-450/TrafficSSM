"""Deterministic mask generation for experimental observation simulation.

Implements Phase 11 missingness mechanism per ADR-0004 and ADR-0007.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from st_dssm.result_schema import RunManifest


class MaskGenerator:
    """
    Generates and applies deterministic node-level experimental observation masks.
    Implements Phase 11 missingness mechanism per ADR-0004 and ADR-0007.

    Canonical mask counts for 325 sensors:
    - 10%: 32 sensors masked
    - 20%: 65 sensors masked
    - 30%: 98 sensors masked
    """

    def __init__(self, num_nodes: int, missing_ratio: float, seed: int):
        """
        Initializes the MaskGenerator with a fixed node mask.

        Args:
            num_nodes: Total number of sensors (e.g., 325 for PEMS-BAY).
            missing_ratio: Fraction of nodes to withhold (e.g., 0.20 for 20%).
            seed: Random seed for reproducibility (canonical seeds: 2026, 2027, 2028).
        """
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be strictly positive, got {num_nodes}")

        if not (0.0 <= missing_ratio < 1.0):
            raise ValueError(
                f"Missing ratio must be in [0.0, 1.0), got {missing_ratio}"
            )

        self.num_nodes = num_nodes
        self.missing_ratio = missing_ratio
        self.seed = seed

        # Generate the deterministic mask
        rng = np.random.default_rng(seed)
        num_missing = round(num_nodes * missing_ratio)

        self.node_mask = np.ones(num_nodes, dtype=bool)
        if num_missing > 0:
            # A prefix of one seeded permutation keeps masks nested across ratios
            # for the same seed (10% within 20% within 30%), per ADR-0010.
            missing_indices = rng.permutation(num_nodes)[:num_missing]
            self.node_mask[missing_indices] = False

        self.checksum = hashlib.sha256(self.node_mask.tobytes()).hexdigest()
        self.mask_condition = f"{round(missing_ratio * 100)}%"
        self.mask_id = f"node{round(missing_ratio * 100)}-seed{seed}"

    def get_masked_sensor_ids(self, sensor_ids: list[str]) -> list[str]:
        """Returns the IDs of the sensors that are masked."""
        if len(sensor_ids) != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} sensor IDs, got {len(sensor_ids)}"
            )
        return [
            sid
            for sid, is_observed in zip(sensor_ids, self.node_mask)
            if not is_observed
        ]

    def save(
        self, path: str | Path | None = None, sensor_ids: list[str] | None = None
    ) -> None:
        """Saves the mask metadata and content to a JSON file."""
        if path is None:
            path = Path("experiments") / "masks" / f"{self.mask_id}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mask_id": self.mask_id,
            "seed": self.seed,
            "missing_ratio": self.missing_ratio,
            "num_nodes": self.num_nodes,
            "masked_indices": np.where(~self.node_mask)[0].tolist(),
            "checksum": self.checksum,
        }

        if sensor_ids is not None:
            data["masked_sensor_ids"] = self.get_masked_sensor_ids(sensor_ids)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> MaskGenerator:
        """Loads a MaskGenerator from a JSON file and validates its checksum."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(
            num_nodes=data["num_nodes"],
            missing_ratio=data["missing_ratio"],
            seed=data["seed"],
        )

        if instance.checksum != data["checksum"]:
            raise ValueError(
                f"Checksum mismatch for mask loaded from {path}. "
                f"Expected {data['checksum']}, got {instance.checksum}"
            )

        stored_indices = sorted(data["masked_indices"])
        actual_indices = np.flatnonzero(~instance.node_mask).tolist()
        if stored_indices != actual_indices:
            raise ValueError(
                f"Masked indices in {path} do not match the mask regenerated "
                f"from seed {data['seed']}"
            )

        return instance

    def fill_manifest(self, manifest: RunManifest, sensor_ids: list[str]) -> None:
        """Fills the mask_* fields of a RunManifest."""
        manifest.mask_seed = self.seed
        manifest.mask_condition = self.mask_condition
        manifest.mask_sensor_ids = self.get_masked_sensor_ids(sensor_ids)
        manifest.mask_checksum = self.checksum

    def apply_mask(
        self, x: np.ndarray, x_mask_native: np.ndarray, node_axis: int = -2
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies the experimental mask to input arrays.
        M_obs = M_native AND M_experiment.
        Unobserved numeric inputs are set to 0.0.

        Args:
            x: Input array, typically [S, L, N, C] or similar.
            x_mask_native: Native missingness mask, same shape as x.
            node_axis: Axis corresponding to the nodes. Default is -2.

        Returns:
            tuple containing (x_masked, x_mask_combined)
        """
        if x.shape != x_mask_native.shape:
            raise ValueError(
                f"Shape mismatch: x {x.shape} != x_mask_native {x_mask_native.shape}"
            )

        try:
            if x.shape[node_axis] != self.num_nodes:
                raise ValueError(
                    f"Expected {self.num_nodes} nodes along axis {node_axis}, got {x.shape[node_axis]}"
                )
        except IndexError:
            raise ValueError(f"Invalid node_axis {node_axis} for shape {x.shape}")

        # Broadcast node mask to x's shape
        reshape_dims = [1] * x.ndim
        reshape_dims[node_axis] = self.num_nodes
        node_mask_bcast = self.node_mask.reshape(*reshape_dims)

        # M_obs = M_native AND M_experiment
        x_mask_combined = x_mask_native & node_mask_bcast

        # Prevent masked ground truth from reaching inputs
        x_masked = np.copy(x)
        x_masked[~x_mask_combined] = 0.0

        return x_masked, x_mask_combined


def generate_canonical_masks(
    sensor_ids: list[str], out_dir: str = "experiments/masks"
) -> None:
    """Generates canonical masks (10%, 20%, 30%) for standard seeds (2026, 2027, 2028)."""
    num_nodes = len(sensor_ids)
    ratios = [0.1, 0.2, 0.3]
    seeds = [2026, 2027, 2028]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for ratio in ratios:
        for seed in seeds:
            generator = MaskGenerator(
                num_nodes=num_nodes, missing_ratio=ratio, seed=seed
            )
            generator.save(
                out_path / f"{generator.mask_id}.json", sensor_ids=sensor_ids
            )
