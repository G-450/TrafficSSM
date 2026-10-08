import numpy as np


class MaskGenerator:
    """
    Generates and applies deterministic node-level experimental observation masks.
    Implements Phase 11 missingness mechanism per ADR-0004 and ADR-0007.
    """

    def __init__(self, num_nodes: int, missing_ratio: float, seed: int):
        """
        Initializes the MaskGenerator with a fixed node mask.

        Args:
            num_nodes: Total number of sensors (e.g., 325 for PEMS-BAY).
            missing_ratio: Fraction of nodes to withhold (e.g., 0.20 for 20%).
            seed: Random seed for reproducibility (canonical seeds: 2026, 2027, 2028).
        """
        if not (0.0 <= missing_ratio < 1.0):
            raise ValueError(f"Missing ratio must be in [0.0, 1.0), got {missing_ratio}")

        self.num_nodes = num_nodes
        self.missing_ratio = missing_ratio
        self.seed = seed

        # Generate the deterministic mask
        rng = np.random.default_rng(seed)
        num_missing = int(round(num_nodes * missing_ratio))

        self.node_mask = np.ones(num_nodes, dtype=bool)
        if num_missing > 0:
            missing_indices = rng.choice(num_nodes, size=num_missing, replace=False)
            self.node_mask[missing_indices] = False

    def get_masked_sensor_ids(self, sensor_ids: list[str]) -> list[str]:
        """Returns the IDs of the sensors that are masked."""
        if len(sensor_ids) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} sensor IDs, got {len(sensor_ids)}")
        return [sid for sid, is_observed in zip(sensor_ids, self.node_mask) if not is_observed]

    def apply_mask(self, x: np.ndarray, x_mask_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies the experimental mask to input arrays.
        M_obs = M_native AND M_experiment.
        Unobserved numeric inputs are set to 0.0.

        Args:
            x: Input array, typically [S, L, N, C] or similar. The last or second to last axis 
               usually contains nodes. Assuming shape [..., N, C] or [..., N].
            x_mask_native: Native missingness mask, same shape as x.

        Returns:
            tuple containing (x_masked, x_mask_combined)
        """
        if x.shape != x_mask_native.shape:
            raise ValueError(f"Shape mismatch: x {x.shape} != x_mask_native {x_mask_native.shape}")

        # Find the node axis. Usually it's axis -2 if C=1, or axis -1 if C is missing.
        if x.shape[-1] == self.num_nodes:
            node_axis = -1
        elif len(x.shape) > 1 and x.shape[-2] == self.num_nodes:
            node_axis = -2
        else:
            raise ValueError(f"Cannot find node dimension of size {self.num_nodes} in shape {x.shape}")

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
