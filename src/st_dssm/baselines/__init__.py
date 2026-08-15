"""Deterministic baseline forecasting models for the ST-DSSM project."""

from st_dssm.baselines.persistence import HistoricalPersistence
from st_dssm.baselines.st_gcn import DeterministicSTGCN, count_trainable_parameters

__all__ = [
    "DeterministicSTGCN",
    "HistoricalPersistence",
    "count_trainable_parameters",
]
