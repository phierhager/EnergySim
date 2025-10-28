import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class TimestepData:
    """
    Holds all exogenous time-series data for a single point in time.

    This is the definitive data container that flows from the dataset
    into the simulation state for each step.
    """
    timestamp: int
    dt_seconds: int
    features: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.features[key]

    def get(self, key: str, default: np.ndarray = np.array([0.0])) -> np.ndarray:
        return self.features.get(key, default)