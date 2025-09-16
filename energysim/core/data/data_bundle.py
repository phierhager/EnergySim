import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class DataBundle:
    """Holds feature data for a timestamp."""

    features: dict[str, np.ndarray]
    dt_seconds: int = 60

    def __getitem__(self, key: str):
        return self.features[key]

    def __contains__(self, key: str) -> bool:
        return key in self.features
