# -------------------------------
# Abstract Sensor Interface
# -------------------------------
from abc import ABC, abstractmethod
from typing import Any
from energysim.core.shared.spaces import Space

class Sensor(ABC):
    """Abstract interface for all sensors."""

    @abstractmethod
    def read(self, *args: Any) -> dict:
        """Return the current observation vector from this sensor."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Return the observation space corresponding to this sensor."""
        pass
