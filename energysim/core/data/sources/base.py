"""Base abstractions for data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True, kw_only=True, slots=True)
class BaseDataSourceConfig:
    """Base configuration for data sources."""

    type: str


@dataclass(frozen=True)
class DataRequest:
    """Request for data at a specific time with optional prediction horizon."""

    timestamp: int
    columns: Tuple[str, ...]
    prediction_horizon: int | None = None
    """Prediction horizon in dt_seconds; None means no prediction requested."""

    def __post_init__(self):
        if self.prediction_horizon is not None and self.prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive.")


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, dt_seconds: int) -> None:
        self._dt_seconds = dt_seconds

    @property
    def dt_seconds(self) -> int:
        return self._dt_seconds

    @abstractmethod
    def get_data(self, request: DataRequest) -> Dict[str, np.ndarray]:
        """
        Get data for the given request.

        Returns:
            A dictionary mapping column names to numpy arrays.
        """
        pass

    @abstractmethod
    def get_time_range(self) -> Tuple[int, int]:
        """Get the available time range (min_time, max_time)."""
        pass

    @abstractmethod
    def get_available_columns(self) -> Tuple[str, ...]:
        """Get the list of columns available from this data source."""
        pass

    @property
    def columns(self) -> Tuple[str, ...]:
        """Pythonic alias for available columns."""
        return self.get_available_columns()
