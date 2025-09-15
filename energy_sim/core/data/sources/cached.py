"""Composite data source implementations."""

from dataclasses import dataclass
from typing import Dict, Tuple, Literal
import numpy as np

from bems_simulation.core.data.sources.base import (
    DataSource,
    DataRequest,
    BaseDataSourceConfig,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class CachedDataSourceConfig(BaseDataSourceConfig):
    """Configuration for cached data sources."""

    wrapped_source_config: BaseDataSourceConfig
    cache_size: int = 1000  # Maximum number of requests to cache

    type: Literal["cached"] = "cached"


class CachedDataSource(DataSource):
    """Wrapper that adds caching to any data source."""

    def __init__(self, source: DataSource, cache_size: int = 1000):
        super().__init__(source.dt_seconds)
        self.source = source
        self.cache_size = cache_size
        self._cache: Dict[DataRequest, Dict[str, np.ndarray]] = {}

    def get_time_range(self) -> Tuple[int, int]:
        return self.source.get_time_range()

    def get_available_columns(self) -> Tuple[str, ...]:
        return self.source.get_available_columns()

    def get_data(self, request: DataRequest) -> Dict[str, np.ndarray]:
        if request in self._cache:
            return self._cache[request]

        data = self.source.get_data(request)

        # Simple LRU eviction
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[request] = data

        return data
