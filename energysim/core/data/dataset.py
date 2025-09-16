import numpy as np
from energysim.core.data.sources.base import DataRequest, DataSource
from energysim.core.data.sources.factory import DataSourceFactory
from dataclasses import dataclass
from energysim.core.data.data_bundle import DataBundle
from energysim.core.data.config import EnergyDatasetConfig, EnergyDatasetParams
from energysim.core.data.transforms import get_time_features


class EnergyDataset:
    def __init__(self, data_source: DataSource, params: EnergyDatasetParams):
        self.data_source = data_source
        self.params = params
        self.start, self.end = self.data_source.get_time_range()
        horizon_seconds = (self.params.prediction_horizon or 0) * self.params.dt_seconds
        self.timestamps = range(
            self.start, self.end - horizon_seconds, self.params.dt_seconds
        )

    def __getitem__(self, idx: int) -> DataBundle:
        timestamp = self.timestamps[idx]
        return self._get_data_bundle(timestamp)

    def __len__(self) -> int:
        return len(self.timestamps)

    def __iter__(self):
        for t in self.timestamps:
            yield self._get_data_bundle(t)

    def _get_data_bundle(self, timestamp: int) -> DataBundle:
        columns = tuple(self.params.feature_columns.values())
        request = DataRequest(
            timestamp=timestamp,
            columns=columns,
            prediction_horizon=self.params.prediction_horizon,
        )
        data = self.data_source.get_data(request)
        features = {
            name: data[col] for name, col in self.params.feature_columns.items()
        }
        if self.params.use_time_features:
            features["time"] = get_time_features(timestamp)
        return DataBundle(features=features, dt_seconds=self.params.dt_seconds)

    @property
    def num_timestamps(self) -> int:
        return len(self.timestamps)

    @property
    def num_features(self) -> int:
        """Return the total number of features per timestamp."""
        sample_bundle = self[0]
        assert all(feat.ndim == 1 for feat in sample_bundle.features.values())
        return sum(feat.shape[0] for feat in sample_bundle.features.values())
