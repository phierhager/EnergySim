"""Tests for energy dataset implementation."""

import pytest
from unittest.mock import Mock
import numpy as np
from energysim.core.data.dataset import EnergyDataset
from energysim.core.data.sources.base import DataSource, DataRequest
from energysim.core.data.config import EnergyDatasetParams
from energysim.core.data.data_bundle import DataBundle


class MockDataSource(DataSource):
    def __init__(self, start_time=0, end_time=86400, data=None):
        self._start = start_time
        self._end = end_time
        self._data = data or {
            "load_kW": np.array([100.0, 110.0, 120.0]),
            "price_": np.array([0.10, 0.12, 0.11]),
            "pv_kW": np.array([50.0, 60.0, 55.0]),
        }

    def get_time_range(self):
        return self._start, self._end

    def get_data(self, request: DataRequest):
        # Simple mock that returns the same data regardless of request
        return self._data

    def get_available_columns(self):
        return tuple(self._data.keys())


def test_energy_dataset_initialization():
    # Arrange
    data_source = MockDataSource(start_time=0, end_time=7200)  # 2 hours
    params = EnergyDatasetParams(
        feature_columns={"load": "load_kW", "price": "price_$"},
        dt_seconds=3600,  # 1 hour intervals
        use_time_features=True,
    )

    # Act
    dataset = EnergyDataset(data_source, params)

    # Assert
    assert dataset.data_source == data_source
    assert dataset.params == params
    assert dataset.start == 0
    assert dataset.end == 7200
    assert len(dataset.timestamps) == 2  # 0 and 3600


def test_energy_dataset_initialization_with_prediction_horizon():
    # Arrange
    data_source = MockDataSource(start_time=0, end_time=10800)  # 3 hours
    params = EnergyDatasetParams(
        feature_columns={"load": "load_kW"},
        dt_seconds=3600,
        prediction_horizon=2,  # 2 steps ahead
        use_time_features=False,
    )

    # Act
    dataset = EnergyDataset(data_source, params)

    # Assert
    # Should stop 2*3600 = 7200 seconds before end
    # Timestamps: 0 (only one timestamp since 3600 >= 10800-7200)
    assert len(dataset.timestamps) == 1


def test_energy_dataset_length():
    # Arrange
    data_source = MockDataSource(start_time=0, end_time=14400)  # 4 hours
    params = EnergyDatasetParams(
        feature_columns={"load": "load_kW"}, dt_seconds=3600, use_time_features=False
    )
    dataset = EnergyDataset(data_source, params)

    # Act & Assert
    assert len(dataset) == 4  # 0, 3600, 7200, 10800


def test_energy_dataset_getitem():
    # Arrange
    mock_data = {"load_kW": np.array([150.0]), "price_$": np.array([0.15])}
    data_source = MockDataSource(data=mock_data)
    params = EnergyDatasetParams(
        feature_columns={"load": "load_kW", "price": "price_$"},
        dt_seconds=3600,
        use_time_features=False,
    )
    dataset = EnergyDataset(data_source, params)

    # Act
    data_bundle = dataset[0]

    # Assert
    assert isinstance(data_bundle, DataBundle)


def test_energy_dataset_iteration():
    # Arrange
    data_source = MockDataSource(start_time=0, end_time=7200)  # 2 hours
    params = EnergyDatasetParams(
        feature_columns={"load": "load_kW"}, dt_seconds=3600, use_time_features=False
    )
    dataset = EnergyDataset(data_source, params)

    # Act
    data_bundles = list(dataset)

    # Assert
    assert len(data_bundles) == 2
    assert all(isinstance(bundle, DataBundle) for bundle in data_bundles)


def test_energy_dataset_get_data_bundle_calls_source_correctly():
    # Arrange
    data_source = Mock(spec=DataSource)
    data_source.get_time_range.return_value = (0, 7200)
    data_source.get_data.return_value = {"load_kW": np.array([100.0])}

    params = EnergyDatasetParams(
        feature_columns={"load": "load_kW"},
        dt_seconds=3600,
        prediction_horizon=1,
        use_time_features=False,
    )
    dataset = EnergyDataset(data_source, params)

    # Act
    dataset[0]

    # Assert
    data_source.get_data.assert_called_once()
    call_args = data_source.get_data.call_args[0][0]
    assert isinstance(call_args, DataRequest)
    assert call_args.timestamp == 0
    assert call_args.columns == ("load_kW",)
    assert call_args.prediction_horizon == 1


def test_energy_dataset_empty_timestamps():
    # Arrange - scenario where no valid timestamps exist
    data_source = MockDataSource(start_time=0, end_time=1800)  # 30 minutes
    params = EnergyDatasetParams(
        feature_columns={"load": "load_kW"},
        dt_seconds=3600,  # 1 hour intervals
        prediction_horizon=2,  # Need 2*3600=7200s buffer
        use_time_features=False,
    )

    # Act
    dataset = EnergyDataset(data_source, params)

    # Assert
    assert len(dataset) == 0
    assert len(list(dataset)) == 0
