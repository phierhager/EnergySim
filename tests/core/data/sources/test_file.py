"""Tests for file data source implementation."""

import pytest
import pandas as pd
from unittest.mock import patch, Mock
from bems_simulation.core.data.sources.file import FileDataSource
from bems_simulation.core.data.sources.config import FileDataSourceConfig
from bems_simulation.core.data.sources.base import DataRequest


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return pd.DataFrame(
        {
            "time": [0, 3600, 7200, 10800, 14400],  # 5 hours of data
            "load_kW": [100.0, 110.0, 120.0, 115.0, 105.0],
            "price_$": [0.10, 0.12, 0.11, 0.13, 0.09],
            "pv_kW": [0.0, 20.0, 50.0, 30.0, 10.0],
        }
    )


def test_file_data_source_initialization(sample_csv_data):
    # Arrange
    config = FileDataSourceConfig(file_path="test.csv", time_column="time")

    with patch("pandas.read_csv", return_value=sample_csv_data):
        # Act
        source = FileDataSource(config, dt_seconds=3600)

        # Assert
        assert source.config == config
        assert len(source._df) == 5
        assert list(source._df.columns) == ["time", "load_kW", "price_$", "pv_kW"]


def test_file_data_source_get_time_range(sample_csv_data):
    # Arrange
    config = FileDataSourceConfig(file_path="test.csv", time_column="time")

    with patch("pandas.read_csv", return_value=sample_csv_data):
        source = FileDataSource(config)

        # Act
        start, end = source.get_time_range()

        # Assert
        assert start == 0
        assert end == 14400


def test_file_data_source_get_data_single_timestamp(sample_csv_data):
    # Arrange
    config = FileDataSourceConfig(file_path="test.csv", time_column="time")

    with patch("pandas.read_csv", return_value=sample_csv_data):
        source = FileDataSource(config)

        request = DataRequest(
            timestamp=3600, columns=("load_kW", "price_$"), prediction_horizon=None
        )

        # Act
        result = source.get_data(request)

        # Assert
        assert "load_kW" in result
        assert "price_$" in result
        assert len(result["load_kW"]) == 1
        assert result["load_kW"][0] == 110.0
        assert result["price_$"][0] == 0.12


def test_file_data_source_get_data_with_prediction_horizon(sample_csv_data):
    # Arrange
    config = FileDataSourceConfig(file_path="test.csv", time_column="time")

    with patch("pandas.read_csv", return_value=sample_csv_data):
        source = FileDataSource(config, dt_seconds=3600)

        request = DataRequest(
            timestamp=3600, columns=("load_kW",), prediction_horizon=2
        )

        # Act
        result = source.get_data(request)

        # Assert
        # Should return current + next 2 timesteps = 3 values
        assert len(result["load_kW"]) == 3
        assert result["load_kW"][0] == 110.0  # Current (t=3600)
        assert result["load_kW"][1] == 120.0  # Next (t=7200)
        assert result["load_kW"][2] == 115.0  # Next+1 (t=10800)


def test_file_data_source_get_data_timestamp_not_found(sample_csv_data):
    # Arrange
    config = FileDataSourceConfig(file_path="test.csv", time_column="time")

    with patch("pandas.read_csv", return_value=sample_csv_data):
        source = FileDataSource(config, dt_seconds=3600)

        request = DataRequest(
            timestamp=1800,  # Not in the data
            columns=("load_kW",),
            prediction_horizon=None,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Timestamp 1800 not found"):
            source.get_data(request)


def test_file_data_source_get_data_insufficient_horizon_data(sample_csv_data):
    # Arrange
    config = FileDataSourceConfig(file_path="test.csv", time_column="time")

    with patch("pandas.read_csv", return_value=sample_csv_data):
        source = FileDataSource(config)

        request = DataRequest(
            timestamp=10800,  # Near end of data
            columns=("load_kW",),
            prediction_horizon=5,  # Asking for more future data than available
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Insufficient data for prediction horizon"
        ):
            source.get_data(request)


def test_file_data_source_get_data_missing_column(sample_csv_data):
    # Arrange
    config = FileDataSourceConfig(file_path="test.csv", time_column="time")

    with patch("pandas.read_csv", return_value=sample_csv_data):
        source = FileDataSource(config)

        request = DataRequest(
            timestamp=3600, columns=("missing_column",), prediction_horizon=None
        )

        # Act & Assert
        with pytest.raises(KeyError):
            source.get_data(request)


@patch("pandas.read_csv")
def test_file_data_source_file_not_found(mock_read_csv):
    # Arrange
    mock_read_csv.side_effect = FileNotFoundError("File not found")
    config = FileDataSourceConfig(file_path="nonexistent.csv", time_column="time")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        FileDataSource(config)


def test_file_data_source_empty_dataframe():
    # Arrange
    empty_df = pd.DataFrame()
    config = FileDataSourceConfig(file_path="empty.csv", time_column="time")

    with patch("pandas.read_csv", return_value=empty_df):
        # Act & Assert
        with pytest.raises(KeyError, match="Time column 'time' not found in file"):
            FileDataSource(config)
