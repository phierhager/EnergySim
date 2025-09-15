"""File-based data source implementation."""

from dataclasses import dataclass
from typing import Tuple, Dict, Sequence, Literal
import numpy as np
import pandas as pd

from bems_simulation.core.data.sources.base import (
    DataSource,
    DataRequest,
    BaseDataSourceConfig,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class FileDataSourceConfig(BaseDataSourceConfig):
    """Configuration for file-based data sources."""

    file_path: str
    time_column: str = "unixtime"

    type: Literal["file"] = "file"


class FileDataSource(DataSource):
    """Data source that reads from CSV or Feather files."""

    def __init__(self, config: FileDataSourceConfig, dt_seconds: int = 900):
        super().__init__(dt_seconds)
        self.config = config

        # Load CSV or Feather
        if config.file_path.endswith(".feather"):
            df = pd.read_feather(config.file_path)
        else:
            df = pd.read_csv(config.file_path)

        if config.time_column not in df.columns:
            raise KeyError(f"Time column '{config.time_column}' not found in file")

        # Use datetime index in seconds
        df.set_index(config.time_column, inplace=True, drop=False)
        df.index = pd.to_datetime(df.index, unit="s")

        # Construct regular grid starting at min timestamp
        start = df.index.min()
        end = df.index.max()
        new_index = pd.date_range(start=start, end=end, freq=f"{self.dt_seconds}s")

        # Reindex and interpolate linearly NOTE: INTERPOLATION!
        df = df.reindex(new_index).interpolate(method="linear")

        # Convert index back to integer seconds since epoch
        df.index = df.index.astype("int64") // 10**9
        df.sort_index(inplace=True)

        # Save
        self._df = df
        self._time_range: Tuple[int, int] = (
            int(df.index.min()),
            int(df.index.max()),
        )
        self._columns: Tuple[str, ...] = tuple(df.columns)

    def get_time_range(self) -> Tuple[int, int]:
        return self._time_range

    def get_available_columns(self) -> Tuple[str, ...]:
        return self._columns

    def get_data(self, request: DataRequest) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}

        if request.timestamp not in self._df.index:
            raise ValueError(f"Timestamp {request.timestamp} not found in data")
        if request.prediction_horizon is not None:
            horizon_end = (
                request.timestamp + request.prediction_horizon * self.dt_seconds
            )
            if horizon_end > self._df.index.max():
                raise ValueError(
                    "Insufficient data for prediction horizon "
                    f"at timestamp {request.timestamp}"
                )

        for col in request.columns:
            if col not in self._df.columns:
                raise KeyError(f"Column '{col}' not found in data")

            # Current value
            current_val = np.array(
                [self._df.at[request.timestamp, col]], dtype=np.float32
            )

            # Future values if prediction horizon is specified
            if request.prediction_horizon is None:
                result[col] = current_val
                continue

            future_vals = []

            for i in range(request.prediction_horizon):
                ts = request.timestamp + (i + 1) * self.dt_seconds
                if ts in self._df.index:
                    future_vals.append(self._df.at[ts, col])
                else:
                    future_vals.append(current_val[0])  # pad with last known value

            result[col] = np.concatenate(
                [current_val, np.array(future_vals, dtype=np.float32)]
            )

        return result
