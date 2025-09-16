from dataclasses import dataclass
from typing import Optional
from energysim.core.data.sources.config import DataSourceConfig


@dataclass(frozen=True)
class EnergyDatasetParams:
    """Configuration parameters for the DataProvider."""

    feature_columns: dict[str, str]  # e.g., {"price": "price_col", "pv": "pv_col"}
    dt_seconds: int
    use_time_features: bool
    prediction_horizon: Optional[int] = None
    """Prediction horizon in dt_seconds; None means no prediction requested."""


@dataclass(frozen=True)
class EnergyDatasetConfig:
    data_source: DataSourceConfig
    params: EnergyDatasetParams
