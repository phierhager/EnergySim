from energysim.core.data.sources.factory import DataSourceFactory
from energysim.core.data.config import EnergyDatasetConfig
from energysim.core.data.dataset import EnergyDataset


def build_dataset(config: EnergyDatasetConfig) -> EnergyDataset:
    """Build an energy dataset from the given configuration."""
    data_source = DataSourceFactory.create(config.data_source)
    return EnergyDataset(data_source, config.params)
