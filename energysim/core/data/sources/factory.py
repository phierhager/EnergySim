
from energysim.core.data.sources.base import (
    DataSource,
    BaseDataSourceConfig,
)
from energysim.core.data.sources.file import FileDataSource, FileDataSourceConfig
from energysim.core.data.sources.cached import (
    CachedDataSource,
    CachedDataSourceConfig,
)


class DataSourceFactory:
    """Factory to create DataSource instances based on configuration."""

    @staticmethod
    def create(config: BaseDataSourceConfig) -> DataSource:
        """Create a DataSource instance from a DataSourceConfig."""
        if not isinstance(config, BaseDataSourceConfig):
            raise ValueError("config must be an instance of DataSourceConfig")
        if isinstance(config, FileDataSourceConfig):
            return FileDataSource(config)
        elif isinstance(config, CachedDataSourceConfig):
            # Extract inner wrapped source config
            wrapped_conf: BaseDataSourceConfig = config.wrapped_source_config
            assert not isinstance(wrapped_conf, CachedDataSourceConfig), (
                "Nested CachedDataSourceConfig is not supported"
            )
            cache_size: int = config.cache_size
            source = DataSourceFactory.create(wrapped_conf)
            return CachedDataSource(source, cache_size=cache_size)
        else:
            raise ValueError(f"Unsupported DataSourceConfig type: {type(config)}")
