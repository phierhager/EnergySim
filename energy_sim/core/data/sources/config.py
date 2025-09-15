from .cached import CachedDataSourceConfig
from .file import FileDataSourceConfig

from typing import Union

DataSourceConfig = Union[FileDataSourceConfig, CachedDataSourceConfig]
