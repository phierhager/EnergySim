"""Config definitions for local components. This is necessary to make dacite able to infer the correct types."""

from .local.config import LocalComponentConfig
from .remote.config import RemoteComponentConfig

ComponentConfig = LocalComponentConfig | RemoteComponentConfig
