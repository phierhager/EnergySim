from .local.config import LocalComponentConfig
from .remote.config import RemoteComponentConfig

ComponentConfig = LocalComponentConfig | RemoteComponentConfig
