from bems_simulation.core.components.shared.component_config import BaseComponentConfig
from dataclasses import dataclass
from bems_simulation.core.components.shared.spaces import Space


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseRemoteComponentConfig(BaseComponentConfig):
    """Remote component configuration."""

    connection: object
    action_space: dict[str, Space]
