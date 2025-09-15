from dataclasses import dataclass
from bems_simulation.core.components.shared.sensors import ComponentSensorConfig
from bems_simulation.core.components.shared.spaces import Space


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseComponentConfig:
    type: str
    sensor: ComponentSensorConfig
