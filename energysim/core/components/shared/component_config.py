from dataclasses import dataclass
from energysim.core.components.shared.sensors import ComponentSensorConfig
from energysim.core.components.shared.spaces import Space


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseComponentConfig:
    type: str
    sensor: ComponentSensorConfig
