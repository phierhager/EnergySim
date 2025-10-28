from dataclasses import dataclass
from energysim.core.components.sensors import ComponentSensorConfig
from energysim.core.components.spaces import Space
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseComponentConfig:
    type: str
    sensor: ComponentSensorConfig
    model: Any
