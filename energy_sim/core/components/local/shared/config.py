from typing import Any
from bems_simulation.core.components.shared.component_config import (
    BaseComponentConfig,
)
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseActuatorConfig:
    """Base actuator configuration."""

    type: str
    space: object


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseLocalComponentConfig(BaseComponentConfig):
    """Local component configuration."""

    model: Any
    actuator: BaseActuatorConfig
