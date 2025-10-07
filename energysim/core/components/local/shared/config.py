from typing import Any
from energysim.core.components.shared.component_config import (
    BaseComponentConfig,
)
from dataclasses import dataclass

@dataclass(frozen=True, slots=True, kw_only=True)
class BaseLocalComponentConfig(BaseComponentConfig):
    """Local component configuration."""

    model: Any
    # the sensor is inherited from BaseComponentConfig

