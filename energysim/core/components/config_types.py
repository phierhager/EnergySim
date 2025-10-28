"""Config definitions for components. This is necessary to make dacite able to infer the correct types."""

from typing import Union
from energysim.core.components.battery.config import BatteryComponentConfig

ComponentConfig = Union[BatteryComponentConfig]
"""Union type for all component configurations."""