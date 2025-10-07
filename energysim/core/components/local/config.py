"""Config definitions for local components. This is necessary to make dacite able to infer the correct types."""

from typing import Union
from .battery.config import BatteryComponentConfig

LocalComponentConfig = Union[BatteryComponentConfig]
