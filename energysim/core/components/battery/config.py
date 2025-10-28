from typing import Literal, Union
from energysim.core.components.config_base import BaseComponentConfig
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class SimpleBatteryModelConfig:
    """Simple battery configuration."""

    efficiency: float = 1.0
    max_power: float = 1.0
    capacity: float = 1.0
    init_soc: float = 0.0
    deadband: float = 0.0

    type: Literal["simple"] = "simple"

    def __post_init__(self):
        if not (0 < self.efficiency <= 1):
            raise ValueError("Efficiency must be between 0 and 1.")
        if not (0 <= self.init_soc <= 1):
            raise ValueError("Initial SoC must be between 0 and 1.")
        if self.max_power <= 0 or self.capacity <= 0:
            raise ValueError("Max power and capacity must be positive.")
        if self.deadband < 0:
            raise ValueError("Deadband must be non-negative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class DegradingBatteryModelConfig:
    """Battery configuration with capacity degradation."""

    efficiency: float = 1.0
    max_power: float = 1.0
    capacity: float = 1.0
    init_soc: float = 0.0
    deadband: float = 0.0

    # degradation settings
    degradation_mode: Literal["linear", "exponential", "polynomial"] = "linear"
    degradation_rate: float = 0.001  # per unit throughput or cycle
    min_capacity_fraction: float = (
        0.5  # cannot degrade below this fraction of initial capacity
    )
    poly_exponent: float = 2.0  # used if degradation_mode = "polynomial"

    type: Literal["degrading"] = "degrading"

    def __post_init__(self):
        if not (0 < self.efficiency <= 1):
            raise ValueError("Efficiency must be between 0 and 1.")
        if not (0 <= self.init_soc <= 1):
            raise ValueError("Initial SoC must be between 0 and 1.")
        if self.max_power <= 0 or self.capacity <= 0:
            raise ValueError("Max power and capacity must be positive.")
        if self.deadband < 0:
            raise ValueError("Deadband must be non-negative.")
        if not (0 < self.min_capacity_fraction <= 1):
            raise ValueError("min_capacity_fraction must be between 0 and 1.")
        if self.degradation_rate < 0:
            raise ValueError("Degradation rate must be non-negative.")
        if self.degradation_mode == "polynomial" and self.poly_exponent <= 0:
            raise ValueError("Polynomial exponent must be positive.")


BatteryModelConfig = Union[SimpleBatteryModelConfig, DegradingBatteryModelConfig]

@dataclass(frozen=True, slots=True, kw_only=True)
class BatteryComponentConfig(BaseComponentConfig):
    model: BatteryModelConfig
    type: Literal["battery"] = "battery"
