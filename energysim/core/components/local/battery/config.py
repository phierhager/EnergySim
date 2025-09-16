from typing import ClassVar, Literal, TypeVar, Union, overload
from energysim.core.components.local.shared.config import BaseLocalComponentConfig
from dataclasses import dataclass, field
from energysim.core.components.shared.spaces import (
    ContinuousSpace,
    DiscreteSpace,
    Space,
)


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
class SimpleBatteryActionSpace:
    action: Space

    def validate_action(self, action) -> None:
        self.action.validate_action(action)


@dataclass(frozen=True, slots=True, kw_only=True)
class SimpleActuatorConfig:
    space: SimpleBatteryActionSpace
    type: Literal["simple"] = "simple"


@dataclass(frozen=True, slots=True, kw_only=True)
class PIBatteryActionSpace:
    action: ContinuousSpace

    def validate_action(self, action) -> None:
        self.action.validate_action(action)


@dataclass(frozen=True, slots=True, kw_only=True)
class PIActuatorConfig:
    space: PIBatteryActionSpace
    kp: float = 0.5
    ki: float = 0.1
    type: Literal["pi"] = "pi"


@dataclass(frozen=True, slots=True, kw_only=True)
class OnOffPowerBatteryActionSpace:
    max_power: ContinuousSpace
    on: DiscreteSpace = field(
        default_factory=lambda: DiscreteSpace(n_actions=2)
    )  # 0: off, 1: on

    def validate_action(self, max_power, on) -> None:
        self.max_power.validate_action(max_power)
        self.on.validate_action(on)


@dataclass(frozen=True, slots=True, kw_only=True)
class OnOffPowerActuatorConfig:
    space: OnOffPowerBatteryActionSpace
    type: Literal["on_off_power"] = "on_off_power"


BatteryActuatorConfig = Union[
    SimpleActuatorConfig, PIActuatorConfig, OnOffPowerActuatorConfig
]


@dataclass(frozen=True, slots=True, kw_only=True)
class BatteryComponentConfig(BaseLocalComponentConfig):
    model: BatteryModelConfig
    actuator: BatteryActuatorConfig
    # the sensor is inherited from BaseComponentConfig

    type: Literal["battery"] = "battery"
