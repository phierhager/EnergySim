from dataclasses import asdict
from energysim.core.components.local.battery.config import (
    OnOffPowerActuatorConfig,
    OnOffPowerBatteryActionSpace,
    SimpleActuatorConfig,
    SimpleBatteryActionSpace,
    PIActuatorConfig,
    PIBatteryActionSpace,
)
from typing import Union
from abc import ABC, abstractmethod
from energysim.core.components.registry import register_actuator
from energysim.core.components.local.shared.actuator import IActuator
from energysim.core.components.shared.spaces import Space
from energysim.core.utils.converter import to_dict_filtered
import numpy as np


class IBatteryActuator(IActuator, ABC):
    @abstractmethod
    def interpret_action(self, action: np.ndarray, max_power: float) -> float:
        """Convert action into requested battery power (specific to battery)."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> dict[str, Space]:
        pass


@register_actuator(SimpleActuatorConfig)
def create_simple_battery_actuator(
    config: SimpleActuatorConfig,
) -> IBatteryActuator:
    return SimpleBatteryActuator(space=config.space)


class SimpleBatteryActuator(IBatteryActuator):
    """Maps actions into requested power for the battery model."""

    def __init__(self, space: SimpleBatteryActionSpace):
        self._space = space

    def interpret_action(self, action: np.ndarray, max_power: float) -> float:
        """Convert action into requested battery power (specific to battery)."""
        self._space.validate_action(action)
        if isinstance(action, np.floating):
            return float(action) * max_power
        elif isinstance(action, np.integer):
            return max_power if action == 1 else -max_power if action == -1 else 0.0
        else:
            raise TypeError("Unsupported action type.")

    @property
    def action_space(self) -> dict[str, Space]:
        return to_dict_filtered(self._space)


@register_actuator(PIActuatorConfig)
def create_pi_battery_actuator(
    config: PIActuatorConfig,
) -> IBatteryActuator:
    return PIBatteryActuator(space=config.space, kp=config.kp, ki=config.ki)


class PIBatteryActuator(IBatteryActuator):
    """
    Converts action using a PI controller to smooth charging/discharging.
    Action is a continuous value [-1,1] representing desired power fraction.
    """

    def __init__(self, space: PIBatteryActionSpace, kp: float = 0.5, ki: float = 0.1):
        self._space = space
        self.kp = kp
        self.ki = ki
        self.integral_error = 0.0

    def interpret_action(self, action: np.ndarray, max_power: float) -> float:
        self._space.validate_action(action)
        error = float(action)  # desired fraction of max_power
        self.integral_error += error
        power_request = self.kp * error + self.ki * self.integral_error
        return np.clip(power_request * max_power, -max_power, max_power)

    @property
    def action_space(self) -> dict[str, Space]:
        return to_dict_filtered(self._space)
