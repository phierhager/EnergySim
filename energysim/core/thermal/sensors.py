from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict
import numpy as np
from energysim.core.sensors import Sensor
from energysim.core.thermal.state import ThermalState
from energysim.core.shared.spaces import DictSpace, ContinuousSpace

@dataclass
class ThermalSensorConfig:
    observe_indoor_temp: bool = True
    observe_temp_error: bool = False
    observe_comfort_violation: bool = False
    observe_zone_temps: bool = False
    temp_noise_std: float = 0.0


# -------------------------------
# Thermal Sensor
# -------------------------------
class ThermalSensorOutputs(TypedDict):
    temperature: Optional[float]
    temperature_error: Optional[float]
    comfort_violation: Optional[float]
    zone_temperatures: Optional[dict[str, float]]
    
class ThermalSensor(Sensor):
    """Robust thermal sensor for single or multi-zone buildings."""

    def __init__(self, config: ThermalSensorConfig):
        self.config = config

    def read(self, thermal_state: ThermalState) -> dict:
        obs = {}

        if self.config.observe_indoor_temp:
            obs["temperature"] = float(thermal_state.temperature + np.random.normal(
                0, self.config.temp_noise_std
            ))

        if self.config.observe_temp_error:
            obs["temperature_error"] = float(thermal_state.temperature_error)

        if self.config.observe_comfort_violation:
            obs["comfort_violation"] = float(thermal_state.comfort_violation)

        if self.config.observe_zone_temps:
            if thermal_state.zone_temperatures is None:
                raise ValueError("Zone temperatures requested but not available.")
            for zone, temp in thermal_state.zone_temperatures.items():
                obs[f"temp_{zone}"] = float(temp + np.random.normal(
                    0, self.config.temp_noise_std
                ))

        return obs

    @property
    def observation_space(
        self, thermal_state: Optional[ThermalState] = None
    ) -> DictSpace:
        observation_space = {}

        if self.config.observe_indoor_temp:
            observation_space["temperature"] = ContinuousSpace(
                lower_bound=-50.0, upper_bound=50.0
            )
        if self.config.observe_temp_error:
            observation_space["temperature_error"] = ContinuousSpace(
                lower_bound=0.0, upper_bound=50.0
            )
        if self.config.observe_comfort_violation:
            observation_space["comfort_violation"] = ContinuousSpace(
                lower_bound=0.0, upper_bound=1.0
            )
        if self.config.observe_zone_temps:
            if thermal_state is None or thermal_state.zone_temperatures is None:
                raise ValueError(
                    "ThermalState must be provided to determine number of zones."
                )
            for zone, _ in thermal_state.zone_temperatures.items():
                observation_space[f"temp_{zone}"] = ContinuousSpace(
                    lower_bound=-50.0, upper_bound=50.0
                )

        return DictSpace(spaces=observation_space)