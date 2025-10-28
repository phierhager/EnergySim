from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from energysim.core.components.base import ComponentOutputs
from energysim.core.thermal.thermal_model_base import ThermalState
from energysim.core.components.spaces import (
    DictSpace,
    Space,
    ContinuousSpace,
)


# -------------------------------
# Abstract Sensor Interface
# -------------------------------
class Sensor(ABC):
    """Abstract interface for all sensors."""

    @abstractmethod
    def read(self, *args: Any) -> dict:
        """Return the current observation vector from this sensor."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Return the observation space corresponding to this sensor."""
        pass


# -------------------------------
# Sensor Configuration Data Classes
# -------------------------------
@dataclass
class ThermalSensorConfig:
    observe_indoor_temp: bool = True
    observe_temp_error: bool = False
    observe_comfort_violation: bool = False
    observe_zone_temps: bool = False
    temp_noise_std: float = 0.0


@dataclass
class ComponentSensorConfig:
    observe_electrical_soc: bool = False
    observe_thermal_soc: bool = False
    observe_electrical_flow: bool = False
    observe_heating_flow: bool = False
    observe_cooling_flow: bool = False
    soc_noise_std: float = 0.0
    flow_noise_std: float = 0.0


# -------------------------------
# Component Sensor
# -------------------------------
class ComponentSensor(Sensor):
    """Sensor for observing component/battery outputs."""

    def __init__(self, config: ComponentSensorConfig):
        self.config = config

    def read(self, comp_out: ComponentOutputs) -> dict:
        obs = {}

        # SOC observations
        if self.config.observe_electrical_soc:
            soc = comp_out.electrical_storage.soc
            soc += np.random.normal(0, self.config.soc_noise_std)
            obs["electrical_soc"] = np.clip(soc, 0.0, 1.0)

        if self.config.observe_thermal_soc:
            soc = comp_out.thermal_storage.soc
            soc += np.random.normal(0, self.config.soc_noise_std)
            obs["thermal_soc"] = np.clip(soc, 0.0, 1.0)

        # Energy flow observations
        if self.config.observe_electrical_flow:
            net_flow = (
                comp_out.electrical_energy.generation_j
                - comp_out.electrical_energy.demand_j
            )
            net_flow += np.random.normal(0, self.config.flow_noise_std)
            obs["electrical_flow"] = net_flow

        if self.config.observe_heating_flow:
            flow = comp_out.thermal_energy.heating_j
            flow += np.random.normal(0, self.config.flow_noise_std)
            obs["heating_flow"] = flow

        if self.config.observe_cooling_flow:
            flow = comp_out.thermal_energy.cooling_j
            flow += np.random.normal(0, self.config.flow_noise_std)
            obs["cooling_flow"] = flow

        return obs

    @property
    def observation_space(self) -> DictSpace:
        observation_space = {}
        MAX_FLOW = 1e6
        if self.config.observe_electrical_soc:
            observation_space["electrical_soc"] = ContinuousSpace(
                lower_bound=0.0, upper_bound=1.0
            )
        if self.config.observe_thermal_soc:
            observation_space["thermal_soc"] = ContinuousSpace(
                lower_bound=0.0, upper_bound=1.0
            )
        if self.config.observe_electrical_flow:
            observation_space["electrical_flow"] = ContinuousSpace(
                lower_bound=-1e6, upper_bound=MAX_FLOW
            )  # example bounds in Joules
        if self.config.observe_heating_flow:
            observation_space["heating_flow"] = ContinuousSpace(
                lower_bound=0.0, upper_bound=MAX_FLOW
            )  # example upper bound in Joules
        if self.config.observe_cooling_flow:
            observation_space["cooling_flow"] = ContinuousSpace(
                lower_bound=0.0, upper_bound=MAX_FLOW
            )  # example upper bound in Joules

        return DictSpace(spaces=observation_space)


# -------------------------------
# Thermal Sensor
# -------------------------------
class ThermalSensor(Sensor):
    """Robust thermal sensor for single or multi-zone buildings."""

    def __init__(self, config: ThermalSensorConfig):
        self.config = config

    def read(self, thermal_state: ThermalState) -> dict:
        obs = {}

        if self.config.observe_indoor_temp:
            obs["temperature"] = thermal_state.temperature + np.random.normal(
                0, self.config.temp_noise_std
            )

        if self.config.observe_temp_error:
            obs["temperature_error"] = thermal_state.temperature_error

        if self.config.observe_comfort_violation:
            obs["comfort_violation"] = float(thermal_state.comfort_violation)

        if self.config.observe_zone_temps:
            if thermal_state.zone_temperatures is None:
                raise ValueError("Zone temperatures requested but not available.")
            for zone, temp in thermal_state.zone_temperatures.items():
                obs[f"temp_{zone}"] = temp + np.random.normal(
                    0, self.config.temp_noise_std
                )

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
