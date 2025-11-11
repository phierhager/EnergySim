from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, TypedDict
import numpy as np

from energysim.core.components.base import ComponentOutputs
from energysim.core.thermal.base import ThermalState
from energysim.core.shared.spaces import (
    DictSpace,
    ContinuousSpace,
)
from energysim.core.shared.sensors import Sensor

# -------------------------------
# Sensor Configuration Data Classes
# -------------------------------

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

@dataclass
class ComponentSensorOutputs(TypedDict):
    electrical_soc: Optional[float]
    thermal_soc: Optional[float]
    electrical_flow: Optional[float]
    heating_flow: Optional[float]
    cooling_flow: Optional[float]


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
            obs["electrical_soc"] = float(np.clip(soc, 0.0, 1.0))

        if self.config.observe_thermal_soc:
            soc = comp_out.thermal_storage.soc
            soc += np.random.normal(0, self.config.soc_noise_std)
            obs["thermal_soc"] = float(np.clip(soc, 0.0, 1.0))

        # Energy flow observations
        if self.config.observe_electrical_flow:
            net_flow = (
                comp_out.electrical_energy.generation_j
                - comp_out.electrical_energy.demand_j
            )
            net_flow += np.random.normal(0, self.config.flow_noise_std)
            obs["electrical_flow"] = float(net_flow)

        if self.config.observe_heating_flow:
            flow = comp_out.thermal_energy.heating_j
            flow += np.random.normal(0, self.config.flow_noise_std)
            obs["heating_flow"] = float(flow)

        if self.config.observe_cooling_flow:
            flow = comp_out.thermal_energy.cooling_j
            flow += np.random.normal(0, self.config.flow_noise_std)
            obs["cooling_flow"] = float(flow)

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
