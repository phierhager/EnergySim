from gymnasium import spaces, Env
import numpy as np
from energysim.core.components.shared.component_base import (
    ComponentBase,
)
from energysim.core.components.shared.component_outputs import (
    ComponentOutputs,
    ElectricalEnergy,
    ThermalEnergy,
)
from energysim.core.thermal.thermal_model_base import ThermalModel
from energysim.core.data.dataset import EnergyDataset
from dataclasses import dataclass
from typing import Mapping, Optional, assert_type, cast
from energysim.core.components.shared.component_config import BaseComponentConfig
from energysim.core.thermal.thermal_model_base import ThermalModelConfig
from energysim.core.data.config import EnergyDatasetConfig
from energysim.rl.rewards.reward_config import RewardConfig
from energysim.rl.rewards.contexts import (
    EconomicContext,
    RewardContext,
)
from energysim.rl.rewards.manager import RewardManager
from energysim.core.components.shared.sensors import Sensor
from enum import StrEnum

from energysim.rl.utils import get_gymnasium_space


class DataColumn(StrEnum):
    LOAD = "load"
    PRICE = "price"
    PV = "pv"


@dataclass
class EnvironmentParameters:
    """Parameters for the modular building environment."""

    random_seed: int


@dataclass(frozen=True)
class EnvironmentConfig:
    """Main configuration for the modular building environment."""

    parameters: EnvironmentParameters
    reward: RewardConfig


class BuildingEnvironment(Env):
    def __init__(
        self,
        components: Mapping[str, ComponentBase],
        comp_sensors: Mapping[str, Sensor],
        thermal_sensor: Sensor,
        dataset: EnergyDataset,
        thermal_model: ThermalModel,
        reward_manager: RewardManager,
        params: EnvironmentParameters,
    ):
        super().__init__()
        self.components = components
        self.comp_sensors = comp_sensors
        self.thermal_sensor = thermal_sensor
        self.dataset = dataset
        self.thermal_model = thermal_model
        self.reward_manager = reward_manager
        self.params = params
        self.economic_context = EconomicContext()

    def reset(self):
        # Initialize components and thermal model
        component_outputs: dict[str, ComponentOutputs] = {}
        for name, component in self.components.items():
            component_outputs[name] = component.initialize()

        self._timestep_index = 0
        data_bundle = self.dataset[self._timestep_index]
        self._timestep_index += 1

        if DataColumn.PRICE in data_bundle:
            self.economic_context.update_price(data_bundle[DataColumn.PRICE][0])

        thermal_state = self.thermal_model.initialize()

        observations = {}
        for sensor_name, sensor in self.comp_sensors.items():
            # NOTE: Sensor names must match component names
            observations[sensor_name] = sensor.read(component_outputs[sensor_name])

        observations["thermal"] = self.thermal_sensor.read(thermal_state)

        return observations, {}

    def step(
        self, action: dict[str, dict[str, np.ndarray]]
    ) -> tuple[dict, float, bool, bool, dict]:
        # Get load, pv, price from dataset for this timestep
        data_bundle = self.dataset[self._timestep_index]
        dt_seconds = data_bundle.dt_seconds
        self._timestep_index += 1

        # Advance components based on actions
        component_outputs: dict[str, ComponentOutputs] = {}
        for name, component in self.components.items():
            if name not in action:
                raise ValueError(f"Missing action for component '{name}'")
            component_outputs[name] = component.advance(
                input=action[name], dt_seconds=dt_seconds
            )

        data_observations = {}
        if DataColumn.LOAD in data_bundle:
            # NOTE: For the observations, we will use this value directly
            load = data_bundle[DataColumn.LOAD]
            component_outputs[DataColumn.LOAD] = ComponentOutputs(
                electrical_energy=ElectricalEnergy(demand_j=load[0]),
                thermal_energy=ThermalEnergy(),
            )
            data_observations[DataColumn.LOAD] = load

        if DataColumn.PV in data_bundle:
            pv = data_bundle[DataColumn.PV]
            component_outputs[DataColumn.PV] = ComponentOutputs(
                electrical_energy=ElectricalEnergy(generation_j=pv[0]),
                thermal_energy=ThermalEnergy(),
            )
            data_observations[DataColumn.PV] = pv
        if DataColumn.PRICE in data_bundle:
            prices = data_bundle[DataColumn.PRICE]
            self.economic_context.update_price(prices[0])
            data_observations[DataColumn.PRICE] = prices

        # Aggregate total energy flows and storage states to update thermal model
        thermal_state = self.thermal_model.advance(
            thermal_energy_j=sum(
                c.thermal_energy.heating_j for c in component_outputs.values()
            )
            - sum(c.thermal_energy.cooling_j for c in component_outputs.values()),
            dt_seconds=dt_seconds,
        )

        # Calculate reward
        context = RewardContext(
            component_outputs=sum(component_outputs.values(), start=ComponentOutputs()),
            thermal_state=thermal_state,
            economic_context=self.economic_context,
        )

        reward, reward_info = self.reward_manager.calculate_reward(context=context)

        # Get observations from sensors
        observations = {}
        for sensor_name, sensor in self.comp_sensors.items():
            # NOTE: Sensor names must match component names
            observations[sensor_name] = sensor.read(component_outputs[sensor_name])
        observations["thermal"] = self.thermal_sensor.read(thermal_state)
        # NOTE: Add dataset observations (also prediction values if available)
        observations["data"] = data_observations

        done = self._timestep_index >= len(self.dataset)

        return observations, reward, done, False, reward_info

    @property
    def action_space(self) -> spaces.Space:
        spaces_dict = {}
        for component_name, component in self.components.items():
            print(
                f"Component '{component_name}' action space: {component.action_space}"
            )
            print(type(component.action_space["action"]))
            spaces_dict[component_name] = get_gymnasium_space(component.action_space)
        return spaces.Dict(spaces_dict)

    @property
    def observation_space(self) -> spaces.Space:
        spaces_dict = {}

        # Add dataset observations
        spaces_dict["data"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dataset.num_features,),
            dtype=np.float32,
        )

        # Add component observation spaces
        for sensor_name, sensor in self.comp_sensors.items():
            spaces_dict[sensor_name] = get_gymnasium_space(sensor.observation_space())

        # Add thermal model observations
        if "thermal" in self.comp_sensors:
            raise ValueError("Thermal sensor must be provided separately.")
        spaces_dict["thermal"] = get_gymnasium_space(
            self.thermal_sensor.observation_space()
        )

        return spaces.Dict(spaces_dict)

    def __repr__(self) -> str:
        return (
            f"<BuildingEnvironment("
            f"components={len(self.components)}, "
            f"sensors={len(self.comp_sensors)}, "
            f"thermal_sensor={'set' if self.thermal_sensor else 'None'}, "
            f"dataset_length={len(self.dataset)}, "
            f"reward_manager={self.reward_manager.__class__.__name__}, "
            f"random_seed={self.params.random_seed}, "
            f"action_space={self.action_space}, "
            f"observation_space={self.observation_space})>"
        )

    __str__ = __repr__
