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
from energysim.core.data.data_bundle import DataBundle
from energysim.core.thermal.state import ThermalState
from energysim.core.thermal.thermal_model_base import ThermalModel
from energysim.core.data.dataset import EnergyDataset
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, assert_type, cast
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

    def _initialize_simulation_state(self) -> Tuple[Dict[str, 'ComponentOutputs'], 'ThermalState']:
        """Initializes all components and the thermal model for a new episode."""
        component_outputs: Dict[str, 'ComponentOutputs'] = {
            name: component.initialize() for name, component in self.components.items()
        }
        thermal_state = self.thermal_model.initialize()
        return component_outputs, thermal_state

    def _get_current_data_bundle(self) -> DataBundle:
        """Retrieves the data for the current timestep."""
        if self._timestep_index >= len(self.dataset):
            raise IndexError("Attempted to access data beyond the dataset length.")
        return self.dataset[self._timestep_index]

    def _update_state_with_exogenous_data(
        self,
        data_bundle: DataBundle,
        component_outputs: Dict[str, 'ComponentOutputs'],
    ) -> Dict[str, np.ndarray]:
        """
        Updates the state based on external data (load, PV, price) for the current step.
        
        Note: This function modifies the `component_outputs` dictionary in-place by adding
        entries for non-controllable data sources like PV and Load.
        """
        data_observations: Dict[str, np.ndarray] = {}

        if DataColumn.LOAD in data_bundle:
            load = data_bundle[DataColumn.LOAD]
            component_outputs[DataColumn.LOAD] = ComponentOutputs(
                electrical_energy=ElectricalEnergy(demand_j=load[0])
            )
            data_observations[DataColumn.LOAD] = load

        if DataColumn.PV in data_bundle:
            pv = data_bundle[DataColumn.PV]
            component_outputs[DataColumn.PV] = ComponentOutputs(
                electrical_energy=ElectricalEnergy(generation_j=pv[0])
            )
            data_observations[DataColumn.PV] = pv

        if DataColumn.PRICE in data_bundle:
            prices = data_bundle[DataColumn.PRICE]
            self.economic_context.update_price(prices[0])
            data_observations[DataColumn.PRICE] = prices
            
        return data_observations

    def _get_current_observations(
        self,
        component_outputs: Dict[str, 'ComponentOutputs'],
        thermal_state: ThermalState,
        data_observations: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Assembles the complete observation dictionary from all sources."""
        observations = {}
        # Get observations from controllable component sensors
        for sensor_name, sensor in self.comp_sensors.items():
            # NOTE: Sensor names must match component names
            observations[sensor_name] = sensor.read(component_outputs[sensor_name])

        # Get observation from the thermal model sensor
        observations["thermal"] = self.thermal_sensor.read(thermal_state)
        
        # Add observations from the external dataset (load, pv, price, etc.)
        observations["data"] = data_observations

        return observations

    # --- Private Helper Methods for Step Logic ---

    def _advance_controllable_components(
        self, action: Dict[str, Dict[str, np.ndarray]], dt_seconds: float
    ) -> Dict[str, 'ComponentOutputs']:
        """Advances all controllable components based on the agent's action."""
        component_outputs: Dict[str, 'ComponentOutputs'] = {}
        for name, component in self.components.items():
            if name not in action:
                raise ValueError(f"Missing action for component '{name}'")
            
            component_action = {k: float(v) for k, v in action[name].items()}
            component_outputs[name] = component.advance(
                input=component_action, dt_seconds=dt_seconds
            )
        return component_outputs

    def _advance_thermal_model(
        self, component_outputs: Dict[str, 'ComponentOutputs'], dt_seconds: float
    ) -> 'ThermalState':
        """Advances the thermal model using the aggregated thermal energy flows."""
        total_heating_j = sum(
            c.thermal_energy.heating_j for c in component_outputs.values()
        )
        total_cooling_j = sum(
            c.thermal_energy.cooling_j for c in component_outputs.values()
        )
        net_thermal_energy_j = total_heating_j - total_cooling_j
        
        return self.thermal_model.advance(
            thermal_energy_j=net_thermal_energy_j, dt_seconds=dt_seconds
        )

    def _calculate_reward_and_info(
        self,
        component_outputs: Dict[str, 'ComponentOutputs'],
        thermal_state: 'ThermalState',
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """Calculates the reward and constructs the info dictionary."""
        total_component_outputs = sum(component_outputs.values(), start=ComponentOutputs())
        
        # Calculate reward
        reward_context = RewardContext(
            component_outputs=total_component_outputs,
            thermal_state=thermal_state,
            economic_context=self.economic_context,
        )
        reward, reward_info = self.reward_manager.calculate_reward(context=reward_context)

        # Assemble info dictionary
        info = {
            "reward": reward_info,
            "energy_consumption": total_component_outputs.electrical_energy.net,
            "temperature_error": thermal_state.temperature_error,
        }
        
        return reward, reward_info, info

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment to its initial state for a new episode.
        """
        super().reset(seed=seed, options=options)

        # 1. Reset internal simulation state (components, thermal model)
        component_outputs, thermal_state = self._initialize_simulation_state()
        
        # 2. Reset time to the beginning of the dataset
        self._timestep_index = 0
        
        # 3. Get the initial data bundle (t=0)
        initial_data_bundle = self._get_current_data_bundle()
        
        # 4. Process exogenous data to update state and get data observations
        data_observations = self._update_state_with_exogenous_data(
            data_bundle=initial_data_bundle,
            component_outputs=component_outputs,
        )

        # 5. Assemble the complete initial observation
        observations = self._get_current_observations(
            component_outputs=component_outputs,
            thermal_state=thermal_state,
            data_observations=data_observations,
        )
        print()
        print()
        print()
        print(f"Initial observation: {observations}")
        # The info dict is typically empty on reset
        info = {}

        return observations, info

    def step(
        self, action: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Advances the environment by one timestep based on the given action.
        """
        # 1. Get data for the current timestep
        data_bundle = self._get_current_data_bundle()
        dt_seconds = data_bundle.dt_seconds

        # 2. Advance controllable components based on agent's action
        component_outputs = self._advance_controllable_components(action, dt_seconds)

        # 3. Process this timestep's exogenous data (load, PV, price)
        #    This adds their energy contributions to the `component_outputs` dict
        data_observations = self._update_state_with_exogenous_data(
            data_bundle=data_bundle,
            component_outputs=component_outputs,
        )

        # 4. Advance the thermal model using the combined thermal effects
        thermal_state = self._advance_thermal_model(component_outputs, dt_seconds)
        
        # 5. Calculate reward and supplemental info for this step
        reward, reward_info, info = self._calculate_reward_and_info(
            component_outputs, thermal_state
        )

        # 6. Assemble the observation for the *next* state
        observations = self._get_current_observations(
            component_outputs, thermal_state, data_observations
        )

        # 7. Advance the simulation time
        self._timestep_index += 1

        # 8. Determine if the episode is finished
        terminated = self._timestep_index >= len(self.dataset)
        truncated = False  # Assuming no other truncation conditions for now

        return observations, reward, terminated, truncated, info


    @property
    def action_space(self) -> spaces.Space:
        spaces_dict = {}
        for component_name, component in self.components.items():
            spaces_dict[component_name] = get_gymnasium_space(component.action_space)
        return spaces.Dict(spaces_dict)

    @property
    def observation_space(self) -> spaces.Space:
        spaces_dict = {}

        # Add dataset observations
        spaces_dict["data"] = spaces.Dict({
            col: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1 + (self.dataset.params.prediction_horizon or 0),),
                dtype=np.float32,
            )
            for col in [DataColumn.LOAD, DataColumn.PV, DataColumn.PRICE] if col in self.dataset.params.feature_columns
        })

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
