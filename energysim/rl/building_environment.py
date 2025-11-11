from gymnasium import spaces, Env
import numpy as np
from energysim.core.components.base import (
    ComponentBase,
)
from energysim.core.components.outputs import (
    ComponentOutputs,
    ElectricalEnergy,
)
from energysim.core.thermal.state import ThermalState
from energysim.core.thermal.base import ThermalModel
from energysim.core.data.dataset import EnergyDataset
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple
from energysim.reward.contexts import (
    RewardContext,
)
from energysim.reward.manager import RewardManager
from energysim.core.components.sensors import Sensor
from energysim.core.state import SimulationState
from energysim.rl.data_column import DataColumn

from energysim.rl.utils import get_gymnasium_space

@dataclass
class EnvironmentParameters:
    """Parameters for the modular building environment."""

    random_seed: int

class BuildingEnvironment(Env):
    """
    A Gymnasium environment for simulating building energy systems.
    
    This environment follows a three-phase simulation loop:
    1.  State Creation: The world's conditions (weather, price, etc.) are established.
    2.  Component Simulation: Controllable components react to the state and an action.
    3.  System Balancing: All energy flows are aggregated to determine the final system
        state, from which rewards and observations are calculated.
    """
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
        # Static elements of the environment
        self.controllable_components = components
        self.comp_sensors = comp_sensors
        self.thermal_sensor = thermal_sensor
        self.dataset = dataset
        self.thermal_model = thermal_model
        self.reward_manager = reward_manager
        self.params = params

        # Internal state, managed by step() and reset()
        self._timestep_index: int = 0
        self._current_thermal_state: Optional[ThermalState] = None
        self._current_component_outputs: Optional[Dict[str, ComponentOutputs]] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment to its initial state for a new episode.
        """
        super().reset(seed=seed)

        # 1. Initialize all stateful models to their starting conditions
        self._current_component_outputs = {
            name: component.initialize() for name, component in self.controllable_components.items()
        }
        self._current_thermal_state = self.thermal_model.initialize()
        self._timestep_index = 0

        # 2. Create the initial SimulationState for timestep t=0
        initial_state = SimulationState(
            timestep_data=self.dataset[self._timestep_index],
            thermal_state=self._current_thermal_state,
            component_outputs=self._current_component_outputs
        )

        # 3. Assemble the initial observation
        # For reset, the "final" state is the same as the initial state.
        observation = self._assemble_observation(
            final_thermal_state=self._current_thermal_state,
            endogenous_outputs=self._current_component_outputs,
            simulation_state=initial_state
        )
        info = {}

        return observation, info

    def step(self, action: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advances the environment by one timestep based on the given action."""

        # --- PHASE 0: Time Advancement & Termination Check ---
        self._timestep_index += 1
        terminated = self._timestep_index >= len(self.dataset)
        if terminated:
            # Create a dummy observation if we step past the end
            last_obs, info = self.reset(seed=self.params.random_seed)
            return last_obs, 0.0, True, False, {"status": "Episode terminated at dataset end"}

        # --- PHASE 1: State Creation (The "Givens") ---
        current_state = SimulationState(
            timestep_data=self.dataset[self._timestep_index],
            thermal_state=self._current_thermal_state,
            component_outputs=self._current_component_outputs
        )
        dt_seconds = current_state.timestep_data.dt_seconds

        # --- PHASE 2: Component Simulation (The "Computed Results") ---
        endogenous_outputs: Dict[str, ComponentOutputs] = {}
        for name, component in self.controllable_components.items():
            component_action = {k: float(v) for k, v in action.get(name, {}).items()}
            endogenous_outputs[name] = component.advance(
                action=component_action,
                state=current_state,
                dt_seconds=dt_seconds
            )

        # --- PHASE 3: System Balancing & Finalization ---
        # Aggregate computed endogenous flows with given exogenous flows
        exogenous_flows = ComponentOutputs(
            electrical_energy=ElectricalEnergy(
                demand_j=current_state.timestep_data.features.get(DataColumn.LOAD)[0],
                generation_j=current_state.timestep_data.features.get(DataColumn.PV)[0]
            )
        )
        system_balance = sum(endogenous_outputs.values(), start=exogenous_flows)
        
        # Advance the thermal model using the final balance
        new_thermal_state = self.thermal_model.advance(
            thermal_energy_j=system_balance.thermal_energy.net_heating,
            ambient_temperature=current_state.timestep_data["ambient_temperature"][0],
            dt_seconds=dt_seconds
        )

        # Calculate reward and info based on the complete step transition
        reward, info = self._calculate_reward_and_info(
            system_balance=system_balance,
            final_thermal_state=new_thermal_state,
            initial_state=current_state
        )

        # --- FINALIZATION ---
        # Update internal state for the next step (t+1)
        self._current_thermal_state = new_thermal_state
        self._current_component_outputs = endogenous_outputs

        # Assemble the observation for the agent to see at the start of t+1
        observation = self._assemble_observation(
            final_thermal_state=new_thermal_state,
            endogenous_outputs=endogenous_outputs,
            simulation_state=current_state
        )
        
        truncated = False # Can be extended with other episode truncation logic
        return observation, reward, terminated, truncated, info

    def _assemble_observation(self, final_thermal_state: ThermalState, endogenous_outputs: Dict[str, ComponentOutputs], simulation_state: SimulationState) -> Dict[str, Any]:
        """Assembles the complete observation dictionary from the final state of a step."""
        observations = {}

        # Get observations from controllable component sensors
        for name, sensor in self.comp_sensors.items():
            # Use the newly computed outputs for the observation
            observations[name] = sensor.read(endogenous_outputs[name])

        # Get observation from the thermal model sensor
        observations["thermal"] = self.thermal_sensor.read(final_thermal_state)
        
        # Add observations from the external dataset (which can include forecasts)
        observations["data"] = simulation_state.timestep_data.features
        
        return observations

    def _calculate_reward_and_info(self, system_balance: ComponentOutputs, final_thermal_state: ThermalState, initial_state: SimulationState) -> Tuple[float, Dict[str, Any]]:
        """Calculates the reward and constructs the info dictionary for the step."""
        
        # Create a rich context for the reward manager
        reward_context = RewardContext(
            simulation_state=initial_state,
            system_balance=system_balance,
            thermal_state=final_thermal_state
        )
        reward, reward_info = self.reward_manager.calculate_reward(context=reward_context)

        # Assemble the info dictionary for logging and analysis
        info = {
            "reward_breakdown": reward_info,
            "grid_interaction_j": system_balance.electrical_energy.net,
            "temperature_c": final_thermal_state.temperature,
            "temperature_error_c": final_thermal_state.temperature_error,
            "timestamp": initial_state.timestep_data.timestamp,
        }
        
        return reward, info

    @property
    def action_space(self) -> spaces.Space:
        spaces_dict = {}
        for component_name, component in self.controllable_components.items():
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
            spaces_dict[sensor_name] = get_gymnasium_space(sensor.observation_space)

        # Add thermal model observations
        if "thermal" in self.comp_sensors:
            raise ValueError("Thermal sensor must be provided separately.")
        spaces_dict["thermal"] = get_gymnasium_space(
            self.thermal_sensor.observation_space
        )

        return spaces.Dict(spaces_dict)

    def __repr__(self) -> str:
        return (
            f"<BuildingEnvironment("
            f"components={len(self.controllable_components)}, "
            f"sensors={len(self.comp_sensors)}, "
            f"thermal_sensor={'set' if self.thermal_sensor else 'None'}, "
            f"dataset_length={len(self.dataset)}, "
            f"reward_manager={self.reward_manager.__class__.__name__}, "
            f"random_seed={self.params.random_seed}, "
            f"action_space={self.action_space}, "
            f"observation_space={self.observation_space})>"
        )

    __str__ = __repr__
