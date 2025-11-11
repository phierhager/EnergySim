
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Any, Union, TypedDict
import numpy as np
import logging

from energysim.core.components.base import ActionDrivenComponent, ComponentBase, StateAwareComponent
from energysim.core.components.outputs import ComponentOutputs, ElectricalEnergy
from energysim.core.components.sensors import ComponentSensor, Sensor, ComponentSensorOutputs, ThermalSensorOutputs
from energysim.core.shared.spaces import DictSpace, Space
from energysim.core.thermal.state import ThermalState
from energysim.core.thermal.models import ThermalModel
from energysim.core.data.dataset import EnergyDataset
from energysim.core.state import SimulationState
from energysim.core.components.sensors import ComponentSensorConfig

logger = logging.getLogger(__name__)

class ObservationDict(TypedDict):
    thermal: dict[str, ThermalSensorOutputs]
    data: dict[str, np.ndarray]
    components: dict[str, ComponentSensorOutputs]


@dataclass(frozen=True, slots=True)
class SimulationTimestepResult:
    """
    Holds all calculated data for a single completed simulation timestep.
    
    This is the "result" of advancing the simulation by one step.
    """
    # The timestamp this data corresponds to (e.g., timestamp at time t)
    timestamp: int
    
    # The state at the *beginning* of the timestep (state_t)
    simulation_state: SimulationState
    
    # The outputs from components calculated *during* the timestep (outputs_t)
    component_outputs: Dict[str, ComponentOutputs]
    
    # The resulting thermal state *at the end* of the timestep (thermal_state_t)
    final_thermal_state: ThermalState
    
    # The aggregated net energy flows *during* the timestep (balance_t)
    system_balance: ComponentOutputs

    # Observations collected during the timestep
    observations: ObservationDict


class BuildingSimulator:
    """
    A bare, decoupled building simulator.
    
    This class orchestrates the core simulation logic without any reference
    to reinforcement learning, objectives, or observations. It is responsible for:
    
    1.  Managing the simulation's internal state (time, thermal, components).
    2.  Stepping the simulation forward given a set of component actions.
    3.  Aggregating energy flows and advancing the thermal model.
    
    It follows the same three-phase simulation loop as the BuildingEnvironment:
    1.  State Creation: Establish "given" conditions for the timestep.
    2.  Component Simulation: Compute results from controllable components based on actions.
    3.  System Balancing: Aggregate flows and advance the building's thermal state.
    """

    def __init__(
        self,
        components: Mapping[str, ComponentBase],
        comp_sensors: Mapping[str, Sensor],
        thermal_sensor: Sensor,
        dataset: EnergyDataset,
        thermal_model: ThermalModel,
    ):
        # Static elements of the simulation
        self.components = components
        self.dataset = dataset
        self.thermal_model = thermal_model
        self.comp_sensors = comp_sensors
        self.thermal_sensor = thermal_sensor

        # NOTE: External components will always be observed fully
        self.external_sensor = ComponentSensor(ComponentSensorConfig(observe_cooling_flow=True, observe_heating_flow=True, observe_electrical_flow=True, observe_electrical_soc=True, observe_thermal_soc=True, soc_noise_std=0.0, flow_noise_std=0.0))

        # Internal state, managed by step() and reset()
        self._timestep_index: int = 0
        self._current_thermal_state: Optional[ThermalState] = None
        self._current_component_outputs: Optional[Dict[str, ComponentOutputs]] = None
        self.is_done: bool = False

    def reset(self) -> SimulationTimestepResult:
        """
        Resets the simulator to its initial state at t=0.
        
        Initializes all stateful models (components, thermal model) and
        returns the complete state of the system at the very first timestep.
        
        Returns:
            SimulationTimestepResult: The state of the system at t=0.
        """
        # 1. Initialize all stateful models to their starting conditions
        self._current_component_outputs = {
            name: component.initialize()
            for name, component in self.components.items()
        }
        self._current_thermal_state = self.thermal_model.initialize()
        self._timestep_index = 0
        self.is_done = False

        # 2. Create the initial SimulationState for timestep t=0
        initial_timestep_data = self.dataset[self._timestep_index]
        initial_sim_state = SimulationState(
            timestep_data=initial_timestep_data,
            thermal_state=self._current_thermal_state,
            component_outputs=self._current_component_outputs
        )

        # 3. Assemble the initial result package
        # At t=0, the "balance" is just the sum of initial component states (e.g., storage)
        initial_balance = sum(
            self._current_component_outputs.values(), start=ComponentOutputs()
        )
        observations = self._assemble_observation(
            final_thermal_state=self._current_thermal_state,
            endogenous_outputs=self._current_component_outputs,
            simulation_state=initial_sim_state
        )

        initial_result = SimulationTimestepResult(
            timestamp=initial_timestep_data.timestamp,
            simulation_state=initial_sim_state,
            component_outputs=self._current_component_outputs,
            final_thermal_state=self._current_thermal_state,
            system_balance=initial_balance,
            observations=observations
        )

        return initial_result

    def step(self, action: Dict[str, Dict[str, float]], external_component_outputs: Optional[Dict[str, ComponentOutputs]] = None) -> Optional[SimulationTimestepResult]:
        """
        Advances the simulation by one timestep based on the given action.
        
        This method simulates the interval from t to t+1 using the action `action_t`.
        
        Args:
            action: A dictionary mapping component names to their specific
                    action dictionaries (e.g., {"battery": {"normalized_power": -0.5}}).
            external_component_outputs: A dictionary of external endogenous component outputs
                                        that should be included in the system balance.
                    
        Returns:
            Optional[SimulationTimestepResult]:
                - The complete result of the simulation step if successful.
                - None if the simulation has reached the end of the dataset.
        """
        if self.is_done:
            raise RuntimeError(
                "Simulation is finished. Call reset() to start a new episode."
            )
        
        if external_component_outputs is not None:
            for k, v in external_component_outputs.items():
                if not isinstance(v, ComponentOutputs):
                    raise ValueError(
                        f"External component output for '{k}' is not a ComponentOutputs instance."
                    )
            logger.warning("MPC formulations should not be used with external component outputs.")

        # --- PHASE 0: Time Advancement & Termination Check ---
        # This logic follows the BuildingEnvironment, where step(action_t)
        # calculates the state for t+1.
        self._timestep_index += 1
        if self._timestep_index >= len(self.dataset):
            self.is_done = True
            return None  # Signifies the end of the simulation episode

        # --- PHASE 1: State Creation (The "Givens" for interval [t, t+1]) ---
        # Note: This state uses data from t+1 but thermal/component state from t
        current_state = SimulationState(
            timestep_data=self.dataset[self._timestep_index],
            thermal_state=self._current_thermal_state,
            component_outputs=self._current_component_outputs
        )
        dt_seconds = current_state.timestep_data.dt_seconds

        # --- PHASE 2: Component Simulation (The "Computed Results") ---
        # Apply action_t to compute the outputs for the *next* state
        endogenous_outputs: Dict[str, ComponentOutputs] = {}
        for name, component in self.components.items():
            if name not in action:
                raise ValueError(f"Missing action for component '{name}' in step().")
            if isinstance(component, ActionDrivenComponent):
                endogenous_outputs[name] = component.advance(
                    action=action[name],
                    dt_seconds=dt_seconds
                )
            elif isinstance(component, StateAwareComponent):
                endogenous_outputs[name] = component.advance(
                    action=action[name],
                    state=current_state,
                    dt_seconds=dt_seconds
                )

        # Include any external endogenous component outputs
        if external_component_outputs is not None:
            for k, v in external_component_outputs.items():
                endogenous_outputs[f"external_{k}"] = v

        # --- PHASE 3: System Balancing & Finalization ---
        # Aggregate computed endogenous flows with given exogenous flows
        # Exogenous flows are from the *current* timestep's data (t+1)
        exogenous_flows = ComponentOutputs(
            electrical_energy=ElectricalEnergy(
                demand_j=current_state.timestep_data.get("load", np.array([0.0]))[0],
                generation_j=current_state.timestep_data.get("pv", np.array([0.0]))[0]
            )
        )
        system_balance = sum(endogenous_outputs.values(), start=exogenous_flows)
        
        # Advance the thermal model using the final balance and the state
        # (which contains ambient temp, etc. for t+1)
        new_thermal_state = self.thermal_model.advance(
            thermal_energy_j=system_balance.thermal_energy.net_heating,
            ambient_temperature=current_state.timestep_data["ambient_temperature"][0],
            dt_seconds=dt_seconds
        )

        # Collect observations from components
        observations = self._assemble_observation(
            final_thermal_state=new_thermal_state,
            endogenous_outputs=endogenous_outputs,
            simulation_state=current_state
        )

        # --- FINALIZATION ---
        # Update internal state to reflect the end of the interval [t, t+1]
        self._current_thermal_state = new_thermal_state
        self._current_component_outputs = endogenous_outputs

        # Package and return the results for this step
        result = SimulationTimestepResult(
            timestamp=current_state.timestep_data.timestamp,
            simulation_state=current_state,
            component_outputs=endogenous_outputs,
            final_thermal_state=new_thermal_state,
            system_balance=system_balance,
            observations=observations
        )
        
        return result
    
    def _assemble_observation(self, final_thermal_state: ThermalState, endogenous_outputs: Dict[str, ComponentOutputs], simulation_state: SimulationState) -> ObservationDict:
        """Assembles the complete observation dictionary from the final state of a step."""
        return ObservationDict(
            thermal=self.thermal_sensor.read(final_thermal_state),
            data=simulation_state.timestep_data.features,
            components={name: sensor.read(endogenous_outputs[name]) for name, sensor in self.comp_sensors.items()}
        )

    @property
    def current_simulation_state(self) -> SimulationState:
        """
        Returns the complete SimulationState for the current timestep.
        
        This represents the "inputs" to the *next* step() call.
        """
        if self.is_done:
            raise RuntimeError("Simulation is done.")
            
        return SimulationState(
            timestep_data=self.dataset[self._timestep_index],
            thermal_state=self._current_thermal_state,
            component_outputs=self._current_component_outputs
        )

    def __repr__(self) -> str:
        return (
            f"<BuildingSimulator("
            f"components={list(self.components.keys())}, "
            f"thermal_model={self.thermal_model.__class__.__name__}, "
            f"dataset_length={len(self.dataset)}, "
            f"current_step={self._timestep_index}, "
            f"is_done={self.is_done})>"
        )

    @property
    def action_space(self) -> Space:
        """
        Returns the combined action space for all components in the simulator.
        
        The returned structure maps component names to their respective action spaces.
        """
        return DictSpace(
            spaces={
                name: component.action_space
                for name, component in self.components.items()
            }
        )