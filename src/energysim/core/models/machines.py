# energysim/core/models/machines.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ApplianceConfig, Array

# --- Protocol: DiscreteModel ---
# step(state, inputs, dt) -> (NewState, OutputTuple)

class MachineState(eqx.Module):
    energy_remaining: Array
    prev_is_available: Array

class MachineOutput(eqx.Module):
    electrical_power_w: Array
    heat_gain_w: Array

class AbstractMachine(eqx.Module):
    config: ApplianceConfig
    target_node_index: int = eqx.field(static=True)
    
    def step(self, state: MachineState, signal: float, dt: float, availability: float) -> tuple[MachineState, MachineOutput]:
        raise NotImplementedError

# --- Implementations ---

class PassiveEquipment(AbstractMachine):
    def __init__(self, config: ApplianceConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index

    def step(self, state: MachineState, signal: float, dt: float, availability: float) -> tuple[MachineState, MachineOutput]:
        # Algebraic calculation
        utilization = jnp.clip(signal, 0.0, 1.0)
        power_w = self.config.nominal_power_w * utilization * availability
        heat_w = power_w * self.config.convective_fraction
        
        # State doesn't change for passive, return as-is
        return state, MachineOutput(power_w, heat_w)

class SmartAppliance(AbstractMachine):
    def __init__(self, config: ApplianceConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index

    def step(self, state: MachineState, signal: float, dt: float, availability: float) -> tuple[MachineState, MachineOutput]:
        # 1. Edge Detection
        just_arrived = availability * (1.0 - state.prev_is_available)
        
        # 2. Logic (Discrete)
        current_demand = jnp.where(
            just_arrived > 0.5, 
            self.config.total_cycle_energy_j, 
            state.energy_remaining
        )
        
        throttle = jnp.clip(signal, 0.0, 1.0)
        should_run = (current_demand > 0) * (availability > 0.5)
        power_w = should_run * throttle * self.config.nominal_power_w
        
        # 3. Physics Update (Integration Step performed internally for discrete model)
        energy_consumed = power_w * dt
        new_energy = jnp.maximum(0.0, current_demand - energy_consumed)
        
        # 4. Pack Output
        heat_w = power_w * self.config.convective_fraction
        
        new_state = MachineState(energy_remaining=new_energy, prev_is_available=jnp.array(availability))
        return new_state, MachineOutput(power_w, heat_w)