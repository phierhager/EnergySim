import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from ..shared.data_structs import ApplianceConfig

# =============================================================================
# 1. Machine State (Unified for PyTree simplicity)
# =============================================================================
class MachineState(eqx.Module):
    """
    Dynamic state for any machine.
    - energy_remaining: (Joules) For smart/shiftable loads. How much work is left?
    - prev_is_available: (0/1) For edge detection. Was it plugged in last step?
    """
    energy_remaining: jnp.ndarray
    prev_is_available: jnp.ndarray

# =============================================================================
# 2. Abstract Base Machine
# =============================================================================
class AbstractMachine(eqx.Module):
    """
    Base for anything that consumes Electricity and emits Heat.
    """
    config: ApplianceConfig
    state: MachineState
    target_node_index: int = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, signal: float, dt: float, availability_mask: float) -> tuple['AbstractMachine', float, float]:
        """
        Returns: (NewModel, Electrical_W, Heat_W)
        """
        raise NotImplementedError

# =============================================================================
# 3. Passive Equipment (Stateless)
# =============================================================================
class PassiveEquipment(AbstractMachine):
    """
    Represents 'Dumb' loads: Lights, Fridge, WiFi, or the 'Ghost' Base Load.
    
    Logic:
    - Power = Nominal_Power * Signal (Utilization)
    - Heat = Power * Convective_Fraction
    - No memory (state is dummy)
    """
    def __init__(self, config: ApplianceConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index
        # Dummy state (zeros) to satisfy type checks
        self.state = MachineState(jnp.array(0.0), jnp.array(0.0))

    @eqx.filter_jit
    def step(self, signal: float, dt: float, availability_mask: float):
        # 1. Electrical Calculation
        # Signal is interpreted as 'utilization fraction' (e.g. 0.5 dimming)
        # Clamp to valid range 0-1
        utilization = jnp.clip(signal, 0.0, 1.0)
        
        # Apply availability (e.g. blackout -> 0 power)
        power_w = self.config.nominal_power_w * utilization * availability_mask
        
        # 2. Thermal Calculation
        # Pure waste heat generation
        heat_w = power_w * self.config.convective_fraction
        
        # Return self (no state update needed)
        return self, power_w, heat_w

# =============================================================================
# 4. Smart Appliance (Stateful / Shiftable)
# =============================================================================
class SmartAppliance(AbstractMachine):
    """
    Represents 'Shiftable' loads: EV Chargers, Washing Machines, Dishwashers.
    
    Logic:
    - Has a 'Job' (Energy Budget in Joules).
    - Detects 'Arrival' (Plug-in event) to reset the budget.
    - Power = Nominal * Signal (Throttle), but ONLY if budget > 0.
    - Stops automatically when budget is exhausted.
    """
    def __init__(self, config: ApplianceConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index
        self.state = MachineState(
            energy_remaining=jnp.array(0.0),
            prev_is_available=jnp.array(0.0)
        )

    @eqx.filter_jit
    def step(self, signal: float, dt: float, availability_mask: float):
        """
        signal: Control throttle (0.0 to 1.0). 1.0 = Charge at max speed.
        availability_mask: 1.0 = Plugged In, 0.0 = Unplugged.
        """
        
        # --- 1. Edge Detection (Arrival Logic) ---
        # Current is 1, Previous was 0 -> Just Arrived (Rising Edge)
        just_arrived = availability_mask * (1.0 - self.state.prev_is_available)

        # --- 2. Manage Energy Budget ---
        # If just arrived -> Refill budget to full capacity (config.total_energy_j)
        # Else -> Keep existing remaining energy
        current_demand_j = jnp.where(
            just_arrived > 0.5,
            self.config.total_cycle_energy_j,
            self.state.energy_remaining
        )

        # --- 3. Calculate Power Draw ---
        # Clamp control signal
        throttle = jnp.clip(signal, 0.0, 1.0)
        
        # Condition: Must be plugged in (avail > 0.5) AND have work left (demand > 0)
        should_run = (current_demand_j > 0) * (availability_mask > 0.5)
        
        power_w = should_run * throttle * self.config.nominal_power_w

        # --- 4. Update Physics ---
        energy_consumed_j = power_w * dt
        
        # Calculate new remaining energy
        # Use max(0, ...) to prevent negative energy if step overshoots slightly
        new_energy_remaining = jnp.maximum(0.0, current_demand_j - energy_consumed_j)
        
        # Safety: If device gets unplugged mid-cycle, we assume the "job" is aborted/paused.
        # Masking ensures if availability=0, remaining energy effectively stored (or cleared depending on preference).
        # Here, we keep the value but it won't discharge because 'should_run' checks mask.
        # However, to reset purely on unplug, one could multiply by availability_mask.
        # We stick to the logic: Unplugging doesn't clear budget, but re-plugging (rising edge) WILL reset it.
        
        heat_w = power_w * self.config.convective_fraction

        # --- 5. Update State ---
        new_state = MachineState(
            energy_remaining=new_energy_remaining,
            prev_is_available=jnp.array(availability_mask) # Store current as prev for next step
        )
        
        # Functional update of the model
        new_model = eqx.tree_at(lambda m: m.state, self, new_state)
        
        return new_model, power_w, heat_w