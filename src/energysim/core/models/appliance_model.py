import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ApplianceConfig

Array = jnp.ndarray

class SmartApplianceState(eqx.Module):
    energy_remaining_j: Array  # How much work is left
    prev_is_available: Array   # State from t-1 to detect arrival (edge detection)

class AbstractApplianceModel(eqx.Module):
    state: SmartApplianceState
    config: ApplianceConfig
    target_node_index: int = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, signal: float, dt: float, availability_mask: float) -> tuple['AbstractApplianceModel', float, float]:
        raise NotImplementedError

class ShiftableLoadModel(AbstractApplianceModel):
    """
    Robust shiftable load with internal edge detection.
    Resets energy demand ONLY when the device transitions from Unavailable -> Available.
    """
    def __init__(self, config: ApplianceConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index
        self.state = SmartApplianceState(
            energy_remaining_j=jnp.array(0.0),
            prev_is_available=jnp.array(0.0)
        )

    @eqx.filter_jit
    def step(self, signal: float, dt: float, availability_mask: float) -> tuple['ShiftableLoadModel', float, float]:
        """
        Returns: (NewModel, ElectricPowerW, HeatGainW)
        """
        # 1. Detect Rising Edge (Arrival)
        # current=1, prev=0  --> just_arrived=1
        # current=1, prev=1  --> just_arrived=0
        just_arrived = availability_mask * (1.0 - self.state.prev_is_available)

        # 2. Determine Energy State BEFORE charging
        # If just arrived: Reset to full target capacity.
        # If continuing: Keep previous remaining energy.
        # We use jnp.where (or arithmetic mixing) for differentiability.
        current_demand_j = jnp.where(
            just_arrived > 0.5, 
            self.config.total_energy_j, 
            self.state.energy_remaining_j
        )

        # 3. Calculate Power Draw
        # Clamp signal (0.0 to 1.0)
        throttle = jnp.clip(signal, 0.0, 1.0)

        # Can only charge if available AND we have demand left
        should_run = (current_demand_j > 0) * (availability_mask > 0.5)
        
        power_w = should_run * throttle * self.config.nominal_power_w
        
        # 4. Update Energy Remaining
        energy_consumed_j = power_w * dt
        
        # If unplugged (availability=0), we can optionally clear the demand to 0 
        # or freeze it. Clearing it is safer to prevent "phantom load" if plugged in later.
        # Here: result is max(0, demand - consumed), masked by availability.
        new_energy_remaining = jnp.maximum(0.0, current_demand_j - energy_consumed_j)
        new_energy_remaining = new_energy_remaining * availability_mask

        # 5. Heat Gain Calculation
        heat_w = power_w * self.config.convective_fraction

        # 6. Update State (Store current availability as 'prev' for next step)
        new_state = SmartApplianceState(
            energy_remaining_j=new_energy_remaining,
            prev_is_available=jnp.array(availability_mask)
        )
        
        new_model = eqx.tree_at(lambda m: m.state, self, new_state)

        return new_model, power_w, heat_w
    

class PassiveApplianceModel(AbstractApplianceModel):
    """
    Represents 'Dumb' loads (Lights, PC, Fridge, Occupants).
    - Stateless (mostly).
    - Power Output = Nominal Power * Input Signal.
    - Input Signal comes from a dataset profile (e.g., Occupancy Schedule) passed by the Env.
    """
    # We need dummy state to satisfy the PyTree structure if we want to stack them later,
    # or simply to adhere to the Abstract Interface.
    def __init__(self, config: ApplianceConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index
        
        # Dummy state (all zeros)
        self.state = SmartApplianceState(
            energy_remaining_j=jnp.array(0.0),
            prev_is_available=jnp.array(0.0)
        )

    @eqx.filter_jit
    def step(self, signal: float, dt: float, availability_mask: float) -> tuple['PassiveApplianceModel', float, float]:
        """
        For a passive appliance:
        - 'signal' is interpreted as 'utilization fraction' (0.0 to 1.0).
        - 'availability_mask' is usually 1.0 (unless there is a power outage).
        """
        
        # 1. Calculate Power
        # Simple: Power = Nominal * Signal (Usage Profile)
        # e.g., If signal is 0.5 (dimmed lights), power is 50%.
        utilization = jnp.clip(signal, 0.0, 1.0)
        power_w = self.config.nominal_power_w * utilization * availability_mask
        
        # 2. Calculate Heat
        # Heat = Grid Power * Convective Fraction + Metabolic Heat (people)
        heat_w = (power_w * self.config.convective_fraction) + \
                 (self.config.metabolic_heat_w * utilization)

        # 3. Return self (no state change)
        return self, power_w, heat_w