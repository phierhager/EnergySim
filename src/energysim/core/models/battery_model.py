# energysim/core/models/battery_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import BatteryConfig, Array

class AbstractBatteryModel(eqx.Module):
    # --- Dynamic State ---
    soc: Array  # All batteries MUST have soc
    soh: Array  # All batteries MUST have soh (even if it's a dummy 1.0)

    # --- Static Config ---
    config: BatteryConfig = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, action_power_w: Array, dt_seconds: float) -> 'AbstractBatteryModel':
        """The 'step' function is the contract all models must fulfill."""
        raise NotImplementedError

class SimpleBatteryModel(AbstractBatteryModel):    
    def __init__(self, config: BatteryConfig, initial_soc: float = 0.5):
        super().__init__(
            soc=jnp.array(initial_soc),
            soh=jnp.array(1.0),
            config=config
        )

    @eqx.filter_jit
    def step(self, action_power_w: Array, dt_seconds: float) -> 'SimpleBatteryModel':
        """
        `self` is auto-filtered (soc=dynamic, config=static)
        `action_power_w` is dynamic (Array)
        `dt_seconds` is auto-static (Python float)
        """
        clipped_power_w = jnp.clip(action_power_w, -self.config.max_power_w, self.config.max_power_w)
        energy_transfer_j = clipped_power_w * dt_seconds
        one_way_eff = jnp.sqrt(self.config.efficiency)
        
        energy_stored_j = jnp.where(
            energy_transfer_j > 0,
            energy_transfer_j * one_way_eff,
            energy_transfer_j / one_way_eff
        )
        
        delta_soc = jnp.where(self.config.capacity_j > 0, energy_stored_j / self.config.capacity_j, 0.0)
        next_soc = jnp.clip(self.soc + delta_soc, 0.0, 1.0)
        
        return eqx.tree_at(lambda model: model.soc, self, next_soc)
    

class DegradationBatteryModel(AbstractBatteryModel):
    def __init__(self, config: BatteryConfig, initial_soc: float = 0.5, initial_soh: float = 1.0):
        super().__init__(
            soc=jnp.array(initial_soc),
            soh=jnp.array(initial_soh),
            config=config
        )
    
    @eqx.filter_jit
    def step(self, action_power_w: Array, dt_seconds: float) -> 'DegradationBatteryModel':

        clipped_power_w = jnp.clip(action_power_w, -self.config.max_power_w, self.config.max_power_w)
        energy_transfer_j = clipped_power_w * dt_seconds
        one_way_eff = jnp.sqrt(self.config.efficiency)
        
        energy_stored_j = jnp.where(
            energy_transfer_j > 0,
            energy_transfer_j * one_way_eff,
            energy_transfer_j / one_way_eff
        )

        effective_capacity_j = self.config.capacity_j * self.soh
        delta_soc = jnp.where(effective_capacity_j > 0, energy_stored_j / effective_capacity_j, 0.0)
        next_soc = jnp.clip(self.soc + delta_soc, 0.0, 1.0)

        # 1. Get total energy cycled (Joules)
        energy_cycled_j = jnp.abs(energy_transfer_j) 
        
        # 2. Convert to equivalent "full cycles"
        # A full cycle is one full discharge + one full charge (2 * capacity)
        # Use the *full* capacity_j, not effective_capacity_j, as the denominator
        full_cycle_energy_j = self.config.capacity_j * 2.0 
        cycles = jnp.where(full_cycle_energy_j > 0, energy_cycled_j / full_cycle_energy_j, 0.0)

        # 3. Calculate SOH loss
        soh_loss = cycles * self.config.degradation_rate_per_cycle
        next_soh = jnp.clip(self.soh - soh_loss, 0.0, 1.0)
    
        # Functionally update both soc and soh
        new_model = eqx.tree_at(lambda m: m.soc, self, next_soc)
        new_model = eqx.tree_at(lambda m: m.soh, new_model, next_soh)
        return new_model
    
class PassthroughBatteryModel(AbstractBatteryModel):
    def __init__(self, config: BatteryConfig):
        # All states are dummy 0.0, but they *exist*
        super().__init__(
            soc=jnp.array(0.0),
            soh=jnp.array(0.0),
            config=config
        )
    
    @eqx.filter_jit
    def step(self, action_power_w: Array, dt_seconds: float) -> 'PassthroughBatteryModel':
        # No state change, just return self
        return self