# energysim/core/models/heat_pump_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import HeatPumpConfig, HeatPumpOutput, Array

# --- 1. Abstract Base Class ---
class AbstractHeatPumpModel(eqx.Module):
    """Abstract base class for all heat pump models."""
    # --- Dynamic State ---
    # All models MUST have this state for PyTree consistency
    current_electrical_w: Array 

    # --- Static Config ---
    config: HeatPumpConfig = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['AbstractHeatPumpModel', HeatPumpOutput]:
        """
        The abstract step function.
        Takes a *request* and returns the new model state and actual output.
        """
        raise NotImplementedError


# --- 2. Stateless (Instant) Implementation ---
class StatelessHeatPumpModel(AbstractHeatPumpModel):
    """
    The original model, refactored.
    It has "instant" ramping. Its state is always equal to the last action.
    """
    def __init__(self, config: HeatPumpConfig):
        super().__init__(
            current_electrical_w=jnp.array(0.0),
            config=config
        )

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['StatelessHeatPumpModel', HeatPumpOutput]:
        
        # 1. Enforce action constraints (electrical power limits)
        actual_electrical_w = jnp.clip(
            requested_electrical_w,
            0.0, # Cannot generate electricity
            self.config.max_electrical_power_w
        )

        # 2. Calculate thermal generation based on COP
        actual_thermal_w = actual_electrical_w * self.config.cop_heating

        output = HeatPumpOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        # 3. Update "state" to match the output
        new_model = eqx.tree_at(
            lambda m: m.current_electrical_w, self, actual_electrical_w
        )

        return new_model, output

# --- 3. Ramping (Stateful) Implementation ---
class RampingHeatPumpModel(AbstractHeatPumpModel):
    """
    A stateful model that limits the rate of change (ramping).
    """
    def __init__(self, config: HeatPumpConfig):
        super().__init__(
            current_electrical_w=jnp.array(0.0),
            config=config
        )

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['RampingHeatPumpModel', HeatPumpOutput]:
        
        # 1. Enforce power constraints on the *target*
        target_electrical_w = jnp.clip(
            requested_electrical_w,
            0.0,
            self.config.max_electrical_power_w
        )
        
        # 2. Calculate max change based on ramp rate
        max_delta_w = self.config.ramp_rate_w_per_sec * dt_seconds
        
        # 3. Clip the target based on ramping constraints
        # Ensure we don't ramp down faster than allowed
        lower_ramp_limit = self.current_electrical_w - max_delta_w
        # Ensure we don't ramp up faster than allowed
        upper_ramp_limit = self.current_electrical_w + max_delta_w
        
        actual_electrical_w = jnp.clip(
            target_electrical_w, lower_ramp_limit, upper_ramp_limit
        )

        # 4. Calculate thermal generation
        actual_thermal_w = actual_electrical_w * self.config.cop_heating

        output = HeatPumpOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        # 5. Update state
        new_model = eqx.tree_at(
            lambda m: m.current_electrical_w, self, actual_electrical_w
        )

        return new_model, output

# --- 4. Passthrough (Dummy) Implementation ---
class PassthroughHeatPumpModel(AbstractHeatPumpModel):
    """A dummy model for when no heat pump is present."""
    def __init__(self, config: HeatPumpConfig):
        super().__init__(
            current_electrical_w=jnp.array(0.0),
            config=config
        )

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['PassthroughHeatPumpModel', HeatPumpOutput]:
        
        # No power, no thermal, no state change
        output = HeatPumpOutput(
            thermal_power_w=0.0,
            electrical_power_w=0.0
        )
        return self, output