# energysim/core/models/air_conditioner_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import AirConditionerConfig, AirConditionerOutput, Array

# --- 1. Abstract Base Class ---
class AbstractAirConditionerModel(eqx.Module):
    """Abstract base class for all AC models."""
    # --- Dynamic State ---
    current_electrical_w: Array 

    # --- Static Config ---
    config: AirConditionerConfig = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['AbstractAirConditionerModel', AirConditionerOutput]:
        raise NotImplementedError


# --- 2. Stateless (Instant) Implementation ---
class StatelessAirConditionerModel(AbstractAirConditionerModel):
    """The original model, refactored. Instant ramping."""
    def __init__(self, config: AirConditionerConfig):
        super().__init__(
            current_electrical_w=jnp.array(0.0),
            config=config
        )

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['StatelessAirConditionerModel', AirConditionerOutput]:
        
        actual_electrical_w = jnp.clip(
            requested_electrical_w,
            0.0,
            self.config.max_electrical_power_w
        )

        # Thermal power is NEGATIVE (removing heat)
        actual_thermal_w = - (actual_electrical_w * self.config.cop_cooling)

        output = AirConditionerOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        new_model = eqx.tree_at(
            lambda m: m.current_electrical_w, self, actual_electrical_w
        )
        return new_model, output

# --- 3. Ramping (Stateful) Implementation ---
class RampingAirConditionerModel(AbstractAirConditionerModel):
    """A stateful model that limits the rate of change (ramping)."""
    def __init__(self, config: AirConditionerConfig):
        super().__init__(
            current_electrical_w=jnp.array(0.0),
            config=config
        )

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['RampingAirConditionerModel', AirConditionerOutput]:
        
        target_electrical_w = jnp.clip(
            requested_electrical_w,
            0.0,
            self.config.max_electrical_power_w
        )
        
        max_delta_w = self.config.ramp_rate_w_per_sec * dt_seconds
        
        lower_ramp_limit = self.current_electrical_w - max_delta_w
        upper_ramp_limit = self.current_electrical_w + max_delta_w
        
        actual_electrical_w = jnp.clip(
            target_electrical_w, lower_ramp_limit, upper_ramp_limit
        )

        actual_thermal_w = - (actual_electrical_w * self.config.cop_cooling)

        output = AirConditionerOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        new_model = eqx.tree_at(
            lambda m: m.current_electrical_w, self, actual_electrical_w
        )
        return new_model, output

# --- 4. Passthrough (Dummy) Implementation ---
class PassthroughAirConditionerModel(AbstractAirConditionerModel):
    """A dummy model for when no AC is present."""
    def __init__(self, config: AirConditionerConfig):
        super().__init__(
            current_electrical_w=jnp.array(0.0),
            config=config
        )

    @eqx.filter_jit
    def step(self, 
             requested_electrical_w: Array, 
             dt_seconds: float
    ) -> tuple['PassthroughAirConditionerModel', AirConditionerOutput]:
        
        output = AirConditionerOutput(
            thermal_power_w=0.0,
            electrical_power_w=0.0
        )
        return self, output