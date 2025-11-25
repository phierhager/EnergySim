import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from ..shared.data_structs import (
    AirConditionerConfig, 
    AirConditionerOutput, 
    Array, 
    ExogenousData, 
    RampingInputs
)

L_V = 2460000.0 # Latent Heat

class AbstractAirConditionerModel(eqx.Module):
    config: AirConditionerConfig
    n_rooms: int = eqx.field(static=True)

    def dynamics(self, t: float, power_state: Array, inputs: RampingInputs) -> Array:
        """Calculates d(Compressor_Power)/dt."""
        raise NotImplementedError

    def calculate_output(
        self, 
        power_state: Array, 
        exo: ExogenousData
    ) -> AirConditionerOutput:
        """
        Algebraic calculation of Sensible/Latent split based on current state & environment.
        """
        raise NotImplementedError

    def _calculate_cop_modifier(self, electrical_w: Array) -> Array:
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        plr = electrical_w / (max_w_per_room + 1e-6)
        c = self.config.part_load_curve_coeffs
        modifier = c[0] + (c[1] * plr) + (c[2] * (plr**2))
        return jnp.clip(modifier, 0.1, 2.0)

    def _split_sensible_latent(self, total_cooling_w: Array, rel_humidity: Array) -> tuple[Array, Array]:
        """
        Pure Algebraic Function: T(RH) -> (Sensible_W, Water_kg_s).
        """
        moisture_availability = jnn.sigmoid((rel_humidity - 0.5) * 20.0)
        effective_shr = 1.0 - (moisture_availability * (1.0 - self.config.sensible_heat_ratio))
        
        sensible_w = total_cooling_w * effective_shr
        latent_w = total_cooling_w * (1.0 - effective_shr)
        
        # Positive mass flow rate (kg/s)
        water_removed_kg_s = jnp.abs(latent_w) / L_V
        
        return sensible_w, water_removed_kg_s


# --- 1. Passthrough / Stateless ---
class PassthroughAirConditionerModel(AbstractAirConditionerModel):
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        self.config = config
        self.n_rooms = n_rooms

    def dynamics(self, t: float, power_state: Array, inputs: RampingInputs) -> Array:
        return jnp.zeros_like(power_state)

    def calculate_output(self, power_state: Array, exo: ExogenousData) -> AirConditionerOutput:
        # Standard constant COP
        total_cooling = -(power_state * self.config.cop_cooling)
        
        sensible, water = self._split_sensible_latent(total_cooling, exo.relative_humidity)
        
        return AirConditionerOutput(
            thermal_power_w=sensible,
            electrical_power_w=power_state,
            water_removed_kg_s=water
        )

class StatelessAirConditionerModel(PassthroughAirConditionerModel):
    pass


# --- 2. Ramping (Differential) ---
class RampingAirConditionerModel(AbstractAirConditionerModel):
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        self.config = config
        self.n_rooms = n_rooms

    def dynamics(self, t: float, power_state: Array, inputs: RampingInputs) -> Array:
        """
        dP/dt = rate * smooth_sign(error)
        """
        target = inputs.target_power_w
        
        max_w = self.config.max_electrical_power_w / self.n_rooms
        min_w = self.config.min_electrical_power_w / self.n_rooms
        
        target_clamped = jnp.clip(target, 0.0, max_w)
        target_clamped = jnp.where(target_clamped < min_w, 0.0, target_clamped)
        
        error = target_clamped - power_state
        rate = self.config.ramp_rate_w_per_sec
        
        return rate * jnp.tanh(error / 50.0)

    def calculate_output(self, power_state: Array, exo: ExogenousData) -> AirConditionerOutput:
        # Algebraic Physics on current state
        plr_mod = self._calculate_cop_modifier(power_state)
        effective_cop = self.config.cop_cooling * plr_mod
        
        total_cooling = -(power_state * effective_cop)
        
        # Psychrometrics are instantaneous/algebraic relative to the 15-min/1-sec step
        sensible, water = self._split_sensible_latent(total_cooling, exo.relative_humidity)
        
        return AirConditionerOutput(
            thermal_power_w=sensible,
            electrical_power_w=power_state,
            water_removed_kg_s=water
        )


# --- 3. Variable COP (Differential + Weather Aware) ---
class VariableCOPAirConditionerModel(RampingAirConditionerModel):
    """
    Interpolates COP based on Ambient Temp, uses Ramping dynamics.
    """
    def calculate_output(self, power_state: Array, exo: ExogenousData) -> AirConditionerOutput:
        # 1. Interpolate Nominal COP
        # AC COP drops as ambient temp rises
        cop_curve_T = jnp.array(self.config.cop_ambient_temps_c)
        cop_curve_V = jnp.array(self.config.cop_values_cooling)
        
        base_cop = jnp.interp(exo.ambient_temp, cop_curve_T, cop_curve_V)
        
        # 2. PLR Modifier
        plr_mod = self._calculate_cop_modifier(power_state)
        effective_cop = base_cop * plr_mod
        
        total_cooling = -(power_state * effective_cop)
        
        # 3. Psychrometrics
        sensible, water = self._split_sensible_latent(total_cooling, exo.relative_humidity)
        
        return AirConditionerOutput(
            thermal_power_w=sensible,
            electrical_power_w=power_state,
            water_removed_kg_s=water
        )