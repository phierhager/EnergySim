import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from ..shared.data_structs import AirConditionerConfig, AirConditionerOutput, Array, ExogenousData

# Physics Constant: Latent Heat of Vaporization for Water (J/kg)
L_V = 2460000.0

# --- 1. Abstract Base Class with Psychrometric Logic ---
class AbstractAirConditionerModel(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array
    config: AirConditionerConfig
    n_rooms: int = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['AbstractAirConditionerModel', AirConditionerOutput]:
        raise NotImplementedError

    def _calculate_cop_modifier(self, electrical_w: Array) -> Array:
        """Calculates efficiency multiplier f(PLR) = c0 + c1*PLR + c2*PLR^2"""
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        plr = electrical_w / (max_w_per_room + 1e-6)
        c = self.config.part_load_curve_coeffs
        modifier = c[0] + (c[1] * plr) + (c[2] * (plr**2))
        return jnp.clip(modifier, 0.1, 2.0)

    def _split_sensible_latent(self, total_cooling_w: Array, rel_humidity: Array) -> tuple[Array, Array]:
        """Splits Total Cooling into Sensible (Watts) and Latent (kg/s)."""
        moisture_availability = jnn.sigmoid((rel_humidity - 0.5) * 20.0)
        effective_shr = 1.0 - (moisture_availability * (1.0 - self.config.sensible_heat_ratio))
        
        sensible_w = total_cooling_w * effective_shr
        latent_w = total_cooling_w * (1.0 - effective_shr)
        
        # Latent W is energy REMOVED. Positive mass flow rate.
        water_removed_kg_s = jnp.abs(latent_w) / L_V
        
        return sensible_w, water_removed_kg_s


# --- 2. Stateless Implementation ---
class StatelessAirConditionerModel(AbstractAirConditionerModel):
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['StatelessAirConditionerModel', AirConditionerOutput]:
        # 1. Electrical Physics
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        actual_electrical_w = jnp.clip(requested_electrical_w, 0.0, max_w_per_room)

        # 2. Thermal Generation with PLR
        plr_modifier = self._calculate_cop_modifier(actual_electrical_w)
        effective_cop = self.config.cop_cooling * plr_modifier
        total_cooling_w = - (actual_electrical_w * effective_cop)

        # 3. Psychrometric Split
        rh_proxy = getattr(exogenous, 'relative_humidity', 0.5)
        sensible_w, water_kg_s = self._split_sensible_latent(total_cooling_w, rh_proxy)

        output = AirConditionerOutput(
            thermal_power_w=sensible_w,
            electrical_power_w=actual_electrical_w,
            water_removed_kg_s=water_kg_s
        )

        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w),
            self,
            (actual_electrical_w, sensible_w)
        )
        return new_model, output


# --- 3. Ramping Implementation ---
class RampingAirConditionerModel(AbstractAirConditionerModel):
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['RampingAirConditionerModel', AirConditionerOutput]:
        
        # 1. Electrical Ramping
        max_w = self.config.max_electrical_power_w / self.n_rooms
        min_w = self.config.min_electrical_power_w / self.n_rooms
        target_w = jnp.where(
            requested_electrical_w < min_w, 0.0, jnp.clip(requested_electrical_w, 0.0, max_w)
        )

        max_delta = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_elec = jnp.clip(target_w, self.current_electrical_w - max_delta, self.current_electrical_w + max_delta)

        # 2. PLR & Total Cooling
        plr_modifier = self._calculate_cop_modifier(actual_elec)
        raw_total_cooling = -1.0 * actual_elec * self.config.cop_cooling * plr_modifier

        # 3. Thermal Lag
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        prev_total_cooling = self.current_thermal_w / self.config.sensible_heat_ratio
        actual_total_cooling = (alpha * raw_total_cooling) + ((1.0 - alpha) * prev_total_cooling)

        # 4. Psychrometric Split
        rh_proxy = getattr(exogenous, 'relative_humidity', 0.5)
        sensible_w, water_kg_s = self._split_sensible_latent(actual_total_cooling, rh_proxy)

        output = AirConditionerOutput(
            thermal_power_w=sensible_w,
            electrical_power_w=actual_elec,
            water_removed_kg_s=water_kg_s
        )

        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w),
            self, (actual_elec, sensible_w)
        )
        return new_model, output


# --- 4. Variable COP Implementation ---
class VariableCOPAirConditionerModel(AbstractAirConditionerModel):
    cop_temps: Array
    cop_values: Array

    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )
        self.cop_temps = jnp.array(config.cop_ambient_temps_c)
        self.cop_values = jnp.array(config.cop_values_cooling)

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['VariableCOPAirConditionerModel', AirConditionerOutput]:

        # 1. Ramping
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        min_w_per_room = self.config.min_electrical_power_w / self.n_rooms

        target_electrical_w = jnp.clip(requested_electrical_w, 0.0, max_w_per_room)
        target_electrical_w = jnp.where(target_electrical_w < min_w_per_room, 0.0, target_electrical_w)

        max_delta_w = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_electrical_w = jnp.clip(
            target_electrical_w,
            self.current_electrical_w - max_delta_w,
            self.current_electrical_w + max_delta_w
        )

        # 2. Variable COP + PLR
        T_amb = exogenous.ambient_temp
        current_cop = jnp.interp(T_amb, self.cop_temps, self.cop_values)
        plr_modifier = self._calculate_cop_modifier(actual_electrical_w)
        
        raw_total_cooling = -1.0 * actual_electrical_w * current_cop * plr_modifier

        # 3. Thermal Lag
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        prev_total_cooling = self.current_thermal_w / self.config.sensible_heat_ratio
        actual_total_cooling = (alpha * raw_total_cooling) + ((1.0 - alpha) * prev_total_cooling)

        # 4. Psychrometric Split
        rh_proxy = getattr(exogenous, 'relative_humidity', 0.5)
        sensible_w, water_kg_s = self._split_sensible_latent(actual_total_cooling, rh_proxy)

        output = AirConditionerOutput(
            thermal_power_w=sensible_w,
            electrical_power_w=actual_electrical_w,
            water_removed_kg_s=water_kg_s
        )

        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w),
            self,
            (actual_electrical_w, sensible_w)
        )
        return new_model, output


# --- 5. Passthrough Implementation ---
class PassthroughAirConditionerModel(AbstractAirConditionerModel):
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['PassthroughAirConditionerModel', AirConditionerOutput]:
        output = AirConditionerOutput(
            thermal_power_w=jnp.zeros_like(self.current_electrical_w),
            electrical_power_w=jnp.zeros_like(self.current_electrical_w),
            water_removed_kg_s=jnp.zeros_like(self.current_electrical_w)
        )
        return self, output