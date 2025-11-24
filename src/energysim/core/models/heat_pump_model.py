import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import HeatPumpConfig, HeatPumpOutput, Array, ExogenousData

class AbstractHeatPumpModel(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array
    config: HeatPumpConfig
    n_rooms: int = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, T_sink_c: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['AbstractHeatPumpModel', HeatPumpOutput]:
        raise NotImplementedError

    def _calculate_cop_modifier(self, electrical_w: Array) -> Array:
        """
        Calculates efficiency multiplier f(PLR).
        PLR = Current_Power / Max_Power
        Modifier = c0 + c1*PLR + c2*PLR^2
        """
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        
        # Avoid division by zero
        plr = electrical_w / (max_w_per_room + 1e-6)
        
        # Extract coefficients
        c = self.config.part_load_curve_coeffs
        
        # Polynomial evaluation
        modifier = c[0] + (c[1] * plr) + (c[2] * (plr**2))
        
        # Clamp for physical realism (e.g., efficiency can't triple or go negative)
        return jnp.clip(modifier, 0.1, 2.0)


class StatelessHeatPumpModel(AbstractHeatPumpModel):
    """Ramps instantly to the requested power, clipped by per-room max."""
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms), 
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(
        self, 
        requested_electrical_w: Array, 
        T_sink_c: Array,
        exogenous: ExogenousData, 
        dt_seconds: float
    ) -> tuple['StatelessHeatPumpModel', HeatPumpOutput]:

        # 1. Clip against per-room power limits
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        actual_electrical_w = jnp.clip(
            requested_electrical_w, 
            0.0, 
            max_w_per_room
        )

        # 2. Calculate Part-Load Efficiency
        plr_modifier = self._calculate_cop_modifier(actual_electrical_w)

        # 3. Instant thermal conversion with PLR penalty
        effective_cop = self.config.cop_heating * plr_modifier
        actual_thermal_w = actual_electrical_w * effective_cop

        output = HeatPumpOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )

        # Update both electrical and thermal state
        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w),
            self,
            (actual_electrical_w, actual_thermal_w)
        )

        return new_model, output


class RampingHeatPumpModel(AbstractHeatPumpModel):
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, T_sink_c: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['RampingHeatPumpModel', HeatPumpOutput]:

        # 1. Minimum Power & Ramping
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        min_w_per_room = self.config.min_electrical_power_w / self.n_rooms

        target_w = jnp.clip(requested_electrical_w, 0.0, max_w_per_room)
        target_w = jnp.where(target_w < min_w_per_room, 0.0, target_w)

        max_delta = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_elec = jnp.clip(
            target_w, 
            self.current_electrical_w - max_delta, 
            self.current_electrical_w + max_delta
        )

        # 2. Calculate Part-Load Efficiency
        plr_modifier = self._calculate_cop_modifier(actual_elec)

        # 3. COP Calculation with PLR
        raw_thermal_gen = actual_elec * self.config.cop_heating * plr_modifier

        # 4. Thermal Lag (First Order Filter)
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        actual_thermal = (alpha * raw_thermal_gen) + ((1.0 - alpha) * self.current_thermal_w)

        output = HeatPumpOutput(
            thermal_power_w=actual_thermal, 
            electrical_power_w=actual_elec
        )

        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w),
            self,
            (actual_elec, actual_thermal)
        )
        return new_model, output


class VariableCOPHeatPumpModel(AbstractHeatPumpModel):
    cop_temps: Array
    cop_values: Array

    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )
        self.cop_temps = jnp.array(config.cop_ambient_temps_c)
        self.cop_values = jnp.array(config.cop_values_heating)

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array,T_sink_c: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['VariableCOPHeatPumpModel', HeatPumpOutput]:

        # 1. Min Power & Ramping
        max_w = self.config.max_electrical_power_w / self.n_rooms
        min_w = self.config.min_electrical_power_w / self.n_rooms

        target_w = jnp.where(
            requested_electrical_w < min_w, 0.0, jnp.clip(requested_electrical_w, 0.0, max_w)
        )

        max_delta = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_elec = jnp.clip(target_w, self.current_electrical_w - max_delta, self.current_electrical_w + max_delta)

        # 2. Variable COP (Ambient Temp)
        cop_nominal = jnp.interp(exogenous.ambient_temp, self.cop_temps, self.cop_values)
        
        # 3. Part-Load Penalty
        plr_modifier = self._calculate_cop_modifier(actual_elec)
        
        effective_cop = cop_nominal * plr_modifier
        raw_thermal_gen = actual_elec * effective_cop

        # 4. Thermal Lag
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        actual_thermal = (alpha * raw_thermal_gen) + ((1.0 - alpha) * self.current_thermal_w)

        output = HeatPumpOutput(thermal_power_w=actual_thermal, electrical_power_w=actual_elec)

        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w),
            self, (actual_elec, actual_thermal)
        )
        return new_model, output


class MechanisticHeatPump(AbstractHeatPumpModel):
    """
    Physics-informed Heat Pump.
    Constraint: Q_delivered <= m_dot_air * Cp_air * (T_supply_limit - T_room)
    """
    # Physics Constants
    CP_AIR: float = 1005.0 
    
    def __init__(self, config, n_rooms):
        # Initialize state as zeros
        self.current_electrical_w = jnp.zeros(n_rooms)
        self.current_thermal_w = jnp.zeros(n_rooms)
        self.config = config
        self.n_rooms = n_rooms

    @eqx.filter_jit
    def step(
        self, 
        requested_electrical_w: Array, 
        T_sink_c: Array,               # This model USES this argument
        exogenous: ExogenousData, 
        dt_seconds: float
    ) -> tuple['MechanisticHeatPump', HeatPumpOutput]:
        
        # 1. Determine Carnot Capacity
        # T_sink_c is the room temperature. 
        # T_source is Ambient (Air-source HP).
        delta_T_lift = T_sink_c - exogenous.ambient_temp
        
        # Avoid singularity if delta_T is 0 or negative (though unlikely in heating mode)
        safe_delta_T = jnp.maximum(5.0, delta_T_lift) 
        
        carnot_cop = (T_sink_c + 273.15) / safe_delta_T
        real_cop = carnot_cop * self.config.cop_heating * 0.4 # 40% Carnot efficiency factor
        
        # 2. Electrical Consumption (Clipped to max)
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        actual_elec_w = jnp.clip(requested_electrical_w, 0.0, max_w_per_room)
        
        # 3. Theoretical Generation (Vapor Cycle)
        q_generation_w = actual_elec_w * real_cop
        
        # 4. Mass Flow Constraint
        # Q_max_flow = m_dot * Cp * (T_max - T_room)
        # We derive m_dot from config (design flow) or hardcoded physics proxy
        m_dot_air = self.config.design_air_flow_m3_s * 1.204 # kg/s
        
        max_supply_temp = self.config.max_supply_temp_c
        max_transferable_q = m_dot_air * self.CP_AIR * (max_supply_temp - T_sink_c)
        
        # 5. Final Output (Physics constrained)
        actual_thermal_w = jnp.minimum(q_generation_w, max_transferable_q)
        
        output = HeatPumpOutput(
            thermal_power_w=actual_thermal_w, 
            electrical_power_w=actual_elec_w
        )
        
        # Update internal state
        new_hp = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w), 
            self, 
            (actual_elec_w, actual_thermal_w)
        )
        
        return new_hp, output
    


class PassthroughHeatPumpModel(AbstractHeatPumpModel):
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exo: ExogenousData, dt: float):
        # Pass through 0s, maintain state as 0s
        return self, HeatPumpOutput(
            thermal_power_w=self.current_thermal_w,
            electrical_power_w=self.current_electrical_w
        )