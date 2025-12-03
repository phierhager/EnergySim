import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, TypeVar
from ..shared.data_structs import (
    AirConditionerConfig, 
    AirConditionerOutput,
    AirConditionerInputs,
    AirConditionerState
)
from ..physics import psychrometrics, thermodynamics
from ..physics.constants import CONSTANTS

# Type alias
Array = jax.Array

class AbstractAirConditionerModel(eqx.Module):
    config: AirConditionerConfig
    n_rooms: int = eqx.field(static=True)

    def dynamics(self, t: float, state: AirConditionerState, inputs: AirConditionerInputs) -> AirConditionerState:
        """Calculates d(Electrical_Power)/dt (Compressor Inertia)."""
        raise NotImplementedError

    def calculate_output(
        self, 
        state: AirConditionerState,
        # --- Pure Physical Inputs (No Data Structures) ---
        outdoor_temp_c: Array,
        indoor_temp_c: Array,
        indoor_rel_humidity: Array,
        atmospheric_pressure_pa: Array
    ) -> AirConditionerOutput:
        """
        Calculates Sensible Cooling, Electrical Draw, and Water Removal.
        """
        raise NotImplementedError

class MechanisticACModel(AbstractAirConditionerModel):
    """
    Platinum Standard AC Model.
    
    Uses:
    1. thermodynamics.py -> To solve the Vapor Compression Cycle (Compressor Limit).
    2. psychrometrics.py -> To solve the Coil Condensation (Latent Heat).
    """
    def dynamics(self, t: float, state: AirConditionerState, inputs: AirConditionerInputs) -> AirConditionerState:
        target = inputs.target_power_w
        max_w = self.config.max_electrical_power_w / self.n_rooms
        min_w = self.config.min_electrical_power_w / self.n_rooms
        
        target_clamped = jnp.clip(target, 0.0, max_w)
        target_final = jnp.where(target_clamped < min_w, 0.0, target_clamped)
        
        error_w = target_final - state.electrical_power_w

        # HEURISTIC: Smoothing band is 2% of max capacity.
        # This ensures the solver behaves consistently regardless of unit size.
        smooth_band_w = (self.config.max_electrical_power_w / self.n_rooms) * 0.02
        
        # Avoid division by zero if max_power is 0
        smooth_band_w = jnp.maximum(smooth_band_w, 1.0) 

        # 5. Derivative Calculation
        # dP/dt = MaxRamp * tanh(Error / Band)
        dP_dt = self.config.ramp_rate_w_per_sec * jnp.tanh(error_w / smooth_band_w)
        
        # 6. Return Derivative State (Fixing the Type Error)
        # We return a AirConditionerState    where 'electrical_power_w' contains the derivative.
        return AirConditionerState(
            electrical_power_w=dP_dt
        )
    def calculate_output(
        self,
        state: AirConditionerState,
        outdoor_temp_c: Array,
        indoor_temp_c: Array,
        indoor_rel_humidity: Array,
        atmospheric_pressure_pa: Array
    ) -> AirConditionerOutput:
        
        # --- A. Setup ---
        max_w = self.config.max_electrical_power_w / self.n_rooms
        
        # 1. Standardized Motor Efficiency
        # We pass the full max_w (per room) and the current state
        eta_motor = thermodynamics.calculate_inverter_efficiency(
            state.electrical_power_w, 
            max_w, 
            self.config.motor_eff_curve_coeffs
        )
        
        shaft_power = state.electrical_power_w * eta_motor
        
        # Calculate PLR for UA scaling
        plr = jnp.clip(state.electrical_power_w / (max_w + 1e-6), 0.0, 1.0)
        
        # Fan and UA Scaling
        flow_ratio = 0.2 + 0.8 * plr 
        m_dot_air = self.config.design_air_flow_m3_s * 1.2 * flow_ratio
        m_cp = m_dot_air * CONSTANTS.SPECIFIC_HEAT_AIR
        ua_cond = self.config.ua_condenser_nominal * (flow_ratio ** 0.8)
        ua_evap = self.config.ua_evaporator_nominal * (flow_ratio ** 0.8)

        # --- B. Thermodynamic Solver ---
        # State: (Q_cooling_mag, T_cond, T_evap, Limit_Flag)
        InitState = TypeVar('InitState', bound=tuple[Array, Array, Array, Array])

        init_carry: InitState = (
            shaft_power * 3.0,      
            outdoor_temp_c + 15.0,  
            indoor_temp_c - 10.0,   
            jnp.array(0.0)
        )

        def cycle_step(carry: InitState, _) -> Tuple[InitState, None]:
            q_cool_mag_prev, _, _, _ = carry
            
            # Heat Rejection Balance
            q_reject = q_cool_mag_prev + shaft_power
            t_cond_new = outdoor_temp_c + (q_reject / (ua_cond + 1e-4)) + (q_reject / (m_cp + 1e-4))

            # Heat Absorption Balance
            t_evap_new = indoor_temp_c - (q_cool_mag_prev / (ua_evap + 1e-4)) - (q_cool_mag_prev / (m_cp + 1e-4))
            
            # --- Volumetric Limit Check ---
            # Calculates the max energy the pump can move based on gas density
            max_q_pump = thermodynamics.calculate_volumetric_limit(
                t_evap_new, 
                plr,
                self.config.compressor.max_displacement_w_per_k
            )
            
            # Efficiency Calculation
            t_cond_k = t_cond_new + 273.15
            t_evap_k = t_evap_new + 273.15
            lift = jnp.maximum(5.0, t_cond_k - t_evap_k)
            
            cop_carnot = t_evap_k / lift
            eta_ii = thermodynamics.calculate_isentropic_efficiency(
                lift, plr,
                self.config.compressor.design_lift_k,
                self.config.compressor.design_speed_ratio,
                self.config.compressor.eta_peak,
                self.config.compressor.k_lift,
                self.config.compressor.k_speed
            )
            
            q_target_thermo = shaft_power * (cop_carnot * eta_ii)
            
            # --- Apply Limit ---
            # We take the minimum of what thermodynamics allows vs what the pump volume allows
            q_target_final = jnp.minimum(q_target_thermo, max_q_pump)
            
            # Determine if we hit the volumetric limit
            # (If Thermo target > Pump Limit, we are limited)
            step_limit_hit = jnp.where(q_target_thermo > max_q_pump, 1.0, 0.0)

            # Soft update for stability
            q_next = 0.6 * q_cool_mag_prev + 0.4 * q_target_final
            
            return (q_next, t_cond_new, t_evap_new, step_limit_hit), None

        # Run Solver
        (q_cooling_mag, _, _, final_limit_hit), _ = jax.lax.scan(
            cycle_step, init_carry, None, length=8
        )
        
        # --- C. Psychrometrics ---
        sensible_w, water_kg_s = psychrometrics.calculate_dx_coil_split(
            total_cooling_w=-q_cooling_mag, # Note negative sign convention
            T_in_c=indoor_temp_c,
            rel_hum_in=indoor_rel_humidity,
            pressure_pa=atmospheric_pressure_pa,
            T_coil_adp_c=self.config.coil_adp_c,
            bypass_factor=self.config.coil_bypass_factor,
            mass_flow_air_kg_s=m_dot_air
        )

        return AirConditionerOutput(
            thermal_power_w=sensible_w,
            electrical_power_w=state.electrical_power_w,
            water_removed_kg_s=water_kg_s,
            volumetric_limit_hit=final_limit_hit # <--- Now exposed
        )

# --- Passthrough Implementation ---

class NullAirConditionerModel(AbstractAirConditionerModel):
    """
    Implements the Null Object Pattern. 
    Used when a room has no AC, or to simulate 'Free Running' conditions.
    """
    def dynamics(self, t: float, state: AirConditionerState, inputs: AirConditionerInputs) -> AirConditionerState:
        return AirConditionerState(
            electrical_power_w=jnp.array(0.0)
        )


    def calculate_output(self,
        state: AirConditionerState,
        # --- Pure Physical Inputs (No Data Structures) ---
        outdoor_temp_c: Array,
        indoor_temp_c: Array,
        indoor_rel_humidity: Array,
        atmospheric_pressure_pa: Array
        ) -> AirConditionerOutput:
        return AirConditionerOutput(
            thermal_power_w=jnp.array(0.0),
            electrical_power_w=jnp.array(0.0),
            water_removed_kg_s=jnp.array(0.0)
        )