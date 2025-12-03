import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, TypeVar
from ..physics.constants import CONSTANTS
from ..shared.data_structs import HeatPumpConfig, HeatPumpOutput, Array, HeatPumpState, HeatPumpInputs
from ..physics import thermodynamics

class AbstractHeatPumpModel(eqx.Module):
    config: HeatPumpConfig
    n_rooms: int = eqx.field(static=True)

    def dynamics(self, t: float, state: HeatPumpState, inputs: HeatPumpInputs) -> HeatPumpState:
        """Calculates d(Electrical_Power)/dt (Compressor Inertia)."""
        raise NotImplementedError

    def calculate_output(
        self, 
        state: HeatPumpState, 
        T_sink_c: Array, 
        ambient_temp_c: Array,
        air_density_kg_m3: Array,
        relative_humidity: Array
    ) -> HeatPumpOutput:
        raise NotImplementedError


class NullHeatPumpModel(AbstractHeatPumpModel):
    """
    Implements the Null Object Pattern.
    Represents a heat pump that is either turned off, broken, or non-existent.
    Returns strictly ZERO for all energy and heat flows.
    """

    def dynamics(self, t: float, state: HeatPumpState, inputs: HeatPumpInputs) -> HeatPumpState:
        return HeatPumpState(
            electrical_power_w=jnp.array(0.0)
        )
    
    def calculate_output(
        self, 
        state: HeatPumpState, 
        T_sink_c: Array, 
        ambient_temp_c: Array,
        air_density_kg_m3: Array,
        relative_humidity: Array
    ) -> HeatPumpOutput:
        
        # Explicitly zero out everything.
        # We ignore power_state because a Null machine consumes nothing regardless of control signal.
        
        zeros = jnp.zeros_like(state.electrical_power_w)
        
        return HeatPumpOutput(
            thermal_power_w=zeros,
            electrical_power_w=zeros,
            cop=zeros,
            
            # Temperatures "pass through" neutrally or return 0?
            # For a Null model, it's safer to return the boundary temperatures 
            # so that if this output is used in a mixing equation, it doesn't 
            # introduce -273C (0K) anomalies.
            # Ideally, the downstream component should see q_thermal=0 and ignore the temp.
            
            supply_temp_c=T_sink_c,      # No heating occurred
            condenser_temp_c=T_sink_c,   # Equilibrated with sink
            evaporator_temp_c=ambient_temp_c, # Equilibrated with ambient
            
            eta_second_law=zeros,
            volumetric_limit_hit=zeros
        )


# --- 2. Mechanistic (High Fidelity) ---

class MechanisticHeatPump(AbstractHeatPumpModel):
    """
    Physics-based HP model solving the Condenser/Evaporator equilibrium
    using the pure functions in physics.thermodynamics.
    """

    def dynamics(self, t: float, state: HeatPumpState, inputs: HeatPumpInputs) -> HeatPumpState:
        """
        Calculates d(Electrical_Power)/dt (Compressor Inertia).
        Returns a state object where fields represent derivatives.
        """
        target = inputs.target_power_w
        
        # 1. Scale Limits
        max_w = self.config.max_electrical_power_w / self.n_rooms
        min_w = self.config.min_electrical_power_w / self.n_rooms
        
        # 2. Control Logic
        # Clamp target to physical limits
        target_clamped = jnp.clip(target, 0.0, max_w)
        
        # Deadband / Minimum Output Logic:
        # If target is below min_w, we force it to 0 (OFF).
        # We do not allow the compressor to run at 10W if min is 300W.
        target_final = jnp.where(target_clamped < min_w, 0.0, target_clamped)
        
        # 3. Error Signal
        error_w = target_final - state.electrical_power_w
        
        # 4. Adaptive Smoothing (From AC Model)
        # Instead of hardcoded 50.0, we use 2% of the machine's capacity.
        # This makes the solver stable for both 500W micro-HPs and 20kW industrial HPs.
        smooth_band_w = max_w * 0.02
        smooth_band_w = jnp.maximum(smooth_band_w, 1.0) # Safety against div/0
        
        # 5. Derivative Calculation
        # dP/dt = MaxRamp * tanh(Error / Band)
        dP_dt = self.config.ramp_rate_w_per_sec * jnp.tanh(error_w / smooth_band_w)
        
        # 6. Return Derivative State (Fixing the Type Error)
        # We return a HeatPumpState where 'electrical_power_w' contains the derivative.
        return HeatPumpState(
            electrical_power_w=dP_dt
        )

    def calculate_output(
        self, 
        power_state: Array, 
        T_sink_c: Array, 
        ambient_temp_c: Array,
        source_air_density_kg_m3: Array, # Specific to Outdoor Air (Source)
        relative_humidity: Array
    ) -> HeatPumpOutput:
        
        # --- A. Input Pre-processing ---
        max_w = self.config.max_electrical_power_w / self.n_rooms
        
        # 1. Part Load Ratio (PLR)
        # Avoid division by zero
        plr = jnp.clip(power_state / (max_w + 1e-6), 0.0, 1.0)
        
        # 2. Variable Speed Flow & UA Scaling
        # We assume pump/fan speeds scale with compressor speed
        flow_ratio = 0.2 + 0.8 * plr 
        
        # --- SINK SIDE (Indoors/Radiators) ---
        # Generic: Works for Air (AC) or Water (Heat Pump) based on Config
        m_dot_sink = self.config.design_sink_flow_m3_s * self.config.sink_fluid_density_kg_m3 * flow_ratio
        m_cp_sink = m_dot_sink * self.config.sink_specific_heat_j_kgk

        # --- SOURCE SIDE (Outdoors) ---
        # Fixed as Air for an Air-Source Heat Pump
        # We assume design_source_flow is in the config, or use a ratio of sink flow
        m_dot_source = self.config.design_source_air_flow_m3_s * source_air_density_kg_m3 * flow_ratio
        m_cp_source = m_dot_source * CONSTANTS.SPECIFIC_HEAT_AIR

        # UA Scaling (Heat Exchanger Conductance)
        ua_cond = self.config.ua_condenser_nominal * (flow_ratio ** 0.8)
        ua_evap = self.config.ua_evaporator_nominal * (flow_ratio ** 0.8) 

        # 3. Motor Efficiency
        # Standardized to use the unified thermodynamics helper
        eta_motor = thermodynamics.calculate_inverter_efficiency(
            power_state, 
            max_w, 
            self.config.motor_eff_curve_coeffs
        )
        shaft_power = power_state * eta_motor

        # --- B. Implicit Physics Solver (Vapor Compression Cycle) ---
        # State: (Q_thermal, T_cond, T_evap, Eta_II, Volumetric_Flag)
        InitState = TypeVar('InitState', bound=Tuple[Array, Array, Array, Array, Array])
        
        # Initial Guesses
        init_carry: InitState = (
            shaft_power * 3.0,          # Guess Q (COP=3)
            T_sink_c + 5.0,             # Guess T_cond > Sink
            ambient_temp_c - 5.0,       # Guess T_evap < Ambient
            jnp.array(0.5),             # Guess Eta
            jnp.array(0.0)              # Limit flag
        )

        def system_balance_step(carry: InitState, _) -> Tuple[InitState, None]:
            q_prev, _, _, _, _ = carry
            
            # 1. Condenser Balance (Reject Q to Sink)
            # T_supply = T_return + Q / (m*cp)
            t_supply = T_sink_c + (q_prev / (m_cp_sink + 1e-4))
            
            # LMTD Approximation: T_cond must be higher than T_supply to transfer heat
            t_cond_new = t_supply + (q_prev / (ua_cond + 1e-4))
            
            # 2. Evaporator Balance (Cooling the Source)
            q_source_needed = q_prev - shaft_power
            
            # Fluid cools down: T_leaving = T_ambient - DeltaT
            t_source_leaving = ambient_temp_c - (q_source_needed / (m_cp_source + 1e-4))
            
            # Evaporator must be colder than the leaving air to absorb that heat
            t_evap_new = t_source_leaving - (q_source_needed / (ua_evap + 1e-4))
            
            # 3. Volumetric Limit (Compressor Displacement Constraint)
            # This is the "Engine Speed Limit" of thermodynamics
            max_q_source = thermodynamics.calculate_volumetric_limit(
                t_evap_new, 
                plr,
                self.config.compressor.max_displacement_w_per_k
            )
            
            # Hard physical limit: Cannot pump more gas than cylinder volume allows
            q_source_limited = jnp.minimum(q_source_needed, max_q_source)
            limit_hit = jnp.where(q_source_needed > max_q_source, 1.0, 0.0)
            
            q_thermal_limited = q_source_limited + shaft_power

            # 4. Isentropic Efficiency (The "Quality" of the compressor)
            t_cond_k = t_cond_new + 273.15
            t_evap_k = t_evap_new + 273.15
            lift = jnp.maximum(8.0, t_cond_k - t_evap_k) # Min lift constraint
            
            cop_carnot = t_cond_k / lift
            
            eta_ii = thermodynamics.calculate_isentropic_efficiency(
                lift, plr,
                self.config.compressor.design_lift_k,
                self.config.compressor.design_speed_ratio,
                self.config.compressor.eta_peak,
                self.config.compressor.k_lift,
                self.config.compressor.k_speed
            )
            
            cop_thermo = cop_carnot * eta_ii
            q_target = jnp.minimum(shaft_power * cop_thermo, q_thermal_limited)

            # Soft Update (Relaxation) for Numerical Stability
            q_update = 0.6 * q_prev + 0.4 * q_target
            
            return (q_update, t_cond_new, t_evap_new, eta_ii, limit_hit), None

        # Solve Fixed Point
        (q_final_raw, t_cond_final, t_evap_final, eta_final, limit_hit), _ = jax.lax.scan(
            system_balance_step, init_carry, None, length=8
        )

        # --- C. Final Output Generation ---
        
        # 1. Defrost Logic (Source Side is Air, so icing happens)
        # Note: If Source was water (Geothermal), you would set enable_defrost=False in config
        defrost_factor = jnp.where(
            self.config.enable_defrost,
            thermodynamics.calculate_defrost_penalty(ambient_temp_c, relative_humidity),
            1.0
        )
        
        q_final = q_final_raw * defrost_factor
        
        # 2. Calculate Supply Temp (The actual temp entering the room/tank)
        t_supply_final = T_sink_c + (q_final / (m_cp_sink + 1e-4))
        
        # 3. Final COP
        final_cop = jnp.where(power_state > 1.0, q_final / power_state, 0.0)

        return HeatPumpOutput(
            thermal_power_w=q_final,
            electrical_power_w=power_state,
            cop=final_cop,
            supply_temp_c=t_supply_final,
            condenser_temp_c=t_cond_final,
            evaporator_temp_c=t_evap_final,
            eta_second_law=eta_final,
            volumetric_limit_hit=limit_hit
        )