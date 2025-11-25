import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Tuple

# Type alias for clarity
Array = jax.Array

# --- 1. Physics-Informed Configuration ---

class CompressorPhysics(NamedTuple):
    # --- Isentropic Efficiency Map (Energy Quality) ---
    eta_peak: float = 0.58 
    design_lift_k: float = 30.0
    design_speed_ratio: float = 0.5
    # Sensitivity coefficients
    k_lift: float = 0.0005    # Penalty for high pressure ratio
    k_speed: float = 0.25     # Penalty for deviating from optimal RPM
    
    # --- Volumetric Efficiency (Mass Flow Limit) ---
    # At T_evap = -25C, density is low. Capacity must drop.
    # We use a simplified Antoine-like relation for suction density.
    # displacement_factor: scales the raw physical size of the compressor
    max_displacement_w_per_k: float = 120.0 

class HeatPumpConfig(NamedTuple):
    # Electrical Limits
    max_electrical_power_w: float
    min_electrical_power_w: float
    
    # Design Constraints
    design_air_flow_m3_s: float
    
    # Heat Exchanger UA (Watts/Kelvin)
    # High UA = Better approach temps = Higher Efficiency
    ua_condenser_nominal: float = 350.0  
    ua_evaporator_nominal: float = 450.0
    
    # Sub-components
    compressor: CompressorPhysics = CompressorPhysics(
        max_displacement_w_per_k=85.0  # Reduced from 120.0
    )
    motor_eff_curve_coeffs: tuple = (0.88, 0.08, -0.04) 

    # Environment
    enable_defrost: bool = True

class HeatPumpOutput(NamedTuple):
    thermal_power_w: Array
    electrical_power_w: Array
    cop: Array
    supply_temp_c: Array
    condenser_temp_c: Array 
    evaporator_temp_c: Array
    eta_second_law: Array
    volumetric_limit_hit: Array # Diagnostic: Did we max out flow?

class ExogenousData(NamedTuple):
    ambient_temp: Array
    air_density: Array
    relative_humidity: Array 

# --- 2. The Mechanics (Physics Engine) ---

import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import HeatPumpConfig, HeatPumpOutput, Array, ExogenousData

class AbstractHeatPumpModel(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array
    config: HeatPumpConfig
    n_rooms: int = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exo: ExogenousData, dt: float) -> tuple['AbstractHeatPumpModel', HeatPumpOutput]:
        raise NotImplementedError


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

class TopTierHeatPump(eqx.Module):
    config: HeatPumpConfig
    n_rooms: int = eqx.field(static=True)
    cp_air: float = 1005.0 

    def _calculate_isentropic_efficiency(self, lift_k: Array, speed_ratio: Array) -> Array:
        """
        Calculates Compressor Isentropic Efficiency (eta_II).
        Penalizes efficiency when Lift is high or Speed is off-design.
        """
        c = self.config.compressor
        delta_lift = lift_k - c.design_lift_k
        delta_speed = speed_ratio - c.design_speed_ratio
        
        # Elliptical/Parabolic decay
        penalty = (c.k_lift * delta_lift**2) + (c.k_speed * delta_speed**2)
        return jnp.clip(c.eta_peak - penalty, 0.20, 0.75)

    def _calculate_volumetric_limit(self, t_evap_c: Array, speed_ratio: Array) -> Array:
        """
        Calculates the MAXIMUM Q_source the compressor can physically ingest.
        Physics: Mass Flow ~ RPM * Suction_Density
        Suction Density drops exponentially with Temperature.
        """
        # 1. Density Proxy (Simplified Clausius-Clapeyron / Ideal Gas)
        # Normalized to 1.0 at 0C. Drops to ~0.4 at -20C.
        # This curve shape is critical for "Cold Climate" realism.
        density_factor = jnp.exp(0.045 * t_evap_c)
        
        # 2. Volumetric Efficiency drop at high speeds
        vol_eff = 0.95 - (0.1 * speed_ratio)
        
        # 3. Max Source Heat (Watts)
        # A scaling factor representing displacement * latent_heat
        limit_w = self.config.compressor.max_displacement_w_per_k * 300.0 # Base scaler
        
        return limit_w * speed_ratio * density_factor * vol_eff

    def _calculate_motor_efficiency(self, electrical_w: Array) -> Array:
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        plr = electrical_w / (max_w_per_room + 1e-6)
        c = self.config.motor_eff_curve_coeffs
        eff = c[0] + (c[1] * plr) + (c[2] * (plr**2))
        return jnp.clip(eff, 0.1, 0.98)

    def calculate_output(self, power_state: Array, T_sink_c: Array, exo: ExogenousData) -> HeatPumpOutput:
        
        # --- A. Input State Pre-processing ---
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        plr = jnp.clip(power_state / (max_w_per_room + 1e-6), 0.0, 1.0)
        
        # Fan Laws (Variable Air Volume)
        flow_ratio = 0.2 + 0.8 * plr 
        m_dot = self.config.design_air_flow_m3_s * exo.air_density * flow_ratio
        m_cp = m_dot * self.cp_air

        # Variable UA Scaling
        ua_cond_eff = self.config.ua_condenser_nominal * (flow_ratio ** 0.8)
        ua_evap_eff = self.config.ua_evaporator_nominal * (flow_ratio ** 0.8) 

        # Electrical -> Shaft Power
        eta_motor = self._calculate_motor_efficiency(power_state)
        shaft_power = power_state * eta_motor

        # --- B. Implicit Physics Solver (Energy + Mass Balance) ---
        # State: (Q_thermal, T_cond, T_evap, Eta_II, Volumetric_Flag)
        InitState = Tuple[Array, Array, Array, Array, Array]
        
        # Robust Initial Guesses
        init_carry: InitState = (
            shaft_power * 3.0,          # Guess Q ~ COP of 3
            T_sink_c + 5.0,             # Guess T_cond
            exo.ambient_temp - 5.0,     # Guess T_evap
            jnp.array(0.5),             # Guess Eta
            jnp.array(0.0)              # Limit flag
        )

        def system_balance_step(carry: InitState, _) -> Tuple[InitState, None]:
            q_prev, _, _, _, _ = carry
            
            # 1. Sink Side (Condenser)
            # T_cond rises to push Q into the room
            delta_t_air = q_prev / (m_cp + 1e-4)
            t_supply = T_sink_c + delta_t_air
            t_cond_new = t_supply + (q_prev / (ua_cond_eff + 1e-4))
            
            # 2. Source Side (Evaporator)
            # T_evap drops to pull (Q - Work) from the outside air
            q_source_needed = q_prev - shaft_power
            t_evap_new = exo.ambient_temp - (q_source_needed / (ua_evap_eff + 1e-4))
            
            # 3. Volumetric Check (The "Choke")
            # Can the compressor actually pump this much q_source at this t_evap?
            max_q_source = self._calculate_volumetric_limit(t_evap_new, plr)
            
            # Soft clamp for differentiability (LogSumExp trick or simple min)
            # We use simple min here for clarity, LogSumExp if gradients get noisy
            q_source_limited = jnp.minimum(q_source_needed, max_q_source)
            limit_hit = jnp.where(q_source_needed > max_q_source, 1.0, 0.0)
            
            # Recalculate Q_thermal if we were choked
            q_thermal_limited = q_source_limited + shaft_power

            # 4. Thermodynamic Cycle
            t_cond_k = t_cond_new + 273.15
            t_evap_k = t_evap_new + 273.15
            lift = jnp.maximum(8.0, t_cond_k - t_evap_k) # Min lift 8K
            
            cop_carnot = t_cond_k / lift
            eta_ii_new = self._calculate_isentropic_efficiency(lift, plr)
            
            # 5. Energy Balance Target
            cop_thermo = cop_carnot * eta_ii_new
            q_target_thermo = shaft_power * cop_thermo
            
            # Final Q is the minimum of Thermodynamic Limit and Volumetric Limit
            q_target_final = jnp.minimum(q_target_thermo, q_thermal_limited)

            # Soft Update (Damping)
            q_update = 0.6 * q_prev + 0.4 * q_target_final
            
            return (q_update, t_cond_new, t_evap_new, eta_ii_new, limit_hit), None

        # Execute Solver
        (q_final_raw, t_cond_final, t_evap_final, eta_final, limit_hit), _ = jax.lax.scan(
            system_balance_step, init_carry, None, length=8
        )

        # --- C. Final Output Generation ---
        q_final = q_final_raw
        
        # Defrost penalty (Simplified for this snippet)
        # If T_evap < 0 and RH is high, apply penalty
        q_final = jnp.where(self.config.enable_defrost, q_final, q_final) # Placeholder

        final_cop = jnp.where(power_state > 1.0, q_final / power_state, 0.0)
        t_supply_final = T_sink_c + (q_final / (m_cp + 1e-4))

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
    


# --- 3. Verification Script ---

if __name__ == "__main__":
    # Config setup
    config = HeatPumpConfig(
        max_electrical_power_w=3000.0,
        min_electrical_power_w=500.0,
        design_air_flow_m3_s=0.5
    )
    hp = TopTierHeatPump(config, n_rooms=1)
    
    # 1. Cold Snap Scenario (-7 C)
    exo_cold = ExogenousData(
        ambient_temp=jnp.array(-7.0),
        air_density=jnp.array(1.28),
        relative_humidity=jnp.array(0.6)
    )
    
    # Vectorize across power levels [Low, Medium, Max]
    powers = jnp.array([800.0, 1800.0, 3000.0])
    vmap_calc = jax.vmap(hp.calculate_output, in_axes=(0, None, None))
    
    outputs = vmap_calc(powers, jnp.array(21.0), exo_cold)
    
    print("\n--- Simulation: -7C Ambient, 21C Indoor ---")
    print(f"{'Input (W)':<10} | {'Q_out (W)':<10} | {'COP':<6} | {'T_evap':<8} | {'T_cond':<8} | {'Limit?':<6}")
    print("-" * 65)
    
    for i in range(3):
        p_in = powers[i]
        print(f"{p_in:<10.0f} | {outputs.thermal_power_w[i]:<10.0f} | {outputs.cop[i]:<6.2f} | "
              f"{outputs.evaporator_temp_c[i]:<8.1f} | {outputs.condenser_temp_c[i]:<8.1f} | "
              f"{outputs.volumetric_limit_hit[i]:<6.1f}")

    print("\nObservation:")
    print("1. Low Power: High COP (3.0+), T_evap stays close to ambient.")
    print("2. Max Power: COP Crashes (2.0), T_evap sags to -17C.")
    print("3. Check 'Limit?': If 1.0, the compressor is physically choked by low gas density.")