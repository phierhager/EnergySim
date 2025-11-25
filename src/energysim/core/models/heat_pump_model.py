import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Tuple

# Type alias for clarity
Array = jax.Array

# --- 1. Configuration & Data Structures ---

class HeatPumpConfig(NamedTuple):
    # Power constraints
    max_electrical_power_w: float
    min_electrical_power_w: float
    ramp_rate_w_per_sec: float
    
    # Physical Design constraints
    design_air_flow_m3_s: float
    max_supply_temp_c: float
    
    # --- Highest Fidelity Parameters ---
    # UA: Overall Heat Transfer Coefficient (W/K) per room.
    # Replaces "Approach Temp". 
    # High UA = Premium unit. Low UA = Budget unit.
    ua_condenser_nominal: float = 850.0 
    
    eta_second_law: float = 0.45           # % of Carnot limit
    motor_eff_curve_coeffs: tuple = (0.7, 0.4, -0.2) # Efficiency of the inverter/motor
    
    # Defrost parameters
    enable_defrost: bool = True
    defrost_max_penalty: float = 0.15      # Max energy loss fraction

class HeatPumpOutput(NamedTuple):
    thermal_power_w: Array
    electrical_power_w: Array
    cop: Array
    supply_temp_c: Array
    condenser_temp_c: Array # Exposed for debugging the physics

class ExogenousData(NamedTuple):
    ambient_temp: Array
    air_density: Array
    relative_humidity: Array # 0.0 to 1.0

# --- 2. The Mechanics ---

class HighFidelityHeatPump(eqx.Module):
    config: HeatPumpConfig
    n_rooms: int = eqx.field(static=True)
    cp_air: float = 1005.0  # J/kg*K

    def _calculate_defrost_penalty(self, ambient_temp: Array, rh: Array) -> Array:
        """
        Calculates efficiency penalty due to frosting.
        Gaussian risk centered at +1.0C, magnified by Humidity > 60%.
        """
        if not self.config.enable_defrost:
            return jnp.array(1.0)
            
        # Physics: Frost formation peaks when air holds moisture but surfaces are freezing.
        # Peak risk at +1.0C ambient (coil is sub-zero).
        temp_risk = jnp.exp(-((ambient_temp - 1.0)**2) / (2 * 4.0**2))
        
        # Physics: No moisture = No frost.
        humidity_risk = jax.nn.sigmoid(15.0 * (rh - 0.60))
        
        penalty_magnitude = self.config.defrost_max_penalty * temp_risk * humidity_risk
        return 1.0 - penalty_magnitude

    def _calculate_motor_efficiency(self, electrical_w: Array) -> Array:
        """
        Separates Electrical Efficiency (Inverter + Motor) from Thermodynamic Efficiency.
        """
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        plr = electrical_w / (max_w_per_room + 1e-6)
        c = self.config.motor_eff_curve_coeffs
        
        # Polynomial approximation of VFD/ECM motor efficiency
        eff = c[0] + (c[1] * plr) + (c[2] * (plr**2))
        return jnp.clip(eff, 0.1, 0.96)

    def calculate_output(self, power_state: Array, T_sink_c: Array, exo: ExogenousData) -> HeatPumpOutput:
        
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        
        # --- A. Airflow & UA Scaling (Fluid Dynamics) ---
        # 1. Variable Air Volume (VAV) logic
        plr = jnp.clip(power_state / (max_w_per_room + 1e-6), 0.0, 1.0)
        # Fan curve: starts at 20%, scales to 100%
        flow_ratio = 0.2 + 0.8 * plr 
        m_dot = self.config.design_air_flow_m3_s * exo.air_density * flow_ratio
        m_cp = m_dot * self.cp_air

        # 2. UA Scaling
        # Nusselt correlation: Heat transfer (h) scales with Velocity^0.8
        # Therefore, UA scales with flow_ratio^0.8
        ua_effective = self.config.ua_condenser_nominal * (flow_ratio ** 0.8)

        # --- B. Electrical Conversion ---
        eta_motor = self._calculate_motor_efficiency(power_state)
        shaft_power = power_state * eta_motor

        # --- C. Implicit Physics Solver (Fixed Point Iteration) ---
        # We solve for the Equilibrium Q where Thermodynamics (Carnot) matches Heat Transfer (UA).
        
        # Initial Guess for Q (prevents cold-start singularities)
        q_guess = power_state * 3.0 
        
        # Carry Tuple: (Current Estimate of Q, Current Estimate of T_cond)
        # We define T_cond guess as T_sink + 5.0 just to start the loop
        InitState = Tuple[Array, Array]
        init_carry: InitState = (q_guess, T_sink_c + 5.0)

        def energy_balance_step(carry: InitState, _) -> Tuple[InitState, None]:
            q_prev, _ = carry
            
            # 1. Air Side Physics
            # Temperature rise required to carry Q away
            delta_t_air = q_prev / (m_cp + 1e-4)
            t_supply_internal = T_sink_c + delta_t_air
            
            # 2. Heat Exchanger Physics (The "UA" Model)
            # To push Q through the HX, T_cond must be higher than T_supply.
            # Q = UA * (T_cond - T_supply)  ->  T_cond = T_supply + Q/UA
            approach_t = q_prev / (ua_effective + 1e-4)
            t_cond_c = t_supply_internal + approach_t
            
            # 3. Thermodynamic Cycle (Carnot)
            t_cond_k = t_cond_c + 273.15
            t_amb_k = exo.ambient_temp + 273.15
            
            # Lift Calculation (Clamped to 5K to avoid singularity)
            lift = jnp.maximum(5.0, t_cond_k - t_amb_k)
            cop_carnot = t_cond_k / lift
            
            # 4. Energy Balance
            cop_thermo = cop_carnot * self.config.eta_second_law
            q_new = shaft_power * cop_thermo
            
            # Soft Update (Damping factor 0.5 for numerical stability)
            q_update = 0.5 * q_prev + 0.5 * q_new
            
            return (q_update, t_cond_c), None

        # Execute Solver
        # We scan 6 times to converge. We only care about the final state (carry).
        (q_final_raw, t_cond_final), _ = jax.lax.scan(
            energy_balance_step, 
            init_carry, 
            None, 
            length=6
        )

        # --- D. Environmental Penalties ---
        defrost_mod = self._calculate_defrost_penalty(exo.ambient_temp, exo.relative_humidity)
        q_final = q_final_raw * defrost_mod
        
        # Handle the edge case where power is 0
        final_cop = jnp.where(power_state > 1.0, q_final / power_state, 0.0)
        
        # Recalculate T_supply based on final (defrosted) output
        # If defrost is active, Q drops, so T_supply drops.
        t_supply_final = T_sink_c + (q_final / (m_cp + 1e-4))

        return HeatPumpOutput(
            thermal_power_w=q_final,
            electrical_power_w=power_state,
            cop=final_cop,
            supply_temp_c=t_supply_final,
            condenser_temp_c=t_cond_final
        )

# --- 3. Verification Script ---

if __name__ == "__main__":
    # Test Parameters
    config = HeatPumpConfig(
        max_electrical_power_w=3000.0,
        min_electrical_power_w=500.0,
        ramp_rate_w_per_sec=100.0,
        design_air_flow_m3_s=0.5,
        max_supply_temp_c=50.0,
        ua_condenser_nominal=900.0
    )
    
    hp_model = HighFidelityHeatPump(config, n_rooms=1)
    
    # Run a sweep of Ambient Temps from -10C to +15C
    ambient_temps = jnp.linspace(-10, 15, 20)
    
    # Condition 1: High Humidity (Defrost Risk), Full Power
    exo_high_rh = ExogenousData(
        ambient_temp=ambient_temps,
        air_density=jnp.ones_like(ambient_temps) * 1.2,
        relative_humidity=jnp.ones_like(ambient_temps) * 0.9 # 90% RH
    )
    
    # Vectorize calculation
    vmap_calc = jax.vmap(hp_model.calculate_output, in_axes=(None, None, 0))
    
    # 2000W Input, 20C Return Temp
    output = vmap_calc(jnp.array(2000.0), jnp.array(20.0), exo_high_rh)
    
    print(f"{'Amb(C)':<8} | {'COP':<6} | {'Q_out(W)':<9} | {'T_supp(C)':<9} | {'T_cond(C)':<9}")
    print("-" * 55)
    for i in range(len(ambient_temps)):
        print(f"{ambient_temps[i]:<8.1f} | {output.cop[i]:<6.2f} | {output.thermal_power_w[i]:<9.0f} | "
              f"{output.supply_temp_c[i]:<9.1f} | {output.condenser_temp_c[i]:<9.1f}")

    # Note on fidelity observation:
    # 1. Check around +1.0C. COP should dip due to Defrost Logic.
    # 2. Check T_cond vs T_supply. The gap is the approach temp, determined by UA.