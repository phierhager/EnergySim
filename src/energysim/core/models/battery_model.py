# High-Fidelity Multi-RC Battery Model (JAX + Equinox)
# Refactored: Vectorized RC interpolation, fully differentiable

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from energysim.core.shared.data_structs import BatteryConfig, BatteryInputs, BatteryOutput, Array, BatteryState

# --- Helper for Differentiable Clamping ---
def soft_clamp(x, min_val, max_val, k=50.0):
    """
    Differentiable clamping using LogSumExp (smooth maximum).
    k: sharpness of the transition.
    """
    # Smooth Max(x, min_val)
    lower = (jnp.logaddexp(k * x, k * min_val) / k)
    # Smooth Min(lower, max_val) -> -SmoothMax(-lower, -max_val)
    return -(jnp.logaddexp(-k * lower, -k * max_val) / k)

def generate_chebyshev_grid(n_points: int) -> Array:
    """
    Generates a non-uniform grid clustered at 0.0 and 1.0.
    Based on Chebyshev nodes, shifted to [0, 1].
    """
    k = jnp.arange(n_points)
    # Cosine distribution: dense at ends, sparse in middle
    # 0.5 * (1 - cos(pi * k / (n-1)))
    nodes = 0.5 * (1.0 - jnp.cos(jnp.pi * k / (n_points - 1)))
    return nodes

# --- 3. Differentiable Sign Helper ---
def soft_sign(x, alpha=100.0):
    """
    Differentiable approximation of sign(x).
    Using tanh allows gradients to flow through zero crossings.
    """
    return jnp.tanh(alpha * x)

# ------------------ ABSTRACT BASE ------------------ #

class AbstractBatteryModel(eqx.Module):
    config: BatteryConfig

    def dynamics(self, t: float, state: BatteryState, inputs: BatteryInputs) -> BatteryState:
        """
        Calculates the time derivative of the state vector.
        Args:
            t: Time (seconds)
            state: BatteryState object containing [SOC, Temp, V_rc1, V_rc2...]
            inputs: Power request and boundary conditions
        Returns:
            Derivatives [dSOC/dt, dTemp/dt, dVrc/dt...]
        """
        raise NotImplementedError

    def calculate_output(self, state: BatteryState, inputs: BatteryInputs) -> BatteryOutput:
        """
        Calculates the instantaneous physical outputs (V, I, P).
        Args:
            state: BatteryState object containing [SOC, Temp, V_rc1, V_rc2...]
            inputs: Power request
        Returns:
            BatteryOutput struct
        """
        raise NotImplementedError


# ------------------ NULL MODEL ------------------ #

class NullBatteryModel(AbstractBatteryModel):
    """
    Represents a missing, disconnected, or bypassed battery.
    
    Physics:
    - Accepts power requests but forces Current = 0.
    - Voltage stays fixed at Nominal Voltage (Open Circuit).
    - SOC and Temperature do not change.
    - Heat generation is 0.
    """
    
    # We store a nominal voltage just so the system doesn't see 0.0V (which might cause divide-by-zero downstream)
    nominal_voltage: float

    def __init__(self, config: BatteryConfig):
        self.config = config
        # Default to 360V or whatever is in config, used purely for 'potential'
        self.nominal_voltage = getattr(config, "v_nom", 360.0) 

    def dynamics(self, t: float, state: BatteryState, inputs: BatteryInputs) -> BatteryState:
        return BatteryState(
            soc=jnp.array(0.0),
            v_rc=jnp.zeros_like(state.v_rc),
            temp_core_k=jnp.array(0.0),
            temp_case_k=jnp.array(0.0)
        )

    def calculate_output(self, state: BatteryState, inputs: BatteryInputs) -> BatteryOutput:
        # It is an open circuit. 
        # Voltage exists (potential), but Current is 0. Therefore Power is 0.
        
        return BatteryOutput(
            actual_power_w=jnp.array(0.0),
            voltage_v=jnp.array(self.nominal_voltage),
            current_a=jnp.array(0.0),
            heat_generation_w=jnp.array(0.0),
        )


# ------------------ HIGH-FIDELITY MODEL ------------------ #

class MultiRCBatteryModel(eqx.Module):
    """
    High-fidelity Li-ion battery model (Refactored).
    Improvements:
    - Entropic (Reversible) Heating included.
    - Configurable Arrhenius activation energy.
    - Gradient-safe power solver (no NaNs).
    """

    config: BatteryConfig

    # --- 1. Grid Lookups ---
    soc_grid: Array
    ocv_avg_grid: Array
    hysteresis_mag_grid: Array
    dudt_grid: Array
    r0_grid: Array
    r_grid: Array
    c_grid: Array

    # --- 2. Thermal Parameters ---
    C_core: float
    C_case: float
    R_core_case: float
    R_case_ambient: float
    ambient_temp: float

    # --- 3. Arrhenius ---
    # Essential for temperature sensitivity gradients
    activation_energy_r0_k: float 
    activation_energy_rc_k: float    

    # --- 4. Hysteresis Parameters ---
    hysteresis_rate: float

    # --- 5. Constraints & Scaling (RESTORED) ---
    v_min: float
    v_max: float
    n_series: int  # Needed for OCV scaling
    capacity_as: float
    n_rc: int
    
    # Optional constraints (allow None for no limit)
    max_charge_current_a: Optional[float]
    max_discharge_current_a: Optional[float]

    def __init__(
        self,
        config: BatteryConfig,
        n_rc: int = 2,
        grid_points: int = 65,
        soc_grid: Optional[Array] = None,
        ocv_avg_grid: Optional[Array] = None,
        hysteresis_mag_grid: Optional[Array] = None,
        r0_grid: Optional[Array] = None,
        r_grid: Optional[Array] = None,
        c_grid: Optional[Array] = None,
    ):
        self.config = config
        self.n_rc = n_rc
        self.capacity_as = config.capacity_ah * 3600.0
        
        # --- Restore Scaling & Constraints ---
        self.n_series = config.n_series
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.max_charge_current_a = config.max_charge_current_a
        self.max_discharge_current_a = config.max_discharge_current_a
        
        # --- Restore Arrhenius ---
        self.activation_energy_r0_k = config.activation_energy_r0_k
        self.activation_energy_rc_k = config.activation_energy_rc_k

        # --- Grid Initialization ---
        if soc_grid is None:
            self.soc_grid = generate_chebyshev_grid(grid_points)
        else:
            self.soc_grid = soc_grid

        if ocv_avg_grid is None:
            # Scale OCV by n_series
            self.ocv_avg_grid = (3.2 + 0.8 * self.soc_grid**0.5) * self.n_series
            
        if hysteresis_mag_grid is None:
            # Scale Hysteresis by n_series
            shape = 0.02 + 0.03 * (self.soc_grid - 0.5)**2
            self.hysteresis_mag_grid = shape * self.n_series
        else:
            self.hysteresis_mag_grid = hysteresis_mag_grid

        self.dudt_grid = jnp.zeros_like(self.soc_grid)
        
        # Scale Resistance by n_series
        self.r0_grid = r0_grid if r0_grid is not None else \
            jnp.full_like(self.soc_grid, 0.01) * self.n_series
            
        self.r_grid = r_grid if r_grid is not None else \
             jnp.tile(jnp.full_like(self.soc_grid, 0.005)[:, None], (1, n_rc))
        self.c_grid = c_grid if c_grid is not None else \
             jnp.full((len(self.soc_grid), n_rc), 1000.0)

        # Thermal
        self.C_core = config.C_core
        self.C_case = config.C_case
        self.R_core_case = config.R_core_case
        self.R_case_ambient = config.R_case_ambient
        self.ambient_temp = config.ambient_temp

        self.hysteresis_rate = config.hysteresis_rate

    # ------------------ INTERPOLATION ------------------ #
    def _get_params(self, soc: float, temp_core_k: float):
        """
        Returns OCV, dU/dT, R0, R_i[], C_i[] for given SOC and temp.
        
        NOTE: We pass the CORE temperature (temp_core_k) here, not case temperature.
        Chemistry happens in the core.
        """
        soc_c = jnp.clip(soc, 0.0, 1.0)
        # Avoid div/0 with soft lower bound on Temp
        temp_safe = jnp.maximum(temp_core_k, 200.0) 
        
        # k_arr = exp( E_a/k_b * (1/T - 1/T_ref) )
        # We assume activation_energy_k is pre-divided by Boltzmann constant
        arrhenius_r0 = jnp.exp(self.config.activation_energy_r0_k * (1.0/temp_safe - 1.0/self.config.ref_temp_k))
        arrhenius_rc = jnp.exp(self.config.activation_energy_rc_k * (1.0/temp_safe - 1.0/self.config.ref_temp_k))

        # 1D Interp
        v_ocv_avg = jnp.interp(soc_c, self.soc_grid, self.ocv_avg_grid)
        v_hyst_max = jnp.interp(soc_c, self.soc_grid, self.hysteresis_mag_grid)
        
        dudt = jnp.interp(soc_c, self.soc_grid, self.dudt_grid)
        r0 = jnp.interp(soc_c, self.soc_grid, self.r0_grid) * arrhenius_r0
        
        # 2D Interp (Manual vmap is correct)
        # r_grid shape: (SOC, RC) -> we want to interp along SOC axis for each RC column
        r_i = jax.vmap(lambda fp: jnp.interp(soc_c, self.soc_grid, fp), in_axes=1, out_axes=0)(self.r_grid) * arrhenius_rc
        c_i = jax.vmap(lambda fp: jnp.interp(soc_c, self.soc_grid, fp), in_axes=1, out_axes=0)(self.c_grid)

        return v_ocv_avg, v_hyst_max, dudt, r0, r_i, c_i
    
    # ------------------ DYNAMICS ------------------ #
    def dynamics(self, t: float, state: BatteryState, inputs: BatteryInputs) -> BatteryState:
        # 1. Unpack
        v_ocv_avg, v_hyst_max, dudt, r0, r_i, c_i = self._get_params(state.soc, state.temp_core_k)
        
        # 2. Calculate Effective Voltage
        # V_eff = OCV_avg + V_hysteresis - V_RC_polarization
        v_eff = v_ocv_avg + state.v_hyst - jnp.sum(state.v_rc)

        # 3. Solve Circuit (Algebraic Step)
        i_flow, _ = self._solve_circuit_state(inputs.power_w, v_eff, r0)
        
        # 3. Derivatives
        # SOC: Coulomb Counting
        dsoc = -i_flow / self.capacity_as
        # V_rc: dV/dt = I/C - V/RC
        dvrc = (i_flow / c_i) - (state.v_rc / (r_i * c_i))

        # Equation: dV_h/dt = |I| * gamma * (sgn(I)*V_max - V_h)
        # We use soft_sign for differentiability.
        # This drives V_hyst toward +V_max during Charge and -V_max during Discharge.
        target_hyst = soft_sign(i_flow) * v_hyst_max
        # Rate is proportional to current magnitude (Coulombic efficiency concept)
        dv_hyst = jnp.abs(i_flow) * self.hysteresis_rate * (target_hyst - state.v_hyst)
        
        # A. Heat Generation (Happens in Core)
        q_irrev = (i_flow**2 * r0) + jnp.sum(state.v_rc**2 / r_i)
        q_rev = state.temp_core_k * dudt * i_flow 
        q_gen = q_irrev + q_rev
        
        # B. Heat Fluxes
        # Flux from Core -> Case (Conduction)
        # Driven by delta between Core (temp_k) and Case (temp_case_k)
        q_core_to_case = (state.temp_core_k - state.temp_case_k) / self.R_core_case
        
        # Flux from Case -> Ambient (Convection)
        q_case_to_amb = (state.temp_case_k - self.ambient_temp) / self.R_case_ambient
        
        # C. Temperature Derivatives
        d_temp_core = (q_gen - q_core_to_case) / self.C_core
        d_temp_case = (q_core_to_case - q_case_to_amb) / self.C_case

        return BatteryState(
            soc=dsoc,
            v_hyst=dv_hyst,
            temp_core_k=d_temp_core,      # Core Temp derivative
            temp_case_k=d_temp_case, # Case Temp derivative (Ensure BatteryState has this field)
            v_rc=dvrc
        )

    # ------------------ OUTPUT ------------------ #
    def calculate_output(self, state: BatteryState, inputs: BatteryInputs) -> BatteryOutput:
        # Re-calculate parameters (cheap compared to solver)
        v_ocv, dudt, r0, r_i, c_i = self._get_params(state.soc, state.temp_core_k)
        v_eff = v_ocv - jnp.sum(state.v_rc)
        
        # Re-solve circuit
        # (JAX JIT will likely merge this with dynamics if called sequentially)
        i_flow, v_term = self._solve_circuit_state(inputs.power_w, v_eff, r0)
        
        # Recalc heat for logging
        q_irrev = (i_flow**2 * r0) + jnp.sum(state.v_rc**2 / r_i)
        q_rev = state.temp_core_k * dudt * i_flow
        
        return BatteryOutput(
            actual_power_w=v_term * i_flow,
            voltage_v=v_term,
            current_a=i_flow,
            heat_generation_w=q_irrev + q_rev,
        )

    def _solve_circuit_state(
        self, 
        power_target: float, 
        v_eff: float, 
        r0: float
    ) -> tuple[Array, Array]:
        """
        Pure function to solve for Current and Terminal Voltage.
        """
        # 1. Quadratic Solve for Constant Power
        # V = V_eff - I*R0  =>  P = V*I = V_eff*I - I^2*R0
        # R0*I^2 - V_eff*I + P = 0
        
        # D = b^2 - 4ac
        D = v_eff**2 - 4.0 * r0 * power_target
        
        # Gradient Safe Sqrt
        sqrt_D = jnp.sqrt(jnp.maximum(D, 0.0))
        
        # Roots: (V_eff +/- sqrt(D)) / 2R0
        # Discharge (P>0): Higher current reduces voltage.
        # Charge (P<0): Negative current increases voltage.
        # We generally want the smaller magnitude current that satisfies the power.
        
        i_cp = (v_eff - sqrt_D) / (2.0 * r0)
        
        # If collapsed (D < 0), we are demanding more power than battery can give.
        # Physics implies we hit the peak of the parabola: I = V_eff / 2R0
        i_cp = jnp.where(D < 0, v_eff / (2.0 * r0), i_cp)

        # 2. Voltage Constraints
        # Calculate what V would be at this current
        v_predicted = v_eff - i_cp * r0
        
        # Smoothly Clamp Voltage
        # This replaces the hard "if/else" logic with a differentiable function
        # that respects v_min and v_max.
        v_clamped = soft_clamp(v_predicted, self.config.v_min, self.config.v_max, k=20.0)
        
        # 3. Recalculate Current based on Clamped Voltage (CV Mode)
        # I = (V_eff - V_term) / R0
        i_final = (v_eff - v_clamped) / r0
        
        # 4. Current Limits (Hard or Soft)
        if self.config.max_discharge_current_a is not None:
             # Equivalent to SoftMin: -SoftMax(-x, -limit)
            neg_i = -i_final
            neg_limit = -self.config.max_discharge_current_a
            # Smooth max of negative values
            neg_i_clamped = (jnp.logaddexp(50.0 * neg_i, 50.0 * neg_limit) / 50.0)
            i_final = -neg_i_clamped
        if self.config.max_charge_current_a is not None:
            limit = -self.config.max_charge_current_a
            i_final = (jnp.logaddexp(50.0 * i_final, 50.0 * limit) / 50.0)

        v_final = v_eff - i_final * r0
        
        return i_final, v_final