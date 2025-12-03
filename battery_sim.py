import jax
import jax.numpy as jnp
import jax.scipy.ndimage as ndimage
import equinox as eqx
import diffrax
from typing import Optional

# ------------------ CONSTANTS & TYPES ------------------ #
CONST_R_GAS = 8.314  # J/(mol K)

class BatteryConfig(eqx.Module):
    # Capacity & Nominal
    capacity_ah: float
    n_series: int
    n_parallel: int = 1
    
    # Thermal Properties
    C_core: float = 800.0   # J/K
    C_case: float = 200.0   # J/K
    R_core_case: float = 1.2 # K/W
    R_case_ambient: float = 4.0 # K/W
    
    # Limits
    v_min_cell: float = 2.5
    v_max_cell: float = 4.2
    i_max_chg: float = 150.0 
    i_max_dch: float = 250.0
    
    # Environment
    ambient_temp_k: float = 298.15

class BatteryInputs(eqx.Module):
    # 0.0 = Current Control, 1.0 = Power Control
    control_mode: float 
    value: float 

class BatteryState(eqx.Module):
    soc: jax.Array           
    v_rc: jax.Array          
    v_hyst: jax.Array        
    temp_core_k: jax.Array   
    temp_case_k: jax.Array   
    ah_throughput: jax.Array 

class BatteryOutput(eqx.Module):
    terminal_voltage: jax.Array
    terminal_current: jax.Array
    actual_power: jax.Array
    heat_total: jax.Array
    temp_core_C: jax.Array
    is_power_limited: jax.Array
    is_voltage_violation: jax.Array

# ------------------ MATH HELPERS ------------------ #

def smooth_sign(x, alpha=50.0):
    return jnp.tanh(alpha * x)

def smooth_clip(x, min_val, max_val, k=20.0):
    # LogSumExp based soft clamp
    lower = (jnp.logaddexp(k * x, k * min_val) / k)
    return -(jnp.logaddexp(-k * lower, -k * max_val) / k)

# ------------------ THE MODEL ------------------ #

class HighFidelityBattery(eqx.Module):
    config: BatteryConfig
    
    # --- Grids ---
    soc_grid: jax.Array
    temp_grid: jax.Array
    
    # --- Parameter Surfaces ---
    # We store these as simple arrays. 
    # Shapes: (n_soc,) or (n_soc, n_temp) or (n_rc, n_soc, n_temp)
    ocv_curve: jax.Array        
    hyst_max_curve: jax.Array   
    hyst_rate_curve: jax.Array  
    dudt_curve: jax.Array       
    
    r0_surface: jax.Array       
    r_rc_surfaces: jax.Array     
    c_rc_surfaces: jax.Array    
    
    capacity_as: float
    n_rc: int

    def __init__(
        self, 
        config: BatteryConfig, 
        n_rc: int = 2,
        grid_soc_n: int = 50,
        grid_temp_n: int = 10
    ):
        self.config = config
        self.n_rc = n_rc
        self.capacity_as = config.capacity_ah * 3600.0
        
        # 1. Grids
        self.soc_grid = jnp.linspace(0, 1, grid_soc_n)
        self.temp_grid = jnp.linspace(253.15, 333.15, grid_temp_n) # -20C to 60C

        # 2. Synthetic Physics Generation
        # Note: In production, these would be loaded from CSVs
        
        # OCV (LFP-like plateau)
        self.ocv_curve = (3.2 + 0.5 * self.soc_grid**0.8 - 0.1 * jnp.exp(-15*self.soc_grid) + 0.05 * jnp.exp(-20*(1-self.soc_grid))) * config.n_series
        
        # Hysteresis
        self.hyst_max_curve = (0.05 + 0.1 * jnp.exp(-10*self.soc_grid)) * config.n_series
        self.hyst_rate_curve = 20.0 + 100.0 * (self.soc_grid - 0.5)**2
        self.dudt_curve = 0.0002 * jnp.sin(self.soc_grid * 4 * jnp.pi) * config.n_series
        
        # Surfaces
        s_mesh, t_mesh = jnp.meshgrid(self.soc_grid, self.temp_grid, indexing='ij')
        
        r_base = 0.02 * config.n_series / config.n_parallel
        temp_factor = jnp.exp(2000 * (1/t_mesh - 1/298.15))
        soc_factor = 1 + 2*jnp.exp(-5*s_mesh) + 2*jnp.exp(-5*(1-s_mesh))
        self.r0_surface = r_base * temp_factor * soc_factor
        
        r_rc1 = self.r0_surface * 0.5
        c_rc1 = jnp.ones_like(r_rc1) * 2000.0
        r_rc2 = self.r0_surface * 1.5
        c_rc2 = jnp.ones_like(r_rc2) * 50000.0
        
        self.r_rc_surfaces = jnp.stack([r_rc1, r_rc2])
        self.c_rc_surfaces = jnp.stack([c_rc1, c_rc2])

    # ------------------ OPTIMIZED INTERPOLATION ------------------ #
    
    def _interp_1d(self, soc_norm, grid_vals):
        # soc_norm is [0, 1]
        # map_coordinates expects index coordinates
        idx = soc_norm * (self.soc_grid.shape[0] - 1)
        return ndimage.map_coordinates(grid_vals, [idx], order=1, mode='nearest')
        
    def _interp_2d(self, soc_norm, temp, surface):
        # 1. Normalize Temp to [0, 1]
        t_min = self.temp_grid[0]
        t_max = self.temp_grid[-1]
        temp_norm = (jnp.clip(temp, t_min, t_max) - t_min) / (t_max - t_min)
        
        # 2. Scale to indices
        s_idx = soc_norm * (self.soc_grid.shape[0] - 1)
        t_idx = temp_norm * (self.temp_grid.shape[0] - 1)
        
        # 3. Stack for map_coordinates: shape (ndim, 1) or (ndim,)
        coords = jnp.stack([s_idx, t_idx])
        return ndimage.map_coordinates(surface, coords, order=1, mode='nearest')

    def _get_physics_params(self, soc, temp):
        # We assume soc is already clamped 0-1 for safety
        
        # 1D
        v_ocv = self._interp_1d(soc, self.ocv_curve)
        v_hyst_lim = self._interp_1d(soc, self.hyst_max_curve)
        gamma = self._interp_1d(soc, self.hyst_rate_curve)
        dudt = self._interp_1d(soc, self.dudt_curve)
        
        # 2D / 3D
        r0 = self._interp_2d(soc, temp, self.r0_surface)
        
        # vmap over the n_rc dimension (axis 0 of surfaces)
        r_rc = jax.vmap(lambda s: self._interp_2d(soc, temp, s))(self.r_rc_surfaces)
        c_rc = jax.vmap(lambda s: self._interp_2d(soc, temp, s))(self.c_rc_surfaces)
        
        return v_ocv, v_hyst_lim, gamma, dudt, r0, r_rc, c_rc

    # ------------------ SOLVER CORE ------------------ #

    def _calculate_limits_and_current(self, params, state, inputs):
        v_ocv, _, _, _, r0, r_rc, _ = params
        
        # 1. Effective Voltage (ECM Topology)
        # # V_term = V_ocv + V_hyst - V_rc - I*R0
        v_eff = v_ocv + state.v_hyst - jnp.sum(state.v_rc)
        
        # 2. Calculate Physical Current Limits (Instantaneous)
        i_lim_v_min = (v_eff - (self.config.v_min_cell * self.config.n_series)) / r0
        i_lim_v_max = (v_eff - (self.config.v_max_cell * self.config.n_series)) / r0
        
        i_upper = smooth_clip(i_lim_v_min, 0.0, self.config.i_max_dch, k=10.0)
        i_lower = smooth_clip(i_lim_v_max, -self.config.i_max_chg, 0.0, k=10.0)
        
        # 3. Determine Desired Current
        def get_current_from_power():
            p_req = inputs.value
            
            # Quadratic solution: R0*I^2 - V_eff*I + P = 0
            discriminant = v_eff**2 - 4 * r0 * p_req
            
            # Robustness: Maximum Power Point Logic
            # If discriminant < 0, power is physically impossible. 
            # We clip discriminant to 0 to prevent NaNs, effectively providing Max Power.
            safe_disc = jnp.maximum(discriminant, 0.0)
            
            # Standard quadratic formula
            i_sol = (v_eff - jnp.sqrt(safe_disc)) / (2 * r0)
            return i_sol

        i_req = jnp.where(inputs.control_mode > 0.5, get_current_from_power(), inputs.value)
        
        # 4. Apply Limits
        i_actual = smooth_clip(i_req, i_lower, i_upper, k=20.0)
        is_limited = jnp.abs(i_actual - i_req) > 0.1
        
        return i_actual, v_eff, is_limited

    def __call__(self, t: float, state: BatteryState, inputs: BatteryInputs) -> tuple[BatteryState, BatteryOutput]:
        
        # --- CRITICAL FIX: Safe SOC for Lookups ---
        # Prevents NaNs in OCV equations if solver attempts soc < 0 or > 1
        soc_safe = jnp.clip(state.soc, 0.0, 1.0)
        
        # A. Parameter Lookup
        params = self._get_physics_params(soc_safe, state.temp_core_k)
        v_ocv, v_hyst_lim, gamma, dudt, r0, r_rc, c_rc = params
        
        # B. Circuit Solution
        i_flow, v_eff, is_lim = self._calculate_limits_and_current(params, state, inputs)
        v_term = v_eff - i_flow * r0
        
        # Check for Voltage Violations (Diagnostic)
        v_min_abs = self.config.v_min_cell * self.config.n_series
        v_max_abs = self.config.v_max_cell * self.config.n_series
        is_violation = (v_term < v_min_abs) | (v_term > v_max_abs)
        
        # C. Derivative Calculations
        
        # 1. SOC (Coulomb Counting)
        d_soc = -i_flow / self.capacity_as
        
        # 2. RC Dynamics
        d_v_rc = (i_flow / c_rc) - (state.v_rc / (r_rc * c_rc))
        
        # 3. Hysteresis Dynamics (Stabilized)
        # 
        hyst_target = smooth_sign(i_flow) * v_hyst_lim
        # We use i_flow magnitude as the driver. If i_flow is 0, d_hyst is 0.
        d_hyst = jnp.abs(i_flow) * gamma * (hyst_target - state.v_hyst)
        
        # 4. Thermal Dynamics
        q_irrev = (i_flow**2 * r0) + jnp.sum(state.v_rc**2 / r_rc)
        # Entropic: Correct sign convention depends on dU/dT data. 
        # Assuming dU/dT is defined such that T*dU/dT*I is heat gen rate.
        q_rev = state.temp_core_k * dudt * i_flow 
        q_gen_total = q_irrev + q_rev
        
        q_core_to_case = (state.temp_core_k - state.temp_case_k) / self.config.R_core_case
        q_case_to_amb = (state.temp_case_k - self.config.ambient_temp_k) / self.config.R_case_ambient
        
        d_temp_core = (q_gen_total - q_core_to_case) / self.config.C_core
        d_temp_case = (q_core_to_case - q_case_to_amb) / self.config.C_case
        
        # 5. Aging
        d_ah = jnp.abs(i_flow) / 3600.0
        
        # D. Packaging
        new_state_derivs = BatteryState(
            soc=d_soc,
            v_rc=d_v_rc,
            v_hyst=d_hyst,
            temp_core_k=d_temp_core,
            temp_case_k=d_temp_case,
            ah_throughput=d_ah
        )
        
        outputs = BatteryOutput(
            terminal_voltage=v_term,
            terminal_current=i_flow,
            actual_power=v_term * i_flow,
            heat_total=q_gen_total,
            temp_core_C=state.temp_core_k - 273.15,
            is_power_limited=jnp.array(is_lim, dtype=float),
            is_voltage_violation=jnp.array(is_violation, dtype=float)
        )
        
        return new_state_derivs, outputs

# ------------------ SIMULATION RUNNER ------------------ #

# 1. Setup
conf = BatteryConfig(capacity_ah=100.0, n_series=96)
model = HighFidelityBattery(conf)

# 2. Initial State
y0 = BatteryState(
    soc=jnp.array(0.8),
    v_rc=jnp.zeros(model.n_rc),
    v_hyst=jnp.array(0.0),
    temp_core_k=jnp.array(298.15),
    temp_case_k=jnp.array(298.15),
    ah_throughput=jnp.array(0.0)
)

# 3. Input (Power Pulse)
def input_func(t):
    # Smooth step from 60kW to 0
    p_demand = 60000.0 * (1.0 - (0.5 * (jnp.tanh(2.0 * (t - 100.0)) + 1.0)))
    return BatteryInputs(control_mode=1.0, value=p_demand)

# 4. Diffrax Wrap
def vector_field(t, y, args):
    inp = input_func(t)
    derivs, _ = model(t, y, inp)
    return derivs

# 5. Solve
term = diffrax.ODETerm(vector_field)
solver = diffrax.Kvaerno5()

# --- FIX 1: Relax Tolerances slightly ---
# rtol=1e-3 is 0.1% accuracy, sufficient for battery engineering.
# Prevents solver from taking nanosecond steps to resolve 6th decimal place.
stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)

# --- FIX 2: Use equinox.filter_jit ---
# This handles the static configuration vs dynamic arrays automatically.
@eqx.filter_jit
def run_sim():
    return diffrax.diffeqsolve(
        term, 
        solver, 
        t0=0.0, 
        t1=200.0, 
        dt0=0.05, # Start with a conservative step
        y0=y0, 
        stepsize_controller=stepsize_controller,
        
        # --- FIX 3: Increase Max Steps ---
        # 200s / 0.01s (typical step) = 20,000 steps. 
        # 10,000 was physically insufficient. We bump to 100k.
        max_steps=100000, 
        
        # --- FIX 4: Prevent Crash ---
        # If it fails, return the partial result so we can see WHERE it failed.
        throw=False 
    )

solution = run_sim()

# 6. Check Results safely
# If result is generic error, max_steps was hit
final_t = solution.ts[-1]
status = solution.result

print(f"Solver Status: {status} (0=Success, 1=MaxSteps)")
print(f"Simulation reached t = {final_t:.2f}s")
print(f"Final SOC: {solution.ys.soc[-1]:.4f}")