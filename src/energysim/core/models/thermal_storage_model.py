"""
Platinum Standard Thermal Storage Model (JAX/Equinox).

Fidelity Features:
 1. Conservative State: Transport solves d(rho*h)/dt, strictly conserving Energy.
 2. Consistent Thermodynamics: Analytic inversion of h(T) for CFL checks.
 3. Isochoric Correction: Newton-Raphson loop reconciles T with Energy Density.
 4. High-Res Numerics: TVD Superbee limiter with Double-Ghost topology.
 5. Stability: Implicit Diffusion via O(N) Thomas Algorithm.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple

# -----------------------
# Configuration & Inputs
# -----------------------
class ThermalStorageConfig(NamedTuple):
    n_nodes: int = 100
    height_m: float = 2.0
    volume_m3: float = 0.3
    loss_coeff_w_k: float = 0.5
    vertical_conductivity_w_mk: float = 0.6

class StorageInputs(NamedTuple):
    mdot_net_kg_s: float  # + Charge (Flow Down), - Discharge (Flow Up)
    T_inlet_c: float
    T_ambient_c: float = 20.0

# -----------------------
# Thermophysics (Kell 1975 & Integrated Cp)
# -----------------------
def density_water_kg_m3(T: jnp.ndarray) -> jnp.ndarray:
    """
    Kell (1975) formulation for liquid water density.
    """
    return (999.842594 + 6.793952e-2 * T - 9.095290e-3 * T**2 + 
            1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536332e-9 * T**5)

def specific_heat_water_j_kgk(T: jnp.ndarray) -> jnp.ndarray:
    """Linear approximation suitable for 0-100C."""
    return 4210.0 - 0.9 * T

def specific_enthalpy_j_kg(T: jnp.ndarray) -> jnp.ndarray:
    """
    Exact integral of Cp(T) = 4210 - 0.9T.
    Reference state: h(0) = 0.
    """
    return 4210.0 * T - 0.45 * T**2

def temp_from_enthalpy_exact(h: jnp.ndarray) -> jnp.ndarray:
    """
    Analytic inverse of h(T).
    Solves 0.45 T^2 - 4210 T + h = 0 for the physical root.
    """
    # Discriminant b^2 - 4ac
    # a = -0.45, b = 4210, c = -h (rearranged)
    # T = (-4210 + sqrt(4210^2 - 4*0.45*h)) / -0.9
    # Simplified: (4210 - sqrt(4210^2 - 1.8*h)) / 0.9
    
    discriminant = 4210.0**2 - 1.8 * h
    # Safe sqrt for numerical stability (prevents NaN on slight negative noise)
    sqrt_D = jnp.sqrt(jnp.maximum(0.0, discriminant))
    return (4210.0 - sqrt_D) / 0.9

# -----------------------
# Solvers
# -----------------------
def limiter_superbee(r: jnp.ndarray) -> jnp.ndarray:
    """
    Superbee Flux Limiter (Compressive).
    Preserves sharp gradients (thermoclines) better than Minmod or Van Leer.
    """
    return jnp.maximum(0.0, jnp.maximum(jnp.minimum(2.0 * r, 1.0), jnp.minimum(r, 2.0)))

def solve_tdma_scan(lower: jnp.ndarray, main: jnp.ndarray, upper: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Thomas Algorithm implemented via jax.lax.scan.
    Solves Tridiagonal System Ax = d in O(N) time.
    """
    # 1. Forward Elimination
    def forward_scan(carry, x):
        c_prev, d_prev = carry
        a, b, c, rhs = x
        
        denom = b - a * c_prev
        # Robust division protection
        denom = jnp.where(jnp.abs(denom) < 1e-15, 1.0, denom)
        
        c_curr = c / denom
        d_curr = (rhs - a * d_prev) / denom
        return (c_curr, d_curr), (c_curr, d_curr)

    init_carry = (0.0, 0.0)
    _, (cp, dp) = jax.lax.scan(forward_scan, init_carry, (lower, main, upper, d))

    # 2. Backward Substitution
    def backward_scan(x_next, inputs):
        c_curr, d_curr = inputs
        x_curr = d_curr - c_curr * x_next
        return x_curr, x_curr

    # Start from last element
    _, x_rev = jax.lax.scan(backward_scan, 0.0, (cp[::-1], dp[::-1]))
    return x_rev[::-1]

def newton_raphson_temperature(E_vol_target: jnp.ndarray, T_guess: jnp.ndarray) -> jnp.ndarray:
    """
    Recovers T from Volumetric Energy Density (E = rho(T) * h(T)).
    Crucial for maintaining consistency between EOS and Conservation Laws.
    """
    def f_val(T):
        return density_water_kg_m3(T) * specific_enthalpy_j_kg(T) - E_vol_target
    
    def df_val(T):
        rho = density_water_kg_m3(T)
        h = specific_enthalpy_j_kg(T)
        cp = specific_heat_water_j_kgk(T)
        
        # d(rho)/dT
        drho = (6.793952e-2 
                - 2 * 9.095290e-3 * T 
                + 3 * 1.001685e-4 * T**2 
                - 4 * 1.120083e-6 * T**3 
                + 5 * 6.536332e-9 * T**4)
        
        return rho * cp + h * drho

    def step(i, T):
        f = f_val(T)
        df = df_val(T)
        return T - f / (df + 1e-12)

    # 3 iterations is sufficient for machine precision given T_guess is previous step
    return jax.lax.fori_loop(0, 3, step, T_guess)

# -----------------------
# Highe-Fidelity Thermal Storage Model
# -----------------------
class ThermalStorageModel(eqx.Module):
    config: ThermalStorageConfig

    def __init__(self, config: ThermalStorageConfig):
        self.config = config

    def step(self, t: float, T: jnp.ndarray, inputs: StorageInputs, dt: float) -> jnp.ndarray:
        """
        Performs one high-fidelity time step.
        """
        N = self.config.n_nodes
        H = self.config.height_m
        V = self.config.volume_m3
        dz = H / N
        A_cross = V / H
        
        # --- 1. Advection Step (Conservative Enthalpy Formulation) ---
        
        # Convert T to Enthalpy
        h = specific_enthalpy_j_kg(T)
        h_inlet = specific_enthalpy_j_kg(inputs.T_inlet_c)
        
        mdot = inputs.mdot_net_kg_s
        is_charging = mdot >= 0
        
        # Double Ghost Padding (Topology Safety)
        # Prevents array wrapping and allows higher-order stencils at boundaries
        hg_top = jnp.where(is_charging, h_inlet, h[0])
        hg_bot = jnp.where(is_charging, h[-1], h_inlet)
        
        # Layout: [G, G, Real(0..N), G, G]
        h_ext = jnp.concatenate([
            jnp.array([hg_top, hg_top]), 
            h, 
            jnp.array([hg_bot, hg_bot])
        ])
        
        faces = jnp.arange(N + 1) # Flux interfaces

        def compute_flux(i):
            idx = i + 1 # Align with h_ext
            
            # Stencil Selection (Upwind logic)
            h_C = jnp.where(is_charging, h_ext[idx], h_ext[idx+1])   # Center
            h_D = jnp.where(is_charging, h_ext[idx+1], h_ext[idx])   # Downstream
            h_U = jnp.where(is_charging, h_ext[idx-1], h_ext[idx+2]) # Upstream
            
            # Smoothness Monitor (r)
            denom = h_D - h_C
            r = jnp.where(jnp.abs(denom) < 1e-9, 0.0, (h_C - h_U) / denom)
            r = jnp.maximum(0.0, r)
            
            # Boundary Clamp: Force 1st order at the physical inlet
            is_inlet_face = jnp.logical_or(is_charging & (i == 0), (not is_charging) & (i == N))
            phi = jnp.where(is_inlet_face, 0.0, limiter_superbee(r))
            
            # Mass Flux (Constant)
            J_mass = jnp.abs(mdot) / A_cross
            
            # Courant Number Calculation (White Box Rigor)
            # Use EXACT inverse to find T_local, then Density
            T_local = temp_from_enthalpy_exact(h_C)
            rho_local = density_water_kg_m3(T_local)
            
            u_local = J_mass / rho_local
            nu = u_local * dt / dz
            
            # TVD Flux
            flux_corr = 0.5 * J_mass * (1 - nu) * phi * (h_D - h_C)
            F_low = J_mass * h_C
            
            return F_low + flux_corr

        F_faces = jax.vmap(compute_flux)(faces)
        
        # Flux Divergence (Watts/m2)
        dF = F_faces[:-1] - F_faces[1:]
        net_flux = jnp.where(is_charging, dF, -dF)
        
        # Update Energy Density [J/m3]
        # E_vol = rho * h
        rho_old = density_water_kg_m3(T)
        E_vol_old = rho_old * h
        
        # Conservation Update
        E_vol_new = E_vol_old + (net_flux / dz) * dt
        
        # Recover T (Thermodynamic Consistency Loop)
        T_star = newton_raphson_temperature(E_vol_new, T_guess=T)

        # --- 2. Implicit Diffusion Step ---
        
        # Update props based on T_star
        rho_s = density_water_kg_m3(T_star)
        cp_s = specific_heat_water_j_kgk(T_star)
        
        # Detect Buoyancy Instability
        d_rho = rho_s[:-1] - rho_s[1:]
        is_unstable = d_rho > 1e-4
        
        # Dynamic Conductivity (Diffusion vs Mixing)
        k_base = self.config.vertical_conductivity_w_mk
        k_mixing = 1000.0
        k_eff = jnp.where(is_unstable, k_mixing, k_base)
        
        # Matrix Construction
        # Thermal inertia per cell [J/K]
        C_cell = rho_s * cp_s * A_cross * dz
        C_term = C_cell / dt
        
        # Conductance [W/K]
        G_cond = k_eff * A_cross / dz
        
        # Losses
        perim = 2 * jnp.sqrt(jnp.pi * A_cross)
        G_loss = self.config.loss_coeff_w_k * perim * dz
        
        # Assemble Tri-Diagonal System
        G_top = jnp.concatenate([jnp.array([0.0]), G_cond])
        G_bot = jnp.concatenate([G_cond, jnp.array([0.0])])
        
        main = C_term + G_top + G_bot + G_loss
        lower = -G_top
        upper = -G_bot
        rhs = C_term * T_star + G_loss * inputs.T_ambient_c
        
        T_final = solve_tdma_scan(lower, main, upper, rhs)
        
        return T_final
    

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    # Define several configurations for debugging
    configs = [
        ThermalStorageConfig(n_nodes=5, height_m=1.0, volume_m3=0.1),
        ThermalStorageConfig(n_nodes=10, height_m=2.0, volume_m3=0.3),
        ThermalStorageConfig(n_nodes=20, height_m=2.0, volume_m3=0.5),
    ]

    # Define several time steps and flows
    dt_list = [1.0, 10.0, 60.0]  # seconds
    mdot_list = [-0.05, 0.0, 0.1]  # kg/s (discharge, idle, charge)

    # Run test loops
    for cfg in configs:
        print(f"\n--- Config: n_nodes={cfg.n_nodes}, height={cfg.height_m}m, volume={cfg.volume_m3}m3 ---")
        model = ThermalStorageModel(cfg)
        T_init = jnp.full(cfg.n_nodes, 25.0)  # start at 25°C
        
        for dt in dt_list:
            for mdot in mdot_list:
                inputs = StorageInputs(mdot_net_kg_s=mdot, T_inlet_c=60.0)
                try:
                    T_next = model.step(t=0.0, T=T_init, inputs=inputs, dt=dt)
                    # Print basic stats instead of full array
                    print(f"dt={dt}s, mdot={mdot}kg/s -> T_min={T_next.min():.2f}, T_max={T_next.max():.2f}, T_mean={T_next.mean():.2f}")
                except Exception as e:
                    print(f"Error for dt={dt}s, mdot={mdot}kg/s: {e}")
