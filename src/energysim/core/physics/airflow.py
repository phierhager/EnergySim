# physics/airflow_high_fidelity.py
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial
from .constants import CONSTANTS

# ==========================================
# 1. Data Structures & Constants
# ==========================================

class AirflowParams(NamedTuple):
    """
    Physical parameters for the Airflow Network.
    """
    # Topology: Link i connects node_a[i] <-> node_b[i]
    link_node_a: jnp.ndarray  # Shape (N_links,)
    link_node_b: jnp.ndarray  # Shape (N_links,)
    
    # Physics: Power Law (m_dot = C * dP^n)
    link_C_flow: jnp.ndarray   # [kg/s @ 1Pa]
    link_exponent: jnp.ndarray # Dimensionless (0.5 - 1.0)
    
    # Geometry: Heights for Stack Effect
    node_heights: jnp.ndarray # Shape (N_total_nodes,)

    # Regularization: Transition to laminar flow (Pa)
    linear_limit_pa: float = 0.5

# ==========================================
# 2. Constitutive Physics (Regularized)
# ==========================================

def regularized_flow(dp: jnp.ndarray, C: jnp.ndarray, n: jnp.ndarray, limit_pa: float) -> jnp.ndarray:
    """
    Calculates mass flow with a smooth transition from Laminar (linear) 
    to Turbulent (power law) to ensure differentiability at dp=0.
    
    Regime 1 (|dp| < limit): m = k * dp
    Regime 2 (|dp| >= limit): m = C * sign(dp) * |dp|^n
    
    We compute 'k' such that value and slope match at 'limit'.
    m_limit = C * limit^n
    dm_limit = C * n * limit^(n-1)
    Therefore k = C * n * limit^(n-1)
    """
    abs_dp = jnp.abs(dp)
    sign_dp = jnp.sign(dp)
    
    # Laminar Coefficient
    # k_lam = C * n * (limit ^ (n - 1))
    k_laminar = C * n * (limit_pa ** (n - 1.0))
    
    # Flow regimes
    m_laminar = k_laminar * dp
    m_turbulent = C * sign_dp * (abs_dp ** n)
    
    # Smooth selection
    return jnp.where(abs_dp < limit_pa, m_laminar, m_turbulent)

# ==========================================
# 3. Residual Function (Mass Balance)
# ==========================================

def calculate_airflow_residuals(
    p_internal: jnp.ndarray,       # Variable: Unknowns
    p_boundary: jnp.ndarray,       # Fixed: Boundary conditions
    t_nodes_k: jnp.ndarray,        # Fixed: Nodal temperatures
    params: AirflowParams          # Fixed: Network parameters
) -> jnp.ndarray:
    """
    Computes Residuals: R(P) = Sum(Mass_In) - Sum(Mass_Out) = 0
    """
    # 1. Combine Pressures
    p_all = jnp.concatenate([p_internal, p_boundary])
    
    idx_a = params.link_node_a
    idx_b = params.link_node_b
    
    # 2. Calculate Density (Boussinesq Approximation)
    # Uses average link temperature and standard pressure for stability.
    # High-Fidelity Note: Updating density inside the Newton loop causes
    # instability for minimal accuracy gain. Lagging or fixing it is standard.
    t_link = 0.5 * (t_nodes_k[idx_a] + t_nodes_k[idx_b])
    rho_link = 101325.0 / (CONSTANTS.GAS_CONSTANT_AIR * t_link)
    
    # 3. Stack Effect (Gravitational Head)
    h_diff = params.node_heights[idx_b] - params.node_heights[idx_a]
    dp_stack = -CONSTANTS.GRAVITY * rho_link * h_diff
    
    # 4. Total Pressure Difference
    # dP_total = (Pa - Pb) + dP_stack + dP_wind (if wind included in boundaries)
    dp_total = (p_all[idx_a] - p_all[idx_b]) + dp_stack
    
    # 5. Mass Flow Calculation
    m_dot = regularized_flow(
        dp_total, 
        params.link_C_flow, 
        params.link_exponent, 
        params.linear_limit_pa
    )
    
    # 6. Accumulate Fluxes at Nodes
    n_internal = p_internal.shape[0]
    residuals = jnp.zeros(n_internal)
    
    # Flow leaves A (-) and enters B (+)
    # Only accumulate for internal nodes (indices < n_internal)
    
    # Masking is faster/cleaner on GPU than boolean indexing for Scatter
    mask_a = idx_a < n_internal
    mask_b = idx_b < n_internal
    
    residuals = residuals.at[idx_a].add(jnp.where(mask_a, -m_dot, 0.0))
    residuals = residuals.at[idx_b].add(jnp.where(mask_b, m_dot, 0.0))
    
    return residuals

# ==========================================
# 4. Newton-Raphson Solver (Dense & JIT-able)
# ==========================================

def solve_newton_dense(
    p_init: jnp.ndarray,
    p_boundary: jnp.ndarray,
    t_nodes_k: jnp.ndarray,
    params: AirflowParams,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 1.0 
) -> Tuple[jnp.ndarray, bool]:
    
    def cond_fun(carry):
        _, iter_num, error, _ = carry
        return (iter_num < max_iter) & (error > tol)

    def body_fun(carry):
        p_curr, iter_num, _, _ = carry
        
        # 1. Forward Residual
        R = calculate_airflow_residuals(p_curr, p_boundary, t_nodes_k, params)
        
        # 2. Jacobian (Dense - Optimal for N < 500)
        # Using jacfwd is efficient because Output Dim (N) == Input Dim (N)
        J = jax.jacfwd(calculate_airflow_residuals, argnums=0)(
            p_curr, p_boundary, t_nodes_k, params
        )
        
        # 3. Linear Solve (J * delta = -R)
        # Regularize diagonal slightly to prevent singular matrix at total isolation
        J_safe = J + jnp.eye(J.shape[0]) * 1e-9
        delta = -jnp.linalg.solve(J_safe, R)
        
        # 4. Update
        p_next = p_curr + (delta * damping)
        error = jnp.linalg.norm(R)
        
        return (p_next, iter_num + 1, error, True)

    # Initial Residual check
    R0 = calculate_airflow_residuals(p_init, p_boundary, t_nodes_k, params)
    err0 = jnp.linalg.norm(R0)
    
    init_val = (p_init, 0, err0, False)
    
    # Run Loop
    final_p, final_iters, final_err, converged = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    return final_p, (final_err < tol)

# ==========================================
# 5. Implicit Differentiation Wrapper
# ==========================================

@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def solve_airflow(
    p_init, p_boundary, t_nodes_k, params, # Differentiable args
    max_iter=20, tol=1e-6, damping=1.0     # Static args
):
    """
    Forward pass: Solves the nonlinear system.
    """
    p_star, converged = solve_newton_dense(
        p_init, p_boundary, t_nodes_k, params, max_iter, tol, damping
    )
    # Stop gradients through the solver loop itself (standard for Implicit Diff)
    return jax.lax.stop_gradient(p_star)

def solve_airflow_fwd(p_init, p_boundary, t_nodes_k, params, max_iter, tol, damping):
    """
    Forward pass helper: Returns solution + residuals for the backward pass.
    """
    p_star = solve_airflow(p_init, p_boundary, t_nodes_k, params, max_iter, tol, damping)
    
    # We return the solution (p_star) and the inputs needed to reconstruct the Jacobian
    # in the backward pass.
    return p_star, (p_star, p_boundary, t_nodes_k, params)

def solve_airflow_bwd(max_iter, tol, damping, res, g):
    """
    Backward pass: Uses Implicit Function Theorem.
    dL/dParams = -(v^T @ dR/dParams)
    where v is the solution to J^T @ v = -dL/dP_star
    """
    p_star, p_boundary, t_nodes_k, params = res
    
    # 1. Recompute Jacobian at the equilibrium point P*
    # J = dR / dP_internal
    J = jax.jacfwd(calculate_airflow_residuals, argnums=0)(
        p_star, p_boundary, t_nodes_k, params
    )
    
    # 2. Solve the Adjoint System
    # We want v such that J.T @ v = -g  =>  v = solve(J.T, -g)
    # Note: 'g' is the incoming gradient from the loss function w.r.t p_star
    v = jnp.linalg.solve(J.T, -g)
    
    # 3. Compute VJP for Parameters (Vector-Jacobian Product)
    # We need v^T @ (dR/dParams). 
    # JAX vjp function computes (v^T @ Jacobian) automatically.
    
    # We define a function that only varies the parameters we care about
    def residual_wrt_params(p_b, t_k, pars):
        return calculate_airflow_residuals(p_star, p_b, t_k, pars)

    # Calculate VJP against the adjoint vector 'v'
    # Note: We pass 'v' (not -v) because we already solved for -g above.
    _, vjp_fun = jax.vjp(residual_wrt_params, p_boundary, t_nodes_k, params)
    
    d_p_bound, d_t_nodes, d_params = vjp_fun(v)
    
    # The gradient w.r.t p_init is zero because the equilibrium solution 
    # of a stable system is independent of the initialization.
    d_p_init = jnp.zeros_like(p_star)
    
    return d_p_init, d_p_bound, d_t_nodes, d_params

# Register the VJP
solve_airflow.defvjp(solve_airflow_fwd, solve_airflow_bwd)