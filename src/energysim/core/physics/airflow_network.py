import jax
import jax.numpy as jnp
import jaxopt
from typing import NamedTuple

class AirflowParams(NamedTuple):
    # Topology: adjacency matrix or list of links
    link_node_a: jnp.ndarray # Indices
    link_node_b: jnp.ndarray
    link_C_flow: jnp.ndarray # Orifice coefficient (m3/s at 1Pa)
    link_exponent: float = 0.65 # 0.5 for turbulent, 1.0 for laminar
    
    # Boundary Conditions
    P_wind: jnp.ndarray      # Wind pressure at external nodes (Pa)
    T_nodes: jnp.ndarray     # Temperatures at all nodes (K)
    node_heights: jnp.ndarray # Elevation (m) for stack effect

@jax.jit
def pressure_residuals(P_internal, params: AirflowParams):
    """
    Returns the mass balance residual for each internal node.
    Sum(m_dot_in) - Sum(m_dot_out) = 0
    """
    # 1. Construct full pressure vector (Internal + Boundary P_wind)
    # Assuming nodes 0..N are internal, N..M are boundary
    # For simplicity, let's assume P_nodes includes ALL nodes, 
    # and we minimize residuals only for internal ones.
    
    # Physics constants
    g = 9.81
    rho_ref = 1.204
    T_ref = 293.15
    
    # 2. Calculate Stack Pressure Difference per Link
    # dP_stack = -rho * g * (h_b - h_a) * (T_link - T_ref)/T_ref
    # (simplified Boussinesq approximation)
    idx_a = params.link_node_a
    idx_b = params.link_node_b
    
    h_diff = params.node_heights[idx_b] - params.node_heights[idx_a]
    dP_stack = -rho_ref * g * h_diff * ((params.T_nodes[idx_a] - T_ref)/T_ref)
    
    # 3. Total Pressure Difference
    P_a = P_internal[idx_a] # Note: Needs careful indexing for boundary nodes
    P_b = P_internal[idx_b]
    
    delta_P = (P_a - P_b) + dP_stack
    
    # 4. Mass Flow (Signed)
    # m_dot = C * sgn(dP) * |dP|^n
    sign = jnp.sign(delta_P)
    m_dot = params.link_C_flow * sign * (jnp.abs(delta_P) ** params.link_exponent)
    
    # 5. Accumulate Net Flow at Nodes
    residuals = jnp.zeros_like(P_internal)
    residuals = residuals.at[idx_a].add(-m_dot) # Leaving A
    residuals = residuals.at[idx_b].add(m_dot)  # Entering B
    
    # Mask out boundary nodes (residuals there don't matter, they are infinite sources)
    # (Implementation detail: assume last K nodes are boundary and return residuals[:N])
    return residuals

class AirflowNetworkSolver:
    def __init__(self):
        # Levenberg-Marquardt is robust for non-linear least squares
        self.solver = jaxopt.LevenbergMarquardt(residual_fun=pressure_residuals)

    @jax.jit
    def solve(self, init_P, params: AirflowParams):
        sol = self.solver.run(init_P, params=params)
        return sol.params