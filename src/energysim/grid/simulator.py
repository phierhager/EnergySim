import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple

class GridState(NamedTuple):
    theta: jnp.ndarray      # Phase angles [radians]
    line_flows: jnp.ndarray # Power flow on lines [Watts]

class GridSimulator(eqx.Module):
    # Parameters (Constant during sim)
    B_bus: jnp.ndarray 
    B_reduced: jnp.ndarray 
    line_incidence: jnp.ndarray 
    line_B: jnp.ndarray
    line_limits: jnp.ndarray 

    @classmethod
    def create(cls, n_nodes, lines):
        """
        lines: list of (from_node, to_node, reactance, limit)
        Node 0 is always the Slack Bus (Reference).
        """
        B_bus = jnp.zeros((n_nodes, n_nodes))
        line_incidence = []
        line_B_vals = []
        line_limits = []

        for u, v, x, limit in lines:
            b = 1.0 / x
            # Build Admittance Matrix
            B_bus = B_bus.at[u, u].add(b).at[v, v].add(b)
            B_bus = B_bus.at[u, v].add(-b).at[v, u].add(-b)
            
            # Build Incidence
            row = jnp.zeros(n_nodes).at[u].set(1).at[v].set(-1)
            line_incidence.append(row)
            line_B_vals.append(b)
            line_limits.append(limit)

        # Pre-compute the inverse of the reduced B matrix for speed
        # We remove row 0 and col 0 (Slack bus)
        B_reduced = B_bus[1:, 1:]
        
        return cls(
            B_bus=B_bus,
            B_reduced=B_reduced,
            line_incidence=jnp.stack(line_incidence),
            line_B=jnp.diag(jnp.array(line_B_vals)),
            line_limits=jnp.array(line_limits)
        )

    def solve_power_flow(self, node_injections):
        """
        node_injections: (N_nodes,) vector. (+ = Gen, - = Load)
        """
        # 1. Separate Slack (idx 0) from others
        P_reduced = node_injections[1:]
        
        # 2. Solve for angles (theta)
        # B_reduced * theta_reduced = P_reduced
        theta_reduced = jnp.linalg.solve(self.B_reduced, P_reduced)
        
        # 3. Add Slack angle (0.0) back
        theta = jnp.concatenate([jnp.array([0.0]), theta_reduced])
        
        # 4. Calc Flows: F = B_line * Incidence * theta
        line_flows = self.line_B @ (self.line_incidence @ theta)
        
        return GridState(theta, line_flows)
