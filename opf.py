import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.example_libraries.optimizers as optimizers
import jax
import jax.numpy as jnp
import jaxopt

# 1. PREPARE THE GRID PHYSICS
def get_ybus(n_nodes, senders, receivers, R, X):
    # Same helper logic as before: Adjacency -> Complex Admittance
    z = R + 1j * X
    y = 1 / z
    Y = jnp.zeros((n_nodes, n_nodes), dtype=jnp.complex64)
    Y = Y.at[senders, receivers].add(-y)
    Y = Y.at[receivers, senders].add(-y)
    diagonal = jnp.zeros(n_nodes, dtype=jnp.complex64)
    diagonal = diagonal.at[senders].add(y)
    diagonal = diagonal.at[receivers].add(y)
    idx = jnp.arange(n_nodes)
    Y = Y.at[idx, idx].add(diagonal)
    return Y

# 3 Node System Data
n_nodes = 3
Y_bus = get_ybus(n_nodes, jnp.array([0,1]), jnp.array([1,2]), jnp.array([0.01, 0.01]), jnp.array([0.1, 0.1]))


# 2. DEFINE THE PROBLEM
def opf_objective(variables, params):
    """
    Minimize Cost + Ramping Penalty
    variables: Dict containing (T, N) arrays for Voltage and Generation
    params: Static data (loads, prices, etc.)
    """
    p_gen = variables['p_gen']
    
    # A. Economic Cost: Quadratic cost curve (a*P^2)
    # Sum over Time (T) and Nodes (N)
    cost = jnp.sum(10.0 * p_gen**2)
    
    # B. Differential "Soft" Constraint (Ramping)
    # If you want ramping to be a hard constraint, move it to the constraint function.
    # Here, we penalize sharp changes between time steps.
    # p_gen shape is (Time, Nodes)
    diffs = jnp.diff(p_gen, axis=0) # P[t+1] - P[t]
    ramping_cost = jnp.sum(diffs**2) * 1000.0
    
    return cost + ramping_cost

def opf_constraints(variables, params):
    """
    All these values must equal ZERO for the solution to be valid.
    """
    # Unpack
    v_mag = variables['v_mag']
    v_ang = variables['v_ang']
    p_gen = variables['p_gen']
    
    # Loads (Time, Nodes)
    p_load = params['p_load']
    q_load = params['q_load']
    
    # --- 1. Power Flow Physics (Vectorized over Time) ---
    # We use vmap to apply the static power flow eq to every time step at once
    def single_step_mismatch(vm, va, pg, pl, ql):
        V = vm * jnp.exp(1j * va)
        I = jnp.dot(Y_bus, V)
        S_calc = V * jnp.conj(I)
        p_mismatch = jnp.real(S_calc) - (pg - pl)
        q_mismatch = jnp.imag(S_calc) - (0.0 - ql) # Assume 0 Q gen
        return jnp.concatenate([p_mismatch, q_mismatch])

    # Apply across T dimension (axis 0)
    # Result shape: (T, 2*N)
    mismatches = jax.vmap(single_step_mismatch)(v_mag, v_ang, p_gen, p_load, q_load)
    
    # Flatten everything into a single 1D array of equality constraints
    return mismatches.flatten()

# 3. SOLVE
def solve_grid(p_loads, q_loads):
    T, N = p_loads.shape
    
    # Initialize variables (Flat start)
    init_vars = {
        'v_mag': jnp.ones((T, N)),
        'v_ang': jnp.zeros((T, N)),
        'p_gen': p_loads.copy() # Start assuming Gen matches Load
    }
    
    # Use Augmented Lagrangian
    # This solver wraps an inner solver (L-BFGS-B) and handles the constraints
    solver = jaxopt.AugmentedLagrangian(
        fun=opf_objective,
        constr_eq=opf_constraints
    )
    
    params = {'p_load': p_loads, 'q_load': q_loads}
    
    # Run the solver
    state = solver.run(init_params=init_vars, params=params)
    return state.params

# Example Run (24 time steps, 3 nodes)
T = 24
dummy_loads_p = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (T, n_nodes)))
dummy_loads_q = dummy_loads_p * 0.1

optimal_solution = solve_grid(dummy_loads_p, dummy_loads_q)
print("Optimization Done.")
print(f"Gen at Node 0 (First 5 hours): {optimal_solution['p_gen'][:5, 0]}")