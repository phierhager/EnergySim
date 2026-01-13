# Assuming your existing imports from the question exist
from energysim.sim.simulator import JAXSimulator
from energysim.core.shared.data_structs import SystemActions
import jax
import jax.numpy as jnp
import equinox as eqx


def create_batched_houses(n_houses, base_config):
    """
    Creates a simulator where every parameter has an extra dimension (n_houses).
    We use jax.tree_map to replicate the config N times.
    """
    # 1. Replicate the base config N times
    # This stacks scalar values: 10.0 -> [10.0, 10.0, ...]
    batched_config = jax.tree.map(
        lambda x: jnp.stack([x] * n_houses), 
        base_config
    )
    
    # # 2. (Optional) Introduce Heterogeneity
    # # Example: Randomize battery capacity for each house
    # key = jax.random.PRNGKey(42)
    # random_capacities = jax.random.uniform(key, (n_houses,), minval=5.0, maxval=20.0)
    
    # Replace the uniform capacities with random ones
    batched_config = eqx.tree_at(
        lambda c: c.b_config.capacity_kwh, 
        batched_config, 
        random_capacities
    )

    # 3. Create the Simulator (Equinox handles the batched config automatically if designed right,
    # but usually we vmap the *method*, not the class construction. 
    # Ideally, we construct the class with Batched params.)
    
    # NOTE: If your JAXSimulator expects scalars in __init__, we must create it differently.
    # The cleanest JAX/Equinox way is to vmap the STEP function, not the object.
    
    return batched_config

# We define a function that steps ONE house
def single_step(sim_state, config, action, prev_action, exo):
    # Re-instantiate simulator with specific config for this house
    # (Assuming JAXSimulator is stateless and just holds config)
    sim = JAXSimulator(**config.__dict__) # Simplified access
    
    # Or if JAXSimulator is an eqx.Module, we might just call:
    # return sim.step(action, prev_action, exo)
    
    # Use the logic from your provided code:
    new_sim, cost = sim.step(action, prev_action, exo)
    
    # CALCULATE NET LOAD (Grid Injection)
    # Injection = Generation - Load
    # This formula depends on your variable names, assuming standard signs:
    # Load (+), Solar (+), Battery Discharge (+)
    
    # This is a placeholder logic based on standard energy flows
    total_load = jnp.sum(new_sim.state.thermal.device_power) # HeatPump + AC
    battery_flow = action.battery_power_w # + is charge, - is discharge
    solar = exo.solar_gains_w # Approximated as electrical gen for this example
    
    injection = solar - total_load - battery_flow
    
    return new_sim.state, cost, injection

# Vectorize the step function!
# in_axes tells JAX: 
# - state: split along axis 0
# - config: split along axis 0
# - action: split along axis 0
# - prev_action: split along axis 0
# - exo: split along axis 0
batched_step_fn = eqx.filter_vmap(single_step, in_axes=(0, 0, 0, 0, 0))