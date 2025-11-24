import jax
import jax.numpy as jnp
import time
from energysim.rl.gymnax_env import EnergyGymnaxEnv, EnvParams
from energysim.sim.simulator import JAXSimulator
from energysim.core.shared.data_structs import (
    ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig, 
    AirConditionerConfig, ThermalStorageConfig, SolarConfig
)
from energysim.core.data.dataset import SimulationDataset
from energysim.sim.helpers import precalculate_exogenous_data

def run_massive_parallel_test():
    # ==========================================
    # 1. Setup the Template (Same as before)
    # ==========================================
    dt = 900.0 
    # ... (Insert your config loading logic here) ...
    # Mocking config for demonstration:
    t_config = ThermalConfig(
        A_matrix=jnp.eye(3)*-0.1, C_inv_vector=jnp.ones(3)*1e-4, B_matrix=jnp.ones((3, 2)),
        node_names=["amb", "room", "wall"], node_map={"amb":0, "room":1, "wall":2}, input_map={},
        u_idx_heating=jnp.array([0]), u_idx_cooling=jnp.array([1]),
        ambient_air_index=0, ground_node_index=-1, room_air_indices=(1,), wall_indices=(2,), mass_indices=()
    )
    
    sim_template = JAXSimulator.create(
        dt_seconds=dt, t_config=t_config, r_config=RewardConfig(), appliances=[],
        dynamic_surfaces=[], solar_surfaces=[], b_config=BatteryConfig(), 
        hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), s_config=SolarConfig()
    )

    # Mock Exo Data (Ensure this is on GPU)
    # In real code: dataset = SimulationDataset(...)
    dataset = SimulationDataset("examples/sample_data.csv", dt)
    exo_trace = precalculate_exogenous_data(
        dataset=dataset, # Mocking pass
        behavioral_models={}, dt_seconds=dt, n_rooms=1, dummy_state=sim_template.state
    ) 
    # Override mock trace with random noise for testing if dataset is None
    T_steps = 10000
    if exo_trace is None:
        print("Generating synthetic exo data...")
        exo_trace = jax.device_put(jax.tree.map(lambda x: jnp.zeros(T_steps), sim_template.solar.calculate(None))) # Mock structure

    # ==========================================
    # 2. Define the Vectorized Functions
    # ==========================================
    
    env = EnergyGymnaxEnv(sim_template, exo_trace)
    env_params = env.default_params
    
    # The Magic: jax.vmap transforms a function that takes (State) -> (Batch of States)
    # in_axes=(0, None) means: Split first arg (RNG) across batch, Duplicate second arg (Params)
    reset_vectorized = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    
    # in_axes=(0, 0, 0, None) means: Split RNG, Split State, Split Action, Duplicate Params
    step_vectorized = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    # ==========================================
    # 3. The "Scan" Loop (Python-Free Stepping)
    # ==========================================
    
    def rollout_loop(carry, _):
        """
        This function compiles into a single CUDA/TPU kernel.
        It steps 'num_envs' environments forward by 1 tick.
        """
        current_state, rng_key = carry
        
        # 1. Split RNG for the step
        rng_key, step_key = jax.random.split(rng_key)
        
        # 2. Split RNGs for the batch (one unique key per env)
        keys_step = jax.random.split(step_key, num_envs)
        
        # 3. Generate Random Actions (or Policy Network output)
        # This happens entirely on accelerator!
        action_key, _ = jax.random.split(step_key)
        action_keys = jax.random.split(action_key, num_envs)
        
        # Sample random actions from the action space (Vectorized sampling)
        # Note: env.action_space().sample is usually not vmapped by default in base Gym,
        # but Gymnax spaces handle keys. We simulate a policy output here:
        actions = jax.random.uniform(
            action_key, 
            shape=(num_envs, env.action_space().shape[0]), 
            minval=-1.0, maxval=1.0
        )
        
        # 4. Step The Environment Batch
        obs, next_state, reward, done, info = step_vectorized(
            keys_step, current_state, actions, env_params
        )
        
        return (next_state, rng_key), (obs, reward, done)

    # ==========================================
    # 4. Execution
    # ==========================================
    
    num_envs = 4096       # Parallel Environments
    rollout_len = 1000    # Steps per rollout
    
    print(f"Preparing JIT compilation for {num_envs} environments x {rollout_len} steps...")
    
    # Initial Reset
    rng = jax.random.PRNGKey(42)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, num_envs)
    
    # Start state (Batch size is implicitly defined here by reset_keys shape)
    start_obs, start_state = reset_vectorized(reset_keys, env_params)
    
    # Run Scan
    # jax.block_until_ready() forces us to wait for the GPU to finish before stopping the timer
    start_time = time.time()
    
    # The Call: Takes (Batch_State) -> Returns (Batch_State, Trace_of_Transitions)
    (final_state, _), (obs_trace, reward_trace, done_trace) = jax.lax.scan(
        rollout_loop, 
        (start_state, rng), 
        None, # iterate over nothing, just count steps
        length=rollout_len
    )
    
    # Force execution
    _ = final_state.time_idx.block_until_ready()
    
    end_time = time.time()
    duration = end_time - start_time
    total_steps = num_envs * rollout_len
    
    print(f"Done!")
    print(f"Total Steps: {total_steps:,}")
    print(f"Duration:    {duration:.4f} s")
    print(f"Throughput:  {total_steps / duration:,.0f} steps/sec")
    
    # Verify Shapes
    print(f"Obs Trace Shape: {obs_trace.shape}") # Should be (Rollout, Num_Envs, Obs_Dim)
    print(f"Reward Mean: {jnp.mean(reward_trace):.4f}")

if __name__ == "__main__":
    run_massive_parallel_test()