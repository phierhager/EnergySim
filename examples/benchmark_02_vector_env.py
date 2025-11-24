import time
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx  # <--- Added import
from energysim.sim.simulator import JAXSimulator
from energysim.rl.gymnax_env import VectorizedEnergyEnv
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, SolarConfig
)
from energysim.behavior import SimpleEVModel
import sample_data_generator
from build_my_house import create_2_room_house

def benchmark_vector():
    print("--- Benchmark: Vectorized RL Environment ---")
    
    # Settings
    NUM_ENVS = 4096  # Simulate 4096 houses in parallel
    STEPS = 1000     # Steps per house
    
    # 1. Setup
    sample_data_generator.create_sample_data(n_days=30)
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, 900)
    t_config = create_2_room_house()
    
    sim = JAXSimulator(
        900, t_config, RewardConfig(), BatteryConfig(), HeatPumpConfig(),
        AirConditionerConfig(), ThermalStorageConfig(), SolarConfig()
    )
    
    # 2. Init Vector Env
    # This handles behavioral logic and data broadcasting automatically
    vec_env = VectorizedEnergyEnv(
        simulator=sim,
        dataset=dataset,
        num_envs=NUM_ENVS,
        behavioral_models={"ev": SimpleEVModel()}
    )
    
    # 3. Define Random Policy
    @partial(jax.jit, static_argnames=("n_envs",))
    def random_policy(key, n_envs):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Generate random values
        new_bat = jax.random.uniform(k1, (n_envs,), minval=-3000, maxval=3000)
        new_hp = jax.random.uniform(k2, (n_envs, 2), minval=0, maxval=2000)
        new_ac = jax.random.uniform(k3, (n_envs, 2), minval=0, maxval=2000)

        # Broadcast the default/dummy action struct to (n_envs, ...)
        # This ensures fields you DON'T update (like storage_discharge_w) 
        # have the correct leading dimension (4096).
        batched_dummy = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_envs,) + x.shape),
            vec_env._dummy_action_struct
        )
        
        # Apply updates to the ALREADY BATCHED structure
        return eqx.tree_at(
            lambda a: (a.battery_power_w, a.heat_pump_power_w, a.ac_power_w),
            batched_dummy, 
            (new_bat, new_hp, new_ac)
        )

    # 4. Rollout Loop
    @jax.jit
    def run_rollout(start_state, key):
        def scan_body(carry, _):
            state, k = carry
            k, pol_k = jax.random.split(k)
            actions = random_policy(pol_k, NUM_ENVS)
            
            # VectorEnv.step handles vmap internally
            next_state, reward, done = vec_env.step(state, actions)
            return (next_state, k), reward

        (final_state, _), rewards = jax.lax.scan(scan_body, (start_state, key), None, length=STEPS)
        return rewards

    # 5. Run
    key = jax.random.PRNGKey(0)
    state = vec_env.reset(key)
    
    print(f"Compiling for {NUM_ENVS} parallel environments...")
    start = time.perf_counter()
    run_rollout(state, key).block_until_ready()
    print(f"Compilation done: {time.perf_counter()-start:.2f}s")
    
    print(f"Simulating...")
    start = time.perf_counter()
    run_rollout(state, key).block_until_ready()
    duration = time.perf_counter() - start
    
    total_steps = NUM_ENVS * STEPS
    print(f"\nRESULTS:")
    print(f"Total Transitions: {total_steps:,}")
    print(f"Time: {duration:.4f}s")
    print(f"Throughput: {total_steps/duration:,.0f} steps/second")

if __name__ == "__main__":
    benchmark_vector()