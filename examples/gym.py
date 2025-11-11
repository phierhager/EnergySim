import jax.numpy as jnp

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    SystemActions, ThermalConfig, BatteryConfig, RewardConfig, 
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, ExogenousData, SystemState
)
import numpy as np
import os
import pandas as pd
from energysim.rl.env import EnergySimEnv

def setup_dummy_data(file_path="/tmp/dummy_data.csv", steps=200, dt_seconds=900):
    """Creates a dummy CSV file for the examples."""
    if os.path.exists(file_path):
        return
        
    print(f"Creating dummy data at {file_path}...")
    time = np.arange(0, steps * dt_seconds, dt_seconds)
    
    # Create sinusoidal data for variety
    sin_wave = np.sin(np.linspace(0, 8 * np.pi, steps)) + 1
    
    df = pd.DataFrame({
        "unixtime": time,
        "load": (sin_wave * 1000) + 1000,  # 1kW to 3kW
        "pv": (sin_wave * 2000),           # 0W to 4kW
        "price": (sin_wave * 0.1) + 0.1,  # 0.1 to 0.3 €/kWh
        "ambient_temp": (sin_wave * 10) + 5 # 5°C to 25°C
    })
    df.to_csv(file_path, index=False)


def my_programmatic_gains(step_idx: int, state: SystemState) -> dict:
    """
    A simple programmatic function to simulate occupants.
    - If the room is cold (below 20°C), occupants are "active" (100W).
    - If the room is warm (>= 20°C), occupants are "inactive" (50W).
    - This creates a dynamic feedback loop.
    """
    
    # We must use JAX operations if we want this to be JIT-compatible
    # (Though for RL, Python ops are fine, but this is good practice)
    gains = jnp.where(state.thermal.room_temp < 20.0, 100.0, 50.0)
    
    # Return a dictionary matching the ExogenousData keys
    return {"internal_gains_w": gains}

def run_programmatic_example():
    """Shows how to run the Gym env with an external programmatic input."""
    
    print("\n--- Running Programmatic Co-Simulation Example ---")
    
    # --- 1. Setup ---
    DATA_FILE = "/tmp/dummy_data.csv"
    DT_SECONDS = 900
    setup_dummy_data(DATA_FILE, steps=100, dt_seconds=DT_SECONDS)
    
    dataset = SimulationDataset(DATA_FILE, dt_seconds=DT_SECONDS)

    # --- 2. Instantiate Simulator ---
    # We don't pass the dataset to the simulator
    simulator = JAXSimulator(
        dt_seconds=DT_SECONDS,
        t_config=ThermalConfig(),
        r_config=RewardConfig(),
        b_config=BatteryConfig(),
        hp_config=HeatPumpConfig(),
        ac_config=None, # No AC
        ts_config=None  # No storage
    )
    
    # --- 3. Instantiate Gym Environment ---
    # We pass the simulator, dataset, AND our new function
    env = EnergySimEnv(
        simulator=simulator,
        dataset=dataset,
        external_fn=my_programmatic_gains  # <--- HERE IS THE HOOK
    )
    
    print(f"Action Space: {env.action_space}")
    print(f"Obs. Space: {env.observation_space}")

    # --- 4. Run the loop ---
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(10):
        action = env.action_space.sample() # Use a random agent
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        print(f"Step {i}: Action={action}, Room Temp={obs[0]:.2f}°C, Reward={reward:.3f}")
        
        if terminated or truncated:
            break
            
    print(f"Total Reward: {total_reward:.3f}")

if __name__ == "__main__":
    run_programmatic_example()