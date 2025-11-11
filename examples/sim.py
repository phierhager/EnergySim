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

# 1. Setup the simulator (e.g., no battery, no storage)
setup_dummy_data()
dataset = SimulationDataset(
    file_path="/tmp/dummy_data.csv",
    dt_seconds=900
)
simulator = JAXSimulator(
    dt_seconds=dataset.dt_seconds,
    t_config=ThermalConfig(),
    r_config=RewardConfig(),
    hp_config=HeatPumpConfig(),
    ac_config=AirConditionerConfig(),
    ts_config=None, # Passthrough
    b_config=None   # No battery
)

# 2. Run a simple Python loop
state = simulator.reset()
total_cost = 0.0

for i in range(len(dataset)):
    exo_data = dataset[i]
    
    # --- Your Rule-Based Controller ---
    room_temp = state.thermal.room_temp
    
    # 1. Decide on heating/cooling
    if room_temp < 20.0:  # Too cold
        heat_power_w = 500.0  # Call for heat
        ac_power_w = 0.0
    elif room_temp > 23.0: # Too hot
        heat_power_w = 0.0
        ac_power_w = 500.0  # Call for cooling
    else:
        heat_power_w = 0.0
        ac_power_w = 0.0
    
    # (Actions for dummy components are ignored)
    actions = SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.array(heat_power_w),
        ac_power_w=jnp.array(ac_power_w),
        storage_discharge_w=jnp.array(0.0) # Ignored by passthrough model
    )
    
    # 3. Step the core simulator
    state, cost = simulator.step(actions, exo_data)
    total_cost += cost

    print(f"Step {i}: Room Temp={state.thermal.room_temp:.2f}°C, Cost=€{cost:.2f}")

print(f"Total cost with RBC: €{total_cost:.2f}")