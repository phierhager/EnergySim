import jax.numpy as jnp
from jax.tree_util import tree_map

from energysim.control.mpc_solver import JAX_MPC_Solver
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    SystemState, ThermalState, BatteryState, ThermalStorageState,
    ThermalConfig, BatteryConfig, RewardConfig, 
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, HeatPumpState, AirConditionerState
)
import numpy as np
import pandas as pd
import os

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

def run_mpc_example():
    """Shows how to run the JAX_MPC_Solver."""
    
    print("\n--- Running MPC Solver Example ---")
    
    # --- 1. Setup ---
    DATA_FILE = "/tmp/dummy_data.csv"
    DT_SECONDS = 900
    HORIZON_N = 10 # Look ahead 10 steps (2.5 hours)
    setup_dummy_data(DATA_FILE, steps=100, dt_seconds=DT_SECONDS)
    
    dataset = SimulationDataset(DATA_FILE, dt_seconds=DT_SECONDS)

    # --- 2. Instantiate All Configs ---
    # MPC needs the *full* set of configs, even for "dummy" components
    # The factory inside the solver will handle `None`
    t_config=ThermalConfig()
    r_config=RewardConfig()
    b_config=BatteryConfig()
    hp_config=HeatPumpConfig()
    ac_config=AirConditionerConfig()
    ts_config=ThermalStorageConfig()

    # --- 3. Instantiate the MPC Solver ---
    # This JIT-compiles the entire simulation horizon!
    print("JIT-compiling MPC solver...")
    solver = JAX_MPC_Solver(
        N_horizon=HORIZON_N,
        dt_seconds=DT_SECONDS,
        t_config=t_config,
        r_config=r_config,
        b_config=b_config,
        hp_config=hp_config,
        ac_config=ac_config,
        ts_config=ts_config
    )
    print("MPC solver compiled.")

    # --- 4. Get Current State and Forecast ---
    
    # The MPC solver needs the *full* PyTree state
    initial_state = SystemState(
        thermal=ThermalState(room_temp=jnp.array(19.0)), # Room is cold
        battery=BatteryState(soc=jnp.array(0.2), soh=jnp.array(1.0)),       # Battery is low
        storage=ThermalStorageState(soc=jnp.array(0.1)), # Storage is low
        heat_pump=HeatPumpState(current_electrical_w=jnp.array(0.0)),
        air_conditioner=AirConditionerState(current_electrical_w=jnp.array(0.0))
    )
    
    # Get the forecast for the next N steps
    forecast = dataset.get_forecast(start_idx=0, horizon=HORIZON_N)
    
    print(f"\nSolving MPC for N={HORIZON_N} steps...")
    print(f"Forecasted Price (€/kWh): {forecast.price}")
    
    # --- 5. Solve ---
    first_action = solver.solve(initial_state, forecast)
    
    print("\n--- MPC Result (Action for Step 0) ---")
    print(f"  Battery Power: {first_action.battery_power_w:.2f} W")
    print(f"  Heat Pump Power: {first_action.heat_pump_power_w:.2f} W")
    print(f"  AC Power: {first_action.ac_power_w:.2f} W")
    print(f"  Storage Discharge: {first_action.storage_discharge_w:.2f} W")
    
    # Example: A simple house with no storage
    print("\n--- MPC for Simple House (No Storage) ---")
    
    solver_simple = JAX_MPC_Solver(
        N_horizon=HORIZON_N,
        dt_seconds=DT_SECONDS,
        t_config=t_config,
        r_config=r_config,
        b_config=b_config,
        hp_config=hp_config,
        ac_config=ac_config,
        ts_config=None  # <-- NO STORAGE
    )
    
    # The `solve` function signature is IDENTICAL
    first_action_simple = solver_simple.solve(initial_state, forecast)
    
    print("--- MPC Result (Simple House) ---")
    print(f"  Battery Power: {first_action_simple.battery_power_w:.2f} W")
    print(f"  Heat Pump Power: {first_action_simple.heat_pump_power_w:.2f} W")
    print(f"  AC Power: {first_action_simple.ac_power_w:.2f} W")
    print(f"  Storage Discharge: {first_action_simple.storage_discharge_w:.2f} W")
    print("(Note: HP power will go to the room, Storage discharge will be 0)")


if __name__ == "__main__":
    run_mpc_example()