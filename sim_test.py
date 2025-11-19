import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from datetime import datetime

# --- Import Simulator Components ---
from energysim.sim.simulator_new import JAXSimulator
from energysim.core.network_builder import RCNetworkBuilder
from energysim.core.shared.data_structs import (
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig,
    ApplianceConfig, SystemActions, ExogenousData
)
from energysim.core.physics.coefficients import PhysicsConfig

def run_test():
    print("--- Setting up Simulation ---")
    
    # ==========================================
    # 1. Define Physics & Thermal Network
    # ==========================================
    # Simple 1-Room House with a massive wall
    builder = RCNetworkBuilder(n_rooms=1)
    
    # Nodes (Capacity in J/K)
    builder.add_node("room_air_0", capacity_j_k=1.5e5) # ~50m3 air
    builder.add_node("wall_outer", capacity_j_k=5.0e6) # Heavy brick wall
    builder.add_node("ground", capacity_j_k=jnp.inf) # Infinite ground
    
    # Connections (Resistances in K/W)
    # Room <-> Wall <-> Ambient
    builder.add_resistor("room_air_0", "wall_outer", R_k_w=0.1)
    builder.add_resistor("wall_outer", "ambient", R_k_w=0.5)
    
    # Room <-> Ambient (Ventilation/Windows)
    builder.add_resistor("room_air_0", "ambient", R_k_w=2.0)
    
    # Wall <-> Ground (Foundation loss)
    builder.add_resistor("wall_outer", "ground", R_k_w=5.0)

    # Map HVAC inputs
    builder.add_input_mapping("heating_w", "room_air_0", room_index=0)
    builder.add_input_mapping("cooling_w", "room_air_0", room_index=0)
    
    t_config = builder.compile()

    # ==========================================
    # 2. Define Surface Metadata (Physics)
    # ==========================================
    # These map external physics (Sun/Wind) to the nodes defined above.
    
    # A South-Facing Wall (Azimuth 180) linked to 'wall_outer'
    solar_surfaces = [{
        "type": "WALL",
        "name": "south_wall",
        "normal": [0, -1, 0], # Pointing South
        "area": 20.0,
        "absorptivity": 0.7,
        "target_node": "wall_outer"
    }]

    # Wind effects on the outer wall
    dynamic_surfaces = [{
        "node_name": "wall_outer",
        "area": 20.0,
        "roughness_mult": 1.5, # Concrete
        "boundary": "ambient"
    }]

    # ==========================================
    # 3. Define Appliances (Smart EV)
    # ==========================================
    # An EV that adds 5% of its power as heat to the garage (or room_0 here)
    appliances = [
        ApplianceConfig(
            name="Tesla_Model_3",
            target_node_name="room_air_0",
            nominal_power_w=7000.0,  # 7kW Charger
            convective_fraction=0.05 # 5% heat loss to room
        )
    ]

    # ==========================================
    # 4. Initialize Simulator
    # ==========================================
    dt = 900.0 # 15 minute steps
    
    sim = JAXSimulator.create(
        dt_seconds=dt,
        t_config=t_config,
        r_config=RewardConfig(price_weight=1.0, comfort_weight=10.0),
        appliances=appliances,
        dynamic_surfaces=dynamic_surfaces,
        solar_surfaces=solar_surfaces,
        b_config=BatteryConfig(capacity_kwh=13.5),
        hp_config=HeatPumpConfig(max_electrical_power_w=3000.0),
        ac_config=AirConditionerConfig(),
        ts_config=ThermalStorageConfig(volume_m3=0.5),
        s_config=SolarConfig(panel_area_m2=25.0) # 25m2 Solar PV
    )

    # ==========================================
    # 5. Generate Dummy Data (2 Days)
    # ==========================================
    n_steps = 96 * 2 # 15min * 96 = 24h
    time_indices = jnp.arange(n_steps)
    
    # Time of year (Start at noon on day 180 - Summer)
    start_sec = 180 * 24 * 3600.0 + 12 * 3600.0
    time_seconds = start_sec + (time_indices * dt)
    
    # Weather
    t_ambient = 20.0 + 10.0 * jnp.sin(2 * jnp.pi * time_indices / 96) # 10C to 30C swing
    dni = jnp.maximum(0, 800.0 * jnp.sin(2 * jnp.pi * (time_indices - 24)/ 96)) # Sun pattern
    dhi = dni * 0.1
    
    exo_data = ExogenousData(
        ambient_temp=t_ambient,
        solar_dni_w_m2=dni,
        solar_dhi_w_m2=dhi,
        wind_speed_m_s=jnp.full((n_steps,), 2.0),
        time_of_year_seconds=time_seconds,
        price=jnp.full((n_steps,), 0.20), # 20 cents flat
        base_load_w=jnp.full((n_steps,), 300.0), # 300W baseload
        carbon_intensity=jnp.full((n_steps,), 0.250), # 250g/kWh
        appliance_profiles=jnp.zeros((n_steps, len(appliances))) # No appliances
    )

    # ==========================================
    # 6. Define Actions
    # ==========================================
    # Strategy:
    # - Run Heat Pump at 1000W constantly
    # - Charge Battery at night (steps 0-30)
    # - Plug in EV at step 50 (arrival), charge it
    
    hp_power = jnp.full((n_steps, 1), 1000.0)
    bat_power = jnp.where(time_indices < 30, 3000.0, 0.0) # Charge early
    
    # EV Control: Signal=1.0 (Charge)
    # EV Availability: Arrives at step 50, leaves at step 90
    ev_signal = jnp.full((n_steps, 1), 1.0)
    ev_avail = jnp.where((time_indices > 50) & (time_indices < 90), 1.0, 0.0)
    ev_avail = ev_avail.reshape(-1, 1) # Shape (T, 1) for 1 appliance

    actions_seq = SystemActions(
        heat_pump_power_w=hp_power,
        ac_power_w=jnp.zeros((n_steps, 1)),
        battery_power_w=bat_power,
        storage_discharge_w=jnp.full((n_steps, 1), 500.0), # Discharge tank slowly
        appliance_signals=ev_signal
    )

    # ==========================================
    # 7. Run Simulation (Scan)
    # ==========================================
    print("--- Running Simulation Loop ---")
    
    def scan_fn(sim, inputs):
        act, prev_act, ex, av = inputs
        new_sim, cost = sim.step(act, prev_act, ex, av)
        
        # Capture data for plotting
        metrics = {
            "room_temp": new_sim.thermal.T_vector[0],
            "wall_temp": new_sim.thermal.T_vector[1], # Wall outer
            "battery_soc": new_sim.battery.soc,
            "tank_temp": jnp.mean(new_sim.storage.temperatures_c),
            "ev_energy_rem": new_sim.appliances[0].state.energy_remaining_j,
            "cost": cost
        }
        return new_sim, metrics

    # Dummy previous action (zeros)
    prev_action_dummy = jax.tree.map(lambda x: jnp.zeros_like(x[0]), actions_seq)
    
    # Need to shift actions to create (current, previous) pairs
    # For simplicity in this test, we just pass current as previous (no slew cost check)
    
    # Stack inputs for scan
    scan_inputs = (actions_seq, actions_seq, exo_data, ev_avail)
    
    start_time = datetime.now()
    
    # JIT compile and run
    final_sim, history = jax.lax.scan(scan_fn, sim, scan_inputs)
    
    # Block until done to measure time
    jax.block_until_ready(history["cost"])
    print(f"--- Done in {(datetime.now() - start_time).total_seconds():.4f}s ---")

    # ==========================================
    # 8. Visualisation
    # ==========================================
    print("--- Plotting Results ---")
    
    # Convert to numpy
    hist = jax.tree.map(np.array, history)
    t_axis = np.arange(n_steps) * dt / 3600.0 # Hours

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Temperatures
    axs[0].plot(t_axis, hist["room_temp"], label="Room Air", linewidth=2)
    axs[0].plot(t_axis, hist["wall_temp"], label="Wall Outer")
    axs[0].plot(t_axis, hist["tank_temp"], label="Thermal Storage", linestyle="--")
    axs[0].plot(t_axis, np.array(exo_data.ambient_temp), label="Ambient", linestyle=":", alpha=0.6)
    axs[0].set_ylabel("Temperature (°C)")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Thermal Dynamics")
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Battery & EV
    axs[1].plot(t_axis, hist["battery_soc"], label="Home Battery SOC", color="green")
    
    # Normalize EV energy for plotting (Max ~ 40kWh = 144e6 J)
    ev_max_j = 40.0 * 3.6e6
    axs[1].plot(t_axis, hist["ev_energy_rem"] / ev_max_j, label="EV Energy Need (Norm)", color="red")
    
    axs[1].fill_between(t_axis, 0, 1, where=(np.array(ev_avail[:,0]) > 0.5), color="red", alpha=0.1, label="EV Plugged In")
    axs[1].set_ylabel("State of Charge (0-1)")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Storage & Smart Appliances")
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Costs
    axs[2].plot(t_axis, np.cumsum(hist["cost"]), label="Cumulative Cost (€)", color="black")
    axs[2].set_ylabel("Euros")
    axs[2].set_xlabel("Time (Hours)")
    axs[2].set_title("Economic Performance")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_test_output.png")

if __name__ == "__main__":
    run_test()