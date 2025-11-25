import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from energysim.core.data.scenario_loader import ScenarioLoader
from energysim.sim.dae_simulator import DAESimulator
from energysim.core.shared.data_structs import (
    ExogenousData, SystemState, DifferentialState, MachineState, SystemActions,
    ThermalConfig, BatteryConfig, HeatPumpConfig, ThermalStorageConfig, MoistureConfig
)
from energysim.core.models.thermal_model import RCNetworkModel
from energysim.core.models.heat_pump_model import RampingHeatPumpModel
from energysim.core.models.battery_model import SimpleBatteryModel
from energysim.core.models.thermal_storage_model import StratifiedThermalStorageModel
from energysim.core.models.moisture_model import EMPDMoistureModel

def create_dummy_sim():
    # Create Minimal Configs
    t_cfg = ThermalConfig(
        A_matrix=jnp.array([[-1.0]]), C_inv_vector=jnp.array([1e-5]), B_matrix=jnp.zeros((1,1)),
        node_names=["Room"], ambient_air_index=0, room_air_indices=(0,),
        mixing_pairs=jnp.zeros((0,2), dtype=int), mixing_conductance=jnp.zeros(0),
        convection_pairs=jnp.zeros((0,2), dtype=int), convection_coefficients=jnp.zeros(0),
        room_vol_m3=50.0, leakage_area_m2=0.01, stack_coeff=0.1, wind_coeff=0.1
    )
    b_cfg = BatteryConfig(capacity_j=3.6e7, efficiency=0.95, max_power_w=5000.0)
    hp_cfg = HeatPumpConfig(
        max_electrical_power_w=3000.0, min_electrical_power_w=0.0, ramp_rate_w_per_sec=100.0,
        cop_heating=3.0, cop_ambient_temps_c=jnp.array([0.0, 20.0]), cop_values_heating=jnp.array([2.5, 4.0])
    )
    ts_cfg = ThermalStorageConfig(
        volume_m3=0.3, height_m=1.5, n_nodes=5, loss_coeff_w_k=2.0, vertical_conductivity_w_mk=0.6
    )
    m_cfg = MoistureConfig(
        air_volume_m3=50.0, buffer_area_m2=100.0, empd_thickness=0.02, 
        material_density=800.0, sorption_slope=0.05, latent_gen_person_kg_s=1e-5
    )

    return DAESimulator(
        thermal=RCNetworkModel(t_cfg),
        hp=RampingHeatPumpModel(hp_cfg, n_rooms=1),
        battery=SimpleBatteryModel(b_cfg),
        storage=StratifiedThermalStorageModel(ts_cfg),
        moisture=EMPDMoistureModel(m_cfg)
    )

def test_pipeline():
    print("--- 1. Generating Dummy Data ---")
    # Mock files for loader
    np.save("dummy_load.npy", np.random.rand(10, 100)) # 10 houses
    with open("dummy.epw", "w") as f: 
        f.write("\n"*8 + "2024,1,1,1,0,?," + "10,5,80,101325,"*10) # Minimal mock
    
    # We use a mock parser for the test to avoid needing real EPW
    # (In real usage, real files are required)
    
    print("--- 2. Simulating 10 Houses ---")
    sim = create_dummy_sim()
    
    # Initialize States
    y0 = DifferentialState(
        T_vector=jnp.array([20.0]),
        storage_T=jnp.full((5,), 45.0),
        battery_soc=jnp.array(0.5),
        hp_power_state=jnp.array(0.0),
        ac_power_state=jnp.array(0.0),
        moisture_w=jnp.array([0.005]),
        moisture_buffer_u=jnp.array([0.05])
    )
    state0 = SystemState(
        diff=y0,
        machines=MachineState(
            energy_remaining=jnp.zeros(10), prev_is_available=jnp.zeros(10)
        )
    )
    
    # Mock Action
    action = SystemActions(
        heat_pump_power_w=jnp.array(1000.0),
        ac_power_w=jnp.array(0.0),
        battery_power_w=jnp.array(100.0),
        storage_discharge_w=jnp.array(0.0),
        smart_appliance_signals=jnp.zeros(10)
    )
    
    # Mock Exo
    exo = jax.device_put(ExogenousData(
        time_of_year_seconds=jnp.array(0.0),
        ambient_temp=jnp.array(5.0),
        solar_dni_w_m2=jnp.array(0.0), solar_dhi_w_m2=jnp.array(0.0),
        wind_speed_m_s=jnp.array(2.0), relative_humidity=jnp.array(0.5),
        atmospheric_pressure=jnp.array(101325.0),
        sky_temp=jnp.array(0.0), ground_temp=jnp.array(10.0),
        price=jnp.array(0.2),
        base_load_w=jnp.array(500.0), # Specific to this house
        occupant_profiles=jnp.zeros(5),
        passive_machine_profiles=jnp.zeros(10),
        smart_device_availability=jnp.zeros(10)
    ))
    
    # Run Single Step
    print("Stepping JAX Simulator...")
    next_state = sim.step(state0, action, exo, dt=900.0)
    
    # Verify Dynamics
    print("Prev T:", state0.diff.T_vector)
    print("Next T:", next_state.diff.T_vector)
    print("Battery SOC:", next_state.diff.battery_soc)
    
    # VMAP Test (10 houses)
    print("--- 3. Testing Batch VMAP ---")
    
    def house_kernel(s, a, e): return sim.step(s, a, e, 900.0)
    
    # Replicate states 10 times
    batch_states = jax.tree_map(lambda x: jnp.stack([x]*10), state0)
    batch_actions = jax.tree_map(lambda x: jnp.stack([x]*10), action)
    # Replicate exo but vary load
    batch_exo = jax.tree_map(lambda x: jnp.stack([x]*10), exo)
    batch_exo = eqx.tree_at(lambda e: e.base_load_w, batch_exo, jnp.linspace(0, 1000, 10))
    
    vmapped_step = jax.vmap(house_kernel)
    batch_next = vmapped_step(batch_states, batch_actions, batch_exo)
    
    print("Batch Results (Temps):", batch_next.diff.T_vector[:, 0])
    assert batch_next.diff.T_vector.shape == (10, 1)
    print("✅ Integration Test Passed")

if __name__ == "__main__":
    test_pipeline()