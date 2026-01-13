    
# Example Usage
import jax
import jax.numpy as jnp
from energysim.grid.simulator import GridSimulator
import equinox as eqx


# create houses
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, SolarConfig, SystemActions
)
import sample_data_generator
from build_my_house import create_2_room_house



def run():
    # 1. Setup Data & Config
    sample_data_generator.create_sample_data(n_days=3)
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, sample_data_generator.DT_SECONDS)
    t_config = create_2_room_house()
    n_rooms = int(len(t_config.room_air_indices))

    build_kwargs = dict(
        dt_seconds=sample_data_generator.DT_SECONDS,
        t_config=t_config,
        r_config=RewardConfig(),
        b_config=BatteryConfig(capacity_kwh=13.0),
        hp_config=HeatPumpConfig(model_type="ramping", max_electrical_power_w=4000.0),
        ac_config=AirConditionerConfig(model_type="ramping", max_electrical_power_w=4000.0),
        ts_config=ThermalStorageConfig(),
        s_config=SolarConfig()
    )
    N_HOUSES = 3
    sims = [JAXSimulator(**build_kwargs) for _ in range(N_HOUSES)]
    batched_sims = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *sims)
    def vmap_step(batched_sim, batched_actions, batched_prev_actions, batched_exo):
        # We vmap over the 'self' (sim), 'actions', 'prev_actions', and 'exo'
        # in_axes=0 means we split all arguments along the first dimension
        step_fn = jax.vmap(lambda s, a, p, e: s.step(a, p, e))
        
        new_sims, costs = step_fn(batched_sim, batched_actions, batched_prev_actions, batched_exo)
        return new_sims, costs
    houses_step = jax.jit(vmap_step)
    
    n_nodes = N_HOUSES + 1  # Including Slack Bus (Node 0)
    lines = [
        (0, 1, 0.1, 100.0),  # from_node, to_node, reactance, limit
        (1, 2, 0.1, 100.0),
        (0, 2, 0.2, 100.0),
        (1, 3, 0.1, 100.0),
        (2, 3, 0.1, 100.0),
    ]

    grid_sim = GridSimulator.create(n_nodes, lines)

    for i in range(10):
        # Prepare batched actions and exogenous data for all houses
        batched_actions = SystemActions(
            battery_power_w=jnp.array([0.0] * N_HOUSES),
            heat_pump_power_w=jnp.array([2000.0] * N_HOUSES),
            ac_power_w=jnp.array([0.0] * N_HOUSES),
            storage_discharge_w=jnp.array([0.0] * N_HOUSES)
        )
        batched_prev_actions = SystemActions(
            battery_power_w=jnp.array([0.0] * N_HOUSES),
            heat_pump_power_w=jnp.array([2000.0] * N_HOUSES),
            ac_power_w=jnp.array([0.0] * N_HOUSES),
            storage_discharge_w=jnp.array([0.0] * N_HOUSES)
        )

        exo_base = dataset[i]
        split_factors = jnp.array([0.6, 0.4]) # Example split for 2 rooms
        exo = eqx.tree_at(
            lambda e: (e.solar_gains_w, e.occupancy_gains_w, e.device_gains_w),
            exo_base,
            (
                exo_base.solar_gains_w * split_factors,      # Scalar -> Vector
                exo_base.occupancy_gains_w * split_factors,  # Scalar -> Vector
                jnp.zeros(n_rooms)                           # Scalar -> Vector
            )
        )
        batched_exo = [exo] * N_HOUSES  # Replicate for each house

        # Step all houses
        batched_sims, costs = houses_step(batched_sims, batched_actions, batched_prev_actions, batched_exo)

        # Calculate grid injections
        injections = jnp.zeros(n_nodes)
        for h in range(N_HOUSES):
            sim = jax.tree_util.tree_map(lambda x: x[h], batched_sims)
            load = sim.heat_pump.current_electrical_w + sim.ac.current_electrical_w
            gen = sim.solar.calculate(batched_exo).electrical_output_w
            injections = injections.at[h + 1].set(gen - load)  # Node indexing

        # Solve power flow
        state = grid_sim.solve_power_flow(injections)

        print(f"Step {i}:")
        print(" Node Angles (radians):", state.theta)
        print(" Line Flows (Watts):", state.line_flows)

if __name__ == "__main__":
    run()