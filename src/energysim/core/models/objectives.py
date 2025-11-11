# energysim/core/models/objectives.py
import jax.numpy as jnp
from jax import jit
from functools import partial
from ..shared.data_structs import (
    SystemState, SystemActions, ExogenousData,
    HeatPumpOutput, AirConditionerOutput, ThermalStorageOutput,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig,
    Array
)

@partial(jit, static_argnames=["configs", "dt_seconds"])
def f_cost_step(
    state: SystemState,
    actions: SystemActions,
    exogenous: ExogenousData,
    hp_output: HeatPumpOutput,
    ac_output: AirConditionerOutput,
    storage_output: ThermalStorageOutput,
    configs: tuple[ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig],
    dt_seconds: float
) -> Array:
    """
    Calculates the total cost for a single timestep.
    """
    t_conf, b_conf, r_conf, hp_conf, ac_conf, ts_conf = configs
    
    # --- 1. Calculate Electrical Cost ---
    hp_electrical_power_w = hp_output.electrical_power_w
    ac_electrical_power_w = ac_output.electrical_power_w
    
    net_grid_power_w = (
        exogenous.load 
        + actions.battery_power_w  # <--- This is CORRECT
        + hp_electrical_power_w
        + ac_electrical_power_w
        - exogenous.pv
    )
    
    net_grid_energy_kwh = (net_grid_power_w * (dt_seconds / 3600.0)) / 1000.0
    cost_euros = jnp.fmax(0.0, net_grid_energy_kwh) * exogenous.price
    
    # --- 2. Calculate Comfort Cost ---
    temp_error = state.thermal.room_temp - t_conf.setpoint
    comfort_violation = jnp.fmax(0.0, jnp.abs(temp_error) - t_conf.comfort_band)
    comfort_penalty = comfort_violation**2
    
    # --- 3. Calculate Waste Penalty ---
    rejected_heat_kwh = (storage_output.rejected_heat_w * (dt_seconds / 3600.0)) / 1000.0
    waste_penalty = rejected_heat_kwh * exogenous.price 
    
    # --- 4. Total Weighted Cost ---
    total_cost = (
        (cost_euros * r_conf.price_weight) +
        (comfort_penalty * r_conf.comfort_weight) +
        (waste_penalty * r_conf.price_weight)
    )
    
    return total_cost