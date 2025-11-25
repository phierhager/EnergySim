import jax.numpy as jnp
import jax.nn as jnn
from jax import jit
from functools import partial
from .shared.data_structs import (
    SystemState, SystemActions, ExogenousData,
    HeatPumpOutput, AirConditionerOutput, ThermalStorageOutput, SolarOutput,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig,
    Array
)

# --- Helper: Soft Deadband Penalty ---
def _calculate_deadband_penalty(power_w: Array, min_power_w: float, weight: float = 1.0) -> float:
    """
    Creates a 'hump' cost in the forbidden zone (0 < p < min_power).
    This acts as a soft constraint to force the optimizer to choose either 0 or >= min_power.
    """
    # If min_power is 0, there is no deadband.
    if min_power_w <= 1.0:
        return 0.0
        
    # Center the penalty hump at half the min power
    mu = min_power_w / 2.0
    # Width of the hump
    sigma = min_power_w / 4.0
    
    # 1. Gaussian hump centered at mu
    # exp(-((x - mu)^2) / (2 * sigma^2))
    gauss = jnp.exp(-((power_w - mu)**2) / (2 * sigma**2))
    
    # 2. Masking:
    # We want the penalty to fade out completely as we approach 0 (to allow turning off)
    # and as we pass min_power (to allow running).
    # The Gaussian naturally fades, but we ensure 0.0 is perfectly safe by multiplying by sigmoid(power).
    # We assume anything < 1 Watt is effectively "off".
    is_active_soft = jnn.sigmoid(power_w - 1.0) 
    
    # Sum penalty across all zones/devices
    total_penalty = jnp.sum(gauss * is_active_soft)
    
    return total_penalty * weight


# --- Helper: Slew Rate Penalty ---
def _calculate_slew_penalty(current_w: Array, prev_w: Array, weight: float = 0.001) -> float:
    """
    Penalizes rapid changes in control actions to prevent oscillation (bang-bang behavior).
    Cost = weight * (u_t - u_{t-1})^2
    """
    delta = current_w - prev_w
    return jnp.sum(delta**2) * weight


@partial(jit, static_argnames=["dt_seconds"])
def f_cost_step(
    state: SystemState,
    actions: SystemActions,
    prev_actions: SystemActions,
    exogenous: ExogenousData,
    hp_output: HeatPumpOutput,
    ac_output: AirConditionerOutput,
    storage_output: ThermalStorageOutput,
    solar_output: SolarOutput,
    configs: tuple[ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig],
    smart_appliance_load_w: float,
    dt_seconds: float
) -> Array:
    """
    Calculates the total cost for a single timestep.
    """
    t_conf, b_conf, r_conf, hp_conf, ac_conf, ts_conf, s_conf = configs

    # ==========================================
    # 1. Economic Cost (Electricity Bill)
    # ==========================================

    hp_electrical_power_w = jnp.sum(hp_output.electrical_power_w)
    ac_electrical_power_w = jnp.sum(ac_output.electrical_power_w)

    # The 'base_load_w' in exogenous comes from the dataset (lights, cooking, etc.)
    # We add the dynamic 'smart_appliance_load_w' (EVs, etc.) calculated in the sim.
    total_uncontrollable_load_w = exogenous.base_load_w + smart_appliance_load_w

    net_grid_power_w = (
        total_uncontrollable_load_w
        + actions.battery_power_w    # + = charging (buying)
        + hp_electrical_power_w
        + ac_electrical_power_w
        - solar_output.pv_generation_w
    )

    net_grid_energy_kwh = (net_grid_power_w * (dt_seconds / 3600.0)) / 1000.0

    # Simple Import Tariff
    cost_euros = jnp.fmax(0.0, net_grid_energy_kwh) * exogenous.price

    # ==========================================
    # 2. Comfort Cost (Thermal Violation)
    # ==========================================

    T_vector = state.thermal.T_vector
    room_temps = T_vector[jnp.array(t_conf.room_air_indices)]

    temp_error = room_temps - t_conf.setpoint
    comfort_violation = jnp.fmax(0.0, jnp.abs(temp_error) - t_conf.comfort_band)
    total_comfort_penalty = jnp.sum(comfort_violation**2)

    # ==========================================
    # 3. Waste Penalty (System Efficiency)
    # ==========================================
    # Penalize dumping heat (e.g. charging thermal storage when full)
    total_rejected_heat_w = jnp.sum(storage_output.rejected_heat_w)
    rejected_heat_kwh = (total_rejected_heat_w * (dt_seconds / 3600.0)) / 1000.0
    waste_penalty = rejected_heat_kwh * exogenous.price

    # ==========================================
    # 4. Robustness: Soft Constraints
    # ==========================================
    
    # Heat Pump Deadband (Prevent micro-cycling)
    hp_deadband_cost = _calculate_deadband_penalty(
        actions.heat_pump_power_w,
        hp_conf.min_electrical_power_w,
        weight=10.0
    )

    # AC Deadband
    ac_deadband_cost = _calculate_deadband_penalty(
        actions.ac_power_w,
        ac_conf.min_electrical_power_w,
        weight=10.0
    )

    # Slew Rate (Prevent rapid switching)
    slew_cost = (
        _calculate_slew_penalty(actions.heat_pump_power_w, prev_actions.heat_pump_power_w) +
        _calculate_slew_penalty(actions.ac_power_w, prev_actions.ac_power_w) +
        _calculate_slew_penalty(actions.battery_power_w, prev_actions.battery_power_w)
    )

    # ==========================================
    # 5. Total Weighted Objective
    # ==========================================
    total_cost = (
        (cost_euros * r_conf.price_weight) +
        (total_comfort_penalty * r_conf.comfort_weight) +
        (waste_penalty * r_conf.price_weight) +
        hp_deadband_cost +
        ac_deadband_cost +
        slew_cost
    )

    return total_cost


@partial(jit, static_argnames=[])
def f_terminal_cost(
    final_state: SystemState,
    initial_state: SystemState,
    configs: tuple,
    exo_forecast_end: ExogenousData
) -> Array:
    """
    Calculates the cost of the final state at step N.
    Prevents myopic behavior (draining battery, overheating) at the end of the horizon.
    """
    t_conf, b_conf, r_conf, _, _, _, _ = configs

    # ==========================================
    # 1. Battery Terminal Value (Price-Aware)
    # ==========================================
    # Strategy: If we leave energy in the battery, that is valuable!
    # Value = Energy_Left (kWh) * Price_At_End (€/kWh)
    # We SUBTRACT this value from the cost (Reward).
    
    final_energy_kwh = final_state.battery.soc * b_conf.capacity_kwh
    
    # Use the forecast price at step N to value the leftover energy.
    # If prices are high at the end, the solver will try to fill the battery (Sell High).
    # If prices are low, it might drain it.
    terminal_energy_value = final_energy_kwh * exo_forecast_end.price
    
    # Alternatively: Robust SOC Target (Simpler, more stable)
    # target_soc = initial_state.battery.soc # Try to return to start state (Net Zero)
    # soc_penalty = jnp.sum((final_state.battery.soc - target_soc)**2) * 100.0
    
    # We use the Value method here as it utilizes exo_forecast_end
    battery_term_cost = -terminal_energy_value * 1.0 # Weighting


    # ==========================================
    # 2. Thermal Terminal Constraint
    # ==========================================
    # Heavy penalty for leaving the house outside comfort bounds at t=N.
    
    T_vector = final_state.thermal.T_vector
    room_temps = T_vector[jnp.array(t_conf.room_air_indices)]
    
    temp_error = room_temps - t_conf.setpoint
    comfort_violation = jnp.fmax(0.0, jnp.abs(temp_error) - t_conf.comfort_band)
    
    # Very high weight to ensure we don't "borrow" comfort from the future
    thermal_term_cost = jnp.sum(comfort_violation**2) * 1000.0

    return battery_term_cost + thermal_term_cost