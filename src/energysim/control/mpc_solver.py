import jax
import jax.numpy as jnp
from jax import jit, grad, lax
from functools import partial
from typing import Optional
import jaxopt
import equinox as eqx

# Import models and factory
from ..core.models.battery_model import AbstractBatteryModel
from ..core.models.thermal_model import AbstractThermalModel
from ..core.models.heat_pump_model import AbstractHeatPumpModel
from ..core.models.air_conditioner_model import AbstractAirConditionerModel
from ..core.models.thermal_storage_model import AbstractThermalStorage
from ..core.models.solar_model import AbstractSolarModel
from ..core.models.factory import (
    create_battery, create_thermal, create_heat_pump,
    create_ac, create_storage, create_solar
)

# --- UPDATED IMPORTS: Added f_terminal_cost ---
from ..core.objectives import f_cost_step, f_terminal_cost

from ..core.shared.data_structs import (
    AirConditionerState, HeatPumpState, SystemState, SystemActions, ExogenousData,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig,
    ThermalState, BatteryState, ThermalStorageState, SolarOutput
)


class JAX_MPC_Solver:
    """
    A full MPC solver built on JAX, supporting modular components.
    """
    def __init__(
        self,
        N_horizon: int,
        dt_seconds: float,
        # --- Configs ---
        t_config: ThermalConfig,
        r_config: RewardConfig,
        b_config: Optional[BatteryConfig] = None,
        hp_config: Optional[HeatPumpConfig] = None,
        ac_config: Optional[AirConditionerConfig] = None,
        ts_config: Optional[ThermalStorageConfig] = None,
        s_config: Optional[SolarConfig] = None
    ):
        self.N = N_horizon
        self.dt = dt_seconds

        self.n_rooms = len(t_config.room_air_indices)
        if self.n_rooms == 0:
            raise ValueError(
                "ThermalConfig has no 'room_air_indices'. MPC solver requires n_rooms > 0."
            )

        # --- 1. Create Models using the Factory ---
        self.battery: AbstractBatteryModel = create_battery(b_config)
        self.thermal: AbstractThermalModel = create_thermal(t_config)
        self.heat_pump: AbstractHeatPumpModel = create_heat_pump(hp_config, self.n_rooms)
        self.ac: AbstractAirConditionerModel = create_ac(ac_config, self.n_rooms)
        self.storage: AbstractThermalStorage = create_storage(ts_config)
        self.solar: AbstractSolarModel = create_solar(s_config)

        # --- 2. Store Configs for Cost Function ---
        self.configs = (
            self.thermal.config, self.battery.config, r_config,
            self.heat_pump.config, self.ac.config, self.storage.config,
            self.solar.config
        )

        # --- 3. Define the Scan Step Function ---
        @jit
        def mpc_scan_step(
            carry: tuple[SystemState, SystemActions], # <--- CHANGED: Carry now holds (State, PrevAction)
            inputs_k: tuple[SystemActions, ExogenousData]
        ):
            state_k, prev_action_k = carry # Unpack previous action for Slew Rate
            action_k, exo_k = inputs_k

            solar_output_k = self.solar.calculate(exo_k)

            # --- Re-hydrate models ---
            battery_k = eqx.tree_at(
                lambda m: (m.soc, m.soh),
                self.battery,
                (state_k.battery.soc, state_k.battery.soh)
            )
            thermal_k = eqx.tree_at(
                lambda m: m.T_vector, self.thermal, state_k.thermal.T_vector
            )
            storage_k = eqx.tree_at(
                lambda m: m.temperatures_c, self.storage, state_k.storage.temperatures_c
            )
            hp_k = eqx.tree_at(
                lambda m: m.current_electrical_w, self.heat_pump, state_k.heat_pump.current_electrical_w
            )
            ac_k = eqx.tree_at(
                lambda m: m.current_electrical_w, self.ac, state_k.air_conditioner.current_electrical_w
            )

            # --- A. Run HVAC models ---
            next_hp, hp_output = hp_k.step(
                action_k.heat_pump_power_w, exo_k, self.dt
            )
            next_ac, ac_output = ac_k.step(
                action_k.ac_power_w, exo_k, self.dt
            )

            # --- B. Run other stateful models ---
            next_battery = battery_k.step(action_k.battery_power_w, self.dt)
            
            next_storage, storage_output = storage_k.step(
                action_k.storage_discharge_w,
                hp_output.thermal_power_w,
                self.dt
            )

            # --- C. Run Thermal Model ---
            heating_w = storage_output.actual_discharge_w
            cooling_w = ac_output.thermal_power_w
            
            # Calculate total waste heat (storage losses + rejected heat)
            total_waste_w = storage_output.standing_loss_w + jnp.sum(storage_output.rejected_heat_w)

            next_thermal = thermal_k.step(
                heating_w,
                cooling_w,
                total_waste_w,
                exo_k,
                self.dt
            )

            # --- D. Calculate cost of current step ---
            # <--- CHANGED: Pass prev_action_k to f_cost_step
            cost_k = f_cost_step(
                state_k, action_k, prev_action_k, exo_k,
                hp_output, ac_output, storage_output,
                solar_output_k,
                self.configs, self.dt
            )

            # --- E. Create next state (data-only) ---
            state_k_plus_1 = SystemState(
                thermal=ThermalState(T_vector=next_thermal.T_vector),
                battery=BatteryState(soc=next_battery.soc, soh=next_battery.soh),
                storage=ThermalStorageState(temperatures_c=next_storage.temperatures_c),
                heat_pump=HeatPumpState(current_electrical_w=next_hp.current_electrical_w, 
                                        current_thermal_w=next_hp.current_thermal_w),
                air_conditioner=AirConditionerState(current_electrical_w=next_ac.current_electrical_w, 
                                                     current_thermal_w=next_ac.current_thermal_w)
            )
            
            # <--- CHANGED: Return (NewState, CurrentAction) as the carry for the next step
            return (state_k_plus_1, action_k), cost_k

        self.mpc_scan_step = mpc_scan_step # Save for reuse

        # --- 4. Define the Total Horizon Cost Function ---
        @jit
        def f_horizon_cost(
            action_sequence: SystemActions, # PyTree of actions [N, ...]
            initial_state: SystemState,     # static
            exo_sequence: ExogenousData     # PyTree of forecasts [N, ...]
        ):
            inputs_over_horizon = (action_sequence, exo_sequence)
            
            # <--- CHANGED: Initialize Carry with dummy "Previous Action" (Zeros) for t=0
            # We use tree_map to create a structure of zeros matching a single time step of actions
            dummy_prev_action = jax.tree.map(
                lambda x: jnp.zeros_like(x[0]), 
                action_sequence
            )
            
            init_carry = (initial_state, dummy_prev_action)

            # 1. Run the Scan
            (final_state, final_action), cost_sequence = lax.scan(
                self.mpc_scan_step, init_carry, inputs_over_horizon
            )
            
            # 2. Calculate Terminal Cost (Robustness) <--- CHANGED
            # Extract the exogenous data for the *end* of the horizon to price remaining energy
            last_exo = jax.tree.map(lambda x: x[-1], exo_sequence)
            
            term_cost = f_terminal_cost(
                final_state, 
                initial_state, 
                self.configs, 
                last_exo
            )
            
            # 3. Total Objective
            return jnp.sum(cost_sequence) + term_cost

        # --- 5. JIT-compile the cost function ---
        self.objective_fn = f_horizon_cost

        # --- 6. Setup the Optimizer ---
        b_conf = self.battery.config
        hp_conf = self.heat_pump.config
        ac_conf = self.ac.config
        ts_conf = self.storage.config

        scalar_shape = (N_horizon,)
        zonal_shape = (N_horizon, self.n_rooms)

        self.action_bounds = (
            SystemActions(
                battery_power_w=jnp.full(scalar_shape, -b_conf.max_power_w),
                heat_pump_power_w=jnp.full(zonal_shape, 0.0),
                ac_power_w=jnp.full(zonal_shape, 0.0),
                storage_discharge_w=jnp.full(zonal_shape, 0.0)
            ),
            SystemActions(
                battery_power_w=jnp.full(scalar_shape, b_conf.max_power_w),
                heat_pump_power_w=jnp.full(zonal_shape, hp_conf.max_electrical_power_w / self.n_rooms),
                ac_power_w=jnp.full(zonal_shape, ac_conf.max_electrical_power_w / self.n_rooms),
                storage_discharge_w=jnp.full(zonal_shape, ts_conf.max_discharge_w / self.n_rooms)
            )
        )

        self.optimizer = jaxopt.ProjectedGradient(
            fun=self.objective_fn,
            projection=jaxopt.projection.projection_box,
            maxiter=50,
            tol=1e-3,
        )

        self.zonal_warm_start = jax.tree.map(
            lambda x: jnp.zeros_like(x), self.action_bounds[0]
        )

    def solve(
        self,
        current_state: SystemState,
        exo_forecast: ExogenousData,
        warm_start_actions: SystemActions = None
    ) -> SystemActions:

        if warm_start_actions is None:
            warm_start_actions = self.zonal_warm_start

        optim_result = self.optimizer.run(
            init_params=warm_start_actions,
            hyperparams_proj=self.action_bounds,
            initial_state=current_state,
            exo_sequence=exo_forecast
        )

        optimal_action_sequence = optim_result.params
        first_action = jax.tree.map(lambda x: x[0], optimal_action_sequence)

        return first_action