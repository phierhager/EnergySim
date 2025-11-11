# energysim/sim/simulator.py
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Optional

from ..core.models.factory import (
    create_battery, create_thermal, create_heat_pump,
    create_ac, create_storage
)
from ..core.models.battery_model import AbstractBatteryModel
from ..core.models.thermal_model import ThermalModel
# --- UPDATED IMPORTS ---
from ..core.models.heat_pump_model import AbstractHeatPumpModel
from ..core.models.air_conditioner_model import AbstractAirConditionerModel
from ..core.models.thermal_storage_model import AbstractThermalStorage
from ..core.models.objectives import f_cost_step
from ..core.shared.data_structs import (
    SystemActions, ExogenousData,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig,
    SystemState, BatteryState, ThermalState, ThermalStorageState,
    HeatPumpState, AirConditionerState  # <-- NEW IMPORTS
)

class JAXSimulator:
    def __init__(
        self,
        dt_seconds: float,
        t_config: ThermalConfig,
        r_config: RewardConfig,
        b_config: Optional[BatteryConfig] = None,
        hp_config: Optional[HeatPumpConfig] = None,
        ac_config: Optional[AirConditionerConfig] = None,
        ts_config: Optional[ThermalStorageConfig] = None
    ):
        self.dt_seconds = dt_seconds

        # --- 1. Create Models using the Factory ---
        self.initial_battery = create_battery(b_config)
        self.initial_thermal = create_thermal(t_config)
        self.initial_heat_pump = create_heat_pump(hp_config)
        self.initial_ac = create_ac(ac_config)
        self.initial_storage = create_storage(ts_config)

        # --- 2. Store Configs for Cost Function ---
        self.configs = (
            self.initial_thermal.config, self.initial_battery.config, r_config,
            self.initial_heat_pump.config, self.initial_ac.config, self.initial_storage.config
        )

        # --- Mutable State Variables ---
        self._battery: AbstractBatteryModel = self.initial_battery
        self._thermal: ThermalModel = self.initial_thermal
        self._heat_pump: AbstractHeatPumpModel = self.initial_heat_pump # <-- Type hint
        self._ac: AbstractAirConditionerModel = self.initial_ac       # <-- Type hint
        self._storage: AbstractThermalStorage = self.initial_storage

        # --- 3. Pre-bind static arguments for the COST function ---
        self.cost_fn = partial(
            jit(f_cost_step, static_argnames=["configs", "dt_seconds"]),
            configs=self.configs,
            dt_seconds=self.dt_seconds
        )
        
        # --- 4. Store active configs (for wrappers) ---
        self.active_configs = {
            "battery": b_config, "heat_pump": hp_config,
            "ac": ac_config, "storage": ts_config
        }

    @property
    def state(self) -> SystemState:
        # --- UPDATED to build full SystemState ---
        return SystemState(
            thermal=ThermalState(room_temp=self._thermal.room_temp),
            battery=BatteryState(soc=self._battery.soc, soh=self._battery.soh),
            storage=ThermalStorageState(soc=self._storage.soc),
            heat_pump=HeatPumpState(current_electrical_w=self._heat_pump.current_electrical_w),
            air_conditioner=AirConditionerState(current_electrical_w=self._ac.current_electrical_w)
        )

    def reset(self) -> SystemState:
        """Resets the stateful models and returns the initial state."""
        self._battery = self.initial_battery
        self._thermal = self.initial_thermal
        self._heat_pump = self.initial_heat_pump
        self._ac = self.initial_ac
        self._storage = self.initial_storage
        return self.state

    def step(self, actions: SystemActions, exo_data: ExogenousData) -> tuple[SystemState, float]:
        """
        Advances the simulation by one step given actions and exogenous data.
        """

        # --- 1. Run HVAC models (now stateful) ---
        next_heat_pump, hp_output = self._heat_pump.step(
            actions.heat_pump_power_w, self.dt_seconds
        )
        next_ac, ac_output = self._ac.step(
            actions.ac_power_w, self.dt_seconds
        )

        # --- 2. Run other stateful models ---
        next_battery = self._battery.step(
            actions.battery_power_w, self.dt_seconds
        )

        next_storage, storage_output = self._storage.step(
            actions.storage_discharge_w,
            hp_output.thermal_power_w, # Use output from new HP
            self.dt_seconds
        )

        # --- 3. Calculate Cost (using state *before* the step) ---
        cost = self.cost_fn(
            self.state, actions, exo_data,
            hp_output, ac_output, storage_output
        )

        # --- 4. Run final stateful model ---
        next_thermal = self._thermal.step(
            storage_output.actual_discharge_w,
            ac_output.thermal_power_w, # Use output from new AC
            exo_data,
            self.dt_seconds
        )

        # --- 5. Update mutable state ---
        self._battery = next_battery
        self._thermal = next_thermal
        self._storage = next_storage
        self._heat_pump = next_heat_pump # <-- NEW
        self._ac = next_ac              # <-- NEW

        # --- 6. Return new state and cost ---
        return self.state, float(cost)