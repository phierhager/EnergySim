# energysim/core/models/factory.py
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

# Import models
from energysim.core.models.battery_model import (
    AbstractBatteryModel, SimpleBatteryModel, 
    DegradationBatteryModel, PassthroughBatteryModel
)
from energysim.core.models.thermal_model import ThermalModel
from energysim.core.models.heat_pump_model import AbstractHeatPumpModel, PassthroughHeatPumpModel, RampingHeatPumpModel, StatelessHeatPumpModel
from energysim.core.models.air_conditioner_model import AbstractAirConditionerModel, PassthroughAirConditionerModel, RampingAirConditionerModel, StatelessAirConditionerModel
from energysim.core.models.thermal_storage_model import (
    AbstractThermalStorage, ThermalStorageModel, ThermalStoragePassthrough
)

# Import configs and dummies
from energysim.core.shared.data_structs import (
    BatteryConfig, ThermalConfig, HeatPumpConfig, 
    AirConditionerConfig, ThermalStorageConfig
)

# --- DUMMY CONFIGS for other optional components ---
DUMMY_STORAGE_CONFIG = ThermalStorageConfig(
    capacity_kwh=0.0,
    max_charge_kw=0.0,
    max_discharge_kw=0.0,
    standing_loss_rate=0.0
)
DUMMY_BATTERY_CONFIG = BatteryConfig(capacity_kwh=0.0, max_power_kw=0.0, efficiency=1.0)
DUMMY_HP_CONFIG = HeatPumpConfig(max_electrical_power_w=0.0, cop_heating=1.0)
DUMMY_AC_CONFIG = AirConditionerConfig(max_electrical_power_w=0.0, cop_cooling=1.0)

# --- Factory Functions ---

def create_battery(config: Optional[BatteryConfig]) -> AbstractBatteryModel:
    if config:
        if config.model_type == "simple":
            return SimpleBatteryModel(config, initial_soc=0.5)
        elif config.model_type == "degradation":
            return DegradationBatteryModel(config, initial_soc=0.5, initial_soh=1.0)
        else:
            raise ValueError(f"Unknown battery model_type: {config.model_type}")
    else:
        # Return the clean Passthrough model instead of a dummy config
        return PassthroughBatteryModel(DUMMY_BATTERY_CONFIG)

def create_heat_pump(config: Optional[HeatPumpConfig]) -> AbstractHeatPumpModel:
    if config:
        if config.model_type == "stateless":
            return StatelessHeatPumpModel(config)
        elif config.model_type == "ramping":
            return RampingHeatPumpModel(config)
        else:
            raise ValueError(f"Unknown heat_pump model_type: {config.model_type}")
    else:
        # Use the PASSTHROUGH model
        return PassthroughHeatPumpModel(DUMMY_HP_CONFIG)

def create_ac(config: Optional[AirConditionerConfig]) -> AbstractAirConditionerModel:
    if config:
        if config.model_type == "stateless":
            return StatelessAirConditionerModel(config)
        elif config.model_type == "ramping":
            return RampingAirConditionerModel(config)
        else:
            raise ValueError(f"Unknown ac model_type: {config.model_type}")
    else:
        # Use the PASSTHROUGH model
        return PassthroughAirConditionerModel(DUMMY_AC_CONFIG)

def create_storage(config: Optional[ThermalStorageConfig]) -> AbstractThermalStorage:
    if config:
        # Use the REAL model
        return ThermalStorageModel(config, initial_soc=0.5)
    else:
        # Use the PASSTHROUGH (bypass) model
        return ThermalStoragePassthrough(DUMMY_STORAGE_CONFIG)

def create_thermal(config: ThermalConfig) -> ThermalModel:
    # The room itself is not optional
    return ThermalModel(config, initial_temp=20.0)