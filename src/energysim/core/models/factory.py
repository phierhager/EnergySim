import jax.numpy as jnp
import equinox as eqx
from typing import Optional

# Import models
from energysim.core.models.battery_model import (
    AbstractBatteryModel, SimpleBatteryModel,
    DegradationBatteryModel, PassthroughBatteryModel
)
from energysim.core.models.thermal_model import (
    AbstractThermalModel, RCNetworkModel
)
from energysim.core.models.heat_pump_model import (
    AbstractHeatPumpModel, PassthroughHeatPumpModel, RampingHeatPumpModel,
    StatelessHeatPumpModel, VariableCOPHeatPumpModel
)
from energysim.core.models.air_conditioner_model import (
    AbstractAirConditionerModel, PassthroughAirConditionerModel, RampingAirConditionerModel,
    StatelessAirConditionerModel, VariableCOPAirConditionerModel
)
from energysim.core.models.thermal_storage_model import (
    AbstractThermalStorage, StratifiedThermalStorageModel, ThermalStoragePassthrough, GridThermalStorageModel
)
from energysim.core.models.solar_model import (
    AbstractSolarModel, SimpleSolarModel, PassthroughSolarModel
)
from energysim.core.models.forecaster import (
    AbstractForecaster, GaussianNoiseForecaster, 
    AR1Forecaster, ForecastConfig
    )
from energysim.core.models.appliance_model import (
    AbstractApplianceModel, PassiveApplianceModel, ShiftableLoadModel
)


# Import configs and dummies
from energysim.core.shared.data_structs import (
    ApplianceConfig, BatteryConfig, ThermalConfig, HeatPumpConfig,
    AirConditionerConfig, ThermalStorageConfig,
    SolarConfig, GridThermalStorageConfig
)

# ... (Dummy configs are unchanged) ...
DUMMY_STORAGE_CONFIG = ThermalStorageConfig()
DUMMY_BATTERY_CONFIG = BatteryConfig()
DUMMY_HP_CONFIG = HeatPumpConfig()
DUMMY_AC_CONFIG = AirConditionerConfig()
DUMMY_SOLAR_CONFIG = SolarConfig(model_type="passthrough")


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
        return PassthroughBatteryModel(DUMMY_BATTERY_CONFIG)

def create_heat_pump(config: Optional[HeatPumpConfig], n_rooms: int) -> AbstractHeatPumpModel:
    if config:
        if config.model_type == "stateless":
            return StatelessHeatPumpModel(config, n_rooms)
        elif config.model_type == "ramping":
            return RampingHeatPumpModel(config, n_rooms)
        elif config.model_type == "variable_cop":
            return VariableCOPHeatPumpModel(config, n_rooms)
        else:
            raise ValueError(f"Unknown heat_pump model_type: {config.model_type}")
    else:
        # Still pass n_rooms to dummy model for state shape consistency
        return PassthroughHeatPumpModel(DUMMY_HP_CONFIG, n_rooms)

def create_ac(config: Optional[AirConditionerConfig], n_rooms: int) -> AbstractAirConditionerModel:
    if config:
        if config.model_type == "stateless":
            return StatelessAirConditionerModel(config, n_rooms)
        elif config.model_type == "ramping":
            return RampingAirConditionerModel(config, n_rooms)
        elif config.model_type == "variable_cop":
            return VariableCOPAirConditionerModel(config, n_rooms)
        else:
            raise ValueError(f"Unknown ac model_type: {config.model_type}")
    else:
        # Still pass n_rooms to dummy model for state shape consistency
        return PassthroughAirConditionerModel(DUMMY_AC_CONFIG, n_rooms)

def create_storage(config: Optional[eqx.Module]) -> AbstractThermalStorage:
    if config is None:
        return ThermalStoragePassthrough(DUMMY_STORAGE_CONFIG)
        
    if isinstance(config, GridThermalStorageConfig):
        # High-Fidelity 2D/3D Model
        return GridThermalStorageModel(config, initial_temp_c=45.0)
        
    elif isinstance(config, ThermalStorageConfig):
        # Standard 1D Stratified Model
        return StratifiedThermalStorageModel(config, initial_temp_c=45.0)
        
    else:
        raise ValueError(f"Unknown storage config type: {type(config)}")

def create_thermal(config: ThermalConfig) -> AbstractThermalModel:
    """Factory function for the RC-Network model."""

    N_nodes = config.C_inv_vector.shape[0]
    initial_T = jnp.full((N_nodes,), config.setpoint)
    
    initial_T = initial_T.at[config.ambient_air_index].set(10.0)
    
    return RCNetworkModel(config, initial_T)

def create_solar(config: Optional[SolarConfig]) -> AbstractSolarModel:
    """Factory function for solar PV models."""
    if config:
        if config.model_type == "simple":
            return SimpleSolarModel(config)
        elif config.model_type == "passthrough":
            return PassthroughSolarModel(config)
        else:
            raise ValueError(f"Unknown solar model_type: {config.model_type}")
    else:
        return PassthroughSolarModel(DUMMY_SOLAR_CONFIG)

def create_forecaster(config: Optional[ForecastConfig] = None) -> AbstractForecaster:
    if config is None:
        config = ForecastConfig()
        
    if config.model_type == "gaussian":
        return GaussianNoiseForecaster(config)
    elif config.model_type == "ar1":
        return AR1Forecaster(config)
    else:
        raise ValueError(f"Unknown forecaster type: {config.model_type}")
    

def create_appliances(
    app_configs: list[ApplianceConfig], 
    node_map: dict[str, int]
) -> list[AbstractApplianceModel]:
    """
    Factory that creates the appropriate model (Smart or Passive) based on config.
    """
    models = []
    
    for app in app_configs:
        # 1. Resolve Topology
        if app.target_node_name in node_map:
            target_idx = node_map[app.target_node_name]
        else:
             # Default to ambient or raise error. 
             # Using 0 ensures code doesn't crash, but raising error is better for config debugging.
            raise ValueError(f"Unknown node '{app.target_node_name}' for appliance '{app.name}'")

        # 2. Dispatch
        # If 'cycle_energy_kwh' is defined and > 0, it's a Smart Load (EV, Washer).
        # Otherwise, it's a Passive Load (Lights, Fridge, People).
        
        if app.cycle_energy_kwh is not None and app.cycle_energy_kwh > 0:
            # SMART
            model = ShiftableLoadModel(
                config=app, 
                target_node_index=target_idx
            )
        else:
            # DUMB (Passive)
            model = PassiveApplianceModel(
                config=app,
                target_node_index=target_idx
            )
            
        models.append(model)

    return models