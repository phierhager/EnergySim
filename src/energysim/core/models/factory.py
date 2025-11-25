import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union

# Import models
from energysim.core.models.battery_model import (
    AbstractBatteryModel, SimpleBatteryModel,
    DegradationBatteryModel, PassthroughBatteryModel
)
from energysim.core.surrogate import ConvectionSurrogate
from energysim.core.models.thermal_model import (
    AbstractThermalModel, RCNetworkModel
)
from energysim.core.models.heat_pump_model import (
    AbstractHeatPumpModel, MechanisticHeatPump, PassthroughHeatPumpModel, RampingHeatPumpModel,
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
from energysim.core.forecaster import (
    AbstractForecaster, GaussianNoiseForecaster, 
    AR1Forecaster, ForecastConfig
    )
from energysim.core.models.machines import (
    AbstractMachine, PassiveEquipment, SmartAppliance
)
from energysim.core.models.occupancy import OccupancyModel
from energysim.core.shared.data_structs import MoistureConfig, MoistureState, ThermalConfig
from energysim.core.models.moisture_model import AbstractMoistureModel, DynamicMoistureModel
from energysim.core.physics.constants import Coefficients
from energysim.core.physics import psychrometrics as psych

# Import configs and dummies
from energysim.core.shared.data_structs import (
    ApplianceConfig, BatteryConfig, OccupantConfig, ThermalConfig, HeatPumpConfig,
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
        if config.model_type == "mechanistic":
            # Physics-Informed Mass Flow Model
            return MechanisticHeatPump(config, n_rooms)
        elif config.model_type == "variable_cop":
            return VariableCOPHeatPumpModel(config, n_rooms)
        elif config.model_type == "ramping":
            return RampingHeatPumpModel(config, n_rooms)
        elif config.model_type == "stateless":
            return StatelessHeatPumpModel(config, n_rooms)
        else:
            raise ValueError(f"Unknown heat_pump model_type: {config.model_type}")
    else:
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

def create_thermal(
    config: ThermalConfig, 
    key: Optional[jnp.ndarray] = None,
    surrogate_weights_path: Optional[str] = None # [NEW] Argument
) -> AbstractThermalModel:
    
    N_nodes = config.C_inv_vector.shape[0]
    initial_T = jnp.full((N_nodes,), config.setpoint)
    initial_T = initial_T.at[config.ambient_air_index].set(10.0)

    surrogate = None
    if config.use_surrogate_convection:
        if key is None:
            raise ValueError("Random Key required for Surrogate")
        
        # Load weights if path provided, otherwise random (for training)
        if surrogate_weights_path:
            surrogate = ConvectionSurrogate.load(surrogate_weights_path, key)
        else:
            surrogate = ConvectionSurrogate(key)

    return RCNetworkModel(config, initial_T, surrogate=surrogate)

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

def create_smart_machines(
    configs: list[ApplianceConfig],
    node_map: dict[str, int]
) -> list[SmartAppliance]:
    """
    Filters and creates ONLY Smart Appliances from the config list.
    """
    models = []
    for cfg in configs:
        if cfg.cycle_energy_kwh is not None and cfg.cycle_energy_kwh > 0:
            target_idx = node_map.get(cfg.target_node_name, 0)
            models.append(SmartAppliance(cfg, target_idx))
    return models

def create_passive_machines(
    configs: list[ApplianceConfig],
    node_map: dict[str, int]
) -> list[PassiveEquipment]:
    """
    Filters and creates ONLY Passive Equipment (Base Load, Lights) from the config list.
    """
    models = []
    for cfg in configs:
        # If it has NO energy budget (or 0), it is passive
        if cfg.cycle_energy_kwh is None or cfg.cycle_energy_kwh <= 0:
            target_idx = node_map.get(cfg.target_node_name, 0)
            models.append(PassiveEquipment(cfg, target_idx))
    return models

def create_occupants(
    configs: list[OccupantConfig],
    node_map: dict[str, int]
) -> list[OccupancyModel]:
    """Creates occupancy models."""
    models = []
    for cfg in configs:
        target_idx = node_map.get(cfg.target_node_name, 0)
        models.append(OccupancyModel(cfg, target_idx))
    return models


def create_moisture(
    t_config: ThermalConfig, 
    initial_rh: float = 0.5
) -> AbstractMoistureModel:
    """
    Creates a moisture model derived from the building's thermal volume.
    """
    # 1. Create Config derived from Thermal Config
    # We divide total volume by n_rooms to get per-room volume if not explicit
    n_rooms = len(t_config.room_air_indices)
    vol_per_room = t_config.room_vol_m3 / n_rooms if n_rooms > 0 else 0.0
    
    config = MoistureConfig(
        air_volume_m3=vol_per_room,
        initial_rel_humidity=initial_rh
    )
    
    # 2. Initialize State (w) based on initial RH and default conditions (20C, 101325Pa)
    # This ensures we start in a valid physics state
    w_init_scalar = psych.calculate_humidity_ratio(
        T_celsius=jnp.array(20.0),
        rel_hum_0_1=jnp.array(initial_rh),
        pressure_pa=jnp.array(101325.0)
    )
    
    state = MoistureState(
        humidity_ratio_kg_kg=jnp.full((n_rooms,), w_init_scalar)
    )
    
    return DynamicMoistureModel(config, state)