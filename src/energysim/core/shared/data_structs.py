# energysim/core/shared/data_structs.py
from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass

# Define array type for clarity
Array = jnp.ndarray

# --- Configuration Structs ---
# These are static and passed as parameters

@dataclass(frozen=True)
class ThermalConfig:
    """Parameters for the thermal model."""
    # --- Model Selection ---
    model_type: Literal["1R1C", "2R2C", "passthrough"] = "1R1C"

    # --- Shared Parameters ---
    setpoint: float = 21.0           # °C
    comfort_band: float = 1.0        # °C
    
    # --- 1R1C Model Parameters (Existing) ---
    air_volume_m3: float = 200.0         # m³ (Renamed from building_volume)
    air_density_kg_m3: float = 1.2       # kg/m³
    air_specific_heat_j_kgk: float = 1005.0 # J/kg·K
    air_to_ambient_coeff_w_k: float = 100.0 # W/K (U-value * Area) (Renamed from heat_transfer_coeff)

    # --- 2R2C Model Parameters (New) ---
    # Note: 2R2C also uses air_volume_m3, air_density_kg_m3, air_specific_heat_j_kgk
    wall_capacity_j_k: float = 5.0e6      # J/K (Thermal mass of walls)
    wall_to_ambient_r_k_w: float = 0.01   # K/W (Resistance from wall to ambient)
    air_to_wall_r_k_w: float = 0.001      # K/W (Resistance from internal air to wall)
    
    @property
    def air_capacity_j_k(self) -> float:
        """Thermal capacity of the internal air mass."""
        return self.air_volume_m3 * self.air_density_kg_m3 * self.air_specific_heat_j_kgk

@dataclass(frozen=True)
class BatteryConfig:
    """Parameters for the battery model."""
    model_type: Literal["simple", "degradation"] = "simple"
    
    capacity_kwh: float = 10.0
    max_power_kw: float = 5.0
    efficiency: float = 0.90 # Round-trip

    # for degradation model
    degradation_rate_per_cycle: float = 0.0001 # Fractional SOH loss per full cycle

    @property
    def capacity_j(self) -> float:
        return self.capacity_kwh * 3.6e6

    @property
    def max_power_w(self) -> float:
        return self.max_power_kw * 1000.0

@dataclass(frozen=True)
class RewardConfig:
    """Weights for the cost/reward function."""
    price_weight: float = 1.0
    comfort_weight: float = 5.0 # Penalize discomfort highly

@dataclass(frozen=True)
class AirConditionerConfig:
    """Parameters for the Air Conditioner model."""
    model_type: Literal["stateless", "ramping"] = "stateless"
    max_electrical_power_w: float = 5000.0 # 5kW max electricity draw
    cop_cooling: float = 3.0

    # Only for ramping model
    ramp_rate_w_per_sec: float = 1000.0 # W of *electrical* power change per sec

@dataclass(frozen=True)
class HeatPumpConfig:
    """Parameters for the Heat Pump model."""
    model_type: Literal["stateless", "ramping"] = "stateless"
    max_electrical_power_w: float = 5000.0 # 5kW max electricity draw
    cop_heating: float = 3.5               # Coeff. of Performance

    # Only for ramping model
    ramp_rate_w_per_sec: float = 1000.0 # W of *electrical* power change per sec

@dataclass(frozen=True)  # <--- NEW CONFIG
class ThermalStorageConfig:
    """Parameters for the Thermal Storage (Water Tank) model."""
    capacity_kwh: float = 50.0  # 50 kWh capacity
    max_charge_kw: float = 15.0   # Max thermal power in
    max_discharge_kw: float = 15.0 # Max thermal power out
    standing_loss_rate: float = 0.01 # 1% of *total capacity* lost per hour

    @property
    def capacity_j(self) -> float:
        return self.capacity_kwh * 3.6e6
    
    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0
    
    @property
    def standing_loss_w_per_soc(self) -> float:
        # Converts %/hr loss of *capacity* to W loss per unit of SOC
        return (self.capacity_kwh * 1000.0 * self.standing_loss_rate)


# --- Dynamic State Structs ---
# These structs hold the JAX arrays that change at each step.

@flax_dataclass
class ThermalState:
    """State of the thermal model."""
    room_temp: Array  # °C
    wall_temp: Array  # °C

@flax_dataclass
class BatteryState:
    """State of the battery model."""
    soc: Array  # [0.0 - 1.0]
    soh: Array  # [0.0 - 1.0], only for degradation model

@flax_dataclass 
class ThermalStorageState:
    """Data-only view of thermal storage state."""
    soc: Array  # [0.0 - 1.0]

@flax_dataclass
class HeatPumpState:
    """Data-only view of heat pump state."""
    current_electrical_w: Array # W

@flax_dataclass
class AirConditionerState:
    """Data-only view of AC state."""
    current_electrical_w: Array # W

@flax_dataclass
class SystemState:
    """The complete internal state of the simulation."""
    thermal: ThermalState
    battery: BatteryState
    storage: ThermalStorageState
    heat_pump: HeatPumpState
    air_conditioner: AirConditionerState

@flax_dataclass
class ExogenousData:
    """All external data for a single timestep."""
    ambient_temp: Array # °C
    load: Array         # W
    pv: Array           # W
    price: Array        # €/kWh
    internal_gains_w: Array # W (e.g., people, computers)
    solar_gains_w: Array    # W (e.g., direct sunlight)

@flax_dataclass
class SystemActions:  # <--- UPDATED
    """All control actions for a single timestep."""
    battery_power_w: Array        # W
    heat_pump_power_w: Array    # W (Electrical)
    ac_power_w: Array           # W (Electrical)
    storage_discharge_w: Array  # W (Thermal)

@flax_dataclass
class HeatPumpOutput:  # <--- RENAMED
    """The calculated outputs of the Heat Pump model for one step."""
    thermal_power_w: Array   # The *actual* (clipped) thermal power *generated*
    electrical_power_w: Array # The electrical power *consumed*

@flax_dataclass  # <--- NEW
class AirConditionerOutput:
    """The calculated outputs of the AC model for one step."""
    thermal_power_w: Array   # The *actual* (clipped) thermal power *removed* (will be negative)
    electrical_power_w: Array # The electrical power *consumed*


@flax_dataclass 
class ThermalStorageOutput:
    """Calculated outputs from the thermal storage step."""
    actual_discharge_w: Array  # Actual thermal power to room
    rejected_heat_w: Array     # Wasted heat (charged when full)