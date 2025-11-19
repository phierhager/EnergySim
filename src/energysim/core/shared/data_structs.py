import jax.numpy as jnp
import equinox as eqx
from dataclasses import field
from typing import Literal, Tuple
import dataclasses
from typing import List, Dict, Optional
import jax.numpy as jnp

class ApplianceConfig(eqx.Module):
    """
    Unified configuration for ALL appliances (Smart or Dumb).
    """
    name: str = eqx.field(static=True)
    target_node_name: str = eqx.field(static=True)
    
    # Physical Props
    nominal_power_w: float = eqx.field(static=True)
    convective_fraction: float = eqx.field(static=True)
    metabolic_heat_w: float = eqx.field(static=True, default=0.0)

    # --- Smart Params (Optional) ---
    # If these are set, the Factory treats it as a ShiftableLoadModel
    cycle_energy_kwh: Optional[float] = eqx.field(static=True, default=None)
    
    @property
    def total_energy_j(self):
        if self.cycle_energy_kwh is None:
            return 0.0
        return self.cycle_energy_kwh * 3.6e6

class ComponentState(eqx.Module):
    """
    State passed from the Behavioral Model to the Simulator.
    0.0 = Off/Absent, 1.0 = On/Present.
    Can be fractional (e.g., 0.5 speed or 0.5 probability).
    """
    activation: jnp.ndarray  # Shape: (n_appliances,)

@dataclasses.dataclass
class Material:
    name: str
    conductivity: float
    density: float
    specific_heat: float
    thickness: float
    absorptivity: float # Default solar absorptivity
    
    @property
    def R_value(self): return self.thickness / self.conductivity
    @property
    def C_area_density(self): return self.thickness * self.density * self.specific_heat

@dataclasses.dataclass
class Surface:
    name: str
    zone_name: str
    type: str # 'WALL', 'ROOF', 'FLOOR', 'WINDOW'
    area: float
    azimuth: float 
    tilt: float
    construction_name: str
    boundary_condition: str 
    
    # New Physics Properties
    roughness: str = "Medium" # Maps to SurfaceRoughness enum
    absorptivity_solar: float = 0.7 # For Opaque (0.0-1.0)
    emissivity_longwave: float = 0.9 # For IR exchange (0.0-1.0)
    
    @property
    def normal(self):
        """Returns 3D normal vector [x, y, z] for solar calc."""
        # Simplified conversion from azimuth/tilt to vector
        rad_az = jnp.radians(self.azimuth)
        rad_tilt = jnp.radians(self.tilt)
        
        x = jnp.sin(rad_tilt) * jnp.sin(rad_az)
        y = jnp.sin(rad_tilt) * jnp.cos(rad_az)
        z = jnp.cos(rad_tilt)
        return jnp.array([x, y, z])
    
@dataclasses.dataclass
class WindowType:
    name: str
    u_value: float       # Datasheet Overall U-value (W/m2K)
    shgc: float         # Solar Heat Gain Coefficient
    
    # New physical properties
    glass_thickness: float = 0.006  # m (Total glass thickness, e.g. 3mm + 3mm)
    density: float = 2500.0         # kg/m3 (Standard glass)
    specific_heat: float = 840.0    # J/kgK
    
    def calculate_properties(self):
        """
        Decomposes datasheet U-value into physical layer resistance.
        1/U_total = 1/h_ext + R_glazing + 1/h_int
        We solve for R_glazing to use in our dynamic model.
        """
        # Standard reference conditions for U-value rating (NFRC/ISO)
        h_ext_ref = 23.0 # W/m2K
        h_int_ref = 8.3  # W/m2K
        
        R_total_ref = 1.0 / self.u_value
        
        # R_glazing includes the glass conduction AND the gas gap
        self.R_glazing_layer = R_total_ref - (1.0/h_ext_ref) - (1.0/h_int_ref)
        
        # Safety clamp (in case U-value is terrible or physics break)
        self.R_glazing_layer = max(self.R_glazing_layer, 0.001)
        
        # Estimate split between Transmission and Absorption based on SHGC
        # Simplify: Transmissivity ~ SHGC * 0.85, Absorptivity ~ The rest
        # (Real EnergyPlus does this spectrally, this is a heuristic)
        self.transmissivity = self.shgc * 0.85
        self.absorptivity = self.shgc * 0.15  # Absorbed in the outer pane mostly


# Define array type for clarity
Array = jnp.ndarray

# ==========================================
# 1. Configuration Modules (Static/Physics)
# ==========================================

# Assuming this code replaces the ThermalConfig class in your data_structs.py (or similar file)

class ThermalConfig(eqx.Module):
    """
    Configuration for a full RC-Network thermal model.
    """
    # --- 1. The Matrices (Dynamic Leaves) ---
    A_matrix: Array
    C_inv_vector: Array
    B_matrix: Array

    # --- 2. Node Indices (Static) ---
    ambient_air_index: int = eqx.field(static=True)
    ground_node_index: int = eqx.field(static=True)
    room_air_indices: Tuple[int, ...] = eqx.field(static=True)
    wall_indices: Tuple[int, ...] = eqx.field(static=True)
    mass_indices: Tuple[int, ...] = eqx.field(static=True)

    # --- 3. Metadata (Static) ---
    node_names: List[str] = eqx.field(static=True)
    node_map: Dict[str, int] = eqx.field(static=True)
    input_map: Dict[str, Dict[str, int]] = eqx.field(static=True)

    # --- 4. B-Matrix Column Indices (CRITICAL for decoupled step) ---
    # These arrays hold the column index in B_matrix corresponding to each room's action.
    u_idx_heating: Array = eqx.field(static=True) # Shape (n_rooms,)
    u_idx_cooling: Array = eqx.field(static=True) # Shape (n_rooms,)
    
    # --- 5. Coupling Indices (Static) ---
    waste_heat_node_index: int = eqx.field(static=True, default=-1)

    # --- 6. Infiltration Parameters (Static/Hyperparams) ---
    use_dynamic_infiltration: bool = eqx.field(static=True, default=False)
    inf_k1: float = eqx.field(static=True, default=0.1)
    inf_k2: float = eqx.field(static=True, default=0.0)
    inf_k3: float = eqx.field(static=True, default=0.0)
    room_vol_m3: float = eqx.field(static=True, default=0.0)

    # --- 7. Cost/Control Parameters (Static) ---
    setpoint: float = eqx.field(static=True, default=21.0)
    comfort_band: float = eqx.field(static=True, default=1.0)
    model_type: str = eqx.field(static=True, default="RCNetwork")


class BatteryConfig(eqx.Module):
    model_type: Literal["simple", "degradation"] = eqx.field(static=True, default="simple")
    capacity_kwh: float = eqx.field(static=True, default=10.0)
    max_power_kw: float = eqx.field(static=True, default=5.0)
    efficiency: float = eqx.field(static=True, default=0.90)
    degradation_rate_per_cycle: float = eqx.field(static=True, default=0.0001)

    @property
    def capacity_j(self) -> float:
        return self.capacity_kwh * 3.6e6

    @property
    def max_power_w(self) -> float:
        return self.max_power_kw * 1000.0


class RewardConfig(eqx.Module):
    price_weight: float = eqx.field(static=True, default=1.0)
    comfort_weight: float = eqx.field(static=True, default=5.0)


class AirConditionerConfig(eqx.Module):
    model_type: Literal["stateless", "ramping", "variable_cop"] = eqx.field(static=True, default="stateless")
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    min_electrical_power_w: float = eqx.field(static=True, default=500.0)
    tau_thermal_seconds: float = eqx.field(static=True, default=60.0)
    
    cop_cooling: float = eqx.field(static=True, default=3.0)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    
    # Note: Converted Tuples to Arrays for easier JAX interpolation if needed. 
    # If these are strictly lookup keys, they can remain static tuples, 
    # but arrays are more flexible in JAX.
    cop_ambient_temps_c: Array = eqx.field(default_factory=lambda: jnp.array([20.0, 25.0, 30.0, 35.0, 40.0]))
    cop_values_cooling: Array = eqx.field(default_factory=lambda: jnp.array([4.5, 4.0, 3.5, 3.0, 2.5]))


class HeatPumpConfig(eqx.Module):
    model_type: Literal["stateless", "ramping", "variable_cop"] = eqx.field(static=True, default="stateless")
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    min_electrical_power_w: float = eqx.field(static=True, default=500.0)
    tau_thermal_seconds: float = eqx.field(static=True, default=60.0)
    
    cop_heating: float = eqx.field(static=True, default=3.5)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    
    cop_ambient_temps_c: Array = eqx.field(default_factory=lambda: jnp.array([-10.0, 0.0, 10.0, 20.0]))
    cop_values_heating: Array = eqx.field(default_factory=lambda: jnp.array([2.5, 3.0, 3.5, 4.0]))


class ThermalStorageConfig(eqx.Module):
    n_nodes: int = eqx.field(static=True, default=5)
    volume_m3: float = eqx.field(static=True, default=0.3)
    height_m: float = eqx.field(static=True, default=1.5)
    
    max_charge_kw: float = eqx.field(static=True, default=15.0)
    max_discharge_kw: float = eqx.field(static=True, default=15.0)
    
    loss_coeff_w_k: float = eqx.field(static=True, default=2.0)
    ambient_temp_c: float = eqx.field(static=True, default=15.0)
    vertical_conductivity_w_mk: float = eqx.field(static=True, default=0.6)

    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0


class GridThermalStorageConfig(eqx.Module):
    grid_shape: Tuple[int, int, int] = eqx.field(static=True, default=(10, 5, 5)) 
    total_volume_m3: float = eqx.field(static=True, default=0.3)
    height_m: float = eqx.field(static=True, default=1.5)
    
    thermal_conductivity_w_mk: float = eqx.field(static=True, default=0.65)
    convection_conductivity_w_mk: float = eqx.field(static=True, default=50.0)
    loss_coeff_to_ambient_w_m2k: float = eqx.field(static=True, default=0.5)
    ambient_temp_c: float = eqx.field(static=True, default=15.0)

    charge_inlet_idx: Tuple[int, int, int] = eqx.field(static=True, default=(0, 2, 2))
    discharge_outlet_idx: Tuple[int, int, int] = eqx.field(static=True, default=(0, 0, 0))
    
    max_charge_kw: float = eqx.field(static=True, default=15.0)
    max_discharge_kw: float = eqx.field(static=True, default=15.0)

    @property
    def voxel_volume_m3(self) -> float:
        z, y, x = self.grid_shape
        return self.total_volume_m3 / (z * y * x)
    
    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0


class SolarConfig(eqx.Module):
    model_type: Literal["simple", "passthrough", "geometric"] = eqx.field(static=True, default="simple")
    
    panel_area_m2: float = eqx.field(static=True, default=20.0)
    efficiency: float = eqx.field(static=True, default=0.20)
    temp_coefficient: float = eqx.field(static=True, default=-0.004)
    reference_temp_c: float = eqx.field(static=True, default=25.0)
    
    latitude_deg: float = eqx.field(static=True, default=48.13)
    longitude_deg: float = eqx.field(static=True, default=11.58)
    panel_azimuth_deg: float = eqx.field(static=True, default=180.0)
    panel_tilt_deg: float = eqx.field(static=True, default=30.0)

# ==========================================
# 2. Dynamic State Modules
# ==========================================
# We replace @flax_dataclass with eqx.Module.
# These are now Pytrees by definition.

class ThermalState(eqx.Module):
    T_vector: Array

class BatteryState(eqx.Module):
    soc: Array 
    soh: Array

class ThermalStorageState(eqx.Module):
    temperatures_c: Array # (Z, Y, X)
    
    @property
    def soc(self) -> Array:
        # Note: Operations inside properties work, but for JIT-compiled
        # functions, prefer passing the resulting Array, not the property access
        # if it involves heavy logic. Here it is fine.
        avg = jnp.mean(self.temperatures_c)
        return jnp.clip((avg - 30.0) / (60.0 - 30.0), 0.0, 1.0)

class HeatPumpState(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array

class AirConditionerState(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array

class SystemState(eqx.Module):
    thermal: ThermalState
    battery: BatteryState
    storage: ThermalStorageState
    heat_pump: HeatPumpState
    air_conditioner: AirConditionerState

class ExogenousData(eqx.Module):
    # Weather
    ambient_temp: Array
    solar_dni_w_m2: jnp.ndarray       # Direct Normal
    solar_dhi_w_m2: jnp.ndarray       # Diffuse Horizontal
    wind_speed_m_s: Array
    time_of_year_seconds: Array 

    # Price
    price: Array
    carbon_intensity: float

    # Loads (W)
    base_load_w: Array

    # Appliance Profiles
    # For Smart devices, this might be 0.0 (or it will be ignored by the mask).
    appliance_profiles: jnp.ndarray # Shape: (n_appliances,)

class SystemActions(eqx.Module):
    """
    Control Actions (Thermostat & Battery).
    """
    heat_pump_power_w: jnp.ndarray  # Per room (if mapped) or global
    ac_power_w: jnp.ndarray
    battery_power_w: jnp.ndarray    # +Charge, -Discharge
    storage_discharge_w: jnp.ndarray 
    # For Passive devices, the agent should output 0.0 (or it will be ignored by the mask).
    appliance_signals: jnp.ndarray  # Shape: (n_appliances,)

class HeatPumpOutput(eqx.Module):
    thermal_power_w: Array     
    electrical_power_w: Array   

class AirConditionerOutput(eqx.Module):
    thermal_power_w: Array     
    electrical_power_w: Array   

class ThermalStorageOutput(eqx.Module):
    actual_discharge_w: Array    
    rejected_heat_w: Array
    standing_loss_w: Array

class SolarOutput(eqx.Module):
    pv_generation_w: Array