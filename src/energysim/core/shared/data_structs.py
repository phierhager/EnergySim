import jax.numpy as jnp
import equinox as eqx
from dataclasses import field
from typing import Literal, Tuple
import dataclasses
from typing import List, Dict, Optional
import jax.numpy as jnp


# Define array type for clarity
Array = jnp.ndarray


class ApplianceConfig(eqx.Module):
    name: str = eqx.field(static=True)
    target_node_name: str = eqx.field(static=True)
    
    # Electrical Physics
    nominal_power_w: float = eqx.field(static=True)
    # "Efficiency" of heating the room. 
    # 1.0 = Resistive heater / Computer / Lights (100% ends up as heat)
    # 0.0 = Energy leaves system (e.g., water pumped out)
    convective_fraction: float = eqx.field(static=True, default=1.0) 
    
    # Smart Dispatch (Optional)
    cycle_energy_kwh: Optional[float] = eqx.field(static=True, default=None)

    @property
    def total_cycle_energy_j(self) -> Optional[float]:
        if self.cycle_energy_kwh is not None:
            return self.cycle_energy_kwh * 3.6e6
        else:
            return 0.0

class OccupantConfig(eqx.Module):
    name: str = eqx.field(static=True)
    target_node_name: str = eqx.field(static=True)
    
    # Biological Physics
    nominal_heat_w: float = eqx.field(static=True, default=100.0) # ~1 person

class ComponentState(eqx.Module):
    """
    State passed from the Behavioral Model to the Simulator.
    0.0 = Off/Absent, 1.0 = On/Present.
    Can be fractional (e.g., 0.5 speed or 0.5 probability).
    """
    activation: Array  # Shape: (n_appliances,)

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


# ==========================================
# 1. Configuration Modules (Static/Physics)
# ==========================================

# Assuming this code replaces the ThermalConfig class in your data_structs.py (or similar file)

class ThermalConfig(eqx.Module):
    """Configuration for the Thermal Model."""
    A_matrix: Array
    C_inv_vector: Array
    B_matrix: Array
    
    # Node Indices
    ambient_air_index: int = eqx.field(static=True)
    ground_node_index: int = eqx.field(static=True)
    room_air_indices: Tuple[int, ...] = eqx.field(static=True)
    room_rad_indices: Tuple[int, ...] = eqx.field(static=True)
    wall_indices: Tuple[int, ...] = eqx.field(static=True)
    mass_indices: Tuple[int, ...] = eqx.field(static=True)
    waste_heat_node_index: int = eqx.field(static=True, default=-1)

    # Input Mappings
    u_idx_heating: Array
    u_idx_cooling: Array
    air_to_rad_map: Array

    # Mixing & Convection
    # Dynamic Convection
    # Pairs: [Node_A, Node_B]
    convection_pairs: Array = eqx.field(default_factory=lambda: jnp.zeros((0, 2), dtype=jnp.int32))
    
    # [NEW] Convection Type Map
    # 0 = Internal (Natural/HVAC driven), 1 = External (Wind driven)
    # Shape: (N_convection_pairs,)
    convection_types: Array = eqx.field(default_factory=lambda: jnp.zeros(0, dtype=jnp.int32))
    convection_coefficients: Array = eqx.field(default_factory=lambda: jnp.zeros(0))
    mixing_pairs: Array = eqx.field(default_factory=lambda: jnp.zeros((0, 2), dtype=jnp.int32))
    mixing_conductance: Array = eqx.field(default_factory=lambda: jnp.zeros(0))

    # Infiltration (Standard Params - kept for fallback)
    use_dynamic_infiltration: bool = eqx.field(static=True, default=False)
    leakage_area_m2: float = eqx.field(static=True, default=0.05)
    stack_coeff: float = eqx.field(static=True, default=0.12)
    wind_coeff: float = eqx.field(static=True, default=0.09)
    room_vol_m3: float = eqx.field(static=True, default=0.0)

    # --- NEW: Advanced Physics Flags ---
    use_surrogate_convection: bool = eqx.field(static=True, default=False)
    
    # Advanced Physics Flags
    use_airflow_network: bool = eqx.field(static=True, default=False)
    airflow_config: Optional["AirflowConfig"] = eqx.field(static=True, default=None)
    
    use_geometric_shading: bool = eqx.field(static=True, default=False)
    geometry_config: Optional["GeometryConfig"] = eqx.field(static=True, default=None)
    
    # Setpoints
    setpoint: float = eqx.field(static=True, default=21.0)
    comfort_band: float = eqx.field(static=True, default=1.0)
    
    # Helpers for dicts
    node_names: List[str] = eqx.field(static=True)
    node_map: Dict[str, int] = eqx.field(static=True)
    input_map: Dict[str, Dict[str, int]] = eqx.field(static=True)


class AirflowConfig(eqx.Module):
    """
    Topology and Physics for the Nodal Airflow Network.
    Based on COMIS/EnergyPlus AirflowNetwork standards.
    """
    # 1. Topology (Links between Nodes)
    # Nodes 0..N-1 are Rooms. Nodes N..M are External/Boundary.
    # Link i connects node_a[i] <-> node_b[i]
    link_node_a: Array = eqx.field(static=True) 
    link_node_b: Array = eqx.field(static=True)
    
    # 2. Component Physics (Power Law: m = C * dP^n)
    # C_flow: Mass Flow Coefficient [kg/s @ 1Pa]
    link_C_flow: Array = eqx.field(static=True)
    link_exponent: Array = eqx.field(static=True) # Usually 0.65 for cracks, 0.5 for large openings
    
    # 3. Boundary Conditions
    # Indices of links connected to the outdoors
    boundary_link_indices: Array = eqx.field(static=True)
    # Wind Pressure Coefficients (Cp) for boundary links (derived from surface orientation)
    boundary_Cp_coeffs: Array = eqx.field(static=True)
    
    # 4. Node Heights (for Stack Effect)
    node_heights: Array = eqx.field(static=True)
    
    n_internal_nodes: int = eqx.field(static=True)
    n_total_nodes: int = eqx.field(static=True)

class GeometryConfig(eqx.Module):
    """
    Triangulated mesh data for Möller-Trumbore Ray Casting.
    """
    # Obstruction Mesh (External buildings, trees, overhangs)
    # Shape: (N_triangles, 3) - Vertices X,Y,Z
    obs_v0: Array = eqx.field(static=True)
    obs_v1: Array = eqx.field(static=True)
    obs_v2: Array = eqx.field(static=True)
    
    # Target Surface Centroids (Windows/Walls to check shading for)
    # Shape: (N_surfaces, 3)
    surface_centroids: Array = eqx.field(static=True)
    
    # Mapping: Which surface index in ThermalConfig corresponds to which centroid here
    # (Used to map calculated shading factors back to thermal/solar models)
    shading_map_indices: Array = eqx.field(static=True)

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
    
    # --- NEW: Moisture Physics ---
    # Sensible Heat Ratio (SHR): Fraction of cooling that is sensible (temp change).
    # 0.75 means 75% temp drop, 25% water removal.
    sensible_heat_ratio: float = eqx.field(static=True, default=0.75) 

    cop_ambient_temps_c: Array = eqx.field(default_factory=lambda: jnp.array([20.0, 25.0, 30.0, 35.0, 40.0]))
    cop_values_cooling: Array = eqx.field(default_factory=lambda: jnp.array([4.5, 4.0, 3.5, 3.0, 2.5]))

    # --- NEW: Part Load Ratio (PLR) Curve ---
    # Modifier = c0 + c1*PLR + c2*PLR^2
    # Default: 0.85 + 0.15*PLR (Efficiency drops to 85% at low load)
    part_load_curve_coeffs: Array = eqx.field(default_factory=lambda: jnp.array([0.85, 0.15, 0.0]))

class HeatPumpConfig(eqx.Module):
    model_type: Literal["stateless", "ramping", "variable_cop", "mechanistic"] = eqx.field(static=True, default="stateless")
    
    # Standard Params
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    min_electrical_power_w: float = eqx.field(static=True, default=500.0)
    cop_heating: float = eqx.field(static=True, default=3.5)
    
    # Ramping / Variable params
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    tau_thermal_seconds: float = eqx.field(static=True, default=60.0)
    cop_ambient_temps_c: Array = eqx.field(default_factory=lambda: jnp.array([-10.0, 0.0, 10.0, 20.0]))
    cop_values_heating: Array = eqx.field(default_factory=lambda: jnp.array([2.5, 3.0, 3.5, 4.0]))
    part_load_curve_coeffs: Array = eqx.field(default_factory=lambda: jnp.array([0.85, 0.15, 0.0]))

    # --- Mechanistic Params ---
    max_supply_temp_c: float = eqx.field(static=True, default=50.0)
    design_air_flow_m3_s: float = eqx.field(static=True, default=0.15) # per room

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
    model_type: Literal["simple", "passthrough", "geometric", "raycast"] = eqx.field(static=True, default="simple")
    
    panel_area_m2: float = eqx.field(static=True, default=20.0)
    efficiency: float = eqx.field(static=True, default=0.20)
    temp_coefficient: float = eqx.field(static=True, default=-0.004)
    reference_temp_c: float = eqx.field(static=True, default=25.0)
    
    latitude_deg: float = eqx.field(static=True, default=48.13)
    longitude_deg: float = eqx.field(static=True, default=11.58)
    panel_azimuth_deg: float = eqx.field(static=True, default=180.0)
    panel_tilt_deg: float = eqx.field(static=True, default=30.0)

    # --- Static Diffuse Shading ---
    # The portion of the sky dome visible to the panel (0.0 to 1.0).
    # If None, the model calculates the unobstructed view factor based on tilt.
    sky_view_factor: Optional[float] = eqx.field(static=True, default=None)


class MoistureConfig(eqx.Module):
    # Physics
    air_volume_m3: float = eqx.field(static=True)

    # Latent Generation (kg_water/sec per person)
    # ASHRAE: ~0.05 kg/h -> 1.4e-5 kg/s for seated light work
    latent_gen_person_kg_s: float = eqx.field(static=True, default=1.4e-5)
    
    # Initial State
    initial_rel_humidity: float = eqx.field(static=True, default=0.5)

    model_type: Literal["dynamic", "empd"] = eqx.field(static=True, default="dynamic")

    # EMPD Parameters (Defaults for Gypsum/Plaster)
    # Effective Moisture Penetration Depth (m)
    empd_thickness: float = eqx.field(static=True, default=0.02) 
    # Surface area available for buffering (m2) (Usually ~3x floor area)
    buffer_area_m2: float = eqx.field(static=True, default=50.0)
    # Density of buffer material (kg/m3)
    material_density: float = eqx.field(static=True, default=800.0)
    # Slope of sorption curve (kg_moisture/kg_material per unit dPhi)
    sorption_slope: float = eqx.field(static=True, default=0.05)
    
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

class MoistureState(eqx.Module):
    # We track Humidity Ratio (w), NOT Relative Humidity.
    # w is conserved mass; RH fluctuates wildly with temperature.
    humidity_ratio_kg_kg: Array # Shape (n_rooms,)
    buffer_moisture_content: Array # Shape (n_rooms,)


class SystemState(eqx.Module):
    thermal: ThermalState
    battery: BatteryState
    storage: ThermalStorageState
    heat_pump: HeatPumpState
    air_conditioner: AirConditionerState
    moisture: MoistureState

# =============================================================================
# CLEANED INPUTS (Actions & Exogenous)
# =============================================================================

class ExogenousData(eqx.Module):
    # Weather
    ambient_temp: Array
    solar_dni_w_m2: Array
    solar_dhi_w_m2: Array
    wind_speed_m_s: Array
    time_of_year_seconds: Array
    price: Array
    
    # Profiles for PASSIVE Machines (Base Load, Fridge, Lights)
    # Shape: (Time, N_passive_machines)
    passive_machine_profiles: Array 

    # Profiles for SMART Machines (EVs, Washers)
    # Shape: (Time, N_smart_machines)
    smart_device_availability: Array
    
    # Profiles for OCCUPANTS (People Presence)
    # Shape: (Time, N_occupants)
    occupant_profiles: Array

    # --- Dynamic Beam Shading ---
    # 1.0 = Unshaded, 0.0 = Fully blocked by horizon/obstacles
    # Scalar for the PV system
    pv_shading_factor: Array = eqx.field(default_factory=lambda: jnp.array(1.0))

    # Vector for the building surfaces (Walls/Windows)
    # Shape matches the order of surfaces in the simulator
    surface_shading_factors: Array = eqx.field(default_factory=lambda: jnp.array([]))
    
    
    carbon_intensity: float = 0.0

class SystemActions(eqx.Module):
    """
    Control Actions.
    Clean Separation: Only Smart/Controllable inputs are here.
    """
    heat_pump_power_w: Array
    ac_power_w: Array
    battery_power_w: Array
    storage_discharge_w: Array
    
    # Controls for Smart Appliances (EVs, Washers)
    # Shape: (N_smart_appliances,)
    # If there are no smart appliances, this can be size 0.
    smart_appliance_signals: Array 


class HeatPumpOutput(eqx.Module):
    thermal_power_w: Array     
    electrical_power_w: Array   
    
class AirConditionerOutput(eqx.Module):
    thermal_power_w: Array      # Sensible cooling (Negative Watts)
    electrical_power_w: Array   # Electricity consumed (Positive Watts)
    water_removed_kg_s: Array   # Moisture removal rate (Positive kg/s)

class ThermalStorageOutput(eqx.Module):
    actual_discharge_w: Array    
    rejected_heat_w: Array
    standing_loss_w: Array

class SolarOutput(eqx.Module):
    pv_generation_w: Array