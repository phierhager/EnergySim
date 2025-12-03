import jax.numpy as jnp
import equinox as eqx
from dataclasses import field
from typing import Literal, Tuple
import dataclasses
from typing import List, Dict, Optional
import jax.numpy as jnp


# Define array type for clarity
Array = jnp.ndarray

# --- Global Constants for Padding ---
MAX_ROOMS = 5
MAX_MACHINES = 10


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

class ThermalConfig(eqx.Module):
    """Configuration for the Thermal Model."""
    # State-Space Matrices
    A_matrix: Array
    C_inv_vector: Array
    B_matrix: Array

    # --- High-Fidelity Radiosity Fields ---
    # Indices of nodes involved in radiation (walls, floor, ceiling)
    # Shape: (N_surfaces,)
    surface_node_indices: Array 
    
    # Pre-computed Radiative Conductance Matrix
    # G_ij = Sigma * Area_i * F_ij * Eps_eff
    # Shape: (N_surfaces, N_surfaces)
    rad_conductance_matrix: Array 

    # Standard Indices
    ambient_air_index: int = eqx.field(static=True)
    ground_node_index: int = eqx.field(static=True)
    
    # Groupings
    room_air_indices: Tuple[int, ...] = eqx.field(static=True)
    room_rad_indices: Tuple[int, ...] = eqx.field(static=True)
    wall_indices: Tuple[int, ...] = eqx.field(static=True)
    mass_indices: Tuple[int, ...] = eqx.field(static=True)

    # Input Mappings
    u_idx_heating: Array
    u_idx_cooling: Array
    air_to_rad_map: Array

    # Dynamic Physics Arrays
    convection_pairs: Array 
    convection_coefficients: Array 
    convection_types: Array 
    mixing_pairs: Array 
    mixing_conductance: Array 


    # Metadata
    node_names: List[str] = eqx.field(static=True)
    node_map: Dict[str, int] = eqx.field(static=True)
    input_map: Dict[str, Dict[str, int]] = eqx.field(static=True)

    # Parameters
    waste_heat_node_index: int = eqx.field(static=True, default=-1)
    use_dynamic_infiltration: bool = eqx.field(static=True, default=False)
    leakage_area_m2: float = eqx.field(static=True, default=0.05)
    stack_coeff: float = eqx.field(static=True, default=0.12)
    wind_coeff: float = eqx.field(static=True, default=0.09)
    room_vol_m3: float = eqx.field(static=True, default=0.0)
    setpoint: float = eqx.field(static=True, default=21.0)
    comfort_band: float = eqx.field(static=True, default=1.0)

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
    # --- Basic Specs ---
    capacity_kwh: float = eqx.field(static=True, default=10.0)
    capacity_ah: Optional[float] = eqx.field(static=True, default=None) # Model checks this in _capacity_as
    max_power_kw: float = eqx.field(static=True, default=5.0)
    efficiency: float = eqx.field(static=True, default=0.90)
    degradation_rate_per_cycle: float = eqx.field(static=True, default=0.0001)

    # --- Pack Architecture ---
    n_series: int = eqx.field(static=True, default=110) # Used to scale OCV/R0 grids

    # --- Voltage Limits (for CC-CV logic) ---
    v_min: float = eqx.field(static=True, default=300.0)
    v_max: float = eqx.field(static=True, default=430.0)

    # --- Current Constraints ---
    max_charge_current_a: Optional[float] = eqx.field(static=True, default=None)
    max_discharge_current_a: Optional[float] = eqx.field(static=True, default=None)

    # --- Thermal Dynamics ---
    C_core: float = eqx.field(static=True, default=5e4)     # Thermal mass (J/K) core
    C_case: float = eqx.field(static=True, default=1e4)     # Thermal mass (J/K) case
    R_core_case: float = eqx.field(static=True, default=0.1)      # Thermal resistance core->case (K/W)
    R_case_ambient: float = eqx.field(static=True, default=0.5)     # Thermal resistance case->ambient (K/W)

    ambient_temp: float = eqx.field(static=True, default=298.15) # Default ambient (K)
    ref_temp_k: float = eqx.field(static=True, default=298.15)   # Reference temp for Arrhenius equation
    activation_energy_r0_k: float = eqx.field(static=True, default=2000.0)  # Activation energy (J/mol)
    activation_energy_rc_k: float = eqx.field(static=True, default=2000.0)  # Activation energy (J/mol)
    hysteresis_rate: float = eqx.field(static=True, default=50.0)  # Hysteresis rate constant


    @property
    def capacity_j(self) -> float:
        """Calculates capacity in Joules from kWh."""
        return self.capacity_kwh * 3.6e6

    @property
    def max_power_w(self) -> float:
        """Calculates max power in Watts."""
        return self.max_power_kw * 1000.0


class RewardConfig(eqx.Module):
    price_weight: float = eqx.field(static=True, default=1.0)
    comfort_weight: float = eqx.field(static=True, default=5.0)

class CompressorConfig(eqx.Module):
    """
    Physical parameters defining the compressor's volumetric and isentropic behavior.
    """
    # 1. Volumetric Limits
    # The maximum theoretical capacity (W/K) the compressor can move at 100% speed.
    # Think of this as (Displacement Volume * RPM * Density_Ref).
    max_displacement_w_per_k: float = 120.0 

    # 2. Isentropic Efficiency Curve Parameters (Grasso's Polynomial / Empirical)
    # The temperature difference (Cond - Evap) where efficiency peaks
    design_lift_k: float = 30.0    
    
    # The speed ratio (0.0 to 1.0) where efficiency peaks
    design_speed_ratio: float = 0.6 
    
    # The maximum possible isentropic efficiency (0.0 to 1.0)
    eta_peak: float = 0.70         
    
    # Sensitivity parameters (Shape of the efficiency hill)
    k_lift: float = 2.5              # Sensitivity to lift deviation
    k_speed: float = 1.8           # Sensitivity to speed deviation

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

    motor_eff_curve_coeffs: Array = eqx.field(default=jnp.array([0.85, 0.1, -0.1]))

    rated_airflow_kg_s: float = 0.5  # Approx 400 CFM for 1 ton
    coil_bypass_factor: float = 0.15 # Typical for residential DX
    coil_adp_c: float = 8.0          # Apparatus Dew Point (Coil Temp)

    # Physics Parameters (Same as HP)
    compressor: CompressorConfig = CompressorConfig()
    
    # Heat Exchanger Sizing
    ua_condenser_nominal: float = 400.0  # W/K (Outdoor coil)
    ua_evaporator_nominal: float = 350.0 # W/K (Indoor coil)
    design_air_flow_m3_s: float = 0.5    # Rated Fan Speed
    
    # We keep these for the "Passthrough" model, 
    # but Mechanistic model ignores them:
    cop_cooling: float = 3.0

class HeatPumpConfig(eqx.Module):
    # --- A. System Sizing ---
    # Total electrical draw allowed
    max_electrical_power_w: float 
    min_electrical_power_w: float 
    
    # Dynamics: How fast the compressor can spin up (Inertia)
    ramp_rate_w_per_sec: float = 50.0

    # --- B. Fluid Dynamics (The Generic Interface) ---
    # SINK (Indoor)
    # For Air-to-Air: use ~1005 J/kgK and ~1.2 kg/m3
    # For Air-to-Water: use ~4186 J/kgK and ~997 kg/m3 (or glycol vals)
    sink_specific_heat_j_kgk: float 
    sink_fluid_density_kg_m3: float
    design_sink_flow_m3_s: float     # e.g., Water flow rate or Indoor Fan airflow

    # SOURCE (Outdoor)
    # Almost always Air for this model class
    design_source_air_flow_m3_s: float 

    # --- C. Heat Exchangers (UA Values) ---
    # Conductance (Watts/Kelvin). Higher = Closer approach temps = Better COP.
    ua_condenser_nominal: float
    ua_evaporator_nominal: float

    # --- D. Motor & Drive ---
    # Polynomial coefficients [c0, c1, c2] for Inverter Efficiency
    # eff = c0 + c1*plr + c2*plr^2
    motor_eff_curve_coeffs: Array 

    # --- E. Sub-Components ---
    compressor: CompressorConfig
    
    # --- F. Logic Flags ---
    enable_defrost: bool = True

    # -----------------------------------------------------------------------
    # Factory Methods (Builders)
    # -----------------------------------------------------------------------

    @classmethod
    def create_air_to_air(
        cls, 
        max_kw: float, 
        n_rooms: int = 1,
        quality: str = 'standard' # 'standard' or 'high_efficiency'
    ):
        """Creates a configuration for a standard Split AC / Heat Pump."""
        
        total_w = max_kw * 1000.0
        
        # Rule of Thumb Sizing
        # Airflow: ~0.05 m3/s per kW of capacity
        airflow = 0.05 * max_kw 
        
        # UA Sizing: High Efficiency units have larger heat exchangers
        ua_factor = 250.0 if quality == 'high_efficiency' else 180.0
        ua_val = ua_factor * max_kw

        return cls(
            max_electrical_power_w=total_w,
            min_electrical_power_w=total_w * 0.15, # 15% turndown
            
            # Physics: Air Properties
            sink_specific_heat_j_kgk=1005.0,
            sink_fluid_density_kg_m3=1.204,
            design_sink_flow_m3_s=airflow,       # Indoor Unit Fan
            design_source_air_flow_m3_s=airflow * 1.5, # Outdoor Fan is usually larger
            
            ua_condenser_nominal=ua_val,
            ua_evaporator_nominal=ua_val,
            
            # Standard Inverter Curve (Peak ~95% at 50% load)
            motor_eff_curve_coeffs=jnp.array([0.85, 0.25, -0.15]),
            
            compressor=CompressorConfig(
                max_displacement_w_per_k=15.0 * max_kw, # Scales with size
                eta_peak=0.75 if quality == 'high_efficiency' else 0.68
            )
        )

    @classmethod
    def create_air_to_water(
        cls, 
        max_kw: float, 
        glycol_percent: float = 0.0
    ):
        """Creates a configuration for a Hydronic Heat Pump."""
        
        total_w = max_kw * 1000.0
        
        # Water/Glycol Physics adjustments
        # Pure Water: 4186 J/kgK, 997 kg/m3
        cp_water = 4186.0 - (glycol_percent * 10.0)
        rho_water = 997.0 + (glycol_percent * 2.0)
        
        # Water flow is usually lower volume than air for same energy (higher density)
        # DeltaT = 5K target -> Flow = Q / (rho*cp*5)
        # Approx 0.17 kg/s for 3.5kW -> ~0.05 m3/h per kW? 
        # Rule of thumb: 0.25 m3/h per kW -> 0.00007 m3/s per kW
        water_flow = (0.25 / 3600.0) * (max_kw * 1000 / 5) * 5 # Rough sizing
        
        ua_val = 300.0 * max_kw # Water HEX usually has better heat transfer

        return cls(
            max_electrical_power_w=total_w,
            min_electrical_power_w=total_w * 0.10, # Water pumps can often turn down lower
            
            # Physics: Water Properties
            sink_specific_heat_j_kgk=cp_water,
            sink_fluid_density_kg_m3=rho_water,
            design_sink_flow_m3_s=water_flow,
            
            # Source is still Air
            design_source_air_flow_m3_s=0.08 * max_kw, 
            
            ua_condenser_nominal=ua_val, # Plate heat exchanger
            ua_evaporator_nominal=ua_val * 0.8, # Air coil
            
            motor_eff_curve_coeffs=jnp.array([0.90, 0.15, -0.08]), # Better motors
            
            compressor=CompressorConfig(
                max_displacement_w_per_k=18.0 * max_kw,
                eta_peak=0.72
            )
        )

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

    # Thermal Physics (Faiman Model)
    thermal_u0: float = 25.0  # W/m²K (Heat transfer const). Ground=25, Roof=15
    thermal_u1: float = 6.84  # W/m²K per m/s (Wind cooling). Ground=6.84, Roof=1.2
    
    # Optical Physics
    iam_b0: float = 0.05      # IAM coeff. Standard=0.05, Anti-Reflective=0.03
    albedo: float = 0.2       # Ground reflection coefficient

class GroundPhysicsConfig(eqx.Module):
    """Parameters for the Kasuda-Achenbach ground model."""
    average_soil_temp_c: float = eqx.field(static=True) # Annual average air temp
    amplitude_diff_c: float = eqx.field(static=True)    # Annual max - avg
    phase_lag_days: float = eqx.field(static=True, default=40.0) # Depends on soil density

class SurfaceMap(eqx.Module):
    """Bridges Geometry (Ray Tracing) to Thermal (Heat Balance)."""
    # Index in GeometryConfig.surface_centroids -> Index in ThermalConfig.wall_indices
    geo_to_thermal_indices: Array = eqx.field(static=True)
    
    # Which geometric surface index corresponds to the PV panel?
    pv_surface_index: int = eqx.field(static=True)


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
# DAE INTERFACE STRUCTURES
# ==========================================

class BatteryState(eqx.Module):
    soc: Array            # Scalar
    temp_core_k: Array         # Scalar (Kelvin)
    temp_case_k: Array    # Scalar (Kelvin)
    v_rc: Array           # Shape (n_rc,)
    v_hyst: Array           # Hysteresis voltage state [Volts]
    
class AirConditionerState(eqx.Module):
    electrical_power_w: Array  # Current electrical power draw (W)

class HeatPumpState(eqx.Module):
    electrical_power_w: Array  # Current electrical power draw (W)

# TODO: Update after implementing local states
class DifferentialState(eqx.Module):
    """
    The subset of SystemState that evolves continuously via ODEs.
    """
    # Building
    T_vector: Array            # (N_nodes,)
    
    # Storage
    storage_T: Array           # 1D: (N_nodes,), 3D: (Z, Y, X)
    
    # Battery
    battery: Array         # Scalar
    
    # HVAC Inertia (Ramping)
    hp_power_state: Array      # Scalar (Current Compressor Watts)
    ac_power_state: Array      # Scalar (Current Compressor Watts)
    
    # Moisture
    moisture_w: Array          # (N_rooms,)
    moisture_buffer_u: Array   # (N_rooms,)


class ThermalInputs(eqx.Module):
    """Inputs required to calculate Building derivatives."""
    Q_solar: Array    # (N_nodes,)
    Q_internal: Array # (N_nodes,) Occupants + Passive Machines
    Q_hvac: Array     # (N_nodes,) Heating/Cooling from HP/AC
    T_ambient: Array  # Scalar
    wind_speed: Array # Scalar


class StorageInputs(eqx.Module):
    """Inputs required to calculate Tank derivatives."""
    charge_power_w: Array    # Scalar (Heat from HP)
    discharge_power_w: Array # Scalar (Heat to House)
    T_inlet_c: Array         # Scalar (Temp coming from HP)
    T_return_c: Array        # Scalar (Temp coming from House)
    T_ambient: Array         # Scalar

class AirConditionerInputs(eqx.Module):
    """Inputs required to calculate AC derivatives."""
    target_power_w: Array  # Requested Power (Positive = Cooling)

class HeatPumpInputs(eqx.Module):
    target_power_w: Array  # Requested Power (Positive = Heating)

class BatteryInputs(eqx.Module):
    power_w: Array       # Requested Power (Positive = Charge)
    T_ambient_c: Array   # Ambient temperature for thermal cooling

class MoistureInputs(eqx.Module):
    T_room_c: Array                   # Room air temperature
    hvac_moisture_removal_kg_s: Array # Dehumidification rate (positive = removal)
    n_occupants: Array                # Number of people
    infiltration_flow_m3_s: Array     # Air exchange rate
    atmospheric_pressure: Array      # Ambient atmospheric pressure
    ambient_temp: Array            # Ambient temperature
    relative_humidity: Array       # Ambient relative humidity

# =============================================================================
# CLEANED INPUTS (Actions & Exogenous)
# =============================================================================

class ExogenousData(eqx.Module):
    """
    IMMUTABLE. Pure external data loaded from files or predicted by forecasters.
    Contains NO simulation-specific or geometry-specific calculated fields.
    """
    # Time
    time_of_year_seconds: Array
    
    # Weather
    ambient_temp: Array
    solar_dni_w_m2: Array
    solar_dhi_w_m2: Array
    wind_speed_m_s: Array
    relative_humidity: Array
    atmospheric_pressure: Array
    
    # Grid
    price: Array
    carbon_intensity: Array
    
    # Usage Profiles
    base_load_w: Array

    # Shape: (Time, Max_Rooms)
    occupant_profiles: Array 
    
    # Shape: (Time, Max_Passive_Machines)
    passive_machine_profiles: Array
    
    # Shape: (Time, Max_Smart_Machines)
    smart_device_availability: Array


class EnvironmentalContext(eqx.Module):
    """
    DERIVED. Calculated by the Physics Engine for a specific timestep.
    Contains everything the models need to run that depends on Geometry + Weather.
    """
    # Reference to raw data (so models can access Temp/Price directly)
    exo: ExogenousData 
    
    # Derived Geometry (The result of Ray-Casting)
    sun_vector: Array            # (3,)
    surface_shading_factors: Array # (N_surfaces,)
    pv_shading_factor: Array       # Scalar
    
    # Derived Physics
    sky_temp_c: Array
    ground_temp_c: Array
    
    # Wind Pressure (for Airflow Network)
    wind_pressure_boundary: Array # (N_boundary_nodes,)

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
    thermal_power_w: float
    electrical_power_w: float
    cop: float
    supply_temp_c: float
    condenser_temp_c: float
    evaporator_temp_c: float
    eta_second_law: float
    volumetric_limit_hit: bool

class AirConditionerOutput(eqx.Module):
    thermal_power_w: Array      # Sensible cooling (Negative Watts)
    electrical_power_w: Array   # Electricity consumed (Positive Watts)
    water_removed_kg_s: Array   # Moisture removal rate (Positive kg/s)
    volumetric_limit_hit: Array # New Flag (0.0 or 1.0)

class ThermalStorageOutput(eqx.Module):
    actual_discharge_w: Array    
    rejected_heat_w: Array
    standing_loss_w: Array

class SolarOutput(eqx.Module):
    pv_generation_w: Array

class BatteryOutput(eqx.Module):
    actual_power_w: Array    # Power actually delivered/absorbed
    voltage_v: Array         # Terminal Voltage
    current_a: Array         # Current (Positive = Charge)
    heat_generation_w: Array # Joule heating + Entropic heat


class SiteProperties(eqx.Module):
    """
    Site-specific physical properties. 
    These must come from the .epw header or configuration, NOT defaults.
    """
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    ground_avg_temp_c: float     # Annual average
    ground_amplitude_c: float    # Seasonal swing amplitude
    ground_reflectivity: float = 0.2


class SolarCache(eqx.Module):
    """
    Pre-computed high-fidelity solar dynamics.
    Loaded onto GPU once; accessed via index during simulation.
    Shape: (8760, ...)
    """
    sun_direction_vectors: jnp.ndarray  # Shape: (Steps, 3)
    surface_shading_factors: jnp.ndarray # Shape: (Steps, N_surfaces)
    incident_angle_modifiers: jnp.ndarray # Shape: (Steps, N_surfaces) - Optional optimization