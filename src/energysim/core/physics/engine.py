
# physics/engine.py
import jax.numpy as jnp
from .constants import CONSTANTS
from ..shared.data_structs import ExogenousData, EnvironmentalContext, ThermalConfig, SiteProperties, SolarCache, SurfaceMap

# Import physics from specific modules to keep engine clean
from .functions import calculate_ground_temperature, get_sun_position
from .psychrometrics import calculate_air_density, calculate_relative_humidity
# Assuming you move sky temp to functions.py or keep it here if it's unique
from .functions import calculate_sky_temperature_bliss 


def compute_environmental_context(
    timestep_idx: int,
    exo: ExogenousData,          # Sliced for CURRENT timestep (scalars)
    solar_cache: SolarCache,     # Full arrays
    t_config: ThermalConfig,
    site: SiteProperties,
    surface_map: SurfaceMap
) -> EnvironmentalContext:
    """
    Derives the physical boundary conditions for the current timestep.
    High-Fidelity: Uses pre-computed solar geometry and dynamic thermodynamics.
    """
    
    # --- 1. Thermodynamics ---
    # Dynamic air density (Standard Model)
    rho_air = calculate_air_density(exo.ambient_temp, exo.atmospheric_pressure)
    
    # Ground temperature (Kasuda & Achenbach)
    grnd_t = calculate_ground_temperature(exo.time_of_year_seconds, site)
    
    # Sky temperature (Bliss correlation)
    sky_t = calculate_sky_temperature_bliss(exo.ambient_temp, exo.relative_humidity)

    # --- 2. Solar Geometry (Lookup) ---
    # Fast array lookup from GPU cache
    current_sun_vec = solar_cache.sun_direction_vectors[timestep_idx]
    current_shading = solar_cache.surface_shading_factors[timestep_idx]
    
    # Optional: If you need Incident Angle Modifiers for PV/Windows
    # current_iam = solar_cache.incident_angle_modifiers[timestep_idx]

    # --- 3. Wind Pressure (Bernoulli) ---
    wind_p_boundary = jnp.array([])
    
    if t_config.airflow_config is not None:
        af = t_config.airflow_config
        # Dynamic pressure q = 0.5 * rho * v^2
        q_wind = 0.5 * rho_air * (exo.wind_speed_m_s ** 2)
        
        # P_boundary = q_wind * Cp
        wind_p_boundary = q_wind * af.boundary_Cp_coeffs

    return EnvironmentalContext(
        exo=exo,
        sun_vector=current_sun_vec,
        surface_shading_factors=current_shading,
        pv_shading_factor=current_shading[surface_map.pv_surface_index],
        sky_temp_c=sky_t,
        ground_temp_c=grnd_t,
        wind_pressure_boundary=wind_p_boundary,
        air_density=rho_air
    )