import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ExogenousData

@eqx.filter_jit
def calculate_sherman_grimsrud_flow(
    config: ThermalConfig, 
    T_rooms: jnp.ndarray, 
    exo: ExogenousData
) -> jnp.ndarray:
    """
    Calculates standard infiltration Mass Flow (kg/s) using the Orifice approximation.
    """
    if not config.use_dynamic_infiltration:
        return jnp.zeros_like(T_rooms)

    delta_T = jnp.abs(exo.ambient_temp - T_rooms)
    v_wind_sq = exo.wind_speed_m_s ** 2

    # Physics: Pressure ~ Stack + Wind
    pressure_term = (config.stack_coeff * delta_T) + (config.wind_coeff * v_wind_sq)
    flow_factor = jnp.sqrt(pressure_term + 1e-6)

    # Volumetric Flow (m3/s)
    vol_flow_m3_s = config.leakage_area_m2 * flow_factor
    
    # Convert to Mass Flow (kg/s)
    # Assume reference density for the flow calculation to align with coefficients
    rho_ref = 1.204
    mass_flow_kg_s = vol_flow_m3_s * rho_ref
    
    return mass_flow_kg_s