import jax.numpy as jnp
from .coefficients import Coefficients

def calculate_saturation_pressure(T_celsius: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates saturation vapor pressure (Psat) in Pascals using the Magnus formula.
    Accurate range: -20°C to 50°C.
    """
    # constants from ASHRAE
    return 611.2 * jnp.exp((17.67 * T_celsius) / (T_celsius + 243.5))

def calculate_humidity_ratio(T_celsius: jnp.ndarray, rel_hum_0_1: jnp.ndarray, pressure_pa: jnp.ndarray) -> jnp.ndarray:
    """
    Converts T, RH, and Pressure -> Humidity Ratio (w) [kg_water / kg_dry_air].
    """
    P_sat = calculate_saturation_pressure(T_celsius)
    P_vap = rel_hum_0_1 * P_sat
    
    # w = 0.622 * P_vap / (P_tot - P_vap)
    # 0.622 is ratio of molecular weight of water to air
    return 0.622 * P_vap / (pressure_pa - P_vap)

def calculate_relative_humidity(T_celsius: jnp.ndarray, w_kg_kg: jnp.ndarray, pressure_pa: jnp.ndarray) -> jnp.ndarray:
    """
    Converts T, w, Pressure -> Relative Humidity (0.0 - 1.0).
    Used for checking comfort and condensation risks.
    """
    P_sat = calculate_saturation_pressure(T_celsius)
    
    # Invert the w equation: P_vap = (w * P_tot) / (0.622 + w)
    P_vap = (w_kg_kg * pressure_pa) / (0.622 + w_kg_kg)
    
    rh = P_vap / P_sat
    return jnp.clip(rh, 0.0, 1.0)

def calculate_air_density(T_celsius: jnp.ndarray, pressure_pa: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates moist air density using Ideal Gas Law.
    rho = P / (R_specific * T_kelvin)
    """
    R_specific_air = 287.058
    T_kelvin = T_celsius + 273.15
    return pressure_pa / (R_specific_air * T_kelvin)