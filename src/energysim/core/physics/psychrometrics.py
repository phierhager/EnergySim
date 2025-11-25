# physics/psychrometrics.py
import jax.numpy as jnp
from .constants import CONSTANTS

def calculate_saturation_pressure(T_celsius: jnp.ndarray) -> jnp.ndarray:
    return 611.2 * jnp.exp((17.67 * T_celsius) / (T_celsius + 243.5))

def calculate_humidity_ratio(T_celsius: jnp.ndarray, rel_hum_0_1: jnp.ndarray, pressure_pa: jnp.ndarray) -> jnp.ndarray:
    P_sat = calculate_saturation_pressure(T_celsius)
    P_vap = rel_hum_0_1 * P_sat
    return 0.622 * P_vap / (pressure_pa - P_vap)

def calculate_relative_humidity(T_celsius: jnp.ndarray, w_kg_kg: jnp.ndarray, pressure_pa: jnp.ndarray) -> jnp.ndarray:
    P_sat = calculate_saturation_pressure(T_celsius)
    P_vap = (w_kg_kg * pressure_pa) / (0.622 + w_kg_kg)
    rh = P_vap / P_sat
    return jnp.clip(rh, 0.0, 1.0)

def calculate_air_density(T_celsius: jnp.ndarray, pressure_pa: jnp.ndarray) -> jnp.ndarray:
    """Calculates moist air density using Ideal Gas Law."""
    T_kelvin = T_celsius + CONSTANTS.T_ABS_ZERO
    return pressure_pa / (CONSTANTS.GAS_CONSTANT_AIR * T_kelvin)