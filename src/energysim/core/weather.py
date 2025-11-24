from enum import Enum
import equinox as eqx
import jax.numpy as jnp

# --- 1. The Data Structure (The "Interface") ---
class WeatherData(eqx.Module):
    """
    Central payload for environment data. 
    The simulation engine only cares about THIS class, 
    not about .epw files or CSV parsing.
    """
    # Time
    time_seconds: jnp.ndarray  # Continuous seconds from start of year
    
    # Air State
    dry_bulb_temp: jnp.ndarray  # deg C
    dew_point_temp: jnp.ndarray # deg C
    relative_humidity: jnp.ndarray # %
    atmospheric_pressure: jnp.ndarray # Pa
    
    # Radiation 
    ghi: jnp.ndarray # Global Horizontal (W/m2)
    dni: jnp.ndarray # Direct Normal (W/m2)
    dhi: jnp.ndarray # Diffuse Horizontal (W/m2)
    
    # Wind
    wind_speed: jnp.ndarray # m/s
    wind_direction: jnp.ndarray # Degrees
    
    # Derived / Calculated Fields
    sky_temp: jnp.ndarray # deg C 
    ground_temp: jnp.ndarray # deg C

def get_ground_temperature(ground_lag_days: float, ground_avg_temp_c: float, ground_amplitude_c: float, time_seconds: float) -> float:
    """
    Calculates ground temperature using a sinusoidal lag model (Kasuda & Achenbach approximation).
    T_ground = T_avg - Amplitude * cos(2pi * (day - shift) / 365)
    """
    seconds_in_year = 365.0 * 24.0 * 3600.0
    # Align phase: cos wave peaks in summer, but ground lags. 
    # Standard cosine peaks at 0, we want peak around day 200 (summer).
    # Shift for air temp usually ~15-20 days delay from solstice. Ground adds ~30-90 days.
    phase_shift_sec = ground_lag_days * 24.0 * 3600.0

    # Calculate simple annual oscillation
    oscillation = jnp.cos(2 * jnp.pi * (time_seconds - phase_shift_sec) / seconds_in_year)
    
    # Ground is colder in spring, warmer in autumn compared to air
    return ground_avg_temp_c - (ground_amplitude_c * oscillation)

def calculate_sky_temperature(dry_bulb_c: jnp.ndarray,
                              dew_point_c: jnp.ndarray) -> jnp.ndarray:
    """
    Clear-sky effective sky temperature using Brunt's equation,
    properly including Stefan-Boltzmann constant.
    """
    sigma = 5.670374419e-8  # W/m²·K⁴

    t_air_k = dry_bulb_c + 273.15
    # Vapor pressure from dew point in kPa
    e_a = 0.61078 * jnp.exp(17.269 * dew_point_c / (dew_point_c + 237.3))

    # Brunt parameters for emissivity
    a = 0.552
    b = 0.065
    epsilon_sky = a + b * jnp.sqrt(e_a)

    # Longwave radiation from sky
    L_sky = epsilon_sky * sigma * t_air_k**4

    # Convert radiation back to effective sky temperature
    t_sky_k = (L_sky / sigma) ** 0.25

    return t_sky_k - 273.15
