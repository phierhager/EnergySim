# physics/functions.py
import jax.numpy as jnp
from ..shared.data_structs import SiteProperties

# --- Geometry & Optical Physics ---

def get_sun_position(time_seconds: float, latitude_deg: float = 48.0) -> jnp.ndarray:
    """Calculates sun position vector [x, y, z] using ASHRAE model."""
    day_seconds = 86400.0
    year_seconds = 365.0 * day_seconds
    day_progress = (time_seconds % day_seconds) / day_seconds
    year_progress = time_seconds / year_seconds
    
    hour_angle = (day_progress - 0.5) * 2 * jnp.pi
    declination = jnp.radians(23.45) * -jnp.cos(2 * jnp.pi * (year_progress + 10.0/365.0))
    lat_rad = jnp.radians(latitude_deg)
    
    sin_elev = (jnp.sin(lat_rad) * jnp.sin(declination) +
                jnp.cos(lat_rad) * jnp.cos(declination) * jnp.cos(hour_angle))
    
    elev = jnp.arcsin(jnp.clip(sin_elev, -1.0, 1.0))
    
    cos_az = (jnp.sin(declination) - jnp.sin(elev) * jnp.sin(lat_rad)) / \
             (jnp.cos(elev) * jnp.cos(lat_rad) + 1e-6)
             
    azimuth = jnp.arccos(jnp.clip(cos_az, -1.0, 1.0))
    
    z = jnp.sin(elev)
    y = jnp.cos(elev) * jnp.cos(azimuth)
    x_sign = jnp.sign(hour_angle)
    x = x_sign * jnp.cos(elev) * jnp.sin(azimuth)
    
    sun_vec = jnp.array([x, y, z])
    is_day = z > 0
    return jnp.where(is_day, sun_vec / (jnp.linalg.norm(sun_vec) + 1e-6), jnp.zeros(3))

def calculate_iam_polynomial(cos_theta: jnp.ndarray) -> jnp.ndarray:
    """Calculates Incidence Angle Modifier (ASHRAE Polynomial)."""
    cos_theta_clamped = jnp.clip(cos_theta, 0.0, 1.0)
    theta_rad = jnp.arccos(cos_theta_clamped)
    theta_deg = jnp.degrees(theta_rad)
    
    c0, c1 = 1.0, -2.2604e-3
    c2, c3, c4 = 1.0963e-4, -4.5913e-6, 7.1685e-8
    
    val = c0 + c1*theta_deg + c2*(theta_deg**2) + c3*(theta_deg**3) + c4*(theta_deg**4)
    return jnp.clip(val, 0.0, 1.0)

# --- Environmental Physics ---

def calculate_ground_temperature(time_seconds: float, site: SiteProperties) -> float:
    """Kasuda & Achenbach ground temperature model."""
    seconds_in_year = 365.0 * 24.0 * 3600.0
    lag_seconds = 40.0 * 24.0 * 3600.0 # Approx 40 days lag
    
    oscillation = jnp.cos(2 * jnp.pi * (time_seconds - lag_seconds) / seconds_in_year)
    return site.ground_avg_temp_c - (site.ground_amplitude_c * oscillation)

def calculate_sky_temperature_bliss(dry_bulb_c: jnp.ndarray, rh: jnp.ndarray) -> jnp.ndarray:
    """Bliss correlation for Sky Temperature."""
    t_k = dry_bulb_c + 273.15
    # Magnus formula for Dew Point
    alpha = 17.27 * dry_bulb_c / (dry_bulb_c + 237.7) + jnp.log(rh + 1e-6)
    t_dp = (237.7 * alpha) / (17.27 - alpha)
    
    # Emissivity correlation
    e_sky = 0.741 + 0.0062 * t_dp
    t_sky_k = t_k * (e_sky ** 0.25)
    return t_sky_k - 273.15

def calculate_sky_temperature_swinbank(dry_bulb_c: jnp.ndarray) -> jnp.ndarray:
    """
    Swinbank (1963) model. Simpler than Brunt, often sufficiently accurate for
    clear sky temperature if cloud cover data is missing.
    T_sky = 0.0552 * T_amb^1.5 (in Kelvin)
    """
    t_k = dry_bulb_c + 273.15
    t_sky_k = 0.0552 * (t_k ** 1.5)
    return t_sky_k - 273.15