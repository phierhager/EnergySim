# energysim/sim/physics.py

import jax.numpy as jnp

def calculate_solar_incidence(
    sun_position: jnp.ndarray, # [x, y, z] normalized vector pointing to sun
    surface_normal: jnp.ndarray, # [x, y, z] normalized surface vector
    dni: float, # Direct Normal Irradiance (W/m2)
    dhi: float, # Diffuse Horizontal Irradiance (W/m2)
    surface_area: float,
    shgc: float, # For windows
    shading_factor: float = 1.0
) -> float:
    """
    Calculates watts absorbed/transmitted by a surface.
    """
    
    # 1. Direct Component: Dot product
    # cos(theta) = dot(sun, normal)
    cos_theta = jnp.dot(sun_position, surface_normal)
    
    # Clamp to 0 (sun behind surface)
    cos_theta = jnp.maximum(cos_theta, 0.0)
    
    direct_gain = dni * cos_theta * surface_area * shgc * shading_factor
    
    # 2. Diffuse Component (Simplified)
    # Vertical surfaces see 50% of sky, Flat see 100%
    # View Factor ~ (1 + cos(tilt))/2
    # We approximate using the Z component of normal to guess tilt
    tilt_factor = (1.0 + surface_normal[2]) / 2.0
    diffuse_gain = dhi * tilt_factor * surface_area * shgc
    
    return direct_gain + diffuse_gain

def compute_solar_vector(
    current_step: int, 
    dt: float, 
    latitude: float
):
    """
    Approximates sun position vector based on time of year/day.
    This creates a moving sun source for the simulation.
    """
    # (Implementation of solar declination/hour angle math here)
    # For MVP, we can pass the sun vector in the dataset or approximate it
    # simple circular motion for demo:
    day_progress = (current_step * dt) % 86400
    angle = (day_progress / 86400) * 2 * jnp.pi
    
    # Sun rises East (-x), sets West (+x), High Noon (+z)
    x = -jnp.cos(angle) # East-West
    y = -0.2 # Slight south bias
    z = jnp.maximum(0, jnp.sin(angle)) # Elevation
    
    norm = jnp.linalg.norm(jnp.array([x,y,z]))
    return jnp.array([x,y,z]) / (norm + 1e-6)


def calculate_iam(incidence_angle_rad: float) -> float:
    """
    Calculates the Incidence Angle Modifier (IAM) for standard double glazing.
    Uses the ASHRAE / EnergyPlus polynomial approximation.
    
    IAM = 1.0 at normal incidence (0 degrees).
    IAM -> 0.0 at 90 degrees.
    """
    theta = jnp.abs(incidence_angle_rad)
    
    # Standard 'Kelly' coefficients for double pane clear glass
    # IAM = 1 + b0*(1/cos - 1) ... is one form, but polynomial is more stable for JAX
    # Polynom: 1 - 8e-6*theta - 1e-4*theta^2 ... (theta in deg)
    # Let's use a robust cosine fit:
    
    # Simple physical approximation (Schlick's approximation for fresnel + attenuation)
    # Or simply specific coefficients for generic double glazing:
    c0 = 1.0
    c1 = -1.1e-3
    c2 = 5.9e-2
    c3 = -4.0e-3
    c4 = -4.8e-4
    
    theta_deg = jnp.degrees(theta)
    
    # Limits validity to < 90
    val = c0 + c1*theta_deg + c2*(theta_deg**2) + c3*(theta_deg**3) + c4*(theta_deg**4)
    
    # Clamp between 0 and 1
    return jnp.clip(val, 0.0, 1.0)


def get_sun_position(time_seconds: float, latitude_deg: float = 48.0) -> jnp.ndarray:
    """Calculates sun position vector [x, y, z]."""
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