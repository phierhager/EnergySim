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