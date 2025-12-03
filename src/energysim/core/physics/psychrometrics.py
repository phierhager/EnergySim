import jax.numpy as jnp
from .constants import CONSTANTS

# --- Basic State Functions (Your existing code) ---

def calculate_saturation_pressure(T_celsius: jnp.ndarray) -> jnp.ndarray:
    return 611.2 * jnp.exp((17.67 * T_celsius) / (T_celsius + 243.5))

def calculate_humidity_ratio(T_celsius: jnp.ndarray, rel_hum_0_1: jnp.ndarray, pressure_pa: jnp.ndarray) -> jnp.ndarray:
    P_sat = calculate_saturation_pressure(T_celsius)
    P_vap = rel_hum_0_1 * P_sat
    # Clip P_vap to prevent singularity if P_vap approx P_atm
    P_vap = jnp.clip(P_vap, 0.0, pressure_pa - 1.0)
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


# --- New Process Function (High Fidelity Coil Physics) ---

def calculate_dx_coil_split(
    total_cooling_w: jnp.ndarray,    # Negative value (energy leaving air)
    T_in_c: jnp.ndarray,             # Entering Dry Bulb
    rel_hum_in: jnp.ndarray,         # Entering RH (0-1)
    pressure_pa: jnp.ndarray,        # Ambient Pressure
    T_coil_adp_c: jnp.ndarray,       # Apparatus Dew Point (Coil Temp)
    bypass_factor: jnp.ndarray,      # Fraction of air bypassing coil (0.1 - 0.2)
    mass_flow_air_kg_s: jnp.ndarray  # Air mass flow rate
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Splits total cooling energy into Sensible and Latent components using
    the Bypass Factor method. Differentiable.
    
    Returns:
        (sensible_cooling_w, water_removed_kg_s) 
        * sensible_cooling_w is negative (cooling).
        * water_removed_kg_s is positive (mass leaving air).
    """
    
    # 1. State of Entering Air
    w_in = calculate_humidity_ratio(T_in_c, rel_hum_in, pressure_pa)
    
    # 2. State of Coil Surface (Saturated at ADP)
    # RH is 1.0 at the wet coil surface
    w_coil_sat = calculate_humidity_ratio(T_coil_adp_c, 1.0, pressure_pa)
    
    # 3. Potential Dehumidification
    # If air is drier than the coil surface (w_in < w_coil_sat), no condensation occurs.
    # Use ReLU (maximum) to handle this switch differentiably.
    delta_w_potential = jnp.maximum(0.0, w_in - w_coil_sat)
    
    # 4. Actual Dehumidification (Apply Bypass Factor)
    # Only the air contacting the coil (1 - BF) is dehumidified
    w_removed_kg_kg = delta_w_potential * (1.0 - bypass_factor)
    
    # 5. Theoretical Latent Power
    # Power = Mass_Flow * Delta_Humidity * Latent_Heat
    latent_cooling_power_mag = mass_flow_air_kg_s * w_removed_kg_kg * CONSTANTS.LATENT_HEAT_VAPORIZATION
    
    # 6. Energy Balance Constraint
    # We cannot remove more latent heat than the total capacity of the compressor.
    total_cooling_mag = jnp.abs(total_cooling_w)
    
    # Clamp latent to total available
    actual_latent_mag = jnp.minimum(latent_cooling_power_mag, total_cooling_mag)
    
    # Sensible is whatever is left
    actual_sensible_mag = total_cooling_mag - actual_latent_mag
    
    # 7. Calculate final Mass Flow of Water
    water_removed_kg_s = actual_latent_mag / CONSTANTS.LATENT_HEAT_VAPORIZATION
    
    # Return Sensible (Negative convention) and Water Mass (Positive convention)
    return -actual_sensible_mag, water_removed_kg_s