from enum import Enum
import jax.numpy as jnp

class SurfaceRoughness(str, Enum):
    """Roughness values for MoWiTT/DOE-2 convection models."""
    VERY_ROUGH = "VeryRough" # Stucco
    ROUGH = "Rough"          # Brick
    MEDIUM = "Medium"        # Concrete
    SMOOTH = "Smooth"        # Glass/Paint

def get_convection_multiplier(roughness: SurfaceRoughness) -> float:
    # DOE-2 Convection multipliers (Rf)
    if roughness == SurfaceRoughness.VERY_ROUGH: return 2.17
    if roughness == SurfaceRoughness.ROUGH: return 1.67
    if roughness == SurfaceRoughness.MEDIUM: return 1.52
    return 1.11 # Smooth

def get_internal_convection(h_int_vertical: float, h_int_horizontal_up: float, h_int_horizontal_down: float, tilt: float) -> float:
    """
    Returns h_int (W/m2K) based on surface tilt.
    tilt: 0 (flat roof), 90 (wall), 180 (floor)
    """
    # Simple classification based on ISO 6946 logic
    is_vertical = jnp.abs(tilt - 90.0) < 30.0
    is_ceiling = tilt < 30.0

    res = jnp.where(is_vertical, h_int_vertical,
                    jnp.where(is_ceiling, h_int_horizontal_up,
                                        h_int_horizontal_down)) # Floor
    return res

def calculate_external_convection(wind_speed: float, roughness_mult: float, windward: bool = True) -> float:
    """
    DOE-2 model for external convection.
    h_ext = h_n + R_f * (a * V + b)
    """
    h_n = 3.5
    # Simple non-magic linear fit:
    return h_n + roughness_mult * wind_speed