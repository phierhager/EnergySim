# energysim/core/physics/coefficients.py
import jax.numpy as jnp
import dataclasses
from enum import Enum

class SurfaceRoughness(str, Enum):
    """Roughness values for MoWiTT/DOE-2 convection models."""
    VERY_ROUGH = "VeryRough" # Stucco
    ROUGH = "Rough"          # Brick
    MEDIUM = "Medium"        # Concrete
    SMOOTH = "Smooth"        # Glass/Paint

    def get_multiplier(self) -> float:
        # DOE-2 Convection multipliers (Rf)
        if self == self.VERY_ROUGH: return 2.17
        if self == self.ROUGH: return 1.67
        if self == self.MEDIUM: return 1.52
        return 1.11 # Smooth

@dataclasses.dataclass
class PhysicsConfig:
    """
    Central source of truth for physical constants.
    Standard: ISO 6946 and Stefan-Boltzmann.
    """
    stefan_boltzmann: float = 5.670374e-8
    air_density_kg_m3: float = 1.225
    air_heat_capacity_j_kgk: float = 1005.0

    # ISO 6946 Standard Surface Resistances (m2K/W)
    h_int_vertical: float = 7.69
    h_int_horizontal_up: float = 10.0
    h_int_horizontal_down: float = 5.88

    # Ground Physics
    ground_avg_temp_c: float = 10.0  # Yearly average
    ground_amplitude_c: float = 5.0  # Amplitude of swing
    ground_lag_days: float = 40.0    # Phase shift (thermal inertia)

    def get_internal_convection(self, tilt: float) -> float:
        """
        Returns h_int (W/m2K) based on surface tilt.
        tilt: 0 (flat roof), 90 (wall), 180 (floor)
        """
        # Simple classification based on ISO 6946 logic
        is_vertical = jnp.abs(tilt - 90.0) < 30.0
        is_ceiling = tilt < 30.0

        res = jnp.where(is_vertical, self.h_int_vertical,
                       jnp.where(is_ceiling, self.h_int_horizontal_up,
                                            self.h_int_horizontal_down)) # Floor
        return res

    def calculate_external_convection(self, wind_speed: float, roughness_mult: float, windward: bool = True) -> float:
        """
        DOE-2 model for external convection.
        h_ext = h_n + R_f * (a * V + b)
        """
        h_n = 3.5
        # Simple non-magic linear fit:
        return h_n + roughness_mult * wind_speed
    
    def get_ground_temperature(self, time_seconds: float) -> float:
        """
        Calculates ground temperature using a sinusoidal lag model (Kasuda & Achenbach approximation).
        T_ground = T_avg - Amplitude * cos(2pi * (day - shift) / 365)
        """
        seconds_in_year = 365.0 * 24.0 * 3600.0
        # Align phase: cos wave peaks in summer, but ground lags. 
        # Standard cosine peaks at 0, we want peak around day 200 (summer).
        # Shift for air temp usually ~15-20 days delay from solstice. Ground adds ~30-90 days.
        
        phase_shift_sec = self.ground_lag_days * 24.0 * 3600.0
        
        # Calculate simple annual oscillation
        oscillation = jnp.cos(2 * jnp.pi * (time_seconds - phase_shift_sec) / seconds_in_year)
        
        # Ground is colder in spring, warmer in autumn compared to air
        return self.ground_avg_temp_c - (self.ground_amplitude_c * oscillation)