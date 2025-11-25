# physics/constants.py
from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicsConstants:
    """
    Fundamental physical constants and standard reference values.
    Freeze this class to ensure immutability during JAX transformations.
    """
    # Fundamental
    STEFAN_BOLTZMANN: float = 5.670374419e-8  # W/m²K⁴
    GRAVITY: float = 9.80665                  # m/s²
    GAS_CONSTANT_AIR: float = 287.058         # J/kgK
    GAS_CONSTANT_VAPOR: float = 461.52        # J/kgK
    
    # Standard Atmosphere
    P_STD_ATM: float = 101325.0               # Pa
    T_ABS_ZERO: float = 273.15                # K offset
    
    # Water Properties
    LATENT_HEAT_VAPORIZATION: float = 2.45e6  # J/kg (at approx 20C)
    SPECIFIC_HEAT_AIR: float = 1005.0         # J/kgK
    
    # Reference values for linearization (only used if dynamic calc fails)
    RHO_AIR_REF: float = 1.204                # kg/m³ (20C, 1atm)

CONSTANTS = PhysicsConstants()