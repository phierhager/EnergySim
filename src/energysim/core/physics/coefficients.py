from dataclasses import dataclass

@dataclass
class Coefficients:
    """
    High-Fidelity Physical Constants.
    Separates Convective and Radiative heat transfer.
    """
    stefan_boltzmann: float = 5.670374e-8
    air_density_kg_m3: float = 1.204 # At 20C
    air_heat_capacity_j_kgk: float = 1005.0
    latent_heat_vaporization: float = 2460000.0

    # --- 1. Convection Coefficients (h_cv) [W/m2K] ---
    # Derived from ISO 13790 / ASHRAE Fundamentals
    # Vertical walls (Natural convection)
    h_cv_vertical: float = 2.5 
    # Floors (Heat flow down - stable air)
    h_cv_floor: float = 0.7
    # Ceilings (Heat flow up - buoyant air)
    h_cv_ceiling: float = 5.0
    # Forced convection (Windows/Exterior) is calculated dynamically by wind

    # --- 2. Radiation Coefficients (h_rad) [W/m2K] ---
    # Linearized Stefan-Boltzmann: h_rad = 4 * sigma * eps * T_mean^3
    # Standard design value for indoor surfaces (eps=0.9) at 20C
    h_rad_interior: float = 5.13 
    
    # Ground Physics
    ground_avg_temp_c: float = 10.0
    ground_amplitude_c: float = 5.0
    ground_lag_days: float = 40.0