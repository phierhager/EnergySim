import jax
import jax.numpy as jnp

Array = jnp.ndarray

def calculate_isentropic_efficiency(
    lift_k: Array, 
    speed_ratio: Array, 
    design_lift_k: float, 
    design_speed_ratio: float, 
    eta_peak: float, 
    k_lift: float, 
    k_speed: float
) -> Array:
    """
    Calculates Isentropic Efficiency (eta_II) based on deviation from design point.
    """
    delta_lift = lift_k - design_lift_k
    delta_speed = speed_ratio - design_speed_ratio
    
    # Elliptical/Parabolic decay for off-design operation
    penalty = (k_lift * delta_lift**2) + (k_speed * delta_speed**2)
    return jnp.clip(eta_peak - penalty, 0.20, 0.75)

def calculate_volumetric_limit(
    t_evap_c: Array, 
    speed_ratio: Array, 
    max_disp_w_per_k: float
) -> Array:
    """
    Calculates the Volumetric limit (Choke flow).
    """
    # Density Proxy (Simplified Clausius-Clapeyron / Ideal Gas)
    # Normalized to 1.0 at 0C. Drops to ~0.4 at -20C.
    density_factor = jnp.exp(0.045 * t_evap_c)
    
    # Volumetric Efficiency drop at high speeds (leakage/backflow)
    vol_eff = 0.95 - (0.1 * speed_ratio)
    
    # Base scaler (displacement * latent_heat)
    limit_w = max_disp_w_per_k * 300.0 
    
    return limit_w * speed_ratio * density_factor * vol_eff

def calculate_defrost_penalty(
    t_ambient_c: Array, 
    rel_humidity: Array
) -> Array:
    """
    Approximation of defrost cycle penalty.
    Penalty peaks when Temp is near 0-4C and Humidity is high.
    """
    # Peak frost formation happens around 2C.
    temp_risk = jnp.exp(-((t_ambient_c - 2.0)**2) / 10.0) 
    
    # Humidity Risk: Sigmoid activation above 60% RH
    hum_risk = jax.nn.sigmoid((rel_humidity - 0.75) * 15.0)
    
    # Penalty factor: 1.0 = No Penalty, 0.85 = 15% energy lost to defrost
    penalty = 1.0 - (0.15 * temp_risk * hum_risk)
    return penalty

def calculate_inverter_efficiency(
    electrical_power_w: Array, 
    max_electrical_power_w: float, 
    curve_coeffs: Array
) -> Array:
    """
    Calculates motor/inverter efficiency using a quadratic curve fit.
    Standardized for both HP and AC.
    
    Args:
        electrical_power_w: Current power draw.
        max_electrical_power_w: Rated max power (for PLR calculation).
        curve_coeffs: Array[3] representing [intercept, slope, curvature].
    """
    # Part Load Ratio (0.0 to 1.0)
    plr = electrical_power_w / (max_electrical_power_w + 1e-6)
    
    # Quadratic Curve: Eff = c0 + c1*PLR + c2*PLR^2
    # This allows modeling the "hump" where efficiency peaks around 50-70% load.
    raw_eff = (
        curve_coeffs[0] 
        + (curve_coeffs[1] * plr) 
        + (curve_coeffs[2] * (plr**2))
    )
    
    # Physical clamps: Efficiency cannot be negative or > 98%
    # We allow a low floor (0.1) to avoid zeros in division elsewhere.
    return jnp.clip(raw_eff, 0.1, 0.98)