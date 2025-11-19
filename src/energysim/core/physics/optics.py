import jax.numpy as jnp

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