# energysim/core/models/solar_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import SolarConfig, ExogenousData, SolarOutput

# --- Protocol: AlgebraicModel ---
# calculate(inputs) -> Output

class AbstractSolarModel(eqx.Module):
    config: SolarConfig
    
    def calculate(self, exo: ExogenousData) -> SolarOutput:
        raise NotImplementedError

class GeometricSolarModel(AbstractSolarModel):
    """
    High-Fidelity model.
    Calculates Plane-of-Array (POA) irradiance using DNI and DHI directly.
    """
    @eqx.filter_jit
    def calculate(self, exo: ExogenousData) -> SolarOutput:
        # Constants
        DEG_2_RAD = jnp.pi / 180.0
        SOLAR_CONSTANT = 1361.0

        # 1. Time Inputs
        day_of_year = (exo.time_of_year_seconds // 86400) + 1
        hour_of_day = (exo.time_of_year_seconds % 86400) / 3600.0

        # 2. Solar Declination (delta)
        delta = 23.45 * jnp.sin(DEG_2_RAD * (360.0 / 365.0) * (day_of_year + 284.0))
        delta_rad = delta * DEG_2_RAD

        # 3. Hour Angle (omega)
        omega = 15.0 * (hour_of_day - 12.0)
        omega_rad = omega * DEG_2_RAD

        # 4. Location & Panel Geometry
        lat_rad = self.config.latitude_deg * DEG_2_RAD
        tilt_rad = self.config.panel_tilt_deg * DEG_2_RAD
        p_azimuth_rad = (self.config.panel_azimuth_deg - 180.0) * DEG_2_RAD

        # 5. Solar Elevation (alpha) & Zenith (theta_z)
        sin_alpha = (jnp.sin(lat_rad) * jnp.sin(delta_rad)) + \
                    (jnp.cos(lat_rad) * jnp.cos(delta_rad) * jnp.cos(omega_rad))
        alpha_rad = jnp.arcsin(jnp.clip(sin_alpha, -1.0, 1.0))
        theta_z_rad = (jnp.pi / 2.0) - alpha_rad

        # 6. Solar Azimuth (gamma_s)
        cos_gamma_s = (jnp.sin(alpha_rad) * jnp.sin(lat_rad) - jnp.sin(delta_rad)) / \
                      (jnp.cos(alpha_rad) * jnp.cos(lat_rad) + 1e-6)
        gamma_s_rad = jnp.sign(omega_rad) * jnp.arccos(jnp.clip(cos_gamma_s, -1.0, 1.0))

        # 7. Angle of Incidence (theta)
        cos_theta = (jnp.cos(theta_z_rad) * jnp.cos(tilt_rad)) + \
                    (jnp.sin(theta_z_rad) * jnp.sin(tilt_rad) * jnp.cos(gamma_s_rad - p_azimuth_rad))
        cos_theta = jnp.fmax(0.0, cos_theta)

        # 8. Plane of Array (POA) Irradiance Calculation

        # --- UPDATED: Direct Beam with Dynamic Shading ---
        # exo.pv_shading_factor comes from the horizon mask (0.0 to 1.0)
        I_beam = exo.solar_dni_w_m2 * cos_theta * exo.pv_shading_factor

        # --- UPDATED: Diffuse with Static Sky View Factor ---
        # If user provided a specific SVF in config (from offline raytracing), use it.
        # Otherwise, use the standard unshaded approximation: (1 + cos(beta))/2
        standard_svf = (1.0 + jnp.cos(tilt_rad)) / 2.0
        
        # Python logic to select efficient path (static check)
        if self.config.sky_view_factor is not None:
            actual_svf = self.config.sky_view_factor
        else:
            actual_svf = standard_svf
            
        I_diffuse = exo.solar_dhi_w_m2 * actual_svf

        # C. Total POA
        poa_irradiance = I_beam + I_diffuse
        poa_irradiance = jnp.clip(poa_irradiance, 0.0, SOLAR_CONSTANT)

        # 9. Power Generation
        temp_factor = 1.0 + (exo.ambient_temp - self.config.reference_temp_c) * self.config.temp_coefficient
        power_w = poa_irradiance * self.config.panel_area_m2 * self.config.efficiency * temp_factor

        return SolarOutput(pv_generation_w=jnp.fmax(0.0, power_w))
    
    
class SimpleSolarModel(AbstractSolarModel):
    """
    Estimates generation without geometric knowledge.
    Assumes total available solar energy is DNI + DHI (Rough Global Proxy).
    Useful for simple "flat plate" estimations or when orientation is unknown.
    """
    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> SolarOutput:
        # 1. Rough Global Irradiance Proxy
        # Summing them is an over-estimation if sun is low, but fair for "Total Potential"
        total_irradiance_w_m2 = exogenous.solar_dni_w_m2 + exogenous.solar_dhi_w_m2
        
        # 2. Calculate temperature correction
        T_amb = exogenous.ambient_temp
        temp_factor = 1.0 + (T_amb - self.config.reference_temp_c) * self.config.temp_coefficient
        
        # 3. Calculate Power (W)
        power_w = (
            total_irradiance_w_m2 
            * self.config.panel_area_m2 
            * self.config.efficiency 
            * temp_factor
        )
        
        # Clip at 0
        pv_generation_w = jnp.fmax(0.0, power_w)
        return SolarOutput(pv_generation_w=pv_generation_w)

class PassthroughSolarModel(AbstractSolarModel):
    """
    A dummy model for backward compatibility or testing.
    Treats (DNI + DHI) as the pre-calculated generation in Watts.
    """
    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> SolarOutput:
        # Sum components to get total "signal"
        total_val = exogenous.solar_dni_w_m2 + exogenous.solar_dhi_w_m2
        return SolarOutput(pv_generation_w=total_val)