# energysim/core/models/solar_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import SolarConfig, EnvironmentalContext, SolarOutput

class AbstractSolarModel(eqx.Module):
    config: SolarConfig
    
    def calculate(self, context: EnvironmentalContext) -> SolarOutput:
        """
        Calculates PV generation based on the full Environmental Context.
        Context includes: 
          - exo (Weather, Time, Location)
          - pv_shading_factor (Ray-traced obstruction)
          - sky/ground temps (Physics-derived boundary)
        """
        raise NotImplementedError

class GeometricSolarModel(AbstractSolarModel):
    """
    Highest-Fidelity "Whitebox" Solar Model.
    
    Architecture:
    1. Time: UTC-based Solar Time (Longitude-corrected).
    2. Geometry: Fourier-series Declination & Equation of Time (Spencer).
    3. Irradiance: Hay-Davies Anisotropy with Cloud Lensing Protection.
    4. Optics: ASHRAE IAM with Grazing Angle Safety.
    5. Thermodynamics: Configurable Faiman Model.
    6. Electrical: Modified Linear Model with Low-Irradiance Loss.
    """
    
    @eqx.filter_jit
    def calculate(self, context: EnvironmentalContext) -> SolarOutput:
        # --- Unpack Context ---
        exo = context.exo
        shading_factor = context.pv_shading_factor
        
        # --- Constants ---
        DEG_2_RAD = jnp.pi / 180.0
        SOLAR_CONSTANT = 1361.0
        
        # --- 1. Robust Time (UTC -> Solar Time) ---
        utc_hour = (exo.time_of_year_seconds % 86400) / 3600.0
        day_of_year = (exo.time_of_year_seconds // 86400) + 1
        
        B = (day_of_year - 1) * (360.0 / 365.0) * DEG_2_RAD

        # Equation of Time (Spencer)
        eot_min = (229.18 * (0.000075 + 0.001868 * jnp.cos(B) - 0.032077 * jnp.sin(B)
                   - 0.014615 * jnp.cos(2*B) - 0.04089 * jnp.sin(2*B)))
        
        # Solar Time
        solar_time_hours = utc_hour + (self.config.longitude_deg / 15.0) + (eot_min / 60.0)
        
        # Hour Angle
        omega_rad = (solar_time_hours - 12.0) * 15.0 * DEG_2_RAD

        # --- 2. Solar Position (Fourier Declination) ---
        delta_rad = (0.006918 - 0.399912 * jnp.cos(B) + 0.070257 * jnp.sin(B)
                     - 0.006758 * jnp.cos(2*B) + 0.000907 * jnp.sin(2*B)
                     - 0.002697 * jnp.cos(3*B) + 0.00148 * jnp.sin(3*B))

        lat_rad = self.config.latitude_deg * DEG_2_RAD
        
        # Zenith (theta_z)
        cos_theta_z = (jnp.sin(lat_rad) * jnp.sin(delta_rad)) + \
                      (jnp.cos(lat_rad) * jnp.cos(delta_rad) * jnp.cos(omega_rad))
        
        # Horizon Clamp
        cos_theta_z = jnp.fmax(0.01, cos_theta_z)
        theta_z_rad = jnp.arccos(cos_theta_z)
        
        # Elevation & Azimuth
        alpha_rad = (jnp.pi / 2.0) - theta_z_rad
        cos_gamma_s = (jnp.sin(alpha_rad) * jnp.sin(lat_rad) - jnp.sin(delta_rad)) / \
                      (jnp.cos(alpha_rad) * jnp.cos(lat_rad) + 1e-6)
        gamma_s_rad = jnp.sign(omega_rad) * jnp.arccos(jnp.clip(cos_gamma_s, -1.0, 1.0))

        # --- 3. Panel Incidence (Theta) ---
        tilt_rad = self.config.panel_tilt_deg * DEG_2_RAD
        p_azimuth_rad = (self.config.panel_azimuth_deg - 180.0) * DEG_2_RAD

        cos_theta = (jnp.cos(theta_z_rad) * jnp.cos(tilt_rad)) + \
                    (jnp.sin(theta_z_rad) * jnp.sin(tilt_rad) * jnp.cos(gamma_s_rad - p_azimuth_rad))
        
        cos_theta = jnp.fmax(0.0, cos_theta)

        # --- 4. Irradiance Model (Hay-Davies) ---
        dist_factor = (1.000110 + 0.034221 * jnp.cos(B) + 0.001280 * jnp.sin(B))
        I_extra = SOLAR_CONSTANT * dist_factor
        
        # Anisotropy Index with CLOUD LENSING PROTECTION
        # Clip max to 1.0 to prevent negative isotropic calculation
        anisotropy_index = jnp.fmin(1.0, exo.solar_dni_w_m2 / (I_extra + 1e-6))

        # Transposition Factor (R_b) with Cap
        R_b = jnp.fmin(cos_theta / cos_theta_z, 10.0)
        
        # Components
        beam_comp = exo.solar_dni_w_m2 * cos_theta * shading_factor
        circumsolar_comp = (exo.solar_dhi_w_m2 * anisotropy_index * R_b) * shading_factor
        
        svf_geom = (1.0 + jnp.cos(tilt_rad)) / 2.0
        effective_svf = jnp.where(
            self.config.sky_view_factor is not None,
            self.config.sky_view_factor,
            svf_geom
        )
        isotropic_comp = exo.solar_dhi_w_m2 * (1.0 - anisotropy_index) * effective_svf
        
        ghi_proxy = (exo.solar_dni_w_m2 * cos_theta_z) + exo.solar_dhi_w_m2
        albedo_comp = ghi_proxy * self.config.albedo * ((1.0 - jnp.cos(tilt_rad)) / 2.0)

        # --- 5. Optical Physics (IAM) ---
        cos_theta_safe = jnp.fmax(0.05, cos_theta)
        iam_beam = 1.0 - self.config.iam_b0 * ((1.0 / cos_theta_safe) - 1.0)
        iam_beam = jnp.clip(iam_beam, 0.0, 1.0)
        iam_diffuse = 0.95 

        poa_absorbed = (beam_comp * iam_beam) + \
                       (circumsolar_comp * iam_beam) + \
                       (isotropic_comp * iam_diffuse) + \
                       (albedo_comp * iam_diffuse)
        
        # --- 6. Thermal Physics (Faiman) ---
        wind_safe = jnp.where(jnp.isnan(exo.wind_speed_m_s), 1.0, exo.wind_speed_m_s)
        heat_transfer = self.config.thermal_u0 + (self.config.thermal_u1 * wind_safe)
        t_cell_c = exo.ambient_temp + (poa_absorbed / heat_transfer)

        # --- 7. Electrical Conversion (with Low-Light Loss) ---
        # Standard Temperature Loss
        temp_loss = 1.0 + (t_cell_c - self.config.reference_temp_c) * self.config.temp_coefficient
        
        # Low Irradiance Loss (Simplified Huld Model proxy)
        # Efficiency drops logarithmically below ~200 W/m2.
        # This is a robust "Whitebox" approximation without needing a full diode model.
        # Factor approaches 1.0 at 1000 W/m2, drops to ~0.9 at 200 W/m2.
        g_ref = 1000.0
        low_light_factor = 1.0 + 0.1 * jnp.log(jnp.fmax(0.1, poa_absorbed) / g_ref)
        # Clip: Can't improve efficiency (max 1.0), can't drop below 0.
        low_light_factor = jnp.clip(low_light_factor, 0.0, 1.0)
        
        power_w = poa_absorbed * self.config.panel_area_m2 * self.config.efficiency * temp_loss * low_light_factor

        return SolarOutput(pv_generation_w=jnp.fmax(0.0, power_w))

class SimpleSolarModel(AbstractSolarModel):
    @eqx.filter_jit
    def calculate(self, context: EnvironmentalContext) -> SolarOutput:
        exo = context.exo
        shading = context.pv_shading_factor
        total_irradiance = exo.solar_dni_w_m2 + exo.solar_dhi_w_m2
        temp_factor = 1.0 + (exo.ambient_temp - self.config.reference_temp_c) * self.config.temp_coefficient
        power_w = total_irradiance * self.config.panel_area_m2 * self.config.efficiency * temp_factor * shading
        return SolarOutput(pv_generation_w=jnp.fmax(0.0, power_w))

class PassthroughSolarModel(AbstractSolarModel):
    @eqx.filter_jit
    def calculate(self, context: EnvironmentalContext) -> SolarOutput:
        return SolarOutput(pv_generation_w=(context.exo.solar_dni_w_m2 + context.exo.solar_dhi_w_m2) * context.pv_shading_factor)
    








import pytest
import jax.numpy as jnp
import jax
from energysim.core.models.solar_model import GeometricSolarModel
from energysim.core.shared.data_structs import (
    SolarConfig, ExogenousData, EnvironmentalContext
)

# --- Helper: Mock Data Factory ---
def create_context(
    dni=800.0, 
    dhi=100.0, 
    temp=25.0, 
    wind=1.0, 
    hour_utc=12.0, 
    shading=1.0,
    lat=48.0,
    lon=0.0
):
    """Creates a minimal valid EnvironmentalContext for testing."""
    
    # Time: Jan 1st (Day 1) at specified UTC hour
    time_sec = (hour_utc * 3600.0) 
    
    # Mock Exogenous Data
    exo = ExogenousData(
        time_of_year_seconds=jnp.array(time_sec),
        ambient_temp=jnp.array(temp),
        solar_dni_w_m2=jnp.array(dni),
        solar_dhi_w_m2=jnp.array(dhi),
        wind_speed_m_s=jnp.array(wind),
        # Fill dummies for unused fields to satisfy Equinox structure
        relative_humidity=jnp.array(0.5),
        atmospheric_pressure=jnp.array(101325.0),
        price=jnp.array(0.2),
        carbon_intensity=jnp.array(0.0),
        base_load_w=jnp.array(0.0),
        occupant_profiles=jnp.array([]),
        passive_machine_profiles=jnp.array([]),
        smart_device_availability=jnp.array([])
    )

    return EnvironmentalContext(
        exo=exo,
        pv_shading_factor=jnp.array(shading),
        # Dummies for unused derived fields
        sun_vector=jnp.zeros(3),
        surface_shading_factors=jnp.array([]),
        sky_temp_c=jnp.array(10.0),
        ground_temp_c=jnp.array(10.0),
        wind_pressure_boundary=jnp.array([])
    )

@pytest.fixture
def solar_model():
    config = SolarConfig(
        panel_area_m2=1.0,
        efficiency=0.20,
        latitude_deg=48.0,  # Munich roughly
        longitude_deg=11.0,
        panel_tilt_deg=30.0,
        panel_azimuth_deg=180.0, # South
        temp_coefficient=-0.004,
        iam_b0=0.05,
        thermal_u0=25.0,
        thermal_u1=6.84
    )
    return GeometricSolarModel(config)

# ==========================================
# 1. Test: Cloud Lensing Protection
# ==========================================
def test_cloud_lensing_stability(solar_model):
    """
    Scenario: The sun is partially obscured but a cloud edge focuses light.
    DNI spikes to 1500 W/m2 (higher than Extraterrestrial ~1400 W/m2).
    
    Expectation: The Anisotropy Index should be clamped to 1.0. 
    If not clamped, (1 - AI) becomes negative, causing negative isotropic diffuse.
    """
    # I_extra in Jan is ~1410 W/m2. We force DNI higher.
    ctx = create_context(dni=2000.0, dhi=200.0, hour_utc=12.0)
    
    output = solar_model.calculate(ctx)
    
    # Generation should be high, but finite and positive
    assert output.pv_generation_w > 0.0
    assert not jnp.isnan(output.pv_generation_w)
    
    # If logic failed, isotropic component might have dragged total down or blown up
    # Simple check: 2000 W/m2 * 0.2 eff * 1 m2 = ~400W. 
    assert output.pv_generation_w > 300.0 

# ==========================================
# 2. Test: Grazing Angle / Horizon Singularity
# ==========================================
def test_horizon_singularity(solar_model):
    """
    Scenario: Sun is exactly at the horizon (Sunset).
    Zenith ~ 90 degrees. cos(z) -> 0.
    
    Expectation: R_b = cos(theta)/cos(z) should be capped.
    Output should be 0 or close to 0, NOT Infinity or NaN.
    """
    # Roughly sunset in Jan at Lat 48 is around 16:30 local time
    # We'll try specific low angles using extremely low solar elevation
    
    # Standard model blows up here without R_b clamping
    # We simulate a low sun angle by picking a time near sunset
    ctx = create_context(dni=100.0, dhi=10.0, hour_utc=15.8) 
    
    output = solar_model.calculate(ctx)
    
    assert not jnp.isnan(output.pv_generation_w)
    assert not jnp.isinf(output.pv_generation_w)
    # Should be small positive power
    assert output.pv_generation_w >= 0.0
    assert output.pv_generation_w < 50.0

# ==========================================
# 3. Test: Low-Light Efficiency Loss
# ==========================================
def test_low_light_loss(solar_model):
    """
    Scenario: Compare High Irradiance vs Low Irradiance efficiency.
    Low light (50 W/m2) should have lower Watts-per-Irradiance ratio than High light (1000 W/m2).
    """
    # Case A: Full Sun
    ctx_high = create_context(dni=1000.0, dhi=0.0)
    out_high = solar_model.calculate(ctx_high)
    eff_high = out_high.pv_generation_w / 1000.0 # Normalized yield
    
    # Case B: Twilight (50 W/m2)
    ctx_low = create_context(dni=50.0, dhi=0.0)
    out_low = solar_model.calculate(ctx_low)
    eff_low = out_low.pv_generation_w / 50.0
    
    # Physics Check: Efficiency drops at low light due to parasitic resistance
    # The logarithmic factor in your code guarantees this.
    assert eff_low < eff_high
    print(f"\nHigh Light Yield: {eff_high:.4f}, Low Light Yield: {eff_low:.4f}")

# ==========================================
# 4. Test: Wind Cooling (Faiman Model)
# ==========================================
def test_wind_cooling_effect(solar_model):
    """
    Scenario: Identical irradiance, different wind speeds.
    
    Expectation: High wind -> Cooler Panel -> Higher Efficiency -> Higher Power.
    """
    # Case A: Stagnant air
    ctx_still = create_context(dni=1000.0, wind=0.0, temp=25.0)
    out_still = solar_model.calculate(ctx_still)
    
    # Case B: Strong Breeze
    ctx_windy = create_context(dni=1000.0, wind=10.0, temp=25.0)
    out_windy = solar_model.calculate(ctx_windy)
    
    # Physics Check
    assert out_windy.pv_generation_w > out_still.pv_generation_w
    diff = out_windy.pv_generation_w - out_still.pv_generation_w
    print(f"\nWind Bonus: +{diff:.2f} Watts")

# ==========================================
# 5. Test: Shading Logic
# ==========================================
def test_shading_logic(solar_model):
    """
    Scenario: Beam is blocked (shading=0.0), but Diffuse exists.
    
    Expectation: Power should effectively disappear for the Beam component,
    but a small amount should remain from Isotropic Diffuse (if model allows).
    Your model applies shading to Beam AND Circumsolar, but not Isotropic.
    """
    # Unshaded
    ctx_clear = create_context(dni=800.0, dhi=200.0, shading=1.0)
    p_clear = solar_model.calculate(ctx_clear).pv_generation_w
    
    # Fully Shaded (by obstacle)
    ctx_shaded = create_context(dni=800.0, dhi=200.0, shading=0.0)
    p_shaded = solar_model.calculate(ctx_shaded).pv_generation_w
    
    # Expect massive drop, but not absolute zero (due to isotropic diffuse)
    assert p_shaded < p_clear * 0.3 
    assert p_shaded > 0.0 

# ==========================================
# 6. Test: Longitude/Time Correction
# ==========================================
def test_solar_time_correction():
    """
    Scenario: Solar Noon should happen earlier at East longitudes compared to Greenwich.
    """
    # Greenwich (0 deg)
    config_uk = SolarConfig(longitude_deg=0.0, latitude_deg=0.0, panel_tilt_deg=0.0)
    model_uk = GeometricSolarModel(config_uk)
    
    # 15 deg East (Should be 1 hour ahead solar wise)
    config_east = SolarConfig(longitude_deg=15.0, latitude_deg=0.0, panel_tilt_deg=0.0)
    model_east = GeometricSolarModel(config_east)
    
    # Check irradiance at 11:00 UTC
    # At 11:00 UTC, 15E is at 12:00 Solar Time (Peak). 0E is at 11:00 Solar Time (Rising).
    # Therefore, 15E should have HIGHER output than 0E at 11:00 UTC.
    
    ctx = create_context(dni=1000.0, hour_utc=11.0) # 11:00 UTC
    
    p_uk = model_uk.calculate(ctx).pv_generation_w
    p_east = model_east.calculate(ctx).pv_generation_w
    
    assert p_east > p_uk