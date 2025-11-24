import jax.numpy as jnp
import pandas as pd
from ..core.weather import WeatherData, calculate_sky_temperature, get_ground_temperature

def load_epw(file_path: str, ground_config: dict = None) -> WeatherData:
    """
    Parses a standard .epw file into JAX arrays.
    
    Args:
        file_path: Path to the .epw file.
        ground_config: Dict with keys 'lag_days', 'avg_temp_c', 'amplitude_c' 
                       to compute ground temps using your weather.py logic.
    """
    # 1. Read with Pandas (Fastest text parser)
    # EPW headers are usually 8 rows. Column indices are standard.
    col_names = [
        "year", "month", "day", "hour", "minute", "flags", 
        "dry_bulb", "dew_point", "rel_hum", "atmos_pressure", 
        "ext_horiz_rad", "ext_dir_rad", "horiz_ir_sky", 
        "ghi", "dni", "dhi", 
        "global_horiz_illum", "dir_norm_illum", "diff_horiz_illum", 
        "zenith_lum", "wind_dir", "wind_spd", 
        "total_sky_cover", "opaque_sky_cover", "visibility", 
        "ceiling_hgt", "pres_weath_obs", "pres_weath_codes", 
        "precip_wtr", "aerosol_opt_depth", "snow_depth", 
        "days_last_snow", "albedo", "liq_precip_depth", "liq_precip_qty"
    ]
    
    # Read CSV, skipping header rows
    df = pd.read_csv(file_path, skiprows=8, header=None, names=col_names)

    # 2. Clean and Convert Data
    # Convert strictly to float32 for JAX
    def to_jax(series):
        return jnp.array(series.values, dtype=jnp.float32)

    # Time handling: Create continuous seconds array
    # Assuming standard 8760 hourly file. 
    # If sub-hourly, we might need to adjust, but index usually works.
    n_steps = len(df)
    dt = 3600.0 # Assuming hourly for standard EPW
    time_seconds = jnp.arange(0, n_steps * dt, dt, dtype=jnp.float32)

    # 3. Extract Core Fields
    dry_bulb = to_jax(df['dry_bulb'])
    dew_point = to_jax(df['dew_point'])
    
    # 4. Derived Fields
    # Calculate Sky Temp (Vectorized)
    sky_temp = calculate_sky_temperature(dry_bulb, dew_point)
    
    # Calculate Ground Temp (Vectorized using your logic)
    if ground_config is None:
        ground_config = {'lag_days': 40.0, 'avg_temp_c': 10.0, 'amplitude_c': 5.0}
    
    ground_temp = get_ground_temperature(
        ground_lag_days=ground_config['lag_days'],
        ground_avg_temp_c=ground_config['avg_temp_c'],
        ground_amplitude_c=ground_config['amplitude_c'],
        time_seconds=time_seconds
    )

    return WeatherData(
        time_seconds=time_seconds,
        dry_bulb_temp=dry_bulb,
        dew_point_temp=dew_point,
        relative_humidity=to_jax(df['rel_hum']),
        atmospheric_pressure=to_jax(df['atmos_pressure']),
        ghi=to_jax(df['ghi']),
        dni=to_jax(df['dni']),
        dhi=to_jax(df['dhi']),
        wind_speed=to_jax(df['wind_spd']),
        wind_direction=to_jax(df['wind_dir']),
        sky_temp=sky_temp,
        ground_temp=ground_temp
    )