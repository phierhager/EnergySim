import pandas as pd
import numpy as np
from typing import Dict, Optional, List

def resample_and_align(
    df: pd.DataFrame, 
    target_freq_min: int = 15, 
    interpolation_method: str = 'linear'
) -> pd.DataFrame:
    """
    Upsamples hourly data (EPW) to simulation frequency (e.g. 15 min).
    Handles datetime indexing and interpolation.
    """
    # Create a dummy datetime index if missing (assuming starts Jan 1st)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2024-01-01", periods=len(df), freq='H')
    
    # Resample
    freq_str = f"{target_freq_min}min"
    df_resampled = df.resample(freq_str).interpolate(method=interpolation_method)
    
    # Forward fill any edges
    df_resampled = df_resampled.ffill().bfill()
    
    return df_resampled

def parse_epw(file_path: str, target_freq_min: int = 60) -> Dict[str, np.ndarray]:
    """
    Reads .epw file and returns aligned numpy arrays.
    """
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
    
    # Read CSV
    try:
        df = pd.read_csv(file_path, skiprows=8, header=None, names=col_names)
    except Exception as e:
        raise ValueError(f"Failed to parse EPW file {file_path}: {e}")

    # Resample if needed (EPW is usually 60min)
    if target_freq_min != 60:
        df = resample_and_align(df, target_freq_min)

    # Extract Dict of Float32 Arrays
    data = {
        "ambient_temp": df["dry_bulb"].values.astype(np.float32),
        "dew_point": df["dew_point"].values.astype(np.float32),
        "relative_humidity": df["rel_hum"].values.astype(np.float32) / 100.0, # Normalize 0-1
        "atmospheric_pressure": df["atmos_pressure"].values.astype(np.float32),
        "solar_dni_w_m2": df["dni"].values.astype(np.float32),
        "solar_dhi_w_m2": df["dhi"].values.astype(np.float32),
        "wind_speed_m_s": df["wind_spd"].values.astype(np.float32),
        # Seconds from start of year
        "time_of_year_seconds": (df.index - df.index[0]).total_seconds().values.astype(np.float32)
    }
    
    return data

def parse_timeseries_csv(
    file_path: str, 
    column_map: Dict[str, str], 
    target_length: int
) -> Dict[str, np.ndarray]:
    """
    Reads generic CSV (Prices, Carbon, etc.).
    
    Args:
        column_map: {csv_column_name: internal_key_name}
        target_length: Expected number of steps (for validation/padding)
    """
    df = pd.read_csv(file_path)
    
    # Simple resampling if length mismatch (Assuming uniform linear stretch)
    # In production, you'd want real datetime alignment
    if len(df) != target_length:
        # Basic resize using numpy interp if no timestamp provided
        old_idx = np.linspace(0, 1, len(df))
        new_idx = np.linspace(0, 1, target_length)
        
        data = {}
        for col, key in column_map.items():
            val = df[col].values.astype(np.float32)
            data[key] = np.interp(new_idx, old_idx, val).astype(np.float32)
        return data

    # Direct map
    return {key: df[col].values.astype(np.float32) for col, key in column_map.items()}