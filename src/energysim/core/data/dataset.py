# energysim/core/data/dataset.py
from typing import Callable
import numpy as np
import pandas as pd
from energysim.core.shared.data_structs import ExogenousData
from energysim.core.shared.control_variables import ExoKey
import jax.numpy as jnp

class SimulationDataset:
    """
    Loads time-series data from a file and serves it step-by-step.
    Initializes all behavioral and calculated fields to 0.0.
    """
    def __init__(self, file_path: str, dt_seconds: int, read_fn: Callable[[str], pd.DataFrame] = pd.read_csv):
        df = read_fn(file_path)

        self.dt_seconds = dt_seconds

        # Assume total_steps is based on a required column
        self.total_steps = len(df)

        # --- Helper function to safely load columns ---
        def load_col_or_zeros(key: ExoKey) -> np.ndarray:
            if key in df.columns:
                return df[key].to_numpy(dtype=np.float32)
            else:
                print(f"Warning: Column '{key}' not found in data. Defaulting to 0.0.")
                return np.zeros(self.total_steps, dtype=np.float32)

        # Store data as lightweight NumPy arrays
        
        # --- Weather ---
        self.ambient_temp = load_col_or_zeros(ExoKey.AMBIENT_TEMP)
        self.solar_dni_w_m2 = load_col_or_zeros(ExoKey.SOLAR_DNI_W_M2)
        self.solar_dhi_w_m2 = load_col_or_zeros(ExoKey.SOLAR_DHI_W_M2)
        self.wind_speed_m_s = load_col_or_zeros(ExoKey.WIND_SPEED_M_S)
        
        # --- Price ---
        self.price = load_col_or_zeros(ExoKey.PRICE)
        
        # --- Loads ---
        self.base_load_w = load_col_or_zeros(ExoKey.LOAD) # <--- RENAMED

        # --- Time ---
        dt_series = pd.to_datetime(df[ExoKey.TIME])
        start_of_year = pd.to_datetime(dt_series.dt.year.astype(str) + "-01-01")
        self.time_of_year_seconds = (dt_series - start_of_year).dt.total_seconds().to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return self.total_steps

    def __getitem__(self, idx: int) -> ExogenousData:
        """Returns data for a single step, converting to JAX arrays."""
        
        # All behavioral/calculated fields are initialized to 0.0
        # The environment (e.g., EnergySimEnv) is responsible for filling them.
        return ExogenousData(
            # --- Weather ---
            ambient_temp=jnp.array(self.ambient_temp[idx]),
            solar_dni_w_m2=jnp.array(self.solar_dni_w_m2[idx]),
            solar_dhi_w_m2=jnp.array(self.solar_dhi_w_m2[idx]),
            wind_speed_m_s=jnp.array(self.wind_speed_m_s[idx]),
            # --- Time ---
            time_of_year_seconds=jnp.array(self.time_of_year_seconds[idx]),
            # --- Price ---
            price=jnp.array(self.price[idx]),
            # --- Loads ---
            base_load_w=jnp.array(self.base_load_w[idx]),
        )

    def get_forecast(self, start_idx: int, horizon: int) -> ExogenousData:
        """Returns a slice of data for MPC forecasts."""
        s = slice(start_idx, start_idx + horizon)
        
        return ExogenousData(
            # --- Weather ---
            ambient_temp=jnp.array(self.ambient_temp[s]),
            solar_dni_w_m2=jnp.array(self.solar_dni_w_m2[s]),
            solar_dhi_w_m2=jnp.array(self.solar_dhi_w_m2[s]),
            wind_speed_m_s=jnp.array(self.wind_speed_m_s[s]),
            # --- Time ---
            time_of_year_seconds=jnp.array(self.time_of_year_seconds[s]),
            # --- Price ---
            price=jnp.array(self.price[s]),
            # --- Loads ---
            base_load_w=jnp.array(self.base_load_w[s]),
        )