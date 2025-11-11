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
        self.load = load_col_or_zeros(ExoKey.LOAD)
        self.pv = load_col_or_zeros(ExoKey.PV)
        self.price = load_col_or_zeros(ExoKey.PRICE)
        self.ambient_temp = load_col_or_zeros(ExoKey.AMBIENT_TEMP)
        
        # --- LOAD NEW OPTIONAL COLUMNS ---
        self.internal_gains = load_col_or_zeros(ExoKey.INTERNAL_GAINS_W)
        self.solar_gains = load_col_or_zeros(ExoKey.SOLAR_GAINS_W)

    def __len__(self) -> int:
        return self.total_steps

    def __getitem__(self, idx: int) -> ExogenousData:
        """Returns data for a single step, converting to JAX arrays."""
        return ExogenousData(
            load=jnp.array(self.load[idx]),
            pv=jnp.array(self.pv[idx]),
            price=jnp.array(self.price[idx]),
            ambient_temp=jnp.array(self.ambient_temp[idx]),
            internal_gains_w=jnp.array(self.internal_gains[idx]), # <--- NEW
            solar_gains_w=jnp.array(self.solar_gains[idx])     # <--- NEW
        )
        
    def get_forecast(self, start_idx: int, horizon: int) -> ExogenousData:
        """Returns a slice of data for MPC forecasts."""
        s = slice(start_idx, start_idx + horizon)
        return ExogenousData(
            load=jnp.array(self.load[s]),
            pv=jnp.array(self.pv[s]),
            price=jnp.array(self.price[s]),
            ambient_temp=jnp.array(self.ambient_temp[s]),
            internal_gains_w=jnp.array(self.internal_gains[s]), # <--- NEW
            solar_gains_w=jnp.array(self.solar_gains[s])     # <--- NEW
        )