from typing import Callable, List, Optional
import numpy as np
import pandas as pd
import jax.numpy as jnp
from energysim.core.shared.data_structs import ExogenousData
from energysim.core.shared.control_variables import ExoKey

class SimulationDataset:
    """
    Loads time-series data and maps it to the new split ExogenousData structure.
    """
    def __init__(
        self, 
        file_path: str, 
        dt_seconds: int, 
        # Metadata needed to shape the arrays correctly
        n_passive_machines: int,
        n_occupants: int,
        # Column mapping: which CSV column goes to which profile index?
        passive_col_map: Optional[List[str]] = None, # e.g. ["base_load", "fridge_power"]
        occupant_col_map: Optional[List[str]] = None, # e.g. ["occupancy_count"]
        read_fn: Callable[[str], pd.DataFrame] = pd.read_csv
    ):
        df = read_fn(file_path)
        self.dt_seconds = dt_seconds
        self.total_steps = len(df)
        self.n_passive = n_passive_machines
        self.n_occupants = n_occupants

        # --- Helper ---
        def load_col_or_zeros(key: str) -> np.ndarray:
            if key in df.columns:
                return df[key].to_numpy(dtype=np.float32)
            return np.zeros(self.total_steps, dtype=np.float32)

        # --- 1. Standard Environment ---
        self.ambient_temp = load_col_or_zeros(ExoKey.AMBIENT_TEMP)
        self.solar_dni = load_col_or_zeros(ExoKey.SOLAR_DNI_W_M2)
        self.solar_dhi = load_col_or_zeros(ExoKey.SOLAR_DHI_W_M2)
        self.wind_speed = load_col_or_zeros(ExoKey.WIND_SPEED_M_S)
        self.price = load_col_or_zeros(ExoKey.PRICE)

        dt_series = pd.to_datetime(df[ExoKey.TIME])
        start_year = pd.to_datetime(dt_series.dt.year.astype(str) + "-01-01")
        self.time_seconds = (dt_series - start_year).dt.total_seconds().to_numpy(dtype=np.float32)

        # --- 2. Passive Machine Profiles ---
        # Shape: (TotalSteps, N_passive)
        self.passive_profiles = np.zeros((self.total_steps, self.n_passive), dtype=np.float32)
        if passive_col_map:
            for i, col_name in enumerate(passive_col_map):
                if i < self.n_passive:
                    self.passive_profiles[:, i] = load_col_or_zeros(col_name)

        # --- 3. Occupant Profiles ---
        # Shape: (TotalSteps, N_occupants)
        self.occupant_profiles = np.zeros((self.total_steps, self.n_occupants), dtype=np.float32)
        if occupant_col_map:
            for i, col_name in enumerate(occupant_col_map):
                if i < self.n_occupants:
                    self.occupant_profiles[:, i] = load_col_or_zeros(col_name)

    def __len__(self) -> int:
        return self.total_steps

    def __getitem__(self, idx: int) -> ExogenousData:
        return ExogenousData(
            ambient_temp=jnp.array(self.ambient_temp[idx]),
            solar_dni_w_m2=jnp.array(self.solar_dni[idx]),
            solar_dhi_w_m2=jnp.array(self.solar_dhi[idx]),
            wind_speed_m_s=jnp.array(self.wind_speed[idx]),
            time_of_year_seconds=jnp.array(self.time_seconds[idx]),
            price=jnp.array(self.price[idx]),
            # New split fields
            passive_machine_profiles=jnp.array(self.passive_profiles[idx]),
            occupant_profiles=jnp.array(self.occupant_profiles[idx])
        )

    def get_forecast(self, start_idx: int, horizon: int) -> ExogenousData:
        s = slice(start_idx, start_idx + horizon)
        return ExogenousData(
            ambient_temp=jnp.array(self.ambient_temp[s]),
            solar_dni_w_m2=jnp.array(self.solar_dni[s]),
            solar_dhi_w_m2=jnp.array(self.solar_dhi[s]),
            wind_speed_m_s=jnp.array(self.wind_speed[s]),
            time_of_year_seconds=jnp.array(self.time_seconds[s]),
            price=jnp.array(self.price[s]),
            passive_machine_profiles=jnp.array(self.passive_profiles[s]),
            occupant_profiles=jnp.array(self.occupant_profiles[s])
        )