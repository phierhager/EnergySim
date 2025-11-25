import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Tuple

from energysim.io.parsers import parse_epw
from energysim.io.profile_loader import ProfileManager
from energysim.core.physics.weather import calculate_sky_temperature, calculate_ground_temperature
from energysim.core.shared.data_structs import ExogenousData, Array, MAX_ROOMS, MAX_MACHINES

class ScenarioLoader:
    def __init__(self, epw_path: str, n_houses: int, dt_seconds: int = 900):
        self.epw_path = epw_path
        self.n_houses = n_houses
        self.dt_seconds = dt_seconds
        self.time_index = pd.date_range("2024-01-01", periods=int(365*24*3600/dt_seconds), freq=f"{dt_seconds}s")
        self.profiler = ProfileManager(self.time_index)

    def load_environment(self, price_file: str, base_load_file: str) -> Tuple[ExogenousData, Array]:
        # 1. Weather (CPU -> JAX)
        w_data = parse_epw(self.epw_path, target_freq_min=self.dt_seconds//60)
        
        # 2. Physics
        sky_t = calculate_sky_temperature(jnp.array(w_data['ambient_temp']), jnp.array(w_data['dew_point']))
        grnd_t = calculate_ground_temperature(jnp.array(w_data['time_of_year_seconds']), 10.0, 5.0, 40.0)
        
        # 3. Profiles (Batching & Padding)
        # Loads are HOUSE SPECIFIC, so they return a separate array to be vmapped later
        # Shape: (N_houses, T, 1)
        house_loads = self.profiler.load_batch_profiles(base_load_file, self.n_houses, max_dim=1)
        house_loads = jax.device_put(jnp.array(house_loads))
        
        # 4. Global Shared Data
        global_exo = ExogenousData(
            time_of_year_seconds=jnp.array(w_data['time_of_year_seconds']),
            ambient_temp=jnp.array(w_data['ambient_temp']),
            relative_humidity=jnp.array(w_data['relative_humidity']),
            atmospheric_pressure=jnp.array(w_data['atmospheric_pressure']),
            solar_dni_w_m2=jnp.array(w_data['solar_dni_w_m2']),
            solar_dhi_w_m2=jnp.array(w_data['solar_dhi_w_m2']),
            wind_speed_m_s=jnp.array(w_data['wind_speed_m_s']),
            
            sky_temp=sky_t,
            ground_temp=grnd_t,
            price=jnp.full(len(sky_t), 0.20), # Mock price
            
            # Placeholders (Padded Shapes) used for type stability in JAX
            base_load_w=jnp.zeros(len(sky_t)), 
            occupant_profiles=jnp.zeros((len(sky_t), MAX_ROOMS)),
            passive_machine_profiles=jnp.zeros((len(sky_t), MAX_MACHINES)),
            smart_device_availability=jnp.zeros((len(sky_t), MAX_MACHINES))
        )
        
        return jax.device_put(global_exo), house_loads