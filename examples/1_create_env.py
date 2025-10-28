# To be able to run this example, please create a dataset.csv file with the script create_random_dataset.py.
# Also, make sure to run the HELICS broker before running this example with start_broker.py.

import logging
import yaml
from dacite import from_dict, Config
from enum import Enum
from energysim.rl.factory import EnvironmentFactory, EnvironmentConfig

logging.basicConfig(level=logging.DEBUG)

yaml_str = """
components:
  battery_1:
    type: battery
    model:
      type: simple
      efficiency: 0.9
      max_power: 10.0
      capacity: 20.0
      init_soc: 0.5
      deadband: 0.1
    sensor:
      observe_electrical_soc: true
      observe_thermal_soc: false
      observe_electrical_flow: true
      observe_heating_flow: false
      observe_cooling_flow: false
      soc_noise_std: 0.01
      flow_noise_std: 0.1

  battery_2:
    type: battery
    model:
      type: degrading
      efficiency: 1.0
      max_power: 1.0
      capacity: 1.0
      init_soc: 0.0
      deadband: 0.0
      degradation_mode: "linear"         # "linear", "exponential", or "polynomial"
      degradation_rate: 0.001            # per unit throughput or cycle
      min_capacity_fraction: 0.5         # cannot degrade below this fraction of initial capacity
      poly_exponent: 2.0     
    sensor:
      observe_electrical_soc: true
      observe_thermal_soc: false
      observe_electrical_flow: true
      observe_heating_flow: false
      observe_cooling_flow: false
      soc_noise_std: 0.01
      flow_noise_std: 0.1       
    config:
      efficiency: 0.95
      max_power: 5.0
      capacity: 15.0
      init_soc: 0.3
      deadband: 0.05
  
  battery_4:
    type: battery
    model:
      type: simple
      efficiency: 0.9
      max_power: 10.0
      capacity: 20.0
      init_soc: 0.5
      deadband: 0.1
    sensor:
      observe_electrical_soc: true
      observe_thermal_soc: false
      observe_electrical_flow: true
      observe_heating_flow: false
      observe_cooling_flow: false
      soc_noise_std: 0.01
      flow_noise_std: 0.1
    config:
      efficiency: 0.95
      max_power: 5.0
      capacity: 15.0
      init_soc: 0.3
      deadband: 0.05

thermal_sensor:
  observe_indoor_temp: true
  observe_temp_error: false
  observe_comfort_violation: false
  observe_zone_temps: false
  temp_noise_std: 0.0

thermal_model:
  thermal_model_type: simple_air
  params:
    building_volume: 300.0

dataset:
  data_source:
    type: file
    file_path: data/building_timeseries.csv
    time_column: timestamp
  params:
    feature_columns: 
      price: electricity_price
      pv: pv_generation
      load: base_load
      ambient_temperature: ambient_temp
    dt_seconds: 900
    use_time_features: true
  
reward_manager:
  energy_weight: 1.0
  comfort_weight: 0.5

economic:
  feed_in_tariff_eur_per_j: 0.0000001
  tax_rate: 0.19
  demand_charge_threshold_j: 50.0
  demand_charge_rate_eur_per_j: 0.1
      
params:
  random_seed: 42

wrappers:
  action_space:
    discrete_bins:
      battery_1: 5
      battery_2: 3
      battery_4: 7
    clip_actions: true

  observation_space:
    noise_std: 0.01
    framestack_size: 4
    normalize: true
    time_aware: true

  misc:
    max_episode_steps: 10
"""

yaml_cfg = yaml.safe_load(yaml_str)

env_config = from_dict(EnvironmentConfig, yaml_cfg, config=Config(cast=[Enum]))

print("Environment configuration loaded successfully.")
print(env_config)

environment = EnvironmentFactory.create_environment(env_config)
print("Environment created successfully.")

environment.reset()
print("Environment reset successfully.")

for _ in range(50):
    structured_action = environment.action_space.sample()
    obs, reward, terminated, truncated, info = environment.step(structured_action)
    print(
        f"Step reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
    )
