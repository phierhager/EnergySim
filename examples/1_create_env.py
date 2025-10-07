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

  battery_3:
    type: helics
    sensor:
      observe_electrical_soc: true
      observe_thermal_soc: false
      observe_electrical_flow: true
      observe_heating_flow: false
      observe_cooling_flow: false
      soc_noise_std: 0.01
      flow_noise_std: 0.1
    action_space:
      normalized_power:
        type: discrete
        n_actions: 5
    connection:
      federate_name: bems_battery_3
      pub_topic: battery_3_control
      sub_topic: battery_3_status
      broker_address: 127.0.0.1
      core_type: zmq

  
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
    file_path: dataset.csv
    time_column: time
  params:
    feature_columns: 
      price: price_$
      pv: pv_kW
      load: load_kW
    dt_seconds: 900
    use_time_features: true

reward_manager:
  rewards:
    energy_cost:
      weight: -1.0
    comfort_violation:
      weight: -10.0
    battery_degradation:
      weight: -0.1
      
params:
  random_seed: 42

"""

yaml_cfg = yaml.safe_load(yaml_str)

env_config = from_dict(EnvironmentConfig, yaml_cfg, config=Config(cast=[Enum]))

print("Environment configuration loaded successfully.")
print(env_config)

environment = EnvironmentFactory.create_environment(env_config)
print("Environment created successfully.")

environment.reset()
print("Environment reset successfully.")

for _ in range(5):
    action = environment.action_space.sample()
    obs, reward, terminated, truncated, info = environment.step(action)
    print(
        f"Step reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
    )
