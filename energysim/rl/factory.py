from enum import Enum
from typing import Dict, Union
from energysim.core.components.shared.component_base import (
    ComponentBase,
    ComponentOutputs,
)
from dataclasses import dataclass

from energysim.core.thermal.thermal_model_base import (
    ThermalModel,
    ThermalModelConfig,
)
from energysim.rl.building_environment import (
    EnvironmentParameters,
)
from energysim.rl.rewards.factory import RewardManagerFactory
from energysim.rl.building_environment import (
    BuildingEnvironment,
)
from energysim.core.components.shared.sensors import (
    Sensor,
    ThermalSensor,
    ThermalSensorConfig,
    ComponentSensorConfig,
    ComponentSensor,
)
from energysim.core.data.config import EnergyDatasetConfig
from energysim.rl.rewards.reward_config import RewardConfig
from energysim.core.data.dataset import EnergyDataset
from omegaconf import OmegaConf
from energysim.core.components.config import ComponentConfig
from energysim.core.components.factory import build_component
from energysim.core.thermal.factory import build_thermal_model
from energysim.core.data.factory import build_dataset

from dacite import from_dict, Config


@dataclass
class EnvironmentConfig:
    components: Dict[str, ComponentConfig]
    thermal_sensor: ThermalSensorConfig
    thermal_model: ThermalModelConfig
    dataset: EnergyDatasetConfig
    reward_manager: RewardConfig
    params: EnvironmentParameters


class EnvironmentFactory:
    @staticmethod
    def create_environment(config: EnvironmentConfig):
        """Create environment with both local and remote components."""

        components = {
            name: build_component(cfg) for name, cfg in config.components.items()
        }
        component_sensors = {
            name: ComponentSensor(cfg)
            for name, cfg in zip(
                config.components.keys(),
                [cfg.sensor for cfg in config.components.values()],
            )
        }
        thermal_sensor = ThermalSensor(config.thermal_sensor)
        thermal_model = build_thermal_model(config.thermal_model)
        dataset = build_dataset(config.dataset)
        reward_manager = RewardManagerFactory.create(config.reward_manager)

        return BuildingEnvironment(
            components=components,
            comp_sensors=component_sensors,
            thermal_sensor=thermal_sensor,
            dataset=dataset,
            thermal_model=thermal_model,
            reward_manager=reward_manager,
            params=config.params,
        )


if __name__ == "__main__":
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
    actuator:
      type: pi
      space:
        action:
          type: continuous
          lower_bound: -1.0
          upper_bound: 1.0

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
    actuator:
      type: simple
      space:
        action:
          type: discrete
          n_actions: 3           
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
      action:
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
    actuator:
      type: simple
      space:
        action:
          type: discrete
          n_actions: 3
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
    import yaml
    import logging

    logging.basicConfig(level=logging.DEBUG)

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
