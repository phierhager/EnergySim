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