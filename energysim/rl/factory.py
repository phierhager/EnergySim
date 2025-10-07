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

from energysim.rl.wrappers.action_discretizer import ActionDiscretizerWrapper
from energysim.rl.wrappers.action_flattener import ActionFlattenerWrapper


from dacite import from_dict, Config
import gymnasium as gym
import numpy as np

@dataclass
class EnvironmentConfig:
    components: Dict[str, ComponentConfig]
    thermal_sensor: ThermalSensorConfig
    thermal_model: ThermalModelConfig
    dataset: EnergyDatasetConfig
    reward_manager: RewardConfig
    params: EnvironmentParameters
    wrappers: dict = None  # Optional wrappers to apply


class EnvironmentFactory:
    @staticmethod
    def create_environment(config: EnvironmentConfig) -> gym.Env:
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

        building_env = BuildingEnvironment(
            components=components,
            comp_sensors=component_sensors,
            thermal_sensor=thermal_sensor,
            dataset=dataset,
            thermal_model=thermal_model,
            reward_manager=reward_manager,
            params=config.params,
        )

        if not config.wrappers:
            return building_env
        
        # 3. Apply wrappers (Decorator Pattern)
        #    The order of wrapping can be important. Generally, action wrappers
        #    should be last (outermost) and observation wrappers first (innermost).
        wrapped_env = building_env

        misc_wrapper_config = config.wrappers.get("misc", {})
        if "max_episode_steps" in misc_wrapper_config:
            wrapped_env = gym.wrappers.TimeLimit(wrapped_env, max_episode_steps=misc_wrapper_config["max_episode_steps"])


        # NOTE: Always apply flattening of Dict observation spaces first
        wrapped_env = gym.wrappers.FlattenObservation(wrapped_env)

        observation_wrapper_config = config.wrappers.get("observation_space", {})
        if "noise_std" in observation_wrapper_config:
            def noise(obs):
                return obs + np.random.normal(0, observation_wrapper_config["noise_std"], size=obs.shape)
            wrapped_env = gym.wrappers.TransformObservation(wrapped_env, func=noise, observation_space=wrapped_env.observation_space)
        if "framestack_size" in observation_wrapper_config and observation_wrapper_config["framestack_size"] > 1:
            wrapped_env = gym.wrappers.FrameStackObservation(wrapped_env, observation_wrapper_config["framestack_size"])
        if "time_aware" in observation_wrapper_config and observation_wrapper_config["time_aware"]:
            wrapped_env = gym.wrappers.TimeAwareObservation(wrapped_env)
        if "normalize" in observation_wrapper_config and observation_wrapper_config["normalize"]:
            wrapped_env = gym.wrappers.NormalizeObservation(wrapped_env, epsilon=1e-8)


        action_wrapper_config = config.wrappers.get("action_space", {})
        if "discrete_bins" in action_wrapper_config:
            wrapped_env = ActionDiscretizerWrapper(wrapped_env, action_wrapper_config["discrete_bins"])

        # NOTE: Always apply flattening of Dict action spaces before clipping
        wrapped_env = ActionFlattenerWrapper(wrapped_env)
        
        if "clip_actions" in action_wrapper_config and action_wrapper_config["clip_actions"]:
            wrapped_env = gym.wrappers.ClipAction(wrapped_env)
            
        print(f"Successfully created environment. Final wrapped env: {wrapped_env}")
        return wrapped_env
