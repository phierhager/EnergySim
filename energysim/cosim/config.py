from dataclasses import dataclass, field
from typing import Dict, Any, Literal
from energysim.core.components.config_types import ComponentConfig
from energysim.core.components.sensors import ThermalSensorConfig
from energysim.core.data.config import EnergyDatasetConfig
from energysim.core.thermal.config import ThermalModelConfig
from energysim.rl.building_environment import EnvironmentParameters
from energysim.rl.factory import EnvironmentConfig

@dataclass
class BuildingSimulationConfig:
    components: Dict[str, ComponentConfig]
    thermal_sensor: ThermalSensorConfig
    thermal_model: ThermalModelConfig
    dataset: EnergyDatasetConfig