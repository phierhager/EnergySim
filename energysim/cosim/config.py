from dataclasses import dataclass
from typing import Dict
from energysim.core.components.config_types import ComponentConfig
from energysim.core.components.sensors import ThermalSensorConfig
from energysim.core.data.config import EnergyDatasetConfig
from energysim.core.thermal.config import ThermalModelConfig

@dataclass
class BuildingSimulationConfig:
    components: Dict[str, ComponentConfig]
    thermal_sensor: ThermalSensorConfig
    thermal_model: ThermalModelConfig
    dataset: EnergyDatasetConfig