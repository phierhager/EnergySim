from energysim.cosim.config import BuildingSimulationConfig
from energysim.cosim.building_sim import BuildingSimulator

from energysim.core.components.factory import build_component
from energysim.core.thermal.factory import build_thermal_model
from energysim.core.data.factory import build_dataset


class SimulatorFactory:
    @staticmethod
    def create_simulator(config: BuildingSimulationConfig) -> BuildingSimulator:
        """Create simulator with components."""

        components = {
            name: build_component(cfg) for name, cfg in config.components.items()
        }
        thermal_model = build_thermal_model(config.thermal_model)
        dataset = build_dataset(config.dataset)

        return BuildingSimulator(
            components=components,
            thermal_model=thermal_model,
            dataset=dataset,
        )