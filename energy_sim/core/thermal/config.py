from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, slots=True, kw_only=True)
class SimpleAirModelConfig:
    # Building parameters
    building_volume: float = 200.0  # m³
    initial_temperature: float = 20.0  # °C
    temperature_setpoint: float = 21.0  # °C

    # Simple air model parameters
    air_density: float = 1.2  # kg/m³
    specific_heat_air: float = 1005.0  # J/kg·°C

    # Thermal mass model parameters (when implemented)
    thermal_mass: float = 50000.0  # J/°C
    heat_transfer_coefficient: float = 100.0  # W/°C

    # External conditions
    ambient_temperature: float = 15.0  # °C

    # Model type
    type: str = "simple_air"

    def __post_init__(self) -> None:
        """Validate thermal model configuration."""
        if self.building_volume <= 0:
            raise ValueError("building_volume must be positive")
        if self.air_density <= 0:
            raise ValueError("air_density must be positive")
        if self.specific_heat_air <= 0:
            raise ValueError("specific_heat_air must be positive")
        if self.thermal_mass <= 0:
            raise ValueError("thermal_mass must be positive")
        if self.heat_transfer_coefficient <= 0:
            raise ValueError("heat_transfer_coefficient must be positive")

        if self.initial_temperature < -50 or self.initial_temperature > 50:
            raise ValueError(
                "initial_temperature should be reasonable for building environment"
            )


ThermalModelConfig = Union[SimpleAirModelConfig]
