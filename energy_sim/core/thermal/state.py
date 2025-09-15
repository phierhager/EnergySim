from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(slots=True)
class ThermalState:
    """Current thermal state of the building."""

    temperature: float  # °C
    temperature_setpoint: float  # °C
    heating_demand: float = 0.0  # J
    cooling_demand: float = 0.0  # J

    # Optional extended state for advanced models
    thermal_mass_temperature: Optional[float] = None  # °C
    zone_temperatures: Optional[Dict[str, float]] = None  # °C per zone

    # TODO: For now, we ignore humidity control
    humidity: float = 0  # %RH
    humidity_setpoint: float = 0  # %RH

    @property
    def humidity_error(self) -> float:
        """Humidity error from setpoint."""
        return abs(self.humidity - self.humidity_setpoint)

    @property
    def temperature_error(self) -> float:
        """Temperature error from setpoint."""
        return abs(self.temperature - self.temperature_setpoint)

    @property
    def comfort_violation(self) -> bool:
        """Whether temperature is significantly outside comfort range."""
        return self.temperature_error > 3.0  # More than 3°C is uncomfortable
