from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from energysim.core.thermal.config import ThermalModelConfig
from energysim.core.thermal.state import ThermalState


class ThermalModel(ABC):
    """
    Abstract base class for building thermal models.

    Thermal models simulate the thermal dynamics of buildings,
    including heat transfer, thermal mass effects, and HVAC interactions.
    """

    def __init__(self, config: ThermalModelConfig):
        """
        Initialize thermal model.

        Args:
            config: Thermal model configuration
        """
        self.config = config

        # Initialize thermal state
        self.state = ThermalState(
            temperature=config.initial_temperature,
            temperature_setpoint=config.temperature_setpoint,
        )

    @abstractmethod
    def advance(self, thermal_energy_j: float, dt_seconds: float) -> ThermalState:
        """
        Advance thermal model by one time step.

        Args:
            thermal_energy_j: Net thermal energy input (positive = heating, negative = cooling)

        Returns:
            Updated thermal state
        """
        pass

    @abstractmethod
    def initialize(self) -> ThermalState:
        """
        Reset thermal model to initial state.

        Returns:
            Initial thermal state
        """
        pass

    def get_thermal_demand(self, target_temperature: Optional[float] = None) -> float:
        """
        Calculate thermal energy demand to reach target temperature.

        Args:
            target_temperature: Target temperature (uses setpoint if None)

        Returns:
            Required thermal energy in joules (positive = heating, negative = cooling)
        """
        target = target_temperature or self.state.temperature_setpoint
        temp_diff = target - self.state.temperature

        # Simple approximation - actual implementation depends on thermal model
        # This is a basic estimate, specific models should override
        thermal_capacity = (
            self.config.building_volume
            * self.config.air_density
            * self.config.specific_heat_air
        )
        return temp_diff * thermal_capacity

    def get_observations(self) -> Dict[str, Any]:
        """Get thermal state as dictionary for observations."""
        return {
            "temperature": self.state.temperature,
            "temperature_setpoint": self.state.temperature_setpoint,
            "temperature_error": self.state.temperature_error,
            "heating_demand": self.state.heating_demand,
            "cooling_demand": self.state.cooling_demand,
        }

    def get_state_dict(self) -> Dict[str, float]:
        """Get thermal state as dictionary for internal use."""
        return asdict(self.state)
