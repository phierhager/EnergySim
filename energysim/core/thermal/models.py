from energysim.core.thermal.thermal_model_base import (
    ThermalModel,
    ThermalState,
)
from energysim.core.thermal.config import SimpleAirModelConfig
from energysim.core.thermal.registry import register


@register(SimpleAirModelConfig)
class SimpleAirModel(ThermalModel):
    """
    Simple air thermal model with no thermal mass.

    This model treats the building as a single air volume with
    instantaneous temperature response to thermal inputs.
    """

    def __init__(self, config: SimpleAirModelConfig):
        super().__init__(config)

        # Calculate thermal properties
        self.air_mass = config.building_volume * config.air_density  # kg
        self.thermal_capacity = self.air_mass * config.specific_heat_air  # J/°C

    def advance(self, thermal_energy_j: float, ambient_temperature: float, dt_seconds: float) -> ThermalState:
        """
        Update temperature based on thermal energy input.

        Uses the formula: ΔT = Q / (m × c)
        where Q is thermal energy, m is air mass, c is specific heat
        """
        # Calculate temperature change from thermal energy
        temp_change = thermal_energy_j / self.thermal_capacity

        # Calculate heat loss to ambient (simple model)
        temp_diff_to_ambient = self.state.temperature - ambient_temperature

        # Heat loss rate (W/°C) - simplified model
        heat_loss_rate = self.config.heat_transfer_coefficient
        heat_loss_j = heat_loss_rate * temp_diff_to_ambient * dt_seconds
        heat_loss_temp_change = -heat_loss_j / self.thermal_capacity

        # Update temperature
        new_temperature = self.state.temperature + temp_change + heat_loss_temp_change

        # Calculate heating/cooling demand to reach setpoint
        temp_error = self.state.temperature_setpoint - new_temperature
        heating_demand = max(0.0, temp_error * self.thermal_capacity)
        cooling_demand = max(0.0, -temp_error * self.thermal_capacity)

        # Update state
        self.state = ThermalState(
            temperature=new_temperature,
            temperature_setpoint=self.state.temperature_setpoint,
            heating_demand=heating_demand,
            cooling_demand=cooling_demand,
        )

        return self.state

    def initialize(self) -> ThermalState:
        """Reset to initial temperature."""
        self.state = ThermalState(
            temperature=self.config.initial_temperature,
            temperature_setpoint=self.config.temperature_setpoint,
        )
        return self.state
