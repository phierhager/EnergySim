from energysim.core.shared.mpc_interfaces import MPCBuilderBase
from energysim.core.thermal.base import (
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

    def add_mpc_dynamics_constraints(self, builder: MPCBuilderBase, k: int, states, actions, exogenous):
        # T_air(k+1) = f(T_air(k), Q_hvac, T_ambient)
        T_k = states['room_temp'][k]
        T_k_plus_1 = states['room_temp'][k+1]
        T_amb = exogenous['ambient_temperature'][k] # from forecast

        # The thermal input is a calculated value from the component actions
        # e.g., Q_hvac_j is a symbolic expression derived in SystemDynamics (or passed here)
        # Assuming Q_hvac_j is computed from actions['hvac_power'][k]
        # Q_hvac_j = actions['hvac_power'][k] * builder.dt_seconds * HVAC_COP_k

        # This requires a *coupling variable* or explicit knowledge of the thermal energy input
        # from all components (hvac, stove, etc.) in the thermal model.

        # For simplicity, let's assume one coupling variable:
        Q_net_thermal = builder.coupling_vars['net_thermal_energy'][k] # Must be defined earlier

        # Symbolic ODE for SimpleAirModel (discretized Euler)
        C_therm = self.air_mass * self.config.specific_heat_air
        R_therm = 1.0 / self.config.heat_transfer_coefficient # R = 1/U-value

        dT = (Q_net_thermal / C_therm) - ((T_k - T_amb) * builder.dt_seconds / (R_therm * C_therm))
        builder.add_constraint(T_k_plus_1 == T_k + dT)

    def add_mpc_operational_constraints(self, builder, k, states, actions, exogenous):
         # Comfort bounds
         T_max = builder.get_param('T_max')
         T_min = builder.get_param('T_min')
         builder.add_constraint(states['room_temp'][k] <= T_max)
         builder.add_constraint(states['room_temp'][k] >= T_min)