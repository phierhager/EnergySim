from dataclasses import dataclass
from energysim.core.components.shared.component_outputs import ComponentOutputs
from energysim.core.thermal.thermal_model_base import ThermalState


@dataclass
class EconomicContext:
    """Economic pricing and tariff information."""

    # Basic pricing
    electricity_price: float = 2.5e-7  # €/J for energy consumption
    feed_in_tariff: float = 1.0e-7  # €/J for energy feed-in
    tax_rate: float = 0.19  # Tax rate (19%)

    # Demand charges (peak pricing)
    demand_charge_threshold: float = 50.0  # kW threshold
    demand_charge_rate: float = 0.1  # €/J for excess demand

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.demand_charge_threshold <= 0:
            raise ValueError("demand_charge_threshold must be positive")

    def calculate_energy_cost(self, net_energy_j: float) -> float:
        """
        Calculate energy cost based on net energy consumption.

        Args:
            net_energy_j: Net energy in joules (positive = consumption, negative = generation)

        Returns:
            Cost in euros (positive = cost, negative = revenue)
        """
        if net_energy_j > 0:
            # Energy consumption
            base_cost = net_energy_j * self.electricity_price
            return base_cost * (1 + self.tax_rate)
        else:
            # Energy generation (feed-in)
            return net_energy_j * self.feed_in_tariff  # Negative value = revenue

    def calculate_demand_charge(self, peak_power_w: float, duration_s: float) -> float:
        """Calculate demand charges for peak power usage."""
        if peak_power_w <= self.demand_charge_threshold * 1000:  # Convert kW to W
            return 0.0

        excess_power_w = peak_power_w - (self.demand_charge_threshold * 1000)
        excess_energy_j = excess_power_w * duration_s
        return excess_energy_j * self.demand_charge_rate

    def update_price(self, new_price: float) -> None:
        """Update the electricity price."""
        if new_price <= 0:
            raise ValueError("Electricity price must be positive.")
        self.electricity_price = new_price


@dataclass
class RewardContext:
    """
    Unified container for system state for reward calculation.
    """

    component_outputs: ComponentOutputs
    thermal_state: ThermalState
    economic_context: EconomicContext
