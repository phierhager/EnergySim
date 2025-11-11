"""Calculation services for extracting KPIs from domain objects."""


from energysim.core.thermal.base import ThermalState
from energysim.core.components.outputs import ComponentOutputs
from energysim.reward.metrics import (
    EnergyMetrics,
    EfficiencyMetrics,
    PeakDemandMetrics,
)


class EnergyMetricsCalculator:
    """Calculates energy-related KPIs from ComponentOutputs and economic context."""

    @staticmethod
    def calculate_renewable_fraction(component_outputs: ComponentOutputs) -> float:
        demand = component_outputs.electrical_energy.demand_j
        generation = component_outputs.electrical_energy.generation_j
        if demand <= 0:
            return 0.0
        return min(1.0, generation / demand)

    @staticmethod
    def calculate_self_consumption_ratio(component_outputs: ComponentOutputs) -> float:
        demand = component_outputs.electrical_energy.demand_j
        generation = component_outputs.electrical_energy.generation_j
        if generation <= 0:
            return 0.0
        return min(1.0, demand / generation)

    @staticmethod
    def calculate_curtailment(component_outputs: ComponentOutputs) -> float:
        demand = component_outputs.electrical_energy.demand_j
        generation = component_outputs.electrical_energy.generation_j
        return max(0.0, generation - demand)

    @staticmethod
    def calculate_carbon_emissions(
        component_outputs: ComponentOutputs, grid_emission_factor: float = 0.4e-6
    ) -> float:
        net_consumption = max(0.0, component_outputs.electrical_energy.net)
        return net_consumption * grid_emission_factor

    @staticmethod
    def get_energy_metrics(
        component_outputs: ComponentOutputs,
        grid_emission_factor: float = 0.4e-6,
    ) -> EnergyMetrics:
        return EnergyMetrics(
            net_energy_j=component_outputs.electrical_energy.net,
            energy_cost_eur=EnergyMetricsCalculator.calculate_energy_cost(
                component_outputs.electrical_energy.net
            ),
            renewable_fraction=EnergyMetricsCalculator.calculate_renewable_fraction(
                component_outputs
            ),
            self_consumption_ratio=EnergyMetricsCalculator.calculate_self_consumption_ratio(
                component_outputs
            ),
            curtailment_j=EnergyMetricsCalculator.calculate_curtailment(
                component_outputs
            ),
            carbon_emission_kg=EnergyMetricsCalculator.calculate_carbon_emissions(
                component_outputs, grid_emission_factor
            ),
            net_heating_j=component_outputs.thermal_energy.net_heating,
        )

    @staticmethod
    def calculate_energy_cost(tax_rate: float, price_eur_per_j: float, net_energy_j: float, feed_in_tariff_eur_per_j: float) -> float:
            """
            Calculate energy cost based on net energy consumption.

            Args:
                net_energy_j: Net energy in joules (positive = consumption, negative = generation)

            Returns:
                Cost in euros (positive = cost, negative = revenue)
            """
            if net_energy_j > 0:
                # Energy consumption
                base_cost = net_energy_j * price_eur_per_j
                return base_cost * (1 + tax_rate)
            else:
                # Energy generation (feed-in)
                return net_energy_j * feed_in_tariff_eur_per_j  # Negative value = revenue

    @staticmethod
    def calculate_demand_charge(demand_charge_threshold_j: float, demand_charge_rate_eur_per_j: float, peak_power_w: float, duration_s: float) -> float:
        """Calculate demand charges for peak power usage."""
        if peak_power_w <= demand_charge_rate_eur_per_j * 1000:  # Convert kW to W
            return 0.0

        excess_power_w = peak_power_w - (demand_charge_threshold_j * 1000)
        excess_energy_j = excess_power_w * duration_s
        return excess_energy_j * demand_charge_rate_eur_per_j

class ComfortMetricsCalculator:
    """Calculates comfort-related KPIs from comfort metrics."""

    @staticmethod
    def calculate_comfort_penalty(
        thermal_state: ThermalState,
        temperature_comfort_band: float = 2.0,
        humidity_comfort_band: float = 0.1,
        max_penalty: float = 10.0,
    ) -> dict[str, float]:
        """
        Calculate comfort penalties for temperature and humidity deviations.

        Args:
            thermal_state: Comfort measurements and setpoints
            temperature_comfort_band: Acceptable temperature deviation in °C
            humidity_comfort_band: Acceptable humidity deviation (fraction)
            max_penalty: Maximum penalty per comfort type

        Returns:
            Dictionary with comfort errors and penalties
        """
        # Temperature penalty
        temp_penalty = 0.0
        if thermal_state.temperature_error > temperature_comfort_band:
            excess_temp_error = (
                thermal_state.temperature_error - temperature_comfort_band
            )
            temp_penalty = min(excess_temp_error**2, max_penalty)

        # Humidity penalty
        humidity_penalty = 0.0
        if thermal_state.humidity_error > humidity_comfort_band:
            excess_humidity_error = thermal_state.humidity_error - humidity_comfort_band
            humidity_penalty = min(
                excess_humidity_error**2 * 100, max_penalty
            )  # Scale humidity penalty

        return {
            "temperature_penalty": temp_penalty,
            "humidity_penalty": humidity_penalty,
        }

    @staticmethod
    def calculate_thermal_comfort_index(thermal_state: ThermalState) -> float:
        """
        Calculate simplified thermal comfort index (0.0 = perfect, 1.0 = max discomfort).

        Args:
            thermal_state: Comfort measurements

        Returns:
            Thermal comfort index between 0.0 and 1.0
        """
        # Simplified comfort index based on temperature and humidity deviations
        temp_factor = min(
            1.0, thermal_state.temperature_error / 5.0
        )  # Max discomfort at 5°C deviation
        humidity_factor = min(
            1.0, thermal_state.humidity_error / 0.3
        )  # Max discomfort at 30% humidity deviation

        # Weighted combination (temperature more important)
        return 0.7 * temp_factor + 0.3 * humidity_factor


class EfficiencyMetricsCalculator:
    """Calculates system efficiency KPIs using ComponentOutputs."""

    @staticmethod
    def calculate_system_efficiency(
        component_outputs: ComponentOutputs,
    ) -> EfficiencyMetrics:
        total_generation = (
            component_outputs.electrical_energy.generation_j
            + component_outputs.thermal_energy.heating_j
            + component_outputs.thermal_energy.cooling_j
        )
        total_demand = component_outputs.electrical_energy.demand_j

        generation_efficiency = (
            min(1.0, total_demand / total_generation) if total_generation > 0 else 0.0
        )

        total_stored_energy = 0.0
        if component_outputs.electrical_storage is not None:
            total_stored_energy += (
                component_outputs.electrical_storage.soc
                * component_outputs.electrical_storage.capacity
            )
        if component_outputs.thermal_storage is not None:
            total_stored_energy += (
                component_outputs.thermal_storage.soc
                * component_outputs.thermal_storage.capacity
            )

        return EfficiencyMetrics(
            generation_efficiency=generation_efficiency,
            total_generation_j=total_generation,
            total_demand_j=total_demand,
            total_stored_energy_j=total_stored_energy,
            energy_balance_j=total_generation - total_demand,
        )

    @staticmethod
    def calculate_peak_demand_metrics(
        component_outputs: ComponentOutputs, peak_demand_threshold: float
    ) -> PeakDemandMetrics:
        demand = component_outputs.electrical_energy.demand_j
        is_peak_exceeded = demand > peak_demand_threshold
        excess_demand = max(0.0, demand - peak_demand_threshold)

        return PeakDemandMetrics(
            is_peak_demand_exceeded=is_peak_exceeded,
            peak_demand_threshold_j=peak_demand_threshold,
            current_demand_j=demand,
            excess_demand_j=excess_demand,
            peak_demand_ratio=(demand / peak_demand_threshold)
            if peak_demand_threshold > 0
            else 0.0,
        )
