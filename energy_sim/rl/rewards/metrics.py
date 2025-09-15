from dataclasses import dataclass


# ---------------------------
# Energy Metrics Data Models
# ---------------------------


@dataclass(frozen=True)
class EnergyMetrics:
    net_energy_j: float  # Joules
    energy_cost_eur: float  # Euros
    renewable_fraction: float  # [0.0 - 1.0]
    self_consumption_ratio: float  # [0.0 - 1.0]
    curtailment_j: float  # Joules
    carbon_emission_kg: float  # Kilograms CO₂
    net_heating_j: float  # Joules


@dataclass(frozen=True)
class ComfortMetrics:
    temperature_error_c: float  # °C deviation
    humidity_error: float  # Fraction [0.0 - 1.0]
    temperature_penalty: float  # Penalty (unitless)
    humidity_penalty: float  # Penalty (unitless)
    total_comfort_penalty: float  # Penalty (unitless)


@dataclass(frozen=True)
class EfficiencyMetrics:
    generation_efficiency: float  # [0.0 - 1.0]
    total_generation_j: float  # Joules
    total_demand_j: float  # Joules
    total_stored_energy_j: float  # Joules
    energy_balance_j: float  # Joules


@dataclass(frozen=True)
class PeakDemandMetrics:
    is_peak_demand_exceeded: bool
    peak_demand_threshold_j: float  # Joules
    current_demand_j: float  # Joules
    excess_demand_j: float  # Joules
    peak_demand_ratio: float  # Dimensionless
