from dataclasses import dataclass


@dataclass(frozen=True)
class EconomicConfig:
    """Immutable configuration for economic calculations."""
    feed_in_tariff_eur_per_j: float = 1.0e-7
    tax_rate: float = 0.19
    demand_charge_threshold_j: float = 50.0
    demand_charge_rate_eur_per_j: float = 0.1

@dataclass(frozen=True)
class RewardConfig:
    """Configuration for modular reward system layers - BEHAVIOR ONLY."""

    # Layer weights (0.0 = disabled, >0.0 = enabled with weight)
    energy_weight: float = 1.0
    comfort_weight: float = 0.1
    efficiency_weight: float = 0.05
    grid_stability_weight: float = 0.02

    # Comfort layer parameters (behavioral thresholds)
    temperature_comfort_band: float = 2.0  # °C tolerance
    humidity_comfort_band: float = 0.1  # fraction tolerance
    max_comfort_penalty: float = 10.0  # max penalty value

    # Efficiency layer parameters (behavioral bonuses)
    renewable_bonus: float = 0.05  # bonus amount
    self_consumption_bonus: float = 0.02  # bonus amount

    # Grid stability layer parameters (behavioral thresholds)
    balance_tolerance: float = 1000.0  # J tolerance for grid balance
    balance_bonus: float = 0.02  # bonus amount
    storage_bonus: float = 0.01  # bonus amount

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.energy_weight < 0:
            raise ValueError("energy_weight must be non-negative")
        if self.comfort_weight < 0:
            raise ValueError("comfort_weight must be non-negative")
        if self.efficiency_weight < 0:
            raise ValueError("efficiency_weight must be non-negative")
        if self.grid_stability_weight < 0:
            raise ValueError("grid_stability_weight must be non-negative")

        if self.temperature_comfort_band <= 0:
            raise ValueError("temperature_comfort_band must be positive")
        if self.max_comfort_penalty <= 0:
            raise ValueError("max_comfort_penalty must be positive")
