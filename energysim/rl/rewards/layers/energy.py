from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

from energysim.rl.rewards.contexts import RewardContext
from energysim.rl.rewards.calculators import (
    EfficiencyMetricsCalculator,
    EnergyMetricsCalculator,
)
from energysim.rl.rewards.layers.base import RewardLayer
from energysim.rl.rewards.reward_config import EconomicConfig
from energysim.rl.data_column import DataColumn


class EnergyRewardLayer(RewardLayer):
    """Energy reward layer without caching."""

    def __init__(
        self,
        economic_config: EconomicConfig,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(weight, enabled, "energy")
        self.economic_config = economic_config

    def calculate_reward(self, context: RewardContext) -> float:
        energy_cost = EnergyMetricsCalculator.calculate_energy_cost(
            self.economic_config.tax_rate,
            context.simulation_state.timestep_data.features[DataColumn.PRICE],
            context.system_balance.electrical_energy.net,
            self.economic_config.feed_in_tariff_eur_per_j,
        )   
        peak_metrics = EfficiencyMetricsCalculator.calculate_peak_demand_metrics(
            context.system_balance,
            self.economic_config.demand_charge_threshold_j,
        )
        demand_penalty = (
            peak_metrics.excess_demand_j * self.economic_config.demand_charge_rate_eur_per_j
        )
        return -(energy_cost + demand_penalty)
