from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

from bems_simulation.rl.rewards.contexts import RewardContext
from bems_simulation.rl.rewards.calculators import (
    EfficiencyMetricsCalculator,
    EnergyMetricsCalculator,
)
from bems_simulation.rl.rewards.layers.base import RewardLayer


class EnergyRewardLayer(RewardLayer):
    """Energy reward layer without caching."""

    def __init__(
        self,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(weight, enabled, "EnergyReward")

    def calculate_reward(self, context: RewardContext) -> float:
        energy_cost = EnergyMetricsCalculator.get_energy_metrics(
            context.component_outputs, context.economic_context
        ).energy_cost_eur
        peak_metrics = EfficiencyMetricsCalculator.calculate_peak_demand_metrics(
            context.component_outputs,
            context.economic_context.demand_charge_threshold,
        )
        demand_penalty = (
            peak_metrics.excess_demand_j * context.economic_context.demand_charge_rate
        )
        return -(energy_cost + demand_penalty)
