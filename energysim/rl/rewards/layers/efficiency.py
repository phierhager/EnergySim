from energysim.rl.rewards.layers.base import RewardLayer
from energysim.rl.rewards.contexts import RewardContext
from energysim.rl.rewards.calculators import (
    EfficiencyMetricsCalculator,
    EnergyMetricsCalculator,
    ComfortMetricsCalculator,
)


class EfficiencyRewardLayer(RewardLayer):
    """Efficiency reward layer without caching."""

    def __init__(
        self,
        weight: float = 1.0,
        renewable_bonus: float = 0.05,
        self_consumption_bonus: float = 0.02,
        enabled: bool = True,
    ):
        super().__init__(weight, enabled, "EfficiencyReward")
        self.renewable_bonus = renewable_bonus
        self.self_consumption_bonus = self_consumption_bonus

    def calculate_reward(self, context: RewardContext) -> float:
        energy_metrics = EnergyMetricsCalculator.get_energy_metrics(
            context.component_outputs, context.economic_context
        )
        renewable_reward = energy_metrics.renewable_fraction * self.renewable_bonus
        self_consumption_reward = (
            energy_metrics.self_consumption_ratio * self.self_consumption_bonus
        )
        return renewable_reward + self_consumption_reward
