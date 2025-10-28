from energysim.rl.rewards.layers.base import RewardLayer
from energysim.rl.rewards.contexts import RewardContext
from energysim.rl.rewards.calculators import (
    EnergyMetricsCalculator,
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
        super().__init__(weight, enabled, "efficiency")
        self.renewable_bonus = renewable_bonus
        self.self_consumption_bonus = self_consumption_bonus

    def calculate_reward(self, context: RewardContext) -> float:
        renewable_fraction = EnergyMetricsCalculator.calculate_renewable_fraction(
            context.system_balance
        )
        self_consumption_ratio = EnergyMetricsCalculator.calculate_self_consumption_ratio(
            context.system_balance
        )
        renewable_reward = renewable_fraction * self.renewable_bonus
        self_consumption_reward = (
            self_consumption_ratio * self.self_consumption_bonus
        )
        return renewable_reward + self_consumption_reward
