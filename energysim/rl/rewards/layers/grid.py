from energysim.rl.rewards.layers.base import RewardLayer
from energysim.rl.rewards.contexts import RewardContext


class GridStabilityRewardLayer(RewardLayer):
    """Grid stability reward layer."""

    def __init__(
        self,
        weight: float = 1.0,
        balance_tolerance: float = 1000.0,
        balance_bonus: float = 0.02,
        storage_bonus: float = 0.01,
        enabled: bool = True,
    ):
        super().__init__(weight, enabled, "GridStabilityReward")
        self.balance_tolerance = balance_tolerance
        self.balance_bonus = balance_bonus
        self.storage_bonus = storage_bonus

    def calculate_reward(self, context: RewardContext) -> float:
        reward = 0.0
        if (
            abs(context.component_outputs.electrical_energy.net)
            < self.balance_tolerance
        ):
            reward += self.balance_bonus
        if context.component_outputs.electrical_storage.soc is not None:
            if 0.2 <= context.component_outputs.electrical_storage.soc <= 0.8:
                reward += self.storage_bonus
        return reward
