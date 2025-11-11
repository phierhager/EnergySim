from energysim.reward.layers.base import RewardLayer
from energysim.reward.contexts import (
    RewardContext,
)
from energysim.reward.calculators import (
    ComfortMetricsCalculator,
)


class ComfortRewardLayer(RewardLayer):
    """Comfort reward layer without caching."""

    def __init__(
        self,
        weight: float = 1.0,
        temperature_comfort_band: float = 2.0,
        humidity_comfort_band: float = 0.1,
        max_penalty: float = 10.0,
        enabled: bool = True,
    ):
        super().__init__(weight, enabled, "comfort")
        self.temperature_comfort_band = temperature_comfort_band
        self.humidity_comfort_band = humidity_comfort_band
        self.max_penalty = max_penalty

    def calculate_reward(self, context: RewardContext) -> float:
        comfort = context.thermal_state
        if (
            comfort.temperature_error <= self.temperature_comfort_band
            and comfort.humidity_error <= self.humidity_comfort_band
        ):
            return 0.0
        penalties = ComfortMetricsCalculator.calculate_comfort_penalty(
            comfort,
            self.temperature_comfort_band,
            self.humidity_comfort_band,
            self.max_penalty,
        )
        return -(penalties["temperature_penalty"] + penalties["humidity_penalty"])
