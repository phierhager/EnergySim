from typing import Dict, Optional
from energysim.reward.contexts import RewardContext
from energysim.reward.layers import RewardLayer


class RewardManager:
    """Reward manager without caching."""

    def __init__(self, name: str = "RewardManager"):
        self.name = name
        self.layers: Dict[str, RewardLayer] = {}
        self.layer_order: list[str] = []
        self._last_reward_breakdown: Dict[str, float] = {}

    def add_layer(self, layer: RewardLayer, layer_id: Optional[str] = None) -> None:
        layer_id = layer_id or layer.name
        self.layers[layer_id] = layer
        if layer_id not in self.layer_order:
            self.layer_order.append(layer_id)

    def calculate_reward(self, context: RewardContext) -> tuple[float, dict]:
        total_reward = 0.0
        self._last_reward_breakdown = {}
        for layer_id in self.layer_order:
            layer = self.layers[layer_id]
            layer_reward = layer.get_weighted_reward(context) if layer.enabled else 0.0
            total_reward += layer_reward
            self._last_reward_breakdown[layer_id] = layer_reward
        return total_reward, self._last_reward_breakdown

    def get_last_reward_breakdown(self) -> Dict[str, float]:
        return self._last_reward_breakdown
