from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
from energysim.rl.rewards.contexts import RewardContext


class RewardLayer(ABC):
    """Abstract base class for reward layers."""

    def __init__(
        self, weight: float = 1.0, enabled: bool = True, name: Optional[str] = None
    ):
        self.weight = weight
        self.enabled = enabled
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def calculate_reward(self, context: RewardContext) -> float:
        pass

    def get_weighted_reward(self, context: RewardContext) -> float:
        if not self.enabled:
            return 0.0
        return self.calculate_reward(context) * self.weight

    def get_info(self, context: RewardContext) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "enabled": self.enabled,
            "raw_reward": self.calculate_reward(context) if self.enabled else 0.0,
            "weighted_reward": self.get_weighted_reward(context),
        }
