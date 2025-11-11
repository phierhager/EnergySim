from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging
from energysim.mpc.base import MPCObjectiveContributor
from energysim.reward.contexts import RewardContext
import casadi as ca

if TYPE_CHECKING:
    from energysim.mpc.builder import MPCBuilder


class RewardLayer(ABC, MPCObjectiveContributor):
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

    def add_mpc_objective_term(
        self,
        builder: MPCBuilder,
        k: int,
        states: Dict[str, ca.SX],
        actions: Dict[str, ca.SX],
        exogenous: Dict[str, ca.SX]
    ) -> ca.SX:
        """
        (Potentially abstract or provide default implementation)
        Return the symbolic cost term for this layer at timestep k.
        Remember: MPC minimizes cost, RL maximizes reward (usually cost = -reward).
        """
        # Default implementation returns zero cost
        return 0.0