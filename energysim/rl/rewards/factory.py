"""Class-level factory for simplified reward system creation."""

import logging

from energysim.rl.rewards.layers import (
    EnergyRewardLayer,
    ComfortRewardLayer,
    EfficiencyRewardLayer,
    GridStabilityRewardLayer,
)
from energysim.rl.rewards.manager import RewardManager
from energysim.rl.rewards.reward_config import EconomicConfig, RewardConfig


class RewardManagerFactory:
    """Class-level factory for creating RewardManager instances."""

    logger = logging.getLogger("RewardManagerFactory")

    @classmethod
    def create(
        cls, config: RewardConfig, economic_config: EconomicConfig, name: str = "BuildingEnvRewards"
    ) -> RewardManager:
        """
        Create a RewardManager configured according to the given RewardConfig.

        Args:
            config: Reward configuration
            name: Name for the reward system

        Returns:
            Configured RewardManager instance
        """
        manager = RewardManager(name)

        # Energy layer
        if config.energy_weight > 0:
            energy_layer = EnergyRewardLayer(weight=config.energy_weight, economic_config=economic_config, enabled=True)
            manager.add_layer(energy_layer, "energy")
            cls.logger.info(f"Added energy layer with weight {config.energy_weight}")

        # Comfort layer
        if config.comfort_weight > 0:
            comfort_layer = ComfortRewardLayer(
                weight=config.comfort_weight,
                temperature_comfort_band=config.temperature_comfort_band,
                humidity_comfort_band=config.humidity_comfort_band,
                max_penalty=config.max_comfort_penalty,
                enabled=True,
            )
            manager.add_layer(comfort_layer, "comfort")
            cls.logger.info(f"Added comfort layer with weight {config.comfort_weight}")

        # Efficiency layer
        if config.efficiency_weight > 0:
            efficiency_layer = EfficiencyRewardLayer(
                weight=config.efficiency_weight,
                renewable_bonus=config.renewable_bonus,
                self_consumption_bonus=config.self_consumption_bonus,
                enabled=True,
            )
            manager.add_layer(efficiency_layer, "efficiency")
            cls.logger.info(
                f"Added efficiency layer with weight {config.efficiency_weight}"
            )

        # Grid stability layer
        if config.grid_stability_weight > 0:
            grid_layer = GridStabilityRewardLayer(
                weight=config.grid_stability_weight,
                balance_tolerance=config.balance_tolerance,
                balance_bonus=config.balance_bonus,
                storage_bonus=config.storage_bonus,
                enabled=True,
            )
            manager.add_layer(grid_layer, "grid_stability")
            cls.logger.info(
                f"Added grid stability layer with weight {config.grid_stability_weight}"
            )

        return manager
