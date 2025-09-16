"""Tests for base reward layer implementation."""
import pytest
from unittest.mock import Mock
from energysim.rl.rewards.layers.base import RewardLayer
from energysim.rl.rewards.contexts import RewardContext


class ConcreteRewardLayer(RewardLayer):
    """Concrete implementation for testing."""
    
    def __init__(self, reward_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reward_value = reward_value
    
    def calculate_reward(self, context: RewardContext) -> float:
        return self.reward_value


def test_reward_layer_default_initialization():
    # Arrange & Act
    layer = ConcreteRewardLayer()
    
    # Assert
    assert layer.weight == 1.0
    assert layer.enabled is True
    assert layer.name == "ConcreteRewardLayer"
    assert layer.logger.name.endswith("ConcreteRewardLayer")


def test_reward_layer_custom_initialization():
    # Arrange & Act
    layer = ConcreteRewardLayer(
        weight=2.5,
        enabled=False,
        name="CustomLayer",
        reward_value=3.0
    )
    
    # Assert
    assert layer.weight == 2.5
    assert layer.enabled is False
    assert layer.name == "CustomLayer"
    assert layer.reward_value == 3.0


def test_calculate_reward_abstract_method():
    # Arrange & Act & Assert
    with pytest.raises(TypeError):
        # Cannot instantiate abstract base class directly
        RewardLayer()


def test_get_weighted_reward_when_enabled():
    # Arrange
    layer = ConcreteRewardLayer(reward_value=4.0, weight=2.5, enabled=True)
    context = Mock(spec=RewardContext)
    
    # Act
    weighted_reward = layer.get_weighted_reward(context)
    
    # Assert
    assert weighted_reward == 10.0  # 4.0 * 2.5


def test_get_weighted_reward_when_disabled():
    # Arrange
    layer = ConcreteRewardLayer(reward_value=4.0, weight=2.5, enabled=False)
    context = Mock(spec=RewardContext)
    
    # Act
    weighted_reward = layer.get_weighted_reward(context)
    
    # Assert
    assert weighted_reward == 0.0


def test_get_weighted_reward_with_negative_weight():
    # Arrange
    layer = ConcreteRewardLayer(reward_value=3.0, weight=-1.5, enabled=True)
    context = Mock(spec=RewardContext)
    
    # Act
    weighted_reward = layer.get_weighted_reward(context)
    
    # Assert
    assert weighted_reward == -4.5  # 3.0 * -1.5


def test_get_weighted_reward_with_zero_weight():
    # Arrange
    layer = ConcreteRewardLayer(reward_value=10.0, weight=0.0, enabled=True)
    context = Mock(spec=RewardContext)
    
    # Act
    weighted_reward = layer.get_weighted_reward(context)
    
    # Assert
    assert weighted_reward == 0.0


def test_get_info_when_enabled():
    # Arrange
    layer = ConcreteRewardLayer(
        reward_value=2.0,
        weight=3.0,
        enabled=True,
        name="TestLayer"
    )
    context = Mock(spec=RewardContext)
    
    # Act
    info = layer.get_info(context)
    
    # Assert
    assert info["name"] == "TestLayer"
    assert info["weight"] == 3.0
    assert info["enabled"] is True
    assert info["raw_reward"] == 2.0
    assert info["weighted_reward"] == 6.0


def test_get_info_when_disabled():
    # Arrange
    layer = ConcreteRewardLayer(
        reward_value=2.0,
        weight=3.0,
        enabled=False,
        name="TestLayer"
    )
    context = Mock(spec=RewardContext)
    
    # Act
    info = layer.get_info(context)
    
    # Assert
    assert info["name"] == "TestLayer"
    assert info["weight"] == 3.0
    assert info["enabled"] is False
    assert info["raw_reward"] == 0.0  # Should be 0 when disabled
    assert info["weighted_reward"] == 0.0


def test_get_info_structure():
    # Arrange
    layer = ConcreteRewardLayer()
    context = Mock(spec=RewardContext)
    
    # Act
    info = layer.get_info(context)
    
    # Assert
    required_keys = {"name", "weight", "enabled", "raw_reward", "weighted_reward"}
    assert set(info.keys()) == required_keys