"""Tests for reward manager implementation."""
from unittest.mock import Mock
from energysim.reward.manager import RewardManager
from energysim.reward.layers.base import RewardLayer
from energysim.reward.contexts import RewardContext


class MockRewardLayer(RewardLayer):
    def __init__(self, reward_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reward_value = reward_value
    
    def calculate_reward(self, context: RewardContext) -> float:
        return self.reward_value


def test_reward_manager_initialization():
    # Arrange & Act
    manager = RewardManager("TestManager")
    
    # Assert
    assert manager.name == "TestManager"
    assert len(manager.layers) == 0
    assert len(manager.layer_order) == 0
    assert manager._last_reward_breakdown == {}


def test_reward_manager_default_name():
    # Arrange & Act
    manager = RewardManager()
    
    # Assert
    assert manager.name == "RewardManager"


def test_add_layer_without_id():
    # Arrange
    manager = RewardManager()
    layer = MockRewardLayer(reward_value=2.0, name="TestLayer")
    
    # Act
    manager.add_layer(layer)
    
    # Assert
    assert "TestLayer" in manager.layers
    assert manager.layers["TestLayer"] == layer
    assert manager.layer_order == ["TestLayer"]


def test_add_layer_with_custom_id():
    # Arrange
    manager = RewardManager()
    layer = MockRewardLayer(reward_value=2.0, name="TestLayer")
    
    # Act
    manager.add_layer(layer, layer_id="custom_id")
    
    # Assert
    assert "custom_id" in manager.layers
    assert manager.layers["custom_id"] == layer
    assert manager.layer_order == ["custom_id"]


def test_add_multiple_layers():
    # Arrange
    manager = RewardManager()
    layer1 = MockRewardLayer(reward_value=1.0, name="Layer1")
    layer2 = MockRewardLayer(reward_value=2.0, name="Layer2")
    
    # Act
    manager.add_layer(layer1)
    manager.add_layer(layer2)
    
    # Assert
    assert len(manager.layers) == 2
    assert manager.layer_order == ["Layer1", "Layer2"]


def test_add_same_layer_id_twice():
    # Arrange
    manager = RewardManager()
    layer1 = MockRewardLayer(reward_value=1.0, name="SameLayer")
    layer2 = MockRewardLayer(reward_value=2.0, name="SameLayer")
    
    # Act
    manager.add_layer(layer1)
    manager.add_layer(layer2)  # Same layer ID
    
    # Assert
    assert len(manager.layers) == 1  # Should overwrite
    assert len(manager.layer_order) == 1  # Should not duplicate
    assert manager.layers["SameLayer"] == layer2  # Should be the second layer


def test_calculate_reward_single_layer():
    # Arrange
    manager = RewardManager()
    layer = MockRewardLayer(reward_value=5.0, weight=2.0)
    manager.add_layer(layer)
    
    context = Mock(spec=RewardContext)
    
    # Act
    total_reward, breakdown = manager.calculate_reward(context)
    
    # Assert
    assert total_reward == 10.0  # 5.0 * 2.0
    assert len(breakdown) == 1
    assert breakdown[layer.name] == 10.0


def test_calculate_reward_multiple_layers():
    # Arrange
    manager = RewardManager()
    layer1 = MockRewardLayer(reward_value=3.0, weight=1.0, name="Layer1")
    layer2 = MockRewardLayer(reward_value=2.0, weight=2.5, name="Layer2")
    layer3 = MockRewardLayer(reward_value=-1.0, weight=3.0, name="Layer3")
    
    manager.add_layer(layer1)
    manager.add_layer(layer2)
    manager.add_layer(layer3)
    
    context = Mock(spec=RewardContext)
    
    # Act
    total_reward, breakdown = manager.calculate_reward(context)
    
    # Assert
    expected_total = 3.0 * 1.0 + 2.0 * 2.5 + (-1.0) * 3.0  # 3 + 5 - 3 = 5
    assert total_reward == expected_total
    assert breakdown["Layer1"] == 3.0
    assert breakdown["Layer2"] == 5.0
    assert breakdown["Layer3"] == -3.0


def test_calculate_reward_with_disabled_layer():
    # Arrange
    manager = RewardManager()
    enabled_layer = MockRewardLayer(reward_value=4.0, weight=1.0, enabled=True, name="EnabledLayer")
    disabled_layer = MockRewardLayer(reward_value=6.0, weight=2.0, enabled=False, name="DisabledLayer")
    
    manager.add_layer(enabled_layer)
    manager.add_layer(disabled_layer)
    
    context = Mock(spec=RewardContext)
    
    # Act
    total_reward, breakdown = manager.calculate_reward(context)
    
    # Assert
    assert total_reward == 4.0  # Only enabled layer contributes
    assert breakdown["EnabledLayer"] == 4.0
    assert breakdown["DisabledLayer"] == 0.0


def test_get_last_reward_breakdown():
    # Arrange
    manager = RewardManager()
    layer = MockRewardLayer(reward_value=2.0, weight=3.0, name="TestLayer")
    manager.add_layer(layer)
    
    context = Mock(spec=RewardContext)
    manager.calculate_reward(context)
    
    # Act
    breakdown = manager.get_last_reward_breakdown()
    
    # Assert
    assert breakdown["TestLayer"] == 6.0


def test_calculate_reward_updates_last_breakdown():
    # Arrange
    manager = RewardManager()
    layer1 = MockRewardLayer(reward_value=1.0, name="Layer1")
    layer2 = MockRewardLayer(reward_value=2.0, name="Layer2")
    
    manager.add_layer(layer1)
    manager.add_layer(layer2)
    
    context = Mock(spec=RewardContext)
    
    # Act
    manager.calculate_reward(context)
    breakdown1 = manager.get_last_reward_breakdown()
    
    # Change layer values and calculate again
    layer1.reward_value = 10.0
    layer2.reward_value = 20.0
    manager.calculate_reward(context)
    breakdown2 = manager.get_last_reward_breakdown()
    
    # Assert
    assert breakdown1["Layer1"] == 1.0
    assert breakdown1["Layer2"] == 2.0
    assert breakdown2["Layer1"] == 10.0
    assert breakdown2["Layer2"] == 20.0


def test_calculate_reward_empty_manager():
    # Arrange
    manager = RewardManager()
    context = Mock(spec=RewardContext)
    
    # Act
    total_reward, breakdown = manager.calculate_reward(context)
    
    # Assert
    assert total_reward == 0.0
    assert breakdown == {}