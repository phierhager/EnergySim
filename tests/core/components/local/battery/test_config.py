"""Tests for battery configuration validation."""

import pytest
from energysim.core.components.local.battery.config import (
    SimpleBatteryModelConfig,
    BatteryComponentConfig,
)
from energysim.core.components.shared.sensors import ComponentSensorConfig
from energysim.core.components.shared.spaces import DiscreteSpace


def test_simple_battery_model_config_valid():
    # Arrange & Act
    config = SimpleBatteryModelConfig(
        efficiency=0.9, max_power=10.0, capacity=50.0, init_soc=0.5, deadband=0.1
    )

    # Assert
    assert config.efficiency == 0.9
    assert config.max_power == 10.0
    assert config.capacity == 50.0
    assert config.init_soc == 0.5
    assert config.deadband == 0.1
    assert config.type == "simple"


def test_simple_battery_model_config_defaults():
    # Arrange & Act
    config = SimpleBatteryModelConfig()

    # Assert
    assert config.efficiency == 1.0
    assert config.max_power == 1.0
    assert config.capacity == 1.0
    assert config.init_soc == 0.0
    assert config.deadband == 0.0
    assert config.type == "simple"


def test_simple_battery_model_config_invalid_efficiency_too_low():
    # Act & Assert
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        SimpleBatteryModelConfig(efficiency=0.0)


def test_simple_battery_model_config_invalid_efficiency_too_high():
    # Act & Assert
    with pytest.raises(ValueError, match="Efficiency must be between 0 and 1"):
        SimpleBatteryModelConfig(efficiency=1.5)


def test_simple_battery_model_config_invalid_init_soc_too_low():
    # Act & Assert
    with pytest.raises(ValueError, match="Initial SoC must be between 0 and 1"):
        SimpleBatteryModelConfig(init_soc=-0.1)


def test_simple_battery_model_config_invalid_init_soc_too_high():
    # Act & Assert
    with pytest.raises(ValueError, match="Initial SoC must be between 0 and 1"):
        SimpleBatteryModelConfig(init_soc=1.1)


def test_simple_battery_model_config_invalid_max_power():
    # Act & Assert
    with pytest.raises(ValueError, match="Max power and capacity must be positive"):
        SimpleBatteryModelConfig(max_power=0.0)

    with pytest.raises(ValueError, match="Max power and capacity must be positive"):
        SimpleBatteryModelConfig(max_power=-1.0)


def test_simple_battery_model_config_invalid_capacity():
    # Act & Assert
    with pytest.raises(ValueError, match="Max power and capacity must be positive"):
        SimpleBatteryModelConfig(capacity=0.0)

    with pytest.raises(ValueError, match="Max power and capacity must be positive"):
        SimpleBatteryModelConfig(capacity=-1.0)


def test_simple_battery_model_config_invalid_deadband():
    # Act & Assert
    with pytest.raises(ValueError, match="Deadband must be non-negative"):
        SimpleBatteryModelConfig(deadband=-0.1)


def test_battery_component_config_valid():
    # Arrange
    model_config = SimpleBatteryModelConfig()
    sensor_config = ComponentSensorConfig()

    # Act
    config = BatteryComponentConfig(
        model=model_config, sensor=sensor_config
    )

    # Assert
    assert config.model == model_config
    assert config.sensor == sensor_config
    assert config.type == "battery"


def test_battery_component_config_frozen():
    # Arrange
    model_config = SimpleBatteryModelConfig()
    sensor_config = ComponentSensorConfig()

    config = BatteryComponentConfig(
        model=model_config, sensor=sensor_config
    )

    # Act & Assert - Should not be able to modify frozen dataclass
    with pytest.raises(AttributeError):
        config.type = "modified"
