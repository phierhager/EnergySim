"""Tests for component factory system."""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
from bems_simulation.core.components.factory import build_component, build_components
from bems_simulation.core.components.shared.component_base import ComponentBase
from bems_simulation.core.components.shared.component_config import BaseComponentConfig
from bems_simulation.core.components.local.shared.config import BaseLocalComponentConfig
from bems_simulation.core.components.remote.shared.config import (
    BaseRemoteComponentConfig,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MockSensorConfig:
    observe_electrical_soc: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class MockLocalConfig(BaseLocalComponentConfig):
    model: object = None
    type: str = "mock_local"
    actuator: object = None
    sensor: MockSensorConfig = field(default_factory=MockSensorConfig)


@dataclass(frozen=True, slots=True, kw_only=True)
class MockRemoteConfig(BaseRemoteComponentConfig):
    connection: object = None
    type: str = "mock_remote"
    action_space: dict = field(default_factory=dict)
    sensor: MockSensorConfig = field(default_factory=MockSensorConfig)


class MockLocalComponent(ComponentBase):
    def __init__(self, model=None, actuator=None):
        self.model = model
        self.actuator = actuator

    def initialize(self):
        return Mock()

    def advance(self, input, dt_seconds):
        return Mock()

    @property
    def action_space(self):
        return {}


class MockRemoteComponent(ComponentBase):
    def __init__(self, connection=None, action_space=None):
        self.connection = connection
        self._action_space = action_space

    def initialize(self):
        return Mock()

    def advance(self, input, dt_seconds):
        return Mock()

    @property
    def action_space(self):
        return self._action_space or {}


@patch("bems_simulation.core.components.factory.registry")
@patch("bems_simulation.core.components.factory.local_component_factory")
def test_build_local_component(mock_local_factory, mock_registry):
    # Arrange
    config = MockLocalConfig()
    mock_registry.components = {MockLocalConfig.__name__: (MockLocalComponent, True)}
    mock_component_instance = MockLocalComponent()
    mock_factory_fn = Mock(return_value=mock_component_instance)
    mock_local_factory.return_value = mock_factory_fn

    # Act
    result = build_component(config)

    # Assert
    mock_local_factory.assert_called_once_with(MockLocalComponent)
    mock_factory_fn.assert_called_once_with(config)
    assert result == mock_component_instance


@patch("bems_simulation.core.components.factory.registry")
@patch("bems_simulation.core.components.factory.remote_component_factory")
def test_build_remote_component(mock_remote_factory, mock_registry):
    # Arrange
    config = MockRemoteConfig()
    mock_registry.components = {MockRemoteConfig.__name__: (MockRemoteComponent, False)}
    mock_component_instance = MockRemoteComponent()
    mock_factory_fn = Mock(return_value=mock_component_instance)
    mock_remote_factory.return_value = mock_factory_fn

    # Act
    result = build_component(config)

    # Assert
    mock_remote_factory.assert_called_once_with(MockRemoteComponent)
    mock_factory_fn.assert_called_once_with(config)
    assert result == mock_component_instance


def test_build_component_raises_for_unknown_config():
    # Arrange
    @dataclass(frozen=True, slots=True, kw_only=True)
    class UnknownConfig(BaseComponentConfig):
        type: str = "unknown"
        action_space: object = None
        sensor: MockSensorConfig = field(default_factory=MockSensorConfig)

    config = UnknownConfig()

    # Act & Assert
    with pytest.raises(KeyError):
        build_component(config)


@patch("bems_simulation.core.components.factory.build_component")
def test_build_components(mock_build_component):
    # Arrange
    config1 = MockLocalConfig()
    config2 = MockRemoteConfig()
    configs = [config1, config2]

    component1 = MockLocalComponent()
    component2 = MockRemoteComponent()
    mock_build_component.side_effect = [component1, component2]

    # Act
    result = list(build_components(configs))

    # Assert
    assert len(result) == 2
    assert result[0] == component1
    assert result[1] == component2
    assert mock_build_component.call_count == 2
