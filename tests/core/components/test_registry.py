"""Tests for component registry system."""
import pytest
from energysim.core.components.registry import (
    registry,
    register_local_component,
    register_remote_component,
    register_model,
    register_actuator,
    register_connection,
)


class MockComponentConfig:
    pass


class MockModel:
    pass


class MockActuator:
    pass


class MockConnection:
    pass


def test_register_local_component():
    # Arrange
    initial_count = len(registry.components)
    
    # Act
    @register_local_component(MockComponentConfig)
    class TestLocalComponent:
        pass
    
    # Assert
    assert len(registry.components) == initial_count + 1
    assert MockComponentConfig.__name__ in registry.components
    component_cls, is_local = registry.components[MockComponentConfig.__name__]
    assert component_cls == TestLocalComponent
    assert is_local is True


def test_register_remote_component():
    # Arrange
    class RemoteComponentConfig:
        pass
    
    initial_count = len(registry.components)
    
    # Act
    @register_remote_component(RemoteComponentConfig)
    class TestRemoteComponent:
        pass
    
    # Assert
    assert len(registry.components) == initial_count + 1
    assert RemoteComponentConfig.__name__ in registry.components
    component_cls, is_local = registry.components[RemoteComponentConfig.__name__]
    assert component_cls == TestRemoteComponent
    assert is_local is False


def test_register_model():
    # Arrange
    class ModelConfig:
        pass
    
    initial_count = len(registry.models)
    
    # Act
    @register_model(ModelConfig)
    class TestModel:
        pass
    
    # Assert
    assert len(registry.models) == initial_count + 1
    assert ModelConfig.__name__ in registry.models
    assert registry.models[ModelConfig.__name__] == TestModel


def test_register_actuator():
    # Arrange
    class ActuatorConfig:
        pass
    
    initial_count = len(registry.actuators)
    
    # Act
    @register_actuator(ActuatorConfig)
    class TestActuator:
        pass
    
    # Assert
    assert len(registry.actuators) == initial_count + 1
    assert ActuatorConfig.__name__ in registry.actuators
    assert registry.actuators[ActuatorConfig.__name__] == TestActuator


def test_register_connection():
    # Arrange
    class ConnectionConfig:
        pass
    
    initial_count = len(registry.connections)
    
    # Act
    @register_connection(ConnectionConfig)
    class TestConnection:
        pass
    
    # Assert
    assert len(registry.connections) == initial_count + 1
    assert ConnectionConfig.__name__ in registry.connections
    assert registry.connections[ConnectionConfig.__name__] == TestConnection


def test_registry_singleton_behavior():
    # Arrange & Act
    from energysim.core.components.registry import registry as registry2
    
    # Assert
    assert registry is registry2