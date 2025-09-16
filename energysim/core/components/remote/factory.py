from typing import Callable
from energysim.core.components.registry import registry
from energysim.core.components.remote.shared.config import (
    BaseRemoteComponentConfig,
)
from energysim.core.components.remote.shared.component import RemoteComponent

RemoteFactory = Callable[[BaseRemoteComponentConfig], RemoteComponent]


def remote_component_factory(component_cls: type[RemoteComponent]) -> RemoteFactory:
    """
    Returns a factory function for a remote component that dynamically
    instantiates model and actuator from the registry using the config.
    """

    def factory(config: BaseRemoteComponentConfig):
        connection_factory = registry.connections[config.connection.__class__.__name__]
        connection = connection_factory(config.connection)
        return component_cls(connection=connection, action_space=config.action_space)

    return factory
