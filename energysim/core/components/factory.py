from typing import Sequence, Iterable
from energysim.core.components.registry import registry
from energysim.core.components.local.factory import local_component_factory
from energysim.core.components.local.shared.config import BaseLocalComponentConfig
from energysim.core.components.remote.factory import remote_component_factory
from energysim.core.components.remote.shared.config import (
    BaseRemoteComponentConfig,
)
from energysim.core.components.shared.component_config import BaseComponentConfig
from energysim.core.components.shared.component_base import ComponentBase


def build_component(config: BaseComponentConfig) -> ComponentBase:
    print(registry.components)
    component_cls, is_local = registry.components[config.__class__.__name__]
    if is_local:
        component_factory = local_component_factory(component_cls)
        assert isinstance(config, BaseLocalComponentConfig)
        component = component_factory(config)
    else:
        component_factory = remote_component_factory(component_cls)
        assert isinstance(config, BaseRemoteComponentConfig)
        component = component_factory(config)
    assert isinstance(component, component_cls)
    return component


def build_components(configs: Iterable[BaseComponentConfig]) -> Iterable[ComponentBase]:
    return (build_component(config) for config in configs)
