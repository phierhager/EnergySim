from typing import Iterable
from energysim.core.components.registry import registry
from energysim.core.components.config_base import BaseComponentConfig
from energysim.core.components.base import ComponentBase


def build_component(config: BaseComponentConfig) -> ComponentBase:
    """
    Builds a component based on its configuration.
    
    This factory is now simplified and no longer distinguishes between 'local' and 'remote'.
    It constructs a component and its internal model based on the registered types.
    """
    if config.__class__.__name__ not in registry.components:
        raise ValueError(f"Component config '{config.__class__.__name__}' not found in registry.")
    if config.model.__class__.__name__ not in registry.models:
        raise ValueError(f"Model config '{config.model.__class__.__name__}' not found in registry.")

    component_cls = registry.components[config.__class__.__name__]
    model_factory = registry.models[config.model.__class__.__name__]
    
    model = model_factory(config.model)
    return component_cls(model=model)


def build_components(configs: Iterable[BaseComponentConfig]) -> Iterable[ComponentBase]:
    """Builds a sequence of components from their configurations."""
    return (build_component(config) for config in configs)
