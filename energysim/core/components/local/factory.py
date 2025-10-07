from typing import Callable
from energysim.core.components.registry import registry
from energysim.core.components.local.shared.config import BaseLocalComponentConfig
from energysim.core.components.local.shared.component import LocalComponent

LocalFactory = Callable[[BaseLocalComponentConfig], LocalComponent]


def local_component_factory(
    component_cls: type[LocalComponent],
) -> LocalFactory:
    """
    Returns a factory function for a local component that dynamically
    instantiates model from the registry using the config.
    """

    def factory(config: BaseLocalComponentConfig):
        model_factory = registry.models[config.model.__class__.__name__]
        model = model_factory(config.model)
        return component_cls(model=model)

    return factory
