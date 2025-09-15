from typing import Callable
from bems_simulation.core.components.registry import registry
from bems_simulation.core.components.local.shared.config import BaseLocalComponentConfig
from bems_simulation.core.components.local.shared.component import LocalComponent

LocalFactory = Callable[[BaseLocalComponentConfig], LocalComponent]


def local_component_factory(
    component_cls: type[LocalComponent],
) -> LocalFactory:
    """
    Returns a factory function for a local component that dynamically
    instantiates model and actuator from the registry using the config.
    """

    def factory(config: BaseLocalComponentConfig):
        model_factory = registry.models[config.model.__class__.__name__]
        actuator_factory = registry.actuators[config.actuator.__class__.__name__]
        model = model_factory(config.model)
        actuator = actuator_factory(config.actuator)
        return component_cls(model=model, actuator=actuator)

    return factory
