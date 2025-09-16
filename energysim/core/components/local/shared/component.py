from abc import ABC, abstractmethod
from energysim.core.components.shared.component_base import ComponentBase
from energysim.core.components.shared.spaces import Space


class LocalComponent(ComponentBase, ABC):
    @abstractmethod
    def __init__(self, model: object, actuator: object):
        pass
