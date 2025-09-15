from abc import ABC, abstractmethod
from bems_simulation.core.components.shared.component_base import ComponentBase
from bems_simulation.core.components.shared.spaces import Space


class LocalComponent(ComponentBase, ABC):
    @abstractmethod
    def __init__(self, model: object, actuator: object):
        pass
