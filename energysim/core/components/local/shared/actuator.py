from abc import ABC, abstractmethod
from typing import Any, Union

from energysim.core.components.shared.spaces import Space


class IActuator(ABC):
    @property
    @abstractmethod
    def action_space(self) -> dict[str, Space]:
        """Return the action space for the actuator."""
        pass

    @abstractmethod
    def interpret_action(self, **kwargs) -> Any:
        """Interpret the action and return control signals.

        The return value has to be compatible with the model's step method."""
        pass
