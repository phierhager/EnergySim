from abc import ABC, abstractmethod
from energysim.core.components.shared.component_outputs import ComponentOutputs
from energysim.core.components.shared.spaces import Space


class ComponentBase(ABC):
    """Abstract base class for all components in the BEMS simulation framework."""

    @abstractmethod
    def initialize(self) -> ComponentOutputs:
        """Reset the component state for the given timestep."""
        pass

    @abstractmethod
    def advance(self, input: dict, dt_seconds) -> ComponentOutputs:
        """Update the component state for the given timestep."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> dict[str, Space]:
        """Return the action space of the component."""
        pass
