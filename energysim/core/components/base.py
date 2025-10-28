from abc import ABC, abstractmethod
from energysim.core.components.outputs import ComponentOutputs
from energysim.core.components.spaces import Space  
from energysim.core.components.model_base import ModelBase
from energysim.core.state import SimulationState

class ComponentBase(ABC):
    """Abstract base class for all components in the BEMS simulation framework."""
    @abstractmethod
    def __init__(self, model: ModelBase) -> None:
        pass

    @abstractmethod
    def initialize(self) -> ComponentOutputs:
        """Reset the component state for the given timestep."""
        pass

    @abstractmethod
    def advance(self, action: dict[str, float], state: SimulationState, dt_seconds: int) -> ComponentOutputs:
        """Update the component state for the given timestep."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Return the action space of the component."""
        pass