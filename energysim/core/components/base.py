from abc import ABC, abstractmethod
from typing import Optional
from energysim.core.components.outputs import ComponentOutputs
from energysim.core.shared.spaces import Space  
from energysim.core.components.model_base import ModelBase
from energysim.core.state import SimulationState

class ComponentBase(ABC):
    """Abstract base class for all components in the BEMS simulation framework."""
    @abstractmethod
    def initialize(self) -> ComponentOutputs:
        """Reset the component state for the given timestep."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Return the action space of the component."""
        pass

    @property
    @abstractmethod
    def model(self) -> ModelBase:
        """Return the underlying model of the component."""
        pass

class ActionDrivenComponent(ComponentBase):
    """Base class for stateless components that do not maintain internal state."""
    
    @abstractmethod
    def advance(self, action: dict[str, float], dt_seconds: int) -> ComponentOutputs:
        """Update the component state for the given timestep."""
        pass

class StateAwareComponent(ComponentBase):
    @abstractmethod
    def advance(self, action: dict[str, float], state: SimulationState, dt_seconds: int) -> ComponentOutputs:
        pass