# energysim/core/state.py (NEW FILE)

from dataclasses import dataclass, field
from typing import Dict

from .thermal.state import ThermalState
from .components.outputs import ComponentOutputs
from .timestep_data import TimestepData

@dataclass(frozen=True, slots=True)
class SimulationState:
    """
    The single source of truth for the simulation's state at the beginning of a timestep.
    
    This object is constructed by the simulation orchestrator (e.g., BuildingEnvironment)
    and passed to all components and models during the 'advance' call.
    """
    # Dynamic, time-varying data from external sources
    timestep_data: TimestepData

    # Internal state of the simulation from the *previous* step
    thermal_state: ThermalState
    component_outputs: Dict[str, ComponentOutputs] = field(default_factory=dict)