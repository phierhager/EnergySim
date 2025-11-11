from dataclasses import dataclass
from energysim.core.components.outputs import ComponentOutputs
from energysim.core.state import SimulationState
from energysim.core.thermal.base import ThermalState

@dataclass
class RewardContext:
    """
    Unified container for system state for reward calculation.
    """

    system_balance: ComponentOutputs
    thermal_state: ThermalState
    simulation_state: SimulationState
