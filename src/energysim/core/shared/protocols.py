# energysim/core/shared/protocols.py
from typing import Protocol, Any, Tuple, runtime_checkable
from .data_structs import Array

@runtime_checkable
class AlgebraicModel(Protocol):
    """
    Models that calculate instantaneous output based on current input.
    No internal state evolution (memoryless relative to the solver step).
    Example: Solar Panels, Occupancy Profiles.
    """
    def calculate(self, inputs: Any) -> Any:
        ...

@runtime_checkable
class DynamicModel(Protocol):
    """
    Models that define a system of differential equations.
    Used by the continuous DAE/ODE solver.
    Example: Building Thermal Mass, Battery SOC, Tank Temperatures.
    """
    def dynamics(self, t: float, state: Array, inputs: Any) -> Array:
        """Returns the time derivative (dState/dt)."""
        ...

@runtime_checkable
class DiscreteModel(Protocol):
    """
    Models that evolve via discrete logic steps or events.
    Used by the outer loop / discrete stepper.
    Example: Washing Machines, User Setpoint Logic.
    """
    def step(self, state: Any, signal: float, dt: float, availability: float) -> Tuple[Any, Any]:
        """Returns (NewState, Output)."""
        ...

@runtime_checkable
class HybridModel(DynamicModel, Protocol):
    """
    Models that have both state evolution AND complex algebraic outputs.
    Example: Heat Pumps (Compressor inertia is Dynamic, but COP/HeatOutput is Algebraic).
    """
    def calculate_output(self, state: Array, inputs: Any) -> Any:
        ...