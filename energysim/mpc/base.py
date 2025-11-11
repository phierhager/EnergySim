# energysim/core/mpc/base.py (NEW FILE)
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List
import casadi as ca

if TYPE_CHECKING:
    from .builder import MPCBuilder # Avoid circular import

class MPCObjectiveContributor(ABC):
    """
    An interface for any object that can contribute a term
    to the MPC objective function.
    """
    @abstractmethod
    def add_mpc_objective_term(
        self,
        builder: MPCBuilder,
        k: int, # Timestep index within the horizon
        states: Dict[str, ca.SX],
        actions: Dict[str, ca.SX],
        exogenous: Dict[str, ca.SX]
    ) -> ca.SX:
        """
        Return the symbolic cost term for this contributor at timestep k.
        Should return 0.0 if no cost is added for this step.
        """
        pass

class MPCFormulatable(ABC):
    """
    An interface for any simulation model (component, thermal, etc.)
    that can describe its dynamics and constraints for an MPC problem.
    """

    @abstractmethod
    def get_mpc_state_variables(self, builder: "MPCBuilder") -> List[ca.SX]:
        """
        Return the symbolic state variables this model manages.
        e.g., [soc] for a battery, [T_air, T_mass] for a thermal model.
        """
        pass

    @abstractmethod
    def get_mpc_action_variables(self, builder: "MPCBuilder") -> List[ca.SX]:
        """
        Return the symbolic action variables this model controls.
        e.g., [p_charge, p_discharge] for a battery.
        """
        pass

    @abstractmethod
    def add_mpc_dynamics_constraints(
        self,
        builder: "MPCBuilder",
        k: int,
        states: Dict[str, ca.SX],
        actions: Dict[str, ca.SX],
        exogenous: Dict[str, ca.SX]
    ):
        """
        Add the symbolic state transition constraints to the optimizer.
        e.g., builder.add_constraint(states['soc_next'] == states['soc'] + ...)
        """
        pass

    @abstractmethod
    def add_mpc_operational_constraints(
        self,
        builder: "MPCBuilder",
        k: int,
        states: Dict[str, ca.SX],
        actions: Dict[str, ca.SX],
        exogenous: Dict[str, ca.SX]
    ):
        """
        Add operational constraints for this timestep.
        e.g., builder.add_constraint(actions['p_charge'] <= self.max_power)
        """
        pass