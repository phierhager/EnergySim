# in energysim/core/interfaces/mpc_interface.py
from abc import ABC, abstractmethod
from typing import List
import casadi as ca

class MPCBuilderBase(ABC):
    @property
    @abstractmethod
    def opti(self) -> ca.Opti: ...
    
    @property
    @abstractmethod
    def dt_seconds(self) -> float: ...
    
    @abstractmethod
    def add_constraint(self, expr) -> ca.SX: ...
    
    @abstractmethod
    def set_initial_state(self, name: str, var: ca.SX) -> None: ...


class MPCObjectiveContributor(ABC):
    """
    An interface for any object that can contribute a term
    to the MPC objective function.
    """
    @abstractmethod
    def add_mpc_objective_term(
        self,
        builder: MPCBuilderBase,
        k: int, # Timestep index within the horizon
        states: dict[str, ca.SX],
        actions: dict[str, ca.SX],
        exogenous: dict[str, ca.SX]
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
    def get_mpc_state_variables(self, builder: MPCBuilderBase) -> List[ca.SX]:
        """
        Return the symbolic state variables this model manages.
        e.g., [soc] for a battery, [T_air, T_mass] for a thermal model.
        """
        pass

    @abstractmethod
    def get_mpc_action_variables(self, builder: MPCBuilderBase) -> List[ca.SX]:
        """
        Return the symbolic action variables this model controls.
        e.g., [p_charge, p_discharge] for a battery.
        """
        pass

    @abstractmethod
    def add_mpc_dynamics_constraints(
        self,
        builder: MPCBuilderBase,
        k: int,
        states: dict[str, ca.SX],
        actions: dict[str, ca.SX],
        exogenous: dict[str, ca.SX]
    ):
        """
        Add the symbolic state transition constraints to the optimizer.
        e.g., builder.add_constraint(states['soc_next'] == states['soc'] + ...)
        """
        pass

    @abstractmethod
    def add_mpc_operational_constraints(
        self,
        builder: MPCBuilderBase,
        k: int,
        states: dict[str, ca.SX],
        actions: dict[str, ca.SX],
        exogenous: dict[str, ca.SX]
    ):
        """
        Add operational constraints for this timestep.
        e.g., builder.add_constraint(actions['p_charge'] <= self.max_power)
        """
        pass