# energysim/core/mpc/builder.py (NEW FILE)
import casadi as ca
from typing import List, Dict

from energysim.core.components.base import ComponentBase
from energysim.core.thermal.base import ThermalModel
from energysim.mpc.base import MPCFormulatable
from energysim.reward.manager import RewardManager

class MPCBuilder:
    def __init__(
        self,
        components: Dict[str, ComponentBase],
        thermal_model: ThermalModel,
        reward_manager: RewardManager,
        horizon_steps: int,
        dt_seconds: int
    ):
        self.all_components = components # Store for reference if needed
        self.reward_manager = reward_manager
        self.thermal_model = thermal_model
        self.N = horizon_steps
        self.dt_seconds = dt_seconds

        self.formulatable_models: Dict[str, MPCFormulatable] = {}
        for name, comp in components.items():
            if isinstance(comp, MPCFormulatable):
                self.formulatable_models[name] = comp.model
            else:
                raise ValueError(f"Component {name} does not implement MPCFormulatable interface.")
        if isinstance(thermal_model, MPCFormulatable):
            self.formulatable_models["thermal"] = thermal_model
        else:
            raise ValueError("Thermal model does not implement MPCFormulatable interface.")

        self.opti = ca.Opti()
        self.states: Dict[str, ca.SX] = {}
        self.actions: Dict[str, ca.SX] = {}
        self.exogenous: Dict[str, ca.SX] = {}
        
        self.initial_states: Dict[str, ca.SX] = {}
        
        self._build_problem()

    def _build_problem(self):
        for comp_name, comp_model in self.formulatable_models.items():
            assert isinstance(comp_model, MPCFormulatable), "Model must implement MPCFormulatable"
            # Register state variables
            for state_var in comp_model.get_mpc_state_variables(self, comp_name):
                assert isinstance(state_var, ca.SX), "State variable must be casadi SX"
                self.states[state_var.name()] = self.opti.variable(self.N + 1, 1)
            
            # Register action variables
            for action_var in comp_model.get_mpc_action_variables(self, comp_name):
                assert isinstance(action_var, ca.SX), "Action variable must be casadi SX"
                self.actions[action_var.name()] = self.opti.variable(self.N, 1)
        
        # 2. Define parameters for exogenous data (forecasts)
        # (e.g., 'load', 'pv', 'price', 'ambient_temperature')
        # This is simplified; you'd get these names from your dataset config
        for name in self.feature_columns:
             self.exogenous[name] = self.opti.parameter(self.N, 1)

        # 3. Define Objective Function
        total_objective = 0
        for k in range(self.N):
            for layer in self.reward_layers:
                total_objective += layer.add_mpc_objective_term(self, k, self.states, self.actions, self.exogenous)

        self.opti.minimize(total_objective)

        # 4. Add Dynamics and Constraints
        for k in range(self.N):
            # Pass dictionaries sliced at timestep k
            k_states = {name: var[k] for name, var in self.states.items()}
            k_next_states = {name: var[k+1] for name, var in self.states.items()}
            k_actions = {name: var[k] for name, var in self.actions.items()}
            k_exogenous = {name: var[k] for name, var in self.exogenous.items()}

            # Add component dynamics & constraints
            for comp in self.components + [self.thermal_model]:
                comp.add_mpc_dynamics_constraints(self, k, k_states, k_next_states, k_actions, k_exogenous)
                comp.add_mpc_operational_constraints(self, k, k_states, k_actions, k_exogenous)
        
        # 5. Add Initial State Constraints
        for name, param in self.initial_states.items():
            self.add_constraint(self.states[name][0] == param)
            
        # 6. Setup Solver
        self.opti.solver('ipopt') # IPOPT is a good, free non-linear solver

    def add_constraint(self, constr):
        self.opti.subject_to(constr)
        
    def set_initial_state(self, name: str, var: ca.SX):
        # Called by get_mpc_state_variables
        param = self.opti.parameter(1)
        self.initial_states[name] = param
        var.set_name(name) # Ensure var has a name

    def solve(self, current_state_dict: Dict, forecast_dict: Dict) -> Dict:
        # 1. Set parameter values
        for name, value in current_state_dict.items():
            self.opti.set_value(self.initial_states[name], value)
            
        for name, value_array in forecast_dict.items():
            self.opti.set_value(self.exogenous[name], value_array)

        # 2. Solve the problem
        sol = self.opti.solve()

        # 3. Extract and return the *first* action
        first_action = {}
        for name, var in self.actions.items():
            first_action[name] = sol.value(var)[0]
            
        return first_action