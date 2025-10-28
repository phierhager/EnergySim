# energysim/rl/wrappers/action_discretizer.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from copy import deepcopy

class ActionDiscretizerWrapper(gym.ActionWrapper):
    """
    Wraps a continuous Dict action space to discretize specific components.
    """
    def __init__(self, env: gym.Env, comp_to_num_bins: dict):
        super().__init__(env)
        self.comp_to_num_bins = comp_to_num_bins
        self.original_action_space = env.action_space
        self.action_mappings = {}

        # Build the new, discretized action space
        new_action_space = {}
        assert isinstance(self.original_action_space, spaces.Dict), "Original action space must be a Dict."
        for comp_name, space in self.original_action_space.items():
            if comp_name in self.comp_to_num_bins:
                n_bins = self.comp_to_num_bins[comp_name]
                # Create a mapping from integer action to continuous value
                if not isinstance(space, spaces.Dict):
                    raise ValueError(f"Component {comp_name} action space must be a Dict. Got {space}.")
                sub_action_mappings = {}
                sub_action_spaces = {}
                for sub_key, sub_space in space.spaces.items():
                    if not (isinstance(sub_space, spaces.Box) and sub_space.shape == (1,)):
                        raise ValueError(f"Sub-action {sub_key} of component {comp_name} must have a Box action space with shape (1,). Got {sub_space}.")
                    sub_action_mappings[sub_key] = np.linspace(sub_space.low, sub_space.high, n_bins)
                    # NOTE: We assume for simplicity that each component sub action has the same number of bins
                    sub_action_spaces[sub_key] = spaces.Discrete(n_bins)
                self.action_mappings[comp_name] = sub_action_mappings
                new_action_space[comp_name] = spaces.Dict(sub_action_spaces)
            else:
                new_action_space[comp_name] = space  # Keep original space for non-discretized components
        self.action_space = spaces.Dict(new_action_space)

    def _discrete_to_continuous(self, discrete_action: dict) -> dict:
        """Converts a discrete action dict to a continuous one."""
        continuous_action = deepcopy(discrete_action)
        for comp_name, sub_actions in discrete_action.items():
            assert isinstance(sub_actions, dict), f"Expected sub-actions for component {comp_name} to be a dict, got {type(sub_actions)}"
            if comp_name in self.action_mappings:
                for sub_key, action_val in sub_actions.items():
                    action_mapping = self.action_mappings[comp_name][sub_key]
                    continuous_action[comp_name][sub_key] = action_mapping[action_val]
            else:
                pass # Keep original action for non-discretized components
        
        return continuous_action
    
    def action(self, action):
        return self._discrete_to_continuous(action)