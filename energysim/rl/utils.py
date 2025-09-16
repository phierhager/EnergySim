from energysim.core.components.shared.spaces import (
    Space,
    DiscreteSpace,
    ContinuousSpace,
)
from gymnasium import spaces
import numpy as np


def get_gymnasium_space(space: Space | dict[str, Space]) -> spaces.Space:
    """Convert a Space or a dictionary of Spaces into a Gymnasium space."""
    if isinstance(space, dict):
        return spaces.Dict(
            {k: get_gymnasium_space(v) for k, v in space.items() if k != "type"}
        )
    elif isinstance(space, DiscreteSpace):
        return spaces.Discrete(space.n_actions)
    elif isinstance(space, ContinuousSpace):
        return spaces.Box(
            low=np.array([space.lower_bound]),
            high=np.array([space.upper_bound]),
            dtype=np.float32,
        )
    else:
        raise TypeError(f"Unsupported Space type: {type(space)}")
