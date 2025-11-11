from energysim.core.shared.spaces import (
    Space,
    DiscreteSpace,
    ContinuousSpace,
    DictSpace
)
from gymnasium import spaces
import numpy as np

def get_gymnasium_space(space: Space) -> spaces.Space:
    """Convert a Space into a Gymnasium space."""
    if isinstance(space, DiscreteSpace):
        return spaces.Discrete(space.n_actions)
    elif isinstance(space, ContinuousSpace):
        return spaces.Box(
            low=np.array([space.lower_bound]),
            high=np.array([space.upper_bound]),
            dtype=np.float32,
        )
    elif isinstance(space, DictSpace):
        return spaces.Dict(
            {k: get_gymnasium_space(v) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError(f"Unsupported Space type: {type(space)}")