import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TransformAction
from gymnasium.core import ActType, ObsType, WrapperActType
import numpy as np

class ActionFlattenerWrapper(
    TransformAction[ObsType, WrapperActType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Flattens the environment's action space and the actions passed to `step`.

    This wrapper is a more robust version of action flattening. It handles
    hybrid action spaces (containing both `Box` and `Discrete` spaces)
    by using an `argmax` mechanism for the discrete components. This makes it
    compatible with standard RL agents that output continuous values and allows
    for direct use of `env.action_space.sample()` for testing.

    Example:
        >>> import gymnasium as gym
        >>> from energysim.rl.wrappers.action_flattener import ActionFlattenerWrapper
        >>> space = gym.spaces.Dict({
        ...     "discrete": gym.spaces.Discrete(2),
        ...     "continuous": gym.spaces.Box(low=-1, high=1, shape=(3,))
        ... })
        >>> env = gym.make("SomeEnv-v0")
        >>> env.action_space = space
        >>> env = ActionFlattenerWrapper(env)
        >>> env.action_space.shape
        (5,)
        >>> # The sampled action is now valid for env.step()
        >>> random_action = env.action_space.sample()
        >>> env.step(random_action)
    """

    def __init__(self, env: gym.Env):
        """Initializes the wrapper, calculates the flat action space, and sets up the robust unflattening function."""
        gym.utils.RecordConstructorArgs.__init__(self)

        # The flattened space is a simple Box representing all combined action dimensions.
        flat_action_space = spaces.utils.flatten_space(env.action_space)

        # The function maps a flat vector back to the original structured action.
        # We pass a reference to our custom, robust unflattening method.
        TransformAction.__init__(
            self,
            env=env,
            func=lambda flat_action: self._robust_unflatten(
                env.action_space, flat_action
            ),
            action_space=flat_action_space,
        )

    def _robust_unflatten(
        self, space: spaces.Space, flat_action: np.ndarray
    ) -> ActType:
        """Recursively unflattens a flat action vector into a structured action.

        This method is the core of the wrapper's robustness. It correctly
        handles nested `Dict` and `Tuple` spaces while using a special `argmax`
        method for `Discrete` spaces. This allows a continuous vector of values
        (from a sampler or RL agent) to be correctly interpreted as discrete actions.

        Args:
            space: The (potentially nested) action space to unflatten into.
            flat_action: The flat numpy array representing the action.

        Returns:
            A structured action that conforms to the original action space.
        """
        if isinstance(space, spaces.Discrete):
            # THE KEY CHANGE: Instead of expecting a one-hot vector, we take the
            # index of the highest value in the corresponding action slice. This
            # robustly converts a continuous output into a discrete choice.
            return int(np.argmax(flat_action))

        if isinstance(space, spaces.Box):
            # Standard unflattening for Box spaces.
            return flat_action.reshape(space.shape)

        if isinstance(space, spaces.Tuple):
            # Recursively unflatten for each subspace in the Tuple.
            unflattened_actions = []
            start_idx = 0
            for subspace in space.spaces:
                end_idx = start_idx + spaces.utils.flatdim(subspace)
                unflattened_actions.append(
                    self._robust_unflatten(subspace, flat_action[start_idx:end_idx])
                )
                start_idx = end_idx
            return tuple(unflattened_actions)

        if isinstance(space, spaces.Dict):
            # Recursively unflatten for each subspace in the Dict.
            unflattened_actions = {}
            start_idx = 0
            for key, subspace in space.spaces.items():
                end_idx = start_idx + spaces.utils.flatdim(subspace)
                unflattened_actions[key] = self._robust_unflatten(
                    subspace, flat_action[start_idx:end_idx]
                )
                start_idx = end_idx
            return unflattened_actions

        if isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete)):
            # Handle these common spaces similarly to Box.
            return flat_action.reshape(space.shape).astype(space.dtype)

        raise TypeError(f"Unsupported action space type to unflatten: {type(space)}")
