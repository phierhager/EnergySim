import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import OccupantConfig

class OccupancyModel(eqx.Module):
    """
    Pure thermal gains. No electrical connection.
    """
    config: OccupantConfig
    target_node_index: int = eqx.field(static=True)

    def __init__(self, config: OccupantConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index

    @eqx.filter_jit
    def step(self, count_signal: float):
        """
        Args:
            count_signal: Number of people (or fraction of max capacity)
        Returns:
            heat_w: The metabolic heat gain
        """
        # No electrical power return needed!
        heat_w = self.config.nominal_heat_w * count_signal
        return self, heat_w