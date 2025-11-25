# energysim/core/models/occupancy.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import OccupantConfig

class OccupancyOutput(eqx.Module):
    heat_gain_w: float

class OccupancyModel(eqx.Module):
    config: OccupantConfig
    target_node_index: int = eqx.field(static=True)
    
    def __init__(self, config: OccupantConfig, target_node_index: int):
        self.config = config
        self.target_node_index = target_node_index

    def calculate(self, count_signal: float) -> OccupancyOutput:
        """Pure algebraic mapping from profile to heat."""
        heat_w = self.config.nominal_heat_w * count_signal
        return OccupancyOutput(heat_w)