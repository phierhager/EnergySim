import jax
import jax.numpy as jnp
import equinox as eqx
import os

class ConvectionSurrogate(eqx.Module):
    """
    Predicts h_convection [W/m2K].
    Input:  [T_surf, T_air, V_air, Char_Len, Tilt_Cos]
    Output: [h_c]
    """
    mlp: eqx.nn.MLP
    
    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=5, out_size=1, width_size=64, depth=3,
            activation=jax.nn.gelu, key=key
        )

    def __call__(self, T_surf, T_air, v_air_mag, length_scale, tilt_cosine):
        # Standardize inputs (assuming typical building ranges)
        # T ~ 20C, V ~ 0-5 m/s, L ~ 2.5m
        x = jnp.stack([
            (T_surf - 20.0) / 20.0,
            (T_air - 20.0) / 20.0,
            v_air_mag / 5.0,
            length_scale / 3.0,
            tilt_cosine
        ], axis=-1)
        
        raw_out = self.mlp(x)[0]
        
        # Ensure physical positivity (h_c > 0.1)
        return jax.nn.softplus(raw_out) + 0.1

    @staticmethod
    def load(path: str, key: jax.Array) -> 'ConvectionSurrogate':
        """Loads pre-trained weights from a file."""
        # 1. Initialize random skeleton
        skeleton = ConvectionSurrogate(key)
        
        if not os.path.exists(path):
            print(f"[Warning] Surrogate weights not found at {path}. Using Random Init.")
            return skeleton
            
        # 2. Load leaves
        return eqx.tree_deserialise_leaves(path, skeleton)