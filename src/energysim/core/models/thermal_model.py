# energysim/core/models/thermal_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ThermalInputs, Array

class AbstractThermalModel(eqx.Module):
    config: ThermalConfig
    
    def dynamics(self, t: float, T_vector: Array, inputs: ThermalInputs) -> Array:
        raise NotImplementedError

class RCNetworkModel(AbstractThermalModel):
    def __init__(self, config: ThermalConfig):
        self.config = config

    def dynamics(self, t: float, T_vector: Array, inputs: ThermalInputs) -> Array:
        """
        dT/dt = C_inv * (A @ T + Q_forcing)
        """
        # 1. Algebraic Physics (Non-linear Convection/Mixing)
        Q_mix = self._calculate_mixing_flux(T_vector)
        
        # Note: We removed Surrogate. Using standard coefficients baked into A or calculated here.
        # If convection is purely linear, it's in A_matrix. 
        # If simple non-linear (dT^1/3), we add it here.
        Q_conv = self._calculate_nonlinear_convection(T_vector)

        # 2. Total Forcing Function (Watts)
        Q_forcing = inputs.Q_solar + inputs.Q_internal + inputs.Q_hvac + Q_conv + Q_mix

        # 3. Linear Dynamics (Conduction)
        Q_conduction = self.config.A_matrix @ T_vector

        # 4. Derivative Calculation
        # dT/dt = (1/C) * (Q_cond + Q_force)
        dTdt = self.config.C_inv_vector * (Q_conduction + Q_forcing)

        # 5. Boundary Condition: Ambient Node
        # Force derivative of ambient node to 0 (it is controlled externally)
        amb_idx = self.config.ambient_air_index
        dTdt = dTdt.at[amb_idx].set(0.0)

        return dTdt

    def _calculate_mixing_flux(self, T_k: Array) -> Array:
        pairs = self.config.mixing_pairs
        if pairs.shape[0] == 0:
            return jnp.zeros_like(T_k)
            
        idx_a, idx_b = pairs[:, 0], pairs[:, 1]
        flux = self.config.mixing_conductance * (T_k[idx_b] - T_k[idx_a])
        
        Q_mix = jnp.zeros_like(T_k)
        Q_mix = Q_mix.at[idx_a].add(flux)
        Q_mix = Q_mix.at[idx_b].add(-flux)
        return Q_mix

    def _calculate_nonlinear_convection(self, T_k: Array) -> Array:
        # Fallback empirical model (Turbulent Natural Convection ~ dT^1/3)
        pairs = self.config.convection_pairs
        if pairs.shape[0] == 0:
            return jnp.zeros_like(T_k)

        idx_a, idx_b = pairs[:, 0], pairs[:, 1]
        delta_T = T_k[idx_b] - T_k[idx_a]
        
        G_dynamic = self.config.convection_coefficients * (jnp.maximum(jnp.abs(delta_T), 1e-3) ** 0.33)
        flux = G_dynamic * delta_T

        Q_conv = jnp.zeros_like(T_k)
        Q_conv = Q_conv.at[idx_a].add(flux)
        Q_conv = Q_conv.at[idx_b].add(-flux)
        return Q_conv