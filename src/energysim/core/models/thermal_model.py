# src/energysim/core/models/thermal_model.py

import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ExogenousData

Array = jnp.ndarray 

class AbstractThermalModel(eqx.Module):
    T_vector: Array
    config: ThermalConfig

    @eqx.filter_jit
    def step(self, 
             U_vector: Array,     
             waste_heat_w: float, 
             exogenous: ExogenousData, 
             Q_flux_injection: Array,     #
             dt_seconds: float
            ) -> 'AbstractThermalModel':
        raise NotImplementedError


class RCNetworkModel(AbstractThermalModel):
    """
    Solves the core thermal dynamics: dT/dt = C_inv * (A*T + U_vector + Q_inf + Q_waste + Q_flux_injection)
    The model assumes U_vector and Q_flux_injection are pre-calculated externally.
    """
    def __init__(self, config: ThermalConfig, initial_T_vector: Array):
        super().__init__(
            T_vector=initial_T_vector,
            config=config
        )

    @eqx.filter_jit
    def _calculate_dynamic_infiltration(self, T_k: Array, T_amb: float, wind_speed: float) -> Array:
        """Calculates heat flow due to air infiltration/ventilation (Q_inf). (Unchanged)"""
        if not self.config.use_dynamic_infiltration:
            return jnp.zeros_like(T_k)

        room_temps = T_k[jnp.array(self.config.room_air_indices)]
        avg_room_temp = jnp.mean(room_temps)
        delta_T = jnp.abs(T_amb - avg_room_temp)
        ach = self.config.inf_k1 + (self.config.inf_k2 * delta_T) + (self.config.inf_k3 * wind_speed)
        conductance = (ach * self.config.room_vol_m3 * 1200.0) / 3600.0
        n_rooms = len(self.config.room_air_indices)
        g_per_room = conductance / n_rooms
        Q_inf_vector = jnp.zeros_like(T_k)
        q_rooms = g_per_room * (T_amb - room_temps)
        Q_inf_vector = Q_inf_vector.at[jnp.array(self.config.room_air_indices)].set(q_rooms)
        
        return Q_inf_vector

    @eqx.filter_jit
    def step(self, 
             U_vector: Array, 
             waste_heat_w: float, 
             exogenous: ExogenousData, 
             Q_flux_injection: Array, 
             dt_seconds: float
            ) -> 'RCNetworkModel':
        
        T_k = self.T_vector
        
        # 1. Set Ambient Node
        T_k = T_k.at[self.config.ambient_air_index].set(exogenous.ambient_temp)

        # 2. Linear Dynamics: A*T + B*U
        A_T = self.config.A_matrix @ T_k
        
        # 3. Dynamic Infiltration
        Q_inf = self._calculate_dynamic_infiltration(T_k, exogenous.ambient_temp, exogenous.wind_speed_m_s)

        # 4. Waste Heat Coupling (Q_waste)
        Q_waste = jnp.zeros_like(T_k)
        valid_node = self.config.waste_heat_node_index >= 0
        waste_node_idx = jnp.where(valid_node, self.config.waste_heat_node_index, 0)
        added_heat = jnp.where(valid_node, waste_heat_w, 0.0)
        Q_waste = Q_waste.at[waste_node_idx].add(added_heat)

        # 5. Integration: Summation of all heat flows (W)
        total_heat_flow = A_T + U_vector + Q_inf + Q_waste + Q_flux_injection
        
        # dT/dt = C_inv * Q_total
        dT_dt_vector = self.config.C_inv_vector * total_heat_flow

        # Zero out derivative for ambient node (infinite capacity)
        dT_dt_vector = dT_dt_vector.at[self.config.ambient_air_index].set(0.0)

        T_k_plus_1 = T_k + dT_dt_vector * dt_seconds

        return eqx.tree_at(lambda m: m.T_vector, self, T_k_plus_1)