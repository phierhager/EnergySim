# src/energysim/core/models/thermal_model.py

from typing import Optional 
import jax
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

    # @eqx.filter_jit
    # def _calculate_dynamic_infiltration(self, T_k: Array, T_amb: float, wind_speed: float) -> Array:
    #     """
    #     Calculates infiltration using the Orifice Equation (Sherman-Grimsrud).
    #     Q = A_leak * sqrt( C_stack * |dT| + C_wind * v^2 )
    #     """
    #     if not self.config.use_dynamic_infiltration:
    #         return jnp.zeros_like(T_k)

    #     room_temps = T_k[jnp.array(self.config.room_air_indices)]
        
    #     # Physics Delta
    #     delta_T = jnp.abs(T_amb - room_temps)
    #     v_wind_sq = wind_speed ** 2
        
    #     # Pressure term approximation (Pascal-like units)
    #     # We use a soft-sqrt (pseudo-huber) to maintain differentiability at 0 flow
    #     pressure_term = (self.config.stack_coeff * delta_T) + (self.config.wind_coeff * v_wind_sq)
    #     flow_factor = jnp.sqrt(pressure_term + 1e-6) # Epsilon for gradient safety
        
    #     # Flow Rate (m3/s)
    #     vol_flow_m3_s = self.config.leakage_area_m2 * flow_factor
        
    #     # Heat Flow = m_dot * Cp * dT
    #     # rho ~ 1.2, Cp ~ 1005 -> Volumetric Heat Capacity ~ 1206
    #     conductance = vol_flow_m3_s * 1206.0
        
    #     # Distribute to room nodes
    #     Q_inf_vector = jnp.zeros_like(T_k)
    #     q_rooms = conductance * (T_amb - room_temps)
        
    #     Q_inf_vector = Q_inf_vector.at[jnp.array(self.config.room_air_indices)].set(q_rooms)
        
    #     return Q_inf_vector
    
    @eqx.filter_jit
    def _calculate_mixing_flux(self, T_k: Array) -> Array:
        """
        Calculates heat transfer due to inter-zone air mixing.
        Q_a = G_mix * (T_b - T_a)
        Q_b = G_mix * (T_a - T_b)
        """
        pairs = self.config.mixing_pairs
        conductance = self.config.mixing_conductance
        
        if pairs.shape[0] == 0:
            return jnp.zeros_like(T_k)

        # 1. Extract Temperatures
        idx_a = pairs[:, 0]
        idx_b = pairs[:, 1]
        
        T_a = T_k[idx_a]
        T_b = T_k[idx_b]
        
        # 2. Calculate Flux (Watts)
        # Positive Q means heat flows INTO the node
        # Q_a_gain = G * (T_b - T_a)
        flux = conductance * (T_b - T_a)
        
        # 3. Accumulate into a total flux vector
        Q_mix = jnp.zeros_like(T_k)
        
        # Add flux to A, subtract from B (conservation)
        Q_mix = Q_mix.at[idx_a].add(flux)
        Q_mix = Q_mix.at[idx_b].add(-flux)
        
        return Q_mix
    
    @eqx.filter_jit
    def _calculate_nonlinear_convection(
        self, 
        T_k: Array, 
        exo: Optional[ExogenousData] = None
    ) -> Array:
        pairs = self.config.convection_pairs
        
        if pairs.shape[0] == 0:
            return jnp.zeros_like(T_k)

        idx_a = pairs[:, 0]
        idx_b = pairs[:, 1]
        T_a = T_k[idx_a]
        T_b = T_k[idx_b]
        delta_T = T_b - T_a

        if self.surrogate is not None and exo is not None:
            # 1. Construct Velocity Vector
            # Use the pre-compiled type map (0=Internal, 1=External)
            types = self.config.convection_types # Shape (N_pairs,)
            
            # External edges get Wind Speed
            v_wind = exo.wind_speed_m_s
            
            # Internal edges get HVAC proxy (e.g. 0.15 m/s still air + disturbance)
            # For high-fidelity, pass hvac_flow in step() and use here.
            # For now, assume constant internal mixing.
            v_int = 0.15 
            
            # Select based on type
            v_air_vector = jnp.where(types == 1, v_wind, v_int)
            
            # 2. Metadata Unpacking (Coefficients store geometric data)
            # Assuming we packed [Length_Scale] into coeffs during factory creation
            length_scale = self.config.convection_coefficients 
            
            # 3. Vectorized Surrogate Call
            # We use a fixed tilt of 1.0 (Vertical) for now, or pack tilt into coeffs too
            h_c = jax.vmap(self.surrogate)(
                T_surf=T_a, 
                T_air=T_b, 
                v_air_mag=v_air_vector, # <--- The Fix: Correct Velocity Used
                length_scale=length_scale, 
                tilt_cosine=jnp.ones_like(T_a)
            )
            
            # Flux = h_c * Area * dT
            # IMPORTANT: RCNetwork builder usually stores 1/R_base in coeffs.
            # If using surrogate, coeffs should represent AREA.
            # We assume coeffs = Area here for the surrogate path.
            flux = h_c * self.config.convection_coefficients * delta_T
            
        else:
            # Fallback Physics
            abs_delta_T = jnp.abs(delta_T)
            G_dynamic = self.config.convection_coefficients * (jnp.maximum(abs_delta_T, 1e-3) ** 0.33)
            flux = G_dynamic * delta_T

        Q_conv = jnp.zeros_like(T_k)
        Q_conv = Q_conv.at[idx_a].add(flux)
        Q_conv = Q_conv.at[idx_b].add(-flux)
        return Q_conv

    @eqx.filter_jit
    def step(
        self,
        U_vector: Array,          # Controlled Inputs (Heating/Cooling devices)
        Q_advection_w: Array,     # <--- NEW: Pre-calculated airflow heat gains/losses
        Q_flux_injection: Array,  # Solar + Internal Gains
        waste_heat_w: float,
        dt_seconds: float
    ) -> 'RCNetworkModel':

        T_k = self.T_vector
        
        # 1. Linear Dynamics (Conduction)
        A_T = self.config.A_matrix @ T_k
        
        # 2. Non-Linear Convection (Surface <-> Air)
        # This stays internal because it depends strictly on local T_k
        Q_conv_internal = self._calculate_nonlinear_convection(T_k)

        # 4. Inter-Zone Mixing
        Q_mixing = self._calculate_mixing_flux(T_k)

        # 5. Waste Heat
        Q_waste = jnp.zeros_like(T_k)
        valid_node = self.config.waste_heat_node_index >= 0
        waste_node_idx = jnp.where(valid_node, self.config.waste_heat_node_index, 0)
        added_heat = jnp.where(valid_node, waste_heat_w, 0.0)
        Q_waste = Q_waste.at[waste_node_idx].add(added_heat)

        # 6. Integration
        # Sum all heat flows
        # Q_advection_w is now just a number passed in.
        # The ThermalModel doesn't care if it came from a CFD solver or a constant.
        total_heat_flow = A_T + U_vector + Q_advection_w + Q_mixing + Q_waste + Q_flux_injection + Q_conv_internal

        # dT/dt = C_inv * Q_total
        dT_dt_vector = self.config.C_inv_vector * total_heat_flow

        # Zero out derivative for ambient node
        dT_dt_vector = dT_dt_vector.at[self.config.ambient_air_index].set(0.0)

        T_k_plus_1 = T_k + dT_dt_vector * dt_seconds

        return eqx.tree_at(lambda m: m.T_vector, self, T_k_plus_1)