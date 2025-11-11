# energysim/core/models/thermal_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ExogenousData, Array

# --- 1. Abstract Base Class ---
class AbstractThermalModel(eqx.Module):
    """Abstract base class for all thermal models."""
    # --- Dynamic State (must be consistent) ---
    room_temp: Array
    wall_temp: Array # Dummy for 1R1C, real for 2R2C

    # --- Static Config ---
    config: ThermalConfig = eqx.field(static=True)

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'AbstractThermalModel':
        """The 'step' function is the contract all models must fulfill."""
        raise NotImplementedError

# --- 2. 1R1C Implementation ---
class ThermalModel_1R1C(AbstractThermalModel):
    """
    1-Resistor, 1-Capacitor model.
    Models only the room air, with a single resistance to ambient.
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Dummy state, tracks room temp
            config=config
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'ThermalModel_1R1C':
        
        # 1. Get model parameters from config
        C_air = self.config.air_capacity_j_k
        R_air_amb = 1.0 / self.config.air_to_ambient_coeff_w_k
        
        T_room = self.room_temp
        T_amb = exogenous.ambient_temp

        # 2. Total thermal power (W)
        net_controllable_w = storage_discharge_w + ac_thermal_w
        net_passive_w = exogenous.internal_gains_w + exogenous.solar_gains_w
        net_thermal_input_w = net_controllable_w + net_passive_w

        # 3. Calculate temperature change
        # dT/dt = (1/C) * (P_in + (T_amb - T_room) / R)
        
        # Change from gains
        dT_gains_k = (net_thermal_input_w / C_air) * dt_seconds
        
        # Change from losses
        dT_loss_k = ((T_amb - T_room) / (R_air_amb * C_air)) * dt_seconds
        
        next_temp = T_room + dT_gains_k + dT_loss_k

        # Return new model with updated state
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp), 
            self, 
            (next_temp, next_temp) # wall_temp just follows room_temp
        )

# --- 3. 2R2C Implementation (New model) ---
class ThermalModel_2R2C(AbstractThermalModel):
    """
    2-Resistor, 2-Capacitor model.
    Models room air (C_air) and building envelope (C_wall) separately.
    Heat path: Ambient <-> Wall <-> Air <-> InternalGains
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Real state
            config=config
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'ThermalModel_2R2C':
        
        # 1. Get model parameters
        C_air = self.config.air_capacity_j_k
        C_wall = self.config.wall_capacity_j_k
        R_wall_amb = self.config.wall_to_ambient_r_k_w
        R_air_wall = self.config.air_to_wall_r_k_w
        
        T_room = self.room_temp
        T_wall = self.wall_temp
        T_amb = exogenous.ambient_temp

        # 2. Get heat inputs (W)
        net_controllable_w = storage_discharge_w + ac_thermal_w
        net_passive_w = exogenous.internal_gains_w + exogenous.solar_gains_w
        net_internal_gains_w = net_controllable_w + net_passive_w # All gains go to air

        # 3. Calculate heat flows (W)
        Q_wall_to_amb = (T_wall - T_amb) / R_wall_amb
        Q_air_to_wall = (T_room - T_wall) / R_air_wall
        
        # 4. Calculate temperature change for Wall
        # dT_wall/dt = (1/C_wall) * (Q_air_to_wall - Q_wall_to_amb)
        dT_wall_k = ((Q_air_to_wall - Q_wall_to_amb) / C_wall) * dt_seconds
        next_wall_temp = T_wall + dT_wall_k

        # 5. Calculate temperature change for Room Air
        # dT_room/dt = (1/C_air) * (Q_internal_gains - Q_air_to_wall)
        dT_room_k = ((net_internal_gains_w - Q_air_to_wall) / C_air) * dt_seconds
        next_room_temp = T_room + dT_room_k

        # 6. Return new model with updated state
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp), 
            self, 
            (next_room_temp, next_wall_temp)
        )

# --- 4. Passthrough (Dummy) Implementation ---
class PassthroughThermalModel(AbstractThermalModel):
    """
    A dummy model that ignores all physics and clamps the room
    temperature to the setpoint.
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Dummy state
            config=config
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'PassthroughThermalModel':
        
        # Room temp is always the setpoint
        next_room_temp = jnp.array(self.config.setpoint)
        # Wall temp just tracks ambient
        next_wall_temp = exogenous.ambient_temp
        
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp), 
            self, 
            (next_room_temp, next_wall_temp)
        )