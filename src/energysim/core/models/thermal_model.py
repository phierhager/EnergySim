# energysim/core/models/thermal_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ExogenousData, Array

class ThermalModel(eqx.Module):
    room_temp: Array
    config: ThermalConfig = eqx.field(static=True)
    
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        self.config = config
        self.room_temp = jnp.array(initial_temp)

    @eqx.filter_jit
    def step(self, 
             storage_discharge_w: Array, 
             ac_thermal_w: Array, 
             exogenous: ExogenousData, 
             dt_seconds: float
    ) -> 'ThermalModel':
        """
        JIT-compiled step.
        Heated/cooled by storage/AC, and now also by passive gains.
        """
        C_th = self.config.building_volume * self.config.air_density * self.config.specific_heat_air
        R_th = 1.0 / self.config.heat_transfer_coeff
        
        T_room = self.room_temp
        T_amb = exogenous.ambient_temp
        
        # --- LOGIC UPDATED ---
        
        # 1. Controllable thermal power (W)
        net_controllable_w = storage_discharge_w + ac_thermal_w
        
        # 2. Uncontrollable (passive) thermal power (W)
        #    This is where your "external components" are added.
        net_passive_w = exogenous.internal_gains_w + exogenous.solar_gains_w
        
        # 3. Total thermal gains (W)
        net_thermal_input_w = net_controllable_w + net_passive_w
        
        # 4. Calculate energy change from gains (J)
        dT_gains_j = (net_thermal_input_w / C_th) * dt_seconds
        
        # 5. Calculate energy change from ambient losses (J)
        dT_loss_j = ((T_amb - T_room) / (R_th * C_th)) * dt_seconds
        
        # 6. Apply temperature change
        next_temp = T_room + dT_gains_j + dT_loss_j
        # --- END OF UPDATE ---
        
        return eqx.tree_at(lambda model: model.room_temp, self, next_temp)