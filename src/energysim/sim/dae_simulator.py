import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax
from ..core.shared.data_structs import (
    SystemState, SystemActions, ExogenousData, DifferentialState,
    ThermalInputs, StorageInputs, BatteryInputs, RampingInputs, MoistureInputs
)
from ..core.models.thermal_model import RCNetworkModel
from ..core.models.heat_pump_model import RampingHeatPumpModel
from ..core.models.thermal_storage_model import StratifiedThermalStorageModel
from ..core.models.battery_model import LithiumIonSimple
from ..core.models.moisture_model import EMPDMoistureModel

class DAESimulator(eqx.Module):
    # Physics Models
    thermal: RCNetworkModel
    hp: RampingHeatPumpModel
    storage: StratifiedThermalStorageModel
    battery: LithiumIonSimple
    moisture: EMPDMoistureModel

    def get_derivatives(self, t: float, y: DifferentialState, args: tuple) -> DifferentialState:
        """The Master DAE Function."""
        actions, exo, machine_loads = args
        exo: ExogenousData
        
        # 1. Algebraic Solves (Instantaneous)
        # HP thermal output depends on current compressor state
        # Simplified T_sink (average room temp)
        T_sink = jnp.mean(y.T_vector) 
        hp_out = self.hp.calculate_output(y.hp_power_state, T_sink, exo)
        
        # 2. Differential Solves
        
        # A. Building
        th_in = ThermalInputs(
            Q_solar=jnp.zeros_like(y.T_vector), # Placeholder for complex solar
            Q_internal=machine_loads, # Injected from discrete loop
            Q_hvac=jnp.zeros_like(y.T_vector), # Simplified mapping needed here
            T_ambient=exo.ambient_temp,
            wind_speed=exo.wind_speed_m_s
        )
        # Inject HP heat into node 0 (Air) for this demo
        th_in = eqx.tree_at(lambda x: x.Q_hvac, th_in, 
                            th_in.Q_hvac.at[0].set(hp_out.thermal_power_w))
        
        dTdt_bldg = self.thermal.dynamics(t, y.T_vector, th_in)
        
        # B. Storage
        st_in = StorageInputs(
            charge_power_w=0.0, # Decoupled in this simplified flow
            discharge_power_w=actions.storage_discharge_w,
            T_inlet_c=60.0,
            T_return_c=35.0,
            T_ambient=exo.ambient_temp
        )
        dTdt_tank = self.storage.dynamics(t, y.storage_T, st_in)
        
        # C. Battery
        bat_in = BatteryInputs(power_w=actions.battery_power_w)
        dSOCdt = self.battery.dynamics(t, y.battery_soc, bat_in)
        
        # D. HVAC Ramping
        hp_in = RampingInputs(target_power_w=actions.heat_pump_power_w)
        dPdt_hp = self.hp.dynamics(t, y.hp_power_state, hp_in)
        
        # E. Moisture
        # Needs infiltration calculation coupling
        mst_in = MoistureInputs(
            T_room_c=y.T_vector[:self.moisture.config.air_volume_m3.shape[0]], # Map to rooms
            hvac_moisture_removal_kg_s=jnp.zeros_like(y.moisture_w),
            n_occupants=jnp.zeros_like(y.moisture_w),
            infiltration_flow_m3_s=jnp.full_like(y.moisture_w, 0.05), # Const approx
            exo=exo
        )
        dMdt = self.moisture.dynamics(t, y, mst_in)
        
        return DifferentialState(
            T_vector=dTdt_bldg,
            storage_T=dTdt_tank,
            battery_soc=dSOCdt,
            hp_power_state=dPdt_hp,
            ac_power_state=jnp.array(0.0),
            moisture_w=dMdt[:len(y.moisture_w)],
            moisture_buffer_u=dMdt[len(y.moisture_w):]
        )

    def step(self, state: SystemState, actions: SystemActions, exo: ExogenousData, dt: float):
        # 1. Setup Solver
        terms = diffrax.ODETerm(self.get_derivatives)
        solver = diffrax.Tsit5()
        
        # 2. Discrete Loads
        # Sum scalar base load + specific profile loads
        # Here we assume 1-node mapping for simplicity
        machine_loads = jnp.zeros_like(state.diff.T_vector)
        machine_loads = machine_loads.at[0].set(exo.base_load_w)

        # 3. Solve
        sol = diffrax.diffeqsolve(
            terms, solver, t0=0, t1=dt, dt0=dt,
            y0=state.diff,
            args=(actions, exo, machine_loads),
            max_steps=500
        )
        
        # 4. Update State
        new_diff_state = sol.ys
        
        # (Update discrete machines here if needed)
        
        return eqx.tree_at(lambda s: s.diff, state, new_diff_state)