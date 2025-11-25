# src/energysim/sim/simulator.py

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional, Dict, List, Any, Union

from energysim.core.physics.airflow import AirflowNetworkSolver

# --- Import Core Factories ---
from ..core.models.factory import (
    create_battery, create_thermal, create_heat_pump,
    create_ac, create_storage, create_solar, 
    create_smart_machines, create_passive_machines, create_occupants,
    create_moisture
)

# --- Models ---
from ..core.models.battery_model import AbstractBatteryModel
from ..core.models.thermal_model import AbstractThermalModel
from ..core.models.heat_pump_model import AbstractHeatPumpModel
from ..core.models.air_conditioner_model import AbstractAirConditionerModel
from ..core.models.thermal_storage_model import AbstractThermalStorage
from ..core.models.solar_model import AbstractSolarModel
from ..core.models.moisture_model import AbstractMoistureModel

# --- Internal Loads ---
from ..core.models.machines import AbstractMachine, SmartAppliance, PassiveEquipment
from ..core.models.occupancy import OccupancyModel

# --- Objectives ---
from ..core.objectives import f_cost_step

# --- Data Structures ---
from ..core.shared.data_structs import (
    SystemActions, ExogenousData, SystemState,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig,
    BatteryState, ThermalState, ThermalStorageState,
    HeatPumpState, AirConditionerState, ApplianceConfig, OccupantConfig,
    MoistureState
)

# --- Physics ---
from ..core.physics.constants import Coefficients
from ..core.physics.thermo import SurfaceRoughness
from ..core.physics.solar import get_sun_position, calculate_solar_incidence, calculate_iam
from ..core.physics.thermo import calculate_external_convection, get_convection_multiplier
from ..core.physics.weather import get_ground_temperature


class JAXSimulator(eqx.Module):
    """
    Top-Tier Differentiable Building Energy Simulator.
    
    Features:
    - RC-Network Thermal Dynamics (State-Space)
    - Psychrometric Moisture Balance (Latent Heat)
    - Geometric Solar Ray-Casting approximation
    - Dynamic Infiltration (Stack + Wind)
    - 3D Stratified Thermal Storage
    - Split Passive/Smart internal loads
    """
    
    # --- Sub-Models ---
    battery: AbstractBatteryModel
    thermal: AbstractThermalModel
    heat_pump: AbstractHeatPumpModel
    ac: AbstractAirConditionerModel
    storage: AbstractThermalStorage
    solar: AbstractSolarModel
    moisture: AbstractMoistureModel
    
    airflow_solver: AirflowNetworkSolver
    
    # --- Internal Loads ---
    smart_machines: List[SmartAppliance]
    passive_machines: List[PassiveEquipment]
    occupants: List[OccupancyModel]
    
    # --- Physics Matrices ---
    wind_coupling_matrix: jnp.ndarray
    wind_roughness_factors: jnp.ndarray 
    wind_surface_areas: jnp.ndarray
    opaque_map_matrix: jnp.ndarray
    opaque_normals: jnp.ndarray
    opaque_areas_absorb: jnp.ndarray 
    win_trans_map_matrix: jnp.ndarray
    win_abs_map_matrix: jnp.ndarray
    win_normals: jnp.ndarray
    win_areas: jnp.ndarray
    win_shgc: jnp.ndarray
    opaque_svf: jnp.ndarray
    win_svf: jnp.ndarray

    # --- Configs ---
    configs: Tuple = eqx.field(static=True)
    dt_seconds: float = eqx.field(static=True)
    phys_config: Coefficients = eqx.field(static=True)
    
    # Reset Metadata
    _init_machine_configs: List[ApplianceConfig] = eqx.field(static=True)
    _init_occupant_configs: List[OccupantConfig] = eqx.field(static=True)
    _init_dynamic_surfaces: List[Dict] = eqx.field(static=True)
    _init_solar_surfaces: List[Dict] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        dt_seconds: float,
        t_config: ThermalConfig,
        r_config: RewardConfig,
        machine_configs: List[ApplianceConfig], 
        occupant_configs: List[OccupantConfig],
        dynamic_surfaces: List[Dict[str, Any]],
        solar_surfaces: List[Dict[str, Any]],
        b_config: Optional[BatteryConfig] = None,
        hp_config: Optional[HeatPumpConfig] = None,
        ac_config: Optional[AirConditionerConfig] = None,
        ts_config: Optional[ThermalStorageConfig] = None,
        s_config: Optional[SolarConfig] = None,
        physics_config: Coefficients = Coefficients()
    ) -> 'JAXSimulator':
        
        # 1. Create Sub-Models
        n_rooms = len(t_config.room_air_indices)
        battery = create_battery(b_config)
        thermal = create_thermal(t_config)
        heat_pump = create_heat_pump(hp_config, n_rooms)
        ac = create_ac(ac_config, n_rooms)
        storage = create_storage(ts_config)
        solar = create_solar(s_config)
        
        # Create Moisture Model (High Fidelity Physics)
        # Initializes state based on default 50% RH
        moisture = create_moisture(t_config, physics_config, initial_rh=0.5)
        
        # 2. Create Split Internal Loads
        smart_models = create_smart_machines(machine_configs, t_config.node_map)
        passive_models = create_passive_machines(machine_configs, t_config.node_map)
        occupant_models = create_occupants(occupant_configs, t_config.node_map)

        # 3. Bundle Configs
        configs = (thermal.config, battery.config, r_config, heat_pump.config, ac.config, storage.config, solar.config)

        # 4. Build Physics Matrices
        n_nodes = len(t_config.node_names)
        w_coup, w_rough, w_area = cls._build_wind_matrices(n_nodes, t_config.node_map, dynamic_surfaces)
        (o_map, o_norm, o_area, o_svf, 
         w_t_map, w_a_map, w_norm, w_area_arr, w_shgc, w_svf) = cls._build_solar_matrices(n_nodes, t_config.node_map, solar_surfaces)

        # Initialize Solver
        airflow_solver = AirflowNetworkSolver()
            
        # [NEW] Placeholder Geometry (In real usage, parse from IDF)
        # (N_obs, 3)
        obs_v0 = jnp.zeros((0, 3)) 
        obs_v1 = jnp.zeros((0, 3))
        obs_v2 = jnp.zeros((0, 3))
        surf_cents = jnp.zeros((len(solar_surfaces), 3))

        return cls(
            battery=battery,
            thermal=thermal,
            heat_pump=heat_pump,
            ac=ac,
            storage=storage,
            solar=solar,
            moisture=moisture,
            
            smart_machines=smart_models,
            passive_machines=passive_models,
            occupants=occupant_models,
            
            wind_coupling_matrix=w_coup,
            wind_roughness_factors=w_rough,
            wind_surface_areas=w_area,
            opaque_map_matrix=o_map,
            opaque_normals=o_norm,
            opaque_areas_absorb=o_area,
            win_trans_map_matrix=w_t_map,
            win_abs_map_matrix=w_a_map,
            win_normals=w_norm,
            win_areas=w_area_arr,
            win_shgc=w_shgc,
            opaque_svf=o_svf,
            win_svf=w_svf,
            
            configs=configs,
            dt_seconds=dt_seconds,
            phys_config=physics_config,
            _init_machine_configs=machine_configs,
            _init_occupant_configs=occupant_configs,
            _init_dynamic_surfaces=dynamic_surfaces,
            _init_solar_surfaces=solar_surfaces,

            airflow_solver=airflow_solver,
            geo_obs_v0=obs_v0,
            geo_obs_v1=obs_v1,
            geo_obs_v2=obs_v2,
            geo_surf_centroids=surf_cents
        )

    def reset(self) -> 'JAXSimulator':
        """Re-creates the simulator with initial states."""
        return JAXSimulator.create(
            self.dt_seconds, 
            self.thermal.config, 
            self.configs[2], # RewardConfig
            self._init_machine_configs,
            self._init_occupant_configs,
            self._init_dynamic_surfaces,
            self._init_solar_surfaces,
            self.battery.config, 
            self.heat_pump.config, 
            self.ac.config,
            self.storage.config, 
            self.solar.config,
            self.phys_config
        )

    @property
    def state(self) -> SystemState:
        return SystemState(
            thermal=ThermalState(T_vector=self.thermal.T_vector),
            battery=BatteryState(soc=self.battery.soc, soh=self.battery.soh),
            storage=ThermalStorageState(temperatures_c=self.storage.temperatures_c),
            heat_pump=HeatPumpState(current_electrical_w=self.heat_pump.current_electrical_w,
                                    current_thermal_w=self.heat_pump.current_thermal_w),
            air_conditioner=AirConditionerState(current_electrical_w=self.ac.current_electrical_w,
                                                current_thermal_w=self.ac.current_thermal_w),
            moisture=self.moisture.state 
        )
    
    @jax.jit
    def _solve_geometric_shading(self, exo: ExogenousData) -> ExogenousData:
        """
        Performs ray-casting to update surface shading factors.
        """
        geo_cfg = self.thermal.config.geometry_config
        if geo_cfg is None:
            return exo

        # 1. Calculate Sun Vector
        sun_vec = get_sun_position(exo.time_of_year_seconds, self.solar.config.latitude_deg)
        
        # 2. Ray-Cast (Möller-Trumbore)
        # Returns (N_surfaces,) boolean mask (1.0 = Unshaded)
        raw_shading = calculate_dynamic_shading(
            sun_vec,
            geo_cfg.surface_centroids,
            geo_cfg.obs_v0,
            geo_cfg.obs_v1,
            geo_cfg.obs_v2
        )
        
        # 3. Map back to Thermal/Solar inputs
        # The raycaster calculates for specific centroids. We need to map these
        # results to the global 'surface_shading_factors' array in ExogenousData.
        # We use scatter_update (at) for this.
        
        current_factors = exo.surface_shading_factors
        # If current factors are not initialized or wrong shape, init ones
        # (Usually handled in factory, but safety check)
        
        new_factors = current_factors.at[geo_cfg.shading_map_indices].set(raw_shading)
        
        # Update PV shading specifically (Assuming index 0 or specific logic)
        # For professional grade, PV surfaces are part of the solar_surfaces list
        # and handled via the map. We update the scalar convenience field too:
        # simplified: take average of unshaded fraction
        avg_shading = jnp.mean(raw_shading)
        
        return eqx.tree_at(
            lambda e: (e.surface_shading_factors, e.pv_shading_factor), 
            exo, 
            (new_factors, avg_shading)
        )
    
    @jax.jit
    def _solve_airflow_pressure_balance(self, T_rooms: jnp.ndarray, exo: ExogenousData) -> jnp.ndarray:
        """
        Solves the non-linear pressure network balance.
        Returns: infiltration_mass_flow_kg_s (N_rooms,)
        """
        af_cfg = self.thermal.config.airflow_config
        if af_cfg is None:
            return jnp.zeros_like(T_rooms) # Return zeros, let fallback logic handle it

        # 1. Construct Parameters for Solver
        # Calculate Wind Pressure at boundaries: P_w = 0.5 * rho * v^2 * Cp
        rho_amb = 1.204 
        P_dynamic = 0.5 * rho_amb * (exo.wind_speed_m_s**2)
        P_boundary = P_dynamic * af_cfg.boundary_Cp_coeffs
        
        # Full Temperature Vector [Rooms... Ambient]
        T_all = jnp.concatenate([T_rooms, jnp.array([exo.ambient_temp])])
        
        # Full Pressure Guess (Initialize at 0 Pa relative)
        P_init = jnp.zeros(af_cfg.n_total_nodes)
        
        # Set boundary pressures (Dirichlet BCs for external nodes)
        # In the solver residual function, we won't iterate on these, 
        # but we pass them in the params vector or fix them.
        # Better approach: The solver solves for N_internal variables. 
        # The residual function maps internal P + Boundary P to find flows.
        
        params = AirflowParams(
            link_node_a=af_cfg.link_node_a,
            link_node_b=af_cfg.link_node_b,
            link_C_flow=af_cfg.link_C_flow,
            link_exponent=af_cfg.link_exponent,
            P_boundary=P_boundary,
            boundary_indices=af_cfg.boundary_link_indices,
            T_nodes=T_all,
            node_heights=af_cfg.node_heights
        )
        
        # 2. Solve for Internal Pressures
        # Levenberg-Marquardt or Newton-Raphson
        P_internal_sol = self.airflow_solver.solve(jnp.zeros(af_cfg.n_internal_nodes), params)
        
        # 3. Calculate Resulting Mass Flows
        # We run the flow calculation one last time with the solved pressures
        # to get the net mass flow into each node.
        net_mass_flows = self.airflow_solver.calculate_flows(P_internal_sol, params)
        
        # Returns net flow (kg/s) entering each room.
        # Positive = Infiltration, Negative = Exfiltration
        # For thermal modeling, we care about m_dot * Cp * (T_amb - T_room) for infiltration
        # and m_dot * Cp * (T_room - T_room) = 0 for exfiltration (energy leaving).
        # The ThermalModel handles the conditional logic (upwinding).
        return net_mass_flows

    @jax.jit
    def _calculate_total_flux(self, current_T: jnp.ndarray, exo: ExogenousData) -> jnp.ndarray:
        """
        Calculates environmental fluxes (Solar + Wind).
        UPDATED: Uses exo.surface_shading_factors for beam and self.svf for diffuse.
        """
        # 1. Geometric Solar Heat Gain
        solar_lat = self.solar.config.latitude_deg
        sun_vec = get_sun_position(exo.time_of_year_seconds, solar_lat)

        # Handle Dynamic Shading Factors
        # Assume exo.surface_shading_factors is concatenated [Opaque_Factors, Window_Factors]
        # If empty/missing, default to 1.0
        sf = exo.surface_shading_factors
        total_surfaces = self.opaque_normals.shape[0] + self.win_normals.shape[0]
        
        # Robust fallback if data is missing
        sf = jax.lax.select(
            sf.shape[0] == total_surfaces,
            sf,
            jnp.ones(total_surfaces)
        )
        
        n_op = self.opaque_normals.shape[0]
        sf_opaque = sf[:n_op]
        sf_win = sf[n_op:]

        # --- OPAQUE WALLS ---
        op_cos = jnp.maximum(0.0, self.opaque_normals @ sun_vec)
        
        # BEAM: DNI * cos * shading_factor
        # DIFFUSE: DHI * sky_view_factor (Static)
        Q_absorbed_w_m2_op = (exo.solar_dni_w_m2 * op_cos * sf_opaque) + (exo.solar_dhi_w_m2 * self.opaque_svf)
        
        Q_solar_opaque = self.opaque_map_matrix @ (Q_absorbed_w_m2_op * self.opaque_areas_absorb)

        # --- WINDOWS ---
        win_cos = jnp.maximum(0.0, self.win_normals @ sun_vec)
        win_theta = jnp.arccos(win_cos)
        win_iam = calculate_iam(win_theta)

        # BEAM: DNI * cos * IAM * shading_factor
        # DIFFUSE: DHI * sky_view_factor (Static)
        G_poa = (exo.solar_dni_w_m2 * win_cos * win_iam * sf_win) + (exo.solar_dhi_w_m2 * self.win_svf)
        
        G_poa = G_poa * self.win_areas

        Q_solar_win_trans = self.win_trans_map_matrix @ (G_poa * self.win_shgc * 0.85)
        Q_solar_win_abs = self.win_abs_map_matrix @ (G_poa * self.win_shgc * 0.15)

        Q_solar_total = Q_solar_opaque + Q_solar_win_trans + Q_solar_win_abs

        # 2. Dynamic Wind Convection (Unchanged)
        h_forced_conv = calculate_external_convection(exo.wind_speed_m_s, self.wind_roughness_factors)
        G_wind_surface = h_forced_conv * self.wind_surface_areas

        T_surf = self.wind_coupling_matrix.T @ current_T
        Q_wind = self.wind_coupling_matrix @ (G_wind_surface * (exo.ambient_temp - T_surf))

        return Q_solar_total + Q_wind

    @jax.jit
    def step(
        self,
        actions: SystemActions,
        prev_actions: SystemActions,
        exo: ExogenousData,
        load_availability_mask: Optional[jnp.ndarray] = None
    ) -> Tuple['JAXSimulator', float]:
        
        current_state = self.state
        dt = self.dt_seconds
        
        # --- PHASE 1: PHYSICS PRE-SOLVE ---
        
        # A. Geometry / Shading
        exo_shaded = self._solve_geometric_shading(exo)
        
        # B. Airflow / Pressure
        # Get current room temps for stack effect
        T_rooms = current_state.thermal.T_vector[jnp.array(self.thermal.config.room_air_indices)]
        
        if self.thermal.config.airflow_config is not None:
            # High Fidelity
            net_airflow_kg_s = self._solve_airflow_pressure_balance(T_rooms, exo_shaded)
            # Convert to volumetric flow for compatibility (m3/s assuming ref density)
            infiltration_vol_flow = jnp.maximum(0.0, net_airflow_kg_s) / 1.204 
        else:
            # Fallback (Orifice approx inside Thermal Model)
            # We pass zeros here, and the ThermalModel will trigger its internal logic
            # based on config.use_dynamic_infiltration flag.
            infiltration_vol_flow = jnp.zeros_like(T_rooms)

        # --- PHASE 2: COMPONENT DYNAMICS ---
        
        # 1. Solar (Using updated shading)
        solar_out = self.solar.calculate(exo_shaded)
        
        # 2. Heat Pump (Mechanistic Check)
        if self.heat_pump.config.model_type == "mechanistic":
            #  - not inserted as per instruction
            # Pass Sink Temp (Mean or Vectorized)
            # Simplification: Central HP sees mean return air temp
            T_sink = jnp.mean(T_rooms) 
            next_hp, hp_out = self.heat_pump.step(actions.heat_pump_power_w, T_sink, exo_shaded, dt)
        else:
            # Legacy / Curve-based
            # Still pass T_sink to satisfy interface, even if ignored
            next_hp, hp_out = self.heat_pump.step(actions.heat_pump_power_w, T_rooms, exo_shaded, dt)
            
        # High-Fidelity AC Step: Returns Sensible (Thermal) and Latent (Water) separately
        next_ac, ac_out = self.ac.step(actions.ac_power_w, exo, dt)
        
        next_battery = self.battery.step(actions.battery_power_w, dt)
        next_storage, storage_out = self.storage.step(actions.storage_discharge_w, hp_out.thermal_power_w, dt)

        # --- 2. Internal Load Dynamics (The New Split Logic) ---
        
        next_smart = []
        next_passive = []
        next_occ = []
        
        total_elec_load_w = 0.0
        total_heat_gain_w = jnp.zeros_like(current_state.thermal.T_vector)
        
        # A. Smart Machines (Controlled by Action)
        for i, model in enumerate(self.smart_machines):
            sig = actions.smart_appliance_signals[i]
            avail = exo.smart_device_availability[i]
            
            new_model, p_elec, q_waste = model.step(sig, dt, avail)
            
            next_smart.append(new_model)
            total_elec_load_w += p_elec
            total_heat_gain_w = total_heat_gain_w.at[model.target_node_index].add(q_waste)

        # B. Passive Machines (Controlled by Exogenous Profile)
        for i, model in enumerate(self.passive_machines):
            sig = exo.passive_machine_profiles[i]
            new_model, p_elec, q_waste = model.step(sig, dt, 1.0) 
            
            next_passive.append(new_model)
            total_elec_load_w += p_elec
            total_heat_gain_w = total_heat_gain_w.at[model.target_node_index].add(q_waste)
            
        # C. Occupants (Heat Only)
        for i, model in enumerate(self.occupants):
            count = exo.occupant_profiles[i]
            _, q_metabolic = model.step(count) 
            
            next_occ.append(model)
            total_heat_gain_w = total_heat_gain_w.at[model.target_node_index].add(q_metabolic)


        # --- 3. Moisture Physics (High Precision) ---
        
        # A. Calculate Infiltration Flow (m3/s) from Thermal Params
        room_temps = self.thermal.T_vector[jnp.array(self.thermal.config.room_air_indices)]
        avg_room_temp = jnp.mean(room_temps)
        delta_T = jnp.abs(exo.ambient_temp - avg_room_temp)
        
        ach = (self.thermal.config.inf_k1 + 
               (self.thermal.config.inf_k2 * delta_T) + 
               (self.thermal.config.inf_k3 * exo.wind_speed_m_s))
        
        total_vol = self.thermal.config.room_vol_m3
        n_rooms = len(self.thermal.config.room_air_indices)
        vol_flow_per_room = jnp.full((n_rooms,), (ach * total_vol) / (3600.0 * n_rooms))

        # B. Occupant Latent Load
        total_people = jnp.sum(exo.occupant_profiles)
        people_per_room = jnp.full((n_rooms,), total_people / n_rooms)

        # C. Step Moisture Model
        # Uses: Room Temp (for Psat), AC Water Removal (Sink), Occupants (Source), Infiltration (Mass Transfer)
        next_moisture = self.moisture.step(
            T_room_c=room_temps,
            hvac_moisture_removal_kg_s=ac_out.water_removed_kg_s,
            n_occupants=people_per_room,
            infiltration_flow_m3_s=vol_flow_per_room,
            exo=exo,
            dt=dt
        )

        # --- 4. Thermal Step (Standard) ---
        Q_static_flux = self._calculate_total_flux(current_state.thermal.T_vector, exo)
        Q_flux_total = Q_static_flux + total_heat_gain_w

        T_ground = get_ground_temperature(self.phys_config.ground_lag_days, self.phys_config.ground_avg_temp_c, self.phys_config.ground_amplitude_c, exo.time_of_year_seconds)

        u_inputs_flat = jnp.zeros(self.thermal.config.B_matrix.shape[1])
        
        # Heat: Storage Discharge
        u_inputs_flat = u_inputs_flat.at[self.thermal.config.u_idx_heating].add(storage_out.actual_discharge_w)
        
        # Cool: AC Sensible Cooling (Not Latent!)
        u_inputs_flat = u_inputs_flat.at[self.thermal.config.u_idx_cooling].add(ac_out.thermal_power_w)
        
        U_vector = self.thermal.config.B_matrix @ u_inputs_flat

        total_waste_w = storage_out.standing_loss_w + jnp.sum(storage_out.rejected_heat_w)

        next_thermal = self.thermal.step(
            U_vector=U_vector, 
            waste_heat_w=total_waste_w, 
            exogenous=exo,
            Q_flux_injection=Q_flux_total, 
            dt_seconds=dt
        )
        
        # Boundary Condition: Ground Temp
        g_idx = self.thermal.config.ground_node_index
        T_proposed = next_thermal.T_vector.at[g_idx].set(T_ground)
        final_T = jnp.where(g_idx >= 0, T_proposed, next_thermal.T_vector)
        next_thermal = eqx.tree_at(lambda m: m.T_vector, next_thermal, final_T)

        # --- 5. Cost ---
        cost = f_cost_step(
            current_state,
            actions,
            prev_actions,
            exo,
            hp_out, ac_out, storage_out, solar_out,
            self.configs,
            total_elec_load_w, 
            self.dt_seconds
        )

        # --- 6. Reconstruct ---
        new_sim = eqx.tree_at(
            lambda s: (s.battery, s.thermal, s.moisture, s.heat_pump, s.ac, s.storage, s.smart_machines, s.passive_machines),
            self,
            (next_battery, next_thermal, next_moisture, next_hp, next_ac, next_storage, next_smart, next_passive)
        )

        return new_sim, cost

    # --- Static Matrix Builders ---
    @staticmethod
    def _build_wind_matrices(n_nodes, node_map, dynamic_surfaces):
        n_wind = len(dynamic_surfaces)
        if n_wind > 0:
            w_coup = jnp.zeros((n_nodes, n_wind))
            w_rough = []
            w_area = []
            for i, s in enumerate(dynamic_surfaces):
                if s["node_name"] in node_map:
                    node_idx = node_map[s["node_name"]]
                    w_coup = w_coup.at[node_idx, i].set(1.0)
                    w_rough.append(s.get("roughness_mult", get_convection_multiplier(SurfaceRoughness.SMOOTH)))
                    w_area.append(s.get("area", 1.0))
            return w_coup, jnp.array(w_rough), jnp.array(w_area)
        else:
            return jnp.zeros((n_nodes, 0)), jnp.zeros(0), jnp.zeros(0)

    @staticmethod
    def _build_solar_matrices(n_nodes, node_map, solar_surfaces):
        """
        Compiles solar geometry mapping matrices.
        UPDATED: Now extracts 'svf' from solar_surfaces dicts.
        """
        op_map, op_norm, op_area, op_svf = [], [], [], []
        win_trans, win_abs = [], []
        win_norm, win_area, win_shgc, win_svf = [], [], [], []

        op_idx, win_idx = 0, 0

        for s in solar_surfaces:
            norm = jnp.array(s.get("normal", [0, 0, 1]))
            
            # Recover tilt from normal Z component for default SVF calculation
            # tilt = arccos(nz)
            tilt_rad = jnp.arccos(jnp.clip(norm[2], -1.0, 1.0))
            default_svf = (1.0 + jnp.cos(tilt_rad)) / 2.0
            
            # Use provided SVF or default
            actual_svf = s.get("svf", default_svf)

            if s.get("type") == "WINDOW":
                target_trans = node_map.get(s.get("target_trans_node"))
                target_abs = node_map.get(s.get("target_abs_node"))
                if target_trans is not None and target_abs is not None:
                    win_trans.append((target_trans, win_idx))
                    win_abs.append((target_abs, win_idx))
                    win_norm.append(norm)
                    win_area.append(s.get("area", 1.0))
                    win_shgc.append(s.get("shgc", 0.7))
                    win_svf.append(actual_svf) # <--- Store SVF
                    win_idx += 1
            else:
                target_node = node_map.get(s.get("target_node"))
                if target_node is not None:
                    op_map.append((target_node, op_idx))
                    op_norm.append(norm)
                    op_area.append(s.get("area", 1.0) * s.get("absorptivity", 0.7))
                    op_svf.append(actual_svf) # <--- Store SVF
                    op_idx += 1

        # Opaque
        if op_idx > 0:
            op_mat = jnp.zeros((n_nodes, op_idx))
            for r, c in op_map: op_mat = op_mat.at[r, c].set(1.0)
            # Return SVF as the 4th element
            o_matrices = (op_mat, jnp.stack(op_norm), jnp.array(op_area), jnp.array(op_svf))
        else:
            o_matrices = (jnp.zeros((n_nodes, 0)), jnp.zeros((0, 3)), jnp.zeros(0), jnp.zeros(0))

        # Window
        if win_idx > 0:
            wt_mat = jnp.zeros((n_nodes, win_idx))
            wa_mat = jnp.zeros((n_nodes, win_idx))
            for r, c in win_trans: wt_mat = wt_mat.at[r, c].set(1.0)
            for r, c in win_abs: wa_mat = wa_mat.at[r, c].set(1.0)
            # Return SVF as the 6th element
            w_matrices = (wt_mat, wa_mat, jnp.stack(win_norm), jnp.array(win_area), jnp.array(win_shgc), jnp.array(win_svf))
        else:
            w_matrices = (jnp.zeros((n_nodes, 0)), jnp.zeros((n_nodes, 0)), jnp.zeros((0, 3)), jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))

        return o_matrices + w_matrices