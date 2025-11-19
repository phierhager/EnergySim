import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional, Dict, List, Any

# --- Import Core Components ---
from ..core.models.factory import (
    create_battery, create_thermal, create_heat_pump,
    create_ac, create_storage, create_solar, create_appliances
)
from ..core.models.battery_model import AbstractBatteryModel
from ..core.models.thermal_model import AbstractThermalModel
from ..core.models.heat_pump_model import AbstractHeatPumpModel
from ..core.models.air_conditioner_model import AbstractAirConditionerModel
from ..core.models.thermal_storage_model import AbstractThermalStorage
from ..core.models.solar_model import AbstractSolarModel
from ..core.models.appliance_model import AbstractApplianceModel 

from ..core.models.objectives import f_cost_step

from ..core.shared.data_structs import (
    SystemActions, ExogenousData, SystemState,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig,
    BatteryState, ThermalState, ThermalStorageState,
    HeatPumpState, AirConditionerState, ApplianceConfig
)
from ..core.physics.coefficients import PhysicsConfig, SurfaceRoughness
from ..core.physics.optics import calculate_iam

# --- Helper: Sun Position ---
def get_sun_position(time_seconds: float, latitude_deg: float = 48.0) -> jnp.ndarray:
    """Calculates sun position vector [x, y, z]."""
    day_seconds = 86400.0
    year_seconds = 365.0 * day_seconds
    day_progress = (time_seconds % day_seconds) / day_seconds
    year_progress = time_seconds / year_seconds
    hour_angle = (day_progress - 0.5) * 2 * jnp.pi
    declination = jnp.radians(23.45) * -jnp.cos(2 * jnp.pi * (year_progress + 10.0/365.0))
    lat_rad = jnp.radians(latitude_deg)
    sin_elev = (jnp.sin(lat_rad) * jnp.sin(declination) + 
                jnp.cos(lat_rad) * jnp.cos(declination) * jnp.cos(hour_angle))
    elev = jnp.arcsin(jnp.clip(sin_elev, -1.0, 1.0))
    cos_az = (jnp.sin(declination) - jnp.sin(elev) * jnp.sin(lat_rad)) / \
             (jnp.cos(elev) * jnp.cos(lat_rad) + 1e-6)
    azimuth = jnp.arccos(jnp.clip(cos_az, -1.0, 1.0))
    z = jnp.sin(elev)
    y = jnp.cos(elev) * jnp.cos(azimuth)
    x_sign = jnp.sign(hour_angle)
    x = x_sign * jnp.cos(elev) * jnp.sin(azimuth)
    sun_vec = jnp.array([x, y, z])
    is_day = z > 0
    return jnp.where(is_day, sun_vec / (jnp.linalg.norm(sun_vec) + 1e-6), jnp.zeros(3))


class JAXSimulator(eqx.Module):
    """
    A purely functional Simulator using JAX.
    Initialized via the `create()` factory method to ensure immutability.
    """
    # --- Sub-Models ---
    battery: AbstractBatteryModel
    thermal: AbstractThermalModel
    heat_pump: AbstractHeatPumpModel
    ac: AbstractAirConditionerModel
    storage: AbstractThermalStorage
    solar: AbstractSolarModel
    
    # List of ALL appliances
    appliances: List[AbstractApplianceModel]
    
    # Mask: 1.0 = Smart, 0.0 = Passive
    appliance_type_mask: jnp.ndarray

    # --- Pre-compiled Physics Matrices ---
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

    # --- Configs & Constants ---
    configs: Tuple = eqx.field(static=True)
    dt_seconds: float = eqx.field(static=True)
    phys_config: PhysicsConfig = eqx.field(static=True)
    
    # Reset Metadata (Stored for reproduction)
    _init_app_configs: List[ApplianceConfig] = eqx.field(static=True)
    _init_dynamic_surfaces: List[Dict] = eqx.field(static=True)
    _init_solar_surfaces: List[Dict] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        dt_seconds: float,
        t_config: ThermalConfig,
        r_config: RewardConfig,
        appliances: List[ApplianceConfig],
        dynamic_surfaces: List[Dict[str, Any]],
        solar_surfaces: List[Dict[str, Any]],
        b_config: Optional[BatteryConfig] = None,
        hp_config: Optional[HeatPumpConfig] = None,
        ac_config: Optional[AirConditionerConfig] = None,
        ts_config: Optional[ThermalStorageConfig] = None,
        s_config: Optional[SolarConfig] = None,
        physics_config: PhysicsConfig = PhysicsConfig()
    ) -> 'JAXSimulator':
        """
        Factory method to initialize the simulator. 
        Performs all pre-computation before instantiating the frozen class.
        """
        
        # 1. Create Sub-Models
        n_rooms = len(t_config.room_air_indices)
        battery = create_battery(b_config)
        thermal = create_thermal(t_config)
        heat_pump = create_heat_pump(hp_config, n_rooms)
        ac = create_ac(ac_config, n_rooms)
        storage = create_storage(ts_config)
        solar = create_solar(s_config)
        
        # 2. Create Appliances & Type Mask
        app_models = create_appliances(appliances, t_config.node_map)
        
        mask_list = []
        for app in appliances:
            is_smart = (app.cycle_energy_kwh is not None and app.cycle_energy_kwh > 0)
            mask_list.append(1.0 if is_smart else 0.0)
        
        appliance_type_mask = jnp.array(mask_list) if mask_list else jnp.zeros(0)

        # 3. Bundle Configs
        configs = (
            thermal.config, battery.config, r_config,
            heat_pump.config, ac.config,
            storage.config, solar.config
        )

        # 4. Build Physics Matrices
        n_nodes = len(t_config.node_names)
        node_map = t_config.node_map
        
        w_coup, w_rough, w_area = cls._build_wind_matrices(n_nodes, node_map, dynamic_surfaces)
        
        (o_map, o_norm, o_area, 
         w_t_map, w_a_map, w_norm, w_area_arr, w_shgc) = cls._build_solar_matrices(n_nodes, node_map, solar_surfaces)

        # 5. Instantiate Immutable Class
        return cls(
            battery=battery,
            thermal=thermal,
            heat_pump=heat_pump,
            ac=ac,
            storage=storage,
            solar=solar,
            appliances=app_models,
            appliance_type_mask=appliance_type_mask,
            
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
            
            configs=configs,
            dt_seconds=dt_seconds,
            phys_config=physics_config,
            
            _init_app_configs=appliances,
            _init_dynamic_surfaces=dynamic_surfaces,
            _init_solar_surfaces=solar_surfaces
        )

    def reset(self) -> 'JAXSimulator':
        """Re-creates the simulator with initial states."""
        return JAXSimulator.create(
            self.dt_seconds, 
            self.thermal.config, 
            self.configs[2], # RewardConfig
            self._init_app_configs,
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
                                                current_thermal_w=self.ac.current_thermal_w)
        )

    @jax.jit
    def _calculate_total_flux(self, current_T: jnp.ndarray, exo: ExogenousData) -> jnp.ndarray:
        """Calculates environmental fluxes (Solar + Wind)."""
        # 1. Geometric Solar Heat Gain
        solar_lat = self.solar.config.latitude_deg
        sun_vec = get_sun_position(exo.time_of_year_seconds, solar_lat)

        # Opaque Walls
        op_cos = jnp.maximum(0.0, self.opaque_normals @ sun_vec)
        op_z_normal = self.opaque_normals[:, 2]
        op_view_factor = (1.0 + op_z_normal) / 2.0
        
        Q_absorbed_w_m2_op = (exo.solar_dni_w_m2 * op_cos) + (exo.solar_dhi_w_m2 * op_view_factor)
        Q_solar_opaque = self.opaque_map_matrix @ (Q_absorbed_w_m2_op * self.opaque_areas_absorb)

        # Windows
        win_cos = jnp.maximum(0.0, self.win_normals @ sun_vec)
        win_theta = jnp.arccos(win_cos)
        win_iam = calculate_iam(win_theta)

        win_z_normal = self.win_normals[:, 2]
        win_view_factor = (1.0 + win_z_normal) / 2.0

        G_poa = (exo.solar_dni_w_m2 * win_cos * win_iam) + (exo.solar_dhi_w_m2 * win_view_factor)
        G_poa = G_poa * self.win_areas

        Q_solar_win_trans = self.win_trans_map_matrix @ (G_poa * self.win_shgc * 0.85)
        Q_solar_win_abs = self.win_abs_map_matrix @ (G_poa * self.win_shgc * 0.15)

        Q_solar_total = Q_solar_opaque + Q_solar_win_trans + Q_solar_win_abs

        # 2. Dynamic Wind Convection
        h_forced_conv = self.phys_config.calculate_external_convection(exo.wind_speed_m_s, self.wind_roughness_factors)
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
        appliance_availability: jnp.ndarray 
    ) -> Tuple['JAXSimulator', float]:
        
        current_state = self.state
        dt = self.dt_seconds

        # --- 1. Multiplex Signals ---
        active_signals = actions.appliance_signals * self.appliance_type_mask
        passive_profiles = exo.appliance_profiles * (1.0 - self.appliance_type_mask)
        effective_signals = active_signals + passive_profiles

        # --- 2. Component Dynamics ---
        solar_out = self.solar.calculate(exo)
        next_hp, hp_out = self.heat_pump.step(actions.heat_pump_power_w, exo, dt)
        next_ac, ac_out = self.ac.step(actions.ac_power_w, exo, dt)
        next_battery = self.battery.step(actions.battery_power_w, dt)
        next_storage, storage_out = self.storage.step(
            actions.storage_discharge_w, hp_out.thermal_power_w, dt
        )

        # --- 3. Appliance Dynamics ---
        next_apps = []
        total_elec_load_w = 0.0
        total_heat_gain_w = jnp.zeros_like(current_state.thermal.T_vector)
        
        # JAX will unroll this loop because self.appliances is a static list
        for i, app_model in enumerate(self.appliances):
            sig = effective_signals[i]
            avail = appliance_availability[i]
            
            new_app_model, p_elec, q_heat = app_model.step(sig, dt, avail)
            
            next_apps.append(new_app_model)
            total_elec_load_w += p_elec
            
            node_idx = app_model.target_node_index
            total_heat_gain_w = total_heat_gain_w.at[node_idx].add(q_heat)

        # --- 4. Physics & Boundaries ---
        Q_static_flux = self._calculate_total_flux(current_state.thermal.T_vector, exo)
        Q_flux_total = Q_static_flux + total_heat_gain_w

        T_ground = self.phys_config.get_ground_temperature(exo.time_of_year_seconds)

        # --- 5. Thermal Step ---
        u_inputs_flat = jnp.zeros(self.thermal.config.B_matrix.shape[1])
        u_inputs_flat = u_inputs_flat.at[self.thermal.config.u_idx_heating].add(storage_out.actual_discharge_w)
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
        
        # Ground Override
        g_idx = self.thermal.config.ground_node_index
        T_proposed = next_thermal.T_vector.at[g_idx].set(T_ground)
        final_T = jnp.where(g_idx >= 0, T_proposed, next_thermal.T_vector)
        next_thermal = eqx.tree_at(lambda m: m.T_vector, next_thermal, final_T)

        # --- 6. Cost ---
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

        # --- 7. Reconstruct ---
        # Standard Equinox functional update
        new_sim = eqx.tree_at(
            lambda s: (s.battery, s.thermal, s.heat_pump, s.ac, s.storage, s.appliances),
            self,
            (next_battery, next_thermal, next_hp, next_ac, next_storage, next_apps)
        )

        return new_sim, cost

    # --- Static Matrix Builders (Pure Functions) ---
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
                    w_rough.append(s.get("roughness_mult", SurfaceRoughness.SMOOTH.get_multiplier()))
                    w_area.append(s.get("area", 1.0))
            return w_coup, jnp.array(w_rough), jnp.array(w_area)
        else:
            return jnp.zeros((n_nodes, 0)), jnp.zeros(0), jnp.zeros(0)

    @staticmethod
    def _build_solar_matrices(n_nodes, node_map, solar_surfaces):
        op_map, op_norm, op_area = [], [], []
        win_trans, win_abs = [], []
        win_norm, win_area, win_shgc = [], [], []
        
        op_idx, win_idx = 0, 0

        for s in solar_surfaces:
            norm = jnp.array(s.get("normal", [0, 0, 1]))

            if s.get("type") == "WINDOW":
                target_trans = node_map.get(s.get("target_trans_node"))
                target_abs = node_map.get(s.get("target_abs_node"))
                if target_trans is not None and target_abs is not None:
                    win_trans.append((target_trans, win_idx))
                    win_abs.append((target_abs, win_idx))
                    win_norm.append(norm)
                    win_area.append(s.get("area", 1.0))
                    win_shgc.append(s.get("shgc", 0.7))
                    win_idx += 1
            else:
                target_node = node_map.get(s.get("target_node"))
                if target_node is not None:
                    op_map.append((target_node, op_idx))
                    op_norm.append(norm)
                    op_area.append(s.get("area", 1.0) * s.get("absorptivity", 0.7))
                    op_idx += 1

        # Opaque
        if op_idx > 0:
            op_mat = jnp.zeros((n_nodes, op_idx))
            for r, c in op_map: op_mat = op_mat.at[r, c].set(1.0)
            o1, o2, o3 = op_mat, jnp.stack(op_norm), jnp.array(op_area)
        else:
            o1, o2, o3 = jnp.zeros((n_nodes, 0)), jnp.zeros((0, 3)), jnp.zeros(0)

        # Window
        if win_idx > 0:
            wt_mat = jnp.zeros((n_nodes, win_idx))
            wa_mat = jnp.zeros((n_nodes, win_idx))
            for r, c in win_trans: wt_mat = wt_mat.at[r, c].set(1.0)
            for r, c in win_abs: wa_mat = wa_mat.at[r, c].set(1.0)
            w1, w2, w3, w4, w5 = wt_mat, wa_mat, jnp.stack(win_norm), jnp.array(win_area), jnp.array(win_shgc)
        else:
            w1, w2, w3, w4, w5 = jnp.zeros((n_nodes, 0)), jnp.zeros((n_nodes, 0)), jnp.zeros((0, 3)), jnp.zeros(0), jnp.zeros(0)

        return o1, o2, o3, w1, w2, w3, w4, w5