import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import MoistureConfig, MoistureState, ExogenousData, Array
from ..physics import psychrometrics as psych

class AbstractMoistureModel(eqx.Module):
    state: MoistureState
    config: MoistureConfig
    
    @eqx.filter_jit
    def step(self, 
             T_room_c: Array, 
             hvac_moisture_removal_kg_s: Array, 
             n_occupants: Array, 
             infiltration_flow_m3_s: Array, # Coupling from Thermal Model
             exo: ExogenousData, 
             dt: float
             ) -> 'AbstractMoistureModel':
        raise NotImplementedError

class DynamicMoistureModel(AbstractMoistureModel):
    def __init__(self, config: MoistureConfig, state: MoistureState):
        self.config = config
        self.state = state

    @eqx.filter_jit
    def step(self, 
             T_room_c: Array, 
             hvac_moisture_removal_kg_s: Array, 
             n_occupants: Array, 
             infiltration_flow_m3_s: Array,
             exo: ExogenousData, 
             dt: float
             ) -> 'DynamicMoistureModel':
        
        # 1. Calculate Ambient Moisture State
        # Use Exogenous P and RH to get Ambient Humidity Ratio (w_amb)
        # Note: We handle scalar or vector inputs via broadcasting
        w_amb = psych.calculate_humidity_ratio(
            exo.ambient_temp, 
            exo.relative_humidity, 
            exo.atmospheric_pressure
        )
        
        # 2. Internal Generation (Latent Load)
        # People breathing/sweating
        m_dot_gen = n_occupants * self.config.latent_gen_person_kg_s
        
        # 3. Infiltration Mass Transfer
        # m_dot_inf = Vol_flow * rho_air * (w_amb - w_room)
        # We calculate dynamic air density based on current room T and Ambient P
        rho_air = psych.calculate_air_density(T_room_c, exo.atmospheric_pressure)
        
        # Mass flow of dry air entering (kg_air/s)
        m_dot_air_inf = infiltration_flow_m3_s * rho_air
        
        # Moisture added/lost via airflow
        m_dot_inf_water = m_dot_air_inf * (w_amb - self.state.humidity_ratio_kg_kg)
        
        # 4. HVAC Removal (Dehumidification)
        # This input is negative (removal), coming from AC model
        m_dot_hvac = -hvac_moisture_removal_kg_s
        
        # 5. Mass Balance Integration
        # dw/dt = Sum(m_dot_water) / Mass_Air_Dry_Room
        # Mass_Air_Dry_Room approx constant or dynamic. Let's use config volume * const density for stability,
        # or use the dynamic density calculated above. Dynamic is better.
        room_dry_air_mass = self.config.air_volume_m3 * rho_air
        
        dw_dt = (m_dot_inf_water + m_dot_gen + m_dot_hvac) / room_dry_air_mass
        
        w_next = self.state.humidity_ratio_kg_kg + (dw_dt * dt)
        
        # Physics Clamp: 0 < w < 0.030 (approx saturation at 35C)
        w_next = jnp.clip(w_next, 1e-5, 0.030)
        
        # Update State
        new_state = eqx.tree_at(lambda s: s.humidity_ratio_kg_kg, self.state, w_next)
        return eqx.tree_at(lambda m: m.state, self, new_state)

    def get_relative_humidity(self, T_room_c: Array, P_amb: Array) -> Array:
        """Helper to get current RH% for comfort observation."""
        return psych.calculate_relative_humidity(T_room_c, self.state.humidity_ratio_kg_kg, P_amb) * 100.0
    
class EMPDMoistureModel(AbstractMoistureModel):
    """
    Effective Moisture Penetration Depth (EMPD) Model.
    Simulates moisture buffering in hygroscopic materials (gypsum, wood, textiles).
    """
    def __init__(self, config: MoistureConfig, state: MoistureState):
        self.config = config
        self.state = state

        # Dry buffer mass (kg_material)
        self.buffer_mass_dry = (
            config.buffer_area_m2
            * config.empd_thickness
            * config.material_density
        )

    @eqx.filter_jit
    def step(
        self,
        T_room_c: Array,
        hvac_moisture_removal_kg_s: Array,
        n_occupants: Array,
        infiltration_flow_m3_s: Array,
        exo: ExogenousData,
        dt: float,
    ) -> "EMPDMoistureModel":

        # ---- States ----
        w_room = self.state.humidity_ratio_kg_kg           # kg_water/kg_dry_air
        u_wall = self.state.buffer_moisture_content        # kg_water/kg_material

        # ---- Air properties ----
        p_atm = exo.atmospheric_pressure
        rho_air = psych.calculate_air_density(T_room_c, p_atm)

        # ---- Sorption interface ----
        phi_surf = jnp.clip(u_wall / self.config.sorption_slope, 0.01, 0.99)
        w_surf = psych.calculate_humidity_ratio(T_room_c, phi_surf, p_atm)

        # ---- Mass transfer between room air and buffer ----
        beta = 2.0e-3  # m/s
        m_dot_buffer = beta * self.config.buffer_area_m2 * rho_air * (w_room - w_surf)

        # ---- Ambient infiltration ----
        w_amb = psych.calculate_humidity_ratio(
            exo.ambient_temp, exo.relative_humidity, p_atm
        )
        m_dot_inf_dry = infiltration_flow_m3_s * rho_air
        m_dot_inf_water = m_dot_inf_dry * (w_amb - w_room)

        # ---- Internal gains + HVAC ----
        m_dot_gen = n_occupants * self.config.latent_gen_person_kg_s
        m_dot_hvac = -hvac_moisture_removal_kg_s

        # ---- Room moisture balance ----
        m_dry_air_room = self.config.air_volume_m3 * rho_air
        dw_dt = (m_dot_inf_water + m_dot_gen + m_dot_hvac - m_dot_buffer) / m_dry_air_room

        w_next = jnp.clip(w_room + dt * dw_dt, 1e-6, 0.03)

        # ---- Buffer moisture balance ----
        du_dt = m_dot_buffer / self.buffer_mass_dry
        u_next = jnp.clip(u_wall + dt * du_dt, 0.0, 0.30)

        # ---- Update ----
        new_state = MoistureState(
            humidity_ratio_kg_kg=w_next,
            buffer_moisture_content=u_next,
        )
        return eqx.tree_at(lambda m: m.state, self, new_state)
