import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import MoistureConfig, MoistureInputs, Array, DifferentialState
from ..physics import psychrometrics as psych

class AbstractMoistureModel(eqx.Module):
    config: MoistureConfig
    
    def dynamics(self, t: float, state: DifferentialState, inputs: MoistureInputs) -> Array:
        """

        Returns an array concatenating [dw/dt, du/dt].        Calculates derivatives for moisture states.
        """
        raise NotImplementedError


class DynamicMoistureModel(AbstractMoistureModel):
    """
    Standard Mass Balance Model.
    Tracks only air humidity ratio (w).
    Assumes no moisture buffering in walls.
    """
    def __init__(self, config: MoistureConfig):
        self.config = config

    def dynamics(self, t: float, state: DifferentialState, inputs: MoistureInputs) -> Array:
        """
        dw/dt = (Sum m_dot_water) / Mass_Air_Dry_Room
        """
        # Unpack State
        w_room = state.moisture_w
        
        # Unpack Inputs
        T_room = inputs.T_room_c
        atmospheric_pressure = inputs.atmospheric_pressure
        ambient_temp = inputs.ambient_temp
        relative_humidity = inputs.relative_humidity

        # 1. Physics Constants
        # Calculate dynamic air density based on current state
        rho_air = psych.calculate_air_density(T_room, atmospheric_pressure)
        mass_air_dry = self.config.air_volume_m3 * rho_air
        
        # Ambient Humidity Ratio
        w_amb = psych.calculate_humidity_ratio(
            ambient_temp, 
            relative_humidity, 
            atmospheric_pressure
        )
        
        # 2. Sources & Sinks (kg_water/s)
        
        # A. Infiltration Mass Transfer
        # m_dot_vapor = m_dot_dry_air * (w_amb - w_room)
        m_dot_air_inf = inputs.infiltration_flow_m3_s * rho_air
        m_dot_inf_vapor = m_dot_air_inf * (w_amb - w_room)
        
        # B. Internal Generation (Occupants)
        m_dot_gen = inputs.n_occupants * self.config.latent_gen_person_kg_s
        
        # C. HVAC Removal
        # Input is defined as removal rate (positive), so we subtract it
        m_dot_hvac = -inputs.hvac_moisture_removal_kg_s
        
        # 3. Derivative Calculation
        dw_dt = (m_dot_inf_vapor + m_dot_gen + m_dot_hvac) / mass_air_dry
        
        # EMPD buffer state doesn't exist here, return 0 derivative for it
        # (assuming fixed state vector size in global DAE)
        du_dt = jnp.zeros_like(state.moisture_buffer_u)
        
        return jnp.concatenate([dw_dt, du_dt])


class EMPDMoistureModel(AbstractMoistureModel):
    """
    Effective Moisture Penetration Depth (EMPD) Model.
    Tracks air humidity (w) AND wall buffer moisture content (u).
    Simulates hygroscopic inertia (e.g., gypsum board absorbing humidity peaks).
    
    State Vector: [w_room, u_buffer]
    """
    buffer_mass_dry: float

    def __init__(self, config: MoistureConfig):
        self.config = config
        # Pre-calculate static mass
        self.buffer_mass_dry = (
            config.buffer_area_m2 * config.empd_thickness * config.material_density
        )

    def dynamics(self, t: float, state: DifferentialState, inputs: MoistureInputs) -> Array:
        """
        Coupled ODE system:
        dw/dt = (Sources - BufferFlux) / Mass_Air
        du/dt = BufferFlux / Mass_Buffer
        """
        # Unpack State
        w_room = state.moisture_w
        u_wall = state.moisture_buffer_u # kg_water / kg_material
        
        # Unpack Inputs
        T_room = inputs.T_room_c
        atmospheric_pressure = inputs.atmospheric_pressure
        ambient_temp = inputs.ambient_temp
        relative_humidity = inputs.relative_humidity
        
        # 1. Physics Constants
        p_atm = atmospheric_pressure
        rho_air = psych.calculate_air_density(T_room, p_atm)
        mass_air_dry = self.config.air_volume_m3 * rho_air
        
        # 2. Sorption Isotherm (Physics of the Wall Surface)
        # Relates Moisture Content (u) to Surface Relative Humidity (phi)
        # Linear approximation: u = slope * phi  =>  phi = u / slope
        phi_surf = jnp.clip(u_wall / self.config.sorption_slope, 0.01, 0.99)
        
        # Convert Surface RH to Surface Humidity Ratio (w_surf)
        # This drives the gradient between air and wall
        w_surf = psych.calculate_humidity_ratio(T_room, phi_surf, p_atm)
        
        # 3. Buffer Mass Transfer Rate (m_dot_buffer)
        # m_dot = beta * Area * rho * (w_air - w_surf)
        # beta ~ 2e-3 m/s (mass transfer coefficient)
        beta = 2.0e-3
        m_dot_buffer = beta * self.config.buffer_area_m2 * rho_air * (w_room - w_surf)
        
        # 4. Room Air Balance
        w_amb = psych.calculate_humidity_ratio(ambient_temp, relative_humidity, p_atm)
        
        m_dot_air_inf = inputs.infiltration_flow_m3_s * rho_air
        m_dot_inf = m_dot_air_inf * (w_amb - w_room)
        
        m_dot_gen = inputs.n_occupants * self.config.latent_gen_person_kg_s
        m_dot_hvac = -inputs.hvac_moisture_removal_kg_s
        
        # dw/dt: Note that positive m_dot_buffer means moisture LEAVING air into wall
        dw_dt = (m_dot_inf + m_dot_gen + m_dot_hvac - m_dot_buffer) / mass_air_dry
        
        # 5. Buffer Storage Balance
        # du/dt = Mass_Flow_Into_Wall / Dry_Mass_Of_Wall
        du_dt = m_dot_buffer / self.buffer_mass_dry
        
        return jnp.concatenate([dw_dt, du_dt])