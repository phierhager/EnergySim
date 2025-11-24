import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from typing import Tuple, Optional
from dataclasses import dataclass
from gymnax.environments import environment, spaces

from ..sim.simulator import JAXSimulator
from ..core.shared.data_structs import SystemState, SystemActions, ExogenousData

@dataclass
class EnvParams:
    price_weight: float = 1.0
    comfort_weight: float = 5.0
    max_steps_per_episode: int = 24 * 30

class EnvState(eqx.Module):
    sim: JAXSimulator
    prev_action: SystemActions
    time_idx: int
    episode_step: int
    key: chex.PRNGKey

class EnergyGymnaxEnv(environment.Environment):
    def __init__(self, simulator_template: JAXSimulator, exogenous_data: ExogenousData):
        super().__init__()
        self.sim_template = simulator_template
        self.exo_data = exogenous_data
        
        self.n_rooms = len(simulator_template.thermal.config.room_air_indices)
        self.dt = simulator_template.dt_seconds
        
        # Count controllable devices for action space sizing
        self.n_smart = len(simulator_template.smart_machines)
        
        # Normalization constants
        self.T_MIN, self.T_MAX = 10.0, 40.0
        self.PRICE_MAX = 0.50
        self.LOAD_MAX = 10000.0
        self.SOLAR_MAX = 1200.0

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(self, key: chex.PRNGKey, state: EnvState, action: jnp.ndarray, params: EnvParams):
        # 1. Map Action Vector to SystemActions
        # Structure: [Bat, HP(n_rooms), AC(n_rooms), Storage(n_rooms), SmartApps(n_smart)]
        
        # Battery
        bat_act = action[0] * self.sim_template.battery.config.max_power_w
        
        idx = 1
        # HP
        hp_acts = jnp.clip(action[idx:idx+self.n_rooms], 0.0, 1.0) * (self.sim_template.heat_pump.config.max_electrical_power_w / self.n_rooms)
        idx += self.n_rooms
        
        # AC
        ac_acts = jnp.clip(action[idx:idx+self.n_rooms], 0.0, 1.0) * (self.sim_template.ac.config.max_electrical_power_w / self.n_rooms)
        idx += self.n_rooms
        
        # Storage
        st_acts = jnp.clip(action[idx:idx+self.n_rooms], 0.0, 1.0) * (self.sim_template.storage.config.max_discharge_w / self.n_rooms)
        idx += self.n_rooms
        
        # Smart Appliances
        # We only generate signals for SMART machines. Passive machines use profiles inside Simulator.step.
        smart_acts = jnp.zeros(self.n_smart)
        if self.n_smart > 0:
            smart_acts = jnp.clip(action[idx:idx+self.n_smart], 0.0, 1.0)
            
        sys_actions = SystemActions(
            battery_power_w=bat_act,
            heat_pump_power_w=hp_acts,
            ac_power_w=ac_acts,
            storage_discharge_w=st_acts,
            smart_appliance_signals=smart_acts
        )

        # 2. Get Data
        exo_step = jax.tree.map(lambda x: x[state.time_idx], self.exo_data)
        
        # 3. Simulator Step
        # We need availability mask for smart machines. 
        # For now, assume always available (1.0) or pass from exo if you added an availability field there.
        smart_avail = jnp.ones(self.n_smart)
        
        next_sim, cost_val = state.sim.step(
            sys_actions, 
            state.prev_action, 
            exo_step, 
            load_availability_mask=smart_avail
        )

        # 4. Reward & State Update
        reward = -1.0 * cost_val
        
        next_step = state.episode_step + 1
        next_time = state.time_idx + 1
        done = (next_step >= params.max_steps_per_episode) | (next_time >= len(self.exo_data.price))
        
        next_state = EnvState(
            sim=next_sim,
            prev_action=sys_actions,
            time_idx=next_time,
            episode_step=next_step,
            key=key
        )
        
        return self.get_obs(next_state), next_state, reward, done, {}

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        new_sim = self.sim_template.reset()
        
        # Random start
        max_start = len(self.exo_data.price) - params.max_steps_per_episode - 1
        start_time = jax.random.randint(key, shape=(), minval=0, maxval=max_start)
        
        dummy_action = SystemActions(
            battery_power_w=jnp.array(0.0),
            heat_pump_power_w=jnp.zeros(self.n_rooms),
            ac_power_w=jnp.zeros(self.n_rooms),
            storage_discharge_w=jnp.zeros(self.n_rooms),
            smart_appliance_signals=jnp.zeros(self.n_smart)
        )
        
        state = EnvState(new_sim, dummy_action, start_time, 0, key)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState):
        # ... (Same logic as before, but ensure you don't access deleted fields) ...
        sim = state.sim
        exo = jax.tree.map(lambda x: x[state.time_idx], self.exo_data)
        
        # Add smart machine states to observation?
        # e.g. energy_remaining for EV
        
        # Minimal Obs:
        t_vec = sim.thermal.T_vector
        return t_vec # Expand this based on your needs

    @property
    def action_space(self):
        # 1 Bat + 3*Rooms + SmartApps
        dims = 1 + (3 * self.n_rooms) + self.n_smart
        return spaces.Box(low=-1.0, high=1.0, shape=(dims,), dtype=jnp.float32)

    @property
    def observation_space(self):
        return spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=jnp.float32) # Placeholder shape