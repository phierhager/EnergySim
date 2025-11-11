import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, List, Callable, Any

from ..sim.simulator import JAXSimulator
from ..core.data.dataset import SimulationDataset
from ..core.shared.data_structs import (
    SystemActions, SystemState, ExogenousData
)

# Define the signature for the programmatic function
# It takes the current step index and current state, and returns a dict
ProgrammaticFn = Callable[[int, SystemState], Dict[str, float]]

class EnergySimEnv(gym.Env):
    """
    A Gymnasium Env wrapper that orchestrates the JAXSimulator.
    
    It manages:
    - The simulator instance (state machine)
    - The data source (dataset)
    - The programmatic external inputs (external_fn)
    - The simulation loop (time step)
    """
    metadata = {"render_modes": []}

    def __init__(
        self, 
        simulator: JAXSimulator,
        dataset: SimulationDataset,
        external_fn: Optional[ProgrammaticFn] = None
    ):
        super().__init__()
        self.simulator = simulator
        self.dataset = dataset
        self.external_fn = external_fn
        self._current_step = 0
        
        # Dynamically build action/obs spaces
        self._build_spaces() 

    def _build_spaces(self):
        self.action_map: Dict[str, int] = {}
        action_lows: List[float] = []
        action_highs: List[float] = []
        idx_a = 0
        if self.simulator.active_configs["battery"]:
            conf = self.simulator.initial_battery.config
            self.action_map["battery_power_w"] = idx_a
            action_lows.append(-conf.max_power_w)
            action_highs.append(conf.max_power_w)
            idx_a += 1
        if self.simulator.active_configs["heat_pump"]:
            conf = self.simulator.initial_heat_pump.config
            self.action_map["heat_pump_power_w"] = idx_a
            action_lows.append(0.0)
            action_highs.append(conf.max_electrical_power_w)
            idx_a += 1
        if self.simulator.active_configs["ac"]:
            conf = self.simulator.initial_ac.config
            self.action_map["ac_power_w"] = idx_a
            action_lows.append(0.0)
            action_highs.append(conf.max_electrical_power_w)
            idx_a += 1
        if self.simulator.active_configs["storage"]:
            conf = self.simulator.initial_storage.config
            self.action_map["storage_discharge_w"] = idx_a
            action_lows.append(0.0)
            action_highs.append(conf.max_discharge_w)
            idx_a += 1
        self.action_space = spaces.Box(
            low=np.array(action_lows, dtype=np.float32),
            high=np.array(action_highs, dtype=np.float32)
        )
        self.obs_map: Dict[str, int] = {}
        obs_lows: List[float] = []
        obs_highs: List[float] = []
        idx_o = 0
        
        # Core thermal states
        self.obs_map["room_temp"] = idx_o
        obs_lows.append(-np.inf)
        obs_highs.append(np.inf)
        idx_o += 1
        
        # ADD wall_temp if using a model that has it (we'll just add it always)
        self.obs_map["wall_temp"] = idx_o 
        obs_lows.append(-np.inf)
        obs_highs.append(np.inf)
        idx_o += 1
        
        if self.simulator.active_configs["battery"]:
            self.obs_map["battery_soc"] = idx_o
            obs_lows.append(0.0)
            obs_highs.append(1.0)
            idx_o += 1
        if self.simulator.active_configs["storage"]:
            self.obs_map["storage_soc"] = idx_o
            obs_lows.append(0.0)
            obs_highs.append(1.0)
            idx_o += 1

        # --- NEW Observations ---
        if self.simulator.active_configs["heat_pump"]:
            conf = self.simulator.initial_heat_pump.config
            self.obs_map["hp_current_w"] = idx_o
            obs_lows.append(0.0)
            obs_highs.append(conf.max_electrical_power_w)
            idx_o += 1
        if self.simulator.active_configs["ac"]:
            conf = self.simulator.initial_ac.config
            self.obs_map["ac_current_w"] = idx_o
            obs_lows.append(0.0)
            obs_highs.append(conf.max_electrical_power_w)
            idx_o += 1

        # Exogenous data
        exo_keys = ["ambient_temp", "load", "pv", "price"]
        for key in exo_keys:
            self.obs_map[key] = idx_o
            obs_lows.append(-np.inf)
            obs_highs.append(np.inf)
            idx_o += 1
            
        self.observation_space = spaces.Box(
            low=np.array(obs_lows, dtype=np.float32),
            high=np.array(obs_highs, dtype=np.float32)
        )

    def _unflatten_action(self, action: np.ndarray) -> SystemActions:
        all_actions = {"battery_power_w": 0.0, "heat_pump_power_w": 0.0, "ac_power_w": 0.0, "storage_discharge_w": 0.0}
        for key, idx in self.action_map.items():
            all_actions[key] = float(action[idx])
        return SystemActions(
            battery_power_w=jnp.array(all_actions["battery_power_w"]),
            heat_pump_power_w=jnp.array(all_actions["heat_pump_power_w"]),
            ac_power_w=jnp.array(all_actions["ac_power_w"]),
            storage_discharge_w=jnp.array(all_actions["storage_discharge_w"])
        )

    def _get_merged_exo_data(self, step_idx: int) -> ExogenousData:
        """Helper to get data from dataset and merge with programmatic inputs."""
        
        # 1. Get base data from the dataset
        base_exo = self.dataset[step_idx]
        
        # 2. If no external function, return base data
        if self.external_fn is None:
            return base_exo
            
        # 3. Get programmatic data
        # We pass the *current* state (from *before* the step)
        external_data_dict = self.external_fn(step_idx, self.simulator.state)
        
        # 4. Merge it
        # .replace() is a handy flax.struct method for an immutable update
        merged_exo = base_exo
        for key, value in external_data_dict.items():
            if hasattr(merged_exo, key):
                merged_exo = merged_exo.replace(**{key: jnp.array(value)})
            else:
                # This could be a warning, but for simplicity we'll raise an error
                raise KeyError(f"Programmatic key '{key}' not found in ExogenousData PyTree.")
                
        return merged_exo

    def _build_obs(self, state: SystemState, exo: ExogenousData) -> np.ndarray:
        # --- UPDATED ---
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Core states
        obs[self.obs_map["room_temp"]] = state.thermal.room_temp
        obs[self.obs_map["wall_temp"]] = state.thermal.wall_temp
        
        if "battery_soc" in self.obs_map:
            obs[self.obs_map["battery_soc"]] = state.battery.soc
        if "storage_soc" in self.obs_map:
            obs[self.obs_map["storage_soc"]] = state.storage.soc
            
        # New HVAC states
        if "hp_current_w" in self.obs_map:
            obs[self.obs_map["hp_current_w"]] = state.heat_pump.current_electrical_w
        if "ac_current_w" in self.obs_map:
            obs[self.obs_map["ac_current_w"]] = state.air_conditioner.current_electrical_w

        # Exogenous data
        obs[self.obs_map["ambient_temp"]] = exo.ambient_temp
        obs[self.obs_map["load"]] = exo.load
        obs[self.obs_map["pv"]] = exo.pv
        obs[self.obs_map["price"]] = exo.price
        
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Reset the simulator's internal state
        state = self.simulator.reset()
        
        # Reset time
        self._current_step = 0
        
        # Get the *first* set of merged exogenous data for the observation
        exo_data = self._get_merged_exo_data(self._current_step)
        
        obs = self._build_obs(state, exo_data)
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        # 1. Get merged exogenous data for the *current* step
        exo_data_k = self._get_merged_exo_data(self._current_step)
        
        # 2. Convert action from RL agent to simulator format
        actions_struct = self._unflatten_action(action)
        
        # 3. Step the simulator
        # The simulator is given the data, it doesn't fetch it
        next_state, cost = self.simulator.step(actions_struct, exo_data_k)
        
        # 4. Advance time
        self._current_step += 1
        
        # 5. Check for termination
        terminated = self._current_step >= len(self.dataset)
        truncated = False
        
        # 6. Get exogenous data for the *next* step (to build the observation)
        if terminated:
            next_exo_data = exo_data_k # Use last available data
        else:
            next_exo_data = self._get_merged_exo_data(self._current_step)
        
        # 7. Build results for Gym
        obs = self._build_obs(next_state, next_exo_data)
        reward = -cost
        info = {"cost": cost}
        
        return obs, reward, terminated, truncated, info