import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from .shared.data_structs import ThermalConfig

@dataclass
class _RCNode:
    name: str
    capacity_j_k: float
    index: int = -1

@dataclass
class _Resistor:
    node_a_name: str
    node_b_name: str
    R_k_w: float

@dataclass
class _InputMapping:
    input_key: str
    node_name: str
    room_index: Optional[int]
    fraction: float

class RCNetworkBuilder:
    """
    Constructs the state-space matrices (A, B) and capacitance vectors
    for the RC Thermal Network.
    """
    def __init__(self, n_rooms: int):
        self.n_rooms = n_rooms
        self._nodes: Dict[str, _RCNode] = {}
        self._resistors: List[_Resistor] = []
        self._mappings: List[_InputMapping] = []
        
        # Infiltration / Coupling Settings
        self._infiltration_enabled = False
        self._inf_params = (0.1, 0.0, 0.0) # k1, k2, k3
        self._total_volume = 0.0
        self._waste_heat_node_name = None

        # Allowed inputs for the B-Matrix. Extendable as needed.
        self._input_keys_order = [
            "heating_w", 
            "cooling_w", 
        ]
        
        # Always exist
        self.add_node("ambient", capacity_j_k=jnp.inf)

    def add_node(self, name: str, capacity_j_k: float):
        if name in self._nodes: 
            raise ValueError(f"Node {name} exists.")
        self._nodes[name] = _RCNode(name, capacity_j_k)

    def add_resistor(self, node_a: str, node_b: str, R_k_w: float):
        if R_k_w <= 0: 
            raise ValueError("R must be > 0")
        self._resistors.append(_Resistor(node_a, node_b, R_k_w))

    def add_input_mapping(self, input_key: str, node_name: str, room_index: int, fraction: float = 1.0):
        """
        Maps an input vector index (e.g., "heating for room 0") to a specific thermal node.
        """
        # We allow dynamic keys now, but warn if they aren't in standard list
        if input_key not in self._input_keys_order:
            # If using the geometric engine, solar inputs are handled separately.
            # We permit the key for metadata tracking, but it might not end up in B-matrix columns
            # unless we expand _input_keys_order. 
            # For this implementation, we restrict to strict keys to prevent matrix shape mismatch.
            # If you need custom keys, add them to _input_keys_order in __init__.
            pass 

        if node_name not in self._nodes:
            raise ValueError(f"Unknown node {node_name}")
            
        self._mappings.append(_InputMapping(input_key, node_name, room_index, fraction))

    def set_infiltration(self, total_volume_m3: float, k1: float = 0.1, k2: float = 0.0, k3: float = 0.0):
        """Enables dynamic infiltration model."""
        self._infiltration_enabled = True
        self._total_volume = total_volume_m3
        self._inf_params = (k1, k2, k3)

    def set_waste_heat_node(self, node_name: str):
        """Sets the node where storage/HVAC waste heat is dumped."""
        if node_name not in self._nodes:
            raise ValueError(f"Unknown node {node_name}")
        self._waste_heat_node_name = node_name

    def _get_input_col_index(self, key: str, room_idx: int) -> int:
        """Calculates the column index in the B matrix."""
        if key not in self._input_keys_order:
            return -1
        base_offset = self._input_keys_order.index(key) * self.n_rooms
        return base_offset + room_idx

    def compile(self) -> ThermalConfig:
        # 1. Node ordering and Indexing
        node_names_sorted = sorted([n for n in self._nodes if n != "ambient"])
        final_node_order = ["ambient"] + node_names_sorted
        N_nodes = len(final_node_order)
        for i, name in enumerate(final_node_order):
            self._nodes[name].index = i

        # 2. Metadata Generation
        node_map = {name: i for i, name in enumerate(final_node_order)}
        c_inv_vector = np.zeros(N_nodes, dtype=np.float32)
        for name, node in self._nodes.items():
            if np.isinf(node.capacity_j_k):
                c_inv_vector[node.index] = 0.0
            else:
                c_inv_vector[node.index] = 1.0 / node.capacity_j_k

        # 3. A Matrix (omitted for brevity)
        A_matrix = np.zeros((N_nodes, N_nodes), dtype=np.float32)
        for res in self._resistors:
            i = self._nodes[res.node_a_name].index
            j = self._nodes[res.node_b_name].index
            G = 1.0 / res.R_k_w
            A_matrix[i, i] -= G; A_matrix[j, j] -= G
            A_matrix[i, j] += G; A_matrix[j, i] += G

        # 4. B Matrix Construction
        N_inputs_flat = len(self._input_keys_order) * self.n_rooms
        B_matrix = np.zeros((N_nodes, N_inputs_flat), dtype=np.float32)
        input_map_metadata = {k: {} for k in self._input_keys_order}

        for m in self._mappings:
            if m.input_key in self._input_keys_order:
                r_idx = self._nodes[m.node_name].index
                c_idx = self._get_input_col_index(m.input_key, m.room_index)
                
                B_matrix[r_idx, c_idx] += m.fraction
                input_map_metadata[m.input_key][m.node_name] = c_idx
        
        # 5. Extract Final Index Arrays (REQUIRED for JAXSimulator.step)
        
        u_idx_heating = []
        u_idx_cooling = []
        
        # Iterate over canonical room indices (0 to n_rooms-1)
        for i in range(self.n_rooms):
            # Check heating input map explicitly (guaranteed to exist if input_key is in _input_keys_order)
            heating_map = input_map_metadata.get("heating_w")
            if heating_map is None:
                 raise ValueError("Internal error: 'heating_w' key missing from input_map metadata.")
            
            # Since the builder guarantees 'heating_w' indices are sequential:
            h_col_idx = self._get_input_col_index("heating_w", i)
            u_idx_heating.append(h_col_idx)

            c_col_idx = self._get_input_col_index("cooling_w", i)
            u_idx_cooling.append(c_col_idx)

        # 6. Special Indices
        def find_indices(prefix):
            return tuple(sorted([n.index for name, n in self._nodes.items() if name.startswith(prefix)]))
        
        waste_idx = -1
        if self._waste_heat_node_name:
            waste_idx = self._nodes[self._waste_heat_node_name].index
            

        return ThermalConfig(
            A_matrix=jnp.array(A_matrix),
            C_inv_vector=jnp.array(c_inv_vector),
            B_matrix=jnp.array(B_matrix),
            
            # Metadata
            node_names=final_node_order,
            node_map=node_map,
            input_map=input_map_metadata,
            
            # B-Matrix Index Arrays (CRITICAL)
            u_idx_heating=jnp.array(u_idx_heating, dtype=jnp.int32),
            u_idx_cooling=jnp.array(u_idx_cooling, dtype=jnp.int32),
            
            # Indices
            ambient_air_index=self._nodes["ambient"].index,
            ground_node_index=self._nodes["ground"].index,
            room_air_indices=find_indices("room_air_"),
            wall_indices=find_indices("wall_"),
            mass_indices=find_indices("mass_"),
            waste_heat_node_index=waste_idx,
            
            # Params
            use_dynamic_infiltration=self._infiltration_enabled,
            room_vol_m3=self._total_volume,
            inf_k1=self._inf_params[0],
            inf_k2=self._inf_params[1],
            inf_k3=self._inf_params[2]
        )