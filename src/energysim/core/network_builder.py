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
    dynamic: bool = False

@dataclass
class _MixingConnection:
    node_a_name: str
    node_b_name: str
    flow_rate_m3_s: float

@dataclass
class _InputMapping:
    input_key: str
    node_name: str
    room_index: Optional[int]
    fraction: float

class RCNetworkBuilder:
    def __init__(self, n_rooms: int):
        self.n_rooms = n_rooms
        self._nodes: Dict[str, _RCNode] = {}
        self._resistors: List[_Resistor] = []
        self._mappings: List[_InputMapping] = []
        self._mixing_links: List[_MixingConnection] = [] 
        
        self._infiltration_enabled = False
        self._inf_params = (0.1, 0.0, 0.0) 
        self._total_volume = 0.0
        self._waste_heat_node_name = None
        self._input_keys_order = ["heating_w", "cooling_w"]
        # 0 = Internal (Natural/HVAC), 1 = External (Wind)
        self._convection_types: List[int] = []
        self.add_node("ambient", capacity_j_k=jnp.inf)

    def add_node(self, name: str, capacity_j_k: float):
        if name in self._nodes: raise ValueError(f"Node {name} exists.")
        self._nodes[name] = _RCNode(name, capacity_j_k)

    def add_resistor(
        self, 
        node_a: str, 
        node_b: str, 
        R_k_w: float, 
        dynamic: bool = False, 
        is_external: bool = False # [NEW] Argument
    ):
        """
        Adds a thermal resistance between two nodes.

        Args:
            node_a: Name of first node.
            node_b: Name of second node.
            R_k_w: Thermal resistance (Kelvin/Watt).
            dynamic: If True, this is a non-linear/surrogate convection edge.
                     It will be excluded from the static A-matrix.
            is_external: [NEW] If True (and dynamic=True), this edge uses 
                         wind speed for convection calculations.
        """
        if R_k_w <= 0: 
            raise ValueError("R must be > 0")
        
        if node_a not in self._nodes: raise ValueError(f"Unknown node {node_a}")
        if node_b not in self._nodes: raise ValueError(f"Unknown node {node_b}")

        self._resistors.append(_Resistor(node_a, node_b, R_k_w, dynamic))

        if dynamic:
            # Store the type metadata needed for the ThermalConfig
            # 1 for External (Wind), 0 for Internal (Still/HVAC)
            self._convection_types.append(1 if is_external else 0)

    def add_mixing(self, zone_a_name: str, zone_b_name: str, flow_m3_s: float):
        """
        Adds an airflow connection between two zones.
        Args:
            zone_a_name: Name of first zone node (must be an Air node)
            zone_b_name: Name of second zone node
            flow_m3_s: Volumetric flow rate (m3/s). 
                       Heat Transfer Coeff = Flow * Density * Cp
        """
        if zone_a_name not in self._nodes: raise ValueError(f"Unknown node {zone_a_name}")
        if zone_b_name not in self._nodes: raise ValueError(f"Unknown node {zone_b_name}")
        
        self._mixing_links.append(_MixingConnection(zone_a_name, zone_b_name, flow_m3_s))

    def add_capacitive_chain(self, base_name: str, R_total: float, C_total: float, n_slices: int = 2) -> Tuple[str, str]:
        """
        Creates a finite-difference chain of nodes and resistors to model a high-mass wall.
        Uses a Pi-Network discretization approach.
        
        Args:
            base_name: Prefix for the nodes (e.g. "wall_LivingRoom_South").
            R_total: Total conductive resistance of the layer (K/W).
            C_total: Total thermal capacity of the layer (J/K).
            n_slices: Number of slices to subdivide the material into.
                      n=1 -> 2 nodes (Int, Ext) connected by 1 resistor.
                      n=2 -> 3 nodes (Int, Mid, Ext) connected by 2 resistors.
        
        Returns:
            (node_int_name, node_ext_name): Names of the boundary nodes to connect to convection/radiation.
        """
        if n_slices < 1:
            raise ValueError("n_slices must be >= 1")

        # Per-slice properties
        R_slice = R_total / n_slices
        C_slice = C_total / n_slices

        # Boundary Node Names
        node_int = f"{base_name}_int"
        node_ext = f"{base_name}_ext"

        # 1. Internal Surface Node (Half slice capacity)
        self.add_node(node_int, capacity_j_k=C_slice * 0.5)

        prev_node = node_int

        # 2. Intermediate Nodes (Full slice capacity)
        # If n_slices > 1, we insert (n_slices - 1) nodes in the middle
        for i in range(1, n_slices):
            mid_node = f"{base_name}_mid_{i}"
            self.add_node(mid_node, capacity_j_k=C_slice)
            self.add_resistor(prev_node, mid_node, R_slice)
            prev_node = mid_node

        # 3. External Surface Node (Half slice capacity)
        self.add_node(node_ext, capacity_j_k=C_slice * 0.5)
        # Connect the last resistor
        self.add_resistor(prev_node, node_ext, R_slice)

        return node_int, node_ext
    
    def add_input_mapping(self, input_key: str, node_name: str, room_index: int, fraction: float = 1.0):
        if input_key not in self._input_keys_order: pass 
        if node_name not in self._nodes: raise ValueError(f"Unknown node {node_name}")
        self._mappings.append(_InputMapping(input_key, node_name, room_index, fraction))

    def set_infiltration(self, total_volume_m3: float, k1: float = 0.1, k2: float = 0.0, k3: float = 0.0):
        self._infiltration_enabled = True
        self._total_volume = total_volume_m3
        self._inf_params = (k1, k2, k3)

    def set_waste_heat_node(self, node_name: str):
        if node_name not in self._nodes: raise ValueError(f"Unknown node {node_name}")
        self._waste_heat_node_name = node_name

    def _get_input_col_index(self, key: str, room_idx: int) -> int:
        if key not in self._input_keys_order: return -1
        base_offset = self._input_keys_order.index(key) * self.n_rooms
        return base_offset + room_idx

    def compile(self) -> ThermalConfig:
        # 1. Define Node Order: Ambient -> Ground -> (Sorted Internal Nodes)
        # This ensures consistent indexing for JAX
        node_names_sorted = sorted([n for n in self._nodes if n not in ["ambient", "ground"]])
        final_node_order = ["ambient"]
        if "ground" in self._nodes: final_node_order.append("ground")
        final_node_order.extend(node_names_sorted)

        N_nodes = len(final_node_order)
        
        # Assign indices to nodes
        for i, name in enumerate(final_node_order): 
            self._nodes[name].index = i
        
        node_map = {name: i for i, name in enumerate(final_node_order)}

        # 2. Build Capacity Vector (C)
        # Invert C for state space: dT/dt = C_inv * Q
        c_inv_vector = np.zeros(N_nodes, dtype=np.float32)
        for name, node in self._nodes.items():
            # Handle infinite capacity (Ambient/Ground) by setting 1/C = 0
            if np.isinf(node.capacity_j_k) or node.capacity_j_k > 1e12:
                c_inv_vector[node.index] = 0.0
            else:
                c_inv_vector[node.index] = 1.0 / node.capacity_j_k

        # 3. Build Static Conductance Matrix (A)
        A_matrix = np.zeros((N_nodes, N_nodes), dtype=np.float32)
        
        # Lists for Dynamic (Non-Linear) Components
        conv_pairs = []
        conv_coeffs = []

        for res in self._resistors:
            i = self._nodes[res.node_a_name].index
            j = self._nodes[res.node_b_name].index

            # Conductance G = 1/R
            G = 1.0 / res.R_k_w

            if res.dynamic:
                # Exclude from static A-Matrix.
                # These will be calculated at runtime (e.g., Surface Convection ~ |dT|^0.33)
                conv_pairs.append([i, j])
                conv_coeffs.append(G)
            else:
                # Standard Linear Conduction: Bake into A-Matrix
                # Conservation: Heat leaving i goes to j
                A_matrix[i, i] -= G
                A_matrix[j, j] -= G
                A_matrix[i, j] += G
                A_matrix[j, i] += G

        # Convert dynamic lists to JAX-ready arrays
        if len(conv_pairs) > 0:
            conv_pairs_arr = jnp.array(conv_pairs, dtype=jnp.int32)
            conv_coeffs_arr = jnp.array(conv_coeffs, dtype=jnp.float32)
            # [NEW] Pack types
            conv_types_arr = jnp.array(self._convection_types, dtype=jnp.int32)
        else:
            conv_pairs_arr = jnp.zeros((0, 2), dtype=jnp.int32)
            conv_coeffs_arr = jnp.zeros((0,), dtype=jnp.float32)
            conv_types_arr = jnp.zeros((0,), dtype=jnp.int32)

        # 4. Build Input Matrix (B)
        # Inputs are: Heating, Cooling, Solar (if applicable), Internal Gains
        N_inputs_flat = len(self._input_keys_order) * self.n_rooms
        B_matrix = np.zeros((N_nodes, N_inputs_flat), dtype=np.float32)
        input_map_metadata = {k: {} for k in self._input_keys_order}

        for m in self._mappings:
            if m.input_key in self._input_keys_order:
                r_idx = self._nodes[m.node_name].index
                c_idx = self._get_input_col_index(m.input_key, m.room_index)
                
                # Add fraction to B-Matrix
                B_matrix[r_idx, c_idx] += m.fraction
                input_map_metadata[m.input_key][m.node_name] = c_idx

        # Pre-calculate indices for HVAC input vectors
        u_idx_heating = []
        u_idx_cooling = []
        for i in range(self.n_rooms):
            u_idx_heating.append(self._get_input_col_index("heating_w", i))
            u_idx_cooling.append(self._get_input_col_index("cooling_w", i))

        # 5. Topology Helpers
        def find_indices(prefix):
            return tuple(sorted([n.index for name, n in self._nodes.items() if name.startswith(prefix)]))

        # Build Air-to-Radiant mapping (Deprecated in full mesh mode, but kept for backward compat)
        # In Full Mesh, we don't strictly have a "room_rad_X" node, but we might for symmetry.
        air_to_rad = np.full(N_nodes, -1, dtype=np.int32)
        for name, node in self._nodes.items():
            if name.startswith("room_air_"):
                suffix = name.replace("room_air_", "")
                # Try to find a representative radiative node if it exists
                rad_name = f"room_rad_{suffix}" 
                if rad_name in self._nodes:
                    rad_idx = self._nodes[rad_name].index
                    air_to_rad[node.index] = rad_idx

        # Indices for specific special nodes
        waste_idx = self._nodes[self._waste_heat_node_name].index if self._waste_heat_node_name else -1
        ground_idx = self._nodes["ground"].index if "ground" in self._nodes else -1

        # 6. Mixing Logic (Inter-zone airflow)
        mix_pairs = []
        mix_cond = []
        RHO_AIR = 1.204
        CP_AIR = 1005.0

        for link in self._mixing_links:
            idx_a = self._nodes[link.node_a_name].index
            idx_b = self._nodes[link.node_b_name].index
            
            # G = m_dot * Cp = (VolFlow * Rho) * Cp
            G_mix = link.flow_rate_m3_s * RHO_AIR * CP_AIR
            
            mix_pairs.append([idx_a, idx_b])
            mix_cond.append(G_mix)

        if len(mix_pairs) > 0:
            mix_pairs_arr = jnp.array(mix_pairs, dtype=jnp.int32)
            mix_cond_arr = jnp.array(mix_cond, dtype=jnp.float32)
        else:
            mix_pairs_arr = jnp.zeros((0, 2), dtype=jnp.int32)
            mix_cond_arr = jnp.zeros((0,), dtype=jnp.float32)

        # 7. Infiltration Physics Parameters (Orifice Flow)
        # We map the generic params to the new Physics fields
        # Leakage Area approx: if not set, assume tight building (0.0005 * Vol)
        if self._infiltration_enabled:
            # Orifice Flow parameters
            # k1 stores leakage area, k2 stores stack coeff, k3 stores wind coeff
            leakage_area = self._inf_params[0]
            stack_c = self._inf_params[1] if self._inf_params[1] > 0 else 0.12
            wind_c = self._inf_params[2] if self._inf_params[2] > 0 else 0.09
        else:
            leakage_area = 0.0
            stack_c = 0.0
            wind_c = 0.0

        # 8. Final Assembly
        return ThermalConfig(
            # Matrices
            A_matrix=jnp.array(A_matrix),
            C_inv_vector=jnp.array(c_inv_vector),
            B_matrix=jnp.array(B_matrix),

            # Dynamic Physics Arrays
            convection_pairs=conv_pairs_arr,
            convection_coefficients=conv_coeffs_arr,
            convection_types=conv_types_arr,

            # Metadata
            node_names=final_node_order,
            node_map=node_map,
            input_map=input_map_metadata,

            # Input Indices
            u_idx_heating=jnp.array(u_idx_heating, dtype=jnp.int32),
            u_idx_cooling=jnp.array(u_idx_cooling, dtype=jnp.int32),

            # Topology Map
            air_to_rad_map=jnp.array(air_to_rad, dtype=jnp.int32),

            # Specific Node Indices
            ambient_air_index=self._nodes["ambient"].index,
            ground_node_index=ground_idx,
            room_air_indices=find_indices("room_air_"),
            room_rad_indices=find_indices("room_rad_"),
            wall_indices=find_indices("wall_"),
            mass_indices=find_indices("mass_"),
            waste_heat_node_index=waste_idx,

            # Infiltration Params (Updated for Orifice Flow)
            use_dynamic_infiltration=self._infiltration_enabled,
            room_vol_m3=self._total_volume,
            leakage_area_m2=leakage_area,
            stack_coeff=stack_c,
            wind_coeff=wind_c,

            # Mixing Params
            mixing_pairs=mix_pairs_arr,
            mixing_conductance=mix_cond_arr,
        )