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
    # --- Radiation Physics Metadata ---
    area_m2: float = 0.0
    emissivity: float = 0.9
    # Critical for High-Fidelity: Which room does this surface belong to?
    # -1 = External/Ambient (No internal radiation)
    # 0, 1, 2... = Specific Enclosure indices
    enclosure_id: int = -1 

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
        
        # Infiltration / Airflow
        self._infiltration_enabled = False
        self._inf_params = (0.1, 0.0, 0.0)
        self._total_volume = 0.0
        
        # Metadata
        self._waste_heat_node_name = None
        self._input_keys_order = ["heating_w", "cooling_w"]
        self._convection_types: List[int] = []
        
        # Initialize Ambient (Infinite Capacity, Enclosure -1)
        self.add_node("ambient", capacity_j_k=jnp.inf, enclosure_id=-1)

    def add_node(self, name: str, capacity_j_k: float, area_m2: float = 0.0, emissivity: float = 0.9, enclosure_id: int = -1):
        """
        Adds a thermal node to the network.
        
        Args:
            enclosure_id: ID of the room/zone this surface faces. 
                          Surfaces with the same enclosure_id will radiate to each other.
                          Use -1 for internal material nodes or external surfaces.
        """
        if name in self._nodes: raise ValueError(f"Node {name} exists.")
        self._nodes[name] = _RCNode(
            name, 
            capacity_j_k, 
            area_m2=area_m2, 
            emissivity=emissivity, 
            enclosure_id=enclosure_id
        )

    def add_resistor(self, node_a: str, node_b: str, R_k_w: float, dynamic: bool = False, is_external: bool = False):
        """Adds a thermal resistance between two nodes."""
        if R_k_w <= 0: raise ValueError("R must be > 0")
        if node_a not in self._nodes: raise ValueError(f"Unknown node {node_a}")
        if node_b not in self._nodes: raise ValueError(f"Unknown node {node_b}")
        
        self._resistors.append(_Resistor(node_a, node_b, R_k_w, dynamic))
        
        if dynamic:
            # Metadata for the model to know which convection correlation to use
            # 1 = External (Wind-driven), 0 = Internal (Natural/HVAC)
            self._convection_types.append(1 if is_external else 0)

    def add_mixing(self, zone_a_name: str, zone_b_name: str, flow_m3_s: float):
        """Adds an airflow connection (mixing) between two zones."""
        if zone_a_name not in self._nodes: raise ValueError(f"Unknown node {zone_a_name}")
        if zone_b_name not in self._nodes: raise ValueError(f"Unknown node {zone_b_name}")
        self._mixing_links.append(_MixingConnection(zone_a_name, zone_b_name, flow_m3_s))

    def add_capacitive_chain(self, base_name: str, R_total: float, C_total: float, n_slices: int = 2, enclosure_id: int = -1) -> Tuple[str, str]:
        """
        Creates a finite-difference chain (Pi-Network) for a wall.
        """
        if n_slices < 1: raise ValueError("n_slices must be >= 1")

        R_slice = R_total / n_slices
        C_slice = C_total / n_slices

        node_int = f"{base_name}_int"
        node_ext = f"{base_name}_ext"

        # 1. Internal Surface (Faces the room -> participates in Radiation)
        self.add_node(node_int, capacity_j_k=C_slice * 0.5, enclosure_id=enclosure_id)

        prev_node = node_int

        # 2. Core Nodes (Inside the wall -> No radiation)
        for i in range(1, n_slices):
            mid_node = f"{base_name}_mid_{i}"
            self.add_node(mid_node, capacity_j_k=C_slice, enclosure_id=-1)
            self.add_resistor(prev_node, mid_node, R_slice)
            prev_node = mid_node

        # 3. External Surface (Faces ambient -> No *Internal* radiation)
        self.add_node(node_ext, capacity_j_k=C_slice * 0.5, enclosure_id=-1)
        self.add_resistor(prev_node, node_ext, R_slice)

        return node_int, node_ext

    def add_input_mapping(self, input_key: str, node_name: str, room_index: int, fraction: float = 1.0):
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
        
    def _compute_radiation_matrix_vectorized(self, surface_indices: List[int]) -> jnp.ndarray:
        """
        Pre-computes the Radiative Conductance Matrix G_ij using Vectorized JAX/Numpy.
        Replaces nested Python loops for O(N^2) efficiency and differentiability.
        
        G_ij = Sigma * Area_i * F_ij * epsilon_eff_ij
        """
        STEFAN_BOLTZMANN = 5.670374419e-8
        N = len(surface_indices)
        
        if N < 2:
            return jnp.zeros((N, N), dtype=jnp.float32)

        # 1. Gather Properties (Flattened)
        areas = []
        emissivities = []
        enclosure_ids = []
        
        idx_to_node = {node.index: node for node in self._nodes.values()}
        sorted_indices = sorted(surface_indices)
        
        for idx in sorted_indices:
            node = idx_to_node[idx]
            areas.append(node.area_m2)
            emissivities.append(node.emissivity)
            enclosure_ids.append(node.enclosure_id)
            
        A = np.array(areas, dtype=np.float32)      # (N,)
        E = np.array(emissivities, dtype=np.float32) # (N,)
        IDs = np.array(enclosure_ids, dtype=np.int32) # (N,)
        
        # 2. Create Topology Mask (Block Diagonal Construction)
        # Mask[i, j] = 1 if (ID_i == ID_j) AND (ID_i != -1), else 0
        # This ensures surfaces only talk to others in the same room.
        id_col = IDs[:, None] # (N, 1)
        id_row = IDs[None, :] # (1, N)
        
        # Surfaces see each other if IDs match and aren't "External" (-1)
        topology_mask = (id_col == id_row) & (id_col != -1)
        
        # Remove diagonal (Self-Shading) immediately
        np.fill_diagonal(topology_mask, False)
        topology_mask = topology_mask.astype(np.float32)

        # 3. Compute Total Area per Enclosure (Vectorized)
        # A_total[i] = Sum of areas of all j visible to i
        # This handles the "A_total" in Carroll's method specifically for each room block.
        A_row = A[None, :] # (1, N)
        # Mask * Area sums up areas only for valid connections
        A_total_per_node = np.sum(topology_mask * A_row, axis=1) # (N,)
        # Add own area back because A_total includes self in Carroll's denominator
        A_total_per_node += A 

        # 4. View Factors (F_ij) - Carroll's Method
        # F_ij = A_j / (A_total_enclosure - A_i)
        
        numerator = np.tile(A, (N, 1)) # (N, N) -> A_j
        denominator = A_total_per_node[:, None] - A[:, None] # (N, 1) -> A_tot - A_i
        
        # Avoid div/0
        F_raw = np.divide(numerator, denominator, where=denominator > 1e-9)
        
        # Apply Topology Mask (zeros out non-room interactions)
        F_matrix = F_raw * topology_mask
        
        # Normalize Rows (Energy Conservation)
        # Ensure sum(F_i->j) <= 1.0
        row_sums = np.sum(F_matrix, axis=1, keepdims=True)
        F_matrix = np.divide(F_matrix, row_sums, where=row_sums > 1e-9)
        
        # 5. Grey Body Effective Emissivity
        # eps_eff = 1 / (1/e_i + 1/e_j - 1)
        inv_e = 1.0 / (E + 1e-6)
        denom_e = inv_e[:, None] + inv_e[None, :] - 1.0
        eps_eff_matrix = 1.0 / denom_e
        
        # 6. Final Conductance Matrix
        # G_ij = Sigma * A_i * F_ij * eps_eff_ij
        G_matrix = STEFAN_BOLTZMANN * A[:, None] * F_matrix * eps_eff_matrix
        
        return jnp.array(G_matrix, dtype=jnp.float32)

    def compile(self) -> ThermalConfig:
        # 1. Node Ordering
        node_names_sorted = sorted([n for n in self._nodes if n not in ["ambient", "ground"]])
        final_node_order = ["ambient"]
        if "ground" in self._nodes: final_node_order.append("ground")
        final_node_order.extend(node_names_sorted)
        
        N_nodes = len(final_node_order)
        for i, name in enumerate(final_node_order):
            self._nodes[name].index = i
            
        node_map = {name: i for i, name in enumerate(final_node_order)}
        
        # 2. Capacities
        c_inv_vector = np.zeros(N_nodes, dtype=np.float32)
        for name, node in self._nodes.items():
            if np.isinf(node.capacity_j_k) or node.capacity_j_k > 1e12:
                c_inv_vector[node.index] = 0.0
            else:
                c_inv_vector[node.index] = 1.0 / node.capacity_j_k
                
        # 3. Static Conduction Matrix (A)
        A_matrix = np.zeros((N_nodes, N_nodes), dtype=np.float32)
        conv_pairs = []
        conv_coeffs = []
        
        for res in self._resistors:
            i = self._nodes[res.node_a_name].index
            j = self._nodes[res.node_b_name].index
            G = 1.0 / res.R_k_w
            
            if res.dynamic:
                conv_pairs.append([i, j])
                conv_coeffs.append(G)
            else:
                A_matrix[i, i] -= G
                A_matrix[j, j] -= G
                A_matrix[i, j] += G
                A_matrix[j, i] += G
                
        if len(conv_pairs) > 0:
            conv_pairs_arr = jnp.array(conv_pairs, dtype=jnp.int32)
            conv_coeffs_arr = jnp.array(conv_coeffs, dtype=jnp.float32)
            conv_types_arr = jnp.array(self._convection_types, dtype=jnp.int32)
        else:
            conv_pairs_arr = jnp.zeros((0, 2), dtype=jnp.int32)
            conv_coeffs_arr = jnp.zeros((0,), dtype=jnp.float32)
            conv_types_arr = jnp.zeros((0,), dtype=jnp.int32)
            
        # 4. Input Matrix (B)
        N_inputs_flat = len(self._input_keys_order) * self.n_rooms
        B_matrix = np.zeros((N_nodes, N_inputs_flat), dtype=np.float32)
        input_map_metadata = {k: {} for k in self._input_keys_order}

        for m in self._mappings:
            if m.input_key in self._input_keys_order:
                r_idx = self._nodes[m.node_name].index
                c_idx = self._get_input_col_index(m.input_key, m.room_index)
                B_matrix[r_idx, c_idx] += m.fraction
                input_map_metadata[m.input_key][m.node_name] = c_idx

        # 5. Radiation Matrix (Fully Vectorized & Block Diagonal)
        surface_indices = []
        for name, node in self._nodes.items():
            # Valid surface if area > 0 and it belongs to a specific enclosure (not -1)
            if node.area_m2 > 1e-6 and node.enclosure_id != -1:
                surface_indices.append(node.index)
        
        surface_indices = sorted(surface_indices)
        # This now calls the vectorized version
        rad_G_matrix = self._compute_radiation_matrix_vectorized(surface_indices)
        surface_indices_arr = jnp.array(surface_indices, dtype=jnp.int32)

        # 6. Mixing & Helpers
        mix_pairs = []
        mix_cond = []
        RHO_AIR, CP_AIR = 1.204, 1005.0
        for link in self._mixing_links:
            mix_pairs.append([self._nodes[link.node_a_name].index, self._nodes[link.node_b_name].index])
            mix_cond.append(link.flow_rate_m3_s * RHO_AIR * CP_AIR)
            
        if len(mix_pairs) > 0:
            mix_pairs_arr = jnp.array(mix_pairs, dtype=jnp.int32)
            mix_cond_arr = jnp.array(mix_cond, dtype=jnp.float32)
        else:
            mix_pairs_arr = jnp.zeros((0, 2), dtype=jnp.int32)
            mix_cond_arr = jnp.zeros((0,), dtype=jnp.float32)

        u_idx_heating = [self._get_input_col_index("heating_w", i) for i in range(self.n_rooms)]
        u_idx_cooling = [self._get_input_col_index("cooling_w", i) for i in range(self.n_rooms)]

        # Topology Helpers
        def find_indices(prefix):
            return tuple(sorted([n.index for name, n in self._nodes.items() if name.startswith(prefix)]))

        air_to_rad = np.full(N_nodes, -1, dtype=np.int32)
        for name, node in self._nodes.items():
            if name.startswith("room_air_"):
                suffix = name.replace("room_air_", "")
                rad_name = f"room_rad_{suffix}"
                if rad_name in self._nodes:
                    air_to_rad[node.index] = self._nodes[rad_name].index

        waste_idx = self._nodes[self._waste_heat_node_name].index if self._waste_heat_node_name else -1
        ground_idx = self._nodes["ground"].index if "ground" in self._nodes else -1

        # 7. Infiltration
        if self._infiltration_enabled:
            leakage_area = self._inf_params[0]
            stack_c = self._inf_params[1] if self._inf_params[1] > 0 else 0.12
            wind_c = self._inf_params[2] if self._inf_params[2] > 0 else 0.09
        else:
            leakage_area = 0.0
            stack_c = 0.0
            wind_c = 0.0

        # 8. Return Config
        return ThermalConfig(
            A_matrix=jnp.array(A_matrix),
            C_inv_vector=jnp.array(c_inv_vector),
            B_matrix=jnp.array(B_matrix),
            
            convection_pairs=conv_pairs_arr,
            convection_coefficients=conv_coeffs_arr,
            convection_types=conv_types_arr,
            
            mixing_pairs=mix_pairs_arr,
            mixing_conductance=mix_cond_arr,
            
            # High-Fidelity Zone-Aware Radiation
            surface_node_indices=surface_indices_arr,
            rad_conductance_matrix=rad_G_matrix,
            
            node_names=final_node_order,
            node_map=node_map,
            input_map=input_map_metadata,
            
            u_idx_heating=jnp.array(u_idx_heating, dtype=jnp.int32),
            u_idx_cooling=jnp.array(u_idx_cooling, dtype=jnp.int32),
            air_to_rad_map=jnp.array(air_to_rad, dtype=jnp.int32),
            
            ambient_air_index=self._nodes["ambient"].index,
            ground_node_index=ground_idx,
            room_air_indices=find_indices("room_air_"),
            room_rad_indices=find_indices("room_rad_"),
            wall_indices=find_indices("wall_"),
            mass_indices=find_indices("mass_"),
            waste_heat_node_index=waste_idx,

            use_dynamic_infiltration=self._infiltration_enabled,
            room_vol_m3=self._total_volume,
            leakage_area_m2=leakage_area,
            stack_coeff=stack_c,
            wind_coeff=wind_c,
        )