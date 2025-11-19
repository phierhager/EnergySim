import jax.numpy as jnp
from typing import List, Dict, Tuple, Any, Optional
import dataclasses

# Internal library imports
from energysim.core.network_builder import RCNetworkBuilder
from energysim.core.shared.data_structs import Material, Surface, WindowType
from energysim.core.physics.coefficients import PhysicsConfig, SurfaceRoughness
from energysim.core.models.thermal_model import ThermalConfig

class BuildingCompiler:
    """
    Parses building geometry and material definitions into a JAX-compatible
    RC-Network Graph.
    
    Produces:
    1. ThermalConfig (A, B matrices)
    2. Dynamic Surfaces (for Wind)
    3. Solar Surfaces (for Geometric Solar Engine)
    """

    def __init__(self, lat_lon: Tuple[float, float], physics_config: Optional[PhysicsConfig] = None):
        self.lat_lon = lat_lon
        self.phys = physics_config if physics_config else PhysicsConfig()
        
        # Registry
        self.materials: Dict[str, Material] = {}
        self.window_types: Dict[str, WindowType] = {}
        self.zones: List[Dict[str, Any]] = []
        self.surfaces: List[Surface] = []
        
        # Output Data
        self.dynamic_surfaces_data = [] 
        self.solar_surfaces_data = []

    def add_material(self, mat: Material):
        self.materials[mat.name] = mat

    def add_window_type(self, win: WindowType):
        self.window_types[win.name] = win

    def add_zone(self, name: str, volume: float, capacitance_multiplier: float = 1.2):
        """
        Adds a thermal zone.
        :param capacitance_multiplier: Account for furniture/internal mass (default 1.2x air mass).
        """
        # C = V * rho * cp * multiplier
        c_air = volume * self.phys.air_density_kg_m3 * self.phys.air_heat_capacity_j_kgk * capacitance_multiplier
        self.zones.append({"name": name, "C": c_air, "volume": volume})

    def add_surface(self, surf: Surface):
        self.surfaces.append(surf)

    def compile(self) -> Tuple[ThermalConfig, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Compiles the physical definitions into matrices.
        """
        n_zones = len(self.zones)
        builder = RCNetworkBuilder(n_rooms=n_zones)

        # Map zone names to indices for B-matrix mapping
        zone_idx_map = {z["name"]: i for i, z in enumerate(self.zones)}

        # 1. Nodes
        # Ground Coupling - Now tracked as a specific input node index in builder if needed,
        # or we treat "ground" string as a boundary condition handled by PhysicsConfig.
        # For RCBuilder, "ambient" is index 0. We add "ground" as a distinct boundary.
        builder.add_node("ground", capacity_j_k=jnp.inf) 
        
        # Create Zone Air Nodes
        for i, z in enumerate(self.zones):
            canonical_name = f"room_air_{i}"
            builder.add_node(canonical_name, capacity_j_k=z["C"])
            builder.add_input_mapping("heating_w", canonical_name, room_index=i)
            builder.add_input_mapping("cooling_w", canonical_name, room_index=i)

        # 2. Process Surfaces
        for surf in self.surfaces:
            zone_idx = zone_idx_map[surf.zone_name]
            zone_node_name = f"room_air_{zone_idx}"

            # --- A. WINDOWS ---
            if surf.type == 'WINDOW':
                if surf.construction_name not in self.window_types:
                    raise ValueError(f"Window Type '{surf.construction_name}' not found.")

                w_type = self.window_types[surf.construction_name]
                w_type.calculate_properties()

                # --- OPTIMIZATION: Negligible Mass Check ---
                # Glass capacity is usually small (~20-50 kJ/K). If small relative to air (~5000 kJ/K),
                # it makes the differential equations stiff. 
                # Threshold: 50 kJ/K (approx 20m2 of glass)
                C_glass = w_type.glass_thickness * w_type.density * w_type.specific_heat * surf.area
                
                if C_glass < 50000.0: 
                    # Optimization: Steady State approximation
                    # Collapse Int-Glass-Ext into one resistor
                    h_int = 8.3
                    h_ext_nat = 25.0
                    
                    R_int = 1.0 / (h_int * surf.area)
                    R_ext_nat = 1.0 / (h_ext_nat * surf.area)
                    R_glazing = w_type.R_glazing_layer / surf.area
                    
                    R_total = R_int + R_glazing + R_ext_nat
                    
                    # Connect Room directly to Ambient
                    # Note: We lose dynamic wind cooling on glass specifically, but for simplified models this is fine.
                    builder.add_resistor(zone_node_name, "ambient", R_total)
                    
                    # Solar logic needs to know where to dump absorbed heat.
                    # If no glass node, we dump absorbed heat 50/50 to Room and Ambient or 100 to Room.
                    # Let's dump to Room for worst-case cooling load.
                    target_abs_node = zone_node_name 
                    target_trans_node = zone_node_name

                else:
                    # Standard Dynamic Model
                    node_glass = f"win_{surf.name}_glass"
                    builder.add_node(node_glass, capacity_j_k=C_glass)
                    
                    h_int = 8.3
                    R_int = 1.0 / (h_int * surf.area)
                    R_total_path_in = R_int + (w_type.R_glazing_layer / surf.area)
                    h_ext_nat = 25.0
                    R_ext_nat = 1.0 / (h_ext_nat * surf.area)

                    builder.add_resistor(node_glass, zone_node_name, R_total_path_in)
                    builder.add_resistor(node_glass, "ambient", R_ext_nat)
                    
                    target_abs_node = node_glass
                    target_trans_node = zone_node_name

                    # Register for Dynamic Wind only if node exists
                    self.dynamic_surfaces_data.append({
                        "node_name": node_glass,
                        "area": surf.area,
                        "roughness_mult": SurfaceRoughness.SMOOTH.get_multiplier(),
                        "boundary": "ambient"
                    })

                # Register for Geometric Solar
                self.solar_surfaces_data.append({
                    "type": "WINDOW",
                    "name": surf.name,
                    "normal": surf.normal,
                    "area": surf.area,
                    "shgc": w_type.shgc,
                    "target_trans_node": target_trans_node,
                    "target_abs_node": target_abs_node
                })

            # --- B. OPAQUE SURFACES ---
            else:
                if surf.construction_name not in self.materials:
                    raise ValueError(f"Material '{surf.construction_name}' not found.")

                mat = self.materials[surf.construction_name]
                node_int = f"wall_{surf.name}_int"
                node_ext = f"wall_{surf.name}_ext"

                # Physics
                R_cond = mat.thickness / (mat.conductivity * surf.area)
                C_tot = mat.thickness * mat.density * mat.specific_heat * surf.area

                h_int = self.phys.get_internal_convection(surf.tilt)
                R_conv_int = 1.0 / (h_int * surf.area)

                # Nodes
                builder.add_node(node_int, capacity_j_k=C_tot * 0.5)
                builder.add_node(node_ext, capacity_j_k=C_tot * 0.5)

                # Connect Internal
                builder.add_resistor(zone_node_name, node_int, R_conv_int)
                builder.add_resistor(node_int, node_ext, R_cond)

                # Connect Boundary
                if surf.boundary_condition == 'AMBIENT':
                    h_ext_nat = 25.0
                    R_nat = 1.0 / (h_ext_nat * surf.area)
                    builder.add_resistor(node_ext, "ambient", R_nat)

                    self.dynamic_surfaces_data.append({
                        "node_name": node_ext,
                        "area": surf.area,
                        "roughness_mult": SurfaceRoughness(surf.roughness).get_multiplier(),
                        "boundary": "ambient"
                    })
                    
                    self.solar_surfaces_data.append({
                        "type": "WALL",
                        "name": surf.name,
                        "normal": surf.normal,
                        "area": surf.area,
                        "absorptivity": mat.absorptivity,
                        "target_node": node_ext
                    })

                elif surf.boundary_condition == 'GROUND':
                    R_ground = 5.0 / surf.area
                    builder.add_resistor(node_ext, "ground", R_ground)

                elif surf.boundary_condition.startswith("ZONE:"):
                    neighbor_name_usr = surf.boundary_condition.split(":")[1]
                    neighbor_idx = zone_idx_map[neighbor_name_usr]
                    neighbor_node = f"room_air_{neighbor_idx}"
                    builder.add_resistor(node_ext, neighbor_node, R_conv_int)

        # 3. Compile
        config = builder.compile()
        return config, self.dynamic_surfaces_data, self.solar_surfaces_data