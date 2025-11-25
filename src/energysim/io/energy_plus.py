import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import asdict
import warnings

# Check for eppy
try:
    from eppy.modeleditor import IDF
except ImportError:
    raise ImportError("The 'eppy' library is required for EnergyPlusBridge. Install it via `pip install eppy`.")

# Internal imports
from energysim.core.shared.data_structs import (
    GeometryConfig, Material, Surface, WindowType, 
    ApplianceConfig, OccupantConfig, 
    ThermalConfig, AirflowConfig
)
from energysim.core.physics.constants import Coefficients
from energysim.core.network_builder import RCNetworkBuilder
from energysim.utils.geometry import get_polygon_normal, get_polygon_area_3d, get_azimuth_tilt
from energysim.core.physics.thermo import get_internal_convection

import equinox as eqx





class EnergyPlusBridge:
    """
    A high-fidelity bridge to import EnergyPlus (.idf) models into EnergySim (JAX).
    """

    def __init__(
        self, 
        idf_path: str, 
        idd_path: str, 
        # NEW ARGUMENT: Dictionary mapping substrings to kWh cycle energy
        # Example: {"DISHWASHER": 1.2, "EV_CHARGER": 40.0}
        smart_appliance_specs: Dict[str, float] = None 
    ):
        IDF.setiddname(idd_path)
        self.idf = IDF(idf_path)
        
        # Store the user specs, defaulting to empty if None
        self.smart_specs = smart_appliance_specs if smart_appliance_specs else {}

        # Registries
        self.materials: Dict[str, Material] = {}
        self.window_types: Dict[str, WindowType] = {}
        self.constructions: Dict[str, Any] = {}

        self.zones: Dict[str, Dict] = {}
        self.zone_lists: Dict[str, List[str]] = {}
        self.surfaces: List[Surface] = []

        self.thermal_config: Optional[ThermalConfig] = None
        self.dynamic_surfaces: List[Dict] = []
        self.solar_surfaces: List[Dict] = []
        self.machine_configs: List[ApplianceConfig] = []
        self.occupant_configs: List[OccupantConfig] = []
        self.schedules: Dict[str, np.ndarray] = {}

        self.coord_sys = "Relative"
        self._zone_to_node_map: Dict[str, str] = {}

    # --- HELPER THAT WAS MISSING ---
    def _parse_float(self, val) -> Optional[float]:
        """Helper to parse raw list items that might be strings or empty."""
        if val is None or str(val).strip() == "":
            return None
        try:
            return float(val)
        except:
            return None

    def _get_strict_float(self, ep_object, attr_name: str) -> Optional[float]:
        """Strictly gets a float attribute if it exists and is not empty."""
        if hasattr(ep_object, attr_name):
            val = getattr(ep_object, attr_name)
            if val is not None and str(val).strip() != "":
                try:
                    return float(val)
                except:
                    pass
        return None

    def _resolve_power_instances(self, ep_object) -> List[Tuple[str, float]]:
        # 1. Identify Target Node & Area
        target_ref = ep_object.obj[2].upper()
        target_nodes = []

        if target_ref in self._zone_to_node_map:
            node = self._zone_to_node_map[target_ref]
            area = self.zones[target_ref]['floor_area']
            target_nodes.append((node, area))
        elif target_ref in self.zone_lists:
            for z_child in self.zone_lists[target_ref]:
                if z_child in self._zone_to_node_map:
                    node = self._zone_to_node_map[z_child]
                    area = self.zones[z_child]['floor_area']
                    target_nodes.append((node, area))
        else:
            return []

        # 2. Robust Value Parsing (The fix from previous steps)
        raw_list = ep_object.obj
        method = str(raw_list[4]).upper() if len(raw_list) > 4 else ""

        # Logic for LIGHTS and ELECTRICEQUIPMENT
        if ep_object.key.upper() in ["LIGHTS", "ELECTRICEQUIPMENT"]:
            if "AREA" in method: 
                if len(raw_list) > 6:
                    val = self._parse_float(raw_list[6])
                    if val is not None:
                        return [(n, val * area) for n, area in target_nodes]
            elif "LEVEL" in method or "WATTS" in method:
                if len(raw_list) > 5:
                    val = self._parse_float(raw_list[5])
                    if val is not None:
                        return [(n, val) for n, _ in target_nodes]

        # Logic for PEOPLE
        elif ep_object.key.upper() == "PEOPLE":
            if "PEOPLE" in method and "AREA" not in method:
                if len(raw_list) > 5:
                    val = self._parse_float(raw_list[5])
                    if val is not None:
                        return [(n, val * 120.0) for n, _ in target_nodes]
            elif "AREA" in method:
                if len(raw_list) > 6:
                    val = self._parse_float(raw_list[6])
                    if val is not None:
                        return [(n, val * area * 120.0) for n, area in target_nodes]

        # Fallback
        val_abs = self._get_strict_float(ep_object, 'Design_Level') or \
                  self._get_strict_float(ep_object, 'Lighting_Level')
        if val_abs: return [(n, val_abs) for n, _ in target_nodes]

        val_area = self._get_strict_float(ep_object, 'Watts_per_Zone_Floor_Area')
        if val_area: return [(n, val_area * area) for n, area in target_nodes]

        print(f"  [Zero Warning] {ep_object.Name}: No valid power definition found.")
        return []

    def _parse_loads(self):
        print("--- Parsing Loads ---")

        # 1. Parse People
        for p in self.idf.idfobjects["PEOPLE"]:
            instances = self._resolve_power_instances(p)
            for node, val in instances:
                self.occupant_configs.append(OccupantConfig(
                    name=f"{p.Name}_{node}", 
                    target_node_name=node, 
                    nominal_heat_w=val
                ))

        # 2. Parse Electric Equipment (Logic for Smart vs Passive)
        for eq in self.idf.idfobjects["ELECTRICEQUIPMENT"]:
            instances = self._resolve_power_instances(eq)
            for node, val in instances:
                if val > 0:
                    
                    # --- CHECK FOR SMART APPLIANCE MATCH ---
                    cycle_kwh = 0.0
                    # Iterate over user-provided dictionary
                    for smart_key, smart_val in self.smart_specs.items():
                        if smart_key.upper() in eq.Name.upper():
                            cycle_kwh = smart_val
                            print(f"  [Smart Config] Found {smart_key}: {eq.Name} -> {cycle_kwh} kWh")
                            break
                    # ---------------------------------------

                    self.machine_configs.append(ApplianceConfig(
                        name=f"{eq.Name}_{node}", 
                        target_node_name=node,
                        nominal_power_w=val, 
                        convective_fraction=0.5,
                        cycle_energy_kwh=cycle_kwh  # 0.0 = Passive, >0 = Smart
                    ))

        # 3. Parse Lights (Always Passive)
        for lg in self.idf.idfobjects["LIGHTS"]:
            instances = self._resolve_power_instances(lg)
            for node, val in instances:
                if val > 0:
                    self.machine_configs.append(ApplianceConfig(
                        name=f"{lg.Name}_{node}", 
                        target_node_name=node,
                        nominal_power_w=val, 
                        convective_fraction=1.0,
                        cycle_energy_kwh=0.0 
                    ))
    
    # =========================================================================
    # PROFESSIONAL FEATURE 1: Geometry & Triangulation
    # =========================================================================

    def _triangulate_polygon(self, coords: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decomposes an N-sided 3D polygon (coplanar) into N-2 triangles using a fan method.
        Robust enough for convex EnergyPlus surfaces.
        """
        if len(coords) < 3: return []
        v0 = coords[0]
        triangles = []
        for i in range(1, len(coords) - 1):
            v1 = coords[i]
            v2 = coords[i+1]
            triangles.append((v0, v1, v2))
        return triangles

    def _compile_geometry(self) -> GeometryConfig:
        """
        Extracts all 'Shading:*' objects and building surfaces to build a collision mesh.
        """
        print("--- Compiling High-Fidelity Geometry ---")
        obs_v0, obs_v1, obs_v2 = [], [], []
        
        # 1. Collect Obstructions (Shading:Site, Shading:Building, Shading:Overhang)
        shading_objs = (self.idf.idfobjects["SHADING:SITE:DETAILED"] + 
                        self.idf.idfobjects["SHADING:BUILDING:DETAILED"] +
                        self.idf.idfobjects["SHADING:OVERHANG"]) # Overhang requires specialized parsing logic not shown for brevity
        
        for obj in shading_objs:
            # Simplified: Assuming DETAILED objects with coords
            if hasattr(obj, 'coords'):
                # Convert eppy coords to numpy
                coords = [np.array(c) for c in obj.coords]
                tris = self._triangulate_polygon(coords)
                for v0, v1, v2 in tris:
                    obs_v0.append(v0); obs_v1.append(v1); obs_v2.append(v2)

        # 2. Collect Receiving Surfaces (Windows only? Or Walls too?)
        # Usually we calculate shading for Fenestration surfaces primarily.
        target_centroids = []
        map_indices = []
        
        # We iterate strictly in the order they appear in the ThermalConfig lists
        # This ensures index alignment.
        # (Assuming self.solar_surfaces is populated as per previous implementation)
        for idx, s_dict in enumerate(self.solar_surfaces):
            # Re-calculate centroid from stored normal/area? 
            # Better to store centroids during the parse phase.
            # Retrieving cached centroid:
            c = s_dict.get('centroid', np.array([0.0, 0.0, 0.0]))
            target_centroids.append(c)
            map_indices.append(idx)

        if not obs_v0:
            # If no shading, return dummy config
            return None

        return GeometryConfig(
            obs_v0=jnp.array(np.stack(obs_v0)),
            obs_v1=jnp.array(np.stack(obs_v1)),
            obs_v2=jnp.array(np.stack(obs_v2)),
            surface_centroids=jnp.array(np.stack(target_centroids)),
            shading_map_indices=jnp.array(map_indices)
        )

    # =========================================================================
    # PROFESSIONAL FEATURE 2: Airflow Network Synthesis
    # =========================================================================

    def _synthesize_airflow_network(self, n_internal_nodes: int) -> AirflowConfig:
        """
        Generates a crack-flow model if explicit AirflowNetwork is missing.
        Reference: ASHRAE Fundamentals - Air Leakage.
        """
        print("--- Synthesizing Airflow Network (Crack Model) ---")
        
        links_a = []
        links_b = []
        links_C = []
        links_n = []
        boundary_indices = []
        boundary_cp = []
        node_heights = np.zeros(n_internal_nodes + 1) # Last one is Ambient
        
        # Map zone names to indices
        z_map = {name: i for i, name in enumerate(sorted(self.zones.keys()))}
        ambient_idx = n_internal_nodes # The virtual external node
        
        link_counter = 0

        for surf in self.surfaces:
            if surf.zone_name not in z_map: continue
            
            z_idx = z_map[surf.zone_name]
            
            # Populate Node Height (Centroid Z)
            # Simple approx: Zone Origin Z + Surface Centroid Z offset
            # In a real engine, we compute exact node heights.
            node_heights[z_idx] = self.zones[surf.zone_name]['origin'][2] + 1.5 # Approx mid-height
            
            # Determine Leakage Characteristic
            # Flow (kg/s) = C_flow * dP^n
            # C_flow approx = Area * LeakageFactor * Density_ref
            # Standard Leakage for tight construction: 0.0003 m3/s per m2 of surface area @ 4Pa
            
            is_window = (surf.type == 'WINDOW')
            
            # Base Leakage Area (ELA) estimation
            if is_window:
                # Crack flow around perimeter
                # Approx perimeter ~ 4 * sqrt(Area)
                L_crack = 4.0 * np.sqrt(surf.area)
                # ASHRAE: 0.15 L/s per meter crack for loose, 0.05 for tight
                flow_coeff = L_crack * 0.0001  # m3/(s·Pa^0.65) approx
                exponent = 0.65
            else:
                # Wall porosity
                flow_coeff = surf.area * 0.00005 
                exponent = 0.65

            # Determine Connectivity
            node_a = z_idx
            node_b = -1
            cp = 0.0
            
            if surf.boundary_condition == "AMBIENT":
                node_b = ambient_idx
                # Calculate Wind Pressure Coeff (Cp) based on Surface Azimuth vs Wind Dir
                # This happens at runtime, but we store the "Base Azimuth" here?
                # Actually, AFN solvers need a Cp map.
                # Simplified: Use surface normal to determine Windward/Leeward logic in solver.
                # We store a rough "Face Index" (0=N, 1=E, 2=S, 3=W, 4=Roof)
                # For now, we let the solver calculate Cp dynamically based on normals.
                # We mark this link as a boundary.
                boundary_indices.append(link_counter)
                boundary_cp.append(0.6) # Default windward, solver will adjust
                
            elif surf.boundary_condition.startswith("ZONE:"):
                neighbor_zone = surf.boundary_condition.split(":")[1]
                if neighbor_zone in z_map:
                    node_b = z_map[neighbor_zone]
            
            if node_b != -1:
                links_a.append(node_a)
                links_b.append(node_b)
                links_C.append(flow_coeff)
                links_n.append(exponent)
                link_counter += 1

        return AirflowConfig(
            link_node_a=jnp.array(links_a, dtype=jnp.int32),
            link_node_b=jnp.array(links_b, dtype=jnp.int32),
            link_C_flow=jnp.array(links_C, dtype=jnp.float32),
            link_exponent=jnp.array(links_n, dtype=jnp.float32),
            boundary_link_indices=jnp.array(boundary_indices, dtype=jnp.int32),
            boundary_Cp_coeffs=jnp.array(boundary_cp, dtype=jnp.float32),
            node_heights=jnp.array(node_heights, dtype=jnp.float32),
            n_internal_nodes=n_internal_nodes,
            n_total_nodes=n_internal_nodes + 1
        )

    def process_all(self):
        """Executes the full import pipeline."""
        print(f"--- Processing IDF: {self.idf.idfname} ---")
        
        self._parse_geometry_rules()
        self._parse_materials()
        self._parse_constructions()
        self._parse_zones()
        self._parse_surfaces()     
        self._parse_fenestration() 
        
        self._compile_network()
        
        self._parse_schedules()
        self._parse_loads()

        n_zones = len(self.zones)
        af_config = self._synthesize_airflow_network(n_zones)
        geo_config = self._compile_geometry()
        
        # Inject into ThermalConfig
        # We must rebuild the ThermalConfig created by builder.compile()
        # because eqx modules are immutable.
        self.thermal_config = eqx.tree_at(
            lambda t: (t.airflow_config, t.geometry_config),
            self.thermal_config,
            (af_config, geo_config)
        )
        
        print(f"--- Import Complete ---")
        if self.thermal_config:
            print(f"Nodes: {len(self.thermal_config.node_names)}")
            print(f"Total Volume: {self.thermal_config.room_vol_m3:.1f} m3")
        print(f"Machine Configs: {len(self.machine_configs)}")
        print(f"Occupant Configs: {len(self.occupant_configs)}")

    # =========================================================================
    # PHASE 1: Physics (Materials & Constructions)
    # =========================================================================

    def _parse_geometry_rules(self):
        rules = self.idf.idfobjects["GLOBALGEOMETRYRULES"]
        if rules:
            self.coord_sys = rules[0].Coordinate_System.capitalize()

    def _parse_materials(self):
        for mat in self.idf.idfobjects["MATERIAL"]:
            name = mat.Name.upper()
            try:
                self.materials[name] = Material(
                    name=name,
                    conductivity=float(mat.Conductivity),
                    density=float(mat.Density),
                    specific_heat=float(mat.Specific_Heat),
                    thickness=float(mat.Thickness),
                    absorptivity=float(getattr(mat, 'Solar_Absorptance', 0.7))
                )
            except ValueError: pass

        for mat in self.idf.idfobjects["MATERIAL:NOMASS"]:
            name = mat.Name.upper()
            r_val = float(mat.Thermal_Resistance)
            self.materials[name] = Material(
                name=name,
                conductivity=0.03,
                density=30.0,
                specific_heat=1400.0,
                thickness=r_val * 0.03,
                absorptivity=float(getattr(mat, 'Solar_Absorptance', 0.7))
            )

        for glass in self.idf.idfobjects["WINDOWMATERIAL:GLAZING"]:
            name = glass.Name.upper()
            th = float(glass.Thickness)
            k = float(glass.Conductivity)
            trans = float(glass.Solar_Transmittance_at_Normal_Incidence)
            self.window_types[name] = WindowType(
                name=name,
                u_value=k/th,
                shgc=trans * 1.1,
                glass_thickness=th
            )

        for glass in self.idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"]:
            name = glass.Name.upper()
            self.window_types[name] = WindowType(
                name=name,
                u_value=float(glass.UFactor),
                shgc=float(glass.Solar_Heat_Gain_Coefficient)
            )

    def _parse_constructions(self):
        for constr in self.idf.idfobjects["CONSTRUCTION"]:
            name = constr.Name.upper()
            layers = []
            for field in constr.obj:
                if isinstance(field, str) and field != constr.key and field != constr.Name:
                    layers.append(field.upper())
            
            if not layers: continue

            glass_layer = next((l for l in layers if l in self.window_types), None)
            if glass_layer:
                self.constructions[name] = {"type": "WINDOW", "obj": self.window_types[glass_layer]}
            else:
                mat = next((l for l in layers if l in self.materials), None)
                if mat:
                    self.constructions[name] = {"type": "OPAQUE", "obj": self.materials[mat]}

    # =========================================================================
    # PHASE 2: Geometry
    # =========================================================================

    def _parse_zones(self):
        for zone in self.idf.idfobjects["ZONE"]:
            name = zone.Name.upper()
            vol = float(zone.Volume) if hasattr(zone, 'Volume') and zone.Volume else 0.0
            
            self.zones[name] = {
                "origin": np.array([float(getattr(zone, k, 0)) for k in ['X_Origin', 'Y_Origin', 'Z_Origin']]),
                "volume": vol,
                "multiplier": float(getattr(zone, 'Multiplier', 1.0)),
                "floor_area": 0.0
            }
        
        for zl in self.idf.idfobjects["ZONELIST"]:
            name = zl.Name.upper()
            zones_in_list = []
            for field in zl.obj:
                if isinstance(field, str) and field != zl.key and field != zl.Name:
                    zones_in_list.append(field.upper())
            self.zone_lists[name] = zones_in_list

    def _get_coords(self, obj, zone_name: str) -> List[np.ndarray]:
        raw_coords = []
        for v in obj.coords:
            raw_coords.append(np.array(v))
            
        if self.coord_sys == "Relative" and zone_name in self.zones:
            origin = self.zones[zone_name]["origin"]
            return [v + origin for v in raw_coords]
        
        return raw_coords

    def _parse_surfaces(self):
        for surf in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            name = surf.Name.upper()
            zone_name = surf.Zone_Name.upper()
            constr_name = surf.Construction_Name.upper()
            
            if constr_name not in self.constructions: continue 
            
            coords = self._get_coords(surf, zone_name)
            if len(coords) < 3: continue
            
            area = get_polygon_area_3d(coords)
            
            if surf.Surface_Type.upper() == 'FLOOR' and zone_name in self.zones:
                self.zones[zone_name]["floor_area"] += area

            bc = surf.Outside_Boundary_Condition.upper()
            boundary = "AMBIENT"
            if bc == "GROUND": boundary = "GROUND"
            elif bc == "SURFACE": boundary = f"SURF_LINK:{surf.Outside_Boundary_Condition_Object.upper()}"
            elif bc == "ADIABATIC": boundary = "ADIABATIC"
            
            normal = get_polygon_normal(coords)
            az, tilt = get_azimuth_tilt(normal)

            self.surfaces.append(Surface(
                name=name,
                zone_name=zone_name,
                type=surf.Surface_Type.upper(),
                area=area,
                azimuth=az,
                tilt=tilt,
                construction_name=constr_name,
                boundary_condition=boundary
            ))

    def _parse_fenestration(self):
        wall_to_zone = {s.name: s.zone_name for s in self.surfaces}
        
        for win in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
            name = win.Name.upper()
            parent_name = win.Building_Surface_Name.upper()
            
            if parent_name not in wall_to_zone: continue
            zone_name = wall_to_zone[parent_name]
            
            coords = self._get_coords(win, zone_name)
            area = get_polygon_area_3d(coords)
            normal = get_polygon_normal(coords)
            az, tilt = get_azimuth_tilt(normal)
            
            self.surfaces.append(Surface(
                name=name,
                zone_name=zone_name,
                type='WINDOW',
                area=area,
                azimuth=az,
                tilt=tilt,
                construction_name=win.Construction_Name.upper(),
                boundary_condition='AMBIENT'
            ))

    # =========================================================================
    # PHASE 3: Network Compilation
    # =========================================================================

    def _resolve_boundaries(self):
        surf_map = {s.name: s.zone_name for s in self.surfaces}
        for s in self.surfaces:
            if s.boundary_condition.startswith("SURF_LINK:"):
                neighbor_surf = s.boundary_condition.split(":")[1]
                if neighbor_surf in surf_map:
                    neighbor_zone = surf_map[neighbor_surf]
                    s.boundary_condition = f"ZONE:{neighbor_zone}"
                else:
                    s.boundary_condition = "AMBIENT"

    def _compile_network(self):
        self._resolve_boundaries()
        n_zones = len(self.zones)
        builder = RCNetworkBuilder(n_rooms=n_zones)
        phys = Coefficients()

        sorted_zones = sorted(self.zones.keys())
        zone_idx_map = {name: i for i, name in enumerate(sorted_zones)}

        total_volume = 0.0
        
        # -- Step B Preparation: Container for Radiosity Mesh Data --
        # Map[zone_idx] -> List[Dict{'node_name': str, 'area': float, 'eps': float}]
        zone_surface_nodes = {i: [] for i in range(n_zones)}

        # --- 1. Create Air Nodes & Infiltration ---
        for z_name in sorted_zones:
            z_data = self.zones[z_name]
            z_idx = zone_idx_map[z_name]
            
            # Volume Handling
            vol = z_data['volume']
            if vol <= 0.1:
                if z_data['floor_area'] > 0.1:
                    vol = z_data['floor_area'] * 3.0 # Assume 3m ceiling
                else:
                    vol = 50.0 # Fallback
            total_volume += vol

            # Create Air Node
            c_air = vol * phys.air_density_kg_m3 * phys.air_heat_capacity_j_kgk
            node_air = f"room_air_{z_idx}"
            builder.add_node(node_air, capacity_j_k=c_air)
            self._zone_to_node_map[z_name] = node_air

            # Input Mappings (Connect HVAC directly to Air)
            # Note: In high-fidelity mode, we might split this, but for now 100% convective input is safer
            # unless we have a specific radiant system model.
            builder.add_input_mapping("heating_w", node_air, z_idx, fraction=1.0)
            builder.add_input_mapping("cooling_w", node_air, z_idx, fraction=1.0)

        # Setup Orifice Infiltration (Step C)
        # Heuristic: ELA (Effective Leakage Area) ~ Volume / 10,000 (approx 0.5 ACH50)
        # Stack Coeff ~ 0.12 (standard house), Wind Coeff ~ 0.09
        ela_total = total_volume / 10000.0
        builder.set_infiltration(total_volume, k1=ela_total, k2=0.12, k3=0.09)
        
        builder.add_node("ground", capacity_j_k=float("inf"))

        # --- 2. Process Surfaces & Build Wall Chains ---
        for surf in self.surfaces:
            if surf.zone_name not in zone_idx_map: continue
            z_idx = zone_idx_map[surf.zone_name]
            node_air = f"room_air_{z_idx}"

            if surf.construction_name not in self.constructions: continue
            data = self.constructions[surf.construction_name]

            # Determine Convection Coefficient based on orientation
            if 60.0 < surf.tilt < 120.0: h_cv_val = phys.h_cv_vertical
            elif surf.tilt <= 60.0: h_cv_val = phys.h_cv_ceiling
            else: h_cv_val = phys.h_cv_floor
            
            r_cv = 1.0 / (h_cv_val * surf.area)

            # -- WINDOWS --
            if surf.type == 'WINDOW' and data['type'] == 'WINDOW':
                w_def = data['obj']
                w_def.calculate_properties()
                
                g_node = f"win_{surf.name}_glass"
                c_glass = w_def.density * w_def.specific_heat * w_def.glass_thickness * surf.area
                builder.add_node(g_node, capacity_j_k=c_glass)

                # Connect Convection (Air <-> Glass)
                builder.add_resistor(node_air, g_node, r_cv, dynamic=True)
                
                # Connect Conduction/Gap (Glass <-> Ambient)
                builder.add_resistor(g_node, "ambient", w_def.R_glazing_layer + (1.0/23.0))

                # Register for Radiosity Mesh
                zone_surface_nodes[z_idx].append({
                    'name': g_node,
                    'area': surf.area,
                    'eps': 0.84 # Standard glass emissivity
                })

                # Register for Solar/Wind
                self.dynamic_surfaces.append({"node_name": g_node, "area": surf.area, "roughness_mult": 1.0})
                self.solar_surfaces.append({
                    "type": "WINDOW", "name": surf.name, "normal": surf.normal, "area": surf.area,
                    "shgc": w_def.shgc, "target_trans_node": node_air, "target_abs_node": g_node
                })

            # -- OPAQUE WALLS --
            elif data['type'] == 'OPAQUE':
                mat = data['obj']
                
                r_cond = mat.thickness / (mat.conductivity * surf.area)
                c_tot = mat.thickness * mat.density * mat.specific_heat * surf.area
                
                # High-fidelity slicing
                n_slices = 3 if mat.thickness > 0.15 else 2
                
                # Create the RC Chain
                n_int, n_ext = builder.add_capacitive_chain(f"wall_{surf.name}", r_cond, c_tot, n_slices)

                # Connect Convection (Air <-> Interior Surface)
                builder.add_resistor(node_air, n_int, r_cv, dynamic=True)

                # Register Interior Node for Radiosity Mesh (Crucial Step B)
                zone_surface_nodes[z_idx].append({
                    'name': n_int,
                    'area': surf.area,
                    'eps': surf.emissivity_longwave
                })

                # Handle Boundary Conditions (Exterior)
                if surf.boundary_condition == "AMBIENT":
                    builder.add_resistor(n_ext, "ambient", 1.0/(23.0*surf.area))
                    self.dynamic_surfaces.append({"node_name": n_ext, "area": surf.area, "roughness_mult": 1.5})
                    self.solar_surfaces.append({
                        "type": "WALL", "name": surf.name, "normal": surf.normal, "area": surf.area,
                        "absorptivity": mat.absorptivity, "target_node": n_ext
                    })
                elif surf.boundary_condition == "GROUND":
                    builder.add_resistor(n_ext, "ground", r_cond * 2.0) # Approximate
                elif surf.boundary_condition.startswith("ZONE:"):
                    neigh = surf.boundary_condition.split(":")[1]
                    if neigh in zone_idx_map:
                        neigh_air = f"room_air_{zone_idx_map[neigh]}"
                        builder.add_resistor(n_ext, neigh_air, r_cv)

        # --- 3. Build Explicit Radiosity Mesh (Step B) ---
        # We create a mesh of resistors connecting all interior surfaces within a zone.
        # G_ij = h_rad_linear * Area_i * ViewFactor_ij
        
        STEFAN_BOLTZMANN = 5.67e-8
        T_REF = 293.15 
        # Linearized radiative coefficient ~5.7 W/m2K
        H_RAD_LINEAR = 4.0 * STEFAN_BOLTZMANN * (T_REF**3) 

        for z_idx, surf_nodes in zone_surface_nodes.items():
            if len(surf_nodes) < 2: continue

            total_area = sum(s['area'] for s in surf_nodes)
            
            # Connect every surface to every other surface
            for i, s_i in enumerate(surf_nodes):
                for j, s_j in enumerate(surf_nodes):
                    if i >= j: continue # Avoid duplicates and self-loops

                    # Approximation: "Mean Radiant View Factor" for convex zones
                    # F_ij = A_j / (Total_Area - A_i)
                    # This assumes radiation leaving i hits everything else proportional to area.
                    F_ij = s_j['area'] / (total_area - s_i['area'] + 1e-4)
                    
                    # Calculate Conductance
                    # G = h_rad * A_i * F_ij * epsilon_factor
                    # (Assuming eps ~ 0.9 for most building materials)
                    G_rad = H_RAD_LINEAR * s_i['area'] * F_ij * 0.9 
                    
                    R_rad = 1.0 / (G_rad + 1e-9)
                    
                    # Add DIRECT resistor between surface nodes
                    builder.add_resistor(s_i['name'], s_j['name'], R_rad)

        # --- 4. Zone Mixing ---
        for mix in self.idf.idfobjects["ZONEMIXING"]:
            dest_zone = mix.Zone_Name.upper()
            source_zone = mix.Source_Zone_Name.upper()
            flow_rate = float(mix.Design_Flow_Rate) if mix.Design_Flow_Rate else 0.1

            if dest_zone in self._zone_to_node_map and source_zone in self._zone_to_node_map:
                node_a = self._zone_to_node_map[dest_zone]
                node_b = self._zone_to_node_map[source_zone]
                builder.add_mixing(node_a, node_b, flow_rate)

        self.thermal_config = builder.compile()

    # =========================================================================
    # PHASE 4: Schedules & Loads
    # =========================================================================

    def _parse_schedule_compact(self, sched_obj) -> np.ndarray:
        profile_24h = np.zeros(24)
        current_hour = 0
        fields = sched_obj.obj[3:]
        i = 0
        while i < len(fields):
            val = str(fields[i]).upper()
            if "UNTIL" in val:
                try:
                    time_str = val.replace("UNTIL", "").replace(":", "").strip()
                    if len(time_str) <= 2: h, m = int(time_str), 0
                    else: h, m = int(time_str.split()[0]), 0
                    end_hour = h + (m/60.0)
                    val_next = float(fields[i+1])
                    start_idx, end_idx = int(current_hour), int(end_hour)
                    if end_idx > 24: end_idx = 24
                    profile_24h[start_idx:end_idx] = val_next
                    current_hour = end_hour
                    i += 2
                except: i += 1
            elif "THROUGH" in val or "FOR" in val: i += 1
            else: i += 1
            if current_hour >= 24: break
        return np.tile(profile_24h, 365)

    def _parse_schedules(self):
        for sched in self.idf.idfobjects["SCHEDULE:COMPACT"]:
            try: self.schedules[sched.Name.upper()] = self._parse_schedule_compact(sched)
            except: self.schedules[sched.Name.upper()] = np.zeros(8760)


if __name__ == "__main__":
    idf_file = "RefBldgWarehouseNew2004_Chicago.idf"
    idd_file = "Energy+.idd"

    smart_appliance_specs = {
        "DISHWASHER": 1.5,
        "EV_CHARGER": 40.0
    } # this building has no smart appliances, just for demo
    
    bridge = EnergyPlusBridge(idf_file, idd_file, smart_appliance_specs)
    bridge.process_all()
    
    thermal_config = bridge.thermal_config
    dynamic_surfaces = bridge.dynamic_surfaces
    solar_surfaces = bridge.solar_surfaces
    machine_configs = bridge.machine_configs
    occupant_configs = bridge.occupant_configs
    schedules = bridge.schedules
    
    print("Thermal Config", thermal_config)
    print("Dynamic Surfaces", dynamic_surfaces)
    print("Solar Surfaces", solar_surfaces)
    print("Machine Configs", machine_configs)
    print("Occupant Configs", occupant_configs)