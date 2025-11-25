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
    Material, Surface, WindowType, 
    ApplianceConfig, OccupantConfig, 
    ThermalConfig
)
from energysim.core.physics.constants import Coefficients
from energysim.core.network_builder import RCNetworkBuilder
from energysim.utils.geometry import get_polygon_normal, get_polygon_area_3d, get_azimuth_tilt
from energysim.core.physics.thermo import get_internal_convection







class EnergyPlusBridge:
    """
    A high-fidelity bridge to import EnergyPlus (.idf) models into EnergySim (JAX).
    Includes robust parsing for loads, schedules, and 'High-Fidelity' radiant network topology.
    """
    
    def __init__(self, idf_path: str, idd_path: str):
        IDF.setiddname(idd_path)
        self.idf = IDF(idf_path)
        
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

        for z_name in sorted_zones:
            z_data = self.zones[z_name]
            vol = z_data['volume']
            if vol <= 0.1:
                if z_data['floor_area'] > 0.1:
                    vol = z_data['floor_area'] * 3.0
                else:
                    vol = 50.0
            total_volume += vol

            c_air = vol * 1200.0 * 5.0 
            node_air = f"room_air_{zone_idx_map[z_name]}"
            builder.add_node(node_air, capacity_j_k=c_air)
            
            c_rad = c_air * 0.05 
            node_rad = f"room_rad_{zone_idx_map[z_name]}"
            builder.add_node(node_rad, capacity_j_k=c_rad)
            
            self._zone_to_node_map[z_name] = node_air 
            
            idx = zone_idx_map[z_name]
            builder.add_input_mapping("heating_w", node_air, idx, fraction=0.6)
            builder.add_input_mapping("heating_w", node_rad, idx, fraction=0.4)
            builder.add_input_mapping("cooling_w", node_air, idx, fraction=0.7)
            builder.add_input_mapping("cooling_w", node_rad, idx, fraction=0.3)

        builder.set_infiltration(total_volume, 0.5)
        builder.add_node("ground", capacity_j_k=float("inf"))

        for surf in self.surfaces:
            if surf.zone_name not in zone_idx_map: continue
            z_idx = zone_idx_map[surf.zone_name]
            node_air = f"room_air_{z_idx}"
            node_rad = f"room_rad_{z_idx}"
            
            if surf.construction_name not in self.constructions: continue
            data = self.constructions[surf.construction_name]
            
            if 60.0 < surf.tilt < 120.0: h_cv_val = phys.h_cv_vertical
            elif surf.tilt <= 60.0: h_cv_val = phys.h_cv_ceiling
            else: h_cv_val = phys.h_cv_floor
            
            r_cv = 1.0 / (h_cv_val * surf.area)
            r_rad = 1.0 / (phys.h_rad_interior * surf.area)

            if surf.type == 'WINDOW' and data['type'] == 'WINDOW':
                w_def = data['obj']
                w_def.calculate_properties()
                g_node = f"win_{surf.name}_glass"
                c_glass = w_def.density * w_def.specific_heat * w_def.glass_thickness * surf.area
                builder.add_node(g_node, capacity_j_k=c_glass)
                
                builder.add_resistor(node_air, g_node, r_cv)
                builder.add_resistor(node_rad, g_node, r_rad)
                builder.add_resistor(g_node, "ambient", w_def.R_glazing_layer + (1.0/23.0))

                self.dynamic_surfaces.append({"node_name": g_node, "area": surf.area, "roughness_mult": 1.0})
                self.solar_surfaces.append({
                    "type": "WINDOW", "name": surf.name, "normal": surf.normal, "area": surf.area, 
                    "shgc": w_def.shgc, "target_trans_node": node_rad, "target_abs_node": g_node
                })

            elif data['type'] == 'OPAQUE':
                mat = data['obj']
                n_int = f"wall_{surf.name}_int"
                n_ext = f"wall_{surf.name}_ext"
                c = mat.thickness * mat.density * mat.specific_heat * surf.area
                builder.add_node(n_int, capacity_j_k=c*0.5)
                builder.add_node(n_ext, capacity_j_k=c*0.5)
                
                builder.add_resistor(node_air, n_int, r_cv)
                builder.add_resistor(node_rad, n_int, r_rad)
                
                r_cond = mat.thickness / (mat.conductivity * surf.area)
                builder.add_resistor(n_int, n_ext, r_cond)
                
                if surf.boundary_condition == "AMBIENT":
                    builder.add_resistor(n_ext, "ambient", 1.0/(23.0*surf.area))
                    self.dynamic_surfaces.append({"node_name": n_ext, "area": surf.area, "roughness_mult": 1.5})
                    self.solar_surfaces.append({
                        "type": "WALL", "name": surf.name, "normal": surf.normal, "area": surf.area, 
                        "absorptivity": mat.absorptivity, "target_node": n_ext
                    })
                elif surf.boundary_condition == "GROUND":
                    builder.add_resistor(n_ext, "ground", r_cond * 2.0)
                elif surf.boundary_condition.startswith("ZONE:"):
                    neigh = surf.boundary_condition.split(":")[1]
                    if neigh in zone_idx_map:
                        neigh_air = f"room_air_{zone_idx_map[neigh]}"
                        builder.add_resistor(n_ext, neigh_air, r_cv)

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

    def _get_strict_float(self, ep_object, attr_name: str) -> Optional[float]:
        """Strictly gets a float attribute if it exists and is truthy."""
        if hasattr(ep_object, attr_name):
            val = getattr(ep_object, attr_name)
            if val:
                try: return float(val)
                except: pass
        return None

    def _resolve_power_instances(self, ep_object) -> List[Tuple[str, float]]:
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
            print(f"  [Load Warning] '{ep_object.Name}' targets unknown '{target_ref}'")
            return []

        # Strategy A: Absolute Watts
        val_abs = self._get_strict_float(ep_object, 'Design_Level') or self._get_strict_float(ep_object, 'Lighting_Level')
        if val_abs is not None:
            return [(n, val_abs) for n, _ in target_nodes]

        # Strategy B: Watts per Area
        val_area = self._get_strict_float(ep_object, 'Watts_per_Zone_Floor_Area')
        if val_area is not None:
            return [(n, val_area * area) for n, area in target_nodes]
            
        # Strategy C: People
        if ep_object.key.upper() == "PEOPLE":
            n_people = self._get_strict_float(ep_object, 'Number_of_People')
            if n_people is not None:
                return [(n, n_people * 120.0) for n, _ in target_nodes]
            p_area = self._get_strict_float(ep_object, 'People_per_Zone_Floor_Area')
            if p_area is not None:
                return [(n, p_area * area * 120.0) for n, area in target_nodes]

        print(f"  [Zero Warning] {ep_object.Name}: No valid power definition found (Design Level, W/m2, or People).")
        return []

    def _parse_loads(self):
        print("--- Parsing Loads ---")
        
        for p in self.idf.idfobjects["PEOPLE"]:
            instances = self._resolve_power_instances(p)
            for node, val in instances:
                self.occupant_configs.append(OccupantConfig(
                    name=f"{p.Name}_{node}", target_node_name=node, nominal_heat_w=val
                ))

        for eq in self.idf.idfobjects["ELECTRICEQUIPMENT"]:
            instances = self._resolve_power_instances(eq)
            for node, val in instances:
                self.machine_configs.append(ApplianceConfig(
                    name=f"{eq.Name}_{node}", target_node_name=node, 
                    nominal_power_w=val, convective_fraction=0.5, cycle_energy_kwh=0.0
                ))

        for lg in self.idf.idfobjects["LIGHTS"]:
            instances = self._resolve_power_instances(lg)
            for node, val in instances:
                self.machine_configs.append(ApplianceConfig(
                    name=f"{lg.Name}_{node}", target_node_name=node, 
                    nominal_power_w=val, convective_fraction=0.5, cycle_energy_kwh=0.0
                ))

if __name__ == "__main__":
    idf_file = "RefBldgWarehouseNew2004_Chicago.idf"
    idd_file = "Energy+.idd"
    
    bridge = EnergyPlusBridge(idf_file, idd_file)
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