import jax.numpy as jnp
from ..shared.data_structs import GeometryConfig, SolarCache
from .functions import get_sun_position
from .geometry_engine import calculate_dynamic_shading
from .functions import calculate_iam_polynomial

import jax


def precompute_solar_data(
    lat_deg: float, 
    geometry: GeometryConfig, 
    time_seconds_array: jnp.ndarray
) -> dict:
    """
    Runs ONCE before simulation start.
    """
    # 1. Vectorized Sun Position
    sun_vectors = jax.vmap(get_sun_position, in_axes=(0, None))(
        time_seconds_array, lat_deg
    ) # Shape: (T, 3)

    # --- 2. Calculate Surface Normals ---
    # Assuming geometry defined by triangles (v0, v1, v2)
    # Normal = Cross(v1-v0, v2-v0) normalized
    edge1 = geometry.obs_v1 - geometry.obs_v0
    edge2 = geometry.obs_v2 - geometry.obs_v0
    raw_normals = jnp.cross(edge1, edge2)
    # Robust normalization (avoid divide by zero)
    norms = jnp.linalg.norm(raw_normals, axis=1, keepdims=True)
    surface_normals = raw_normals / (norms + 1e-6) # Shape: (N_surf, 3)

    # --- 3. Calculate Incidence Angles (Cos Theta) ---
    # We need Dot Product of Sun(t) and Normal(n)
    # sun_vectors: (T, 3)
    # surface_normals: (N, 3)
    # Result: (T, N)
    
    # Einsum is perfect here: 'ti, ni -> tn' (Time, XYZ; Surf, XYZ -> Time, Surf)
    cos_theta_matrix = jnp.einsum('ti, ni -> tn', sun_vectors, surface_normals)
    
    # Physics check: Sun behind surface?
    # If cos_theta < 0, the sun is hitting the back of the surface (or surface facing away).
    # Solar gain should be 0. We clamp cos_theta to 0 for the IAM calc.
    cos_theta_clamped = jnp.maximum(cos_theta_matrix, 0.0)

    # --- 4. Calculate IAM (Vectorized) ---
    # Apply polynomial to the whole (T, N) matrix
    iam_matrix = calculate_iam_polynomial(cos_theta_clamped)
    
    # Masking: If cos_theta was negative (sun behind wall), IAM should strictly be 0? 
    # The polynomial might give 1.0 for 0 degrees (which is wrong if it's 180 degrees).
    # However, we passed clamped 0.0 (which implies 90 deg grazing). 
    # Let's explicitly mask back-facing surfaces to 0.0 IAM.
    is_front_facing = cos_theta_matrix > 0.0
    iam_matrix = jnp.where(is_front_facing, iam_matrix, 0.0)

    # --- 5. Calculate Shading (Ray Casting) ---
    # vmap calculate_dynamic_shading over the sun_vectors (axis 0)
    calc_shading_vmap = jax.vmap(
        calculate_dynamic_shading,
        in_axes=(0, None, None, None, None)
    )
    
    # Shape: (T, N_surf)
    shading = calc_shading_vmap(
        sun_vectors,
        geometry.surface_centroids,
        geometry.obs_v0,
        geometry.obs_v1,
        geometry.obs_v2
    )

    return SolarCache(
        sun_direction_vectors=sun_vectors,
        surface_shading_factors=shading,
        incident_angle_modifiers=iam_matrix
    )