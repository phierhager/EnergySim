import jax
import jax.numpy as jnp
from functools import partial

def ray_intersects_triangle(
    ray_origin: jnp.ndarray,    # (3,)
    ray_dir: jnp.ndarray,       # (3,) Normalized
    v0: jnp.ndarray, v1: jnp.ndarray, v2: jnp.ndarray # Vertices (3,)
) -> bool:
    """
    Möller-Trumbore intersection algorithm. Differentiable-friendly.
    Returns True (1.0) if intersection occurs, False (0.0) otherwise.
    """
    EPSILON = 1e-6
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = jnp.cross(ray_dir, edge2)
    a = jnp.dot(edge1, h)
    
    # Parallel check (a ~ 0)
    is_parallel = (a > -EPSILON) & (a < EPSILON)
    
    f = 1.0 / (a + 1e-9) # Avoid div/0
    s = ray_origin - v0
    u = f * jnp.dot(s, h)
    
    # Barycentric Coordinate U check
    u_invalid = (u < 0.0) | (u > 1.0)
    
    q = jnp.cross(s, edge1)
    v = f * jnp.dot(ray_dir, q)
    
    # Barycentric Coordinate V check
    v_invalid = (v < 0.0) | (u + v > 1.0)
    
    # Distance t check
    t = f * jnp.dot(edge2, q)
    t_invalid = t < EPSILON # Intersection is behind ray
    
    hit = ~(is_parallel | u_invalid | v_invalid | t_invalid)
    return hit.astype(jnp.float32)

@jax.jit
def calculate_dynamic_shading(
    sun_vec: jnp.ndarray,       # (3,) Direction TO sun
    surface_centroids: jnp.ndarray, # (N_surf, 3)
    obstructors_v0: jnp.ndarray,    # (N_obs, 3) Triangle vertices
    obstructors_v1: jnp.ndarray,
    obstructors_v2: jnp.ndarray
) -> jnp.ndarray:
    """
    Vectorized All-vs-All Shadow Check.
    Returns: shading_factor (N_surf,) where 1.0 = Unshaded, 0.0 = Blocked.
    """
    
    # Inner function: Does Ray(i) hit Triangle(j)?
    def check_one_ray_one_tri(origin, v0, v1, v2):
        return ray_intersects_triangle(origin, sun_vec, v0, v1, v2)
    
    # vmap over Obstructors (j)
    check_all_obs = jax.vmap(check_one_ray_one_tri, in_axes=(None, 0, 0, 0))
    
    # vmap over Surfaces (i)
    check_all_surfs = jax.vmap(check_all_obs, in_axes=(0, None, None, None))
    
    # Matrix of hits: (N_surf, N_obs)
    hit_matrix = check_all_surfs(surface_centroids, obstructors_v0, obstructors_v1, obstructors_v2)
    
    # If ANY object blocks the ray, the surface is shaded.
    # sum(hits) > 0 -> Blocked
    blocked = jnp.sum(hit_matrix, axis=1) > 0
    
    # Differentiable "soft" logic can be used here if needed (sigmoid), 
    # but hard logic is usually fine for shading state.
    return jnp.where(blocked, 0.0, 1.0)