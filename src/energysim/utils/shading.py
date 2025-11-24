# src/energysim/utils/shading.py

import numpy as np

def calculate_horizon_shading(
    solar_azimuths_deg: np.ndarray,
    solar_elevations_deg: np.ndarray,
    surface_azimuth_deg: float,
    obstruction_profile: list[tuple[float, float]] = None
) -> np.ndarray:
    """
    Returns a shading factor array (1 = unshaded, 0 = shaded) based on a horizon mask.
    """

    if obstruction_profile is None or len(obstruction_profile) == 0:
        return np.ones_like(solar_azimuths_deg, dtype=float)

    # Sort and separate azimuth/elevation
    profile = np.array(sorted(obstruction_profile, key=lambda x: x[0]), dtype=float)
    obs_az = profile[:, 0]
    obs_el = profile[:, 1]

    # Ensure full 0–360 coverage
    if obs_az[0] > 0:
        obs_az = np.insert(obs_az, 0, 0.0)
        obs_el = np.insert(obs_el, 0, obs_el[-1])
    if obs_az[-1] < 360:
        obs_az = np.append(obs_az, 360.0)
        obs_el = np.append(obs_el, obs_el[0])

    # Normalize sun azimuths
    sun_az = np.mod(solar_azimuths_deg, 360.0)

    # Interpolated horizon height
    horizon = np.interp(sun_az, obs_az, obs_el)

    # Blocked if sun elevation is below horizon
    blocked = solar_elevations_deg < horizon

    shading = np.ones_like(solar_azimuths_deg, dtype=float)
    shading[blocked] = 0.0
    return shading

def calculate_obstructed_svf(
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
    obstruction_profile: list[tuple[float, float]] = None
) -> float:
    """
    Computes the sky-view factor using an isotropic blocked-sky integration.
    """

    # Unobstructed SVF
    if not obstruction_profile:
        return (1.0 + np.cos(np.radians(surface_tilt_deg))) * 0.5

    # --- Horizon interpolation (0–360°) ---
    prof = np.array(sorted(obstruction_profile, key=lambda x: x[0]), dtype=float)
    az_p = prof[:, 0]
    el_p = prof[:, 1]

    if az_p[0] > 0:
        az_p = np.insert(az_p, 0, 0.0)
        el_p = np.insert(el_p, 0, el_p[-1])
    if az_p[-1] < 360:
        az_p = np.append(az_p, 360.0)
        el_p = np.append(el_p, el_p[0])

    horizon = np.interp(np.arange(360), az_p, el_p)

    # --- Surface normal ---
    t = np.radians(surface_tilt_deg)
    a = np.radians(surface_azimuth_deg)
    n = np.array([
        np.sin(t) * np.sin(a),
        np.sin(t) * np.cos(a),
        np.cos(t)
    ])

    # --- Discretization grid ---
    step = 2.0
    az_deg = np.arange(0, 360, step)
    el_deg = np.arange(step/2, 90, step)

    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    dΩ = np.cos(el_rad) * (np.radians(step)**2)

    # Precompute sin/cos
    sin_az = np.sin(az_rad)
    cos_az = np.cos(az_rad)
    sin_el = np.sin(el_rad)
    cos_el = np.cos(el_rad)

    visible = 0.0

    # --- Integration ---
    for i, az in enumerate(az_deg):
        h_el = horizon[int(az)]
        mask_el = el_deg > h_el
        if not np.any(mask_el):
            continue

        # Patch direction vectors for all elevations at this azimuth
        sx = cos_el[mask_el] * sin_az[i]
        sy = cos_el[mask_el] * cos_az[i]
        sz = sin_el[mask_el]
        dirs = np.stack((sx, sy, sz), axis=1)

        cos_theta = dirs @ n
        pos = cos_theta > 0
        if np.any(pos):
            visible += np.sum(cos_theta[pos] * dΩ[mask_el][pos])

    svf = visible / np.pi
    return float(np.clip(svf, 0.0, 1.0))
