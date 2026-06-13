"""Orbital dynamics and numerical propagation."""

from __future__ import annotations

import numpy as np

from .constants import (
    AU,
    EARTH_ORBIT_MEAN_MOTION_RADPS,
    EARTH_ROTATION_RATE,
    J2,
    MOON_MEAN_MOTION_RADPS,
    MOON_ORBIT_RADIUS,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    R_EARTH,
    R_SUN,
    SOLAR_RADIATION_PRESSURE_1AU,
)


def kepler_to_cartesian(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_rad: float,
    raan_rad: float,
    arg_perigee_rad: float,
    true_anomaly_rad: float,
) -> np.ndarray:
    """Convert Keplerian elements to ECI Cartesian state."""
    a = semi_major_axis_m
    e = eccentricity
    i = inclination_rad
    raan = raan_rad
    argp = arg_perigee_rad
    nu = true_anomaly_rad

    p = a * (1.0 - e**2)
    r_pqw = np.array(
        [p * np.cos(nu) / (1.0 + e * np.cos(nu)), p * np.sin(nu) / (1.0 + e * np.cos(nu)), 0.0],
        dtype=np.float64,
    )
    v_pqw = np.array(
        [
            -np.sqrt(MU_EARTH / p) * np.sin(nu),
            np.sqrt(MU_EARTH / p) * (e + np.cos(nu)),
            0.0,
        ],
        dtype=np.float64,
    )

    c_raan = np.cos(raan)
    s_raan = np.sin(raan)
    c_i = np.cos(i)
    s_i = np.sin(i)
    c_argp = np.cos(argp)
    s_argp = np.sin(argp)

    rot = np.array(
        [
            [c_raan * c_argp - s_raan * s_argp * c_i, -c_raan * s_argp - s_raan * c_argp * c_i, s_raan * s_i],
            [s_raan * c_argp + c_raan * s_argp * c_i, -s_raan * s_argp + c_raan * c_argp * c_i, -c_raan * s_i],
            [s_argp * s_i, c_argp * s_i, c_i],
        ],
        dtype=np.float64,
    )
    r_eci = rot @ r_pqw
    v_eci = rot @ v_pqw
    return np.hstack([r_eci, v_eci]).astype(np.float64)


def atmospheric_density_exponential(
    altitude_m: float,
    rho_ref: float = 4.0e-11,
    h_ref_m: float = 400e3,
    scale_height_m: float = 60e3,
) -> float:
    h = max(0.0, altitude_m)
    return float(rho_ref * np.exp(-(h - h_ref_m) / scale_height_m))


def sun_position_eci(t_s: float, initial_phase_rad: float = 0.0) -> np.ndarray:
    """Approximate Sun position in geocentric equatorial frame."""
    theta = EARTH_ORBIT_MEAN_MOTION_RADPS * t_s + initial_phase_rad
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([AU * c, AU * s, 0.0], dtype=np.float64)


def moon_position_eci(t_s: float, initial_phase_rad: float = 0.0) -> np.ndarray:
    """Approximate Moon position in geocentric equatorial frame."""
    theta = MOON_MEAN_MOTION_RADPS * t_s + initial_phase_rad
    c = np.cos(theta)
    s = np.sin(theta)
    # Mild inclination surrogate to avoid a strictly planar model.
    z = 0.089 * MOON_ORBIT_RADIUS * np.sin(theta + 0.3)
    xy = np.sqrt(max(MOON_ORBIT_RADIUS**2 - z**2, 1.0))
    return np.array([xy * c, xy * s, z], dtype=np.float64)


def third_body_acceleration(r_sat: np.ndarray, r_body: np.ndarray, mu_body: float) -> np.ndarray:
    d = r_body - r_sat
    d_norm = float(np.linalg.norm(d))
    rb_norm = float(np.linalg.norm(r_body))
    if d_norm < 1.0 or rb_norm < 1.0:
        return np.zeros(3, dtype=np.float64)
    return mu_body * (d / (d_norm**3) - r_body / (rb_norm**3))


def in_earth_shadow_cylinder(r_sat: np.ndarray, r_sun: np.ndarray) -> bool:
    """Fast eclipse test via cylindrical shadow approximation."""
    sun_dir = r_sun / max(float(np.linalg.norm(r_sun)), 1.0)
    proj = float(np.dot(r_sat, sun_dir))
    if proj > 0.0:
        return False
    perp = r_sat - proj * sun_dir
    return float(np.linalg.norm(perp)) <= R_EARTH


def srp_acceleration(
    r_sat: np.ndarray,
    r_sun: np.ndarray,
    area_to_mass_m2_per_kg: float,
    cr: float,
) -> np.ndarray:
    if area_to_mass_m2_per_kg <= 0.0:
        return np.zeros(3, dtype=np.float64)
    if in_earth_shadow_cylinder(r_sat=r_sat, r_sun=r_sun):
        return np.zeros(3, dtype=np.float64)
    rel = r_sat - r_sun  # vector from Sun to spacecraft
    dist = float(np.linalg.norm(rel))
    if dist < R_SUN:
        return np.zeros(3, dtype=np.float64)
    unit = rel / dist
    pressure = SOLAR_RADIATION_PRESSURE_1AU * (AU / dist) ** 2
    a_mag = pressure * cr * area_to_mass_m2_per_kg
    return a_mag * unit


def acceleration_eci(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    ballistic_coeff_m2_per_kg: float,
    t_s: float = 0.0,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> np.ndarray:
    x, y, z = r_eci
    r2 = float(np.dot(r_eci, r_eci))
    r = np.sqrt(r2)
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)

    # Two-body gravity
    a_grav = -MU_EARTH * r_eci / (r**3)

    # J2 perturbation
    z2 = z * z
    factor = 1.5 * J2 * MU_EARTH * (R_EARTH**2) / (r**5)
    common = 5.0 * z2 / r2
    a_j2 = factor * np.array([x * (common - 1.0), y * (common - 1.0), z * (common - 3.0)], dtype=np.float64)

    # Drag with rotating atmosphere
    altitude = r - R_EARTH
    rho = atmospheric_density_exponential(
        altitude_m=altitude,
        rho_ref=drag_rho_ref,
        h_ref_m=drag_h_ref_m,
        scale_height_m=drag_scale_height_m,
    )
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE], dtype=np.float64)
    v_atm = np.cross(omega, r_eci)
    v_rel = v_eci - v_atm
    v_rel_norm = float(np.linalg.norm(v_rel))
    a_drag = -0.5 * ballistic_coeff_m2_per_kg * rho * v_rel_norm * v_rel

    a_total = a_grav + a_j2 + a_drag

    if enable_third_body or enable_srp:
        r_sun = sun_position_eci(t_s=t_s, initial_phase_rad=sun_initial_phase_rad)
    if enable_third_body:
        r_moon = moon_position_eci(t_s=t_s, initial_phase_rad=moon_initial_phase_rad)
        a_total = a_total + third_body_acceleration(r_sat=r_eci, r_body=r_sun, mu_body=MU_SUN)
        a_total = a_total + third_body_acceleration(r_sat=r_eci, r_body=r_moon, mu_body=MU_MOON)
    if enable_srp:
        a_total = a_total + srp_acceleration(
            r_sat=r_eci,
            r_sun=r_sun,
            area_to_mass_m2_per_kg=srp_area_to_mass_m2_per_kg,
            cr=srp_cr,
        )
    return a_total


def state_derivative(
    state_eci: np.ndarray,
    ballistic_coeff_m2_per_kg: float,
    t_s: float = 0.0,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> np.ndarray:
    r = state_eci[:3]
    v = state_eci[3:]
    a = acceleration_eci(
        r,
        v,
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        t_s=t_s,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
        enable_third_body=enable_third_body,
        enable_srp=enable_srp,
        srp_area_to_mass_m2_per_kg=srp_area_to_mass_m2_per_kg,
        srp_cr=srp_cr,
        sun_initial_phase_rad=sun_initial_phase_rad,
        moon_initial_phase_rad=moon_initial_phase_rad,
    )
    return np.hstack([v, a]).astype(np.float64)


def rk4_step(
    state_eci: np.ndarray,
    dt: float,
    ballistic_coeff_m2_per_kg: float,
    t_s: float = 0.0,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> np.ndarray:
    kwargs = {
        "ballistic_coeff_m2_per_kg": ballistic_coeff_m2_per_kg,
        "drag_rho_ref": drag_rho_ref,
        "drag_h_ref_m": drag_h_ref_m,
        "drag_scale_height_m": drag_scale_height_m,
        "enable_third_body": enable_third_body,
        "enable_srp": enable_srp,
        "srp_area_to_mass_m2_per_kg": srp_area_to_mass_m2_per_kg,
        "srp_cr": srp_cr,
        "sun_initial_phase_rad": sun_initial_phase_rad,
        "moon_initial_phase_rad": moon_initial_phase_rad,
    }
    k1 = state_derivative(state_eci, t_s=t_s, **kwargs)
    k2 = state_derivative(state_eci + 0.5 * dt * k1, t_s=t_s + 0.5 * dt, **kwargs)
    k3 = state_derivative(state_eci + 0.5 * dt * k2, t_s=t_s + 0.5 * dt, **kwargs)
    k4 = state_derivative(state_eci + dt * k3, t_s=t_s + dt, **kwargs)
    return (state_eci + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float64)


def propagate_orbit(
    initial_state_eci: np.ndarray,
    dt: float,
    steps: int,
    ballistic_coeff_m2_per_kg: float,
    process_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> np.ndarray:
    states = np.zeros((steps, 6), dtype=np.float64)
    states[0] = np.asarray(initial_state_eci, dtype=np.float64)
    for k in range(1, steps):
        states[k] = rk4_step(
            states[k - 1],
            dt=dt,
            ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
            t_s=(k - 1) * dt,
            drag_rho_ref=drag_rho_ref,
            drag_h_ref_m=drag_h_ref_m,
            drag_scale_height_m=drag_scale_height_m,
            enable_third_body=enable_third_body,
            enable_srp=enable_srp,
            srp_area_to_mass_m2_per_kg=srp_area_to_mass_m2_per_kg,
            srp_cr=srp_cr,
            sun_initial_phase_rad=sun_initial_phase_rad,
            moon_initial_phase_rad=moon_initial_phase_rad,
        )
        if process_noise_std > 0.0 and rng is not None:
            states[k] += rng.normal(0.0, process_noise_std, size=6)
    return states


def numerical_jacobian(
    func, x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Finite-difference Jacobian of `func(x)`."""
    x = np.asarray(x, dtype=np.float64)
    y0 = np.asarray(func(x), dtype=np.float64)
    jac = np.zeros((y0.size, x.size), dtype=np.float64)
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = eps
        yp = np.asarray(func(x + d), dtype=np.float64)
        ym = np.asarray(func(x - d), dtype=np.float64)
        jac[:, i] = (yp - ym) / (2.0 * eps)
    return jac
