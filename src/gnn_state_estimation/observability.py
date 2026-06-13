"""Measurement-information features for sparse OD diagnostics and learning."""

from __future__ import annotations

import math

import numpy as np

from .coordinates import StationGeometry, WGS84_A, WGS84_E2, line_of_sight_measurement
from .dataset import STATE_SCALE


OBSERVABILITY_CONTEXT_DIM = 6


def ecef_to_geodetic(xyz_m: np.ndarray) -> tuple[float, float, float]:
    """Convert ECEF XYZ to geodetic latitude, longitude, and altitude."""
    x, y, z = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / max(math.cos(lat), 1.0e-12) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), 1.0e-12) - n
    return float(lat), float(lon), float(alt)


def stations_from_ecef(station_ecef: np.ndarray, *, min_elevation_deg: float = -90.0) -> tuple[StationGeometry, ...]:
    stations: list[StationGeometry] = []
    for idx, xyz in enumerate(np.asarray(station_ecef, dtype=np.float64).reshape(-1, 3)):
        lat, lon, alt = ecef_to_geodetic(xyz)
        stations.append(
            StationGeometry(
                name=f"station_{idx}",
                lat_deg=float(np.rad2deg(lat)),
                lon_deg=float(np.rad2deg(lon)),
                alt_m=float(alt),
                min_elevation_deg=float(min_elevation_deg),
            )
        )
    return tuple(stations)


def wrapped_angle_difference(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def scaled_measurement_jacobian(
    state: np.ndarray,
    station: StationGeometry,
    time_s: float,
    measurement_std: np.ndarray,
    *,
    fd_relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Finite-difference d(measurement/std)/d(state/state_scale)."""
    state_scale = STATE_SCALE.astype(np.float64)
    std = np.asarray(measurement_std, dtype=np.float64).reshape(4)
    jac = np.zeros((4, 6), dtype=np.float64)
    state64 = np.asarray(state, dtype=np.float64).reshape(6)
    for col, scale in enumerate(state_scale):
        step = max(float(scale) * fd_relative_step, 1.0e-6)
        plus = state64.copy()
        minus = state64.copy()
        plus[col] += step
        minus[col] -= step
        z_plus, _ = line_of_sight_measurement(plus, station, float(time_s))
        z_minus, _ = line_of_sight_measurement(minus, station, float(time_s))
        diff = z_plus - z_minus
        diff[1] = wrapped_angle_difference(float(diff[1]))
        diff[2] = wrapped_angle_difference(float(diff[2]))
        jac[:, col] = (diff / (2.0 * step)) * scale / np.clip(std, 1.0e-12, None)
    return jac


def gramian_summary_features(
    gramian: np.ndarray,
    *,
    rank_threshold: float = 1.0e-6,
    visible_fraction: float = 0.0,
) -> np.ndarray:
    """Return normalized local information features from a 6x6 Gramian.

    Channels:
    0: clipped log10 trace / 12
    1: clipped log10 pseudo-determinant / 72
    2: clipped log10 condition / 12, where 1 is poorly conditioned
    3: rank fraction
    4: visible-station fraction
    5: clipped log10 minimum eigenvalue / 12
    """
    g = np.asarray(gramian, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(0.5 * (g + g.T))
    eigvals = np.clip(eigvals, 0.0, None)
    positive = eigvals[eigvals > rank_threshold]
    rank = int(positive.size)
    trace = float(np.trace(g))
    log_trace = float(np.log10(trace + 1.0e-12))
    log_pdet = float(np.sum(np.log10(eigvals + 1.0e-12)))
    if positive.size >= 2:
        condition = float(positive[-1] / positive[0])
        log_condition = float(np.log10(condition)) if math.isfinite(condition) else 12.0
    else:
        log_condition = 12.0
    min_log_eig = float(np.log10(eigvals[0] + 1.0e-12))
    return np.array(
        [
            np.clip(log_trace, -12.0, 12.0) / 12.0,
            np.clip(log_pdet, -72.0, 72.0) / 72.0,
            np.clip(log_condition, 0.0, 12.0) / 12.0,
            float(rank) / 6.0,
            float(np.clip(visible_fraction, 0.0, 1.0)),
            np.clip(min_log_eig, -12.0, 12.0) / 12.0,
        ],
        dtype=np.float32,
    )


def compute_observability_context_features(
    prior_states: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    meas_std_vector: np.ndarray,
    *,
    rank_threshold: float = 1.0e-6,
) -> np.ndarray:
    """Compute local measurement-information context from prior states.

    The features are causal with respect to the estimator state: they use the
    prior trajectory, declared visible stations, station geometry, and the
    configured measurement noise. They do not use truth states.
    """
    prior = np.asarray(prior_states, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    meas_std = np.asarray(meas_std_vector, dtype=np.float64).reshape(4)
    n_traj, t_steps = prior.shape[:2]
    n_stations = max(len(stations), 1)
    out = np.zeros((n_traj, t_steps, OBSERVABILITY_CONTEXT_DIM), dtype=np.float32)
    for traj_idx in range(n_traj):
        for step_idx in range(t_steps):
            gramian = np.zeros((6, 6), dtype=np.float64)
            active = np.where(vis[traj_idx, step_idx] >= 0.5)[0]
            for station_idx in active:
                jac = scaled_measurement_jacobian(
                    prior[traj_idx, step_idx],
                    stations[int(station_idx)],
                    float(times[traj_idx, step_idx]),
                    meas_std,
                )
                gramian += jac.T @ jac
            out[traj_idx, step_idx] = gramian_summary_features(
                gramian,
                rank_threshold=rank_threshold,
                visible_fraction=float(active.size) / float(n_stations),
            )
    return out
