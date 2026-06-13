"""Reference-frame transforms and measurement geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import EARTH_ROTATION_RATE, R_EARTH


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class StationGeometry:
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    min_elevation_deg: float = 10.0

    @property
    def lat_rad(self) -> float:
        return np.deg2rad(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        return np.deg2rad(self.lon_deg)

    @property
    def min_elevation_rad(self) -> float:
        return np.deg2rad(self.min_elevation_deg)


def rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def station_to_ecef(station: StationGeometry) -> np.ndarray:
    return geodetic_to_ecef(station.lat_rad, station.lon_rad, station.alt_m)


def eci_to_ecef(r_eci: np.ndarray, v_eci: np.ndarray, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)
    theta = EARTH_ROTATION_RATE * t_s
    rot = rot_z(theta)
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE], dtype=np.float64)
    r_ecef = rot @ r
    v_ecef = rot @ (v - np.cross(omega, r))
    return r_ecef, v_ecef


def ecef_to_eci(r_ecef: np.ndarray, v_ecef: np.ndarray, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r_ecef, dtype=np.float64)
    v = np.asarray(v_ecef, dtype=np.float64)
    theta = EARTH_ROTATION_RATE * t_s
    rot = rot_z(-theta)
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE], dtype=np.float64)
    r_eci = rot @ r
    v_eci = rot @ v + np.cross(omega, r_eci)
    return r_eci, v_eci


def ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> np.ndarray:
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def line_of_sight_measurement(
    state_eci: np.ndarray, station: StationGeometry, t_s: float
) -> tuple[np.ndarray, bool]:
    """Return [range, azimuth, elevation, range_rate] and visibility flag."""
    r_eci = state_eci[:3]
    v_eci = state_eci[3:]
    r_ecef, v_ecef = eci_to_ecef(r_eci, v_eci, t_s)

    station_ecef = station_to_ecef(station)
    rel = r_ecef - station_ecef
    # `eci_to_ecef` already maps inertial velocity to ECEF-frame relative velocity.
    # Ground stations are fixed in ECEF, so their velocity is zero in this frame.
    rel_v = v_ecef
    rho = np.linalg.norm(rel)
    if rho < 1e-6:
        return np.zeros(4, dtype=np.float64), False

    enu_rot = ecef_to_enu_matrix(station.lat_rad, station.lon_rad)
    east, north, up = enu_rot @ rel
    az = np.arctan2(east, north)
    if az < 0.0:
        az += 2.0 * np.pi
    el = np.arcsin(np.clip(up / rho, -1.0, 1.0))
    rho_dot = float(np.dot(rel, rel_v) / rho)
    visible = bool(el >= station.min_elevation_rad)
    return np.array([rho, az, el, rho_dot], dtype=np.float64), visible


def orbital_altitude_m(state_eci: np.ndarray) -> float:
    return float(np.linalg.norm(state_eci[:3]) - R_EARTH)
