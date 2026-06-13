"""High-fidelity-ish orbital simulation and sensor measurement generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coordinates import StationGeometry, line_of_sight_measurement, station_to_ecef
from .dynamics import kepler_to_cartesian, propagate_orbit


@dataclass(frozen=True)
class OrbitSamplingConfig:
    altitude_min_km: float
    altitude_max_km: float
    eccentricity_min: float
    eccentricity_max: float
    inclination_min_deg: float
    inclination_max_deg: float


@dataclass(frozen=True)
class MeasurementNoiseConfig:
    range_std_m: float
    az_std_deg: float
    el_std_deg: float
    range_rate_std_mps: float
    outlier_prob: float
    outlier_scale: float
    range_bias_std_m: float = 0.0
    az_bias_std_deg: float = 0.0
    el_bias_std_deg: float = 0.0
    range_rate_bias_std_mps: float = 0.0
    clock_bias_std_s: float = 0.0
    clock_jitter_std_s: float = 0.0
    random_dropout_prob: float = 0.0

    @property
    def std_vector(self) -> np.ndarray:
        return np.array(
            [
                self.range_std_m,
                np.deg2rad(self.az_std_deg),
                np.deg2rad(self.el_std_deg),
                self.range_rate_std_mps,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class DynamicsConfig:
    dt_s: float
    steps: int
    ballistic_coeff_m2_per_kg: float
    process_noise_std: float
    drag_rho_ref: float
    drag_h_ref_m: float
    drag_scale_height_m: float
    enable_third_body: bool = False
    enable_srp: bool = False
    srp_area_to_mass_m2_per_kg: float = 0.02
    srp_cr: float = 1.3
    sun_initial_phase_rad: float = 0.0
    moon_initial_phase_rad: float = 0.0


@dataclass(frozen=True)
class DatasetConfig:
    orbit_sampling: OrbitSamplingConfig
    measurement_noise: MeasurementNoiseConfig
    dynamics: DynamicsConfig
    stations: tuple[StationGeometry, ...]


def parse_dataset_config(cfg: dict[str, Any]) -> DatasetConfig:
    stations = tuple(
        StationGeometry(
            name=s["name"],
            lat_deg=float(s["lat_deg"]),
            lon_deg=float(s["lon_deg"]),
            alt_m=float(s.get("alt_m", 0.0)),
            min_elevation_deg=float(s.get("min_elevation_deg", 10.0)),
        )
        for s in cfg["stations"]
    )
    return DatasetConfig(
        orbit_sampling=OrbitSamplingConfig(**cfg["orbit_sampling"]),
        measurement_noise=MeasurementNoiseConfig(**cfg["measurement_noise"]),
        dynamics=DynamicsConfig(**cfg["dynamics"]),
        stations=stations,
    )


def sample_initial_state(cfg: OrbitSamplingConfig, rng: np.random.Generator) -> np.ndarray:
    earth_radius_m = 6378.1363e3
    altitude_m = rng.uniform(cfg.altitude_min_km, cfg.altitude_max_km) * 1e3
    semi_major_axis_m = earth_radius_m + altitude_m
    eccentricity = rng.uniform(cfg.eccentricity_min, cfg.eccentricity_max)
    inclination_rad = np.deg2rad(rng.uniform(cfg.inclination_min_deg, cfg.inclination_max_deg))
    raan_rad = rng.uniform(0.0, 2.0 * np.pi)
    arg_perigee_rad = rng.uniform(0.0, 2.0 * np.pi)
    true_anomaly_rad = rng.uniform(0.0, 2.0 * np.pi)
    return kepler_to_cartesian(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=eccentricity,
        inclination_rad=inclination_rad,
        raan_rad=raan_rad,
        arg_perigee_rad=arg_perigee_rad,
        true_anomaly_rad=true_anomaly_rad,
    )


def _wrap_azimuth(az: float) -> float:
    if az < 0.0:
        az += 2.0 * np.pi
    if az >= 2.0 * np.pi:
        az -= 2.0 * np.pi
    return az


def simulate_single_trajectory(
    cfg: DatasetConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    dyn = cfg.dynamics
    n_stations = len(cfg.stations)
    times = np.arange(dyn.steps, dtype=np.float64) * dyn.dt_s

    x0 = sample_initial_state(cfg.orbit_sampling, rng)
    states = propagate_orbit(
        initial_state_eci=x0,
        dt=dyn.dt_s,
        steps=dyn.steps,
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        process_noise_std=dyn.process_noise_std,
        rng=rng,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
        enable_third_body=dyn.enable_third_body,
        enable_srp=dyn.enable_srp,
        srp_area_to_mass_m2_per_kg=dyn.srp_area_to_mass_m2_per_kg,
        srp_cr=dyn.srp_cr,
        sun_initial_phase_rad=dyn.sun_initial_phase_rad,
        moon_initial_phase_rad=dyn.moon_initial_phase_rad,
    )

    meas = np.zeros((dyn.steps, n_stations, 4), dtype=np.float64)
    vis = np.zeros((dyn.steps, n_stations), dtype=np.float64)
    meas_true = np.zeros_like(meas)
    noise_std = cfg.measurement_noise.std_vector
    bias_std = np.array(
        [
            cfg.measurement_noise.range_bias_std_m,
            np.deg2rad(cfg.measurement_noise.az_bias_std_deg),
            np.deg2rad(cfg.measurement_noise.el_bias_std_deg),
            cfg.measurement_noise.range_rate_bias_std_mps,
        ],
        dtype=np.float64,
    )
    station_bias = rng.normal(0.0, bias_std, size=(n_stations, 4))
    station_clock_bias = rng.normal(0.0, cfg.measurement_noise.clock_bias_std_s, size=n_stations)
    for t_idx, t_s in enumerate(times):
        for s_idx, station in enumerate(cfg.stations):
            t_eff = (
                t_s
                + station_clock_bias[s_idx]
                + rng.normal(0.0, cfg.measurement_noise.clock_jitter_std_s)
            )
            z_true, visible = line_of_sight_measurement(states[t_idx], station, t_eff)
            z_true = z_true + station_bias[s_idx]
            z_true[1] = _wrap_azimuth(float(z_true[1]))
            z_true[2] = float(np.clip(z_true[2], -0.5 * np.pi, 0.5 * np.pi))
            meas_true[t_idx, s_idx] = z_true
            if not visible:
                continue
            if rng.uniform() < cfg.measurement_noise.random_dropout_prob:
                continue
            eps = rng.normal(0.0, noise_std)
            if rng.uniform() < cfg.measurement_noise.outlier_prob:
                eps *= cfg.measurement_noise.outlier_scale
            z_noisy = z_true + eps
            z_noisy[1] = _wrap_azimuth(float(z_noisy[1]))
            z_noisy[2] = float(np.clip(z_noisy[2], -0.5 * np.pi, 0.5 * np.pi))
            meas[t_idx, s_idx] = z_noisy
            vis[t_idx, s_idx] = 1.0

    station_ecef = np.vstack([station_to_ecef(s) for s in cfg.stations]).astype(np.float64)
    station_llh = np.array(
        [[s.lat_rad, s.lon_rad, s.alt_m] for s in cfg.stations], dtype=np.float64
    )

    return {
        "states": states,
        "measurements": meas,
        "measurements_true": meas_true,
        "visibility": vis,
        "times": times,
        "station_ecef": station_ecef,
        "station_llh": station_llh,
    }


def generate_dataset(
    cfg: DatasetConfig,
    num_trajectories: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    trajs = [simulate_single_trajectory(cfg, rng) for _ in range(num_trajectories)]

    out: dict[str, np.ndarray] = {}
    for key in (
        "states",
        "measurements",
        "measurements_true",
        "visibility",
        "times",
    ):
        out[key] = np.stack([tr[key] for tr in trajs], axis=0)

    # Station metadata are shared across trajectories.
    out["station_ecef"] = trajs[0]["station_ecef"][None, ...]
    out["station_llh"] = trajs[0]["station_llh"][None, ...]
    return out
