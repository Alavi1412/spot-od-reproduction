"""Semi-real replay helpers built from archived public TLEs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sgp4.api import Satrec

from .coordinates import line_of_sight_measurement, station_to_ecef
from .simulation import DatasetConfig


@dataclass(frozen=True)
class TLEEntry:
    name: str
    line1: str
    line2: str
    source: str


EARTH_RADIUS_M = 6378.1363e3


def load_tle_catalog(path: str | Path) -> tuple[TLEEntry, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        TLEEntry(
            name=str(row["name"]),
            line1=str(row["line1"]),
            line2=str(row["line2"]),
            source=str(row.get("source", "archived")),
        )
        for row in payload
    )


def _wrap_azimuth(az: float) -> float:
    if az < 0.0:
        az += 2.0 * np.pi
    if az >= 2.0 * np.pi:
        az -= 2.0 * np.pi
    return az


def describe_tle(entry: TLEEntry) -> dict[str, float]:
    sat = Satrec.twoline2rv(entry.line1, entry.line2)
    err, r_km, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation failed for {entry.name}: error={err}")
    altitude_km = float(np.linalg.norm(np.asarray(r_km, dtype=np.float64) * 1e3) - EARTH_RADIUS_M) / 1e3
    mean_motion_rev_per_day = float(sat.no_kozai * 1440.0 / (2.0 * np.pi))
    return {
        "altitude_km": altitude_km,
        "eccentricity": float(sat.ecco),
        "inclination_deg": float(np.rad2deg(sat.inclo)),
        "mean_motion_rev_per_day": mean_motion_rev_per_day,
    }


def filter_tle_catalog(
    tle_catalog: tuple[TLEEntry, ...],
    *,
    min_altitude_km: float | None = None,
    max_altitude_km: float | None = None,
    max_eccentricity: float | None = None,
    min_inclination_deg: float | None = None,
    max_inclination_deg: float | None = None,
    min_mean_motion_rev_per_day: float | None = None,
) -> tuple[TLEEntry, ...]:
    filtered: list[TLEEntry] = []
    for entry in tle_catalog:
        desc = describe_tle(entry)
        if min_altitude_km is not None and desc["altitude_km"] < float(min_altitude_km):
            continue
        if max_altitude_km is not None and desc["altitude_km"] > float(max_altitude_km):
            continue
        if max_eccentricity is not None and desc["eccentricity"] > float(max_eccentricity):
            continue
        if min_inclination_deg is not None and desc["inclination_deg"] < float(min_inclination_deg):
            continue
        if max_inclination_deg is not None and desc["inclination_deg"] > float(max_inclination_deg):
            continue
        if min_mean_motion_rev_per_day is not None and desc["mean_motion_rev_per_day"] < float(min_mean_motion_rev_per_day):
            continue
        filtered.append(entry)
    return tuple(filtered)


def _propagate_tle(entry: TLEEntry, times_s: np.ndarray) -> np.ndarray:
    sat = Satrec.twoline2rv(entry.line1, entry.line2)
    jd = np.full(times_s.shape, sat.jdsatepoch, dtype=np.float64)
    fr = np.full(times_s.shape, sat.jdsatepochF, dtype=np.float64) + times_s / 86400.0
    err, r_km, v_kmps = sat.sgp4_array(jd, fr)
    if np.any(err != 0):
        raise RuntimeError(f"SGP4 propagation failed for {entry.name}: errors={np.unique(err)}")
    return np.hstack([r_km * 1e3, v_kmps * 1e3]).astype(np.float64)


def generate_semireal_replay_dataset(
    cfg: DatasetConfig,
    *,
    tle_catalog: tuple[TLEEntry, ...],
    num_trajectories: int,
    seed: int,
    tle_filters: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dyn = cfg.dynamics
    times = np.arange(dyn.steps, dtype=np.float64) * dyn.dt_s
    n_stations = len(cfg.stations)
    station_ecef = np.vstack([station_to_ecef(s) for s in cfg.stations]).astype(np.float64)
    station_llh = np.array([[s.lat_rad, s.lon_rad, s.alt_m] for s in cfg.stations], dtype=np.float64)

    states_all = np.zeros((num_trajectories, dyn.steps, 6), dtype=np.float64)
    meas_all = np.zeros((num_trajectories, dyn.steps, n_stations, 4), dtype=np.float64)
    vis_all = np.zeros((num_trajectories, dyn.steps, n_stations), dtype=np.float64)
    meas_true_all = np.zeros_like(meas_all)

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
    tle_name = np.empty(num_trajectories, dtype="<U64")
    tle_altitude_km = np.zeros(num_trajectories, dtype=np.float64)
    tle_eccentricity = np.zeros(num_trajectories, dtype=np.float64)
    tle_inclination_deg = np.zeros(num_trajectories, dtype=np.float64)
    tle_mean_motion_rev_per_day = np.zeros(num_trajectories, dtype=np.float64)

    filter_cfg = dict(tle_filters or {})
    if not filter_cfg:
        filter_cfg = {
            "min_altitude_km": 300.0,
            "max_altitude_km": 2000.0,
            "max_eccentricity": max(0.05, float(cfg.orbit_sampling.eccentricity_max) + 0.03),
            "min_mean_motion_rev_per_day": 10.0,
        }
    tle_catalog = filter_tle_catalog(tle_catalog, **filter_cfg)
    if not tle_catalog:
        raise ValueError("No TLEs remain after applying semi-real replay filters.")
    selected = rng.integers(0, len(tle_catalog), size=num_trajectories)

    for traj_idx, tle_idx in enumerate(selected):
        entry = tle_catalog[int(tle_idx)]
        tle_desc = describe_tle(entry)
        tle_name[traj_idx] = entry.name
        tle_altitude_km[traj_idx] = tle_desc["altitude_km"]
        tle_eccentricity[traj_idx] = tle_desc["eccentricity"]
        tle_inclination_deg[traj_idx] = tle_desc["inclination_deg"]
        tle_mean_motion_rev_per_day[traj_idx] = tle_desc["mean_motion_rev_per_day"]
        states = _propagate_tle(entry, times)
        states_all[traj_idx] = states
        station_bias = rng.normal(0.0, bias_std, size=(n_stations, 4))
        station_clock_bias = rng.normal(0.0, cfg.measurement_noise.clock_bias_std_s, size=n_stations)
        for t_idx, t_s in enumerate(times):
            for s_idx, station in enumerate(cfg.stations):
                t_eff = t_s + station_clock_bias[s_idx] + rng.normal(0.0, cfg.measurement_noise.clock_jitter_std_s)
                z_true, visible = line_of_sight_measurement(states[t_idx], station, t_eff)
                z_true = z_true + station_bias[s_idx]
                z_true[1] = _wrap_azimuth(float(z_true[1]))
                z_true[2] = float(np.clip(z_true[2], -0.5 * np.pi, 0.5 * np.pi))
                meas_true_all[traj_idx, t_idx, s_idx] = z_true
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
                meas_all[traj_idx, t_idx, s_idx] = z_noisy
                vis_all[traj_idx, t_idx, s_idx] = 1.0

    return {
        "states": states_all,
        "measurements": meas_all,
        "measurements_true": meas_true_all,
        "visibility": vis_all,
        "times": np.repeat(times[None, :], num_trajectories, axis=0),
        "station_ecef": station_ecef[None, ...],
        "station_llh": station_llh[None, ...],
        "tle_name": tle_name,
        "tle_altitude_km": tle_altitude_km,
        "tle_eccentricity": tle_eccentricity,
        "tle_inclination_deg": tle_inclination_deg,
        "tle_mean_motion_rev_per_day": tle_mean_motion_rev_per_day,
        "source_type": np.array(["semi_real_replay"] * num_trajectories, dtype="<U32"),
    }
