"""Public catalog replay helpers built from live orbital catalogs and station metadata."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sgp4.api import Satrec
from sgp4.omm import initialize as initialize_omm

from .coordinates import StationGeometry, line_of_sight_measurement, station_to_ecef
from .simulation import DatasetConfig


EARTH_RADIUS_M = 6378.1363e3


@dataclass(frozen=True)
class PublicCatalogEntry:
    fields: dict[str, Any]

    @property
    def object_name(self) -> str:
        return str(self.fields["OBJECT_NAME"])

    @property
    def object_id(self) -> str:
        return str(self.fields["OBJECT_ID"])

    @property
    def norad_cat_id(self) -> int:
        return int(self.fields["NORAD_CAT_ID"])

    @property
    def epoch(self) -> str:
        return str(self.fields["EPOCH"])

    @property
    def eccentricity(self) -> float:
        return float(self.fields["ECCENTRICITY"])

    @property
    def inclination_deg(self) -> float:
        return float(self.fields["INCLINATION"])

    @property
    def mean_motion_rev_per_day(self) -> float:
        return float(self.fields["MEAN_MOTION"])


@dataclass(frozen=True)
class PublicStationEntry:
    id: int
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    min_elevation_deg: float
    observations: int
    success_rate: float
    status: str


def load_public_catalog(path: str | Path) -> tuple[PublicCatalogEntry, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    return tuple(PublicCatalogEntry(fields=dict(row)) for row in payload)


def load_public_station_snapshot(path: str | Path) -> tuple[PublicStationEntry, ...]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    out: list[PublicStationEntry] = []
    for row in payload:
        try:
            out.append(
                PublicStationEntry(
                    id=int(row["id"]),
                    name=str(row["name"]),
                    lat_deg=float(row["lat"]),
                    lon_deg=float(row["lng"]),
                    alt_m=float(row.get("altitude", 0.0)),
                    min_elevation_deg=float(row.get("min_horizon", 10.0)),
                    observations=int(row.get("observations") or 0),
                    success_rate=float(row.get("success_rate") or 0.0),
                    status=str(row.get("status", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


def _wrap_azimuth(az: float) -> float:
    if az < 0.0:
        az += 2.0 * np.pi
    if az >= 2.0 * np.pi:
        az -= 2.0 * np.pi
    return az


def _entry_to_satrec(entry: PublicCatalogEntry) -> Satrec:
    sat = Satrec()
    fields = {
        "OBJECT_NAME": entry.object_name,
        "OBJECT_ID": entry.object_id,
        "CENTER_NAME": "EARTH",
        "REF_FRAME": "TEME",
        "TIME_SYSTEM": "UTC",
        "MEAN_ELEMENT_THEORY": "SGP4",
        "CLASSIFICATION_TYPE": str(entry.fields.get("CLASSIFICATION_TYPE", "U")),
        "EPHEMERIS_TYPE": str(entry.fields.get("EPHEMERIS_TYPE", 0)),
        "ELEMENT_SET_NO": str(entry.fields.get("ELEMENT_SET_NO", 0)),
        "REV_AT_EPOCH": str(entry.fields.get("REV_AT_EPOCH", 0)),
        "EPOCH": entry.epoch,
        "ARG_OF_PERICENTER": str(entry.fields.get("ARG_OF_PERICENTER", 0.0)),
        "BSTAR": str(entry.fields.get("BSTAR", 0.0)),
        "ECCENTRICITY": str(entry.fields.get("ECCENTRICITY", 0.0)),
        "INCLINATION": str(entry.fields.get("INCLINATION", 0.0)),
        "MEAN_ANOMALY": str(entry.fields.get("MEAN_ANOMALY", 0.0)),
        "MEAN_MOTION": str(entry.fields.get("MEAN_MOTION", 0.0)),
        "MEAN_MOTION_DOT": str(entry.fields.get("MEAN_MOTION_DOT", 0.0)),
        "MEAN_MOTION_DDOT": str(entry.fields.get("MEAN_MOTION_DDOT", 0.0)),
        "RA_OF_ASC_NODE": str(entry.fields.get("RA_OF_ASC_NODE", 0.0)),
        "NORAD_CAT_ID": str(entry.fields.get("NORAD_CAT_ID", 0)),
    }
    initialize_omm(sat, fields)
    return sat


def describe_public_entry(entry: PublicCatalogEntry) -> dict[str, float | str]:
    sat = _entry_to_satrec(entry)
    err, r_km, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation failed for {entry.object_name}: error={err}")
    altitude_km = float(np.linalg.norm(np.asarray(r_km, dtype=np.float64) * 1e3) - EARTH_RADIUS_M) / 1e3
    inclination_deg = entry.inclination_deg
    if inclination_deg < 40.0:
        bucket = "low_inclination"
    elif inclination_deg < 75.0:
        bucket = "mid_inclination"
    elif inclination_deg < 92.0:
        bucket = "high_inclination"
    else:
        bucket = "sunsync_like"
    return {
        "object_name": entry.object_name,
        "object_id": entry.object_id,
        "altitude_km": altitude_km,
        "eccentricity": entry.eccentricity,
        "inclination_deg": inclination_deg,
        "mean_motion_rev_per_day": entry.mean_motion_rev_per_day,
        "bucket": bucket,
        "epoch": entry.epoch,
    }


def filter_public_catalog(
    catalog: tuple[PublicCatalogEntry, ...],
    *,
    min_altitude_km: float | None = None,
    max_altitude_km: float | None = None,
    max_eccentricity: float | None = None,
    min_inclination_deg: float | None = None,
    max_inclination_deg: float | None = None,
    min_mean_motion_rev_per_day: float | None = None,
) -> tuple[PublicCatalogEntry, ...]:
    filtered: list[PublicCatalogEntry] = []
    for entry in catalog:
        desc = describe_public_entry(entry)
        if min_altitude_km is not None and float(desc["altitude_km"]) < float(min_altitude_km):
            continue
        if max_altitude_km is not None and float(desc["altitude_km"]) > float(max_altitude_km):
            continue
        if max_eccentricity is not None and float(desc["eccentricity"]) > float(max_eccentricity):
            continue
        if min_inclination_deg is not None and float(desc["inclination_deg"]) < float(min_inclination_deg):
            continue
        if max_inclination_deg is not None and float(desc["inclination_deg"]) > float(max_inclination_deg):
            continue
        if min_mean_motion_rev_per_day is not None and float(desc["mean_motion_rev_per_day"]) < float(min_mean_motion_rev_per_day):
            continue
        filtered.append(entry)
    return tuple(filtered)


def _propagate_public_entry(entry: PublicCatalogEntry, times_s: np.ndarray) -> np.ndarray:
    sat = _entry_to_satrec(entry)
    jd = np.full(times_s.shape, sat.jdsatepoch, dtype=np.float64)
    fr = np.full(times_s.shape, sat.jdsatepochF, dtype=np.float64) + times_s / 86400.0
    err, r_km, v_kmps = sat.sgp4_array(jd, fr)
    if np.any(err != 0):
        raise RuntimeError(f"SGP4 propagation failed for {entry.object_name}: errors={np.unique(err)}")
    return np.hstack([r_km * 1e3, v_kmps * 1e3]).astype(np.float64)


def _great_circle_distance_deg(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    a_lat_r = math.radians(a_lat)
    b_lat_r = math.radians(b_lat)
    d_lat = b_lat_r - a_lat_r
    d_lon = math.radians(b_lon - a_lon)
    s = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(a_lat_r) * math.cos(b_lat_r) * math.sin(d_lon / 2.0) ** 2
    )
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(max(0.0, s)))))


def select_public_ground_stations(
    stations: tuple[PublicStationEntry, ...],
    *,
    count: int = 8,
    min_success_rate: float = 60.0,
    min_observations: int = 1_000,
    require_online: bool = True,
) -> tuple[StationGeometry, ...]:
    deduped: dict[str, dict[str, Any]] = {}
    for station in stations:
        if require_online and station.status.lower() != "online":
            continue
        if station.observations < int(min_observations):
            continue
        if station.success_rate < float(min_success_rate):
            continue
        if not (-90.0 <= station.lat_deg <= 90.0 and -180.0 <= station.lon_deg <= 180.0):
            continue
        norm_name = station.name.strip().lower()
        score = math.log1p(max(station.observations, 0)) * (0.5 + station.success_rate / 100.0)
        current = deduped.get(norm_name)
        if current is None or score > float(current["score"]):
            deduped[norm_name] = {"station": station, "score": score}
    candidates = list(deduped.values())
    if len(candidates) < int(count):
        raise ValueError("Insufficient public stations remain after filtering.")

    scores = np.array([item["score"] for item in candidates], dtype=np.float64)
    score_min = float(scores.min())
    score_span = float(scores.max() - score_min)

    candidates.sort(key=lambda item: (-item["score"], item["station"].name))
    selected: list[PublicStationEntry] = [candidates[0]["station"]]
    remaining = candidates[1:]

    while len(selected) < int(count):
        best_idx = 0
        best_value = -float("inf")
        for idx, candidate in enumerate(remaining):
            station = candidate["station"]
            min_dist = min(
                _great_circle_distance_deg(
                    station.lat_deg,
                    station.lon_deg,
                    chosen.lat_deg,
                    chosen.lon_deg,
                )
                for chosen in selected
            )
            score_norm = 1.0 if score_span <= 1e-9 else (candidate["score"] - score_min) / score_span
            dist_norm = min_dist / 180.0
            value = 0.35 * score_norm + 0.65 * dist_norm
            if value > best_value:
                best_idx = idx
                best_value = value
        selected.append(remaining.pop(best_idx)["station"])

    return tuple(
        StationGeometry(
            name=station.name,
            lat_deg=station.lat_deg,
            lon_deg=station.lon_deg,
            alt_m=station.alt_m,
            min_elevation_deg=station.min_elevation_deg,
        )
        for station in selected
    )


def apply_public_station_selection(
    sim_cfg: dict[str, Any],
    scenario_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_cfg = copy.deepcopy(sim_cfg)
    station_path = scenario_cfg.get("station_snapshot_path")
    if not station_path:
        raise ValueError("public_catalog_replay requires station_snapshot_path.")
    stations = load_public_station_snapshot(station_path)
    station_filters = dict(scenario_cfg.get("station_filters", {}))
    selected = select_public_ground_stations(
        stations,
        count=int(station_filters.get("count", 8)),
        min_success_rate=float(station_filters.get("min_success_rate", 60.0)),
        min_observations=int(station_filters.get("min_observations", 1000)),
        require_online=bool(station_filters.get("require_online", True)),
    )
    out_cfg["stations"] = [
        {
            "name": station.name,
            "lat_deg": station.lat_deg,
            "lon_deg": station.lon_deg,
            "alt_m": station.alt_m,
            "min_elevation_deg": station.min_elevation_deg,
        }
        for station in selected
    ]
    summary = {
        "station_snapshot_path": str(station_path),
        "selected_station_names": [station.name for station in selected],
        "selected_station_count": len(selected),
    }
    return out_cfg, summary


def _select_catalog_entries(
    catalog: tuple[PublicCatalogEntry, ...],
    *,
    num_trajectories: int,
    rng: np.random.Generator,
    sampling_strategy: str,
) -> list[PublicCatalogEntry]:
    if sampling_strategy == "stratified_inclination":
        buckets: dict[str, list[PublicCatalogEntry]] = {
            "low_inclination": [],
            "mid_inclination": [],
            "high_inclination": [],
            "sunsync_like": [],
        }
        for entry in catalog:
            bucket = str(describe_public_entry(entry)["bucket"])
            buckets.setdefault(bucket, []).append(entry)
        active_buckets = [bucket for bucket, items in buckets.items() if items]
        if not active_buckets:
            raise ValueError("No public catalog entries remain after filtering.")
        per_bucket = max(1, num_trajectories // len(active_buckets))
        selected: list[PublicCatalogEntry] = []
        for bucket in active_buckets:
            items = buckets[bucket]
            take = min(len(items), per_bucket)
            idx = rng.choice(len(items), size=take, replace=False)
            selected.extend(items[int(i)] for i in np.asarray(idx).reshape(-1))
        while len(selected) < num_trajectories:
            entry = catalog[int(rng.integers(0, len(catalog)))]
            selected.append(entry)
        rng.shuffle(selected)
        return selected[:num_trajectories]

    idx = rng.choice(len(catalog), size=num_trajectories, replace=len(catalog) < num_trajectories)
    return [catalog[int(i)] for i in np.asarray(idx).reshape(-1)]


def generate_public_catalog_replay_dataset(
    cfg: DatasetConfig,
    *,
    catalog: tuple[PublicCatalogEntry, ...],
    num_trajectories: int,
    seed: int,
    catalog_filters: dict[str, Any] | None = None,
    sampling_strategy: str = "stratified_inclination",
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

    filter_cfg = dict(catalog_filters or {})
    if not filter_cfg:
        filter_cfg = {
            "min_altitude_km": 350.0,
            "max_altitude_km": 1500.0,
            "max_eccentricity": 0.05,
            "min_mean_motion_rev_per_day": 10.0,
        }
    catalog = filter_public_catalog(catalog, **filter_cfg)
    if not catalog:
        raise ValueError("No public-catalog entries remain after applying filters.")
    selected = _select_catalog_entries(
        catalog,
        num_trajectories=num_trajectories,
        rng=rng,
        sampling_strategy=sampling_strategy,
    )

    object_name = np.empty(num_trajectories, dtype="<U96")
    object_id = np.empty(num_trajectories, dtype="<U32")
    source_epoch = np.empty(num_trajectories, dtype="<U32")
    object_bucket = np.empty(num_trajectories, dtype="<U32")
    catalog_altitude_km = np.zeros(num_trajectories, dtype=np.float64)
    catalog_eccentricity = np.zeros(num_trajectories, dtype=np.float64)
    catalog_inclination_deg = np.zeros(num_trajectories, dtype=np.float64)
    catalog_mean_motion_rev_per_day = np.zeros(num_trajectories, dtype=np.float64)

    for traj_idx, entry in enumerate(selected):
        desc = describe_public_entry(entry)
        object_name[traj_idx] = str(desc["object_name"])
        object_id[traj_idx] = str(desc["object_id"])
        source_epoch[traj_idx] = str(desc["epoch"])
        object_bucket[traj_idx] = str(desc["bucket"])
        catalog_altitude_km[traj_idx] = float(desc["altitude_km"])
        catalog_eccentricity[traj_idx] = float(desc["eccentricity"])
        catalog_inclination_deg[traj_idx] = float(desc["inclination_deg"])
        catalog_mean_motion_rev_per_day[traj_idx] = float(desc["mean_motion_rev_per_day"])
        states = _propagate_public_entry(entry, times)
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
        "station_name": np.array([station.name for station in cfg.stations], dtype="<U96")[None, ...],
        "object_name": object_name,
        "object_id": object_id,
        "source_epoch": source_epoch,
        "object_bucket": object_bucket,
        "catalog_altitude_km": catalog_altitude_km,
        "catalog_eccentricity": catalog_eccentricity,
        "catalog_inclination_deg": catalog_inclination_deg,
        "catalog_mean_motion_rev_per_day": catalog_mean_motion_rev_per_day,
        "source_type": np.array(["public_catalog_replay"] * num_trajectories, dtype="<U32"),
    }
