"""Replay helpers built from public SatNOGS observation windows."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sgp4.api import Satrec, jday

from .coordinates import StationGeometry, line_of_sight_measurement, station_to_ecef
from .simulation import DatasetConfig


EARTH_RADIUS_M = 6378.1363e3
EARTH_ROTATION_PERIOD_S = 86164.0905


@dataclass(frozen=True)
class PublicObservationEntry:
    fields: dict[str, Any]

    @property
    def observation_id(self) -> int:
        return int(self.fields["id"])

    @property
    def norad_cat_id(self) -> int:
        return int(self.fields["norad_cat_id"])

    @property
    def ground_station_id(self) -> int:
        return int(self.fields["ground_station"])

    @property
    def station_name(self) -> str:
        return str(self.fields["station_name"])

    @property
    def station_lat_deg(self) -> float:
        return float(self.fields["station_lat"])

    @property
    def station_lon_deg(self) -> float:
        return float(self.fields["station_lng"])

    @property
    def station_alt_m(self) -> float:
        return float(self.fields.get("station_alt") or 0.0)

    @property
    def start(self) -> str:
        return str(self.fields["start"])

    @property
    def end(self) -> str:
        return str(self.fields["end"])

    @property
    def tle1(self) -> str:
        return str(self.fields["tle1"])

    @property
    def tle2(self) -> str:
        return str(self.fields["tle2"])

    @property
    def status(self) -> str:
        return str(self.fields.get("status", ""))

    @property
    def vetted_status(self) -> str:
        return str(self.fields.get("vetted_status", ""))


def _parse_iso8601_z(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _datetime_to_unix(dt: datetime) -> float:
    return dt.timestamp()


def _tle_to_satrec(entry: PublicObservationEntry) -> Satrec:
    return Satrec.twoline2rv(entry.tle1, entry.tle2)


def describe_public_observation(entry: PublicObservationEntry) -> dict[str, float | str]:
    sat = _tle_to_satrec(entry)
    err, r_km, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation failed for observation {entry.observation_id}: error={err}")
    altitude_km = float(np.linalg.norm(np.asarray(r_km, dtype=np.float64) * 1e3) - EARTH_RADIUS_M) / 1e3
    inclination_deg = float(sat.inclo * 180.0 / np.pi)
    mean_motion_rev_per_day = float(sat.no_kozai * 1440.0 / (2.0 * np.pi))
    if inclination_deg < 40.0:
        bucket = "low_inclination"
    elif inclination_deg < 75.0:
        bucket = "mid_inclination"
    elif inclination_deg < 92.0:
        bucket = "high_inclination"
    else:
        bucket = "sunsync_like"
    return {
        "observation_id": entry.observation_id,
        "norad_cat_id": entry.norad_cat_id,
        "station_name": entry.station_name,
        "altitude_km": altitude_km,
        "inclination_deg": inclination_deg,
        "mean_motion_rev_per_day": mean_motion_rev_per_day,
        "bucket": bucket,
        "status": entry.status,
        "vetted_status": entry.vetted_status,
    }


def load_public_observation_snapshot(path: str | Path) -> tuple[PublicObservationEntry, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        payload = payload.get("observations", [])
    return tuple(PublicObservationEntry(fields=dict(row)) for row in payload)


def filter_public_observations(
    observations: tuple[PublicObservationEntry, ...],
    *,
    require_good_status: bool = True,
    min_altitude_km: float | None = None,
    max_altitude_km: float | None = None,
    min_mean_motion_rev_per_day: float | None = None,
) -> tuple[PublicObservationEntry, ...]:
    filtered: list[PublicObservationEntry] = []
    for entry in observations:
        if require_good_status and str(entry.status).lower() != "good":
            continue
        try:
            desc = describe_public_observation(entry)
        except RuntimeError:
            continue
        if min_altitude_km is not None and float(desc["altitude_km"]) < float(min_altitude_km):
            continue
        if max_altitude_km is not None and float(desc["altitude_km"]) > float(max_altitude_km):
            continue
        if min_mean_motion_rev_per_day is not None and float(desc["mean_motion_rev_per_day"]) < float(
            min_mean_motion_rev_per_day
        ):
            continue
        filtered.append(entry)
    return tuple(filtered)


def _normalize_station_key(name: str) -> str:
    return name.strip().lower()


def _select_station_keys(
    observations: tuple[PublicObservationEntry, ...],
    *,
    max_count: int | None,
) -> set[str]:
    if max_count is None or max_count <= 0:
        return {_normalize_station_key(entry.station_name) for entry in observations}
    counts: dict[str, int] = {}
    for entry in observations:
        key = _normalize_station_key(entry.station_name)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {key for key, _ in ranked[:max_count]}


def _filter_observations_to_station_keys(
    observations: tuple[PublicObservationEntry, ...],
    station_keys: set[str],
) -> tuple[PublicObservationEntry, ...]:
    return tuple(entry for entry in observations if _normalize_station_key(entry.station_name) in station_keys)


def _sort_observations_by_start(
    observations: tuple[PublicObservationEntry, ...],
) -> tuple[PublicObservationEntry, ...]:
    return tuple(sorted(observations, key=lambda entry: (entry.start, entry.observation_id), reverse=True))


def _slice_observations(
    observations: tuple[PublicObservationEntry, ...],
    *,
    source_slice: dict[str, Any] | None,
) -> tuple[tuple[PublicObservationEntry, ...], dict[str, int]]:
    ordered = _sort_observations_by_start(observations)
    total_count = len(ordered)
    if not source_slice:
        return ordered, {
            "source_pool_count": total_count,
            "slice_offset": 0,
            "slice_limit": total_count,
            "slice_count": total_count,
        }
    offset = int(source_slice.get("offset", 0) or 0)
    limit = source_slice.get("count")
    if offset < 0:
        raise ValueError("source_slice.offset must be non-negative.")
    if limit is None:
        sliced = ordered[offset:]
    else:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("source_slice.count must be positive when provided.")
        sliced = ordered[offset : offset + limit]
    if not sliced:
        raise ValueError(
            f"source_slice offset/count {offset}/{limit} selects no public observations from a pool of {total_count}."
        )
    return sliced, {
        "source_pool_count": total_count,
        "slice_offset": offset,
        "slice_limit": int(limit) if limit is not None else total_count,
        "slice_count": len(sliced),
    }


def _unique_observation_stations(
    observations: tuple[PublicObservationEntry, ...],
    *,
    station_keys: set[str] | None = None,
) -> list[StationGeometry]:
    seen: set[str] = set()
    stations: list[StationGeometry] = []
    for entry in observations:
        station_key = _normalize_station_key(entry.station_name)
        if station_keys is not None and station_key not in station_keys:
            continue
        if station_key in seen:
            continue
        seen.add(station_key)
        stations.append(
            StationGeometry(
                name=entry.station_name,
                lat_deg=entry.station_lat_deg,
                lon_deg=entry.station_lon_deg,
                alt_m=entry.station_alt_m,
                min_elevation_deg=8.0,
            )
        )
    return stations


def apply_public_observation_station_bank(
    sim_cfg: dict[str, Any],
    scenario_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_cfg = copy.deepcopy(sim_cfg)
    observation_path = scenario_cfg.get("observation_snapshot_path")
    if not observation_path:
        raise ValueError("public_observation_replay requires observation_snapshot_path.")
    observations = load_public_observation_snapshot(observation_path)
    filters = dict(scenario_cfg.get("observation_filters", {}))
    observations = filter_public_observations(
        observations,
        require_good_status=bool(filters.get("require_good_status", True)),
        min_altitude_km=filters.get("min_altitude_km"),
        max_altitude_km=filters.get("max_altitude_km"),
        min_mean_motion_rev_per_day=filters.get("min_mean_motion_rev_per_day"),
    )
    station_limit = scenario_cfg.get("station_filters", {}).get("count")
    station_keys = _select_station_keys(observations, max_count=station_limit)
    observations = _filter_observations_to_station_keys(observations, station_keys)
    sliced_observations, slice_summary = _slice_observations(
        observations,
        source_slice=scenario_cfg.get("source_slice"),
    )
    stations = _unique_observation_stations(sliced_observations, station_keys=station_keys)
    if not stations:
        raise ValueError("No observation-backed stations remain after filtering.")
    out_cfg["stations"] = [
        {
            "name": station.name,
            "lat_deg": station.lat_deg,
            "lon_deg": station.lon_deg,
            "alt_m": station.alt_m,
            "min_elevation_deg": station.min_elevation_deg,
        }
        for station in stations
    ]
    summary = {
        "observation_snapshot_path": str(observation_path),
        "selected_station_names": [station.name for station in stations],
        "selected_station_count": len(stations),
        "selected_observation_count": len(observations),
        "slice_observation_count": len(sliced_observations),
        **slice_summary,
    }
    return out_cfg, summary


def _select_anchor_observations(
    observations: tuple[PublicObservationEntry, ...],
    *,
    num_trajectories: int,
    rng: np.random.Generator,
) -> list[PublicObservationEntry]:
    by_bucket: dict[str, list[PublicObservationEntry]] = {
        "low_inclination": [],
        "mid_inclination": [],
        "high_inclination": [],
        "sunsync_like": [],
    }
    for entry in observations:
        bucket = str(describe_public_observation(entry)["bucket"])
        by_bucket.setdefault(bucket, []).append(entry)
    nonempty = [bucket for bucket, items in by_bucket.items() if items]
    if not nonempty:
        raise ValueError("No usable public observations remain after filtering.")
    selected: list[PublicObservationEntry] = []
    per_bucket = max(1, num_trajectories // len(nonempty))
    for bucket in nonempty:
        items = by_bucket[bucket]
        take = min(len(items), per_bucket)
        if take == len(items):
            chosen = list(items)
        else:
            idx = rng.choice(len(items), size=take, replace=False)
            chosen = [items[int(i)] for i in np.asarray(idx).reshape(-1)]
        chosen.sort(key=lambda item: item.start, reverse=True)
        selected.extend(chosen)
    selected_ids = {item.observation_id for item in selected}
    remaining_unique = [item for item in observations if item.observation_id not in selected_ids]
    remaining_unique.sort(key=lambda item: item.start, reverse=True)
    if len(selected) < num_trajectories:
        need = min(len(remaining_unique), num_trajectories - len(selected))
        if need > 0:
            idx = rng.choice(len(remaining_unique), size=need, replace=False)
            selected.extend(remaining_unique[int(i)] for i in np.asarray(idx).reshape(-1))
    while len(selected) < num_trajectories:
        selected.append(observations[int(rng.integers(0, len(observations)))])
    selected.sort(key=lambda item: item.start, reverse=True)
    return selected[:num_trajectories]


def _propagate_tle_window(entry: PublicObservationEntry, abs_unix_times: np.ndarray) -> np.ndarray:
    sat = _tle_to_satrec(entry)
    jd = np.empty(abs_unix_times.shape[0], dtype=np.float64)
    fr = np.empty(abs_unix_times.shape[0], dtype=np.float64)
    for idx, unix_time in enumerate(abs_unix_times):
        dt = datetime.fromtimestamp(float(unix_time), tz=timezone.utc)
        second = dt.second + dt.microsecond / 1.0e6
        jd[idx], fr[idx] = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, second)
    err, r_km, v_kmps = sat.sgp4_array(jd, fr)
    if np.any(err != 0):
        raise RuntimeError(
            f"SGP4 propagation failed for observation {entry.observation_id}: errors={np.unique(err).tolist()}"
        )
    return np.hstack([r_km * 1e3, v_kmps * 1e3]).astype(np.float64)


def _best_rotation_offset_s(
    states: np.ndarray,
    times_s: np.ndarray,
    station: StationGeometry,
    anchor_start_rel_s: float,
    anchor_end_rel_s: float,
) -> float:
    midpoint = 0.5 * (anchor_start_rel_s + anchor_end_rel_s)
    mid_idx = int(np.clip(np.searchsorted(times_s, midpoint), 0, len(times_s) - 1))
    candidate_offsets = np.linspace(0.0, EARTH_ROTATION_PERIOD_S, num=181, endpoint=False, dtype=np.float64)
    best_offset = 0.0
    best_score = -float("inf")
    for offset in candidate_offsets:
        z_mid, _ = line_of_sight_measurement(states[mid_idx], station, float(times_s[mid_idx] + offset))
        score = float(z_mid[2])
        if score > best_score:
            best_score = score
            best_offset = float(offset)
    return best_offset


def _station_intervals_for_object(
    observations: tuple[PublicObservationEntry, ...],
    *,
    norad_cat_id: int,
    station_index_by_name: dict[str, int],
    trajectory_start_unix: float,
) -> dict[int, list[tuple[float, float]]]:
    intervals: dict[int, list[tuple[float, float]]] = {}
    for entry in observations:
        if entry.norad_cat_id != norad_cat_id:
            continue
        station_idx = station_index_by_name.get(entry.station_name)
        if station_idx is None:
            continue
        start_rel = _datetime_to_unix(_parse_iso8601_z(entry.start)) - trajectory_start_unix
        end_rel = _datetime_to_unix(_parse_iso8601_z(entry.end)) - trajectory_start_unix
        intervals.setdefault(station_idx, []).append((start_rel, end_rel))
    return intervals


def generate_public_observation_replay_dataset(
    cfg: DatasetConfig,
    *,
    observations: tuple[PublicObservationEntry, ...],
    num_trajectories: int,
    seed: int,
    observation_filters: dict[str, Any] | None = None,
    source_slice: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    filters = dict(observation_filters or {})
    filtered = filter_public_observations(
        observations,
        require_good_status=bool(filters.get("require_good_status", True)),
        min_altitude_km=filters.get("min_altitude_km"),
        max_altitude_km=filters.get("max_altitude_km"),
        min_mean_motion_rev_per_day=filters.get("min_mean_motion_rev_per_day"),
    )
    if not filtered:
        raise ValueError("No public observations remain after applying replay filters.")
    station_limit = filters.get("station_count")
    dyn = cfg.dynamics
    times = np.arange(dyn.steps, dtype=np.float64) * dyn.dt_s
    trajectory_duration_s = float(times[-1]) if times.size else 0.0

    station_keys = _select_station_keys(filtered, max_count=int(station_limit) if station_limit else None)
    filtered = _filter_observations_to_station_keys(filtered, station_keys)
    filtered, slice_summary = _slice_observations(filtered, source_slice=source_slice)
    if len(filtered) < num_trajectories:
        raise ValueError(
            f"Observation replay has only {len(filtered)} usable passes after station selection, "
            f"but {num_trajectories} trajectories were requested."
        )
    anchors = _select_anchor_observations(filtered, num_trajectories=num_trajectories, rng=rng)
    station_bank = _unique_observation_stations(filtered, station_keys=station_keys)
    station_index_by_name = {station.name: idx for idx, station in enumerate(station_bank)}
    n_stations = len(station_bank)
    station_ecef = np.vstack([station_to_ecef(s) for s in station_bank]).astype(np.float64)
    station_llh = np.array([[s.lat_rad, s.lon_rad, s.alt_m] for s in station_bank], dtype=np.float64)

    states_all = np.zeros((len(anchors), dyn.steps, 6), dtype=np.float64)
    meas_all = np.zeros((len(anchors), dyn.steps, n_stations, 4), dtype=np.float64)
    vis_all = np.zeros((len(anchors), dyn.steps, n_stations), dtype=np.float64)
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

    observation_id = np.zeros(len(anchors), dtype=np.int64)
    object_id = np.zeros(len(anchors), dtype=np.int64)
    object_bucket = np.empty(len(anchors), dtype="<U32")
    source_start = np.empty(len(anchors), dtype="<U32")
    source_end = np.empty(len(anchors), dtype="<U32")
    anchor_station_name = np.empty(len(anchors), dtype="<U96")
    anchor_alignment_offset_s = np.zeros(len(anchors), dtype=np.float64)

    for traj_idx, anchor in enumerate(anchors):
        anchor_mid_abs = 0.5 * (
            _datetime_to_unix(_parse_iso8601_z(anchor.start))
            + _datetime_to_unix(_parse_iso8601_z(anchor.end))
        )
        trajectory_start_unix = anchor_mid_abs - 0.5 * trajectory_duration_s
        abs_times = trajectory_start_unix + times
        states = _propagate_tle_window(anchor, abs_times)
        states_all[traj_idx] = states

        desc = describe_public_observation(anchor)
        observation_id[traj_idx] = anchor.observation_id
        object_id[traj_idx] = anchor.norad_cat_id
        object_bucket[traj_idx] = str(desc["bucket"])
        source_start[traj_idx] = anchor.start
        source_end[traj_idx] = anchor.end
        anchor_station_name[traj_idx] = anchor.station_name

        station_bias = rng.normal(0.0, bias_std, size=(n_stations, 4))
        station_clock_bias = rng.normal(0.0, cfg.measurement_noise.clock_bias_std_s, size=n_stations)
        station_intervals = _station_intervals_for_object(
            filtered,
            norad_cat_id=anchor.norad_cat_id,
            station_index_by_name=station_index_by_name,
            trajectory_start_unix=trajectory_start_unix,
        )
        anchor_idx = station_index_by_name[anchor.station_name]
        anchor_start_rel = _datetime_to_unix(_parse_iso8601_z(anchor.start)) - trajectory_start_unix
        anchor_end_rel = _datetime_to_unix(_parse_iso8601_z(anchor.end)) - trajectory_start_unix
        time_offset_s = _best_rotation_offset_s(
            states,
            times,
            station_bank[anchor_idx],
            anchor_start_rel,
            anchor_end_rel,
        )
        anchor_alignment_offset_s[traj_idx] = time_offset_s

        for t_idx, t_s in enumerate(times):
            for s_idx, station in enumerate(station_bank):
                intervals = station_intervals.get(s_idx, [])
                in_window = any(start <= t_s <= end for start, end in intervals)
                if not in_window:
                    continue
                t_eff = (
                    t_s
                    + time_offset_s
                    + station_clock_bias[s_idx]
                    + rng.normal(0.0, cfg.measurement_noise.clock_jitter_std_s)
                )
                z_true, _ = line_of_sight_measurement(states[t_idx], station, float(t_eff))
                z_true = z_true + station_bias[s_idx]
                z_true[1] = float(np.mod(z_true[1], 2.0 * np.pi))
                z_true[2] = float(np.clip(z_true[2], station.min_elevation_rad, 0.5 * np.pi))
                meas_true_all[traj_idx, t_idx, s_idx] = z_true
                if rng.uniform() < cfg.measurement_noise.random_dropout_prob:
                    continue
                eps = rng.normal(0.0, noise_std)
                if rng.uniform() < cfg.measurement_noise.outlier_prob:
                    eps *= cfg.measurement_noise.outlier_scale
                z_noisy = z_true + eps
                z_noisy[1] = float(np.mod(z_noisy[1], 2.0 * np.pi))
                z_noisy[2] = float(np.clip(z_noisy[2], station.min_elevation_rad, 0.5 * np.pi))
                meas_all[traj_idx, t_idx, s_idx] = z_noisy
                vis_all[traj_idx, t_idx, s_idx] = 1.0

    return {
        "states": states_all,
        "measurements": meas_all,
        "measurements_true": meas_true_all,
        "visibility": vis_all,
        # Each trajectory's measurements are generated against an
        # Earth-rotation phase of ``t_s + anchor_alignment_offset_s[traj]``
        # (see ``time_offset_s`` above). Return that same phase base so
        # downstream EKF/UKF/WLS evaluate ``line_of_sight_measurement`` at the
        # geometry the measurements were produced from. A constant per-trajectory
        # offset preserves every per-step ``dt`` used for dynamics propagation.
        "times": times[None, :] + anchor_alignment_offset_s[:, None],
        "station_ecef": station_ecef[None, ...],
        "station_llh": station_llh[None, ...],
        "station_name": np.array([station.name for station in station_bank], dtype="<U96")[None, ...],
        "source_observation_id": observation_id,
        "source_norad_cat_id": object_id,
        "source_bucket": object_bucket,
        "source_start": source_start,
        "source_end": source_end,
        "anchor_station_name": anchor_station_name,
        "anchor_alignment_offset_s": anchor_alignment_offset_s,
        "source_slice_offset": np.full(len(anchors), int(slice_summary["slice_offset"]), dtype=np.int64),
        "source_slice_pool_count": np.full(len(anchors), int(slice_summary["source_pool_count"]), dtype=np.int64),
        "source_type": np.array(["satnogs_observation_replay"] * len(anchors), dtype="<U32"),
    }
