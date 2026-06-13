import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.coordinates import StationGeometry, line_of_sight_measurement
from gnn_state_estimation.observation_replay import (
    apply_public_observation_station_bank,
    filter_public_observations,
    generate_public_observation_replay_dataset,
    load_public_observation_snapshot,
    _slice_observations,
)
from gnn_state_estimation.simulation import parse_dataset_config


OBS_ROWS = [
    {
        "id": 13821681,
        "start": "2026-04-15T22:00:40Z",
        "end": "2026-04-15T22:06:16Z",
        "ground_station": 100,
        "norad_cat_id": 25544,
        "station_name": "Alpha",
        "station_lat": 34.0,
        "station_lng": -118.0,
        "station_alt": 250.0,
        "status": "good",
        "vetted_status": "good",
        "tle1": "1 25544U 98067A   26105.45127315  .00016566  00000+0  30117-3 0  9994",
        "tle2": "2 25544  51.6330  37.7856 0003392 107.0047 288.9815 15.49938272552942",
    },
    {
        "id": 13821682,
        "start": "2026-04-15T22:04:00Z",
        "end": "2026-04-15T22:09:20Z",
        "ground_station": 101,
        "norad_cat_id": 25544,
        "station_name": "Beta",
        "station_lat": 40.0,
        "station_lng": -3.5,
        "station_alt": 650.0,
        "status": "good",
        "vetted_status": "good",
        "tle1": "1 25544U 98067A   26105.45127315  .00016566  00000+0  30117-3 0  9994",
        "tle2": "2 25544  51.6330  37.7856 0003392 107.0047 288.9815 15.49938272552942",
    },
    {
        "id": 13821683,
        "start": "2026-04-15T22:02:00Z",
        "end": "2026-04-15T22:08:00Z",
        "ground_station": 102,
        "norad_cat_id": 46907,
        "station_name": "Gamma",
        "station_lat": -33.4,
        "station_lng": -70.6,
        "station_alt": 500.0,
        "status": "good",
        "vetted_status": "good",
        "tle1": "1 46907U 20081B   26105.08909339  .00006611  00000+0  43262-3 0  9998",
        "tle2": "2 46907  37.0012 342.6294 0007856 139.4645 220.6865 15.00414065294990",
    },
]


class ObservationReplayTests(unittest.TestCase):
    def test_load_and_filter_public_observation_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "obs.json"
            path.write_text(json.dumps({"observations": OBS_ROWS}), encoding="utf-8")
            rows = load_public_observation_snapshot(path)
            self.assertEqual(len(rows), 3)
            filtered = filter_public_observations(
                rows,
                require_good_status=True,
                min_altitude_km=350.0,
                max_altitude_km=1500.0,
                min_mean_motion_rev_per_day=10.0,
            )
            self.assertGreaterEqual(len(filtered), 2)

    def test_apply_public_observation_station_bank_overrides_station_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = Path(tmpdir) / "obs.json"
            obs_path.write_text(json.dumps({"observations": OBS_ROWS}), encoding="utf-8")
            sim_cfg = {
                "orbit_sampling": {
                    "altitude_min_km": 450.0,
                    "altitude_max_km": 900.0,
                    "eccentricity_min": 0.0,
                    "eccentricity_max": 0.02,
                    "inclination_min_deg": 20.0,
                    "inclination_max_deg": 98.0,
                },
                "dynamics": {
                    "dt_s": 20.0,
                    "steps": 8,
                    "ballistic_coeff_m2_per_kg": 0.018,
                    "process_noise_std": 0.0,
                    "drag_rho_ref": 4.0e-11,
                    "drag_h_ref_m": 400000.0,
                    "drag_scale_height_m": 60000.0,
                    "enable_third_body": False,
                    "enable_srp": False,
                },
                "measurement_noise": {
                    "range_std_m": 30.0,
                    "az_std_deg": 0.02,
                    "el_std_deg": 0.02,
                    "range_rate_std_mps": 0.08,
                    "outlier_prob": 0.0,
                    "outlier_scale": 6.0,
                },
                "stations": [{"name": "Legacy", "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0}],
            }
            out_cfg, summary = apply_public_observation_station_bank(
                sim_cfg,
                {
                    "observation_snapshot_path": str(obs_path),
                    "observation_filters": {
                        "require_good_status": True,
                        "min_altitude_km": 350.0,
                        "max_altitude_km": 1500.0,
                        "min_mean_motion_rev_per_day": 10.0,
                    },
                },
            )
            self.assertEqual(len(out_cfg["stations"]), 3)
            self.assertEqual(summary["selected_station_count"], 3)
            self.assertEqual(summary["selected_observation_count"], 3)
            self.assertNotIn("Legacy", summary["selected_station_names"])

    def test_apply_public_observation_station_bank_honors_station_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = Path(tmpdir) / "obs.json"
            obs_path.write_text(json.dumps({"observations": OBS_ROWS}), encoding="utf-8")
            sim_cfg = {
                "orbit_sampling": {
                    "altitude_min_km": 450.0,
                    "altitude_max_km": 900.0,
                    "eccentricity_min": 0.0,
                    "eccentricity_max": 0.02,
                    "inclination_min_deg": 20.0,
                    "inclination_max_deg": 98.0,
                },
                "dynamics": {
                    "dt_s": 20.0,
                    "steps": 8,
                    "ballistic_coeff_m2_per_kg": 0.018,
                    "process_noise_std": 0.0,
                    "drag_rho_ref": 4.0e-11,
                    "drag_h_ref_m": 400000.0,
                    "drag_scale_height_m": 60000.0,
                    "enable_third_body": False,
                    "enable_srp": False,
                },
                "measurement_noise": {
                    "range_std_m": 30.0,
                    "az_std_deg": 0.02,
                    "el_std_deg": 0.02,
                    "range_rate_std_mps": 0.08,
                    "outlier_prob": 0.0,
                    "outlier_scale": 6.0,
                },
                "stations": [{"name": "Legacy", "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0}],
            }
            out_cfg, summary = apply_public_observation_station_bank(
                sim_cfg,
                {
                    "observation_snapshot_path": str(obs_path),
                    "observation_filters": {
                        "require_good_status": True,
                        "min_altitude_km": 350.0,
                        "max_altitude_km": 1500.0,
                        "min_mean_motion_rev_per_day": 10.0,
                    },
                    "station_filters": {"count": 2},
                },
            )
            self.assertEqual(len(out_cfg["stations"]), 2)
            self.assertEqual(summary["selected_station_count"], 2)
            self.assertEqual(summary["selected_observation_count"], 2)

    def test_public_observation_replay_dataset_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = Path(tmpdir) / "obs.json"
            obs_path.write_text(json.dumps({"observations": OBS_ROWS}), encoding="utf-8")
            sim_cfg = {
                "orbit_sampling": {
                    "altitude_min_km": 450.0,
                    "altitude_max_km": 900.0,
                    "eccentricity_min": 0.0,
                    "eccentricity_max": 0.02,
                    "inclination_min_deg": 20.0,
                    "inclination_max_deg": 98.0,
                },
                "dynamics": {
                    "dt_s": 20.0,
                    "steps": 6,
                    "ballistic_coeff_m2_per_kg": 0.018,
                    "process_noise_std": 0.0,
                    "drag_rho_ref": 4.0e-11,
                    "drag_h_ref_m": 400000.0,
                    "drag_scale_height_m": 60000.0,
                    "enable_third_body": False,
                    "enable_srp": False,
                },
                "measurement_noise": {
                    "range_std_m": 30.0,
                    "az_std_deg": 0.02,
                    "el_std_deg": 0.02,
                    "range_rate_std_mps": 0.08,
                    "outlier_prob": 0.0,
                    "outlier_scale": 6.0,
                    "random_dropout_prob": 0.0,
                    "range_bias_std_m": 0.0,
                    "az_bias_std_deg": 0.0,
                    "el_bias_std_deg": 0.0,
                    "range_rate_bias_std_mps": 0.0,
                    "clock_bias_std_s": 0.0,
                    "clock_jitter_std_s": 0.0,
                },
                "stations": [
                    {"name": "Legacy", "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "min_elevation_deg": 8.0},
                ],
            }
            out_cfg, _ = apply_public_observation_station_bank(
                sim_cfg,
                {
                    "observation_snapshot_path": str(obs_path),
                    "observation_filters": {
                        "require_good_status": True,
                        "min_altitude_km": 350.0,
                        "max_altitude_km": 1500.0,
                        "min_mean_motion_rev_per_day": 10.0,
                    },
                },
            )
            cfg = parse_dataset_config(out_cfg)
            dataset = generate_public_observation_replay_dataset(
                cfg,
                observations=load_public_observation_snapshot(obs_path),
                num_trajectories=3,
                seed=11,
                observation_filters={
                    "require_good_status": True,
                    "min_altitude_km": 350.0,
                    "max_altitude_km": 1500.0,
                    "min_mean_motion_rev_per_day": 10.0,
                    "station_count": 3,
                },
            )
            self.assertEqual(dataset["states"].shape, (3, 6, 6))
            self.assertEqual(dataset["station_name"].shape[1], 3)
            self.assertEqual(dataset["visibility"].shape[:3], (3, 6, 3))
            self.assertTrue(np.all(dataset["source_type"] == "satnogs_observation_replay"))
            self.assertIn("anchor_alignment_offset_s", dataset)
            self.assertEqual(len(set(dataset["source_observation_id"].tolist())), 3)

    def test_replay_times_carry_station_rotation_phase(self) -> None:
        """Returned ``times`` must reproduce the Earth-rotation phase the
        measurements were generated against.

        With all measurement noise / bias / clock terms zeroed, each
        ``measurements_true`` entry equals the normalized line-of-sight
        geometry at ``t_s + anchor_alignment_offset_s[traj]``. Recomputing the
        line-of-sight at the *returned* per-trajectory times must therefore
        match exactly. Under the old behaviour (``np.repeat`` of the relative
        ``t_s``) the alignment offset is stripped, so the recomputed geometry
        is evaluated a full station-rotation phase away and diverges.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = Path(tmpdir) / "obs.json"
            obs_path.write_text(json.dumps({"observations": OBS_ROWS}), encoding="utf-8")
            sim_cfg = {
                "orbit_sampling": {
                    "altitude_min_km": 450.0,
                    "altitude_max_km": 900.0,
                    "eccentricity_min": 0.0,
                    "eccentricity_max": 0.02,
                    "inclination_min_deg": 20.0,
                    "inclination_max_deg": 98.0,
                },
                "dynamics": {
                    "dt_s": 20.0,
                    "steps": 6,
                    "ballistic_coeff_m2_per_kg": 0.018,
                    "process_noise_std": 0.0,
                    "drag_rho_ref": 4.0e-11,
                    "drag_h_ref_m": 400000.0,
                    "drag_scale_height_m": 60000.0,
                    "enable_third_body": False,
                    "enable_srp": False,
                },
                "measurement_noise": {
                    "range_std_m": 30.0,
                    "az_std_deg": 0.02,
                    "el_std_deg": 0.02,
                    "range_rate_std_mps": 0.08,
                    "outlier_prob": 0.0,
                    "outlier_scale": 6.0,
                    "random_dropout_prob": 0.0,
                    "range_bias_std_m": 0.0,
                    "az_bias_std_deg": 0.0,
                    "el_bias_std_deg": 0.0,
                    "range_rate_bias_std_mps": 0.0,
                    "clock_bias_std_s": 0.0,
                    "clock_jitter_std_s": 0.0,
                },
                "stations": [
                    {"name": "Legacy", "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "min_elevation_deg": 8.0},
                ],
            }
            out_cfg, _ = apply_public_observation_station_bank(
                sim_cfg,
                {
                    "observation_snapshot_path": str(obs_path),
                    "observation_filters": {
                        "require_good_status": True,
                        "min_altitude_km": 350.0,
                        "max_altitude_km": 1500.0,
                        "min_mean_motion_rev_per_day": 10.0,
                    },
                },
            )
            cfg = parse_dataset_config(out_cfg)
            dataset = generate_public_observation_replay_dataset(
                cfg,
                observations=load_public_observation_snapshot(obs_path),
                num_trajectories=3,
                seed=11,
                observation_filters={
                    "require_good_status": True,
                    "min_altitude_km": 350.0,
                    "max_altitude_km": 1500.0,
                    "min_mean_motion_rev_per_day": 10.0,
                    "station_count": 3,
                },
            )

            times = dataset["times"]
            states = dataset["states"]
            vis = dataset["visibility"]
            meas_true = dataset["measurements_true"]
            llh = dataset["station_llh"][0]
            names = dataset["station_name"][0]
            offsets = dataset["anchor_alignment_offset_s"]

            # The returned times must encode a non-trivial per-trajectory
            # phase, otherwise the test could pass vacuously against the old
            # behaviour.
            self.assertGreater(float(np.max(np.abs(offsets))), 1.0)
            np.testing.assert_allclose(
                times,
                np.arange(cfg.dynamics.steps, dtype=np.float64)[None, :] * cfg.dynamics.dt_s
                + offsets[:, None],
                rtol=0.0,
                atol=1e-9,
            )

            stations = [
                StationGeometry(
                    name=str(names[s_idx]),
                    lat_deg=float(np.rad2deg(llh[s_idx, 0])),
                    lon_deg=float(np.rad2deg(llh[s_idx, 1])),
                    alt_m=float(llh[s_idx, 2]),
                    min_elevation_deg=8.0,
                )
                for s_idx in range(llh.shape[0])
            ]

            checked = 0
            for traj in range(states.shape[0]):
                for t_idx in range(states.shape[1]):
                    for s_idx, station in enumerate(stations):
                        if vis[traj, t_idx, s_idx] < 0.5:
                            continue
                        z_chk, _ = line_of_sight_measurement(
                            states[traj, t_idx], station, float(times[traj, t_idx])
                        )
                        z_chk[1] = float(np.mod(z_chk[1], 2.0 * np.pi))
                        z_chk[2] = float(
                            np.clip(z_chk[2], station.min_elevation_rad, 0.5 * np.pi)
                        )
                        np.testing.assert_allclose(
                            z_chk,
                            meas_true[traj, t_idx, s_idx],
                            rtol=1e-6,
                            atol=1e-4,
                        )
                        checked += 1
            self.assertGreater(checked, 0, "no visible anchor steps were exercised")

    def test_source_slice_is_deterministic_and_reflected_in_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = Path(tmpdir) / "obs.json"
            obs_path.write_text(json.dumps({"observations": OBS_ROWS}), encoding="utf-8")
            rows = load_public_observation_snapshot(obs_path)
            filtered = filter_public_observations(
                rows,
                require_good_status=True,
                min_altitude_km=350.0,
                max_altitude_km=1500.0,
                min_mean_motion_rev_per_day=10.0,
            )
            sliced, meta = _slice_observations(filtered, source_slice={"offset": 1, "count": 2})
            self.assertEqual(len(sliced), 2)
            self.assertEqual(meta["source_pool_count"], len(filtered))
            self.assertEqual(meta["slice_offset"], 1)
            self.assertEqual(meta["slice_count"], 2)

    def test_station_bank_summary_reports_source_slice_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_path = Path(tmpdir) / "obs.json"
            obs_path.write_text(json.dumps({"observations": OBS_ROWS}), encoding="utf-8")
            sim_cfg = {
                "orbit_sampling": {
                    "altitude_min_km": 450.0,
                    "altitude_max_km": 900.0,
                    "eccentricity_min": 0.0,
                    "eccentricity_max": 0.02,
                    "inclination_min_deg": 20.0,
                    "inclination_max_deg": 98.0,
                },
                "dynamics": {
                    "dt_s": 20.0,
                    "steps": 8,
                    "ballistic_coeff_m2_per_kg": 0.018,
                    "process_noise_std": 0.0,
                    "drag_rho_ref": 4.0e-11,
                    "drag_h_ref_m": 400000.0,
                    "drag_scale_height_m": 60000.0,
                    "enable_third_body": False,
                    "enable_srp": False,
                },
                "measurement_noise": {
                    "range_std_m": 30.0,
                    "az_std_deg": 0.02,
                    "el_std_deg": 0.02,
                    "range_rate_std_mps": 0.08,
                    "outlier_prob": 0.0,
                    "outlier_scale": 6.0,
                },
                "stations": [{"name": "Legacy", "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0}],
            }
            _, summary = apply_public_observation_station_bank(
                sim_cfg,
                {
                    "observation_snapshot_path": str(obs_path),
                    "observation_filters": {
                        "require_good_status": True,
                        "min_altitude_km": 350.0,
                        "max_altitude_km": 1500.0,
                        "min_mean_motion_rev_per_day": 10.0,
                    },
                    "source_slice": {"offset": 0, "count": 2},
                },
            )
            self.assertEqual(summary["source_pool_count"], 3)
            self.assertEqual(summary["slice_observation_count"], 2)


if __name__ == "__main__":
    unittest.main()
