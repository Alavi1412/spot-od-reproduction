import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.public_data import (
    PublicCatalogEntry,
    PublicStationEntry,
    apply_public_station_selection,
    filter_public_catalog,
    generate_public_catalog_replay_dataset,
    select_public_ground_stations,
)
from gnn_state_estimation.simulation import parse_dataset_config


def sample_catalog_entry(
    *,
    name: str,
    object_id: str,
    norad_cat_id: int,
    inclination_deg: float,
    mean_motion_rev_per_day: float,
    eccentricity: float,
) -> PublicCatalogEntry:
    return PublicCatalogEntry(
        fields={
            "OBJECT_NAME": name,
            "OBJECT_ID": object_id,
            "NORAD_CAT_ID": norad_cat_id,
            "CLASSIFICATION_TYPE": "U",
            "EPHEMERIS_TYPE": 0,
            "ELEMENT_SET_NO": 1,
            "REV_AT_EPOCH": 1,
            "EPOCH": "2026-04-15T20:55:53.384160",
            "ARG_OF_PERICENTER": 15.0,
            "BSTAR": 0.0001,
            "ECCENTRICITY": eccentricity,
            "INCLINATION": inclination_deg,
            "MEAN_ANOMALY": 25.0,
            "MEAN_MOTION": mean_motion_rev_per_day,
            "MEAN_MOTION_DOT": 0.0,
            "MEAN_MOTION_DDOT": 0.0,
            "RA_OF_ASC_NODE": 30.0,
        }
    )


class PublicDataTests(unittest.TestCase):
    def test_public_catalog_filter_removes_non_leo_entries(self) -> None:
        catalog = (
            sample_catalog_entry(
                name="LEO-1",
                object_id="2026-001A",
                norad_cat_id=10001,
                inclination_deg=51.6,
                mean_motion_rev_per_day=15.4,
                eccentricity=0.001,
            ),
            sample_catalog_entry(
                name="GEO-1",
                object_id="2026-002A",
                norad_cat_id=10002,
                inclination_deg=10.0,
                mean_motion_rev_per_day=1.0,
                eccentricity=0.0001,
            ),
        )
        filtered = filter_public_catalog(
            catalog,
            min_altitude_km=300.0,
            max_altitude_km=2000.0,
            max_eccentricity=0.05,
            min_mean_motion_rev_per_day=10.0,
        )
        self.assertEqual([entry.object_name for entry in filtered], ["LEO-1"])

    def test_station_selection_prefers_filtered_and_geographically_diverse_rows(self) -> None:
        stations = (
            PublicStationEntry(1, "Alpha", 10.0, 10.0, 100.0, 8.0, 5000, 85.0, "Online"),
            PublicStationEntry(2, "Beta", -20.0, 140.0, 50.0, 10.0, 7000, 80.0, "Online"),
            PublicStationEntry(3, "Gamma", 55.0, -120.0, 20.0, 12.0, 4000, 78.0, "Online"),
            PublicStationEntry(4, "Offline", 0.0, 0.0, 0.0, 8.0, 10000, 99.0, "Offline"),
        )
        selected = select_public_ground_stations(
            stations,
            count=3,
            min_success_rate=70.0,
            min_observations=1000,
            require_online=True,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual({station.name for station in selected}, {"Alpha", "Beta", "Gamma"})

    def test_apply_public_station_selection_overrides_station_geometry(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
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
        scenario_cfg = {
            "station_snapshot_path": str(repo_root / "configs" / "public_satnogs_stations_snapshot.json"),
            "station_filters": {
                "count": 4,
                "min_success_rate": 70.0,
                "min_observations": 1000,
                "require_online": True,
            },
        }
        out_cfg, summary = apply_public_station_selection(sim_cfg, scenario_cfg)
        self.assertEqual(len(out_cfg["stations"]), 4)
        self.assertEqual(summary["selected_station_count"], 4)
        self.assertNotIn("Legacy", summary["selected_station_names"])

    def test_public_catalog_replay_dataset_smoke(self) -> None:
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
                {"name": "Alpha", "lat_deg": 35.0, "lon_deg": -120.0, "alt_m": 50.0, "min_elevation_deg": 8.0},
                {"name": "Beta", "lat_deg": -30.0, "lon_deg": 120.0, "alt_m": 60.0, "min_elevation_deg": 8.0},
            ],
        }
        cfg = parse_dataset_config(sim_cfg)
        catalog = (
            sample_catalog_entry(
                name="LEO-1",
                object_id="2026-001A",
                norad_cat_id=10001,
                inclination_deg=51.6,
                mean_motion_rev_per_day=15.4,
                eccentricity=0.001,
            ),
            sample_catalog_entry(
                name="LEO-2",
                object_id="2026-003A",
                norad_cat_id=10003,
                inclination_deg=97.4,
                mean_motion_rev_per_day=14.9,
                eccentricity=0.002,
            ),
        )
        dataset = generate_public_catalog_replay_dataset(
            cfg,
            catalog=catalog,
            num_trajectories=2,
            seed=7,
            catalog_filters={
                "min_altitude_km": 300.0,
                "max_altitude_km": 2000.0,
                "max_eccentricity": 0.05,
                "min_mean_motion_rev_per_day": 10.0,
            },
            sampling_strategy="stratified_inclination",
        )
        self.assertEqual(dataset["states"].shape, (2, 6, 6))
        self.assertEqual(dataset["visibility"].shape[:3], (2, 6, 2))
        self.assertTrue(np.all(dataset["source_type"] == "public_catalog_replay"))
        self.assertEqual(dataset["station_name"].shape, (1, 2))
        self.assertEqual(dataset["object_bucket"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
