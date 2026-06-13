import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.coordinates import StationGeometry
from gnn_state_estimation.dataset import compute_prior_bank_stats
from gnn_state_estimation.evaluation import run_model_inference, run_model_inference_batched
from gnn_state_estimation.models import TemporalGraphEstimator
from gnn_state_estimation.observability import (
    OBSERVABILITY_CONTEXT_DIM,
    compute_observability_context_features,
)
from gnn_state_estimation.semireal import filter_tle_catalog, load_tle_catalog
from gnn_state_estimation.utils.runtime import (
    build_run_manifest,
    duration_metadata,
    resolve_device,
    sha256_source_snapshot,
)


class PriorBankAndRuntimeTests(unittest.TestCase):
    def test_prior_bank_stats_are_scaled_and_finite(self) -> None:
        ekf = np.ones((2, 3, 6), dtype=np.float64) * np.array([7.0e6, 7.1e6, 7.2e6, 7500.0, 10.0, -20.0])
        ukf = ekf + np.array([100.0, -50.0, 30.0, 0.2, -0.1, 0.15], dtype=np.float64)
        aukf = ekf + np.array([-80.0, 25.0, -10.0, -0.15, 0.05, -0.2], dtype=np.float64)
        stats = compute_prior_bank_stats(ekf, ukf, aukf)
        self.assertEqual(stats.shape, (2, 3, 18))
        self.assertTrue(np.isfinite(stats).all())
        self.assertLess(float(np.max(np.abs(stats))), 1.0)

    def test_observability_context_features_are_prior_based_and_finite(self) -> None:
        station = StationGeometry(
            name="TestStation",
            lat_deg=35.0,
            lon_deg=-120.0,
            alt_m=100.0,
            min_elevation_deg=-90.0,
        )
        prior = np.array([[[7_000_000.0, 1_100_000.0, 1_500_000.0, -900.0, 7_300.0, 800.0]]], dtype=np.float64)
        features = compute_observability_context_features(
            prior_states=prior,
            visibility=np.ones((1, 1, 1), dtype=np.float64),
            times_s=np.array([[1234.5]], dtype=np.float64),
            stations=(station,),
            meas_std_vector=np.array([30.0, np.deg2rad(0.02), np.deg2rad(0.02), 0.08], dtype=np.float64),
        )
        self.assertEqual(features.shape, (1, 1, OBSERVABILITY_CONTEXT_DIM))
        self.assertTrue(np.isfinite(features).all())

    def test_multi_prior_forward_outputs_expected_debug_heads(self) -> None:
        torch.manual_seed(0)
        model = TemporalGraphEstimator(
            hidden_dim=16,
            gnn_layers=1,
            gru_layers=1,
            dropout=0.0,
            use_ekf_prior=True,
            use_graph=True,
            use_innovation_features=True,
            use_context_budget=True,
            use_prior_bank_fusion=True,
            prior_bank_size=3,
            prior_stats_dim=18,
            predict_noise_scale=True,
        )
        model.eval()

        batch_size, window_size, n_stations = 2, 4, 3
        out = model(
            measurements=torch.randn(batch_size, window_size, n_stations, 4),
            visibility=torch.ones(batch_size, window_size, n_stations, 1),
            station_xyz=torch.randn(batch_size, n_stations, 3),
            innovation_features=torch.randn(batch_size, window_size, n_stations, 6),
            prior_bank=torch.randn(batch_size, 3, 6),
            prior_bank_stats=torch.randn(batch_size, 18),
        )

        self.assertEqual(tuple(out["state"].shape), (batch_size, 6))
        self.assertEqual(tuple(out["fusion_weights"].shape), (batch_size, 6, 3))
        self.assertEqual(tuple(out["noise_scale"].shape), (batch_size, 4))
        sums = out["fusion_weights"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_local_no_graph_layers_match_graph_layer_capacity(self) -> None:
        torch.manual_seed(0)
        graph_model = TemporalGraphEstimator(
            hidden_dim=16,
            gnn_layers=2,
            gru_layers=1,
            dropout=0.0,
            use_graph=True,
        )
        local_model = TemporalGraphEstimator(
            hidden_dim=16,
            gnn_layers=2,
            gru_layers=1,
            dropout=0.0,
            use_graph=False,
            use_local_layers_when_no_graph=True,
        )
        graph_stack_params = sum(p.numel() for name, p in graph_model.named_parameters() if name.startswith("graph_layers."))
        local_stack_params = sum(p.numel() for name, p in local_model.named_parameters() if name.startswith("local_layers."))
        self.assertEqual(graph_stack_params, local_stack_params)
        self.assertEqual(len(local_model.graph_layers), 0)

        out = local_model(
            measurements=torch.randn(2, 4, 3, 4),
            visibility=torch.ones(2, 4, 3, 1),
            station_xyz=torch.randn(2, 3, 3),
        )
        self.assertEqual(tuple(out["state"].shape), (2, 6))

    def test_batched_inference_matches_single_trajectory_loop(self) -> None:
        torch.manual_seed(0)
        model = TemporalGraphEstimator(
            hidden_dim=16,
            gnn_layers=1,
            gru_layers=1,
            dropout=0.0,
            use_graph=False,
            use_local_layers_when_no_graph=True,
        )
        model.eval()
        rng = np.random.default_rng(123)
        states = rng.normal(size=(3, 6, 6)).astype(np.float64)
        measurements = rng.normal(size=(3, 6, 2, 4)).astype(np.float64)
        visibility = np.ones((3, 6, 2), dtype=np.float64)
        station_ecef = rng.normal(size=(2, 3)).astype(np.float64)

        loop_pred = run_model_inference(
            model,
            states=states,
            measurements=measurements,
            visibility=visibility,
            station_ecef=station_ecef,
            window_size=3,
        )
        batch_pred = run_model_inference_batched(
            model,
            states=states,
            measurements=measurements,
            visibility=visibility,
            station_ecef=station_ecef,
            window_size=3,
            batch_size=2,
        )
        mask = np.isfinite(loop_pred)
        self.assertTrue(np.allclose(loop_pred[mask], batch_pred[mask], atol=1e-6))
        self.assertTrue(np.array_equal(np.isnan(loop_pred), np.isnan(batch_pred)))

    def test_observability_context_forward_accepts_augmented_prior_stats(self) -> None:
        torch.manual_seed(0)
        model = TemporalGraphEstimator(
            hidden_dim=16,
            gnn_layers=1,
            gru_layers=1,
            dropout=0.0,
            use_ekf_prior=True,
            use_graph=True,
            use_innovation_features=True,
            use_context_budget=True,
            use_prior_bank_fusion=True,
            prior_bank_size=3,
            prior_stats_dim=18 + OBSERVABILITY_CONTEXT_DIM,
            use_observability_context=True,
            predict_noise_scale=True,
        )
        model.eval()

        batch_size, window_size, n_stations = 2, 4, 3
        out = model(
            measurements=torch.randn(batch_size, window_size, n_stations, 4),
            visibility=torch.ones(batch_size, window_size, n_stations, 1),
            station_xyz=torch.randn(batch_size, n_stations, 3),
            innovation_features=torch.randn(batch_size, window_size, n_stations, 6),
            prior_bank=torch.randn(batch_size, 3, 6),
            prior_bank_stats=torch.randn(batch_size, 18 + OBSERVABILITY_CONTEXT_DIM),
        )

        self.assertEqual(tuple(out["state"].shape), (batch_size, 6))
        self.assertEqual(tuple(out["fusion_weights"].shape), (batch_size, 6, 3))
        self.assertTrue(model.use_observability_context)

    def test_resolve_device_rejects_missing_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")
            self.assertEqual(str(resolve_device("cpu")), "cpu")

    def test_tle_filter_removes_non_leo_entries(self) -> None:
        catalog = load_tle_catalog(Path(__file__).resolve().parents[1] / "configs" / "archived_tles.json")
        filtered = filter_tle_catalog(
            catalog,
            min_altitude_km=300.0,
            max_altitude_km=2000.0,
            max_eccentricity=0.05,
            min_mean_motion_rev_per_day=10.0,
        )
        names = {entry.name for entry in filtered}
        self.assertIn("ISS (ZARYA)", names)
        self.assertIn("CSS (TIANHE)", names)
        self.assertNotIn("METEOSAT-9 (MSG-2)", names)

    def test_source_snapshot_hash_exists_for_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        digest = sha256_source_snapshot(repo_root)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)

    def test_run_manifest_can_record_timing(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        out_path = repo_root / "results" / "smoke_v2" / "timing_manifest_test.json"
        timing = duration_metadata(start_perf_counter=0.0, started_at_utc="2026-04-13T00:00:00Z")
        manifest = build_run_manifest(
            command=["pytest"],
            config_text="seed: 1\n",
            config_path=repo_root / "configs" / "experiment.yaml",
            output_path=out_path,
            device=torch.device("cpu"),
            seed=1,
            repo_root=repo_root,
            timing=timing,
        )
        self.assertIn("timing", manifest)
        self.assertEqual(manifest["timing"]["started_at_utc"], "2026-04-13T00:00:00Z")
        self.assertIn("duration_sec", manifest["timing"])


if __name__ == "__main__":
    unittest.main()
