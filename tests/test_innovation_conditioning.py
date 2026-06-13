import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.coordinates import StationGeometry, line_of_sight_measurement
from gnn_state_estimation.innovation import compute_innovation_features
from gnn_state_estimation.models import TemporalGraphEstimator


class InnovationConditioningTests(unittest.TestCase):
    def test_innovation_features_are_zero_for_consistent_measurement(self) -> None:
        station = StationGeometry(
            name="TestStation",
            lat_deg=35.0,
            lon_deg=-120.0,
            alt_m=100.0,
            min_elevation_deg=-90.0,
        )
        state = np.array([7_000_000.0, 1_100_000.0, 1_500_000.0, -900.0, 7_300.0, 800.0], dtype=np.float64)
        t_s = np.array([[1234.5]], dtype=np.float64)
        meas, visible = line_of_sight_measurement(state, station, float(t_s[0, 0]))

        features = compute_innovation_features(
            prior_states=state.reshape(1, 1, 6),
            measurements=meas.reshape(1, 1, 1, 4),
            visibility=np.array([[[1.0 if visible else 0.0]]], dtype=np.float64),
            times_s=t_s,
            stations=(station,),
            meas_std_vector=np.array([30.0, np.deg2rad(0.02), np.deg2rad(0.02), 0.08], dtype=np.float64),
        )

        self.assertEqual(features.shape, (1, 1, 1, 6))
        np.testing.assert_allclose(features[0, 0, 0, :5], np.zeros(5, dtype=np.float32), atol=1e-6)
        self.assertEqual(float(features[0, 0, 0, 5]), 1.0)

    def test_model_forward_accepts_innovation_features(self) -> None:
        torch.manual_seed(0)
        model = TemporalGraphEstimator(
            hidden_dim=16,
            gnn_layers=1,
            gru_layers=1,
            dropout=0.0,
            use_ekf_prior=True,
            use_innovation_features=True,
            use_context_budget=True,
            use_dual_prior_fusion=True,
        )
        model.eval()

        batch_size, window_size, n_stations = 2, 4, 3
        out = model(
            measurements=torch.randn(batch_size, window_size, n_stations, 4),
            visibility=torch.ones(batch_size, window_size, n_stations, 1),
            station_xyz=torch.randn(batch_size, n_stations, 3),
            ekf_prior=torch.randn(batch_size, 6),
            secondary_prior=torch.randn(batch_size, 6),
            innovation_features=torch.randn(batch_size, window_size, n_stations, 6),
        )

        self.assertEqual(tuple(out["state"].shape), (batch_size, 6))
        self.assertEqual(tuple(out["logvar"].shape), (batch_size, 6))
        self.assertEqual(tuple(out["budget"].shape), (batch_size, 1))
        self.assertIn("context_stats", out)
        self.assertIn("prior_gate", out)


if __name__ == "__main__":
    unittest.main()
