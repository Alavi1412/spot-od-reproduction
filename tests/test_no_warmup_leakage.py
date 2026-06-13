import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.evaluation import run_model_inference
from gnn_state_estimation.models import TemporalGraphEstimator


class WarmupLeakageTests(unittest.TestCase):
    def test_inference_does_not_copy_ground_truth_in_warmup(self) -> None:
        torch.manual_seed(0)
        model = TemporalGraphEstimator(
            hidden_dim=16,
            gnn_layers=1,
            gru_layers=1,
            dropout=0.0,
            use_ekf_prior=False,
        )
        model.eval()

        n_traj, t_steps, n_stations = 2, 8, 3
        window_size = 4
        states = np.random.randn(n_traj, t_steps, 6).astype(np.float64)
        measurements = np.random.randn(n_traj, t_steps, n_stations, 4).astype(np.float64)
        visibility = np.ones((n_traj, t_steps, n_stations), dtype=np.float64)
        station_ecef = np.random.randn(n_stations, 3).astype(np.float64) * 6_300_000.0

        pred = run_model_inference(
            model=model,
            states=states,
            measurements=measurements,
            visibility=visibility,
            station_ecef=station_ecef,
            window_size=window_size,
        )

        warmup = pred[:, : window_size - 1]
        forecast = pred[:, window_size - 1 :]

        self.assertTrue(np.isnan(warmup).all())
        self.assertTrue(np.isfinite(forecast).all())


if __name__ == "__main__":
    unittest.main()
