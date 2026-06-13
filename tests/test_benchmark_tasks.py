import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.benchmark_tasks import (
    binary_classification_metrics,
    build_selector_outputs,
    compute_trajectory_feature_frame,
    trajectory_diverged_mask,
)


class BenchmarkTaskTests(unittest.TestCase):
    def test_trajectory_diverged_mask_flags_invalid_and_large_errors(self) -> None:
        mask = trajectory_diverged_mask(
            np.array([1.0, np.nan, 2.0e8, 3.0]),
            np.array([1.0, 2.0, 3.0, 2.0e5]),
        )
        self.assertTrue(np.array_equal(mask, np.array([False, True, True, True])))

    def test_compute_trajectory_feature_frame_extracts_visibility_and_prior_gap(self) -> None:
        arrays = SimpleNamespace(
            visibility=np.array(
                [
                    [[1, 0], [0, 0], [1, 1]],
                    [[0, 0], [1, 0], [0, 1]],
                ],
                dtype=np.float64,
            ),
            innovation_features=np.array(
                [
                    [
                        [[0, 0, 0, 0, 1.0, 1.0], [0, 0, 0, 0, 5.0, 0.0]],
                        [[0, 0, 0, 0, 2.0, 0.0], [0, 0, 0, 0, 6.0, 0.0]],
                        [[0, 0, 0, 0, 3.0, 1.0], [0, 0, 0, 0, 7.0, 1.0]],
                    ],
                    [
                        [[0, 0, 0, 0, 4.0, 0.0], [0, 0, 0, 0, 8.0, 0.0]],
                        [[0, 0, 0, 0, 5.0, 1.0], [0, 0, 0, 0, 9.0, 0.0]],
                        [[0, 0, 0, 0, 6.0, 0.0], [0, 0, 0, 0, 10.0, 1.0]],
                    ],
                ],
                dtype=np.float64,
            ),
            ekf_prior=np.zeros((2, 3, 6), dtype=np.float64),
            ukf_prior=np.ones((2, 3, 6), dtype=np.float64),
            aukf_prior=np.full((2, 3, 6), 2.0, dtype=np.float64),
        )
        frame = compute_trajectory_feature_frame(
            arrays,
            scenario_name="satnogs_observation_replay_test",
            eval_start=1,
        )
        self.assertEqual(frame.shape[0], 2)
        self.assertAlmostEqual(float(frame.loc[0, "mean_visible_stations"]), 1.0)
        self.assertAlmostEqual(float(frame.loc[0, "fraction_zero_visibility"]), 0.5)
        self.assertAlmostEqual(float(frame.loc[0, "fraction_two_plus_visibility"]), 0.5)
        self.assertGreater(float(frame.loc[0, "mean_prior_gap_pos_m"]), 0.0)
        self.assertEqual(float(frame.loc[0, "is_public_observation"]), 1.0)
        self.assertEqual(float(frame.loc[0, "is_stress"]), 0.0)

    def test_binary_classification_metrics_reports_perfect_rank_separation(self) -> None:
        metrics = binary_classification_metrics(
            np.array([0, 0, 1, 1], dtype=np.float64),
            np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        )
        self.assertEqual(metrics["n_samples"], 4.0)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["auprc"], 1.0)
        self.assertLess(metrics["brier"], 0.05)

    def test_build_selector_outputs_preserves_zero_regret_oracle(self) -> None:
        feature_cols = {
            "mean_visible_stations": [0.1, 1.2],
            "fraction_zero_visibility": [0.9, 0.1],
            "fraction_one_visibility": [0.1, 0.4],
            "fraction_two_plus_visibility": [0.0, 0.5],
            "mean_innovation_energy": [1.0, 2.0],
            "max_innovation_energy": [2.0, 3.0],
            "mean_visibility_mismatch": [0.2, 0.1],
            "mean_prior_gap_pos_m": [100.0, 50.0],
            "max_prior_gap_pos_m": [120.0, 70.0],
            "mean_prior_gap_vel_mps": [0.5, 0.2],
            "is_public_catalog": [0.0, 0.0],
            "is_public_observation": [1.0, 1.0],
            "is_stress": [0.0, 1.0],
        }
        train_features = pd.DataFrame(
            {
                "scenario": ["test", "stress_test"],
                "traj_id": [0, 1],
                **feature_cols,
            }
        )
        eval_features = pd.DataFrame(
            {
                "scenario": ["satnogs_observation_replay_test", "satnogs_observation_replay_stress_test"],
                "traj_id": [0, 1],
                **feature_cols,
            }
        )
        train_stability = pd.DataFrame(
            [
                {"scenario": "test", "traj_id": 0, "method": "EKF", "traj_pos_rmse_m": 10.0, "is_stable": 1, "is_unstable": 0},
                {"scenario": "test", "traj_id": 0, "method": "AUKF", "traj_pos_rmse_m": 20.0, "is_stable": 1, "is_unstable": 0},
                {"scenario": "stress_test", "traj_id": 1, "method": "EKF", "traj_pos_rmse_m": 50.0, "is_stable": 1, "is_unstable": 0},
                {"scenario": "stress_test", "traj_id": 1, "method": "AUKF", "traj_pos_rmse_m": 15.0, "is_stable": 1, "is_unstable": 0},
            ]
        )
        eval_stability = pd.DataFrame(
            [
                {"scenario": "satnogs_observation_replay_test", "traj_id": 0, "method": "EKF", "traj_pos_rmse_m": 100.0, "is_stable": 1},
                {"scenario": "satnogs_observation_replay_test", "traj_id": 0, "method": "AUKF", "traj_pos_rmse_m": 90.0, "is_stable": 1},
                {"scenario": "satnogs_observation_replay_stress_test", "traj_id": 1, "method": "EKF", "traj_pos_rmse_m": 80.0, "is_stable": 1},
                {"scenario": "satnogs_observation_replay_stress_test", "traj_id": 1, "method": "AUKF", "traj_pos_rmse_m": 60.0, "is_stable": 1},
            ]
        )
        detail_df, summary_df = build_selector_outputs(
            train_features=train_features,
            train_stability=train_stability,
            eval_features=eval_features,
            eval_stability=eval_stability,
            selector_methods=("EKF", "AUKF"),
            divergence_penalty_m=1.0e6,
        )
        self.assertFalse(detail_df.empty)
        oracle_row = summary_df[
            (summary_df["selector"] == "Oracle stable selector") & (summary_df["scope"] == "combined")
        ].iloc[0]
        ekf_row = summary_df[
            (summary_df["selector"] == "Always EKF") & (summary_df["scope"] == "combined")
        ].iloc[0]
        self.assertEqual(float(oracle_row["mean_regret_m"]), 0.0)
        self.assertEqual(float(oracle_row["oracle_match_rate"]), 1.0)
        self.assertGreater(float(ekf_row["mean_regret_m"]), 0.0)


if __name__ == "__main__":
    unittest.main()
