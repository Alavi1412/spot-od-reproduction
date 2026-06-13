import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.guarded_selection import (
    GuardedSelectorConfig,
    build_cost_matrix,
    evaluate_selection,
    fit_guarded_selector,
    load_guarded_selector,
    save_guarded_selector,
    select_methods,
)


class GuardedSelectionTests(unittest.TestCase):
    def test_cost_predictor_selector_trains_and_round_trips(self) -> None:
        features = pd.DataFrame(
            {
                "scenario": ["nominal"] * 4 + ["stress"] * 4,
                "traj_id": list(range(4)) + list(range(4)),
                "is_stress": [0.0] * 4 + [1.0] * 4,
                "mean_visible_stations": [1.0, 1.2, 1.1, 1.3, 0.4, 0.5, 0.6, 0.4],
            }
        )
        rows = []
        for item in features.itertuples(index=False):
            method_a = 100.0 if item.scenario == "nominal" else 500.0
            method_b = 400.0 if item.scenario == "nominal" else 90.0
            rows.append(
                {
                    "scenario": item.scenario,
                    "traj_id": item.traj_id,
                    "method": "MethodA",
                    "traj_pos_rmse_m": method_a,
                    "traj_vel_rmse_mps": 1.0,
                }
            )
            rows.append(
                {
                    "scenario": item.scenario,
                    "traj_id": item.traj_id,
                    "method": "MethodB",
                    "traj_pos_rmse_m": method_b,
                    "traj_vel_rmse_mps": 1.0,
                }
            )
        traj = pd.DataFrame(rows)
        methods = ("MethodA", "MethodB")
        aligned, costs = build_cost_matrix(features, traj, methods=methods)
        selector = fit_guarded_selector(
            aligned,
            costs,
            methods=methods,
            feature_names=("is_stress", "mean_visible_stations"),
            config=GuardedSelectorConfig(epochs=80, val_fraction=0.0, hidden_dim=12, seed=7),
            device=torch.device("cpu"),
        )
        selected = select_methods(selector, features)
        self.assertEqual(selected.shape[0], features.shape[0])
        self.assertTrue(np.isfinite(selected["predicted_cost_m"].to_numpy(dtype=np.float64)).all())

        detail, summary = evaluate_selection(selected, traj)
        self.assertIn("aggregate_pos_rmse_m", summary.columns)
        self.assertEqual(detail["selected_diverged"].sum(), 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selector.pt"
            save_guarded_selector(selector, path)
            loaded = load_guarded_selector(path)
            reselected = select_methods(loaded, features)
            self.assertEqual(reselected.shape[0], selected.shape[0])


if __name__ == "__main__":
    unittest.main()
