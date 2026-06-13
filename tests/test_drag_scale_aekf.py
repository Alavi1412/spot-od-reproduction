"""Focused tests for the Drag-Scale Adaptive EKF (DSA-EKF), loop 45.

Covers ``run_drag_scale_aekf`` against the same small synthetic case used by
the EKF/AUKF/DMC tests, so the new multiplicative drag-scale channel is
exercised cheaply on every run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.coordinates import StationGeometry, line_of_sight_measurement
from gnn_state_estimation.filters import (
    DragScaleAEKFConfig,
    EKFConfig,
    run_drag_scale_aekf,
    run_ekf,
)


def _synthetic_case() -> dict[str, np.ndarray]:
    stations = (
        StationGeometry(name="S0", lat_deg=35.0, lon_deg=-120.0, alt_m=100.0, min_elevation_deg=-90.0),
        StationGeometry(name="S1", lat_deg=-25.0, lon_deg=27.0, alt_m=1500.0, min_elevation_deg=-90.0),
    )
    t_len = 8
    times_s = np.arange(t_len, dtype=np.float64) * 20.0
    true_state = np.array(
        [6_900_000.0, 1_200_000.0, 1_400_000.0, -950.0, 7_350.0, 750.0], dtype=np.float64
    )
    measurements = np.zeros((t_len, len(stations), 4), dtype=np.float64)
    visibility = np.zeros((t_len, len(stations)), dtype=np.float64)
    for t in range(t_len):
        for s_idx, station in enumerate(stations):
            z, _vis = line_of_sight_measurement(true_state, station, float(times_s[t]))
            measurements[t, s_idx] = z
            visibility[t, s_idx] = 1.0 if (t >= 1 and (t + s_idx) % 2 == 0) else 0.0
    x0_est = true_state + np.array([1500.0, -1200.0, 900.0, 3.0, -2.5, 1.5], dtype=np.float64)
    return {
        "measurements": measurements,
        "visibility": visibility,
        "times_s": times_s,
        "stations": stations,
        "meas_std": np.array([30.0, 0.02 * np.pi / 180.0, 0.02 * np.pi / 180.0, 0.08]),
        "x0_est": x0_est,
        "true_state": true_state,
    }


class DragScaleAEKFTests(unittest.TestCase):
    def test_returns_finite_state_and_drag_scale(self) -> None:
        case = _synthetic_case()
        cfg = DragScaleAEKFConfig()
        x_hist, p_hist, diag = run_drag_scale_aekf(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            ballistic_coeff_m2_per_kg=0.018,
            meas_std_vector=case["meas_std"],
            x0_est=case["x0_est"],
            cfg=cfg,
        )
        T = case["measurements"].shape[0]
        self.assertEqual(x_hist.shape, (T, 6))
        self.assertEqual(p_hist.shape, (T, 6, 6))
        self.assertTrue(np.all(np.isfinite(x_hist)))
        self.assertTrue(np.all(np.isfinite(p_hist)))
        self.assertIn("drag_scale_history", diag)
        beta_hist = diag["drag_scale_history"]
        self.assertEqual(beta_hist.shape, (T,))
        self.assertTrue(np.all(np.isfinite(beta_hist)))
        # Beta initial condition is 1.0
        self.assertAlmostEqual(float(beta_hist[0]), 1.0, places=9)
        # Beta should stay finite and within the soft bound used by the filter.
        self.assertTrue(np.all(beta_hist >= 0.04))
        self.assertTrue(np.all(beta_hist <= 5.01))

    def test_reduces_to_ekf_when_scale_channel_disabled(self) -> None:
        """When the drag-scale channel is fully suppressed (initial covariance
        and steady-state variance both zero), the DSA-EKF is expected to track
        the EKF posterior up to numerical tolerance, because beta stays
        identically one and the compact-state dynamics are identical to the
        EKF baseline."""
        case = _synthetic_case()
        ekf_cfg = EKFConfig()
        dsa_cfg = DragScaleAEKFConfig(
            q_pos_m=ekf_cfg.q_pos_m,
            q_vel_mps=ekf_cfg.q_vel_mps,
            init_pos_std_m=ekf_cfg.init_pos_std_m,
            init_vel_std_mps=ekf_cfg.init_vel_std_mps,
            init_drag_scale_std=0.0,
            drag_scale_sigma_ss=0.0,
            drag_scale_tau_s=600.0,
            gating_threshold=ekf_cfg.gating_threshold,
        )
        x_ekf, _ = run_ekf(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            ballistic_coeff_m2_per_kg=0.018,
            meas_std_vector=case["meas_std"],
            x0_est=case["x0_est"],
            cfg=ekf_cfg,
        )
        x_dsa, _, diag = run_drag_scale_aekf(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            ballistic_coeff_m2_per_kg=0.018,
            meas_std_vector=case["meas_std"],
            x0_est=case["x0_est"],
            cfg=dsa_cfg,
        )
        # Drag scale should stay exactly one when the channel is disabled.
        self.assertTrue(np.allclose(diag["drag_scale_history"], 1.0, atol=1e-12))
        # The (r, v) recursion is the same dynamics as the EKF up to
        # finite-difference Jacobian noise on the augmented state.
        self.assertTrue(
            np.allclose(x_dsa, x_ekf, atol=5e1),
            f"DSA-EKF posterior diverged from EKF by max {np.max(np.abs(x_dsa - x_ekf))}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
