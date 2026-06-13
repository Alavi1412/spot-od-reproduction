"""Focused tests for the process-noise-adaptive UKF (PUKF) (loop 41).

Covers the new ``run_process_noise_adaptive_ukf`` and
``ProcessNoiseAdaptiveUKFConfig`` against the same small synthetic case
used by the AUKF tests, so the predeclared Q-adaptive rule's filter
recursion is exercised cheaply on every run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.coordinates import StationGeometry, line_of_sight_measurement
from gnn_state_estimation.filters.ukf import (
    ProcessNoiseAdaptiveUKFConfig,
    run_process_noise_adaptive_ukf,
    run_ukf,
    UKFConfig,
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


class PUKFTests(unittest.TestCase):
    def test_returns_finite_state_and_records(self) -> None:
        case = _synthetic_case()
        cfg = ProcessNoiseAdaptiveUKFConfig()
        x_hist, p_hist, recs = run_process_noise_adaptive_ukf(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            ballistic_coeff_m2_per_kg=0.018,
            meas_std_vector=case["meas_std"],
            x0_est=case["x0_est"],
            cfg=cfg,
        )
        self.assertEqual(x_hist.shape, (case["measurements"].shape[0], 6))
        self.assertEqual(p_hist.shape, (case["measurements"].shape[0], 6, 6))
        self.assertTrue(np.all(np.isfinite(x_hist)))
        self.assertTrue(np.all(np.isfinite(p_hist)))
        self.assertEqual(len(recs), case["measurements"].shape[0] - 1)
        for r in recs:
            self.assertGreaterEqual(r["q_scale_used"], 1.0)
            self.assertLessEqual(r["q_scale_used"], cfg.q_scale_max)

    def test_reduces_to_ukf_with_q_scale_one(self) -> None:
        """If the windowed NIS never exceeds the warn ratio, q_scale stays 1
        and the PUKF posterior should be very close to the fixed-noise UKF."""
        case = _synthetic_case()
        # Set thresholds so the rule never fires (warn_ratio above any plausible value).
        pukf_cfg = ProcessNoiseAdaptiveUKFConfig(
            nis_warn_ratio=1e9,
            nis_alarm_ratio=1e9,
            q_scale_warn=1.0,
            q_scale_alarm=1.0,
            q_scale_max=1.0,
            smoothing=0.0,
        )
        ukf_cfg = UKFConfig()
        x_pukf, _, _ = run_process_noise_adaptive_ukf(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            ballistic_coeff_m2_per_kg=0.018,
            meas_std_vector=case["meas_std"],
            x0_est=case["x0_est"],
            cfg=pukf_cfg,
        )
        x_ukf, _ = run_ukf(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            ballistic_coeff_m2_per_kg=0.018,
            meas_std_vector=case["meas_std"],
            x0_est=case["x0_est"],
            cfg=ukf_cfg,
        )
        # Same Q-scale rule, same NIS expectation: posteriors should match closely.
        self.assertTrue(
            np.allclose(x_pukf, x_ukf, atol=1e-6),
            f"PUKF posterior diverged from UKF by max {np.max(np.abs(x_pukf - x_ukf))}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
