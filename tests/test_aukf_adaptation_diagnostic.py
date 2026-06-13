"""Focused tests for the AUKF mechanistic-diagnostic helpers.

These cover the instrumented AUKF twin and the comparable R-only innovation
diagnostic on small synthetic arrays. They are intentionally cheap (a handful
of time steps, two stations, no third-body/SRP) so they add negligible runtime.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.coordinates import StationGeometry, line_of_sight_measurement
from gnn_state_estimation.filters.ukf import (
    AUKF_MEAS_CHANNELS,
    AdaptiveUKFConfig,
    predicted_innovation_nis,
    run_adaptive_ukf,
    run_adaptive_ukf_instrumented,
)


def _synthetic_case() -> dict[str, np.ndarray]:
    stations = (
        StationGeometry(name="S0", lat_deg=35.0, lon_deg=-120.0, alt_m=100.0, min_elevation_deg=-90.0),
        StationGeometry(name="S1", lat_deg=-25.0, lon_deg=27.0, alt_m=1500.0, min_elevation_deg=-90.0),
    )
    t_len = 6
    times_s = np.arange(t_len, dtype=np.float64) * 20.0
    true_state = np.array([6_900_000.0, 1_200_000.0, 1_400_000.0, -950.0, 7_350.0, 750.0], dtype=np.float64)

    measurements = np.zeros((t_len, len(stations), 4), dtype=np.float64)
    visibility = np.zeros((t_len, len(stations)), dtype=np.float64)
    for t in range(t_len):
        for s_idx, station in enumerate(stations):
            z, _vis = line_of_sight_measurement(true_state, station, float(times_s[t]))
            measurements[t, s_idx] = z
            # Deterministic visibility pattern with several visible updates.
            visibility[t, s_idx] = 1.0 if (t >= 1 and (t + s_idx) % 2 == 0) else 0.0

    x0_est = true_state + np.array([1500.0, -1200.0, 900.0, 3.0, -2.5, 1.5], dtype=np.float64)
    return {
        "measurements": measurements,
        "visibility": visibility,
        "times_s": times_s,
        "stations": stations,
        "x0_est": x0_est,
        "true_state": true_state,
    }


_AUKF_KWARGS = dict(
    ballistic_coeff_m2_per_kg=0.018,
    meas_std_vector=np.array([30.0, np.deg2rad(0.02), np.deg2rad(0.02), 0.08], dtype=np.float64),
    cfg=AdaptiveUKFConfig(),
    enable_third_body=False,
    enable_srp=False,
)


class AukfInstrumentedTwinTests(unittest.TestCase):
    def test_instrumented_matches_baseline_bit_for_bit(self) -> None:
        case = _synthetic_case()
        x_base, p_base = run_adaptive_ukf(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            x0_est=case["x0_est"],
            **_AUKF_KWARGS,
        )
        x_inst, p_inst, records = run_adaptive_ukf_instrumented(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            x0_est=case["x0_est"],
            **_AUKF_KWARGS,
        )
        # The instrumented twin must not alter the baseline outputs at all.
        np.testing.assert_array_equal(x_inst, x_base)
        np.testing.assert_array_equal(p_inst, p_base)
        self.assertGreater(len(records), 0)

    def test_record_count_matches_visible_updates(self) -> None:
        case = _synthetic_case()
        _x, _p, records = run_adaptive_ukf_instrumented(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            x0_est=case["x0_est"],
            **_AUKF_KWARGS,
        )
        # k starts at 1 in the recursion, so step 0 visibilities are not updated.
        expected = int(np.sum(case["visibility"][1:] >= 0.5))
        self.assertEqual(len(records), expected)

    def test_record_fields_are_well_formed(self) -> None:
        case = _synthetic_case()
        _x, _p, records = run_adaptive_ukf_instrumented(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            x0_est=case["x0_est"],
            **_AUKF_KWARGS,
        )
        soft_gate = float(_AUKF_KWARGS["cfg"].nis_soft_gate)
        for rec in records:
            self.assertGreaterEqual(rec["pre_adapt_nis"], 0.0)
            self.assertGreaterEqual(rec["robust_scale"], 1.0 - 1e-12)
            self.assertEqual(rec["nis_exceeds_soft_gate"], rec["pre_adapt_nis"] > soft_gate)
            # Effective scale must respect the configured clamp band.
            self.assertGreaterEqual(rec["r_eff_scale_mean"], _AUKF_KWARGS["cfg"].min_r_scale - 1e-9)
            self.assertLessEqual(rec["r_eff_scale_mean"], _AUKF_KWARGS["cfg"].max_r_scale + 1e-9)
            self.assertGreaterEqual(rec["state_update_norm"], 0.0)
            np.testing.assert_allclose(
                rec["state_update_norm"],
                np.hypot(rec["state_update_pos_norm_m"], rec["state_update_vel_norm_mps"]),
                rtol=1e-9,
                atol=1e-6,
            )
            for ch in AUKF_MEAS_CHANNELS:
                for prefix in ("r_scale_pre_", "r_proposal_scale_", "r_scale_post_", "r_eff_scale_"):
                    self.assertIn(f"{prefix}{ch}", rec)

    def test_robust_inflation_triggers_on_biased_init(self) -> None:
        """A large initial offset should drive some updates over the soft gate."""
        case = _synthetic_case()
        x0_biased = case["true_state"] + np.array(
            [60_000.0, -45_000.0, 30_000.0, 40.0, -35.0, 25.0], dtype=np.float64
        )
        _x, _p, records = run_adaptive_ukf_instrumented(
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            x0_est=x0_biased,
            **_AUKF_KWARGS,
        )
        self.assertTrue(any(r["nis_exceeds_soft_gate"] for r in records))
        self.assertTrue(any(r["robust_scale"] > 1.0 for r in records))


class PredictedInnovationNisTests(unittest.TestCase):
    def test_zero_residual_gives_zero_nis(self) -> None:
        case = _synthetic_case()
        # pred == true at every step => residual is exactly zero where visible.
        pred = np.tile(case["true_state"], (case["times_s"].size, 1))
        out = predicted_innovation_nis(
            pred_states=pred,
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            meas_std_vector=_AUKF_KWARGS["meas_std_vector"],
        )
        self.assertEqual(len(out), int(np.sum(case["visibility"] >= 0.5)))
        for entry in out:
            self.assertAlmostEqual(entry["nis_r"], 0.0, places=9)
            self.assertAlmostEqual(entry["whitened_norm"], 0.0, places=9)

    def test_biased_prediction_gives_positive_nis(self) -> None:
        case = _synthetic_case()
        pred = np.tile(case["true_state"] + np.array([5000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (case["times_s"].size, 1))
        out = predicted_innovation_nis(
            pred_states=pred,
            measurements=case["measurements"],
            visibility=case["visibility"],
            times_s=case["times_s"],
            stations=case["stations"],
            meas_std_vector=_AUKF_KWARGS["meas_std_vector"],
        )
        self.assertTrue(all(entry["nis_r"] > 0.0 for entry in out))


if __name__ == "__main__":
    unittest.main()
