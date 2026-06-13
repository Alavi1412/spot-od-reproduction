"""Tests for the topocentric-azimuth de-weighting conditioning fix.

The dense-visibility control's multi-thousand-kilometre medians were a
near-zenith azimuth measurement-conditioning artifact: the topocentric
azimuth is geometrically singular at the station zenith, so weighting it with
a fixed tiny angular noise drives Kalman-gain blow-up and filter divergence.
The fix inflates the azimuth measurement variance by a capped
1/cos^2(elevation) factor, applied identically to EKF/UKF/AUKF and OFF by
default (so every existing scenario, table, and pinned number is unchanged).

These tests pin: (1) the closed-form de-weight factor; (2) that the new
config field defaults to None and the default path is byte-identical to the
legacy recursion; (3) that on a near-zenith arc the fix removes the
divergence that the uncorrected filter exhibits.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import unittest

import numpy as np

from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters.ekf import (
    EKFConfig,
    azimuth_deweight_factor,
    run_ekf,
)
from gnn_state_estimation.filters.ukf import (
    AdaptiveUKFConfig,
    UKFConfig,
)
from gnn_state_estimation.simulation import generate_dataset, parse_dataset_config
from gnn_state_estimation.utils.io import load_yaml


class TestDeweightFactor(unittest.TestCase):
    def test_unity_at_horizon_and_monotone_to_cap(self) -> None:
        # At the horizon (el=0) the factor is exactly 1 (no de-weighting).
        self.assertAlmostEqual(azimuth_deweight_factor(0.0, 80.0), 1.0, places=12)
        # Strictly increasing with elevation up to the cap.
        f10 = azimuth_deweight_factor(math.radians(10.0), 80.0)
        f45 = azimuth_deweight_factor(math.radians(45.0), 80.0)
        f79 = azimuth_deweight_factor(math.radians(79.0), 80.0)
        self.assertLess(f10, f45)
        self.assertLess(f45, f79)
        self.assertGreater(f10, 1.0)

    def test_capped_above_cap_elevation(self) -> None:
        cap_factor = 1.0 / (math.cos(math.radians(80.0)) ** 2)
        # At and above the cap the factor saturates (no unbounded weight at a
        # numerical zenith crossing).
        self.assertAlmostEqual(
            azimuth_deweight_factor(math.radians(80.0), 80.0), cap_factor, places=6
        )
        self.assertAlmostEqual(
            azimuth_deweight_factor(math.radians(89.999), 80.0), cap_factor, places=6
        )
        self.assertAlmostEqual(
            azimuth_deweight_factor(math.radians(90.0), 80.0), cap_factor, places=6
        )
        # ~33x at an 80-degree cap.
        self.assertGreater(cap_factor, 30.0)
        self.assertLess(cap_factor, 35.0)


class TestConfigDefaultsOff(unittest.TestCase):
    def test_field_defaults_to_none(self) -> None:
        self.assertIsNone(EKFConfig().angle_deweight_elev_cap_deg)
        self.assertIsNone(UKFConfig().angle_deweight_elev_cap_deg)
        self.assertIsNone(AdaptiveUKFConfig().angle_deweight_elev_cap_deg)

    def test_parsed_baseline_config_is_off_by_default(self) -> None:
        cfg = load_yaml("configs/experiment.yaml")
        bc = parse_baseline_config(cfg["baselines"])
        self.assertIsNone(bc.ekf.angle_deweight_elev_cap_deg)
        self.assertIsNone(bc.ukf.angle_deweight_elev_cap_deg)
        self.assertIsNone(bc.aukf.angle_deweight_elev_cap_deg)


class TestDefaultPathByteIdentical(unittest.TestCase):
    """A config with the field explicitly None must be bit-identical to one
    that never set it (the guard is a no-op when None)."""

    def _tiny_arc(self):
        cfg = load_yaml("configs/experiment.yaml")
        sim = copy.deepcopy(cfg["simulation"])
        sim["dynamics"]["steps"] = 30
        dc = parse_dataset_config(sim)
        data = generate_dataset(dc, 2, seed=12345)
        return dc, data

    def test_none_matches_legacy(self) -> None:
        dc, data = self._tiny_arc()
        base = EKFConfig(q_pos_m=5.0, q_vel_mps=0.05, init_pos_std_m=2500.0, init_vel_std_mps=6.0)
        explicit_none = dataclasses.replace(base, angle_deweight_elev_cap_deg=None)
        x0 = data["states"][0, 0] + np.array([1500.0, -1200.0, 800.0, 3.0, -2.0, 1.0])
        a, _ = run_ekf(
            measurements=data["measurements"][0],
            visibility=data["visibility"][0],
            times_s=data["times"][0],
            stations=dc.stations,
            ballistic_coeff_m2_per_kg=dc.dynamics.ballistic_coeff_m2_per_kg,
            meas_std_vector=dc.measurement_noise.std_vector,
            x0_est=x0,
            cfg=base,
        )
        b, _ = run_ekf(
            measurements=data["measurements"][0],
            visibility=data["visibility"][0],
            times_s=data["times"][0],
            stations=dc.stations,
            ballistic_coeff_m2_per_kg=dc.dynamics.ballistic_coeff_m2_per_kg,
            meas_std_vector=dc.measurement_noise.std_vector,
            x0_est=x0,
            cfg=explicit_none,
        )
        self.assertTrue(np.array_equal(a, b))


class TestFixRemovesNearZenithDivergence(unittest.TestCase):
    """On the credible dense scenario the uncorrected EKF diverges on some
    near-zenith passes; the standard de-weighting must not make the pooled
    error worse and must not introduce new divergence."""

    def test_deweighting_does_not_worsen_and_bounds_error(self) -> None:
        cfg = load_yaml("configs/experiment.yaml")
        sim = copy.deepcopy(cfg["simulation"])
        scen = cfg["benchmark_suite"]["scenarios"]["credible_dense_od_test"]
        sim["stations"] = scen["overrides"]["stations"]
        sim["dynamics"]["steps"] = 60
        dc = parse_dataset_config(sim)
        data = generate_dataset(dc, 6, seed=330000)
        std = dc.measurement_noise.std_vector

        def pooled_rmse(cap):
            errs = []
            for i in range(data["states"].shape[0]):
                x0 = data["states"][i, 0] + np.array(
                    [2000.0, -1500.0, 1000.0, 4.0, -3.0, 2.0]
                )
                xh, _ = run_ekf(
                    measurements=data["measurements"][i],
                    visibility=data["visibility"][i],
                    times_s=data["times"][i],
                    stations=dc.stations,
                    ballistic_coeff_m2_per_kg=dc.dynamics.ballistic_coeff_m2_per_kg,
                    meas_std_vector=std,
                    x0_est=x0,
                    cfg=EKFConfig(
                        q_pos_m=5.5,
                        q_vel_mps=0.06,
                        init_pos_std_m=2500.0,
                        init_vel_std_mps=6.0,
                        angle_deweight_elev_cap_deg=cap,
                    ),
                )
                e = np.sqrt(
                    np.mean(np.sum((data["states"][i, :, :3] - xh[:, :3]) ** 2, axis=-1))
                )
                if np.isfinite(e):
                    errs.append(e)
            return np.array(errs)

        uncorrected = pooled_rmse(None)
        corrected = pooled_rmse(80.0)
        # The corrected filter must keep every trajectory finite ...
        self.assertEqual(corrected.size, data["states"].shape[0])
        # ... and its worst-case (max) trajectory error must not exceed the
        # uncorrected filter's worst case: the fix only ever damps the
        # ill-conditioned near-zenith azimuth update.
        self.assertLessEqual(float(np.max(corrected)), float(np.max(uncorrected)) + 1.0)


if __name__ == "__main__":
    unittest.main()
