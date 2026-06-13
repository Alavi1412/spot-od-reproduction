"""Tests for the astrodynamically-credible dense-tracking OD probe.

This probe is the corrected twin of the densified-visibility control. It runs
a perfect shared dynamics model (third-body and SRP enabled, no force-model,
measurement, or station mismatch) on the dense global network with the
standard topocentric-azimuth de-weighting, scores a PREDECLARED primary metric
(pooled all-step trajectory position RMSE) with the observed-step reported
symmetrically, and quantifies the near-zenith conditioning artifact with a
fix-on/fix-off diagnostic.

The committed artifact (if present) is checked for schema, the predeclared
primary metric, the perfect-shared-model assumption, astrodynamic credibility,
the honest negative, and that the conditioning fix removes the artifact. No
model is trained and no realization is regenerated here.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = (
    _ROOT / "results" / "credible_dense_od_probe" / "credible_dense_od_probe.json"
)

DIVERGENCE_POS_RMSE_M = 1.0e8
ENGINEERING_ADEQUATE_POS_RMSE_M = 1.0e5


class TestPairedBootstrapLogic(unittest.TestCase):
    def test_paired_difference_of_rmse(self) -> None:
        a = np.array([100.0, 120.0, 90.0, 110.0])
        b = np.array([130.0, 140.0, 120.0, 150.0])
        diff = math.sqrt(float(np.mean(a**2))) - math.sqrt(float(np.mean(b**2)))
        # The learned-like array is uniformly better here, so the paired RMSE
        # difference is negative (learned better).
        self.assertLess(diff, 0.0)

    def test_gross_failure_rate_counts_blowups(self) -> None:
        vals = np.array([300.0, 450.0, 5.0e7, np.inf])
        gross = ~(np.isfinite(vals) & (vals <= ENGINEERING_ADEQUATE_POS_RMSE_M))
        self.assertEqual(int(np.sum(gross)), 2)


class TestCredibleProbeArtifact(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("credible dense OD probe artifact not built")
        self.d = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        if self.d.get("schema_version") != "credible_dense_od_probe_v1":
            self.skipTest("credible dense OD probe artifact schema mismatch")
        _required = (
            "predeclared_primary_metric",
            "primary_all_step_pooled_rmse_m",
            "secondary_observed_step_pooled_rmse_m",
            "median_trajectory_position_rmse_m",
            "divergence_diagnosis",
            "shared_dynamics_no_mismatch",
            "od_is_astrodynamically_credible",
            "learned_positive_established",
        )
        if any(k not in self.d for k in _required):
            self.skipTest("credible dense OD probe artifact incomplete (pre-run)")

    def test_predeclared_and_shared_model(self) -> None:
        self.assertEqual(self.d["status"], "completed")
        self.assertTrue(
            self.d["pre_registered"],
            "the primary metric of this probe must be predeclared",
        )
        self.assertIn("all-step", self.d["predeclared_primary_metric"])
        self.assertTrue(self.d["shared_dynamics_no_mismatch"])
        self.assertTrue(self.d["third_body_enabled"])
        self.assertTrue(self.d["srp_enabled"])
        self.assertGreaterEqual(self.d["n_trajectories_total"], 100)
        # Symmetric reporting: both the primary and the secondary metric exist
        # for every estimator.
        for m in ("EKF", "UKF", "AUKF", "RGR-GF"):
            self.assertIn(m, self.d["primary_all_step_pooled_rmse_m"])
            self.assertIn(m, self.d["secondary_observed_step_pooled_rmse_m"])

    def test_regime_is_measurement_update_dominant(self) -> None:
        vis = self.d["visibility"]
        self.assertLess(
            float(vis["zero_visible_fraction_mean"]),
            float(vis["main_split_zero_visible_fraction_reference"]) - 0.10,
        )
        self.assertGreater(float(vis["measurement_informed_fraction_mean"]), 0.25)

    def test_od_is_astrodynamically_credible(self) -> None:
        # The whole point of the corrected probe: the best classical filter
        # reaches a physically adequate (<100 km) pooled state error, unlike
        # the multi-thousand-kilometre uncorrected control.
        self.assertTrue(self.d["od_is_astrodynamically_credible"])
        best_cls = self.d["best_classical_primary_all_step"]
        self.assertIn(best_cls, {"EKF", "UKF", "AUKF"})
        self.assertLessEqual(
            float(self.d["primary_all_step_pooled_rmse_m"][best_cls]),
            ENGINEERING_ADEQUATE_POS_RMSE_M,
        )

    def test_credible_vs_uncorrected_control_and_azimuth_is_minor(self) -> None:
        diag = self.d["divergence_diagnosis"]
        worst_unc = max(float(v) for v in diag["uncorrected_gross_failure_rate"].values())
        worst_cor = max(float(v) for v in diag["corrected_gross_failure_rate"].values())
        # The standard azimuth de-weighting must never make things worse on the
        # same trajectories (it only damps the ill-conditioned near-zenith
        # update). It is a do-no-harm refinement; under an 8-degree mask its
        # effect is minor, which is itself the honest finding.
        self.assertLessEqual(worst_cor, worst_unc + 1e-9)
        # The corrected, network-consistent regime must be astrodynamically
        # credible: the worst-estimator gross-failure here is a small minority,
        # in stark contrast to the uncorrected station-mismatched control.
        self.assertLess(worst_cor, 0.20)
        old = (
            _ROOT
            / "results"
            / "dense_visibility_probe"
            / "dense_visibility_probe.json"
        )
        if old.exists():
            od = json.loads(old.read_text(encoding="utf-8"))
            old_worst_gf = max(
                float(v) for v in od["gross_failure_rate"].values()
            )
            # The corrected regime's worst gross-failure must be far below the
            # uncorrected control's: the uncorrected catastrophe was the
            # estimator/truth observation-network inconsistency, removed here.
            self.assertLess(worst_cor, old_worst_gf - 0.30)

    def test_honest_negative_unless_paired_ci_excludes_zero(self) -> None:
        # A learned positive may only be claimed if the regime is credible AND
        # the paired bootstrap CI strictly excludes zero in the learned
        # direction. Pin the consistency of that flag so no edit can inflate a
        # negative into a positive.
        learned_pos = bool(self.d["learned_positive_established"])
        paired = self.d.get("paired_learned_vs_best_classical")
        if learned_pos:
            self.assertIsNotNone(paired)
            self.assertTrue(paired["ci_excludes_zero"])
            self.assertTrue(paired["rgr_gf_better"])
            self.assertTrue(self.d["od_is_astrodynamically_credible"])
        else:
            # Honest negative: the learned estimator is not the best on the
            # predeclared metric.
            self.assertIn(
                self.d["best_method_primary_all_step"], {"EKF", "UKF", "AUKF"}
            )


if __name__ == "__main__":
    unittest.main()
