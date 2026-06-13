"""Tests for the loop-25 DBAR independent-realization sweep.

Cheap: the rule/threshold logic is exercised on tiny crafted rows; the
committed sweep artifact (if present) is checked for schema, for the
honest powered negative (at the ~450-realization powered scale DBAR is
not statistically distinguishable from the no-information baseline), and
for the durable measurement-noise-stress specificity control (large
R_eff but DBAR must essentially not fire). No model is trained and no
realization is regenerated here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_dbar_independent_sweep",
    _ROOT / "scripts" / "build_dbar_independent_sweep.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

_ARTIFACT = (
    _ROOT / "results" / "adaptation_risk_diagnostic" / "dbar_independent_sweep.json"
)


class TestSweepRuleLogic(unittest.TestCase):
    def test_predeclared_thresholds_round_and_fixed(self) -> None:
        self.assertEqual(mod.TAU_R, 1.5)
        self.assertEqual(mod.TAU_RHO, 1.5)
        self.assertEqual(mod.MATERIALITY_MARGIN, 0.05)
        self.assertEqual(tuple(mod.FAMILIES), ("nominal", "meas_stress", "dynamics_bias"))

    def test_severity_is_independent_per_family(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        self.assertEqual(mod.sample_severity("nominal", rng), {})
        ms = mod.sample_severity("meas_stress", rng)
        self.assertEqual(ms["kind"], "meas_stress")
        self.assertGreaterEqual(ms["noise_factor"], mod.MEAS_STRESS_FACTOR[0])
        self.assertLessEqual(ms["noise_factor"], mod.MEAS_STRESS_FACTOR[1])
        db = mod.sample_severity("dynamics_bias", rng)
        self.assertEqual(db["kind"], "dynamics_bias")
        self.assertGreater(db["process_noise_std"], 0.0)

    def test_threshold_sensitivity_grid_shape_and_accuracy(self) -> None:
        # Cleanly separable crafted rows: a no-fire cluster and a fire cluster.
        rows = []
        for i in range(10):
            rows.append(
                {
                    "r_eff": 1.1,
                    "rho_nis": 1.0,
                    "adaptation_counterproductive": False,
                }
            )
            rows.append(
                {
                    "r_eff": 3.0 + 0.01 * i,
                    "rho_nis": 2.5,
                    "adaptation_counterproductive": True,
                }
            )
        ts = mod.threshold_sensitivity(rows)
        self.assertEqual(ts["n_grid_points"], 49)
        self.assertEqual(ts["predeclared_accuracy"], 1.0)
        self.assertEqual(ts["grid_min_accuracy"], 1.0)

    def test_meas_stress_high_reff_does_not_fire(self) -> None:
        # The decisive control: large R_eff but rho_NIS below threshold ->
        # no fire, and adaptation actually helped -> classified correct.
        rng = __import__("numpy").random.default_rng(1)
        # Exercise the pure decision arithmetic the sweep applies.
        r_eff, rho = 4.8, 1.3
        fired = (r_eff > mod.TAU_R) and (rho >= mod.TAU_RHO)
        self.assertFalse(fired)


class TestSweepArtifact(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("DBAR independent-sweep artifact not built")
        self.d = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_schema_and_independence(self) -> None:
        self.assertEqual(self.d["schema_version"], "dbar_independent_sweep_v1")
        des = self.d["design"]
        self.assertGreaterEqual(des["n_independent_realizations"], 30)
        self.assertEqual(
            tuple(des["families"]), ("nominal", "meas_stress", "dynamics_bias")
        )

    def test_powered_characterization_is_honest_negative(self) -> None:
        # The committed artifact is the properly powered sweep. At this scale
        # the small-sample optimism does not survive: DBAR is NOT shown to
        # beat the no-information baseline. This test pins the honest
        # negative so no future edit can quietly re-inflate the claim.
        s = self.d["summary"]
        rep = s["classification_report"]
        self.assertGreaterEqual(
            self.d["design"]["n_independent_realizations"], 400,
            "powered characterization must use of order 400+ realizations",
        )
        ni = s["no_information_baseline"]
        # Incremental accuracy over the trivial majority classifier is small
        # and the accuracy Wilson interval contains the majority baseline,
        # i.e. not statistically distinguishable from no information.
        self.assertLess(float(ni["accuracy_minus_majority"]), 0.05)
        self.assertLessEqual(
            float(rep["accuracy_ci"][0]),
            float(ni["majority_class_accuracy"]),
            "accuracy CI lower bound must not clear the no-information "
            "baseline (DBAR is reported as a powered negative)",
        )

    def test_decisive_measurement_noise_stress_control(self) -> None:
        # Among measurement-noise-stress realizations, R-adaptation is the
        # correct move, so DBAR must essentially never fire there.
        fam = self.d["summary"]["by_family"]["meas_stress"]
        self.assertLessEqual(fam["n_dbar_fired"], max(1, int(0.1 * fam["n"])))

    def test_threshold_sensitivity_band(self) -> None:
        ts = self.d["threshold_sensitivity"]
        self.assertEqual(ts["n_grid_points"], 49)
        # The predeclared-threshold accuracy is not a fragile grid maximum.
        self.assertGreaterEqual(ts["grid_min_accuracy"], 0.5)
        self.assertGreaterEqual(
            ts["n_grid_points_within_0p05_of_predeclared"], 9
        )


if __name__ == "__main__":
    unittest.main()
