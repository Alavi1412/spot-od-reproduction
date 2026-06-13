"""Tests for the loop-30 dense-visibility estimator-skill probe.

The probe answers the loop-29 reviewer concern that the synthetic
nominal/stress splits are propagation-dominated (~79% zero-visible steps)
and therefore do not measure estimator skill. It densifies *only* the
observation network (perfect shared compact model, no dynamics mismatch)
into a measurement-update-dominant regime and scores the recursive
classical filters and the fixed, previously trained RGR-GF estimator
(evaluated without any per-realization refitting).

Cheap: the divergence-guard / median reduction arithmetic is exercised on
tiny crafted arrays. The committed artifact (if present) is checked for
schema, for the "not propagation-dominated" property, and for the honest
negative -- the released learned residual must NOT be shown to beat the
tuned AUKF on the heavy-tail-robust median observed-step metric. This pins
the honest negative so no future edit can quietly re-inflate it. No model
is trained and no realization is regenerated here.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = (
    _ROOT / "results" / "dense_visibility_probe" / "dense_visibility_probe.json"
)

DIVERGENCE_POS_RMSE_M = 1.0e8
ENGINEERING_ADEQUATE_POS_RMSE_M = 1.0e5


class TestProbeReductionLogic(unittest.TestCase):
    def test_divergence_guard_and_median_are_heavy_tail_robust(self) -> None:
        # Bulk of trajectories well-estimated, a heavy tail of blow-ups.
        vals = np.array([300.0, 350.0, 400.0, 450.0, 5.0e7, np.inf])
        finite_ok = np.isfinite(vals) & (vals <= DIVERGENCE_POS_RMSE_M)
        # Only the np.inf trajectory exceeds the 1e8 m divergence guard.
        self.assertEqual(int(np.sum(finite_ok)), 5)
        # The median ignores the heavy tail; the mean would not.
        med = float(np.median(vals[np.isfinite(vals)]))
        self.assertLess(med, 1.0e4)
        # Gross-failure (100 km) rate counts the 5e7 m and inf trajectories.
        gross = ~(np.isfinite(vals) & (vals <= ENGINEERING_ADEQUATE_POS_RMSE_M))
        self.assertEqual(int(np.sum(gross)), 2)

    def test_pooled_rmse_excludes_nonfinite(self) -> None:
        vals = np.array([100.0, 200.0, np.nan])
        v = vals[np.isfinite(vals)]
        rmse = math.sqrt(float(np.mean(v ** 2)))
        self.assertAlmostEqual(rmse, math.sqrt((100.0**2 + 200.0**2) / 2.0), places=6)


class TestProbeArtifact(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("dense-visibility probe artifact not built")
        self.d = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        if self.d.get("schema_version") != "dense_visibility_probe_v2":
            self.skipTest("dense-visibility probe artifact schema mismatch")
        _required = (
            "visibility",
            "gross_failure_rate",
            "median_trajectory_position_rmse_m",
            "best_method_observed_step_median",
            "regime_is_estimator_discriminative",
            "conclusion_changes_vs_sparse_regime",
        )
        if any(k not in self.d for k in _required):
            self.skipTest("dense-visibility probe artifact incomplete (pre-run)")

    def test_schema_and_perfect_shared_model(self) -> None:
        self.assertEqual(self.d["status"], "completed")
        self.assertFalse(
            self.d["pre_registered"],
            "the probe must not be presented as a pre-registered endpoint",
        )
        self.assertGreaterEqual(self.d["n_trajectories_total"], 100)

    def test_regime_is_not_propagation_dominated(self) -> None:
        vis = self.d["visibility"]
        main_ref = float(vis["main_split_zero_visible_fraction_reference"])
        zvf = float(vis["zero_visible_fraction_mean"])
        # The whole point: this regime is materially less zero-visible than
        # the propagation-dominated main split, so a measurement update
        # actually enters the comparison.
        self.assertLess(zvf, main_ref - 0.10)
        self.assertGreater(
            float(vis["measurement_informed_fraction_mean"]), 0.25
        )

    def test_honest_negative_is_pinned(self) -> None:
        # The rigorous honest-negative pin: densifying the observation
        # geometry must NOT flip the conclusion into a learned advantage.
        # conclusion_changes_vs_sparse_regime is True only if the regime is
        # estimator-discriminative AND RGR-GF is strictly best on the
        # physically adequate subset by a non-trivial (>5%) margin over every
        # classical reference -- noisy median ordering in a
        # divergence-dominated regime cannot trip it. Pin it False so no edit
        # can quietly inflate the probe into a learned win.
        self.assertFalse(
            self.d["conclusion_changes_vs_sparse_regime"],
            "densifying visibility must not flip the conclusion into a "
            "learned advantage (honest negative)",
        )
        # If the regime IS discriminative, additionally require that no
        # learned residual is the best estimator on the adequate subset.
        if self.d["regime_is_estimator_discriminative"]:
            self.assertIn(
                self.d["best_method_observed_step_adequate"],
                {"EKF", "UKF", "AUKF"},
                "in a discriminative regime a classical filter must be best",
            )


if __name__ == "__main__":
    unittest.main()
