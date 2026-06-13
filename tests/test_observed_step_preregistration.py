"""Tests for the fresh-independent observed-step endpoint-fixation support record.

Cheap: validates the committed
``results/observed_step_preregistration/observed_step_preregistration.json``
schema and the honest endpoint-fixation invariants: observed-step is the
adopted primary endpoint with all-step as the reference; the
realization base seed is disjoint from training/validation/model
selection; the decision rule is fixed before results; and no learned
positive is asserted unless the fixed rule actually fires. No model is run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_paper_assets", _ROOT / "scripts" / "build_paper_assets.py"
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

_ARTIFACT = (
    _ROOT
    / "results"
    / "observed_step_preregistration"
    / "observed_step_preregistration.json"
)


class ObservedStepPreregistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("observed_step_preregistration.json not present")
        self.data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_preregistration_block(self) -> None:
        pr = self.data["pre_registration"]
        self.assertEqual(pr["primary_metric"], "observed_step_position_rmse_m")
        self.assertEqual(pr["reference_metric"], "all_step_position_rmse_m")
        self.assertTrue(pr["fixed_before_results"])
        # Base seed disjoint from the 41-55 cohort and the 90000 resampling
        # base; far from every model-selection validation split.
        self.assertGreaterEqual(int(pr["realization_base_seed"]), 700000)
        self.assertIn("disjoint", pr["seed_disjointness"])

    def test_scenarios_and_decision_rule(self) -> None:
        rows = self.data["scenarios"]
        self.assertGreaterEqual(len(rows), 3)
        for r in rows:
            self.assertIn("primary_observed_step_pos_rmse_m", r)
            self.assertIn("reference_all_step_pos_rmse_m", r)
            self.assertIn("RGR-GF", r["primary_observed_step_pos_rmse_m"])
            # learned_positive is only true if RGR-GF is the per-scenario
            # best on the primary endpoint with a CI strictly below zero.
            if r["learned_positive_under_predeclared_rule"]:
                self.assertEqual(r["best_method_primary"], "RGR-GF")
                self.assertLess(
                    r["rgr_gf_minus_best_classical_primary_ci_high_m"], 0.0
                )

    def test_summary_consistent(self) -> None:
        s = self.data["summary"]
        positives = sum(
            1
            for r in self.data["scenarios"]
            if r["learned_positive_under_predeclared_rule"]
        )
        self.assertEqual(
            s["scenarios_with_learned_positive_under_predeclared_rule"], positives
        )

    def test_table_renders(self) -> None:
        tex = mod.build_observed_step_preregistration_table(_ARTIFACT)
        self.assertIn("tab:observed_step_preregistration", tex)
        self.assertIn("Submitted observed-step endpoint-fixation support record", tex)
        self.assertIn("lacks a created/finalized timestamp field", tex)
        self.assertIn("$K{=}8$ endpoint-fixation support", tex)
        self.assertIn("no confirmatory status", tex)
        self.assertNotIn("predeclared", tex.lower())
        self.assertNotIn("$K{=}8$ predeclared", tex)


if __name__ == "__main__":
    unittest.main()
