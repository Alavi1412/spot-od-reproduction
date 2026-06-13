"""Tests for the less-constrained learned residual comparator (loop-37 FF1).

Cheap: validates the committed
``results/unconstrained_residual_comparator/unconstrained_residual_comparator.json``
schema and the honest pre-registration invariants. The comparator is the
canonical residual architecture with the residual budget, the gate, the
context budget, and the prior-anchoring/auxiliary penalties removed, so a
learned positive cannot be blamed on the bound. No model is run.
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
    / "unconstrained_residual_comparator"
    / "unconstrained_residual_comparator.json"
)


class UnconstrainedResidualComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("unconstrained_residual_comparator.json not present")
        self.data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_comparator_is_less_constrained(self) -> None:
        mk = self.data["comparator"]["model_kwargs"]
        # The defining relaxation vs the canonical bounded/anchored residual.
        self.assertFalse(bool(mk.get("bounded_residual", True)))
        self.assertFalse(bool(mk.get("use_gating", True)))
        self.assertFalse(bool(mk.get("use_context_budget", True)))
        self.assertGreaterEqual(float(mk.get("residual_scale", 0.0)), 1.0)

    def test_preregistration_block(self) -> None:
        pr = self.data["pre_registration"]
        self.assertEqual(pr["primary_metric"], "observed_step_position_rmse_m")
        self.assertEqual(pr["reference_metric"], "all_step_position_rmse_m")
        self.assertTrue(pr["fixed_before_results"])
        # Disjoint from the 770000 pre-registration seed and the 41-55 cohort.
        self.assertGreaterEqual(int(pr["realization_base_seed"]), 800000)
        self.assertIn("disjoint", pr["seed_disjointness"])
        self.assertIn("WLS", pr["classical_references"])

    def test_scenarios_and_decision_rule(self) -> None:
        rows = self.data["scenarios"]
        self.assertGreaterEqual(len(rows), 3)
        for r in rows:
            self.assertIn("RGR-U", r["primary_observed_step_pos_rmse_m"])
            self.assertIn("WLS", r["primary_observed_step_pos_rmse_m"])
            if r["learned_positive_under_predeclared_rule"]:
                self.assertEqual(r["best_method_primary"], "RGR-U")
                self.assertLess(
                    r["rgr_u_minus_best_classical_primary_ci_high_m"], 0.0
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
        tex = mod.build_unconstrained_residual_comparator_table(_ARTIFACT)
        self.assertIn("tab:unconstrained_residual_comparator", tex)
        self.assertIn("removed", tex)


if __name__ == "__main__":
    unittest.main()
