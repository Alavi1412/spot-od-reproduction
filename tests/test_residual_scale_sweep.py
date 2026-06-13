"""Tests for the residual-scale sweep comparator (loop-38 R3).

Cheap: validates the committed
``results/residual_scale_sweep/residual_scale_sweep.json`` schema and the
honest pre-registration invariants. The sweep extends the canonical-vs-RGR-U
two-point comparison with two tethered intermediate residual scales (0.30
and 1.00), characterising whether the canonical-vs-unconstrained gap is
binary or graded. No model is run.
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
    / "residual_scale_sweep"
    / "residual_scale_sweep.json"
)


class ResidualScaleSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("residual_scale_sweep.json not present")
        self.data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_preregistration_block(self) -> None:
        pr = self.data["pre_registration"]
        self.assertEqual(pr["primary_metric"], "observed_step_position_rmse_m")
        self.assertEqual(pr["reference_metric"], "all_step_position_rmse_m")
        self.assertTrue(pr["fixed_before_results"])
        # Disjoint from the 770000 pre-registration seed, 880000 unconstrained.
        self.assertGreaterEqual(int(pr["realization_base_seed"]), 880001)
        self.assertIn("disjoint", pr["seed_disjointness"])
        self.assertIn("WLS", pr["classical_references"])

    def test_sweep_covers_endpoints_and_intermediates(self) -> None:
        pr = self.data["pre_registration"]
        scales = sorted({float(pt["residual_scale"]) for pt in pr["sweep_points"]})
        # Endpoints (canonical 0.03, RGR-U 1.0) plus intermediates.
        self.assertIn(0.03, scales)
        self.assertIn(1.0, scales)
        self.assertGreaterEqual(len(scales), 3)
        variants = {pt["variant"] for pt in pr["sweep_points"]}
        self.assertIn("tethered", variants)
        self.assertIn("untethered", variants)

    def test_scenarios_complete(self) -> None:
        rows = self.data["scenarios"]
        self.assertGreaterEqual(len(rows), 3)
        for r in rows:
            self.assertIn("EKF", r["primary_observed_step_pos_rmse_m"])
            self.assertIn("WLS", r["primary_observed_step_pos_rmse_m"])
            for entry in r["per_learned"].values():
                self.assertIn("minus_best_classical_mean_m", entry)
                self.assertIn("minus_best_classical_ci_low_m", entry)
                self.assertIn("minus_best_classical_ci_high_m", entry)

    def test_table_renders(self) -> None:
        tex = mod.build_residual_scale_sweep_table(_ARTIFACT)
        self.assertIn("tab:residual_scale_sweep", tex)
        self.assertIn("intermediate", tex.lower())

    def test_k_is_at_least_eight(self) -> None:
        """MC-1 (loop-40): K>=8 with non-degenerate paired-bootstrap CIs."""
        pr = self.data["pre_registration"]
        self.assertGreaterEqual(int(pr["num_realizations_per_scenario"]), 8)
        for r in self.data["scenarios"]:
            self.assertGreaterEqual(int(r["n_realizations"]), 8)
            for entry in r["per_learned"].values():
                lo = float(entry["minus_best_classical_ci_low_m"])
                hi = float(entry["minus_best_classical_ci_high_m"])
                # CI must not collapse to a point (the K=1 failure mode that
                # MC-1 fixes).
                self.assertNotEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
