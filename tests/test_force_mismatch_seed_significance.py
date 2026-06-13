"""Tests for the force-model-mismatch paired-significance hardening.

Cheap: validates the committed
``results/force_mismatch_seed_significance.json`` schema and the honest
invariants the manuscript relies on (deterministic classical observed-step
ordering EKF < UKF < AUKF; classical paired CIs present; the 15-seed
RGR-GF cohort never beats EKF; descriptive-only Wilcoxon labelling). It
also checks the generated table renders the two panels. No model is run.
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

_ARTIFACT = _ROOT / "results" / "force_mismatch_seed_significance.json"


class ForceMismatchSeedSignificanceTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("force_mismatch_seed_significance.json not present")
        self.data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_classical_observed_ordering_and_cis(self) -> None:
        pooled = self.data["classical_pooled_observed_pos_rmse_m"]
        # Causal EKF best, tuned AUKF worst under true dynamics mismatch.
        self.assertLess(pooled["EKF"], pooled["UKF"])
        self.assertLess(pooled["UKF"], pooled["AUKF"])
        crows = {r["comparison"]: r for r in self.data["classical_paired_rows"]}
        for key in ("EKF vs AUKF", "EKF vs UKF", "UKF vs AUKF"):
            r = crows[key]
            # Positive mean gain favours the first-named (better) filter and
            # the paired bootstrap CI is reported.
            self.assertGreater(r["mean_paired_gain_m"], 0.0)
            self.assertLessEqual(
                r["paired_bootstrap_ci_low_m"], r["paired_bootstrap_ci_high_m"]
            )
            self.assertIn("wilcoxon_p", r)

    def test_cohort_never_beats_ekf(self) -> None:
        rows = {r["comparison"]: r for r in self.data["seed_cohort_rows"]}
        self.assertEqual(rows["vs EKF"]["n_seeds"], 15)
        # Honest negative: across all 15 trained seeds the learned residual
        # never beats the causal EKF, and the seed CI stays negative.
        self.assertEqual(rows["vs EKF"]["seed_wins"], 0)
        self.assertLess(rows["vs EKF"]["seed_bootstrap_ci_high_m"], 0.0)
        # Versus AUKF the cohort does not establish a positive either.
        self.assertLessEqual(rows["vs AUKF"]["mean_seed_observed_step_gain_m"], 0.0)
        for r in self.data["seed_cohort_rows"]:
            self.assertIn("descriptive diagnostic", r["wilcoxon_note"])

    def test_table_renders_two_panels(self) -> None:
        tex = mod.build_force_mismatch_significance_table(_ARTIFACT)
        self.assertIn("tab:force_mismatch_significance", tex)
        self.assertIn("Deterministic classical pairs", tex)
        self.assertIn("15-seed RGR-GF cohort", tex)


if __name__ == "__main__":
    unittest.main()
