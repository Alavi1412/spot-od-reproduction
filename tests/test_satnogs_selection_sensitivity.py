"""Tests for the SatNOGS validation/model-selection sensitivity analysis.

Cheap: validates the committed
``results/satnogs_selection_sensitivity.json`` schema and the honest
claims the manuscript makes from it: the SatNOGS validation item share
rises across curriculum stages (0%, ~22%, 40%); the headline negative is
invariant on the discriminative endpoints (best selectable checkpoint
still loses to the strongest classical reference); and the nominal split
is explicitly flagged as non-headline. No model is run.
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

_ARTIFACT = _ROOT / "results" / "satnogs_selection_sensitivity.json"


class SatnogsSelectionSensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("satnogs_selection_sensitivity.json not present")
        self.data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_item_share_increases_across_stages(self) -> None:
        share = self.data["satnogs_validation_item_share"]
        vals = list(share.values())
        # Stage 1 has no SatNOGS in validation; later stages do, increasing.
        self.assertAlmostEqual(vals[0], 0.0, places=4)
        self.assertGreater(vals[1], 0.0)
        self.assertGreater(vals[2], vals[1])
        self.assertAlmostEqual(vals[2], 0.40, places=2)

    def test_discriminative_endpoints_invariant(self) -> None:
        rows = {r["endpoint_label"]: r for r in self.data["rows"]}
        for label in ("Measurement-noise stress", "Controlled force-model mismatch"):
            r = rows[label]
            self.assertTrue(r["is_headline_endpoint"])
            self.assertEqual(r["seeds_beating_reference"], 0)
            self.assertFalse(r["best_case_beats_reference"])
            self.assertGreater(
                r["best_selectable_rgr_gf_obs_pos_rmse_m"],
                r["reference_obs_pos_rmse_m"],
            )
        self.assertTrue(self.data["headline_invariant_to_satnogs_selection_pathway"])

    def test_nominal_is_not_headline(self) -> None:
        nominal = next(
            r for r in self.data["rows"] if r["endpoint_label"].startswith("Nominal")
        )
        self.assertFalse(nominal["is_headline_endpoint"])

    def test_table_renders(self) -> None:
        tex = mod.build_satnogs_selection_sensitivity_table(_ARTIFACT)
        self.assertIn("tab:satnogs_selection_sensitivity", tex)
        self.assertIn("worst-case bound", tex)


if __name__ == "__main__":
    unittest.main()
