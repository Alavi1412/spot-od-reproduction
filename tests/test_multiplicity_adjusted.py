"""Tests for the familywise/FDR multiplicity-adjusted companion table.

Cheap: the Holm and Benjamini--Hochberg adjustment math is checked
against hand-computed values and standard properties (monotonicity, cap
at 1, never below the raw p, BH never above Holm). If the committed
artifact exists it is checked for schema and for internal consistency
with the pure adjustment functions.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_paper_assets",
    _ROOT / "scripts" / "build_paper_assets.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

_ARTIFACT = _ROOT / "results" / "multiplicity_adjusted.json"
_PAPER_TABLE = _ROOT / "paper" / "tables" / "multiplicity_adjusted.tex"


class HolmTests(unittest.TestCase):
    def test_known_values(self) -> None:
        # m=4; Holm multipliers are 4,3,2,1 on the sorted p-values with a
        # cumulative max enforcing monotonicity, all capped at 1.
        p = [0.01, 0.04, 0.03, 0.005]
        adj = mod.holm_adjusted(p)
        # sorted: 0.005(*4=0.02), 0.01(*3=0.03), 0.03(*2=0.06), 0.04(*1=0.04->0.06)
        self.assertAlmostEqual(adj[3], 0.02)
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertAlmostEqual(adj[2], 0.06)
        self.assertAlmostEqual(adj[1], 0.06)

    def test_capped_and_not_below_raw(self) -> None:
        p = [0.5, 0.9, 0.8]
        adj = mod.holm_adjusted(p)
        for raw, a in zip(p, adj):
            self.assertLessEqual(a, 1.0)
            self.assertGreaterEqual(a + 1e-12, raw)


class BenjaminiHochbergTests(unittest.TestCase):
    def test_known_values(self) -> None:
        p = [0.01, 0.04, 0.03, 0.005]
        adj = mod.benjamini_hochberg_adjusted(p)
        # sorted ranks 1..4: 0.005*4/1=0.02, 0.01*4/2=0.02, 0.03*4/3=0.04,
        # 0.04*4/4=0.04 with step-up monotone min from the largest down.
        self.assertAlmostEqual(adj[3], 0.02)
        self.assertAlmostEqual(adj[0], 0.02)
        self.assertAlmostEqual(adj[2], 0.04)
        self.assertAlmostEqual(adj[1], 0.04)

    def test_bh_never_exceeds_holm(self) -> None:
        p = [0.001, 0.2, 0.04, 0.5, 0.009, 0.3]
        holm = mod.holm_adjusted(p)
        bh = mod.benjamini_hochberg_adjusted(p)
        for h, b in zip(holm, bh):
            self.assertLessEqual(b, h + 1e-12)


class TableBuilderTests(unittest.TestCase):
    def test_builds_latex_and_writes_artifact(self) -> None:
        records = [
            {"source": "Seed-aware", "comparison": "RGR-GF Stress vs UKF", "p": 3.05e-05},
            {"source": "Seed-aware", "comparison": "RGR-GF Stress vs AUKF", "p": 0.97},
            {"source": "Seed observed", "comparison": "Stress vs UKF", "p": 3.05e-05},
        ]
        # Build into a scratch artifact path. The canonical production
        # artifact must never be mutated by the test suite, so its bytes
        # are snapshotted and re-checked after the build.
        canonical_before = (
            _ARTIFACT.read_bytes() if _ARTIFACT.exists() else None
        )
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "multiplicity_adjusted.json"
            latex = mod.build_multiplicity_adjusted_table(
                records, artifact_path=scratch
            )
            self.assertIn("tab:multiplicity_adjusted", latex)
            self.assertIn("Holm", latex)
            self.assertIn("Benjamini--Hochberg", latex)
            self.assertIn("descriptive diagnostic", latex)
            art = json.loads(scratch.read_text(encoding="utf-8"))
            self.assertEqual(art["family_size"], 3)
            self.assertEqual(len(art["records"]), 3)
        # Regression guard for the loop-34 integration bug: building with an
        # explicit test family must leave the canonical artifact untouched.
        canonical_after = (
            _ARTIFACT.read_bytes() if _ARTIFACT.exists() else None
        )
        self.assertEqual(canonical_before, canonical_after)

    def test_empty_family_is_safe(self) -> None:
        self.assertIn("unavailable", mod.build_multiplicity_adjusted_table([]))


class CommittedArtifactTests(unittest.TestCase):
    def test_artifact_schema_and_consistency(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("multiplicity artifact not present")
        art = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        for key in ("family_size", "records", "min_raw_p", "min_holm_p", "min_bh_p"):
            self.assertIn(key, art)
        self.assertEqual(art["family_size"], len(art["records"]))
        raw = [r["p"] for r in art["records"]]
        holm = mod.holm_adjusted(raw)
        bh = mod.benjamini_hochberg_adjusted(raw)
        for i, r in enumerate(art["records"]):
            self.assertAlmostEqual(r["holm_p"], holm[i], places=10)
            self.assertAlmostEqual(r["bh_p"], bh[i], places=10)

    def test_artifact_family_size_matches_paper_table(self) -> None:
        # The committed JSON companion must agree with the displayed
        # paper table's family size ($m=...$). This catches the loop-34
        # regression where pytest overwrote the canonical 14-record
        # artifact with a 3-record test fixture.
        if not _ARTIFACT.exists() or not _PAPER_TABLE.exists():
            self.skipTest("multiplicity artifact or paper table not present")
        tex = _PAPER_TABLE.read_text(encoding="utf-8")
        m = re.search(r"\$m=(\d+)\$", tex)
        self.assertIsNotNone(m, "paper table is missing the $m=...$ family size")
        family_size = int(m.group(1))
        art = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(art["family_size"], family_size)
        self.assertEqual(len(art["records"]), family_size)
        # Displayed table body rows must equal the family size too.
        body_rows = [
            ln for ln in tex.splitlines()
            if "&" in ln and "\\\\" in ln and "Source table" not in ln
        ]
        self.assertEqual(len(body_rows), family_size)


if __name__ == "__main__":
    unittest.main()
