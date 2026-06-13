"""Tests for the validation-tuned KalmanNet-style learned-gain sweep.

Cheap: only the predeclared grid and the pure validation-loss selection
rule (with its deterministic tie-break) are exercised on tiny crafted
rows. No model is trained here. If the committed sweep artifact exists it
is checked for schema and for internal consistency (the recorded selected
hyperparameters must match the configuration the selection rule yields
from the recorded validation scores).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_kalmannet_gain_tuning_sweep",
    _ROOT / "scripts" / "run_kalmannet_gain_tuning_sweep.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

_ARTIFACT = _ROOT / "results" / "kalmannet_gain_tuning_sweep.json"


class PredeclaredGridTests(unittest.TestCase):
    def test_grid_is_small_and_well_formed(self) -> None:
        grid = mod.PREDECLARED_GRID
        self.assertGreaterEqual(len(grid), 3)
        self.assertLessEqual(len(grid), 6, "grid must stay small/predeclared")
        ids = [g["id"] for g in grid]
        self.assertEqual(len(ids), len(set(ids)), "grid ids must be unique")
        for g in grid:
            self.assertGreater(float(g["kalmannet_gain_scale"]), 0.0)
            self.assertGreater(float(g["kalmannet_correction_clip"]), 0.0)

    def test_default_config_is_in_grid(self) -> None:
        # The prior committed default (1e-3 / 5e-3) must be one grid point so
        # the tuned selection is a fair superset of the previous comparator.
        combos = {
            (float(g["kalmannet_gain_scale"]), float(g["kalmannet_correction_clip"]))
            for g in mod.PREDECLARED_GRID
        }
        self.assertIn((1.0e-3, 5.0e-3), combos)


class SelectionRuleTests(unittest.TestCase):
    def test_final_stage_val_score_ignores_non_final_and_non_finite(self) -> None:
        stages = [
            {"stage": "s1", "history": {"val_loss": [-1.0, -2.0]}},
            {"stage": "s3", "history": {"val_loss": [float("nan"), -0.5, -0.9]}},
        ]
        # Only the final stage counts; NaN is ignored; minimum is taken.
        self.assertAlmostEqual(mod.final_stage_val_score(stages), -0.9)

    def test_final_stage_val_score_all_non_finite_is_inf(self) -> None:
        stages = [{"stage": "s3", "history": {"val_loss": [float("inf")]}}]
        self.assertEqual(mod.final_stage_val_score(stages), float("inf"))

    def test_select_winner_picks_lowest_val_score(self) -> None:
        results = [
            {"id": "a", "kalmannet_gain_scale": 1e-3, "kalmannet_correction_clip": 5e-3, "val_score": -1.0},
            {"id": "b", "kalmannet_gain_scale": 2e-3, "kalmannet_correction_clip": 5e-3, "val_score": -1.5},
            {"id": "c", "kalmannet_gain_scale": 5e-4, "kalmannet_correction_clip": 5e-3, "val_score": -0.9},
        ]
        self.assertEqual(mod.select_winner(results), "b")

    def test_select_winner_tie_break_prefers_smaller_gain_then_clip(self) -> None:
        results = [
            {"id": "big_gain", "kalmannet_gain_scale": 2e-3, "kalmannet_correction_clip": 5e-3, "val_score": -1.0},
            {"id": "small_gain", "kalmannet_gain_scale": 5e-4, "kalmannet_correction_clip": 1e-2, "val_score": -1.0},
            {"id": "small_gain_tight", "kalmannet_gain_scale": 5e-4, "kalmannet_correction_clip": 5e-3, "val_score": -1.0},
        ]
        self.assertEqual(mod.select_winner(results), "small_gain_tight")

    def test_select_winner_requires_results(self) -> None:
        with self.assertRaises(ValueError):
            mod.select_winner([])


class CommittedArtifactConsistencyTests(unittest.TestCase):
    def test_artifact_schema_and_self_consistency(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("sweep artifact not present")
        data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        for key in (
            "selection_rule",
            "predeclared_grid",
            "selected_id",
            "selected_kalmannet_gain_scale",
            "selected_kalmannet_correction_clip",
            "results",
        ):
            self.assertIn(key, data)
        # The recorded selection must be reproducible from the recorded
        # validation scores via the pure selection rule (no test-set use).
        recomputed = mod.select_winner(data["results"])
        self.assertEqual(recomputed, data["selected_id"])
        sel = next(r for r in data["results"] if r["id"] == data["selected_id"])
        self.assertAlmostEqual(
            float(sel["kalmannet_gain_scale"]),
            float(data["selected_kalmannet_gain_scale"]),
        )
        self.assertAlmostEqual(
            float(sel["kalmannet_correction_clip"]),
            float(data["selected_kalmannet_correction_clip"]),
        )


if __name__ == "__main__":
    unittest.main()
