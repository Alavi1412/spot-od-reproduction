"""Tests for the practical-significance-floor sensitivity sweep."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_MOD_PATH = _REPO / "scripts" / "build_floor_sensitivity_sweep.py"
_spec = importlib.util.spec_from_file_location("build_floor_sensitivity_sweep", _MOD_PATH)
floor_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(floor_mod)


def test_floor_sweep_artifact_present_and_well_formed():
    art_path = _REPO / "results" / "floor_sensitivity_sweep" / "floor_sensitivity_sweep.json"
    assert art_path.exists()
    art = json.loads(art_path.read_text(encoding="utf-8"))
    assert art["schema_version"] == "floor_sensitivity_sweep_v1"
    assert art["swept_floors_pct"] == [1.0, 2.0, 3.0, 5.0]
    assert art["predeclared_floor_pct"] == 3.0
    assert len(art["rows"]) == 4


def test_headline_finding_preserved_under_all_floors():
    art_path = _REPO / "results" / "floor_sensitivity_sweep" / "floor_sensitivity_sweep.json"
    art = json.loads(art_path.read_text(encoding="utf-8"))
    for row in art["rows"]:
        assert row["qualitative_conclusion_unchanged"], row
    assert art["summary"]["n_floors_with_unchanged_conclusion"] == art["summary"]["n_floors"]


def test_swept_floor_scales_with_baseline():
    art_path = _REPO / "results" / "floor_sensitivity_sweep" / "floor_sensitivity_sweep.json"
    art = json.loads(art_path.read_text(encoding="utf-8"))
    row_3pct = next(r for r in art["rows"] if r["floor_pct"] == 3.0)
    for sc in row_3pct["per_scenario"]:
        expected = round(0.03 * sc["baseline_m"], 2)
        assert abs(sc["floor_m"] - expected) < 0.05, (sc, expected)


def test_paper_table_present_and_lists_all_floors():
    tex = (_REPO / "paper" / "tables" / "floor_sensitivity_sweep.tex").read_text(encoding="utf-8")
    assert "tab:floor_sensitivity_sweep" in tex
    for floor in ("1\\%", "2\\%", "3\\%", "5\\%"):
        assert floor in tex
    assert "Nominal" in tex
    assert "Measurement-noise stress" in tex
    assert "Controlled force-model mismatch" in tex
