"""Unit tests for the deterministic precise-SLR-reduction component audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gnn_state_estimation import frames, slr

import importlib.util

_MOD_PATH = Path(__file__).resolve().parent.parent / "scripts" / "audit_correction_components.py"
_spec = importlib.util.spec_from_file_location("audit_correction_components", _MOD_PATH)
audit_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(audit_mod)


def test_marini_murray_zenith_in_band():
    rows = audit_mod.audit_marini_murray()
    zenith_row = next(r for r in rows if "zenith" in r["component"])
    assert zenith_row["passed"], zenith_row


def test_centre_of_mass_exact_lageos_value():
    rows = audit_mod.audit_centre_of_mass()
    assert rows[0]["passed"]
    assert math.isclose(rows[0]["computed_m"], 0.251, abs_tol=1e-9)


def test_shapiro_one_way_in_millimetre_band():
    rows = audit_mod.audit_shapiro_delay()
    assert rows[0]["passed"]
    assert 5e-3 <= rows[0]["computed_m"] <= 3e-2


def test_iau76_precession_theta_one_century():
    rows = audit_mod.audit_frame_components()
    theta_row = next(r for r in rows if "theta_A" in r["component"])
    assert theta_row["passed"], theta_row
    # Lieske 1977 / Vallado 2013 Table 3-7: theta_A = 2004.3109'' / century.
    assert 1995.0 <= theta_row["computed_m"] <= 2015.0


def test_full_audit_runs_and_passes(tmp_path: Path):
    payload = audit_mod.run_audit(Path(__file__).resolve().parent.parent)
    assert payload["all_passed"]
    assert payload["n_components"] == payload["n_passed"]
    assert payload["schema_version"] == "correction_component_audit_v1"
    # Persisted artifact must already exist after main run.
    json_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "correction_component_audit"
        / "correction_component_audit.json"
    )
    disk_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert disk_payload["all_passed"]
    # Every component row carries the expected schema fields.
    for r in disk_payload["components"]:
        assert {"component", "expected_sign", "expected_abs_range_m",
                "expected_source", "computed_m", "passed"}.issubset(r.keys())


def test_paper_table_artifact_present():
    tex_path = (
        Path(__file__).resolve().parent.parent
        / "paper"
        / "tables"
        / "correction_component_audit.tex"
    )
    assert tex_path.exists()
    text = tex_path.read_text(encoding="utf-8")
    assert "tab:correction_component_audit" in text
    assert "pass" in text or "FAIL" in text
    assert "Marini" in text
    assert "Shapiro" in text
    assert "IAU-76" in text
