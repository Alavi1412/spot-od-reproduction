"""Focused checks for the structural-channel recoverability diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = (
    REPO_ROOT
    / "results"
    / "structural_channel_recoverability"
    / "structural_channel_recoverability.json"
)
TABLE = REPO_ROOT / "paper" / "tables" / "structural_channel_recoverability.tex"
SCRIPT = REPO_ROOT / "scripts" / "run_structural_channel_recoverability_diagnostic.py"
RULE = (
    REPO_ROOT
    / "release"
    / "predeclarations"
    / "structural_channel_recoverability_loop70.json"
)


def _ensure_artifacts() -> dict:
    if not RESULT_JSON.is_file() or not TABLE.is_file():
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--predeclared-rule",
                str(RULE),
                "--output-json",
                str(RESULT_JSON),
                "--output-table",
                str(TABLE),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=120,
        )
    return json.loads(RESULT_JSON.read_text())


def test_structural_channel_recoverability_artifacts_and_assertions() -> None:
    payload = _ensure_artifacts()
    rule = json.loads(RULE.read_text())

    dsa = payload["dsa_drag_scale"]
    dmc = payload["dmc_empirical_acceleration"]

    assert dsa["beta_relative_error"] <= rule["acceptance"]["dsa_beta_relative_tolerance"]
    assert dsa["rmse_m"]["DSA_EKF"] < dsa["rmse_m"]["EKF"]

    assert (
        dmc["empirical_acceleration_relative_l2_error"]
        <= rule["acceptance"]["dmc_acceleration_relative_l2_tolerance"]
    )
    assert dmc["rmse_m"]["DMC_EKF"] < dmc["rmse_m"]["EKF"]

    assert TABLE.is_file()
    table_text = TABLE.read_text()
    assert "not an operational orbit-determination setting" in table_text
    assert "not a primary endpoint result" in table_text
