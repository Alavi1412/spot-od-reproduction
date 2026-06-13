"""Tests for the temporal public CRD/SP3 OD campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_real_slr_sp3_temporal_od_campaign import (
    LEARNED_LABEL,
    paired_gap_summary,
    select_lowest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_select_lowest_ignores_nonfinite_values() -> None:
    assert select_lowest({"b": 4.0, "a": 2.0}) == "a"
    assert select_lowest({"bad": float("nan"), "ok": 3.0}) == "ok"


def test_paired_gap_summary_reports_candidate_minus_reference() -> None:
    rows = [
        {"rmse": {"candidate": 7.0, "reference": 5.0}},
        {"rmse": {"candidate": 4.0, "reference": 6.0}},
        {"rmse": {"candidate": 10.0, "reference": 5.0}},
    ]
    out = paired_gap_summary(rows, "candidate", "reference", field="rmse")
    assert out["n"] == 3
    assert out["mean_gap_m"] == round(float(np.mean([2.0, -2.0, 5.0])), 2)
    assert out["n_a_lower_rmse"] == 1
    assert "positive means candidate a has larger held-out RMSE" in out[
        "gap_convention"
    ]


def test_temporal_od_campaign_artifact_boundary_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_od_campaign"
        / "real_slr_sp3_temporal_od_campaign.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_temporal_od_campaign_v1"
    assert d["selection_integrity"]["test_set_information_used_for_selection"] is False
    assert d["selection_integrity"]["calibrator_fit_uses_only_train_weeks"] is True
    assert d["selection_integrity"]["learned_ridge_selected_on_validation_only"] is True
    assert LEARNED_LABEL in d["selection"]["validation_mean_rms_m"]
    cb = d["claim_boundary"]
    assert cb["defensible_status"] == "bounded_public_real_measurement_od_probe"
    assert cb["can_be_used_as_central_external_validation"] is False
    assert cb["can_be_used_as_bounded_public_real_measurement_od_probe"] is True
    assert cb["is_operational_validation"] is False
    assert cb["is_powered_confirmatory_campaign"] is False
