"""Tests for the bounded public SP3 state-scoring campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_real_slr_sp3_state_scoring_campaign import (
    paired_gap_summary,
    select_lowest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_select_lowest_uses_validation_means_only() -> None:
    assert select_lowest({"b": 2.0, "a": 1.0}) == "a"
    assert select_lowest({"z": float("nan"), "x": 3.0}) == "x"


def test_paired_gap_summary_convention() -> None:
    rows = [
        {"rmse": {"candidate": 5.0, "reference": 3.0}},
        {"rmse": {"candidate": 2.0, "reference": 4.0}},
        {"rmse": {"candidate": 6.0, "reference": 1.0}},
    ]
    out = paired_gap_summary(
        rows, "candidate", "reference", field="rmse"
    )
    assert out["n"] == 3
    assert out["mean_gap_m"] == round(float(np.mean([2.0, -2.0, 5.0])), 2)
    assert out["n_a_lower_rmse"] == 1
    assert len(out["bootstrap95_mean_gap_m"]) == 2


def test_state_scoring_campaign_artifact_boundary_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_state_scoring_campaign"
        / "real_slr_sp3_state_scoring_campaign.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_state_scoring_campaign_v1"
    assert d["selection_integrity"]["test_set_information_used_for_selection"] is False
    cb = d["claim_boundary"]
    assert cb["defensible_status"] == "bounded_public_precise_reference_probe"
    assert cb["is_operational_validation"] is False
    assert cb["is_central_learned_vs_classical_validation"] is False
    assert cb["does_not_relabel_provenance_as_validation"] is True
    sparse = d["campaigns"]["sparse_slr_temporal_selector"]
    assert sparse["uses_public_crd_normal_points"] is True
    assert sparse["candidate_pool_includes_learned_model"] is False
    controlled = d["campaigns"]["controlled_sp3_dynamics_temporal_selector"]
    assert controlled["candidate_pool_includes_learned_model"] is True
