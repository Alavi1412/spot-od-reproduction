"""Tests for the temporal public OD selection-stability audit."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_real_slr_sp3_temporal_od_campaign import LEARNED_LABEL
from scripts.run_real_slr_sp3_temporal_selection_stability import (
    bootstrap_selection,
    candidate_margin,
    select_on_validation_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _toy_validation():
    learned_rows = [
        {
            "arc_id": "arc-1",
            "learned_ridge_rmse_m": {"1e+00": 4.0, "1e+03": 9.0},
        },
        {
            "arc_id": "arc-2",
            "learned_ridge_rmse_m": {"1e+00": 6.0, "1e+03": 8.0},
        },
    ]
    hifi_by_arc = {
        "arc-1": {
            "held_out_position_rmse_m": {
                "UKF (compact)": 5.0,
                "UKF (higher-fidelity)": 7.0,
                "AUKF (higher-fidelity)": 9.0,
            }
        },
        "arc-2": {
            "held_out_position_rmse_m": {
                "UKF (compact)": 8.0,
                "UKF (higher-fidelity)": 9.0,
                "AUKF (higher-fidelity)": 10.0,
            }
        },
    }
    return learned_rows, hifi_by_arc


def test_select_on_validation_indices_uses_validation_rows_only() -> None:
    learned_rows, hifi_by_arc = _toy_validation()
    out = select_on_validation_indices(
        learned_rows,
        hifi_by_arc,
        indices=[0, 1],
    )
    assert out["selected_learned_ridge_lambda"] == "1e+00"
    assert out["selected_candidate"] == LEARNED_LABEL
    assert out["winner_margin_to_runner_up_m"] == 1.5


def test_bootstrap_selection_counts_sum_to_resamples() -> None:
    learned_rows, hifi_by_arc = _toy_validation()
    out = bootstrap_selection(
        learned_rows,
        hifi_by_arc,
        n_resamples=25,
        seed=3,
    )
    assert sum(out["candidate_selection_counts"].values()) == 25
    assert sum(out["learned_ridge_selection_counts"].values()) == 25
    assert LEARNED_LABEL in out["candidate_selection_counts"]


def test_candidate_margin_reports_winner_and_runner_up() -> None:
    out = candidate_margin({"b": 4.0, "a": 2.5, "bad": None})
    assert out["winner"] == "a"
    assert out["runner_up"] == "b"
    assert out["winner_margin_to_runner_up_m"] == 1.5


def test_selection_stability_artifact_boundary_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_selection_stability"
        / "real_slr_sp3_temporal_selection_stability.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_temporal_selection_stability_v1"
    assert d["predeclared"]["test_set_information_used_for_selection"] is False
    assert d["negative_control_sentinels"][
        "test_oracle_is_forbidden_for_model_selection"
    ] is True
    cb = d["claim_boundary"]
    assert cb["can_be_used_as_central_external_validation"] is False
    assert cb["can_be_used_as_public_temporal_probe_stability_audit"] is True
    assert cb["does_not_select_or_tune_on_test"] is True
