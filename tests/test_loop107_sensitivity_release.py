from __future__ import annotations

import json
from pathlib import Path

from scripts.build_endpoint_selection_sensitivity import build_endpoint_selection_sensitivity


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_endpoint_sensitivity_uses_stored_observed_cis_and_excludes_k16() -> None:
    artifact = build_endpoint_selection_sensitivity(bootstrap_samples=200, bootstrap_seed=12345)

    assert artifact["record_inclusion_assertions"]["realization_counts_present"] == [8, 32]
    assert artifact["record_inclusion_assertions"]["k16_rows_present"] is False
    observed_rows = [
        row for row in artifact["rows"] if row["metric_id"] == "observed_step_position_rmse"
    ]
    all_step_rows = [
        row for row in artifact["rows"] if row["metric_id"] == "all_step_position_rmse"
    ]

    assert observed_rows
    assert all_step_rows
    assert {row["confidence_interval_source"] for row in observed_rows} == {
        "stored_original_endpoint_record"
    }
    assert {row["confidence_interval_source"] for row in all_step_rows} == {
        "sensitivity_recomputed_bootstrap"
    }
    assert all(row["bootstrap_seed_used"] is None for row in observed_rows)
    assert all(row["n_realizations"] in (8, 32) for row in artifact["rows"])


def test_manifest_source_indexes_loop107_sensitivity_artifacts() -> None:
    text = (REPO_ROOT / "scripts" / "build_supplementary_manifest.py").read_text(
        encoding="utf-8"
    )

    for needle in (
        "endpoint_and_tuning_sensitivity_audits",
        "endpoint_choice_sensitivity_audit",
        "pukf_tuning_comparability_sensitivity",
        "scripts/build_endpoint_selection_sensitivity.py",
        "scripts/run_pukf_hifi_tuning_sensitivity.py",
        "results/endpoint_selection_sensitivity/endpoint_selection_sensitivity.json",
        "results/pukf_tuning_sensitivity/pukf_hifi_grid_sensitivity.json",
        "paper/tables/endpoint_selection_sensitivity.tex",
        "paper/tables/pukf_tuning_sensitivity.tex",
    ):
        assert needle in text


def test_pukf_tuning_artifact_records_alignment_metadata() -> None:
    path = REPO_ROOT / "results" / "pukf_tuning_sensitivity" / "pukf_hifi_grid_sensitivity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    heldout = data["heldout_test"]
    alignment = heldout["population_alignment"]

    assert heldout["reference_classical_csv_sha256"]
    assert heldout["reference_result_sha256"]
    assert alignment["status"] == "pass"
    assert alignment["csv_rows"] == alignment["regenerated_population_trajectories"]
    assert alignment["trajectory_ids_available"] is False
    assert alignment["finite_selected_pukf_trajectories"] > 0
    assert "heldout_test" in data["population"]
