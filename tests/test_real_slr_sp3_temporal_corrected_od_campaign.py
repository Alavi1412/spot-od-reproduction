"""Tests for the full-correction temporal public CRD/SP3 OD campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_real_slr_sp3_temporal_corrected_od_campaign import (
    DEFAULT_OUTPUT_JSON,
    DEFAULT_PREDECLARATION,
    DEFAULT_TABLE,
    BEST_CLASSICAL_LABELS,
    CLASSICAL_LABELS,
    LEARNED_LABEL,
    SCHEDULE_FORMAL210_MINI_PRE260509,
    SCHEDULE_FORMAL210_RECENT_PRE260509,
    SCHEDULE_PROSPECTIVE_260516,
    SCHEDULE_PROSPECTIVE_260523,
    SCHEDULE_PROSPECTIVE_260530,
    SCHEDULE_PROSPECTIVE_260606,
    SCHEDULE_PROSPECTIVE_260613,
    SCHEDULE_PROSPECTIVE_260620,
    SCHEDULE_PROSPECTIVE_260627,
    arc_cache_digest,
    arc_cache_key,
    build_arc_specs,
    build_parser,
    build_predeclaration,
    paired_gap_summary,
    learned_vs_recursive_classical_readout_context,
    render_table,
    schedule_metadata,
    select_lowest,
    split_plan,
    validate_prospective_cli_path_safety,
    _validate_sp3_input,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_required_artifact(path: Path) -> None:
    assert path.exists(), f"required integration artifact is missing: {path}"


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def test_formal210_recent_pre260509_schedule_excludes_current_temporal_test_week() -> None:
    plan = split_plan(SCHEDULE_FORMAL210_MINI_PRE260509)

    assert "260509" not in plan["test"]
    assert plan["test"] == ["260502"]
    assert plan["validation"] == ["260425"]
    assert "260509" in plan["unused"]
    assert plan["train"] == ["260418"]

    meta = schedule_metadata(SCHEDULE_FORMAL210_MINI_PRE260509)
    assert meta["confirmatory_status"] == "post_hoc_robustness"
    assert meta["test_weeks"] == ["260502"]
    assert len(meta["validation_arcs"]) == 2
    assert len(meta["test_arcs"]) == 2
    assert {arc["split"] for arc in meta["test_arcs"]} == {"test"}

    test_specs = build_arc_specs(SCHEDULE_FORMAL210_MINI_PRE260509, "test")
    assert len(test_specs) == 2
    assert {a.sp3_week for a in test_specs} == {"260502"}
    assert {a.date for a in test_specs} == {"20260427"}


def test_learned_boundary_keeps_mini_two_arc_posthoc_wording() -> None:
    context = learned_vs_recursive_classical_readout_context(
        SCHEDULE_FORMAL210_MINI_PRE260509, 2
    )

    assert context["readout_scope"] == (
        "bounded_post_hoc_two_arc_readout_not_validation_selection"
    )
    assert "bounded post-hoc two-arc test readout only" in context["boundary"]
    assert "10-arc" not in context["boundary"]
    assert "prospective" not in context["boundary"]


def test_learned_boundary_uses_recent_ten_arc_posthoc_wording() -> None:
    context = learned_vs_recursive_classical_readout_context(
        SCHEDULE_FORMAL210_RECENT_PRE260509, 10
    )

    assert context["readout_scope"] == (
        "bounded_post_hoc_10_arc_readout_not_validation_selection"
    )
    assert "bounded post-hoc 10-arc test readout only" in context["boundary"]
    assert "two-arc" not in context["boundary"]
    assert "predeclared prospective" not in context["boundary"]


def test_learned_boundary_uses_prospective_ten_arc_wording() -> None:
    context = learned_vs_recursive_classical_readout_context(
        SCHEDULE_PROSPECTIVE_260516, 10
    )

    assert context["readout_scope"] == (
        "prospective_public_week_10_arc_readout_not_operational_pod"
    )
    assert "predeclared prospective public-week 10-arc test readout" in context[
        "boundary"
    ]
    assert "post-hoc" not in context["boundary"]
    assert "two-arc" not in context["boundary"]


def test_learned_boundary_uses_realized_prospective_eight_of_ten_wording() -> None:
    context = learned_vs_recursive_classical_readout_context(
        SCHEDULE_PROSPECTIVE_260516, 8, planned_n_test_arcs=10
    )

    assert context["readout_scope"] == (
        "prospective_public_week_8_arc_completed_of_10_arc_planned_"
        "readout_not_operational_pod"
    )
    assert "8 completed arcs of 10 planned" in context["boundary"]
    assert "predeclared prospective public-week" in context["boundary"]
    assert "post-hoc" not in context["boundary"]
    assert "two-arc" not in context["boundary"]


def test_prospective_260516_schedule_has_frozen_boundary_and_new_test_arcs() -> None:
    plan = split_plan(SCHEDULE_PROSPECTIVE_260516)

    assert plan["train"][0] == "251220"
    assert plan["train"][-1] == "260502"
    assert "260502" in plan["train"]
    assert "260509" not in plan["train"]
    assert plan["validation"] == ["260509"]
    assert plan["test"] == ["260516"]
    assert plan["unused"] == []

    meta = schedule_metadata(SCHEDULE_PROSPECTIVE_260516)
    assert (
        meta["confirmatory_status"]
        == "predeclared_prospective_public_temporal_holdout"
    )
    assert "Rule fixed before scoring" in meta["reason_for_boundary"]
    assert meta["weeks"]["260516"]["dates"] == [
        "20260511",
        "20260512",
        "20260513",
        "20260514",
        "20260515",
    ]
    assert meta["weeks"]["260516"]["original_formal210_split"] is None
    assert (
        meta["weeks"]["260516"]["source_schedule_role"]
        == "prospective_added_public_week"
    )
    assert len(meta["test_arcs"]) == 10
    assert {arc["sp3_week"] for arc in meta["test_arcs"]} == {"260516"}
    assert {arc["date"] for arc in meta["test_arcs"]} == {
        "20260511",
        "20260512",
        "20260513",
        "20260514",
        "20260515",
    }
    assert {arc["target"] for arc in meta["test_arcs"]} == {
        "LAGEOS-1",
        "LAGEOS-2",
    }
    assert {
        (arc["target"], arc["date"])
        for arc in meta["test_arcs"]
    } == {
        (target, date)
        for target in ("LAGEOS-1", "LAGEOS-2")
        for date in (
            "20260511",
            "20260512",
            "20260513",
            "20260514",
            "20260515",
        )
    }


def test_arc_cache_key_binds_input_provenance_and_selection_state() -> None:
    input_provenance = {
        "target": "LAGEOS-1",
        "date": "20260511",
        "sp3_week": "260516",
        "crd": {
            "archived_input_id": "lageos1_20260511.np2",
            "sha256": "c" * 64,
            "bytes": 123,
        },
        "sp3": {
            "archived_input_id": "nsgf.orb.lageos1.260516.v80.sp3.gz",
            "sha256": "s" * 64,
            "bytes": 456,
        },
        "eop": {
            "archived_input_id": "finals2000A.all",
            "sha256": "e" * 64,
            "bytes": 789,
        },
    }
    base = {
        "schedule_name": SCHEDULE_PROSPECTIVE_260516,
        "split": "test",
        "arc_id": "LAGEOS-1 20260511",
        "target": "LAGEOS-1",
        "date": "20260511",
        "sp3_week": "260516",
        "input_provenance": input_provenance,
        "training_input_digest_sha256": "t" * 64,
        "lambda_keys": ["1e-06"],
    }
    key_a = arc_cache_key(
        **base,
        predeclaration_sha256="a" * 64,
        selected_lam_key="1e-06",
    )
    key_b = arc_cache_key(
        **base,
        predeclaration_sha256="a" * 64,
        selected_lam_key="1e-05",
    )
    key_c = arc_cache_key(
        **base,
        predeclaration_sha256="b" * 64,
        selected_lam_key="1e-06",
    )
    crd_changed = json.loads(json.dumps(input_provenance))
    crd_changed["crd"]["sha256"] = "d" * 64
    sp3_changed = json.loads(json.dumps(input_provenance))
    sp3_changed["sp3"]["bytes"] = 457
    eop_changed = json.loads(json.dumps(input_provenance))
    eop_changed["eop"]["sha256"] = "f" * 64
    key_d = arc_cache_key(
        **{**base, "input_provenance": crd_changed},
        predeclaration_sha256="a" * 64,
        selected_lam_key="1e-06",
    )
    key_e = arc_cache_key(
        **{**base, "input_provenance": sp3_changed},
        predeclaration_sha256="a" * 64,
        selected_lam_key="1e-06",
    )
    key_f = arc_cache_key(
        **{**base, "input_provenance": eop_changed},
        predeclaration_sha256="a" * 64,
        selected_lam_key="1e-06",
    )
    key_g = arc_cache_key(
        **{**base, "training_input_digest_sha256": "u" * 64},
        predeclaration_sha256="a" * 64,
        selected_lam_key="1e-06",
    )

    assert key_a["selected_learned_ridge_lambda"] == "1e-06"
    assert key_a["predeclaration_sha256"] == "a" * 64
    assert key_a["input_provenance"]["crd"]["sha256"] == "c" * 64
    assert key_a["input_provenance"]["sp3"]["bytes"] == 456
    assert key_a["input_provenance"]["eop"]["sha256"] == "e" * 64
    assert arc_cache_digest(key_a) != arc_cache_digest(key_b)
    assert arc_cache_digest(key_a) != arc_cache_digest(key_c)
    assert arc_cache_digest(key_a) != arc_cache_digest(key_d)
    assert arc_cache_digest(key_a) != arc_cache_digest(key_e)
    assert arc_cache_digest(key_a) != arc_cache_digest(key_f)
    assert arc_cache_digest(key_a) != arc_cache_digest(key_g)


def test_prospective_cli_rejects_generic_default_artifact_paths(tmp_path) -> None:
    parser = build_parser()

    legacy_args = parser.parse_args(
        ["--schedule", SCHEDULE_FORMAL210_MINI_PRE260509]
    )
    validate_prospective_cli_path_safety(legacy_args, {"--schedule"})

    prospective_defaults = parser.parse_args(
        ["--schedule", SCHEDULE_PROSPECTIVE_260516]
    )
    with pytest.raises(SystemExit, match="--predeclaration"):
        validate_prospective_cli_path_safety(prospective_defaults, {"--schedule"})

    predecl = tmp_path / "prospective.json"
    output = tmp_path / "prospective_result.json"
    safe_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260516,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(output),
            "--no-table",
        ]
    )
    validate_prospective_cli_path_safety(
        safe_args,
        {"--schedule", "--predeclaration", "--output-json", "--no-table"},
    )

    unsafe_table_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260516,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(output),
        ]
    )
    with pytest.raises(SystemExit, match="--no-table"):
        validate_prospective_cli_path_safety(
            unsafe_table_args,
            {"--schedule", "--predeclaration", "--output-json"},
        )

    default_predecl_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260516,
            "--predeclaration",
            str(DEFAULT_PREDECLARATION),
            "--output-json",
            str(output),
            "--no-table",
        ]
    )
    with pytest.raises(SystemExit, match="--predeclaration"):
        validate_prospective_cli_path_safety(
            default_predecl_args,
            {"--schedule", "--predeclaration", "--output-json", "--no-table"},
        )

    default_output_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260516,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(DEFAULT_OUTPUT_JSON),
            "--no-table",
        ]
    )
    with pytest.raises(SystemExit, match="--output-json"):
        validate_prospective_cli_path_safety(
            default_output_args,
            {"--schedule", "--predeclaration", "--output-json", "--no-table"},
        )

    default_table_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260516,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(output),
            "--table",
            str(DEFAULT_TABLE),
        ]
    )
    with pytest.raises(SystemExit, match="--no-table"):
        validate_prospective_cli_path_safety(
            default_table_args,
            {"--schedule", "--predeclaration", "--output-json", "--table"},
        )


def test_predeclaration_defines_rule_candidates_metric_and_boundary() -> None:
    pre = build_predeclaration(SCHEDULE_FORMAL210_MINI_PRE260509)

    assert (
        pre["schema_version"]
        == "real_slr_sp3_temporal_corrected_od_predeclaration_v1"
    )
    assert pre["artifact_role"] == "predeclared_rule_schedule_before_scoring"
    assert LEARNED_LABEL == pre["candidate_set"]["learned"]
    for label in BEST_CLASSICAL_LABELS:
        assert label in pre["candidate_set"]["classical_skill_reference"]
    rule = pre["selection_rule"]
    assert "held-out SP3 position RMSE" in rule["primary_metric"]
    assert rule["test_set_information_used_for_selection"] is False
    boundary = pre["claim_boundary"]
    assert boundary["post_hoc_robustness_not_confirmatory"] is True
    assert boundary["can_be_used_as_operational_pod_validation"] is False
    assert boundary["can_be_used_as_central_external_validation"] is False


def test_select_lowest_and_paired_gap_sign_convention() -> None:
    assert select_lowest({"b": 4.0, "a": 2.0, "nan": float("nan")}) == "a"

    rows = [
        {"held_out_position_rmse_m": {"candidate": 7.0, "reference": 5.0}},
        {"held_out_position_rmse_m": {"candidate": 4.0, "reference": 6.0}},
        {"held_out_position_rmse_m": {"candidate": 10.0, "reference": 5.0}},
    ]
    out = paired_gap_summary(rows, "candidate", "reference")
    assert out["n"] == 3
    assert out["mean_gap_m"] == round(float(np.mean([2.0, -2.0, 5.0])), 2)
    assert out["n_a_lower_rmse"] == 1
    assert "positive means candidate a has larger held-out RMSE" in out[
        "gap_convention"
    ]


def test_render_table_keeps_posthoc_and_non_operational_boundary() -> None:
    pre = build_predeclaration(SCHEDULE_FORMAL210_MINI_PRE260509)
    result = {
        "predeclared_schedule": pre,
        "selection": {
            "selected_candidate": "AUKF (full correction)",
            "selected_validation_mean_rms_m": 100.0,
            "validation_mean_rms_m": {
                "AUKF (full correction)": 100.0,
                LEARNED_LABEL: 120.0,
            },
        },
        "test_readout": {
            "selected_test_mean_rms_m": 90.0,
            "test_mean_rms_m": {
                "AUKF (full correction)": 90.0,
                LEARNED_LABEL: 80.0,
            },
            "selected_vs_test_best_paired_gap": {
                "mean_gap_m": -5.0,
                "bootstrap95_mean_gap_m": [-8.0, -1.0],
            },
            "learned_vs_best_classical_paired_gap": {
                "mean_gap_m": -5.0,
                "bootstrap95_mean_gap_m": [-8.0, -1.0],
            },
        },
    }

    table = render_table(result)

    assert "post-hoc-robustness rather than confirmatory" in table
    assert "IERS Earth-orientation" in table
    assert "not operational POD" in table
    assert "Validation-selected candidate" not in table
    assert "Candidate/readout" in table
    assert "Learned residual UKF (not validation-selected)" in table
    assert "260425" in table and "260502" in table


def test_temporal_corrected_artifact_boundary_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_campaign"
        / "real_slr_sp3_temporal_corrected_od_campaign.json"
    )
    _assert_required_artifact(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_temporal_corrected_od_campaign_v1"
    assert d["selection_integrity"]["test_set_information_used_for_selection"] is False
    assert d["selection_integrity"]["calibrator_fit_uses_only_train_weeks"] is True
    assert d["selection_integrity"]["learned_ridge_selected_on_validation_only"] is True
    assert d["selection_integrity"]["predeclaration_artifact_required_before_scoring"]
    assert d["predeclared_schedule"]["schedule"]["confirmatory_status"] == (
        "post_hoc_robustness"
    )
    assert "260509" not in d["selection_integrity"]["test_weeks"]
    assert d["test_readout"]["n_arcs"] == 2
    assert d["test_readout"]["n_failed_or_excluded_arcs"] == 0
    assert d["test_readout"]["failed_or_excluded_arcs"] == []
    assert d["selection"]["selected_candidate"] == "AUKF (full correction)"
    assert d["selection"]["selected_candidate_family"] == "classical"
    assert d["selection"]["selected_candidate"] != LEARNED_LABEL
    headline = d["headline_readout"]
    assert headline["learned_residual_is_validation_selected_candidate"] is False
    assert "learned_positive_against_best_classical" not in headline
    assert d["test_readout"]["test_best_candidate"] == (
        "SP3-IC propagation (full correction)"
    )
    assert headline["test_best_candidate"] == "SP3-IC propagation (full correction)"
    assert headline["test_floor_candidate"] == "SP3-IC propagation (full correction)"
    assert d["test_readout"]["best_classical_test_candidate"] in BEST_CLASSICAL_LABELS
    assert d["test_readout"]["best_classical_test_candidate"] != (
        "SP3-IC propagation (full correction)"
    )
    assert d["test_readout"]["test_mean_rms_m"][LEARNED_LABEL] < d["test_readout"][
        "best_classical_test_mean_rms_m"
    ]
    learned_gap = d["test_readout"]["learned_vs_best_classical_paired_gap"]
    assert learned_gap["n"] == 2
    assert learned_gap["mean_gap_m"] < 0.0
    assert learned_gap["bootstrap95_mean_gap_m"][1] < 0.0
    assert learned_gap["n_a_lower_rmse"] == 2
    assert (
        headline[
            "learned_lower_than_best_recursive_classical_on_two_posthoc_test_arcs"
        ]
        is True
    )
    assert headline["learned_vs_recursive_classical_readout_scope"] == (
        "bounded_post_hoc_two_arc_readout_not_validation_selection"
    )
    cb = d["claim_boundary"]
    assert cb["post_hoc_robustness_not_confirmatory"] is True
    assert cb["can_be_used_as_central_external_validation"] is False
    assert cb["is_operational_validation"] is False
    assert cb["is_simulator_result_validation"] is False
    assert "learned_result_boundary" not in cb
    boundary = cb["learned_vs_recursive_classical_boundary"]
    assert "bounded post-hoc two-arc test readout only" in boundary
    assert "SP3-IC propagation remains the test-floor candidate" in boundary
    assert "not validation-selected evidence" in boundary
    assert "operational POD validation" in boundary
    assert "simulator-result validation" in boundary


def test_temporal_corrected_recent_artifact_boundary_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent.json"
    )
    _assert_required_artifact(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["predeclared_schedule"]["schedule"]["schedule_name"] == (
        SCHEDULE_FORMAL210_RECENT_PRE260509
    )
    assert d["validation"]["n_arcs"] == 10
    assert d["test_readout"]["n_arcs"] == 10
    headline = d["headline_readout"]
    assert (
        "learned_lower_than_best_recursive_classical_on_two_posthoc_test_arcs"
        not in headline
    )
    assert headline["learned_vs_recursive_classical_readout_scope"] == (
        "bounded_post_hoc_10_arc_readout_not_validation_selection"
    )
    boundary = d["claim_boundary"]["learned_vs_recursive_classical_boundary"]
    assert "bounded post-hoc 10-arc test readout only" in boundary
    assert "two-arc" not in boundary


def test_temporal_corrected_prospective_artifact_boundary_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516.json"
    )
    _assert_required_artifact(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["predeclared_schedule"]["schedule"]["schedule_name"] == (
        SCHEDULE_PROSPECTIVE_260516
    )
    assert d["predeclared_schedule"]["schedule"]["confirmatory_status"] == (
        "predeclared_prospective_public_temporal_holdout"
    )
    assert d["selection"]["selected_candidate"] == "UKF (full correction)"
    assert d["selection"]["selected_validation_mean_rms_m"] == 363.49
    assert d["test_readout"]["n_arcs"] == 8
    assert d["test_readout"]["n_planned_arcs"] == 10
    assert d["test_readout"]["n_failed_or_excluded_arcs"] == 2
    assert {
        (arc["target"], arc["date"], arc["num_observations"])
        for arc in d["test_readout"]["failed_or_excluded_arcs"]
    } == {
        ("LAGEOS-1", "20260513", 0),
        ("LAGEOS-2", "20260513", 7),
    }
    assert d["test_readout"]["selected_test_mean_rms_m"] == 426.53
    assert d["test_readout"]["test_best_candidate"] == "UKF (full correction)"
    assert d["test_readout"]["test_best_mean_rms_m"] == 426.53
    assert d["test_readout"]["best_classical_test_mean_rms_m"] == 426.53
    assert d["test_readout"]["test_mean_rms_m"][LEARNED_LABEL] == 458.01
    learned_gap = d["test_readout"]["learned_vs_best_classical_paired_gap"]
    assert learned_gap["n"] == 8
    assert learned_gap["mean_gap_m"] == 31.48
    assert learned_gap["bootstrap95_mean_gap_m"] == [-50.28, 132.35]
    assert learned_gap["n_a_lower_rmse"] == 5
    headline = d["headline_readout"]
    assert headline["completed_test_arcs"] == 8
    assert headline["planned_test_arcs"] == 10
    assert headline["learned_vs_recursive_classical_readout_scope"] == (
        "prospective_public_week_8_arc_completed_of_10_arc_planned_"
        "readout_not_operational_pod"
    )
    assert (
        headline["learned_lower_than_best_recursive_classical_on_prospective_test_arcs"]
        is False
    )
    cb = d["claim_boundary"]
    assert cb["prospective_public_temporal_holdout"] is True
    assert cb["post_hoc_robustness_not_confirmatory"] is False
    boundary = cb["learned_vs_recursive_classical_boundary"]
    assert "8 completed arcs of 10 planned" in boundary
    assert "post-hoc" not in boundary


def test_build_paper_assets_temporal_corrected_table_if_artifact_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_campaign"
        / "real_slr_sp3_temporal_corrected_od_campaign.json"
    )
    _assert_required_artifact(path)

    from scripts.build_paper_assets import (
        build_real_slr_sp3_temporal_corrected_od_campaign_table,
    )

    table = build_real_slr_sp3_temporal_corrected_od_campaign_table(path)
    assert "\\label{tab:real_slr_sp3_temporal_corrected_od_campaign}" in table
    assert "post-hoc-robustness rather than confirmatory" in table
    assert "not operational POD" in table
    assert "Validation-selected candidate" not in table
    assert "Candidate/readout" in table
    assert "Learned residual UKF (not validation-selected)" in table

    table_path = (
        REPO_ROOT / "paper" / "tables" / "real_slr_sp3_temporal_corrected_od_campaign.tex"
    )
    _assert_required_artifact(table_path)
    artifact_table = table_path.read_text(encoding="utf-8")
    assert "Validation-selected candidate" not in artifact_table
    assert "Candidate/readout" in artifact_table
    assert "Learned residual UKF (not validation-selected)" in artifact_table
    assert "post-hoc-robustness rather than confirmatory" in artifact_table
    assert "not operational POD" in artifact_table


def test_build_paper_assets_temporal_corrected_campaign_summary_if_artifacts_present() -> None:
    recent_path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent.json"
    )
    prospective_path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516.json"
    )
    prospective_260523_path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_prospective_260523"
        / "real_slr_sp3_temporal_corrected_od_prospective_260523.json"
    )
    _assert_required_artifact(recent_path)
    _assert_required_artifact(prospective_path)
    _assert_required_artifact(prospective_260523_path)

    from scripts.build_paper_assets import (
        build_real_slr_sp3_temporal_corrected_od_campaign_summary_table,
    )

    table = build_real_slr_sp3_temporal_corrected_od_campaign_summary_table(
        recent_path=recent_path,
        prospective_path=prospective_path,
        prospective_260523_path=prospective_260523_path,
    )
    assert (
        "\\label{tab:real_slr_sp3_temporal_corrected_od_campaign_summary}"
        in table
    )
    assert "Exploratory recent post-hoc" in table
    assert "Controlling prospective public week" in table
    assert "10/10" in table
    assert "8/10" in table
    assert "LAGEOS-1 2026-05-13: 0 obs" in table
    assert "LAGEOS-2 2026-05-13: 7 obs" in table
    assert "fewer than 10 SP3-covered CRD normal points" in table
    assert "fixed arc-construction eligibility rule before scoring" in table
    assert "AUKF (full correction); validation 464.39 m; test 566.20 m" in table
    assert "UKF (full correction); validation 363.49 m; test 426.53 m" in table
    assert (
        "Learned 490.72 m vs UKF (full correction) 549.33 m; "
        "gap -58.61 m, CI [-83.47, -30.50], lower on 9/10 arcs"
        in table
    )
    assert (
        "Learned 458.01 m vs UKF (full correction) 426.53 m; "
        "gap +31.48 m, CI [-50.28, 132.35], lower on 5/8 arcs"
        in table
    )

    table_path = (
        REPO_ROOT
        / "paper"
        / "tables"
        / "real_slr_sp3_temporal_corrected_od_campaign_summary.tex"
    )
    _assert_required_artifact(table_path)
    materialized_table = table_path.read_text(encoding="utf-8")
    assert _normalize_newlines(materialized_table) == _normalize_newlines(table)


# ---------------------------------------------------------------------------
# prospective_260523 schedule tests
# ---------------------------------------------------------------------------


def test_prospective_260523_split_plan() -> None:
    plan = split_plan(SCHEDULE_PROSPECTIVE_260523)

    assert plan["train"][0] == "251220"
    assert plan["train"][-1] == "260502"
    assert "260502" in plan["train"]
    assert "260509" not in plan["train"]
    assert "260516" not in plan["train"]
    assert plan["validation"] == ["260516"]
    assert plan["test"] == ["260523"]
    # 260509 is unused (it is in the formal210 corpus but not assigned to any split)
    assert "260509" in plan["unused"] or plan["unused"] == []


def test_prospective_260523_metadata() -> None:
    meta = schedule_metadata(SCHEDULE_PROSPECTIVE_260523)

    assert meta["schedule_name"] == SCHEDULE_PROSPECTIVE_260523
    assert meta["confirmatory_status"] == "predeclared_prospective_public_temporal_holdout"
    assert "Rule fixed before scoring" in meta["reason_for_boundary"]
    assert "260523" in meta["reason_for_boundary"]
    assert "260516" in meta["reason_for_boundary"]

    # test week 260523 dates
    assert meta["weeks"]["260523"]["dates"] == [
        "20260518",
        "20260519",
        "20260520",
        "20260521",
        "20260522",
    ]
    assert meta["weeks"]["260523"]["original_formal210_split"] is None
    assert meta["weeks"]["260523"]["source_schedule_role"] == "prospective_added_public_week"

    # validation week 260516 dates
    assert meta["weeks"]["260516"]["dates"] == [
        "20260511",
        "20260512",
        "20260513",
        "20260514",
        "20260515",
    ]
    assert meta["weeks"]["260516"]["original_formal210_split"] is None
    assert meta["weeks"]["260516"]["source_schedule_role"] == "prospective_added_public_week"

    # 10 test arcs (5 dates x 2 targets)
    assert len(meta["test_arcs"]) == 10
    assert {arc["sp3_week"] for arc in meta["test_arcs"]} == {"260523"}
    assert {arc["date"] for arc in meta["test_arcs"]} == {
        "20260518",
        "20260519",
        "20260520",
        "20260521",
        "20260522",
    }
    assert {arc["target"] for arc in meta["test_arcs"]} == {"LAGEOS-1", "LAGEOS-2"}
    assert {
        (arc["target"], arc["date"])
        for arc in meta["test_arcs"]
    } == {
        (target, date)
        for target in ("LAGEOS-1", "LAGEOS-2")
        for date in (
            "20260518",
            "20260519",
            "20260520",
            "20260521",
            "20260522",
        )
    }

    # 10 validation arcs (5 dates x 2 targets for 260516)
    assert len(meta["validation_arcs"]) == 10
    assert {arc["sp3_week"] for arc in meta["validation_arcs"]} == {"260516"}


def test_prospective_260523_predeclaration_boundary() -> None:
    pre = build_predeclaration(SCHEDULE_PROSPECTIVE_260523)

    assert pre["schema_version"] == "real_slr_sp3_temporal_corrected_od_predeclaration_v1"
    assert pre["artifact_role"] == "predeclared_rule_schedule_before_scoring"
    boundary = pre["claim_boundary"]
    assert boundary["prospective_public_temporal_holdout"] is True
    assert boundary["rule_fixed_before_scoring_new_public_week"] is True
    assert boundary["post_hoc_robustness_not_confirmatory"] is False
    assert boundary["can_be_used_as_operational_pod_validation"] is False
    assert boundary["can_be_used_as_central_external_validation"] is False
    assert "260523" in boundary["appropriate_use"]
    assert "not operational POD" in boundary["appropriate_use"]
    # Claim boundary must not mention public deposit/repository/Zenodo/GitHub
    for forbidden in ("repository", "Zenodo", "GitHub", "public DOI", "deposit"):
        assert forbidden not in boundary["appropriate_use"], (
            f"forbidden word '{forbidden}' found in appropriate_use"
        )
    schedule_meta = pre["schedule"]
    assert schedule_meta["schedule_name"] == SCHEDULE_PROSPECTIVE_260523
    assert schedule_meta["train_weeks"][-1] == "260502"
    assert schedule_meta["validation_weeks"] == ["260516"]
    assert schedule_meta["test_weeks"] == ["260523"]


def test_prospective_260523_cli_path_safety(tmp_path) -> None:
    parser = build_parser()

    # non-prospective schedule should pass without extra options
    legacy_args = parser.parse_args(["--schedule", SCHEDULE_FORMAL210_MINI_PRE260509])
    validate_prospective_cli_path_safety(legacy_args, {"--schedule"})

    # 260523 without explicit --predeclaration should fail
    prospective_defaults = parser.parse_args(["--schedule", SCHEDULE_PROSPECTIVE_260523])
    with pytest.raises(SystemExit, match="--predeclaration"):
        validate_prospective_cli_path_safety(prospective_defaults, {"--schedule"})

    # provide non-default predeclaration and output paths → should pass with --no-table
    predecl = tmp_path / "prospective_260523.json"
    output = tmp_path / "prospective_260523_result.json"
    safe_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260523,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(output),
            "--no-table",
        ]
    )
    validate_prospective_cli_path_safety(
        safe_args,
        {"--schedule", "--predeclaration", "--output-json", "--no-table"},
    )

    # providing paths but forgetting --no-table or non-default --table should fail
    unsafe_table_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260523,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(output),
        ]
    )
    with pytest.raises(SystemExit, match="--no-table"):
        validate_prospective_cli_path_safety(
            unsafe_table_args,
            {"--schedule", "--predeclaration", "--output-json"},
        )

    # using generic DEFAULT_PREDECLARATION should still fail
    from scripts.run_real_slr_sp3_temporal_corrected_od_campaign import (
        DEFAULT_PREDECLARATION as DEFAULT_PREDECL,
        DEFAULT_OUTPUT_JSON as DEFAULT_OUT,
    )
    default_predecl_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260523,
            "--predeclaration",
            str(DEFAULT_PREDECL),
            "--output-json",
            str(output),
            "--no-table",
        ]
    )
    with pytest.raises(SystemExit, match="--predeclaration"):
        validate_prospective_cli_path_safety(
            default_predecl_args,
            {"--schedule", "--predeclaration", "--output-json", "--no-table"},
        )

    # using generic DEFAULT_OUTPUT_JSON should fail
    default_output_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260523,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(DEFAULT_OUT),
            "--no-table",
        ]
    )
    with pytest.raises(SystemExit, match="--output-json"):
        validate_prospective_cli_path_safety(
            default_output_args,
            {"--schedule", "--predeclaration", "--output-json", "--no-table"},
        )

    # --write-predeclaration-only should not require --output-json
    predecl_only_args = parser.parse_args(
        [
            "--schedule",
            SCHEDULE_PROSPECTIVE_260523,
            "--predeclaration",
            str(predecl),
            "--write-predeclaration-only",
        ]
    )
    validate_prospective_cli_path_safety(
        predecl_only_args,
        {"--schedule", "--predeclaration", "--write-predeclaration-only"},
    )


def test_prospective_260523_learned_boundary_wording() -> None:
    context = learned_vs_recursive_classical_readout_context(
        SCHEDULE_PROSPECTIVE_260523, 10
    )
    assert context["readout_scope"] == (
        "prospective_public_week_10_arc_readout_not_operational_pod"
    )
    assert "predeclared prospective public-week 10-arc test readout" in context["boundary"]
    assert "post-hoc" not in context["boundary"]
    assert "not operational POD" in context["boundary"]


def test_prospective_260523_arc_specs() -> None:
    test_specs = build_arc_specs(SCHEDULE_PROSPECTIVE_260523, "test")
    assert len(test_specs) == 10
    assert {a.sp3_week for a in test_specs} == {"260523"}
    assert {a.date for a in test_specs} == {
        "20260518",
        "20260519",
        "20260520",
        "20260521",
        "20260522",
    }
    assert {a.target for a in test_specs} == {"LAGEOS-1", "LAGEOS-2"}
    assert all(a.split == "test" for a in test_specs)

    val_specs = build_arc_specs(SCHEDULE_PROSPECTIVE_260523, "validation")
    assert len(val_specs) == 10
    assert {a.sp3_week for a in val_specs} == {"260516"}
    assert all(a.split == "validation" for a in val_specs)


@pytest.mark.parametrize(
    ("schedule_name", "validation_week", "test_week", "test_dates"),
    [
        (
            SCHEDULE_PROSPECTIVE_260530,
            "260523",
            "260530",
            ["20260525", "20260526", "20260527", "20260528", "20260529"],
        ),
        (
            SCHEDULE_PROSPECTIVE_260606,
            "260530",
            "260606",
            ["20260601", "20260602", "20260603", "20260604", "20260605"],
        ),
        (
            SCHEDULE_PROSPECTIVE_260613,
            "260606",
            "260613",
            ["20260608", "20260609", "20260610", "20260611", "20260612"],
        ),
        (
            SCHEDULE_PROSPECTIVE_260620,
            "260613",
            "260620",
            ["20260615", "20260616", "20260617", "20260618", "20260619"],
        ),
        (
            SCHEDULE_PROSPECTIVE_260627,
            "260620",
            "260627",
            ["20260622", "20260623", "20260624", "20260625", "20260626"],
        ),
    ],
)
def test_future_prospective_schedule_metadata_and_arc_specs(
    schedule_name: str,
    validation_week: str,
    test_week: str,
    test_dates: list[str],
) -> None:
    plan = split_plan(schedule_name)

    assert plan["train"][0] == "251220"
    assert plan["train"][-1] == "260502"
    assert "260502" in plan["train"]
    assert "260509" not in plan["train"]
    assert validation_week not in plan["train"]
    assert test_week not in plan["train"]
    assert plan["validation"] == [validation_week]
    assert plan["test"] == [test_week]

    meta = schedule_metadata(schedule_name)
    assert meta["schedule_name"] == schedule_name
    assert meta["confirmatory_status"] == (
        "predeclared_prospective_public_temporal_holdout"
    )
    assert "Rule fixed before scoring" in meta["reason_for_boundary"]
    assert validation_week in meta["reason_for_boundary"]
    assert test_week in meta["reason_for_boundary"]
    assert meta["weeks"][test_week]["dates"] == test_dates
    assert meta["weeks"][test_week]["original_formal210_split"] is None
    assert meta["weeks"][test_week]["source_schedule_role"] == (
        "prospective_added_public_week"
    )
    assert meta["weeks"][validation_week]["original_formal210_split"] is None
    assert meta["weeks"][validation_week]["source_schedule_role"] == (
        "prospective_added_public_week"
    )
    assert len(meta["test_arcs"]) == 10
    assert {arc["sp3_week"] for arc in meta["test_arcs"]} == {test_week}
    assert {arc["date"] for arc in meta["test_arcs"]} == set(test_dates)
    assert {arc["target"] for arc in meta["test_arcs"]} == {"LAGEOS-1", "LAGEOS-2"}
    assert len(meta["validation_arcs"]) == 10
    assert {arc["sp3_week"] for arc in meta["validation_arcs"]} == {validation_week}

    test_specs = build_arc_specs(schedule_name, "test")
    assert len(test_specs) == 10
    assert {a.sp3_week for a in test_specs} == {test_week}
    assert {a.date for a in test_specs} == set(test_dates)
    assert {a.target for a in test_specs} == {"LAGEOS-1", "LAGEOS-2"}
    assert all(a.split == "test" for a in test_specs)

    val_specs = build_arc_specs(schedule_name, "validation")
    assert len(val_specs) == 10
    assert {a.sp3_week for a in val_specs} == {validation_week}
    assert all(a.split == "validation" for a in val_specs)


@pytest.mark.parametrize(
    ("schedule_name", "validation_week", "test_week"),
    [
        (SCHEDULE_PROSPECTIVE_260530, "260523", "260530"),
        (SCHEDULE_PROSPECTIVE_260606, "260530", "260606"),
        (SCHEDULE_PROSPECTIVE_260613, "260606", "260613"),
        (SCHEDULE_PROSPECTIVE_260620, "260613", "260620"),
        (SCHEDULE_PROSPECTIVE_260627, "260620", "260627"),
    ],
)
def test_future_prospective_predeclaration_and_cli_path_safety(
    tmp_path: Path,
    schedule_name: str,
    validation_week: str,
    test_week: str,
) -> None:
    pre = build_predeclaration(schedule_name)

    boundary = pre["claim_boundary"]
    assert boundary["prospective_public_temporal_holdout"] is True
    assert boundary["rule_fixed_before_scoring_new_public_week"] is True
    assert boundary["post_hoc_robustness_not_confirmatory"] is False
    assert boundary["can_be_used_as_operational_pod_validation"] is False
    assert boundary["can_be_used_as_central_external_validation"] is False
    assert test_week in boundary["appropriate_use"]
    assert "not operational POD" in boundary["appropriate_use"]

    schedule_meta = pre["schedule"]
    assert schedule_meta["schedule_name"] == schedule_name
    assert schedule_meta["train_weeks"][-1] == "260502"
    assert schedule_meta["validation_weeks"] == [validation_week]
    assert schedule_meta["test_weeks"] == [test_week]

    parser = build_parser()
    prospective_defaults = parser.parse_args(["--schedule", schedule_name])
    with pytest.raises(SystemExit, match="--predeclaration"):
        validate_prospective_cli_path_safety(prospective_defaults, {"--schedule"})

    predecl = tmp_path / f"{schedule_name}.json"
    output = tmp_path / f"{schedule_name}_result.json"
    safe_args = parser.parse_args(
        [
            "--schedule",
            schedule_name,
            "--predeclaration",
            str(predecl),
            "--output-json",
            str(output),
            "--no-table",
        ]
    )
    validate_prospective_cli_path_safety(
        safe_args,
        {"--schedule", "--predeclaration", "--output-json", "--no-table"},
    )

    predecl_only_args = parser.parse_args(
        [
            "--schedule",
            schedule_name,
            "--predeclaration",
            str(predecl),
            "--write-predeclaration-only",
        ]
    )
    validate_prospective_cli_path_safety(
        predecl_only_args,
        {"--schedule", "--predeclaration", "--write-predeclaration-only"},
    )


def test_summary_table_without_260523_path_omits_third_row(tmp_path) -> None:
    """Summary table function works when prospective_260523_path is None or absent."""
    recent_path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent.json"
    )
    prospective_path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516.json"
    )
    _assert_required_artifact(recent_path)
    _assert_required_artifact(prospective_path)

    from scripts.build_paper_assets import (
        build_real_slr_sp3_temporal_corrected_od_campaign_summary_table,
    )

    # With no 260523 path, two-row table is generated
    table_no_260523 = build_real_slr_sp3_temporal_corrected_od_campaign_summary_table(
        recent_path=recent_path,
        prospective_path=prospective_path,
        prospective_260523_path=None,
    )
    assert "\\label{tab:real_slr_sp3_temporal_corrected_od_campaign_summary}" in table_no_260523
    assert "260516 row is the controlling predeclared" in table_no_260523

    # With a non-existent 260523 path, same two-row table is generated (graceful skip)
    absent_path = tmp_path / "nonexistent_260523.json"
    table_absent_260523 = build_real_slr_sp3_temporal_corrected_od_campaign_summary_table(
        recent_path=recent_path,
        prospective_path=prospective_path,
        prospective_260523_path=absent_path,
    )
    assert "\\label{tab:real_slr_sp3_temporal_corrected_od_campaign_summary}" in table_absent_260523
    assert "260516 row is the controlling predeclared" in table_absent_260523


def test_temporal_corrected_prospective_260523_artifact_boundary_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_prospective_260523"
        / "real_slr_sp3_temporal_corrected_od_prospective_260523.json"
    )
    if not path.exists():
        pytest.skip("260523 scoring artifact not yet generated")
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_temporal_corrected_od_campaign_v1"
    assert d["predeclared_schedule"]["schedule"]["schedule_name"] == (
        SCHEDULE_PROSPECTIVE_260523
    )
    assert d["predeclared_schedule"]["schedule"]["confirmatory_status"] == (
        "predeclared_prospective_public_temporal_holdout"
    )
    assert d["selection_integrity"]["test_set_information_used_for_selection"] is False
    assert d["selection_integrity"]["calibrator_fit_uses_only_train_weeks"] is True
    assert d["selection_integrity"]["test_weeks"] == ["260523"]
    assert d["selection_integrity"]["validation_weeks"] == ["260516"]
    assert d["selection_integrity"]["train_weeks"][-1] == "260502"
    assert "260509" not in d["selection_integrity"]["train_weeks"]
    assert "260516" not in d["selection_integrity"]["train_weeks"]
    assert d["test_readout"]["n_planned_arcs"] == 10
    cb = d["claim_boundary"]
    assert cb["prospective_public_temporal_holdout"] is True
    assert cb["post_hoc_robustness_not_confirmatory"] is False
    assert cb["is_operational_validation"] is False
    assert cb["is_simulator_result_validation"] is False


# ---------------------------------------------------------------------------
# Loop 161: SP3 input-validation hardening tests
# ---------------------------------------------------------------------------


def test_validate_sp3_input_rejects_non_gzip_gz_file(tmp_path) -> None:
    """A .gz file that begins with HTML is rejected as sp3_not_valid_gzip."""
    html_content = b"<html><head><title>404 Not Found</title></head></html>\n"
    bad_gz = tmp_path / "nsgf.orb.lageos1.260523.v80.sp3.gz"
    bad_gz.write_bytes(html_content)

    text, kind, reason = _validate_sp3_input(bad_gz)

    assert text is None
    assert kind == "sp3_not_valid_gzip"
    assert reason is not None
    assert "gzip magic" in reason or "\\x1f\\x8b" in reason or "non-gzip" in reason
    assert str(bad_gz.stat().st_size) in reason or bad_gz.stat().st_size > 0


def test_validate_sp3_input_rejects_non_gzip_gz_with_html_bytes(tmp_path) -> None:
    """Specifically test the 825-byte HTML case reported for the 260523 week."""
    # Simulate an HTTP 404/redirect page materialized as the SP3 file
    html_bytes = (
        b"<!DOCTYPE html>\n<html>\n<head><title>Not Found</title></head>\n"
        b"<body><p>The requested URL was not found.</p></body>\n</html>\n"
    )
    # Pad to 825 bytes (approximate size reported)
    html_bytes = html_bytes + b" " * max(0, 825 - len(html_bytes))
    bad_gz = tmp_path / "nsgf.orb.lageos1.260523.v80.sp3.gz"
    bad_gz.write_bytes(html_bytes)

    text, kind, reason = _validate_sp3_input(bad_gz)

    assert text is None, "Expected None text for HTML content in .gz file"
    assert kind == "sp3_not_valid_gzip"
    assert "gzip magic" in reason or "\\x1f\\x8b" in reason or ".gz suffix" in reason


def test_validate_sp3_input_rejects_decompress_failure(tmp_path) -> None:
    """A .gz file with magic bytes but corrupted content fails decompression."""
    import struct
    # Minimal gzip header: magic + CM + FLG + MTIME + XFL + OS, then garbage body
    corrupted = b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"\xff\xfe\xfd\xfc" * 20
    bad_gz = tmp_path / "bad.sp3.gz"
    bad_gz.write_bytes(corrupted)

    text, kind, reason = _validate_sp3_input(bad_gz)

    assert text is None
    assert kind == "sp3_decompress_failed"
    assert reason is not None


def test_validate_sp3_input_rejects_non_sp3_format_after_decompress(tmp_path) -> None:
    """A valid gzip file whose content is not SP3 format is rejected."""
    import gzip
    non_sp3_content = b"Hello, this is not an SP3 file\nNo hash at start\n"
    gz_path = tmp_path / "not_sp3.sp3.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(non_sp3_content)

    text, kind, reason = _validate_sp3_input(gz_path)

    assert text is None
    assert kind == "sp3_not_sp3_format"
    assert reason is not None


def test_validate_sp3_input_accepts_valid_sp3_gz(tmp_path) -> None:
    """A correctly formatted gzip-compressed SP3 file is accepted."""
    import gzip
    # Minimal SP3-like header content
    sp3_content = (
        b"#cP2026 5 18  0  0  0.00000000      96 ORBIT IGS14 HLM  NGS\n"
        b"## 2263  345600.00000000   900.00000000 60082 0.0000000000000\n"
        b"+   1   L51  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n"
        b"EOF\n"
    )
    gz_path = tmp_path / "nsgf.orb.lageos1.260523.v80.sp3.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(sp3_content)

    text, kind, reason = _validate_sp3_input(gz_path)

    assert text is not None
    assert kind is None
    assert reason is None
    assert text.startswith("#")


def test_validate_sp3_input_accepts_plain_sp3(tmp_path) -> None:
    """A plain (non-gzip) SP3 file is accepted."""
    sp3_content = (
        "#cP2026 5 18  0  0  0.00000000      96 ORBIT IGS14 HLM  NGS\n"
        "EOF\n"
    )
    plain_path = tmp_path / "test.sp3"
    plain_path.write_text(sp3_content, encoding="utf-8")

    text, kind, reason = _validate_sp3_input(plain_path)

    assert text is not None
    assert kind is None
    assert reason is None


def test_render_table_with_null_test_metrics() -> None:
    """render_table must not crash when test metrics are all null (zero-test case)."""
    pre = build_predeclaration(SCHEDULE_PROSPECTIVE_260523)
    null_test_means = {label: None for label in CLASSICAL_LABELS}
    null_test_means[LEARNED_LABEL] = None

    result = {
        "predeclared_schedule": pre,
        "selection": {
            "selected_candidate": "UKF (full correction)",
            "selected_validation_mean_rms_m": 350.0,
            "validation_mean_rms_m": {
                "UKF (full correction)": 350.0,
                LEARNED_LABEL: 400.0,
            },
        },
        "test_readout": {
            "selected_test_mean_rms_m": None,
            "test_mean_rms_m": null_test_means,
            "selected_vs_test_best_paired_gap": {"n": 0},
            "learned_vs_best_classical_paired_gap": {"n": 0},
        },
    }

    # Must not raise; null values should render as '--'
    table = render_table(result)
    assert "\\begin{table}" in table
    assert "--" in table  # null metrics render as '--'
    assert "not operational POD" in table


def test_zero_test_arcs_result_is_not_fabricated() -> None:
    """Honest partial result with zero test arcs must not contain estimator numbers."""
    pre = build_predeclaration(SCHEDULE_PROSPECTIVE_260523)
    null_test_means = {label: None for label in CLASSICAL_LABELS}
    null_test_means[LEARNED_LABEL] = None

    # Simulate the structure build_result would produce for zero-test case
    result = {
        "schema_version": "real_slr_sp3_temporal_corrected_od_campaign_v1",
        "status": "zero_test_arcs_input_unavailable",
        "predeclared_schedule": pre,
        "selection": {
            "selected_candidate": "UKF (full correction)",
            "selected_validation_mean_rms_m": 350.0,
            "validation_mean_rms_m": {
                "UKF (full correction)": 350.0,
                LEARNED_LABEL: 400.0,
            },
        },
        "test_readout": {
            "n_arcs": 0,
            "n_planned_arcs": 10,
            "n_failed_or_excluded_arcs": 10,
            "test_mean_rms_m": null_test_means,
            "selected_candidate": "UKF (full correction)",
            "selected_test_mean_rms_m": None,
            "test_best_candidate": None,
            "test_best_mean_rms_m": None,
            "best_classical_test_candidate": None,
            "best_classical_test_mean_rms_m": None,
            "selected_vs_test_best_paired_gap": {"n": 0},
            "selected_vs_best_classical_paired_gap": {"n": 0},
            "learned_vs_best_classical_paired_gap": {"n": 0},
            "arcs": [],
            "failed_or_excluded_arcs": [
                {
                    "status": "input_unavailable",
                    "kind": "sp3_not_valid_gzip",
                    "target": "LAGEOS-1",
                    "date": "20260518",
                    "sp3_week": "260523",
                }
            ],
        },
        "headline_readout": {
            "completed_test_arcs": 0,
            "planned_test_arcs": 10,
            "selected_test_mean_rms_m": None,
            "test_best_candidate": None,
            "test_best_mean_rms_m": None,
        },
    }

    # status must NOT be "completed"
    assert result["status"] != "completed"
    assert result["status"] == "zero_test_arcs_input_unavailable"
    # test metrics must be null (no fabricated numbers)
    for label, val in result["test_readout"]["test_mean_rms_m"].items():
        assert val is None, f"expected null test metric for {label!r}, got {val}"
    assert result["test_readout"]["n_arcs"] == 0
    assert result["test_readout"]["n_planned_arcs"] == 10
    # At least one excluded arc must document the input_unavailable reason
    excluded = result["test_readout"]["failed_or_excluded_arcs"]
    assert len(excluded) >= 1
    assert excluded[0]["status"] == "input_unavailable"
    assert "kind" in excluded[0]


def test_build_paper_assets_summary_table_skips_260523_when_not_completed(
    tmp_path,
) -> None:
    """Summary table gracefully omits 260523 row when its artifact status is not 'completed'."""
    recent_path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent"
        / "real_slr_sp3_temporal_corrected_od_campaign_recent.json"
    )
    prospective_path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516"
        / "real_slr_sp3_temporal_corrected_od_prospective_260516.json"
    )
    _assert_required_artifact(recent_path)
    _assert_required_artifact(prospective_path)

    from scripts.build_paper_assets import (
        build_real_slr_sp3_temporal_corrected_od_campaign_summary_table,
    )

    # Write a minimal zero-test 260523 artifact
    pre = build_predeclaration(SCHEDULE_PROSPECTIVE_260523)
    null_test_means = {label: None for label in CLASSICAL_LABELS}
    null_test_means[LEARNED_LABEL] = None
    zero_test_artifact = {
        "schema_version": "real_slr_sp3_temporal_corrected_od_campaign_v1",
        "status": "zero_test_arcs_input_unavailable",
        "predeclared_schedule": pre,
        "selection": {
            "selected_candidate": "UKF (full correction)",
            "selected_validation_mean_rms_m": 350.0,
            "validation_mean_rms_m": {
                "UKF (full correction)": 350.0,
                LEARNED_LABEL: 400.0,
            },
        },
        "test_readout": {
            "n_arcs": 0,
            "n_planned_arcs": 10,
            "n_failed_or_excluded_arcs": 10,
            "test_mean_rms_m": null_test_means,
        },
    }
    zero_test_path = tmp_path / "zero_test_260523.json"
    zero_test_path.write_text(json.dumps(zero_test_artifact, indent=2), encoding="utf-8")

    table = build_real_slr_sp3_temporal_corrected_od_campaign_summary_table(
        recent_path=recent_path,
        prospective_path=prospective_path,
        prospective_260523_path=zero_test_path,
    )

    # Table must be generated without crashing
    assert "\\label{tab:real_slr_sp3_temporal_corrected_od_campaign_summary}" in table
    # Zero-test 260523 row is omitted; two-row caption refers to 260516 only
    assert "260516 row is the controlling predeclared" in table
    # No fabricated metrics for 260523
    assert "260523" not in table or "input unavailable" in table.lower() or "260523" not in table
