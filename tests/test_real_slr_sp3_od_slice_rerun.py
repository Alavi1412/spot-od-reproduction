from __future__ import annotations

from scripts.run_real_slr_sp3_od_slice_rerun_validation import (
    compare_claim_summaries,
    compare_table_text,
    extract_public_claim_summary,
)


def _tiny_artifact(mean: float = 334.72) -> dict:
    return {
        "schema_version": "real_slr_sp3_od_v1",
        "generated_utc": "ignored",
        "status": "completed",
        "targets": ["LAGEOS-1", "LAGEOS-2"],
        "num_arcs": 10,
        "num_arcs_completed": 10,
        "sp3_analysis_center": "NSGF",
        "sp3_week_product": "260509",
        "fixed_station_subset": ["HERL", "MATM", "WETL", "YARL"],
        "pooled_held_out_position_rmse_m": {
            "EKF": {
                "n_arcs": 10,
                "mean_arc_rms_m": 367.98,
                "median_arc_rms_m": 248.67,
                "arcs_best_of": 1,
            },
            "UKF (fixed-noise)": {
                "n_arcs": 10,
                "mean_arc_rms_m": mean,
                "median_arc_rms_m": 234.04,
                "arcs_best_of": 1,
            },
            "AUKF (adaptive)": {
                "n_arcs": 10,
                "mean_arc_rms_m": 341.33,
                "median_arc_rms_m": 296.13,
                "arcs_best_of": 3,
            },
            "SP3-IC propagation": {
                "n_arcs": 10,
                "mean_arc_rms_m": 402.92,
                "median_arc_rms_m": 260.68,
                "arcs_best_of": 5,
            },
        },
        "dbar_external_validation": {
            "n_arcs_scored": 10,
            "n_correct": 6,
            "classification_accuracy": 0.6,
            "confusion": {
                "true_fire": 0,
                "true_no_fire": 6,
                "false_fire": 2,
                "false_no_fire": 2,
            },
            "sensitivity": 0.0,
            "specificity": 0.75,
            "n_counterproductive_arcs": 2,
            "n_non_counterproductive_arcs": 8,
            "no_information_baseline": {
                "majority_class": False,
                "majority_class_accuracy": 0.8,
                "accuracy_minus_majority": -0.2,
                "beats_majority": False,
            },
        },
    }


def test_extract_public_claim_summary_ignores_timestamps() -> None:
    first = _tiny_artifact()
    second = _tiny_artifact()
    second["generated_utc"] = "also ignored"

    assert extract_public_claim_summary(first) == extract_public_claim_summary(second)


def test_compare_claim_summaries_passes_exact_public_claim_surface() -> None:
    result = compare_claim_summaries(_tiny_artifact(), _tiny_artifact())

    assert result["status"] == "pass"
    assert result["mismatch_count"] == 0


def test_compare_claim_summaries_reports_field_mismatch() -> None:
    result = compare_claim_summaries(_tiny_artifact(), _tiny_artifact(mean=335.0))

    assert result["status"] == "fail"
    assert result["mismatch_count"] == 1
    assert result["mismatches"][0]["field"] == (
        "pooled_held_out_position_rmse_m.UKF (fixed-noise).mean_arc_rms_m"
    )


def test_compare_table_text_ignores_only_line_endings_and_final_newline() -> None:
    result = compare_table_text("a\r\nb\n", "a\nb")

    assert result["status"] == "pass"
    assert result["matches_submitted_table"] is True


def test_compare_table_text_reports_content_difference() -> None:
    result = compare_table_text("a\nchanged", "a\nsubmitted")

    assert result["status"] == "fail"
    assert result["matches_submitted_table"] is False
    assert result["diff_head"]
