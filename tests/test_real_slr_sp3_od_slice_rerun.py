from __future__ import annotations

from scripts.run_real_slr_sp3_od_slice_rerun_validation import (
    PUBLIC_OD_RMSE_ABS_TOLERANCE_M,
    PUBLIC_OD_TABLE_TARGETED_ABS_TOLERANCE,
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
    assert result["tolerated_numeric_difference_count"] == 0


def test_compare_claim_summaries_tolerates_small_pooled_rmse_drift() -> None:
    result = compare_claim_summaries(
        _tiny_artifact(mean=334.72),
        _tiny_artifact(mean=334.76),
    )

    assert result["status"] == "pass"
    assert result["mismatch_count"] == 0
    assert result["tolerated_numeric_difference_count"] == 1
    assert result["tolerated_numeric_differences"][0]["field"] == (
        "pooled_held_out_position_rmse_m.UKF (fixed-noise).mean_arc_rms_m"
    )
    assert result["tolerated_numeric_differences"][0]["abs_tolerance"] == (
        PUBLIC_OD_RMSE_ABS_TOLERANCE_M
    )


def test_compare_claim_summaries_reports_field_mismatch() -> None:
    result = compare_claim_summaries(_tiny_artifact(), _tiny_artifact(mean=335.0))

    assert result["status"] == "fail"
    assert result["mismatch_count"] == 1
    assert result["mismatches"][0]["field"] == (
        "pooled_held_out_position_rmse_m.UKF (fixed-noise).mean_arc_rms_m"
    )


def test_compare_claim_summaries_keeps_counts_strict() -> None:
    changed = _tiny_artifact()
    changed["pooled_held_out_position_rmse_m"]["UKF (fixed-noise)"]["n_arcs"] = 11

    result = compare_claim_summaries(_tiny_artifact(), changed)

    assert result["status"] == "fail"
    assert result["mismatches"][0]["field"] == (
        "pooled_held_out_position_rmse_m.UKF (fixed-noise).n_arcs"
    )
    assert result["mismatches"][0]["abs_tolerance"] is None


def test_compare_claim_summaries_rejects_count_type_change() -> None:
    changed = _tiny_artifact()
    changed["pooled_held_out_position_rmse_m"]["UKF (fixed-noise)"]["n_arcs"] = 10.0

    result = compare_claim_summaries(_tiny_artifact(), changed)

    assert result["status"] == "fail"
    mismatch = result["mismatches"][0]
    assert mismatch["field"] == "pooled_held_out_position_rmse_m.UKF (fixed-noise).n_arcs"
    assert mismatch["expected_type"] == "int"
    assert mismatch["actual_type"] == "float"
    assert mismatch["abs_tolerance"] is None


def test_compare_claim_summaries_rejects_bool_type_change() -> None:
    changed = _tiny_artifact()
    changed["dbar_external_validation"]["no_information_baseline"]["beats_majority"] = 0

    result = compare_claim_summaries(_tiny_artifact(), changed)

    assert result["status"] == "fail"
    mismatch = result["mismatches"][0]
    assert mismatch["field"] == "dbar_external_validation.no_information_baseline.beats_majority"
    assert mismatch["expected_type"] == "bool"
    assert mismatch["actual_type"] == "int"
    assert mismatch["abs_tolerance"] is None


def test_compare_table_text_ignores_only_line_endings_and_final_newline() -> None:
    result = compare_table_text("a\r\nb\n", "a\nb")

    assert result["status"] == "pass"
    assert result["matches_submitted_table"] is True
    assert result["byte_identical_after_normalization"] is True


def _public_od_table_excerpt() -> str:
    return (
        "    EKF & 367.98 & 248.67 & 1/10 \\\\\n"
        "    UKF (fixed-noise) & 334.72 & 234.04 & 1/10 \\\\\n"
        "    AUKF (adaptive) & 341.33 & 296.13 & 3/10 \\\\\n"
        "  \\\\[2pt] {\\footnotesize \\textbf{Compact recursive-filter readout.} "
        "The paired EKF-minus-AUKF gap (positive favors AUKF) is mean 26.65~m, "
        "median 16.77~m, with deterministic 20,000-resample bootstrap 95\\% CI "
        "$[-43.58,108.48]$~m, which spans zero. The fixed-noise UKF remains "
        "slightly best by pooled mean (334.72~m versus 341.33~m for AUKF; "
        "UKF-minus-AUKF mean -6.61~m, 95\\% CI $[-55.13,26.24]$~m), so this "
        "is an underpowered real-measurement discriminative readout.}\n"
    )


def test_compare_table_text_tolerates_known_linux_public_od_drift() -> None:
    submitted = _public_od_table_excerpt()
    generated = (
        "    EKF & 367.98 & 248.67 & 1/10 \\\\\n"
        "    UKF (fixed-noise) & 334.76 & 234.15 & 1/10 \\\\\n"
        "    AUKF (adaptive) & 341.20 & 296.00 & 3/10 \\\\\n"
        "  \\\\[2pt] {\\footnotesize \\textbf{Compact recursive-filter readout.} "
        "The paired EKF-minus-AUKF gap (positive favors AUKF) is mean 26.70~m, "
        "median 16.80~m, with deterministic 20,000-resample bootstrap 95\\% CI "
        "$[-43.50,108.40]$~m, which spans zero. The fixed-noise UKF remains "
        "slightly best by pooled mean (334.76~m versus 341.20~m for AUKF; "
        "UKF-minus-AUKF mean -6.50~m, 95\\% CI $[-55.00,26.30]$~m), so this "
        "is an underpowered real-measurement discriminative readout.}\n"
    )

    result = compare_table_text(generated, submitted)

    assert result["status"] == "pass"
    assert result["matches_submitted_table"] is True
    assert result["byte_identical_after_normalization"] is False
    assert result["tolerated_numeric_difference_count"] == 13
    assert result["max_observed_abs_delta"] <= PUBLIC_OD_TABLE_TARGETED_ABS_TOLERANCE


def test_compare_table_text_tolerates_targeted_table_row_drift() -> None:
    submitted = "UKF (fixed-noise) & 334.72 & 234.04 & 1/10 \\\\\n"
    generated = "UKF (fixed-noise) & 334.76 & 234.15 & 1/10 \\\\\n"

    result = compare_table_text(generated, submitted)

    assert result["status"] == "pass"
    assert result["matches_submitted_table"] is True
    assert result["byte_identical_after_normalization"] is False
    assert result["tolerated_numeric_difference_count"] == 2
    assert result["max_observed_abs_delta"] <= PUBLIC_OD_TABLE_TARGETED_ABS_TOLERANCE


def test_compare_table_text_reports_content_difference() -> None:
    result = compare_table_text("a\nchanged", "a\nsubmitted")

    assert result["status"] == "fail"
    assert result["matches_submitted_table"] is False
    assert result["diff_head"]


def test_compare_table_text_rejects_integer_count_drift() -> None:
    submitted = "AUKF is best on 6/10 arcs.\n"
    generated = "AUKF is best on 6/11 arcs.\n"

    result = compare_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["unchanged_outside_tolerated_fields"] is False


def test_compare_table_text_rejects_reff_threshold_drift() -> None:
    submitted = "The predeclared DBAR rule ($R_{\\mathrm{eff}}>1.5$) is fixed.\n"
    generated = "The predeclared DBAR rule ($R_{\\mathrm{eff}}>2.0$) is fixed.\n"

    result = compare_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["unchanged_outside_tolerated_fields"] is False


def test_compare_table_text_rejects_specificity_drift() -> None:
    submitted = "Agreement is reported; specificity 0.75 on the 8 non-counterproductive arcs.\n"
    generated = "Agreement is reported; specificity 0.30 on the 8 non-counterproductive arcs.\n"

    result = compare_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["unchanged_outside_tolerated_fields"] is False


def test_compare_table_text_rejects_out_of_tolerance_decimal_drift() -> None:
    submitted = "UKF (fixed-noise) & 334.72 & 234.04 & 1/10 \\\\\n"
    generated = "UKF (fixed-noise) & 335.50 & 234.04 & 1/10 \\\\\n"

    result = compare_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["numeric_mismatch_count"] == 1
    assert result["numeric_mismatches"][0]["abs_delta"] > PUBLIC_OD_TABLE_TARGETED_ABS_TOLERANCE
