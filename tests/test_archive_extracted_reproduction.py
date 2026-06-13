from __future__ import annotations

import json

import pytest

from scripts.build_supplementary_manifest import ARTIFACT_GROUPS
from scripts.regenerate_active_manuscript import ROOT, active_artifacts
from scripts.verify_archive_extracted_reproduction import (
    OD_CANONICAL_DIR_REL,
    OD_RERUN_DIR_REL,
    REVIEW_ALIAS_RESTORE_SOURCES,
    compare_public_od_claim_summaries,
    manifest_entries,
    prepare_archive_extracted_od_rerun_directory,
    python_script_from_command,
    restore_review_archive_aliases,
    review_archive_path,
    safe_public_input_name,
    safe_member_path,
)


def test_active_main_artifacts_are_indexed_for_review_archive() -> None:
    active = active_artifacts()
    active_paths = {
        row["path"].replace("\\", "/")
        for row in active["tables"] + active["figures"]
    }
    manifest_paths = {
        path.replace("\\", "/")
        for paths in ARTIFACT_GROUPS.values()
        for path in paths
    }

    assert "paper/tables/main_findings_summary.tex" in active_paths
    assert active_paths <= manifest_paths


def test_archive_extracted_active_regeneration_count_matches_current_active_set() -> None:
    report_path = ROOT / "results" / "validation" / "archive_extracted_reproduction.json"
    if not report_path.exists():
        pytest.skip("archive-extracted reproduction report has not been generated")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    nested = report["checks"]["active_table_regeneration_from_extracted_tree"][
        "nested_report"
    ]
    active = active_artifacts()
    expected_count = len(active["tables"]) + len(active["figures"])

    assert nested["artifact_count"] == expected_count
    assert nested["pass_count"] == expected_count
    assert nested["mismatch_count"] == 0
    assert nested["documented_blocker_count"] == 0


def test_manifest_entries_normalize_paths_and_preserve_group() -> None:
    manifest = {
        "artifact_groups": {
            "validation": [
                {
                    "path": "results\\validation\\archive_extracted_reproduction.json",
                    "exists": True,
                    "bytes": 12,
                    "sha256": "abc",
                }
            ]
        }
    }

    rows = manifest_entries(manifest)

    assert rows == [
        {
            "path": "results\\validation\\archive_extracted_reproduction.json",
            "exists": True,
            "bytes": 12,
            "sha256": "abc",
            "group": "validation",
            "norm_path": "results/validation/archive_extracted_reproduction.json",
        }
    ]


@pytest.mark.parametrize(
    "command,expected",
    [
        ("python -I scripts\\verify_archive_extracted_reproduction.py", "scripts/verify_archive_extracted_reproduction.py"),
        ("python scripts\\build_supplementary_manifest.py", "scripts/build_supplementary_manifest.py"),
        ("python -m pytest tests", None),
    ],
)
def test_python_script_from_command(command: str, expected: str | None) -> None:
    assert python_script_from_command(command) == expected


def test_safe_member_path_rejects_traversal() -> None:
    assert safe_member_path("paper/tables/main_results.tex").as_posix() == "paper/tables/main_results.tex"

    with pytest.raises(ValueError):
        safe_member_path("../paper/main.tex")


def test_safe_public_input_name_rejects_paths() -> None:
    assert safe_public_input_name("lageos1_20260505.np2") == "lageos1_20260505.np2"

    with pytest.raises(ValueError):
        safe_public_input_name("../lageos1_20260505.np2")

    with pytest.raises(ValueError):
        safe_public_input_name("nested/lageos1_20260505.np2")


def test_review_archive_alias_restore_report_excludes_raw_paths(tmp_path) -> None:
    for raw_rel in REVIEW_ALIAS_RESTORE_SOURCES:
        safe_rel = review_archive_path(raw_rel)
        safe_path = tmp_path.joinpath(*safe_rel.split("/"))
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(f"payload for {safe_rel}\n", encoding="utf-8")

    result = restore_review_archive_aliases(tmp_path)
    serialized = json.dumps(result, sort_keys=True)

    assert result["status"] == "pass"
    expected_restore_count = sum(
        1 for raw_rel in REVIEW_ALIAS_RESTORE_SOURCES if review_archive_path(raw_rel) != raw_rel
    )
    assert result["restored_count"] == expected_restore_count
    assert result["missing_alias_count"] == 0
    assert "_loop" not in serialized
    assert "loop42" not in serialized.lower()
    assert "loop91" not in serialized.lower()
    assert all("archive_alias" in row for row in result["restored"])
    assert all("source_digest_id" in row for row in result["restored"])


def test_prepare_archive_extracted_od_rerun_directory_copies_only_from_extracted_tree(tmp_path) -> None:
    source_dir = tmp_path / OD_CANONICAL_DIR_REL.replace("/", "\\")
    source_dir.mkdir(parents=True)
    for name in ("lageos1_20260505.np2", "nsgf.orb.lageos1.260509.v80.sp3.gz"):
        (source_dir / name).write_bytes(f"input:{name}".encode("utf-8"))
    (source_dir / "sp3_residual_calibrator.json").write_text("{}", encoding="utf-8")
    stale_dir = tmp_path / OD_RERUN_DIR_REL.replace("/", "\\")
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")

    copied, support = prepare_archive_extracted_od_rerun_directory(
        extracted_root=tmp_path,
        canonical={
            "input_digests": [
                {"archived_input_id": "lageos1_20260505.np2"},
                {"archived_input_id": "nsgf.orb.lageos1.260509.v80.sp3.gz"},
            ]
        },
    )

    assert len(copied) == 2
    assert len(support) == 1
    assert not (stale_dir / "stale.txt").exists()
    assert (stale_dir / "lageos1_20260505.np2").read_bytes() == b"input:lageos1_20260505.np2"
    assert (stale_dir / "nsgf.orb.lageos1.260509.v80.sp3.gz").is_file()


def _minimal_public_od_payload(mean: float = 334.72) -> dict:
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
            "EKF": {"n_arcs": 10, "mean_arc_rms_m": 367.98, "median_arc_rms_m": 248.67, "arcs_best_of": 1},
            "UKF (fixed-noise)": {"n_arcs": 10, "mean_arc_rms_m": mean, "median_arc_rms_m": 234.04, "arcs_best_of": 1},
            "AUKF (adaptive)": {"n_arcs": 10, "mean_arc_rms_m": 341.33, "median_arc_rms_m": 296.13, "arcs_best_of": 3},
            "SP3-IC propagation": {"n_arcs": 10, "mean_arc_rms_m": 402.92, "median_arc_rms_m": 260.68, "arcs_best_of": 5},
        },
        "dbar_external_validation": {
            "n_arcs_scored": 10,
            "n_correct": 6,
            "classification_accuracy": 0.6,
            "confusion": {"true_fire": 0, "true_no_fire": 6, "false_fire": 2, "false_no_fire": 2},
            "sensitivity": 0.0,
            "specificity": 0.75,
            "n_counterproductive_arcs": 2,
            "n_non_counterproductive_arcs": 8,
            "no_information_baseline": {
                "majority_class": "no_fire (negative)",
                "majority_class_accuracy": 0.8,
                "accuracy_minus_majority": -0.2,
                "beats_majority": False,
            },
        },
    }


def test_archive_extracted_public_od_claim_comparison_ignores_timestamps() -> None:
    first = _minimal_public_od_payload()
    second = _minimal_public_od_payload()
    second["generated_utc"] = "also ignored"

    result = compare_public_od_claim_summaries(first, second)

    assert result["status"] == "pass"
    assert result["mismatch_count"] == 0


def test_archive_extracted_public_od_claim_comparison_reports_field_mismatch() -> None:
    result = compare_public_od_claim_summaries(
        _minimal_public_od_payload(),
        _minimal_public_od_payload(mean=335.0),
    )

    assert result["status"] == "fail"
    assert result["mismatches"][0]["field"] == (
        "pooled_held_out_position_rmse_m.UKF (fixed-noise).mean_arc_rms_m"
    )
