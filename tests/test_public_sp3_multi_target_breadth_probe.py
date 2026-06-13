"""Tests for the public multi-target SP3/CRD breadth probe."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_public_sp3_multi_target_breadth_probe import (
    SCHEMA_VERSION,
    detect_single_sp3_satellite_id,
    detect_sp3_satellite_ids,
    parse_crd_coverage,
    summarize_state_rows,
    summarize_state_weeks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_detect_sp3_satellite_ids_from_header_and_records() -> None:
    text = "\n".join(
        [
            "#cV2026  4 12  0  0  0.00000000       2   SLR   ECF FIT NSGF",
            "## 2414      0.00000000   900.00000000 61142 0.0000000000000",
            "+    2   L50 L53  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
            "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
            "*  2026 04 12 00 00 00.00000000",
            "PL50  1000.0  2000.0  3000.0 0 0 0 0",
            "VL50  1.0  2.0  3.0 0 0 0 0",
            "PL53  4000.0  5000.0  6000.0 0 0 0 0",
        ]
    )
    assert detect_sp3_satellite_ids(text) == ["L50", "L53"]


def test_detect_single_sp3_satellite_id_uses_product_content() -> None:
    text = "\n".join(
        [
            "#cV2026  4 12  0  0  0.00000000       1   SLR   ECF FIT NSGF",
            "+    1   L67  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
            "*  2026 04 12 00 00 00.00000000",
            "PL67  1000.0  2000.0  3000.0 0 0 0 0",
        ]
    )
    assert detect_single_sp3_satellite_id(text) == "L67"


def test_parse_crd_coverage_does_not_filter_station_ids() -> None:
    crd = "\n".join(
        [
            "H1 CRD  2 2026 04 13 00",
            "H2 AAAA 1234 1 2 3 4",
            "H4 1 2026 04 13 01 00 00 2026 04 13 01 30 00 0 0 0 0 1 0",
            "11 3600.0 0.100000000000 0 0 120.0 10 20.0 0 0 0",
            "11 3610.0 0.100000100000 0 0 120.0 12 22.0 0 0 0",
            "H8",
            "H2 BBBB 5678 1 2 3 4",
            "H4 1 2026 04 13 02 00 00 2026 04 13 02 30 00 0 0 0 0 1 0",
            "11 7200.0 0.200000000000 0 0 120.0 8 18.0 0 0 0",
            "H8",
        ]
    )
    out = parse_crd_coverage(crd)
    assert out["normal_point_count"] == 3
    assert out["distinct_station_count"] == 2
    assert out["passes_with_normal_points"] == 2
    assert {s["station_key"] for s in out["station_counts"]} == {
        "AAAA:1234",
        "BBBB:5678",
    }


def test_state_summary_records_paired_improvement_boundary() -> None:
    out = summarize_state_rows(
        [
            {"status": "completed", "compact_rmse_m": 10.0, "hifi_rmse_m": 7.0},
            {"status": "completed", "compact_rmse_m": 8.0, "hifi_rmse_m": 9.0},
        ]
    )
    assert out["n_start_epochs"] == 2
    assert out["compact_mean_rms_m"] == 9.0
    assert out["hifi_mean_rms_m"] == 8.0
    gap = out["hifi_vs_compact"]
    assert gap["mean_improvement_m"] == 1.0
    assert "positive means candidate a has lower SP3-state RMSE" in gap[
        "improvement_convention"
    ]
    assert gap["bootstrap_unit"] == "fixed_start_epoch"
    assert gap["start_epoch_bootstrap95_mean_improvement_m"] == gap[
        "bootstrap95_mean_improvement_m"
    ]


def test_state_week_summary_records_clustered_sensitivity() -> None:
    weeks = [
        {
            "status": "completed",
            "target": "A",
            "week": "w1",
            "start_epoch_scores": [
                {"status": "completed", "compact_rmse_m": 10.0, "hifi_rmse_m": 7.0},
                {"status": "completed", "compact_rmse_m": 12.0, "hifi_rmse_m": 8.0},
            ],
        },
        {
            "status": "completed",
            "target": "A",
            "week": "w2",
            "start_epoch_scores": [
                {"status": "completed", "compact_rmse_m": 9.0, "hifi_rmse_m": 10.0},
                {"status": "completed", "compact_rmse_m": 11.0, "hifi_rmse_m": 9.0},
            ],
        },
        {
            "status": "completed",
            "target": "B",
            "week": "w1",
            "start_epoch_scores": [
                {"status": "completed", "compact_rmse_m": 8.0, "hifi_rmse_m": 5.0},
                {"status": "completed", "compact_rmse_m": 7.0, "hifi_rmse_m": 5.0},
            ],
        },
    ]
    out = summarize_state_weeks(weeks)
    assert out["n_start_epochs"] == 6
    uncertainty = out["cluster_uncertainty"]
    assert uncertainty["n_start_epochs"] == 6
    assert uncertainty["n_target_week_clusters"] == 3
    assert uncertainty["n_target_clusters"] == 2
    assert uncertainty["target_week_mean_improvement_m"] == out["hifi_vs_compact"][
        "mean_improvement_m"
    ]
    assert len(uncertainty["target_week_bootstrap95_mean_improvement_m"]) == 2
    assert len(uncertainty["target_bootstrap95_mean_improvement_m"]) == 2
    assert "finite-probe clustered sensitivity" in uncertainty["uncertainty_scope"]


def test_public_sp3_multi_target_artifact_shape_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "public_sp3_multi_target_breadth_probe"
        / "public_sp3_multi_target_breadth_probe.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == SCHEMA_VERSION
    integrity = d["selection_integrity"]
    assert integrity["train_validation_test_labels_used_as_reporting_strata_only"] is True
    assert integrity["test_set_information_used_for_selection"] is False
    assert integrity["no_model_or_hyperparameter_selection_performed"] is True
    cb = d["claim_boundary"]
    assert cb["is_central_external_validation"] is False
    assert cb["is_operational_pod"] is False
    assert cb["is_centimeter_slr_validation"] is False
    assert cb["claims_real_measurement_od"] is False
    assert cb["crd_used_for_coverage_only"] is True
    assert "sp3_state_scoring" in d
    assert "crd_coverage" in d
    all_summary = d["sp3_state_scoring"]["summary"]["all"]
    assert all_summary["hifi_vs_compact"]["bootstrap_unit"] == "fixed_start_epoch"
    assert all_summary["hifi_vs_compact"][
        "start_epoch_bootstrap95_mean_improvement_m"
    ] == all_summary["hifi_vs_compact"]["bootstrap95_mean_improvement_m"]
    uncertainty = all_summary["cluster_uncertainty"]
    assert uncertainty["n_start_epochs"] == d["headline_readout"][
        "scored_start_epochs"
    ]
    assert uncertainty["n_target_week_clusters"] == d["headline_readout"][
        "completed_target_weeks"
    ]
    assert uncertainty["n_target_clusters"] == d["headline_readout"][
        "sp3_targets_scored"
    ]
    assert len(uncertainty["target_week_bootstrap95_mean_improvement_m"]) == 2
    assert len(uncertainty["target_bootstrap95_mean_improvement_m"]) == 2


def test_manifest_source_indexes_public_multi_target_probe() -> None:
    text = (REPO_ROOT / "scripts" / "build_supplementary_manifest.py").read_text(
        encoding="utf-8"
    )

    assert "public_multi_target_sp3_crd_breadth_probe" in text
    assert "public_multi_target_sp3_crd_archived_inputs" in text
    assert "results/public_sp3_multi_target_breadth_probe/public_sp3_multi_target_breadth_probe.json" in text
    assert "paper/tables/public_sp3_multi_target_breadth_probe.tex" in text
    assert "scripts/run_public_sp3_multi_target_breadth_probe.py" in text
    assert "tests/test_public_sp3_multi_target_breadth_probe.py" in text
    assert "public_breadth_input_rels" in text
